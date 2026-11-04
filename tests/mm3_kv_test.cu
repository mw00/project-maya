// tests/mm3_kv_test.cu - maya-mm3's K / V caches (mm3_kernels.cuh: KvQ) and the sparse attention over them, no model
// needed:
//   * every format (BF16, INT8, FP8, FP4, and K / V mixed), rows rotated or not: qk_norm_rope writes the rows, the
//     test reads the codes back and decodes them on the CPU (its own decoder), and sparse_attn's output must match a
//     double-precision attention over those decoded rows (the queries and the output rotated alike) - the packing,
//     the decode and the rotation exact;
//   * KV streaming (a VRAM ring of 4096 rows, the rest from host memory) gives the same bits as the whole cache in VRAM;
//   * the quality of each format: the attention's output against F32 K / V (relative RMS error, the worst head's),
//     over MiniMax-M3.1's shapes (4 KV heads of 128, 16 query heads each, 16 kept blocks of 128) at positions up to
//     64K, keys with outlier channels and heavy-tailed values - what --kv-cache's default rests on, with the model's
//     own test on top (tools/mm3/kv_gate.py).
//
//   mm3_kv_test            run everything, exit 0 when every exactness check holds
#include "mm3_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = mm3::k;

namespace {

int g_fail = 0;
std::mt19937_64 g_rng(4321);

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

template <typename T>
T* dev(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

float bf16_f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float e4m3_f(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float a = e == 0 ? m * std::ldexp(1.0f, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -a : a;
}
float e2m1_f(int c) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
// a row of 128 values from its codes and scales (format f)
void decode_row(int f, const uint8_t* code, const uint8_t* sc, float* out) {
    for (int d = 0; d < 128; ++d) {
        if (f == k::kKvBf16) {
            uint16_t b;
            std::memcpy(&b, code + 2 * d, 2);
            out[d] = bf16_f(b);
        } else if (f == k::kKvI8) {
            float s;
            std::memcpy(&s, sc, 4);
            out[d] = (float) (int8_t) code[d] * s;
        } else if (f == k::kKvFp8) {
            out[d] = e4m3_f(code[d]) * std::ldexp(1.0f, (int) sc[d / 32] - 127);
        } else {
            out[d] = e2m1_f((code[d / 2] >> (4 * (d & 1))) & 15) * e4m3_f(sc[d / 16]);
        }
    }
}
// the 128-point Sylvester transform over sqrt(128), in place
void had(double* v) {
    for (int m = 1; m < 128; m <<= 1)
        for (int i = 0; i < 128; ++i)
            if (!(i & m)) {
                const double a = v[i], b = v[i | m];
                v[i] = a + b;
                v[i | m] = a - b;
            }
    for (int i = 0; i < 128; ++i) v[i] *= 0.088388347648318;
}

struct Case {
    int T = 0, ctx = 0, Hkv = 4, Hq = 64, topk = 16, blk = 128;
    std::vector<float> q, kk, vv;   // [T][Hq][128], [ctx][Hkv][128] x2 (F32: the reference's)
    std::vector<int> pos, kept;     // [T], [Hkv][T][topk]
};

// keys with a few outlier channels per head (RoPE-rotated pairs keep them), values heavy-tailed; queries at the end of
// the context, their kept blocks their own and 15 others
Case make_case(int ctx, int T) {
    Case c;
    c.ctx = ctx;
    c.T = T;
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::student_t_distribution<float> st(3.0f);
    std::vector<float> chan(128 * 4);
    for (int h = 0; h < 4; ++h)
        for (int d = 0; d < 128; ++d) chan[(size_t) h * 128 + d] = (g_rng() % 16 == 0) ? 6.0f : 1.0f;
    c.kk.resize((size_t) ctx * 4 * 128);
    c.vv.resize(c.kk.size());
    for (int p = 0; p < ctx; ++p)
        for (int h = 0; h < 4; ++h)
            for (int d = 0; d < 128; ++d) {
                c.kk[((size_t) p * 4 + h) * 128 + d] = n01(g_rng) * chan[(size_t) h * 128 + d];
                c.vv[((size_t) p * 4 + h) * 128 + d] = 0.5f * st(g_rng);
            }
    c.q.resize((size_t) T * 64 * 128);
    for (auto& x : c.q) x = n01(g_rng);
    c.pos.resize((size_t) T);
    for (int t = 0; t < T; ++t) c.pos[(size_t) t] = ctx - T + t;
    c.kept.assign((size_t) 4 * T * 16, -1);
    for (int h = 0; h < 4; ++h)
        for (int t = 0; t < T; ++t) {
            const int qb = c.pos[(size_t) t] / 128;
            int* o = &c.kept[((size_t) h * T + t) * 16];
            if (qb + 1 <= 16) {
                for (int i = 0; i <= qb; ++i) o[i] = i;
                continue;
            }
            o[0] = qb;
            std::vector<int> all(qb);
            for (int i = 0; i < qb; ++i) all[(size_t) i] = i;
            std::shuffle(all.begin(), all.end(), g_rng);
            for (int i = 1; i < 16; ++i) o[i] = all[(size_t) i - 1];
        }
    return c;
}

// the attention over rows K, V (the decoded cache, or F32) in double: out [T][Hq][128]; rotq / roto: the queries
// rotated before, the output after
std::vector<float> ref_attn(const Case& c, const std::vector<float>& K, const std::vector<float>& V, bool rotq,
                            bool roto) {
    std::vector<float> out((size_t) c.T * c.Hq * 128);
    const double scale = 1.0 / std::sqrt(128.0);
    for (int t = 0; t < c.T; ++t)
        for (int h = 0; h < c.Hq; ++h) {
            const int g = h / 16;
            double q[128];
            for (int d = 0; d < 128; ++d) q[d] = (double) (float) (c.q[((size_t) t * c.Hq + h) * 128 + d] * (float) scale);
            if (rotq) had(q);
            std::vector<double> s;
            std::vector<int> js;
            for (int i = 0; i < c.topk; ++i) {
                const int b = c.kept[((size_t) g * c.T + t) * c.topk + i];
                if (b < 0) continue;
                for (int j = b * c.blk; j < (b + 1) * c.blk && j <= c.pos[(size_t) t]; ++j) {
                    double dot = 0;
                    for (int d = 0; d < 128; ++d) dot += q[d] * K[((size_t) j * c.Hkv + g) * 128 + d];
                    s.push_back(dot);
                    js.push_back(j);
                }
            }
            const double mx = *std::max_element(s.begin(), s.end());
            double l = 0, o[128] = {0};
            for (size_t i = 0; i < s.size(); ++i) {
                const double e = std::exp(s[i] - mx);
                l += e;
                for (int d = 0; d < 128; ++d) o[d] += e * V[((size_t) js[i] * c.Hkv + g) * 128 + d];
            }
            for (int d = 0; d < 128; ++d) o[d] /= l;
            if (roto) had(o);
            for (int d = 0; d < 128; ++d) out[((size_t) t * c.Hq + h) * 128 + d] = (float) o[d];
        }
    return out;
}

struct Err {
    double rel = 0, worst_head = 0;
};
Err compare(const std::vector<float>& got, const std::vector<float>& ref, int heads) {
    Err e;
    double se = 0, sr = 0;
    const size_t per = got.size() / (size_t) heads;
    for (int h = 0; h < heads; ++h) {
        double a = 0, b = 0;
        for (size_t i = (size_t) h * per; i < (size_t) (h + 1) * per; ++i) {
            const double d = (double) got[i] - ref[i];
            a += d * d;
            b += (double) ref[i] * ref[i];
        }
        se += a;
        sr += b;
        e.worst_head = std::max(e.worst_head, std::sqrt(a / std::max(b, 1e-30)));
    }
    e.rel = std::sqrt(se / std::max(sr, 1e-30));
    return e;
}

// the cache in formats kf / vf (rot), written by qk_norm_rope from the case's F32 rows (unit norms, RoPE off: n_rot 0 -
// the rows as given), then sparse_attn for the case's queries; ring > 0: KV streaming with that many VRAM rows
struct Run {
    std::vector<float> out, Kd, Vd;   // the output, and the cache's rows decoded on the CPU
};
Run run(const Case& c, int kf, int vf, bool rot, int ring) {
    const int ctx = c.ctx, Hkv = 4, Hq = c.Hq;
    const size_t vrows = (size_t) (ring > 0 ? ring : ctx) * Hkv, hrows = (size_t) ctx * Hkv;
    uint8_t* kvb = nullptr;
    CK(cudaMalloc(&kvb, k::kv_bytes(kf, vf, vrows)));
    uint16_t *kc = nullptr, *vc = nullptr;
    k::KvQ q;
    k::kv_carve(kvb, vrows, kf, vf, rot ? 3 : 0, kc, vc, q);
    k::KvHost hs;
    uint8_t* hostb = nullptr;
    if (ring > 0) {
        CK(cudaHostAlloc((void**) &hostb, k::kv_bytes(kf, vf, hrows), cudaHostAllocMapped));
        k::kv_carve(hostb, hrows, kf, vf, rot ? 3 : 0, hs.kc, hs.vc, hs.q);
        hs.ring = ring;
        CK(cudaMalloc(&hs.tag, (size_t) ring * 4));
        CK(cudaMemset(hs.tag, 0xff, (size_t) ring * 4));
    }
    std::vector<float> ones(128, 1.0f), inv(1, 0.0f);
    float* d_one = dev(ones);
    float* d_inv = dev(inv);
    // the rows in chunks of 1024 positions (the prompt path's), K normed by qk's per-head RMSNorm: undo it by giving
    // rows of unit RMS (the test's K is then normalised - the reference reads the cache's own decoded rows anyway)
    const int C = 1024;
    std::vector<float> kn((size_t) ctx * Hkv * 128);
    for (size_t r = 0; r < (size_t) ctx * Hkv; ++r) {
        double ss = 0;
        for (int d = 0; d < 128; ++d) ss += (double) c.kk[r * 128 + d] * c.kk[r * 128 + d];
        const double rr = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
        for (int d = 0; d < 128; ++d) kn[r * 128 + d] = (float) (c.kk[r * 128 + d] * rr);
    }
    float* d_k = nullptr;
    float* d_v = nullptr;
    int* d_pos = nullptr;
    CK(cudaMalloc(&d_k, (size_t) C * Hkv * 128 * 4));
    CK(cudaMalloc(&d_v, (size_t) C * Hkv * 128 * 4));
    CK(cudaMalloc(&d_pos, (size_t) C * 4));
    for (int p0 = 0; p0 < ctx; p0 += C) {
        const int n = std::min(C, ctx - p0);
        std::vector<int> pos((size_t) n);
        for (int i = 0; i < n; ++i) pos[(size_t) i] = p0 + i;
        CK(cudaMemcpy(d_k, c.kk.data() + (size_t) p0 * Hkv * 128, (size_t) n * Hkv * 128 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_v, c.vv.data() + (size_t) p0 * Hkv * 128, (size_t) n * Hkv * 128 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, pos.data(), (size_t) n * 4, cudaMemcpyHostToDevice));
        k::qk_norm_rope(nullptr, d_k, d_v, d_one, d_one, d_pos, n, 0, Hkv, 128, 0, d_inv, 1e-6f, kc, vc, 0, 0, q, hs);
    }
    CK(cudaDeviceSynchronize());
    Run r;
    // the cache's rows decoded on the CPU, rotated back when they were stored rotated (the reference attention
    // rotates q and the output instead - so this undoing is only for the quality numbers below)
    r.Kd.resize((size_t) ctx * Hkv * 128);
    r.Vd.resize(r.Kd.size());
    for (int part = 0; part < 2; ++part) {
        const int f = part == 0 ? kf : vf;
        std::vector<uint8_t> codes, scales;
        const size_t rows = ring > 0 ? hrows : vrows;
        const uint16_t* c16 = part == 0 ? (ring > 0 ? hs.kc : kc) : (ring > 0 ? hs.vc : vc);
        const k::KvQ& qq = ring > 0 ? hs.q : q;
        if (f == k::kKvBf16) {
            codes = host((const uint8_t*) c16, rows * 256);
        } else {
            codes = host(part == 0 ? qq.kc : qq.vc, rows * k::kv_code_bytes(f));
            scales = host(part == 0 ? qq.ks : qq.vs, rows * k::kv_scale_bytes(f));
        }
        std::vector<float>& dst = part == 0 ? r.Kd : r.Vd;
        for (size_t row = 0; row < rows; ++row)
            decode_row(f, codes.data() + row * k::kv_code_bytes(f),
                       f == k::kKvBf16 ? nullptr : scales.data() + row * k::kv_scale_bytes(f), dst.data() + row * 128);
    }
    float* d_q = dev(c.q);
    int* d_qpos = dev(c.pos);
    int* d_kept = dev(c.kept);
    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, c.q.size() * 4));
    k::sparse_attn(d_q, kc, vc, d_kept, d_qpos, c.T, Hq, Hkv, c.topk, c.blk, 1.0f / std::sqrt(128.0f), d_out, 0, q, hs);
    CK(cudaDeviceSynchronize());
    r.out = host(d_out, c.q.size());
    for (void* p : {(void*) kvb, (void*) d_one, (void*) d_inv, (void*) d_k, (void*) d_v, (void*) d_pos, (void*) d_q,
                    (void*) d_qpos, (void*) d_kept, (void*) d_out})
        cudaFree(p);
    if (hostb) cudaFreeHost(hostb);
    if (hs.tag) cudaFree(hs.tag);
    (void) kn;
    return r;
}

std::string fname(int kf, int vf, bool rot) {
    return std::string("K ") + k::kv_name(kf) + " V " + k::kv_name(vf) + (rot && (kf || vf) ? " rot" : "");
}

}  // namespace

int main(int argc, char** argv) {
    (void) argc, (void) argv;
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("mm3_kv_test on %s\n", p.name);
    const int combos[][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {0, 3}, {0, 2}, {2, 0}, {3, 2}};
    // 1. exactness: the GPU's attention over the cache = the reference over the cache's decoded rows, 8K positions
    {
        const Case c = make_case(8192, 8);
        std::printf("exactness (8K positions, 8 queries): the attention over the decoded cache\n");
        for (const auto& cb : combos)
            for (int rot = 0; rot < 2; ++rot) {
                if (rot && cb[0] == 0 && cb[1] == 0) continue;
                const Run r = run(c, cb[0], cb[1], rot != 0, 0);
                const bool rk = rot && cb[0] != 0, rv = rot && cb[1] != 0;
                const Err e = compare(r.out, ref_attn(c, r.Kd, r.Vd, rk, rv), c.T * c.Hq);
                const bool ok = e.rel < 2e-5 && e.worst_head < 1e-4;
                std::printf("  %-26s rel %.2e  worst head %.2e  %s\n", fname(cb[0], cb[1], rot).c_str(), e.rel,
                            e.worst_head, ok ? "ok" : "FAIL");
                if (!ok) ++g_fail;
            }
    }
    // 2. KV streaming: a ring of 4096 rows over 12K positions, the same bits as the cache whole in VRAM
    {
        const Case c = make_case(12288, 4);
        std::printf("KV streaming (12K positions, a 4096-row ring): the same bits as the whole cache\n");
        for (const auto& cb : combos) {
            const Run a = run(c, cb[0], cb[1], true, 0), b = run(c, cb[0], cb[1], true, 4096);
            size_t diff = 0;
            for (size_t i = 0; i < a.out.size(); ++i) diff += std::memcmp(&a.out[i], &b.out[i], 4) != 0;
            std::printf("  %-26s %zu of %zu differ  %s\n", fname(cb[0], cb[1], true).c_str(), diff, a.out.size(),
                        diff ? "FAIL" : "ok");
            if (diff) ++g_fail;
        }
    }
    // 3. quality: each format's attention output against F32 K / V's, at 4K / 16K / 64K positions
    std::printf("quality: the attention's output vs F32 K / V (relative RMS error; the worst head)\n");
    for (int ctx : {4096, 16384, 65536}) {
        const Case c = make_case(ctx, 4);
        // (the F32 reference: K normed as qk_norm_rope norms it - unit RMS a row, its weight 1)
        std::vector<float> kn(c.kk.size());
        for (size_t r = 0; r < c.kk.size() / 128; ++r) {
            double ss = 0;
            for (int d = 0; d < 128; ++d) ss += (double) c.kk[r * 128 + d] * c.kk[r * 128 + d];
            const float rr = 1.0f / std::sqrt((float) (ss / 128.0) + 1e-6f);
            for (int d = 0; d < 128; ++d) kn[r * 128 + d] = c.kk[r * 128 + d] * rr;
        }
        const std::vector<float> ref = ref_attn(c, kn, c.vv, false, false);
        for (const auto& cb : combos)
            for (int rot = 0; rot < 2; ++rot) {
                if (rot && cb[0] == 0 && cb[1] == 0) continue;
                const Run r = run(c, cb[0], cb[1], rot != 0, 0);
                const Err e = compare(r.out, ref, c.T * c.Hq);
                std::printf("  ctx %6d  %-26s %6.0f B a position-layer  rel %.2e  worst head %.2e\n", ctx,
                            fname(cb[0], cb[1], rot).c_str(), (double) k::kv_bytes(cb[0], cb[1], 4), e.rel,
                            e.worst_head);
            }
    }
    std::printf("mm3_kv_test: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
