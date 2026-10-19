// tests/ds4_kv_fp4_parity.cu - strata-ds4's compressed KV and index keys stored as the packed FP4 rows they are
// (Finish::pdst) against the BF16 rows they were stored as: every value of every packed row, unpacked, the BF16 row's bit
// for bit (quant 2: FP4 with an E4M3 scale a 16, the compressed KV; quant 3: FP4 with a power of two a 32, the index
// keys), on random rows with zeros, negative zeros and a wide range of magnitudes; and index_score / sparse_attn on the
// packed rows the same bits as on the BF16 ones.
//   nvcc -O3 -std=c++17 -arch=native -I src/ds4 tests/ds4_kv_fp4_parity.cu src/ds4/ds4_kernels.cu -o build/ds4_kv_fp4_parity
#include "ds4_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

#define CK(x)                                                                         \
    do {                                                                              \
        const cudaError_t e_ = (x);                                                   \
        if (e_ != cudaSuccess) {                                                      \
            std::printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

namespace {

float fp4_host(unsigned c) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8u) ? -mag[c & 7u] : mag[c & 7u];
}
float e4m3_host(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, (int) e - 7);
    return (b & 0x80u) ? -v : v;
}
uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// rows of len values: normal, scaled per row and per group over ~2^-20 .. 2^12, with zeros and tiny negatives
std::vector<float> rows(int n, int len, std::mt19937& rng) {
    std::normal_distribution<float> nd;
    std::uniform_int_distribution<int> ex(-20, 12), pick(0, 15);
    std::vector<float> v((size_t) n * len);
    for (int r = 0; r < n; ++r) {
        const float rs = std::ldexp(1.0f, ex(rng));
        for (int g = 0; g < len; g += 16) {
            const float gs = rs * std::ldexp(1.0f, pick(rng) - 8);
            for (int i = g; i < g + 16 && i < len; ++i) {
                const int k = pick(rng);
                v[(size_t) r * len + i] = k == 0 ? 0.0f : k == 1 ? -1e-30f : nd(rng) * gs;
            }
        }
    }
    return v;
}

// finish(quant) of the rows into BF16 and into packed rows; every packed value against its BF16 one
int finish_parity(int n, int len, int quant, std::mt19937& rng, uint16_t** bf_out, uint8_t** pk_out) {
    const std::vector<float> h = rows(n, len, rng);
    const size_t pr = fp4_row_bytes(len, quant);
    float* x = nullptr;
    uint16_t* bf = nullptr;
    uint8_t* pk = nullptr;
    CK(cudaMalloc(&x, h.size() * 4));
    CK(cudaMalloc(&bf, h.size() * 2));
    CK(cudaMalloc(&pk, (size_t) n * pr));
    CK(cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    Finish f;
    f.quant = quant;
    f.dst = bf;
    f.dst_stride = len;
    finish(x, n, len, len, f, nullptr);
    Finish p;
    p.quant = quant;
    p.pdst = pk;
    p.pdst_stride = (long long) pr;
    finish(x, n, len, len, p, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> hb(h.size());
    std::vector<uint8_t> hp((size_t) n * pr);
    std::vector<float> hx(h.size());
    CK(cudaMemcpy(hb.data(), bf, hb.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hp.data(), pk, hp.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx.data(), x, hx.size() * 4, cudaMemcpyDeviceToHost));
    const int g = quant == 2 ? 16 : 32;
    long long bad = 0, negz = 0;
    for (int r = 0; r < n; ++r) {
        const uint8_t* row = hp.data() + (size_t) r * pr;
        for (int i = 0; i < len; ++i) {
            const unsigned c = (row[i / 2] >> (4 * (i & 1))) & 15u;
            const unsigned sb = row[len / 2 + i / g];
            const float sc = quant == 2 ? e4m3_host(sb) : std::ldexp(1.0f, (int) sb - 127);
            const float v = fp4_host(c) * sc;
            const uint32_t want = (uint32_t) hb[(size_t) r * len + i] << 16;
            if (bits(v) != want && bad++ < 5)
                std::printf("  row %d value %d: packed %g (%08x), BF16 %08x\n", r, i, v, bits(v), want);
            negz += want == 0x80000000u;
        }
    }
    const bool same_x = std::memcmp(hx.data(), h.data(), h.size() * 4) == 0;   // (both leave x as it was)
    std::printf("finish quant %d, %d rows of %d: %lld of %lld values differ (%lld negative zeros among them all); x %s\n",
                quant, n, len, bad, (long long) n * len, negz, same_x ? "untouched" : "CHANGED");
    CK(cudaFree(x));
    *bf_out = bf;
    *pk_out = pk;
    return bad == 0 && same_x ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(20261019);
    int fails = 0;
    // the index keys: quant 3, 128 values; index_score on BF16 against packed keys
    {
        const int n = 6000, hd = 128, heads = 32, T = 5;
        uint16_t* kb = nullptr;
        uint8_t* kp = nullptr;
        fails += finish_parity(n, hd, 3, rng, &kb, &kp);
        std::normal_distribution<float> nd;
        std::vector<float> q((size_t) T * heads * hd), w((size_t) T * 64);
        for (float& v : q) v = nd(rng);
        for (float& v : w) v = nd(rng);
        float *dq, *dw, *o1, *o2;
        CK(cudaMalloc(&dq, q.size() * 4));
        CK(cudaMalloc(&dw, w.size() * 4));
        CK(cudaMalloc(&o1, (size_t) T * n * 4));
        CK(cudaMalloc(&o2, (size_t) T * n * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        Rows lim;
        for (int t = 0; t < kWin; ++t) {
            lim.n[t] = n - 7 * t;
            lim.k[t] = 512;
        }
        index_score(dq, dw, kb, nullptr, heads, hd, T, n, nullptr, 0, 1, lim, o1, n, nullptr);
        index_score(dq, dw, nullptr, kp, heads, hd, T, n, nullptr, 0, 1, lim, o2, n, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t) T * n), b((size_t) T * n);
        CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        std::printf("index_score, %d positions x %d queries: %s\n", n, T, same ? "bit for bit" : "DIFFERENT");
        fails += !same;
    }
    // the compressed KV: quant 2, 512 values; sparse_attn on BF16 against packed rows (and the window rows BF16 in both)
    {
        const int n = 3000, d = 512, heads = 8, T = 5, win = 128;
        uint16_t* cb = nullptr;
        uint8_t* cp = nullptr;
        fails += finish_parity(n, d, 2, rng, &cb, &cp);
        std::normal_distribution<float> nd;
        std::uniform_int_distribution<int> pos(0, n - 1);
        std::vector<float> q((size_t) T * heads * d), sink(heads);
        for (float& v : q) v = nd(rng) * 0.05f;
        for (float& v : sink) v = nd(rng);
        std::vector<uint16_t> ext((size_t) (win + T) * d);
        for (uint16_t& v : ext) {
            const float f = nd(rng);
            uint32_t u;
            std::memcpy(&u, &f, 4);
            v = (uint16_t) (u >> 16);
        }
        AttnRows ar;
        const int nsel[5] = {512, 300, 1, 0, 477};
        std::vector<int> sel((size_t) T * 512, 0);
        for (int t = 0; t < T; ++t) {
            ar.win0[t] = t + 1;
            ar.nwin[t] = win;
            ar.nsel[t] = nsel[t];
            for (int i = 0; i < nsel[t]; ++i) sel[(size_t) t * 512 + i] = pos(rng);
        }
        float *dq, *dsink, *o1, *o2;
        uint16_t* dext;
        int* dsel;
        CK(cudaMalloc(&dq, q.size() * 4));
        CK(cudaMalloc(&dsink, sink.size() * 4));
        CK(cudaMalloc(&dext, ext.size() * 2));
        CK(cudaMalloc(&dsel, sel.size() * 4));
        CK(cudaMalloc(&o1, q.size() * 4));
        CK(cudaMalloc(&o2, q.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsink, sink.data(), sink.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dext, ext.data(), ext.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
        const float scale = 1.0f / std::sqrt((float) d);
        sparse_attn(dq, dext, cb, nullptr, dsel, 512, ar, dsink, heads, d, T, scale, o1, nullptr);
        sparse_attn(dq, dext, nullptr, cp, dsel, 512, ar, dsink, heads, d, T, scale, o2, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> a(q.size()), b(q.size());
        CK(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        std::printf("sparse_attn, %d heads x %d queries (window and selected rows): %s\n", heads, T,
                    same ? "bit for bit" : "DIFFERENT");
        if (!same) {   // (how: the first value apart, the largest gap, and whether BF16 against itself agrees)
            size_t first = 0, nd_ = 0;
            double gap = 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (bits(a[i]) != bits(b[i])) {
                    if (nd_++ == 0) first = i;
                    gap = std::fmax(gap, std::fabs((double) a[i] - b[i]) / (std::fabs((double) a[i]) + 1e-30));
                }
            sparse_attn(dq, dext, cb, nullptr, dsel, 512, ar, dsink, heads, d, T, scale, o2, nullptr);
            CK(cudaDeviceSynchronize());
            std::vector<float> c(q.size());
            CK(cudaMemcpy(c.data(), o2, c.size() * 4, cudaMemcpyDeviceToHost));
            std::printf("  %zu of %zu apart, the first at %zu (query %zu head %zu): %.9g vs %.9g; largest relative gap %.3g;"
                        " BF16 run twice: %s\n", nd_, a.size(), first, first / ((size_t) heads * d),
                        first / d % heads, a[first], b[first], gap,
                        std::memcmp(a.data(), c.data(), a.size() * 4) == 0 ? "the same" : "DIFFERENT");
        }
        fails += !same;
    }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
