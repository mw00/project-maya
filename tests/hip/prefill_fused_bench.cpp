// The GLM fused prompt experts' speed at a real layer's shape: 288 experts, 8 routed per token, IQ2_XXS gate/up and
// IQ3_XXS down by default (unsloth's UD-IQ2_XXS), random valid blocks, a token's experts drawn uniformly.
//
//   hip_prefill_fused_bench [tokens=9139] [reps=10] [gu_type=16] [down_type=18]
//
// Prints the median time of one layer's experts (gate/up + down, as rf::experts launches them), its int8 TOPS and an
// FNV-1a hash of the output's bits (the same hash = the same arithmetic).
#include "strata/prefill/moe_fused_rdna4.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace rf = strata::prefill::rdna4;
int E = 4096, FF = 2048, NE = 288, TOPK = 8;   // GLM-5.3-Flash's; `ds4` as the 5th argument: DeepSeek-V4.1's

void ck(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(hipMalloc(&p, n)); }
    ~Dev() { (void) hipFree(p); }
    template <class T> T* as() { return static_cast<T*>(p); }
};

// random bytes with a valid fp16 block scale in front of every block (as the parity test fills them)
void fill(uint8_t* dst, ggml_type type, int rows, int cols, uint64_t& s) {
    const size_t bs = ggml_type_size(type), nb = (size_t) rows * cols / ggml_blck_size(type);
    for (size_t i = 0; i < nb * bs; i += 8) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        std::memcpy(dst + i, &s, std::min<size_t>(8, nb * bs - i));
    }
    for (size_t b = 0; b < nb; ++b) {
        const auto h = ggml_fp32_to_fp16(0.0008f * (1.0f + (float) (b % 500) / 100.0f));
        std::memcpy(dst + b * bs, &h, 2);
    }
}

int main(int argc, char** argv) {
    try {
        const int T = argc > 1 ? std::atoi(argv[1]) : 9139, reps = argc > 2 ? std::atoi(argv[2]) : 10;
        const auto gu_type = (ggml_type) (argc > 3 ? std::atoi(argv[3]) : GGML_TYPE_IQ2_XXS);
        const auto d_type = (ggml_type) (argc > 4 ? std::atoi(argv[4]) : GGML_TYPE_IQ3_XXS);
        if (argc > 5 && std::strcmp(argv[5], "ds4") == 0) E = 5120, FF = 2304, NE = 384, TOPK = 6;
        ck(hipSetDevice(0));
        if (!rf::supported(gu_type, d_type)) {
            std::printf("SKIP: fused experts unsupported here for %s/%s\n", ggml_type_name(gu_type), ggml_type_name(d_type));
            return 77;
        }
        rf::Geometry geo{gu_type, d_type, ggml_row_size(gu_type, E), ggml_row_size(d_type, FF), 0, 0, 100.0f};
        geo.n_embd = E == 4096 ? 0 : E;
        geo.n_ff = FF == 2048 ? 0 : FF;
        geo.up_off = FF * geo.gu_row;
        geo.down_off = 2 * geo.up_off;
        const size_t unit = std::lcm(ggml_type_size(gu_type), ggml_type_size(d_type));
        const size_t stride = (geo.down_off + E * geo.d_row + unit - 1) / unit * unit;
        std::vector<uint8_t> w(stride);
        Dev dw(NE * stride);
        uint64_t seed = 0x9e3779b97f4a7c15ull;
        for (int e = 0; e < NE; ++e) {
            fill(w.data(), gu_type, 2 * FF, E, seed);
            fill(w.data() + geo.down_off, d_type, E, FF, seed);
            ck(hipMemcpy(dw.as<uint8_t>() + (size_t) e * stride, w.data(), stride, hipMemcpyHostToDevice));
        }
        // each token's 8 distinct experts, uniformly; the rows sorted by expert, src = the token of each row
        std::mt19937 rng(1570);
        std::vector<std::vector<int>> by(NE);
        std::vector<int> pick(NE);
        std::iota(pick.begin(), pick.end(), 0);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < TOPK; ++k) std::swap(pick[k], pick[k + rng() % (NE - k)]);
            for (int k = 0; k < TOPK; ++k) by[pick[k]].push_back(t);
        }
        std::vector<int> bounds(NE + 1, 0), src;
        for (int e = 0; e < NE; ++e) {
            src.insert(src.end(), by[e].begin(), by[e].end());
            bounds[e + 1] = (int) src.size();
        }
        const int R = (int) src.size();
        std::vector<float> x((size_t) T * E);
        for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        int ntiles = 0;
        for (int e = 0; e < NE; ++e) ntiles += (bounds[e + 1] - bounds[e] + 63) / 64;
        Dev dx(x.size() * 4), ds((size_t) R * 4), db(bounds.size() * 4), xa(rf::act_bytes(T, E)), ha(rf::act_bytes(R, FF)),
            out((size_t) R * E * 4), tiles((size_t) 2 * (ntiles + 8) * sizeof(int));
        ck(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice));
        ck(hipMemcpy(ds.p, src.data(), (size_t) R * 4, hipMemcpyHostToDevice));
        ck(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice));
        rf::quantize(dx.as<float>(), T, E, xa.p, nullptr);
        const auto layer = [&] {
            rf::experts(dw.as<uint8_t>(), stride, geo, NE, db.as<int>(), bounds.data(), tiles.p, xa.p, ds.as<int>(), ha.p,
                        out.as<float>(), nullptr);
        };
        layer();
        ck(hipDeviceSynchronize());
        hipEvent_t a, b;
        ck(hipEventCreate(&a));
        ck(hipEventCreate(&b));
        std::vector<float> ms;
        for (int i = 0; i < reps; ++i) {
            ck(hipEventRecord(a, nullptr));
            layer();
            ck(hipEventRecord(b, nullptr));
            ck(hipEventSynchronize(b));
            float v = 0;
            ck(hipEventElapsedTime(&v, a, b));
            ms.push_back(v);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2], ops = 2.0 * R * ((double) 2 * FF * E + (double) E * FF);
        std::vector<uint32_t> o((size_t) R * E);
        ck(hipMemcpy(o.data(), out.p, o.size() * 4, hipMemcpyDeviceToHost));
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t v : o) h = (h ^ v) * 0x100000001b3ull;
        std::printf("%s/%s T=%d rows=%d tiles=%d: %.2f ms (min %.2f, max %.2f), %.1f TOPS, hash %016llx\n",
                    ggml_type_name(gu_type), ggml_type_name(d_type), T, R, ntiles, med, ms.front(), ms.back(),
                    ops / (med * 1e-3) / 1e12, (unsigned long long) h);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
