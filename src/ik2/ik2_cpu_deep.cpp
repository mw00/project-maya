// NVFP4 and Q8_0 host GEMV for deep-context decode on AVX2 systems.
// Eight row tiles share each broadcast input. Within a row the multiply,
// FMA, sub-block accumulation and warp reduction follow the sm_86 kernels.
#include "ik2_cpu_deep.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

namespace ik2::cpu {
struct alignas(32) Block {
    int8_t w[16][8];
    float d[8];
};
inline __m256 w8(const int8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
// PTX multiplies the second term before fusing the first term. Changing
// their order changes the final FP32 logits on this checkpoint.
inline __m256 four(const Block& b, const float* x, int i) {
    __m256 a = _mm256_mul_ps(w8(b.w[i + 1]), _mm256_set1_ps(x[i + 1]));
    a = _mm256_fmadd_ps(w8(b.w[i]), _mm256_set1_ps(x[i]), a);
    a = _mm256_fmadd_ps(w8(b.w[i + 2]), _mm256_set1_ps(x[i + 2]), a);
    return _mm256_fmadd_ps(w8(b.w[i + 3]), _mm256_set1_ps(x[i + 3]), a);
}
static void nvfp4_rows8(const Block* p, const float* x, int K, float* out) {
    __m256 lane[32];
    for (auto& v : lane)
        v = _mm256_setzero_ps();
    for (int s = 0; s < K / 16; ++s) {
        const Block& b = p[s];
        const float* v = x + 16 * s;
        auto a = _mm256_add_ps(four(b, v, 0), four(b, v, 4));
        a = _mm256_add_ps(a, four(b, v, 8));
        a = _mm256_add_ps(a, four(b, v, 12));
        lane[s % 32] = _mm256_fmadd_ps(a, _mm256_load_ps(b.d), lane[s % 32]);
    }
    for (int step = 16; step > 0; step /= 2)
        for (int i = 0; i < step; ++i)
            lane[i] = _mm256_add_ps(lane[i], lane[i + step]);
    _mm256_storeu_ps(out, lane[0]);
}

struct alignas(32) Q8Block {
    int8_t w[32][8];
    float d[8];
};
inline __m256 q8_four(const Q8Block& b, const float* x, int t) {
    auto a = _mm256_mul_ps(w8(b.w[t + 1]), _mm256_set1_ps(x[t + 1]));
    a = _mm256_fmadd_ps(w8(b.w[t]), _mm256_set1_ps(x[t]), a);
    a = _mm256_fmadd_ps(w8(b.w[t + 2]), _mm256_set1_ps(x[t + 2]), a);
    return _mm256_fmadd_ps(w8(b.w[t + 3]), _mm256_set1_ps(x[t + 3]), a);
}
static void q8_rows(const Q8Block* p, const float* x, int K, float* out) {
    __m256 lane[32];
    for (auto& a : lane)
        a = _mm256_setzero_ps();
    for (int sb = 0; sb < K / 32; sb += 2) {
        const float* v = x + sb * 32;
        const Q8Block& a = p[sb];
        const Q8Block& b = p[sb + 1];
        auto g = _mm256_mul_ps(w8(a.w[1]), _mm256_set1_ps(v[1]));
        g = _mm256_fmadd_ps(w8(a.w[0]), _mm256_set1_ps(v[0]), g);
        for (int t = 2; t < 30; t += 4)
            g = _mm256_add_ps(g, q8_four(a, v, t));
        auto z = _mm256_mul_ps(w8(a.w[31]), _mm256_set1_ps(v[31]));
        z = _mm256_fmadd_ps(w8(a.w[30]), _mm256_set1_ps(v[30]), z);
        g = _mm256_add_ps(g, z);
        auto& val = lane[(sb / 2) % 32];
        val = _mm256_fmadd_ps(g, _mm256_load_ps(a.d), val);
        v += 32;
        g = q8_four(b, v, 0);
        for (int t = 4; t < 32; t += 4)
            g = _mm256_add_ps(g, q8_four(b, v, t));
        val = _mm256_fmadd_ps(g, _mm256_load_ps(b.d), val);
    }
    for (int step = 16; step > 0; step /= 2)
        for (int i = 0; i < step; ++i)
            lane[i] = _mm256_add_ps(lane[i], lane[i + step]);
    _mm256_storeu_ps(out, lane[0]);
}
struct PackedLayer {
    int C, F, K, E, nthreads;
    bool q8;
    Block* parts[3];
    const int* picks;
    const float* x;
    float *gu, *y;
    const uint16_t* hidden;
    float* hidden32;
    ~PackedLayer() {
        for (auto* ptr : parts)
            std::free(ptr);
        std::free(hidden32);
    }
};
std::shared_ptr<PackedLayer> pack_layer(int C, int F, int K, int E, const uint8_t* const raw[3], const size_t stride[3],
                                        int nt, bool q8, TransferBuffers buffers) {
    auto L = std::shared_ptr<PackedLayer>(new PackedLayer{C,
                                                          F,
                                                          K,
                                                          E,
                                                          nt,
                                                          q8,
                                                          {},
                                                          buffers.picks,
                                                          buffers.input,
                                                          buffers.gate_up,
                                                          buffers.output,
                                                          buffers.hidden,
                                                          nullptr});
    if (posix_memalign((void**)&L->hidden32, 64, (size_t)K * F * sizeof(float))) throw std::bad_alloc();
    const int8_t table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    float deq[256];
    for (int i = 0; i < 256; ++i) {
        int e = (i >> 3) & 15, m = i & 7;
        deq[i] =
            i == 0 || i == 127 ? 0.f : (e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7)) * .5f;
    }
    for (int p = 0; p < 3; ++p) {
        int n = p == 2 ? C : F, k = p == 2 ? F : C;
        size_t tiles = (size_t)E * (n / 8);
        if (q8) {
            Q8Block* data;
            if (posix_memalign((void**)&data, 64, tiles * (k / 32) * sizeof(Q8Block))) throw std::bad_alloc();
            L->parts[p] = (Block*)data;
#pragma omp parallel for num_threads(nt) schedule(static)
            for (size_t tile = 0; tile < tiles; ++tile) {
                const int expert = tile / (n / 8), r0 = (tile % (n / 8)) * 8;
                Q8Block* dst = data + tile * (k / 32);
                for (int sb = 0; sb < k / 32; ++sb)
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t* q = raw[p] + expert * stride[p] + ((r0 + j) * (k / 32) + sb) * 34;
                        uint16_t h;
                        std::memcpy(&h, q, sizeof(h));
                        dst[sb].d[j] = _cvtsh_ss(h);
                        for (int z = 0; z < 32; ++z)
                            dst[sb].w[z][j] = (int8_t)q[2 + z];
                    }
            }
            continue;
        }
        if (posix_memalign((void**)&L->parts[p], 64, tiles * (k / 16) * sizeof(Block))) throw std::bad_alloc();
#pragma omp parallel for num_threads(nt) schedule(static)
        for (size_t tile = 0; tile < tiles; ++tile) {
            const int expert = tile / (n / 8), r0 = (tile % (n / 8)) * 8;
            const size_t rb = k / 64 * 36;
            Block* dst = L->parts[p] + tile * (k / 16);
            for (int s = 0; s < k / 16; ++s) {
                Block& b = dst[s];
                for (int j = 0; j < 8; ++j) {
                    const uint8_t* q = raw[p] + expert * stride[p] + (r0 + j) * rb + (s / 4) * 36;
                    b.d[j] = deq[q[s % 4]];
                    q += 4 + (s % 4) * 8;
                    for (int z = 0; z < 8; ++z) {
                        uint8_t v = q[z];
                        b.w[z][j] = table[v & 15];
                        b.w[z + 8][j] = table[v >> 4];
                    }
                }
            }
        }
    }
    return L;
}
void gate_up_rows(void* payload) {
    const auto& L = *static_cast<PackedLayer*>(payload);
    const int tiles = L.F / 8;
#pragma omp parallel for num_threads(L.nthreads) schedule(static)
    for (int r = 0; r < L.K * 2 * tiles; ++r) {
        int pick = r / (2 * tiles), part = (r / tiles) % 2, local = r % tiles, expert = L.picks[pick];
        if (L.q8) {
            const auto* a = (Q8Block*)L.parts[part] + ((size_t)expert * tiles + local) * (L.C / 32);
            q8_rows(a, L.x, L.C, L.gu + 8 * r);
            continue;
        }
        const Block* b = L.parts[part] + ((size_t)expert * tiles + local) * (L.C / 16);
        nvfp4_rows8(b, L.x, L.C, L.gu + 8 * r);
    }
}
void down_rows(void* payload) {
    const auto& L = *static_cast<PackedLayer*>(payload);
    const int tiles = L.C / 8;
    for (int j = 0; j < L.K * L.F; j += 8)
        _mm256_storeu_ps(L.hidden32 + j, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(L.hidden + j))));
#pragma omp parallel for num_threads(L.nthreads) schedule(static)
    for (int r = 0; r < L.K * tiles; ++r) {
        int pick = r / tiles, expert = L.picks[pick];
        if (L.q8) {
            const auto* a = (Q8Block*)L.parts[2] + ((size_t)expert * tiles + r % tiles) * (L.F / 32);
            q8_rows(a, L.hidden32 + (size_t)pick * L.F, L.F, L.y + 8 * r);
            continue;
        }
        const Block* b = L.parts[2] + ((size_t)expert * tiles + r % tiles) * (L.F / 16);
        nvfp4_rows8(b, L.hidden32 + (size_t)pick * L.F, L.F, L.y + 8 * r);
    }
}

} // namespace ik2::cpu
