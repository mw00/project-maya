// tests/glm_lat8_test.cu - the INT8 latent records' FP16 reads (src/kernels/cuda/glm_lat8.cuh) against the conversion
// they replaced, (half) ((float) q * (float) d): every code with every finite FP16 scale, bit for bit (needs a GPU).
#include "../src/kernels/cuda/glm_lat8.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::kernels::glmb;

namespace {

constexpr int KV = 512, REC = KV + KV / 16;   // a record: 512 codes, 16 FP16 scales (544 bytes)

// a thread: one record's values [8 c8, 8 c8 + 8), through lat8_h8 and lat8_h4, against the F32 product rounded once;
// the first 8 values that differ go to ex (code, scale bits, got, want)
__global__ void check_kernel(const uint8_t* recs, int n_rec, unsigned long long* bad, unsigned long long* seen,
                             unsigned* ex) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_rec * (KV / 8)) return;
    const int cell = (int) (i / (KV / 8)), c8 = (int) (i % (KV / 8));
    const uint8_t* rec = recs + (size_t) REC * cell;
    const __half d = ((const __half*) (rec + KV))[c8 >> 2];
    if ((__half_as_ushort(d) & 0x7C00u) == 0x7C00u) return;   // Inf and NaN are no scale
    const uint4 v = lat8_h8(rec, KV, c8);
    const uint2 v0 = lat8_h4(rec, KV, 2 * c8), v1 = lat8_h4(rec, KV, 2 * c8 + 1);
    const unsigned got8[4] = {v.x, v.y, v.z, v.w}, got4[4] = {v0.x, v0.y, v1.x, v1.y};
    const float df = __half2float(d);
    const signed char* q = (const signed char*) rec + 8 * c8;
    unsigned long long nbad = 0;
    for (int k = 0; k < 4; ++k) {
        const __half2 r = __floats2half2_rn((float) q[2 * k] * df, (float) q[2 * k + 1] * df);
#if defined(STRATA_LAT8_UNSIGNED_ZEROS)
        // (AMD: a zero has no sign in a record's read - the F32 product's own zero sign varies with the compiler)
        const unsigned ref = lat8_unsigned_zeros(*(const unsigned*) &r);
#else
        const unsigned ref = *(const unsigned*) &r;
#endif
        nbad += (got8[k] != ref) + (got4[k] != ref);
        for (int h = 0; h < 2; ++h)
            if (((got8[k] >> (16 * h)) & 0xFFFFu) != ((ref >> (16 * h)) & 0xFFFFu)) {
                const unsigned at = atomicAdd(ex, 1u);
                if (at < 8) {
                    ex[1 + 4 * at] = (unsigned) (int) q[2 * k + h];
                    ex[2 + 4 * at] = __half_as_ushort(d);
                    ex[3 + 4 * at] = (got8[k] >> (16 * h)) & 0xFFFFu;
                    ex[4 + 4 * at] = (ref >> (16 * h)) & 0xFFFFu;
                }
            }
    }
    if (nbad != 0) atomicAdd(bad, nbad);
    atomicAdd(seen, 1ull);
}

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "glm_lat8_test: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

int main() {
    // 4096 records of 16 scales hold every FP16 bit pattern once; pass p puts code (c + 32 p) & 255 at value c, so over
    // the 8 passes every scale meets all 256 codes
    const int n_rec = 65536 / (KV / 32);
    std::vector<uint8_t> h((size_t) REC * n_rec);
    uint8_t* d_rec = nullptr;
    unsigned long long* d_cnt = nullptr;
    unsigned* d_ex = nullptr;
    if (!ok(cudaMalloc((void**) &d_rec, h.size()), "cudaMalloc") ||
        !ok(cudaMalloc((void**) &d_cnt, 2 * sizeof(unsigned long long)), "cudaMalloc") ||
        !ok(cudaMalloc((void**) &d_ex, 33 * sizeof(unsigned)), "cudaMalloc") ||
        !ok(cudaMemset(d_ex, 0, 33 * sizeof(unsigned)), "cudaMemset"))
        return 1;
    unsigned long long bad = 0, seen = 0;
    for (int p = 0; p < 8; ++p) {
        for (int r = 0; r < n_rec; ++r) {
            uint8_t* rec = h.data() + (size_t) REC * r;
            for (int c = 0; c < KV; ++c) rec[c] = (uint8_t) ((c + 32 * p) & 255);
            for (int gi = 0; gi < KV / 32; ++gi) {
                const uint16_t bits = (uint16_t) (r * (KV / 32) + gi);
                std::memcpy(rec + KV + 2 * gi, &bits, 2);
            }
        }
        unsigned long long c[2] = {0, 0};
        const int64_t n = (int64_t) n_rec * (KV / 8);
        if (!ok(cudaMemcpy(d_rec, h.data(), h.size(), cudaMemcpyHostToDevice), "upload") ||
            !ok(cudaMemset(d_cnt, 0, 2 * sizeof(unsigned long long)), "cudaMemset"))
            return 1;
        check_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(d_rec, n_rec, d_cnt, d_cnt + 1, d_ex);
        if (!ok(cudaGetLastError(), "launch") || !ok(cudaMemcpy(c, d_cnt, sizeof c, cudaMemcpyDeviceToHost), "readback"))
            return 1;
        bad += c[0];
        seen += c[1];
    }
    unsigned ex[33] = {};
    if (!ok(cudaMemcpy(ex, d_ex, sizeof ex, cudaMemcpyDeviceToHost), "readback")) return 1;
    for (unsigned i = 0; i < ex[0] && i < 8; ++i)
        std::printf("  differs: code %d x scale 0x%04x: 0x%04x, the F32 product rounded once 0x%04x\n",
                    (int) ex[1 + 4 * i], ex[2 + 4 * i], ex[3 + 4 * i], ex[4 + 4 * i]);
    cudaFree(d_rec);
    cudaFree(d_cnt);
    cudaFree(d_ex);
    // 8 passes x 4096 records x 64 reads, less the 2048 scales that are Inf or NaN (x 32 values x 8 passes / 8 a read)
    const unsigned long long want = 8ull * n_rec * (KV / 8) - 2048ull * 32;
    std::printf("glm_lat8_test: %llu reads of 8 values (every code x every finite FP16 scale), %llu words differ%s\n",
                seen, bad, bad == 0 && seen == want ? " - OK" : " - FAIL");
    return bad == 0 && seen == want ? 0 : 1;
}
