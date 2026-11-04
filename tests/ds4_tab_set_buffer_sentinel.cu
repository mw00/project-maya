// Standalone sentinel for DS4's host-mapped tab_set input reuse hazard; deliberately not a CTest.
// The unguarded half intentionally demonstrates the race. The tab_set kernel is intentionally
// identical to ds4_kernels.cu's three-line kernel.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void tab_set_kernel(const unsigned long long* __restrict__ pairs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *(unsigned long long*) pairs[2 * i] = pairs[2 * i + 1];
}

__global__ void delay_kernel(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
    std::exit(3);
}

std::vector<unsigned long long> run(bool guarded) {
    unsigned long long* table = nullptr;
    unsigned long long* pairs = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaMalloc(&table, 4 * sizeof(*table)), "cudaMalloc table");
    check(cudaHostAlloc(&pairs, 4 * sizeof(*pairs), cudaHostAllocMapped), "cudaHostAlloc pairs");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    const std::vector<unsigned long long> initial = {1000, 1001, 1002, 1003};
    check(cudaMemcpy(table, initial.data(), 4 * sizeof(*table), cudaMemcpyHostToDevice), "initial table copy");
    delay_kernel<<<1, 1, 0, stream>>>(500000000LL);

    for (int batch = 0; batch < 2; ++batch) {
        if (guarded && batch > 0) check(cudaStreamSynchronize(stream), "guarded predecessor sync");
        if (batch == 0) {
            pairs[0] = (unsigned long long) (table + 0);
            pairs[1] = 0;       // A-only eviction that B must not erase
            pairs[2] = (unsigned long long) (table + 1);
            pairs[3] = 111;
        } else {
            pairs[0] = (unsigned long long) (table + 2);
            pairs[1] = 222;     // B-only values, which overwrite A's mapped input when unguarded
            pairs[2] = (unsigned long long) (table + 3);
            pairs[3] = 333;
        }
        tab_set_kernel<<<1, 128, 0, stream>>>(pairs, 2);
    }
    check(cudaStreamSynchronize(stream), "final sync");
    std::vector<unsigned long long> out(4);
    check(cudaMemcpy(out.data(), table, 4 * sizeof(*table), cudaMemcpyDeviceToHost), "result copy");
    std::printf("%s table: %llu %llu %llu %llu\n", guarded ? "GUARDED" : "UNGUARDED", out[0], out[1], out[2],
                out[3]);
    check(cudaStreamDestroy(stream), "stream destroy");
    check(cudaFreeHost(pairs), "free pairs");
    check(cudaFree(table), "free table");
    return out;
}

int main() {
    const auto unguarded = run(false);
    const bool loss = unguarded[0] == 1000 && unguarded[1] == 1001 && unguarded[2] == 222 && unguarded[3] == 333;
    std::printf("%s: A's zero/111 updates were overwritten by B's mapped input\n",
                loss ? "UNGUARDED_LOSS_OBSERVED" : "UNGUARDED_UNEXPECTED_RESULT");
    const auto guarded = run(true);
    const bool pass = guarded[0] == 0 && guarded[1] == 111 && guarded[2] == 222 && guarded[3] == 333;
    std::printf("%s: predecessor sync preserved both intended batches\n", pass ? "GUARDED_PASS" : "GUARDED_FAIL");
    return loss && pass ? 0 : 2;
}
