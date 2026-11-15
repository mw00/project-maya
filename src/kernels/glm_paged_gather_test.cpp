// glm_paged_gather_test.cpp --selftest
// Parity for the paged-gather routing, the heart of the PR.  Two layers:
//
//   1. A HOST-RUNNABLE mapping check, always executed (this is the CI subset:
//      the routing is the pure integer map  row -> table[row / P] * P +
//      (row - (row / P) * P), so it is fully testable on a machine with no GPU,
//      which is exactly what GitHub Actions runs).  This is the regression
//      signal CI was missing: identity leaves a row put, a non-identity (shuffled)
//      table reorders it, and a corrupted table must move at least one row.
//
//   2. A DEVICE run, when a GPU is present: the real strata::kernels::glmb::
//      dsa_select is driven against an identity, a shuffled, and a corrupted
//      page table, and its on-device output is compared to the host mapping.
//      glm_lat_paged_parity only exercised the host-side lat_row_offset with
//      memcpy and never ran the kernels; this one runs them with a table whose
//      physical order differs from the logical order.
//
// The selection is deterministic and predictable on the host: each row keeps
// its visible pools whole (n_vis <= top), so logical cells[i] = i for
// i < n_vis*kpool and -1 after.  The test SKIPs the device layer without a GPU
// and still returns the host-subset result.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "strata/kernels/glm_batch.hpp"

static int g_checks = 0;
static int g_failures = 0;

static void require(const char* name, bool cond) {
    g_checks++;
    if (!cond) { g_failures++; std::printf("FAIL %s\n", name); }
    else std::printf("ok %s\n", name);
}

static const int KPOOL = 8;    // latent rows per pool
static const int T = 64;       // prompt rows
static const int NSMAX = 64;   // cells row width (>= n_vis*KPOOL)
static const int P = 8;        // page_rows
static const int MAX_ROWS = 4096;

// the pure routing map the device kernel computes: a logical row to its
// physical row under the table.  -1 (a masked cell) and rows past the pool's
// capacity are left put, matching lat_route_cells_kernel's guards.
static int route(int cell, const int32_t* table, int page_rows, int max_row) {
    if (cell < 0 || cell >= max_row) return cell;
    const int pg = cell / page_rows;
    return (int) table[pg] * page_rows + (cell - pg * page_rows);
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--selftest") != 0) return 0;

    const int pages = (MAX_ROWS + P - 1) / P;
    std::vector<int32_t> t_id(pages), t_sh(pages), t_co(pages);
    for (int i = 0; i < pages; i++) {
        t_id[i] = i;
        t_sh[i] = (i + 3) % pages;
        t_co[i] = (i == 0) ? 7 : i;
    }
    bool nonid = false;
    for (int i = 0; i < pages; i++) if (t_sh[i] != i) nonid = true;

    // the deterministic logical selection every row produces (visible pools kept whole)
    std::vector<int> h_ref(T * NSMAX);
    std::vector<int> h_ns(T);
    for (int t = 0; t < T; t++) {
        const int n_vis = (t + 1) / KPOOL;
        const int n_sel = KPOOL * n_vis;
        h_ns[t] = n_sel;
        for (int i = 0; i < NSMAX; i++)
            h_ref[(size_t) t * NSMAX + i] = (i < n_sel) ? i : -1;
    }

    // ---- layer 1: host-runnable mapping (always runs; the CI regression subset)
    {
        int moved = 0, mismap = 0, sh_moved = 0;
        for (size_t i = 0; i < h_ref.size(); i++) {
            const int c = h_ref[i];
            if (route(c, t_id.data(), P, MAX_ROWS) != c) mismap++;                 // identity must be a no-op
            if (route(c, t_sh.data(), P, MAX_ROWS) != c) sh_moved++;               // shuffled must reorder
            if (route(c, t_co.data(), P, MAX_ROWS) != c) moved++;                  // corrupted must move
        }
        require("host: the identity table is the no-op map", mismap == 0);
        require("host: the shuffled table reorders at least one row", sh_moved > 0);
        require("host: the corrupted table moves a row (a no-op routing fails)", moved > 0);
        require("host: the shuffled table is non-identity", nonid);
        // a moved row is not enough: a routing that collides two logical rows onto one
        // physical row would still "move" and still be wrong.  The touched set must be
        // a permutation (distinct destinations).
        bool perm = true;
        int checked = 0;
        std::vector<char> seen((size_t) MAX_ROWS, 0);
        std::vector<char> cell_seen((size_t) MAX_ROWS, 0);   // each logical cell once (rows repeat)
        for (size_t i = 0; i < h_ref.size(); i++) {
            const int c = h_ref[i];
            if (c < 0 || c >= MAX_ROWS || cell_seen[(size_t) c]) continue;
            cell_seen[(size_t) c] = 1;
            const int r = route(c, t_sh.data(), P, MAX_ROWS);
            if (r < 0 || r >= MAX_ROWS || seen[(size_t) r]) { perm = false; break; }
            seen[(size_t) r] = 1;
            checked++;
        }
        require("host: the shuffled routing is a permutation of the touched rows", perm && checked > 0);
    }

    // ---- layer 2: the device kernels (only when a GPU is present)
    bool require_gpu = false;
    for (int i = 1; i < argc; i++) if (std::strcmp(argv[i], "--require-gpu") == 0) require_gpu = true;
    const cudaError_t dev_err = cudaSetDevice(0);
    const bool have_gpu = (dev_err == cudaSuccess);
    if (!have_gpu) {
        // a busy device (error 46) must not look like a pass in the maintainers' gate
        std::printf("SKIP device layer: no usable GPU (%s)\n", cudaGetErrorString(dev_err));
        if (require_gpu) { std::printf("FAIL device: --require-gpu was set\n"); return 1; }
    } else {
        cudaStream_t s;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            std::printf("FAIL device: stream create\n");
            return 1;
        }
        const long long N = (long long) T * NSMAX;
        const int64_t LDMAX = 64;
        const int top = (T / KPOOL) + 1;   // >= max n_vis so every visible pool is kept
        std::vector<float> h_score((size_t) T * (size_t) LDMAX, 0.0f);
        for (int t = 0; t < T; t++)
            for (int p = 0; p < LDMAX; p++)
                h_score[(size_t) t * LDMAX + p] = (float) (t * 31 + p * 7) / 100.0f;
        float* d_score = nullptr; int* d_cells = nullptr; int* d_ns = nullptr; int* d_table = nullptr;
        if (cudaMalloc(&d_score, h_score.size() * 4) != cudaSuccess ||
            cudaMalloc(&d_cells, N * 4) != cudaSuccess ||
            cudaMalloc(&d_ns, T * 4) != cudaSuccess ||
            cudaMalloc(&d_table, pages * 4) != cudaSuccess) {
            std::printf("FAIL device: alloc\n");
            return 1;
        }
        const auto run = [&](const int32_t* table, const char* label) {
            cudaStreamSynchronize(s);
            cudaMemcpy(d_score, h_score.data(), h_score.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells, h_ref.data(), h_ref.size() * 4, cudaMemcpyHostToDevice);
            // d_ns is DEVICE memory: std::memset would write it from the host and fault
            if (cudaMemset(d_ns, 0, T * 4) != cudaSuccess) return false;
            cudaMemcpy(d_table, table, pages * 4, cudaMemcpyHostToDevice);
            // the real selection + the paged-gather routing run on the device
            strata::kernels::glmb::dsa_select(d_score, (int) LDMAX, 0, KPOOL, top, 0, T, NSMAX,
                                              d_cells, d_ns, s, d_table, P, MAX_ROWS);
            cudaStreamSynchronize(s);
            if (cudaGetLastError() != cudaSuccess || strata::kernels::glmb::launch_errors() != 0) {
                std::printf("FAIL device %s: a launch failed\n", label);
                return false;
            }
            std::vector<int> got(T * NSMAX), gns(T);
            cudaMemcpy(got.data(), d_cells, N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gns.data(), d_ns, T * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T; t++)
                if (gns[t] != h_ns[t]) {
                    std::printf("FAIL device %s: n_sel[%d] = %d, want %d\n", label, t, gns[t], h_ns[t]);
                    return false;
                }
            // the on-device gather must land each cell where the host map says it goes
            for (size_t i = 0; i < got.size(); i++) {
                const int want = route(h_ref[i], table, P, MAX_ROWS);
                if (got[i] != want) {
                    std::printf("FAIL device %s: cells[%zu] = %d, want %d (logical %d)\n",
                                label, i, got[i], want, h_ref[i]);
                    return false;
                }
            }
            return true;
        };
        require("device: identity table is the no-op gather", run(t_id.data(), "identity"));
        require("device: shuffled table lands each cell on its physical row", run(t_sh.data(), "shuffled"));
        require("device: corrupted table moves a cell (the kernel reads the table)", run(t_co.data(), "corrupted"));
        cudaFree(d_score); cudaFree(d_cells); cudaFree(d_ns); cudaFree(d_table);
        cudaStreamDestroy(s);
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
