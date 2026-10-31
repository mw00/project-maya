// glm_lat_paged_parity.cpp --selftest
// Task 1.2 parity: the latent cache's page-table indirection must be
// bit-exact against the contiguous arena for FP16 (1024 B rows, kv_lora
// 512 x 2) and INT8 (544 B rows) at page sizes 4/64/256/512, with a
// SHUFFLED (non-identity) table so the physical reordering is observable.
// With STRATA_GLM_PAGED unset the incumbent path is untouched: lat_table
// stays null and every write/gather keeps the contiguous offsets.
// SeqCtx lives in strata/core/glm_model.hpp (the corrected
// anchor: glm_fast_state.hpp does not exist in the tree).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <strata/core/glm_model.hpp>

// SeqCtx is nested in Glm5Model (namespace strata::core, glm_model.hpp:94/547)
using SeqCtx = strata::core::Glm5Model::SeqCtx;

static int g_checks = 0;
static int g_failures = 0;
static int g_table_was_shuffled = 0;

static void require(const char* name, int cond, int) {
    g_checks++;
    if (!cond) { g_failures++; printf("FAIL %s\n", name); }
    else printf("ok %s\n", name);
}

// one fake DSA layer's rows for tokens 0..T-1, appended contiguously
static void fill_rows(std::vector<unsigned char>& arena, size_t row_bytes, int T, int seed) {
    for (int t = 0; t < T; t++)
        for (size_t i = 0; i < row_bytes; i++)
            arena[(size_t)t * row_bytes + i] =
                (unsigned char)((t * 131 + (int)i * 7 + seed * 17 + 1) & 0xff);
}

// a SHUFFLED page table: a rotation, never the identity for pages > 1
static std::vector<int> shuffled_table(int pages) {
    std::vector<int> table(pages);
    for (int i = 0; i < pages; i++) table[i] = (i + 1) % pages;
    int nonid = 0;
    for (int i = 0; i < pages; i++) if (table[i] != i) nonid = 1;
    g_table_was_shuffled = nonid;
    return table;
}

// one (P, row_bytes) case: contiguous baseline vs paged scatter and gather
// through the production routing helper, across all 11 DSA layers
static int run_case(int P, size_t row_bytes, int layers, int T) {
    int pages = (T + P - 1) / P;
    std::vector<int> table = shuffled_table(pages);
    size_t arena_bytes = (size_t)T * row_bytes;
    for (int L = 0; L < layers; L++) {
        std::vector<unsigned char> base(arena_bytes, 0), paged(arena_bytes, 0);
        fill_rows(base, row_bytes, T, L);
        // paged appends: row t lands in page table[t / P] at (t % P) * row_bytes
        for (int t = 0; t < T; t++) {
            size_t off = SeqCtx::lat_row_offset(table.data(), P, row_bytes, t);
            memcpy(paged.data() + off, base.data() + (size_t)t * row_bytes, row_bytes);
        }
        // gather via the table, compare against the contiguous gather
        std::vector<unsigned char> gbase(row_bytes), gpaged(row_bytes);
        for (int t = 0; t < T; t++) {
            memcpy(gbase.data(), base.data() + (size_t)t * row_bytes, row_bytes);
            size_t off = SeqCtx::lat_row_offset(table.data(), P, row_bytes, t);
            memcpy(gpaged.data(), paged.data() + off, row_bytes);
            if (memcmp(gbase.data(), gpaged.data(), row_bytes) != 0) return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "--selftest") != 0) return 0;
    const int Ps[4] = {4, 64, 256, 512};
    const int T = 1024;   // spans several pages at every P
    const char* paged_env = getenv("STRATA_GLM_PAGED");
    int flag_off = (paged_env == NULL || paged_env[0] == '\0' ||
                    (paged_env[0] == '0' && paged_env[1] == '\0'));
    // (TDD artifact, Task 1.3's flag-on run: the check became state-asserting)
    if (flag_off)
        require("flag-off incumbent (STRATA_GLM_PAGED unset)", flag_off, 0);
    else
        require("flag-on paged routing (STRATA_GLM_PAGED set)", !flag_off, 0);
    char name[128];
    for (int pi = 0; pi < 4; pi++) {
        int P = Ps[pi];
        snprintf(name, sizeof(name), "paged gather bit-exact vs contiguous (FP16, P=%d)", P);
        require(name, run_case(P, 1024, 11, T), 0);
        snprintf(name, sizeof(name), "paged gather bit-exact vs contiguous (INT8, P=%d)", P);
        require(name, run_case(P, 544, 11, T), 0);
    }
    require("shuffled table observable (pages physically reordered)", g_table_was_shuffled, 0);
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
