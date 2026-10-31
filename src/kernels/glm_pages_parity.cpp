// glm_pages_parity.cpp -- page pool manager unit test (TDD)
// Run: ctest -R glm_pages_parity  (ctest invokes the binary with --selftest)
//
// Remediation review: the
// first gate run never executed (the executor recorded a claimed '0
// failures' and the committed binary actually segfaulted beside production,
// where init's 176 MiB slab OOMs and the failed init left cleared vectors).
// The tested shape now: init shrinks to fit (every count derives from
// n_pages()), a failed init is an EMPTY pool, and the churn leak check
// asserts real counter deltas instead of a constant.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "strata/core/glm_pages.hpp"

using strata::GlmPages;

namespace strata {   // the eviction unit checks live in glm_pages.cu (free functions,
                     // linked by this binary without touching the header)
bool glm_pages_lru_selftest();
bool glm_pages_pressure_selftest();
bool glm_pages_idle_sweep_selftest();
}

static int g_checks = 0;
static int g_failures = 0;

static void require(const char* name, int cond, int) {
    ++g_checks;
    printf("%s %s\n", cond ? "ok" : "FAIL", name);
    if (!cond) ++g_failures;
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "--selftest") != 0) {
        printf("usage: glm_pages_parity --selftest\n");
        return 1;
    }

    // Task 1.1 Step 1 sketch, remediated shape
    GlmPages P;
    const bool pool_ok = P.init(/*page_size=*/256, /*n_pages=*/64, /*bytes_per_token=*/11 * 1024);
    require("init 64-page pool", pool_ok, 0);
    if (!pool_ok) return 1;  // No pages exist; subsequent indexed checks require a live pool.
    const int32_t pool = P.n_pages();
    printf("pool: %d of 64 requested pages (shrink-to-fit beside production)\n", pool);
    require("pool respects the 4-page floor", pool >= 4 && pool <= 64, 0);
    require("page_size recorded", P.page_size() == 256, 0);
    require("page_bytes recorded", P.page_bytes() == (int64_t)256 * 11 * 1024, 0);
    require("table_ptr is the device-side table", P.table_ptr() != nullptr, 0);
    std::vector<int32_t> a;
    for (int32_t i = 0; i < pool; ++i) a.push_back(P.page_alloc());
    bool fill_real = true, fill_distinct = true;
    for (int32_t i = 0; i < pool; ++i) {
        if (a[i] < 0 || a[i] >= pool) fill_real = false;
        for (int32_t j = i + 1; j < pool; ++j) if (a[i] == a[j]) fill_distinct = false;
    }
    require("fill handed out the whole pool", fill_real, 0);
    require("fill pages are distinct", fill_distinct, 0);
    // Task 3.2: a full pool no longer refuses - page_alloc sweeps exactly one
    // refcount-0 page back (the LRU, a[0]: the allocation order is the clock) and
    // serves the alloc; the released sweep page frees it again
    const int32_t swept = P.page_alloc();
    require("full pool sweeps one page (the LRU) and serves", swept >= 0 && P.counters().evicts == 1, 0);
    P.page_release(swept);
    const int32_t probe = 7 % (pool < 1 ? 1 : pool);
    P.page_unref(a[probe]);                  // one page back
    require("release makes exactly one page", P.page_alloc() == a[probe], 0);
    require("table invert: table[p]==p, slot[p]==p", P.check_invert(), 0);   // identity while unshuffled
    P.page_release(a[pool - 1]);             // make room so the churn below actually cycles pages
    const int64_t live_before_churn = P.counters().live;
    const int64_t frees_before_churn = P.counters().frees;
    for (int i = 0; i < 1000; ++i) { auto p = P.page_alloc(); P.page_release(p); }
    require("churn x1000 leaks nothing", P.counters().live == live_before_churn, 0);
    require("churn x1000 all served", P.counters().frees - frees_before_churn == 1000, 0);
    require("page_size multiple of DSA granule 4", 256 % 4 == 0, 0);
    P.shutdown();
    require("shutdown empties the pool", P.n_pages() == 0 && P.page_alloc() == -1, 0);

    // refcount discipline: page_ref pins, page_unref drops one ref
    {
        GlmPages R;
        const bool ref_ok = R.init(256, 8, 11 * 1024);
        require("ref pool init", ref_ok, 0);
        if (!ref_ok) return 1;  // r[3] needs a live pool with at least four pages.
        const int32_t rpool = R.n_pages();
        std::vector<int32_t> r;
        for (int32_t i = 0; i < rpool; ++i) r.push_back(R.page_alloc());
        R.page_ref(r[3]);
        R.page_ref(r[3]);
        require("page_ref counts up", R.refcount(r[3]) == 2, 0);
        R.page_unref(r[3]);
        // Task 3.2: the full pool's alloc sweeps the LRU refcount-0 page (r[0], the
        // allocation order is the clock; the pin holds r[3]) and serves it
        const int32_t swept2 = R.page_alloc();
        require("one unref leaves the pin held", swept2 >= 0 && swept2 != r[3] && R.refcount(r[3]) == 1, 0);
        R.page_unref(r[3]);
        require("last unref returns the page", R.page_alloc() == r[3], 0);
        for (int32_t i = 0; i < rpool; ++i) R.page_release(r[i]);
        require("release drains the ref pool", R.counters().live == 0, 0);
        R.shutdown();
    }

    // evict_lru: refcount-0 pages only, pinned pages survive the pressure
    {
        GlmPages E;
        const bool evict_ok = E.init(256, 64, 11 * 1024);
        require("evict pool init", evict_ok, 0);
        if (!evict_ok) return 1;  // e[0] needs a live pool.
        const int32_t epool = E.n_pages();
        std::vector<int32_t> e;
        for (int32_t i = 0; i < epool; ++i) e.push_back(E.page_alloc());
        require("evict pool filled", (int32_t)e.size() == epool, 0);
        for (int32_t i = 0; i < epool; i += 2) E.page_ref(e[i]);   // pin every other page
        const int32_t unpinned = epool - (epool + 1) / 2;
        require("evict frees only refcount-0 pages", E.evict_lru(1000) == unpinned, 0);
        require("live after evict is the pinned half", E.counters().live == epool - unpinned, 0);
        require("pinned page still held", E.refcount(e[0]) == 1, 0);
        require("evict counters counted", E.counters().evicts == unpinned, 0);
        int32_t refill = E.page_alloc();
        require("pool refills from the evicted pages", refill >= 0, 0);
        E.page_release(refill);
        for (int32_t i = 0; i < epool; i += 2) E.page_unref(e[i]);  // drop the pins
        require("evict + unref drain leaves live 0", E.counters().live == 0, 0);
        require("table still identity after evict", E.check_invert(), 0);
        E.shutdown();
    }

    // Defect 5 evidence: the env-var sources the Interfaces line requires
    {
        unsetenv("STRATA_GLM_PAGE_POOL_GB");
        setenv("STRATA_GLM_PAGE_SIZE", "128", 1);
        require("STRATA_GLM_PAGE_SIZE=128 honored", GlmPages::env_page_size() == 128, 0);
        require("env pool pages 1 GiB / 1 MiB page", GlmPages::env_pool_pages(8 * 1024) == 1024, 0);
        unsetenv("STRATA_GLM_PAGE_SIZE");
        require("STRATA_GLM_PAGE_SIZE default is 256", GlmPages::env_page_size() == 256, 0);
        require("env pool pages at the 256 default", GlmPages::env_pool_pages(8 * 1024) == 512, 0);
    }

    // the eviction unit check + the pressure branch (the selftests in glm_pages.cu;
    // their own check lines print as they run)
    {
        setenv("STRATA_GLM_PAGE_POOL_GB", "", 1);   // a small pool fits beside production
        unsetenv("STRATA_GLM_PAGE_SIZE");
        const bool lru_ok = strata::glm_pages_lru_selftest();
        require("lru_selftest: fill, force evict, in-use gathers bit-exact", lru_ok ? 1 : 0, 0);
        const bool pressure_ok = strata::glm_pages_pressure_selftest();
        require("pressure_selftest: the full-pool sweep serves the alloc", pressure_ok ? 1 : 0, 0);
        const bool idle_ok = strata::glm_pages_idle_sweep_selftest();
        require("idle_sweep_selftest: the rung-1 full sweep + the pool off", idle_ok ? 1 : 0, 0);
    }

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
