// src/core/glm_pages.cu -- page pool manager
//
// Bump allocator over one GPU slab: page_alloc hands out either a released
// page from the free list or the next never-touched page; page_release and
// the last page_unref push the page back. Refcounts live on the host. The
// logical->physical table and its physical->logical mirror are device
// arrays kept identity here; Task 1.2's paged gather reads them on the
// device. Counters track allocs, frees, evicts and the live count for the
// leak checks (live = outstanding allocations: ++ on alloc, -- on free).
//
// Remediation (Task 1.1 review):
// init() publishes nothing until every allocation and copy has succeeded,
// and when the slab does not fit beside production (~104 MiB allocatable at
// 23.7 GB in use) it shrinks the pool by halves down to a 4-page floor (the
// request itself when smaller) instead of failing; n_pages() reports the
// actual pool size. A failed init now leaves
// a consistent EMPTY pool (n_pages_ 0, vectors sized 0), so page_alloc
// returns -1 instead of writing into cleared vectors: the segfault the
// first gate run recorded as '0 failures' without ever running the test.
#include "strata/core/glm_pages.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata {

int32_t GlmPages::env_page_size() {
    const char* v = getenv("STRATA_GLM_PAGE_SIZE");
    return v ? (int32_t)atoi(v) : 256;
}

int32_t GlmPages::env_pool_pages(int64_t bytes_per_token) {
    const char* v = getenv("STRATA_GLM_PAGE_POOL_GB");
    int64_t gb = v ? (int64_t)atoll(v) : 1;
    int64_t page_bytes = (int64_t)env_page_size() * bytes_per_token;
    if (page_bytes <= 0) return 1;
    int64_t pages = (gb << 30) / page_bytes;
    return (int32_t)(pages < 1 ? 1 : pages);
}

bool GlmPages::init(int32_t page_size, int32_t n_pages, int64_t bytes_per_token) {
    shutdown();
    if (n_pages <= 0 || page_size <= 0 || bytes_per_token <= 0) return false;
    const int64_t page_bytes = (int64_t)page_size * bytes_per_token;
    // Try the requested pool; shrink by halves on OOM (beside-production
    // courtesy), floor 4 pages. Nothing is published until every allocation
    // and copy has succeeded, so any failure leaves the empty pool that
    // shutdown() prepared.
    const int32_t floor_pages = n_pages < 4 ? n_pages : 4;
    int32_t try_pages = n_pages;
    void* slab = nullptr;
    while (try_pages >= floor_pages) {
        void* s = nullptr;
        if (cudaMalloc(&s, (size_t)try_pages * (size_t)page_bytes) == cudaSuccess) {
            slab = s;
            break;
        }
        cudaGetLastError();   // the probe's OOM must not stay sticky: the next
                              // kernel launch would swallow it and fail a clean op
        try_pages >>= 1;
    }
    if (slab == nullptr) {
        cudaGetLastError();   // the last probe's error: the pool is off, the
                              // caller's next launch must start clean
        return false;
    }
    std::vector<int32_t> ident(try_pages);
    for (int32_t i = 0; i < try_pages; ++i) ident[i] = i;
    int32_t* tbl = nullptr;
    int32_t* slot = nullptr;
    if (cudaMalloc(&tbl, sizeof(int32_t) * try_pages) != cudaSuccess ||
        cudaMalloc(&slot, sizeof(int32_t) * try_pages) != cudaSuccess ||
        cudaMemcpy(tbl, ident.data(), sizeof(int32_t) * try_pages,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(slot, ident.data(), sizeof(int32_t) * try_pages,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();   // the failed alloc/copy must not stay sticky
        cudaFree(slab);
        if (tbl) cudaFree(tbl);
        if (slot) cudaFree(slot);
        return false;
    }
    page_size_ = page_size;
    page_bytes_ = page_bytes;
    n_pages_ = try_pages;
    d_slab_ = slab;
    d_table_ = tbl;
    d_slot_ = slot;
    identity_ = true;   // the fresh table is identity; P3's reorders set this false when they land
    bump_ = 0;
    clock_ = 0;
    free_list_.clear();
    refcount_.assign(try_pages, 0);
    last_use_.assign(try_pages, 0);
    allocated_.assign(try_pages, 0);
    counters_ = GlmPagesCounters{};
    return true;
}

void GlmPages::shutdown() {
    if (d_slab_) { cudaFree(d_slab_); d_slab_ = nullptr; }
    if (d_table_) { cudaFree(d_table_); d_table_ = nullptr; }
    if (d_slot_) { cudaFree(d_slot_); d_slot_ = nullptr; }
    free_list_.clear();
    refcount_.clear();
    last_use_.clear();
    allocated_.clear();
    bump_ = 0;
    clock_ = 0;
    page_size_ = 0;
    page_bytes_ = 0;
    n_pages_ = 0;   // a shut-down pool is an EMPTY pool: page_alloc -> -1, no OOB writes
    counters_ = GlmPagesCounters{};
}

int32_t GlmPages::page_alloc() {
    int32_t phys = -1;
    if (!free_list_.empty()) {
        phys = free_list_.back();
        free_list_.pop_back();
    } else if (bump_ < n_pages_) {
        phys = bump_++;
    } else {
        // Task 3.2: LRU runs under pool pressure ONLY -- the free list is
        // empty and the bump is exhausted, so this is the one place the
        // sweep may fire, and it may push back only refcount-0 pages (an
        // in-use page's refcount holds it through the pressure).
        if (evict_lru(1) > 0) {
            phys = free_list_.back();
            free_list_.pop_back();
        }
    }
    if (phys < 0) return -1;
    allocated_[phys] = 1;
    refcount_[phys] = 0;
    last_use_[phys] = ++clock_;
    ++counters_.allocs;
    ++counters_.live;
    return phys;
}

void GlmPages::free_page(int32_t phys, bool evicted) {
    if (phys < 0 || phys >= n_pages_ || !allocated_[phys]) return;
    allocated_[phys] = 0;
    refcount_[phys] = 0;
    free_list_.push_back(phys);
    --counters_.live;
    ++counters_.frees;
    if (evicted) ++counters_.evicts;
}

void GlmPages::page_release(int32_t phys) {
    free_page(phys, false);
}

void GlmPages::page_ref(int32_t phys) {
    if (phys < 0 || phys >= n_pages_ || !allocated_[phys]) return;
    ++refcount_[phys];
    last_use_[phys] = ++clock_;
}

void GlmPages::page_unref(int32_t phys) {
    if (phys < 0 || phys >= n_pages_ || !allocated_[phys]) return;
    if (--refcount_[phys] <= 0) free_page(phys, false);
}

int32_t GlmPages::evict_lru(int32_t n) {
    int32_t freed = 0;
    while (freed < n) {
        int32_t best = -1;
        uint64_t best_use = ~0ULL;
        for (int32_t p = 0; p < n_pages_; ++p) {
            if (allocated_[p] && refcount_[p] == 0 && last_use_[p] < best_use) {
                best_use = last_use_[p];
                best = p;
            }
        }
        if (best < 0) break;
        free_page(best, true);
        ++freed;
    }
    return freed;
}

// Task 3.2 Step 3: the eviction unit check, as a free function so the
// test binary can link it without touching the header (the branch's
// Files list is the three .cu/.cpp files only).  The check: fill the pool,
// force eviction, confirm ONLY refcount-0 pages were pushed back and the
// in-use pages' gathers stay bit-exact (the slab bytes of every refcounted
// page re-read and compared against the pre-eviction snapshot).
bool glm_pages_lru_selftest() {
    using namespace strata;
    GlmPages pages;
    const int32_t kPages = 8;
    const int32_t kPageSize = 16;
    const int64_t kBytesPerToken = 4;
    if (!pages.init(kPageSize, kPages, kBytesPerToken)) {
        printf("lru_selftest: init failed\n");
        return false;
    }
    const int32_t n = pages.n_pages() < kPages ? pages.n_pages() : kPages;
    const size_t page_bytes = (size_t)kPageSize * (size_t)kBytesPerToken;
    int32_t fails = 0;
    auto check = [&](const char* name, bool ok) {
        printf("  %-44s %s\n", name, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    // Fill the pool: every page allocated, three of them refcounted (in use).
    std::vector<int32_t> all;
    for (int32_t i = 0; i < n; ++i) {
        int32_t p = pages.page_alloc();
        if (p >= 0) all.push_back(p);
    }
    check("fill: every page allocated", (int)all.size() == n);
    const int32_t kKeep = n < 3 ? n : 3;
    for (int32_t i = 0; i < kKeep; ++i) pages.page_ref(all[i]);
    // Pre-eviction snapshot of the in-use pages' gather bytes.
    std::vector<std::vector<char>> keep_before(kKeep, std::vector<char>(page_bytes));
    for (int32_t i = 0; i < kKeep; ++i) {
        if (cudaMemcpy(keep_before[i].data(), pages.slab_ptr() + (size_t)all[i] * page_bytes,
                       page_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ++fails;
    }
    // Force eviction: more pages than the pool holds.
    const int32_t freed = pages.evict_lru(n * 2);
    check("force evict: only refcount-0 pages freed", freed == n - kKeep);
    check("force evict: live count == refcounted pages", pages.counters().live == kKeep);
    // The in-use pages' gathers stay bit-exact through the pressure.
    std::vector<char> after(page_bytes);
    for (int32_t i = 0; i < kKeep; ++i) {
        if (cudaMemcpy(after.data(), pages.slab_ptr() + (size_t)all[i] * page_bytes,
                       page_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ++fails;
        check("in-use gather bit-exact", memcmp(keep_before[i].data(), after.data(), page_bytes) == 0);
    }
    // refcounts still hold: unref frees exactly the kept pages.
    for (int32_t i = 0; i < kKeep; ++i) pages.page_unref(all[i]);
    check("refcounts held through eviction", pages.counters().frees >= kKeep && pages.counters().live == 0);
    pages.shutdown();
    printf("lru_selftest: %d checks %d failures\n", 5 + kKeep, fails);
    return fails == 0;
}

// Task 3.2's pressure branch: a FULL pool's page_alloc sweeps one refcount-0
// page back (the LRU), so the alloc is served and exactly one page was evicted.
bool glm_pages_pressure_selftest() {
    GlmPages pages;
    const int32_t kPages = 4, kPageSize = 16, kBytesPerToken = 4;
    if (!pages.init(kPageSize, kPages, kBytesPerToken)) {
        printf("pressure_selftest: init failed\n");
        return false;
    }
    int32_t fails = 0;
    auto check = [&](const char* name, bool ok) {
        printf("  %-44s %s\n", name, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    std::vector<int32_t> q;
    for (int32_t i = 0; i < pages.n_pages(); ++i) {
        const int32_t p = pages.page_alloc();
        if (p >= 0) q.push_back(p);
    }
    check("pressure: the pool filled", (int32_t) q.size() == pages.n_pages());
    const int32_t got = pages.page_alloc();   // the pool is full: the sweep fires
    check("pressure: the sweep served the alloc", got >= 0);
    check("pressure: the sweep evicted exactly one", pages.counters().evicts == 1);
    check("pressure: the live count held", pages.counters().live == pages.n_pages());
    if (got >= 0) pages.page_release(got);
    for (int32_t p : q) pages.page_release(p);
    check("pressure: the drain leaves live 0", pages.counters().live == 0);
    pages.shutdown();
    printf("pressure_selftest: %d failures\n", fails);
    return fails == 0;
}

// Task 4.1's pool-pages rung (the ladder's rung 1, pages_sweep_idle's body): the full-pool
// sweep frees every refcount-0 page, protects the refcounted ones (the capture's tiles), is a
// no-op when nothing is idle, and the pool off is a clean 0.
bool glm_pages_idle_sweep_selftest() {
    GlmPages pages;
    const int32_t kPages = 8, kPageSize = 16, kBytesPerToken = 4;
    if (!pages.init(kPageSize, kPages, kBytesPerToken)) {
        printf("idle_sweep_selftest: init failed\n");
        return false;
    }
    int32_t fails = 0;
    auto check = [&](const char* name, bool ok) {
        printf("  %-44s %s\n", name, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    std::vector<int32_t> all;
    for (int32_t i = 0; i < pages.n_pages(); ++i) {
        const int32_t p = pages.page_alloc();
        if (p >= 0) all.push_back(p);
    }
    const int32_t kKeep = (int32_t) all.size() < 3 ? (int32_t) all.size() : 3;
    for (int32_t i = 0; i < kKeep; ++i) pages.page_ref(all[i]);
    const int32_t swept = pages.evict_lru(pages.n_pages());   // the rung-1 form: the whole pool
    check("idle sweep: every refcount-0 page back", swept == (int32_t) all.size() - kKeep);
    check("idle sweep: the refcounted pages survive", pages.counters().live == kKeep);
    check("idle sweep: a second sweep is a no-op", pages.evict_lru(pages.n_pages()) == 0);
    GlmPages off;   // the pool off (a default GlmPages): the wrapper's early return, as 0
    check("idle sweep: the pool off is a clean 0", off.evict_lru(4) == 0);
    for (int32_t i = 0; i < kKeep; ++i) pages.page_unref(all[i]);
    check("idle sweep: the held pages release clean", pages.counters().live == 0);
    pages.shutdown();
    printf("idle_sweep_selftest: %d checks %d failures\n", 5, fails);
    return fails == 0;
}

bool GlmPages::check_invert() const {
    if (!d_table_ || !d_slot_ || n_pages_ <= 0) return false;
    std::vector<int32_t> table(n_pages_), slot(n_pages_);
    if (cudaMemcpy(table.data(), d_table_, sizeof(int32_t) * n_pages_,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(slot.data(), d_slot_, sizeof(int32_t) * n_pages_,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int32_t p = 0; p < n_pages_; ++p) {
        if (table[p] != p || slot[p] != p) return false;
    }
    return true;
}

} // namespace strata
