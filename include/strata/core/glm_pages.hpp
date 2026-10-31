// include/strata/core/glm_pages.hpp -- page pool manager
//
// One GPU slab, bump-allocated into pages of page_size tokens each. The
// host keeps a free list and per-page refcounts; a logical->physical
// table (plus its physical->logical mirror) lives on the device for the
// paged gather of Task 1.2. evict_lru frees refcount-0 pages only, so a
// pinned page survives pressure. Page size comes from STRATA_GLM_PAGE_SIZE
// (default 256), pool size from STRATA_GLM_PAGE_POOL_GB.
//
// Remediation: init() tries the requested pool and shrinks it by
// halves (floor 4 pages, or the request itself when smaller) until the slab
// fits beside production; a failed init leaves an EMPTY pool (n_pages() 0,
// page_alloc() -1), and shutdown() resets to that same empty state.
#ifndef STRATA_CORE_GLM_PAGES_HPP
#define STRATA_CORE_GLM_PAGES_HPP

#include <cstdint>
#include <vector>

namespace strata {

struct GlmPagesCounters {
    int64_t allocs = 0;   // page_alloc calls that returned a page
    int64_t frees = 0;    // pages returned to the pool
    int64_t evicts = 0;   // pages freed by evict_lru
    int64_t live = 0;     // allocated right now (leak check)
};

class GlmPages {
public:
    bool init(int32_t page_size, int32_t n_pages, int64_t bytes_per_token);
    void shutdown();

    // A physical page index, or -1 when the pool is exhausted.
    int32_t page_alloc();
    // Return a page to the pool regardless of refcount.
    void page_release(int32_t phys);
    // Pin the page: evict_lru skips pages with refcount > 0.
    void page_ref(int32_t phys);
    // Drop one ref; the ref that reaches 0 returns the page to the pool.
    void page_unref(int32_t phys);
    // Free at most n refcount-0 pages, oldest use first; returns the count.
    int32_t evict_lru(int32_t n);

    // Device-side logical->physical table (identity until Task 1.2 shuffles).
    int32_t* table_ptr() const { return d_table_; }
    // True while the table maps every page to itself (the v1 wiring's constant state; P3's
    // reorders flip it): launch sites use it to skip the cells-translation route entirely.
    bool table_identity() const { return identity_; }
    // Device-side physical->logical mirror (the table invert check reads it).
    int32_t* slot_ptr() const { return d_slot_; }
    // The one GPU slab the pages are bump-allocated out of.
    void* slab_ptr() const { return d_slab_; }
    int64_t page_bytes() const { return page_bytes_; }
    // True while the physical page is still in the pool (the prefix store's
    // reconcile reads it: a block whose pages were evicted is forgotten).
    bool page_allocated(int32_t phys) const {
        return phys >= 0 && phys < n_pages_ && allocated_[phys];
    }

    // table[p]==p and slot[p]==p while the pool is unshuffled.
    bool check_invert() const;

    GlmPagesCounters counters() const { return counters_; }
    int32_t refcount(int32_t phys) const { return refcount_[phys]; }
    int32_t n_pages() const { return n_pages_; }
    int32_t page_size() const { return page_size_; }

    // STRATA_GLM_PAGE_SIZE (default 256) and STRATA_GLM_PAGE_POOL_GB.
    static int32_t env_page_size();
    static int32_t env_pool_pages(int64_t bytes_per_token);

private:
    void free_page(int32_t phys, bool evicted);

    int32_t page_size_ = 0;
    int32_t n_pages_ = 0;
    bool identity_ = true;   // the table maps page -> itself until a reorder lands (P3)
    int64_t page_bytes_ = 0;
    void* d_slab_ = nullptr;
    int32_t* d_table_ = nullptr;
    int32_t* d_slot_ = nullptr;
    int32_t bump_ = 0;
    uint64_t clock_ = 0;
    std::vector<int32_t> free_list_;
    std::vector<int32_t> refcount_;
    std::vector<uint64_t> last_use_;
    std::vector<uint8_t> allocated_;
    GlmPagesCounters counters_{};
};

} // namespace strata

#endif // STRATA_CORE_GLM_PAGES_HPP
