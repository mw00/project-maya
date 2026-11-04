// The expert cache's lendable arena (the on-demand vision encoder): its tail is given back to the driver - the free
// VRAM grows by what was lent - the slots before it keep their bytes, and once mapped again at the same addresses and
// refilled every slot holds its expert.  A HIP build (one allocation) must refuse to lend.  Needs a GPU (77: none).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what, const std::string& err = "") {
    if (!ok) {
        ++fails;
        std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    }
}
std::vector<uint8_t> blob(int slot, int64_t n, int round) {
    std::vector<uint8_t> b((size_t) n);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t) (i * 31 + (size_t) slot * 7 + (size_t) round * 13);
    return b;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU: skipped\n");
        return 77;
    }
    std::vector<int64_t> sizes;
    for (int i = 0; i < 256; ++i) sizes.push_back(((int64_t) 1 << 20) + (int64_t) i * 4096 + 100);   // ~1.4 MB each
    strata::core::ExpertCache c;
    std::string err;
    c.set_lendable(true);
    expect(c.open_sized(sizes, 8, 32, err), "open_sized", err);
    if (fails) return 1;
#if defined(STRATA_USE_HIP)
    uint64_t from = 0, given = 0;
    expect(!c.lendable() && c.lendable_bytes(0) == 0, "a HIP build keeps one allocation");
    expect(!c.lend_tail(1 << 20, 0, from, given, err) && given == 0, "a HIP build cannot lend");
    expect(c.reclaim_tail(err) && !c.lent(), "nothing to reclaim");
    strata::core::LendableArena a;
    expect(!a.open(1 << 20, err) && !a.ok() && a.lendable_bytes(0) == 0, "a HIP build has no lendable arena");
    std::printf(fails ? "FAILED\n" : "ok (HIP: no lending)\n");
    return fails ? 1 : 0;
#else
    if (!c.lendable()) {
        std::printf("this GPU has no virtual memory management: skipped\n");
        return 77;
    }
    const int n = (int) c.slots();
    auto slot_end = [&](int s) -> uint64_t {
        return s + 1 < n ? (uint64_t) (c.device_slot(s + 1) - c.device_slot(0)) : (uint64_t) c.bytes();
    };
    for (int s = 0; s < n; ++s) {
        const auto b = blob(s, sizes[(size_t) s], 0);
        expect(c.fill_slot_blocking(s, b.data(), err, (int64_t) b.size()), "fill", err);
    }
    uint8_t* const base = c.device_slot(0);
    const uint64_t keep = (uint64_t) (c.device_slot(16) - base);
    expect(c.lendable_bytes(keep) > (uint64_t) 200 << 20 && c.lendable_bytes(keep) <= (uint64_t) c.bytes(),
           "lendable bytes");
    size_t f0 = 0, f1 = 0, f2 = 0, t = 0;
    cudaMemGetInfo(&f0, &t);
    uint64_t from = 0, given = 0;
    expect(c.lend_tail((uint64_t) 100 << 20, keep, from, given, err), "lend 100 MB", err);
    const uint64_t given0 = given;
    expect(given >= (uint64_t) 100 << 20 && from >= keep && from < (uint64_t) c.bytes() && c.lent(), "lent range");
    cudaMemGetInfo(&f1, &t);
    expect(f1 >= f0 + given - ((uint64_t) 4 << 20), "the free VRAM grew by what was lent");
    // another allocation can use the memory given back (as the encoder's process does)
    void* other = nullptr;
    expect(cudaMalloc(&other, (size_t) given - ((size_t) 8 << 20)) == cudaSuccess, "allocate in the lent memory");
    // a second lend while lent: nothing new is unmapped
    uint64_t from2 = 0, given2 = 0;
    expect(c.lend_tail((uint64_t) 100 << 20, keep, from2, given2, err) && from2 == from, "lend again", err);
    // the slots before the lent part kept their bytes
    for (int s = 0; s < n && slot_end(s) <= from; ++s) {
        const auto b = blob(s, sizes[(size_t) s], 0);
        expect(c.verify_slot(s, b.data(), err, (int64_t) b.size()), "a kept slot", err);
    }
    // reclaim while the memory is taken by the other allocation may fail (the card's free memory decides); after it
    // is freed it must succeed, at the same addresses
    cudaFree(other);
    expect(c.reclaim_tail(err) && !c.lent() && c.device_slot(0) == base, "reclaim", err);
    cudaMemGetInfo(&f2, &t);
    expect(f2 + ((uint64_t) 4 << 20) >= f0 && f2 <= f0 + ((uint64_t) 4 << 20), "the free VRAM is back where it was");
    int refilled = 0;
    for (int s = 0; s < n; ++s) {
        const bool lost = slot_end(s) > from;
        const auto b = blob(s, sizes[(size_t) s], lost ? 1 : 0);
        if (lost) {
            expect(c.fill_slot_blocking(s, b.data(), err, (int64_t) b.size()), "refill", err);
            ++refilled;
        }
        expect(c.verify_slot(s, b.data(), err, (int64_t) b.size()), "a slot after the reclaim", err);
    }
    expect(refilled > 0 && refilled < n, "some slots were lent");
    // the whole cache (no floor: a small card's encoder needs all of it): every byte goes back, every slot refills
    {
        size_t fa = 0, fb = 0, fc = 0;
        cudaMemGetInfo(&fa, &t);
        uint64_t fromw = 1, givenw = 0;
        expect(c.lend_tail((uint64_t) c.bytes(), 0, fromw, givenw, err) && fromw == 0 &&
                   givenw >= (uint64_t) c.bytes() && givenw == c.lendable_bytes(0),
               "lend the whole cache", err);
        cudaMemGetInfo(&fb, &t);
        expect(fb + ((uint64_t) 4 << 20) >= fa + givenw, "the free VRAM grew by the whole cache");
        expect(c.reclaim_tail(err) && !c.lent() && c.device_slot(0) == base, "reclaim the whole cache", err);
        cudaMemGetInfo(&fc, &t);
        expect(fc + ((uint64_t) 4 << 20) >= fa && fc <= fa + ((uint64_t) 4 << 20), "the whole cache is back");
        for (int s = 0; s < n; ++s) {
            const auto b = blob(s, sizes[(size_t) s], 2);
            expect(c.fill_slot_blocking(s, b.data(), err, (int64_t) b.size()) &&
                       c.verify_slot(s, b.data(), err, (int64_t) b.size()),
                   "a slot refilled after the whole cache was lent", err);
        }
    }
    // a plain arena (the draft layer's experts): its tail goes back, the head keeps its bytes, the same address after
    {
        strata::core::LendableArena a;
        const uint64_t ab = ((uint64_t) 130 << 20) + 12345;
        expect(a.open(ab, err) && a.ok() && a.lendable_bytes(0) >= ab, "an arena", err);
        std::vector<uint8_t> h((size_t) ab), back((size_t) ab);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 17 + 3);
        cudaMemcpy(a.data(), h.data(), h.size(), cudaMemcpyHostToDevice);
        uint8_t* const ad = a.data();
        uint64_t af = 0, ag = 0;
        expect(a.lend_tail(1, 0, af, ag, err) && a.lent() && af == a.lent_from() && af > 0 && af < ab && ag > 0,
               "the arena lends its last piece", err);
        expect(a.reclaim_tail(err) && !a.lent() && a.data() == ad, "the arena maps it again", err);
        cudaMemcpy(ad + af, h.data() + af, (size_t) (ab - af), cudaMemcpyHostToDevice);
        cudaMemcpy(back.data(), ad, back.size(), cudaMemcpyDeviceToHost);
        expect(back == h, "the arena's bytes after the reclaim");
        a.close();
        expect(!a.ok() && a.data() == nullptr, "the arena closed");
    }
    // nothing past the floor: refused; a cache closed while lent leaves nothing behind
    expect(!c.lend_tail(1, (uint64_t) c.bytes(), from, given, err), "nothing to lend past the floor");
    const uint64_t lent_first = given0;
    expect(c.lend_tail(1, keep, from, given, err), "lend", err);
    const uint64_t total = (uint64_t) c.bytes();
    c.close();
    cudaMemGetInfo(&f1, &t);
    expect(f1 + ((uint64_t) 4 << 20) >= f0 + total, "closed while lent");
    std::printf(fails ? "FAILED (%d)\n" : "ok: %d of %d slots lent and refilled, %.0f MB given back\n", fails ? fails : refilled,
                n, (double) lent_first / 1048576.0);
    return fails ? 1 : 0;
#endif
}
