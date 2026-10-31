// glm_seq_dynamic_test.cpp --selftest
// Task 2.1: the acquire/release lifecycle.  STRATA_GLM_SEQS is a cap, not a reservation: a
// sequence's own buffers (the state arena, the snapshot pair, the KDA backup) are allocated on
// demand by seq_acquire through the UNCHANGED seq_layout(), freed or parked by seq_release, and
// the pinned hop slots live in one cap-sized reusable pool.  The destructor's per-pointer sweep
// (glm_model.cu:584-592) stays valid unchanged because every parked slot's pointers are null.
// Layout invariance: acquire() builds the SeqCtx via the UNCHANGED seq_layout() (glm_multi.cu:73-107).
//
// Task 4.1: dynamic slots + eviction (the scheduler's primitives).  The driver binds without
// acquire (seq_bind allocates a parked slot), so the bind IS the acquire: the slot must leave
// the free list when the bind allocates it (a second bind of the same slot would alias one
// arena, and a release would push it twice).  seq_release parks a slot exactly once (the guard:
// an already-parked slot has nothing to free and no second entry to push).  The admission
// check seq_ready(want) reports whether the cap can hold one more live sequence for a request
// of `want` tokens: a free-list slot, its parked context holds the want, and the slot's own
// bytes (arena + snapshot pair + speculative backup) fit the device's free VRAM on every part of the split.  The
// pressure ladder's first rung is pages_sweep_idle(): every refcount-0 page back (the store's
// refcounted canonical pages survive), the prefix capture's allocator lock held.
//
// The fixture pack (data/glm5-synth-pack, tools/glm_synth_pack.py) packs the 6-layer synthetic
// GGUF through the REAL packer, so this is the real load_pack path; the model runs on the slow
// path (STRATA_GLM_SLOW) because the synth pack's small BF16 rows resolve only there - the
// lifecycle methods are shared by both paths (glm_multi.cu).
#include "strata/core/glm_model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

static void require(const char* name, bool ok, int) {
    ++g_checks;
    if (ok) printf("ok: %s\n", name);
    else { ++g_failures; printf("FAIL: %s\n", name); }
}

static int free_count(const strata::core::Glm5Model& m, int s) {
    return (int) std::count(m.seq_free_.begin(), m.seq_free_.end(), s);
}

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        fprintf(stderr, "glm_seq_dynamic_test: run with --selftest (the CTest form)\n");
        return 2;
    }
    const std::string pack = argc > 2 ? argv[2] : "data/glm5-synth-pack";
    std::string err;
    // the cap comes from STRATA_GLM_SEQS at load; pick a small one the loop can churn, with a
    // parked-slot context smaller than the first sequence's (the STRATA_GLM_SEQ_CTX form)
    setenv("STRATA_GLM_SLOW", "1", 1);
    setenv("STRATA_GLM_SEQS", "4", 1);
    setenv("STRATA_GLM_SEQ_CTX", "2048", 1);   // the parked slots' own context: the 8320 floor
                                               // applies (STRATA_GLM_SEQ_CTX=2048 under it)
    setenv("STRATA_GLM_SEQ_VRAM_FIT", "1", 1); // the VRAM-fit leg is forced (the unit test pins the
                                               // admission logic; the real query is the driver's gate)
    unsetenv("STRATA_GLM_SEQ_FLOOR");          // the first load pins the default: the floor is the full cap

    strata::core::Glm5Model model;
    if (!model.load_pack(pack, 4096, err)) {
        printf("FAIL: glm_seq_dynamic_test: load_pack(%s) failed: %s\n", pack.c_str(), err.c_str());
        return 1;
    }

    // the load-time capture: the cap parked; the pool sizing subtracts the guaranteed states'
    // floor (seq_bytes_floor_, Task 4.1 amendment) - the first load's floor == the reservation
    require("load parked the cap (bytes outstanding == 0 at load)", model.state_bytes_outstanding() == 0, 0);
    require("the load-time capture: the reservation is anchored", model.seq_bytes_reserved_ > 0, 0);
    require("the load-time capture: the per-slot arena bytes are anchored", model.seq_arena_bytes() > 0, 0);
    // Task 4.1 amendment: with no STRATA_GLM_SEQ_FLOOR the floor defaults to the full cap (the
    // sizing is identical to Task 2.1's reservation), so the first load's two agree
    require("the floor defaults to the full cap", model.seq_bytes_floor_ == model.seq_bytes_reserved_, 0);
    require("the floor is anchored", model.seq_bytes_floor_ > 0, 0);

    // the pooled pipeline buffers: one cudaHostAlloc, two distinct carvings
    void* h0 = model.seq_hop_slot(0);
    void* h1 = model.seq_hop_slot(1);
    require("the pipeline-buffer pool carved two distinct slots", h0 != nullptr && h1 != nullptr && h0 != h1, 0);
    require("the pool is host RAM (bytes outstanding still 0)", model.state_bytes_outstanding() == 0, 0);

    // layout invariance: acquire() builds SeqCtx via the UNCHANGED seq_layout()
    for (int i = 0; i < 1000; ++i) {
        int s = model.seq_acquire();                 // allocates KDA states + conv + DSA caches + snapshot
        require("acquire returned a valid slot", s >= 0 && s < 4, 0);
        model.seq_release(s);                        // frees or returns to the free-list
    }
    require("acquire/release x1000, no leak (bytes outstanding == 0)", model.state_bytes_outstanding() == 0, 0);
    require("layout matches load_pack's check", model.seq_layout_matches(), 0);
    require("cap honoured: acquire past cap fails cleanly", model.seq_acquire_past_cap(), 0);
    require("destructor-sweep invariant: parked slots' pointers null after release", model.seq_parked_slots_null(), 0);

    // ---- Task 4.1: dynamic slots + eviction (the scheduler's primitives)
    // the admission check at load: a parked slot with room, its bytes fit the device (the
    // parked ctx is the 8320 floor: STRATA_GLM_SEQ_CTX=2048 under max_ctx=4096 resolves to it)
    require("a parked slot's own context is known at load", model.seq_max_ctx(1) == 8320, 0);
    require("ready at load: the cap holds one more for a 2048-token want", model.seq_ready(2048), 0);
    require("a want above every context is not ready", !model.seq_ready(8321), 0);
    require("slot 0 is live (the load-time alias)", model.seq_live(0), 0);
    for (int s = 1; s < 4; ++s)
        require("parked slot is not live at load", !model.seq_live(s), 0);

    // the bind IS the scheduler's acquire: the free list pops the bound slot
    model.seq_bind(1);
    require("bind allocated the state", model.seq_live(1), 0);
    require("bind popped the free-list entry (no duplicate)", free_count(model, 1) == 0, 0);
    require("the bound slot is the bound one", model.seq_bound() == 1, 0);
    require("bytes outstanding == one slot after the bind", model.state_bytes_outstanding() > 0, 0);
    // a second bind of the bound slot changes nothing
    model.seq_bind(1);
    require("re-binding the bound slot is a no-op", model.seq_bound() == 1 && free_count(model, 1) == 0, 0);

    // the release parks it exactly once
    model.seq_release(1);
    require("release parked the slot (pointers null)", !model.seq_live(1), 0);
    require("release pushed the slot back exactly once", free_count(model, 1) == 1, 0);
    require("the release rebound slot 0", model.seq_bound() == 0, 0);
    require("slot 0 stayed live through the release", model.seq_live(0), 0);
    require("bytes outstanding back to 0 after the release", model.state_bytes_outstanding() == 0, 0);
    require("destructor-sweep invariant holds after the release", model.seq_parked_slots_null(), 0);

    // the guard: releasing an already-parked slot changes nothing (no second entry, no free)
    model.seq_release(1);
    require("double release does not duplicate the free list", free_count(model, 1) == 1, 0);
    require("double release keeps bytes outstanding 0", model.state_bytes_outstanding() == 0, 0);

    // the cap exhausted: acquire every parked slot, ready goes false; release all, true again
    std::vector<int> taken;
    for (int i = 0; i < 4; ++i) {
        int s = model.seq_acquire();
        if (s >= 0) taken.push_back(s);
    }
    require("the cap holds 3 parked slots (slot 0 stays the alias)", (int) taken.size() == 3, 0);
    require("cap exhausted: ready is false with every slot live", !model.seq_ready(128), 0);
    for (int s : taken) model.seq_release(s);
    require("released all: ready is true again", model.seq_ready(2048), 0);
    require("released all: bytes outstanding back to 0", model.state_bytes_outstanding() == 0, 0);
    require("released all: each parked slot pushed exactly once", [&] {
        for (int s = 1; s < 4; ++s) if (free_count(model, s) != 1) return false;
        return true;
    }(), 0);

    // the pressure ladder's first rung: the pool is off here (STRATA_GLM_PAGED unset), so the
    // sweep is a clean 0 and must not fail; the pages selftests cover the pool-on sweep
    require("the pool sweep with the pool off returns 0", model.pages_sweep_idle() == 0, 0);

    // ---- Task 4.1 amendment: the state floor (the second load, a separate model)
    // STRATA_GLM_SEQ_FLOOR=2 of the cap 4: only slots 1-2's bytes stay guaranteed against the
    // expert-pool sizing; slots 3's bytes are the ladder's to arbitrate. The first model is
    // destroyed by scope, so a fresh one re-runs the full load path (the env read happens in
    // seq_alloc_arenas, at glm_model.cu's load_pack call site, before the pool sizing).
    {
        setenv("STRATA_GLM_SEQ_FLOOR", "2", 1);
        strata::core::Glm5Model floored;
        if (!floored.load_pack(pack, 4096, err)) {
            printf("FAIL: glm_seq_dynamic_test: the floored load_pack(%s) failed: %s\\n", pack.c_str(), err.c_str());
            return 1;
        }
        require("the floored load parked the cap (bytes outstanding == 0)", floored.state_bytes_outstanding() == 0, 0);
        require("the floored reservation is the full cap", floored.seq_bytes_reserved_ == model.seq_bytes_reserved_, 0);
        require("the floor is below the reservation", floored.seq_bytes_floor_ < floored.seq_bytes_reserved_, 0);
        // the floor's arithmetic: (f - 1) * (arena + snaps + spec backup) + snaps
        // (seq_snap_bytes ignores its slot: every slot's pair is the same size, glm_multi.cu:245)
        require("the floor covers f-1 full slots + the slot-0 snapshot pair",
                floored.seq_bytes_floor_ ==
                    ((size_t) 2 - 1) * (floored.seq_arena_bytes() + floored.seq_snap_bytes(floored.seq_[0]) + floored.spec_kda_bytes()) + floored.seq_snap_bytes(floored.seq_[0]), 0);
        require("the floored floor is anchored", floored.seq_bytes_floor_ > 0, 0);
        require("the floored cap still parked every slot", floored.seq_parked_slots_null(), 0);
        require("the floored model's lifecycle is intact", floored.seq_ready(2048), 0);
        int s = floored.seq_acquire();
        require("the floored model acquired a slot", s >= 0, 0);
        if (s >= 0) floored.seq_release(s);
        require("the floored model released cleanly", floored.state_bytes_outstanding() == 0, 0);
    }
    unsetenv("STRATA_GLM_SEQ_FLOOR");
    require("the floor env restored (unset)", getenv("STRATA_GLM_SEQ_FLOOR") == nullptr, 0);

    // ---- Task 4.2: the PAUSE path's registry-owned snapshots.  The legs live in
    // glm_multi.cu's selftest (the byte-identity legs read the state's private
    // regions - the glm_pages.cu selftest pattern: full access, linked by this
    // binary); the test file holds the pack path + the counters only.  Red
    // first: the selftest body lands with the implementation (the registry shape -
    // the undefined reference is the red).
    {
        int pc = 0, pf = 0;
        const bool ps_ok = strata::core::Glm5Model::glm_pause_snapshots_selftest(pack, &pc, &pf);
        g_checks += pc;
        g_failures += pf;
        require("the pause-snapshot selftest passed", ps_ok, 0);
    }

    printf("glm_seq_dynamic_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
