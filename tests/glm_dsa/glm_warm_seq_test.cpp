// tests/glm_dsa/glm_warm_seq_test.cpp - glm-dsa B8.1 (warm, CPU): the file-order expert warm-up's host side
// (include/strata/core/glm_warm_seq.hpp, src/core/glm_warm_seq.cpp).
//   1. the switches: STRATA_GLM_WARM_SEQ defaults on for glm-dsa and off for Flash, both ways settable, an unknown
//      value keeps the default; STRATA_GLM_WARM_SUM defaults off;
//   2. file_order: a stable permutation into (shard, offset) order - ties keep their order, a later shard comes after
//      any offset of an earlier one;
//   3. read_order and runs: a batch's reads sorted into file order, adjacent reads one run, a shard change a new run;
//   4. fast_warm's job list in Maya's slot order (each layer's VRAM slots in a scrambled profile order, then the RAM
//      tier round-robin over the layers) on a mock pack (several shards, each layer's down / gate / up tensors): in
//      file order it is a permutation of the same jobs, the experts of a layer come together in id order, and every
//      batch of 8 reads as 3 contiguous runs; the slot order's batches, even with each batch's reads sorted, read as
//      many more.  Printed at GLM-5.3's shape (76 layers x 256 experts, 114 VRAM slots a layer as served on the test PC);
//   5. hash_bytes / mix: a flipped bit anywhere (low or high bit of a word, the tail), the length and the order all
//      change the hash; equal bytes hash equal;
//   6. the counters: note() adds up, busy_enter / busy_leave keep the most warm-ups streaming at once; the kept warm
//      digests fold in part order whatever order the parts finished in, another assignment of parts folds differently,
//      a failed part makes the load's digest not ok, stats_reset clears them.
#include "strata/core/glm_warm_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

using namespace strata::core::glmwarm;

void test_switches() {
    std::printf("[1. switches]\n");
    unsetenv("STRATA_GLM_WARM_SEQ");
    unsetenv("STRATA_GLM_WARM_SUM");
    check(seq_on(true) && !seq_on(false), "STRATA_GLM_WARM_SEQ unset: on for glm-dsa, off for Flash");
    setenv("STRATA_GLM_WARM_SEQ", "0", 1);
    check(!seq_on(true) && !seq_on(false), "STRATA_GLM_WARM_SEQ=0: off for both (Maya's slot order)");
    setenv("STRATA_GLM_WARM_SEQ", "on", 1);
    check(seq_on(true) && seq_on(false), "STRATA_GLM_WARM_SEQ=on: on for both");
    setenv("STRATA_GLM_WARM_SEQ", "sideways", 1);
    check(seq_on(true) && !seq_on(false), "an unknown value keeps each arch's default");
    unsetenv("STRATA_GLM_WARM_SEQ");
    check(!sum_on(), "STRATA_GLM_WARM_SUM unset: off");
    setenv("STRATA_GLM_WARM_SUM", "1", 1);
    check(sum_on(), "STRATA_GLM_WARM_SUM=1: on");
    unsetenv("STRATA_GLM_WARM_SUM");
}

void test_file_order() {
    std::printf("[2. file_order]\n");
    std::vector<At> at;
    at.push_back(At{0, 5});
    at.push_back(At{1, 0});
    at.push_back(At{0, 5});
    at.push_back(At{0, 1000000000000ull});
    at.push_back(At{0, 1});
    const std::vector<size_t> p = file_order(at);
    const std::vector<size_t> want = {4, 0, 2, 3, 1};
    check(p == want, "(0,5) (1,0) (0,5) (0,1e12) (0,1) -> 4 0 2 3 1 (ties in order, shard 1 last)");
    check(file_order(std::vector<At>()).empty(), "no jobs: an empty order");
}

void test_runs() {
    std::printf("[3. read_order / runs]\n");
    std::vector<Piece> pc;
    pc.push_back(Piece{0, 300, 100, 0});
    pc.push_back(Piece{1, 0, 50, 1});
    pc.push_back(Piece{0, 100, 100, 2});
    pc.push_back(Piece{0, 200, 100, 3});
    pc.push_back(Piece{0, 600, 10, 4});
    check(runs(pc) == 4, "as given: 4 runs, [100,300) the only join (" + std::to_string(runs(pc)) + ")");
    read_order(pc);
    bool sorted = true;
    for (size_t i = 1; i < pc.size(); ++i)
        sorted = sorted && (pc[i - 1].shard < pc[i].shard ||
                            (pc[i - 1].shard == pc[i].shard && pc[i - 1].off <= pc[i].off));
    check(sorted && pc[0].job == 2 && pc[3].job == 4 && pc[4].job == 1, "read_order: shard 0 by offset, then shard 1");
    check(runs(pc) == 3, "sorted: [100,400) one run, 600 another, shard 1 a third (" + std::to_string(runs(pc)) + ")");
    std::vector<Piece> cross;
    cross.push_back(Piece{0, 0, 100, 0});
    cross.push_back(Piece{1, 100, 100, 1});
    check(runs(cross) == 2, "the same offsets in another shard do not continue a run");
}

// a mock pack: layer il in shard il * shards / L, each layer's tensors down | gate | up (E experts each) after a
// 4 KB gap; gu / dn the gate-or-up and the down slice of one expert
struct Mock {
    int L = 0, E = 0, nv = 0, shards = 1;
    uint64_t gu = 0, dn = 0;
    std::vector<int> shard;
    std::vector<uint64_t> down_off, gate_off, up_off;
};

Mock make_mock(int L, int E, int nv, int shards, uint64_t gu, uint64_t dn) {
    Mock m;
    m.L = L;
    m.E = E;
    m.nv = nv;
    m.shards = shards;
    m.gu = gu;
    m.dn = dn;
    std::vector<uint64_t> at((size_t) shards, 0);
    for (int il = 0; il < L; ++il) {
        const int s = il * shards / L;
        uint64_t& o = at[(size_t) s];
        o += 4096;
        m.shard.push_back(s);
        m.down_off.push_back(o);
        o += (uint64_t) E * dn;
        m.gate_off.push_back(o);
        o += (uint64_t) E * gu;
        m.up_off.push_back(o);
        o += (uint64_t) E * gu;
    }
    return m;
}

struct Job {
    int il, e;
    bool vram;
};

// fast_warm's job list in Maya's order: each layer's first nv experts of its (scrambled) order into VRAM, then the
// RAM tier one expert per layer per round
std::vector<Job> slot_order(const Mock& m, unsigned seed) {
    std::vector<std::vector<int>> order((size_t) m.L);
    std::mt19937 rng(seed);
    for (int il = 0; il < m.L; ++il) {
        order[(size_t) il].resize((size_t) m.E);
        std::iota(order[(size_t) il].begin(), order[(size_t) il].end(), 0);
        std::shuffle(order[(size_t) il].begin(), order[(size_t) il].end(), rng);
    }
    std::vector<Job> jobs;
    std::vector<int> next((size_t) m.L, m.nv);
    for (int il = 0; il < m.L; ++il)
        for (int j = 0; j < m.nv; ++j) jobs.push_back(Job{il, order[(size_t) il][(size_t) j], true});
    for (bool progress = true; progress;) {
        progress = false;
        for (int il = 0; il < m.L; ++il)
            if (next[(size_t) il] < m.E) {
                jobs.push_back(Job{il, order[(size_t) il][(size_t) next[(size_t) il]++], false});
                progress = true;
            }
    }
    return jobs;
}

// the runs of every 8-job batch, each batch's 24 reads sorted into file order (fast_warm's file-order reader)
size_t batch_runs(const Mock& m, const std::vector<Job>& jobs, size_t& batches) {
    size_t n = 0;
    batches = 0;
    std::vector<Piece> pc;
    for (size_t b = 0; b < jobs.size(); b += 8) {
        pc.clear();
        for (size_t k = 0; k < 8 && b + k < jobs.size(); ++k) {
            const Job& J = jobs[b + k];
            const int s = m.shard[(size_t) J.il];
            pc.push_back(Piece{s, m.gate_off[(size_t) J.il] + (uint64_t) J.e * m.gu, (size_t) m.gu, (int) (3 * k)});
            pc.push_back(Piece{s, m.up_off[(size_t) J.il] + (uint64_t) J.e * m.gu, (size_t) m.gu, (int) (3 * k + 1)});
            pc.push_back(Piece{s, m.down_off[(size_t) J.il] + (uint64_t) J.e * m.dn, (size_t) m.dn, (int) (3 * k + 2)});
        }
        read_order(pc);
        n += runs(pc);
        ++batches;
    }
    return n;
}

void warm_case(const char* name, int L, int E, int nv, int shards, uint64_t gu, uint64_t dn, bool print) {
    const Mock m = make_mock(L, E, nv, shards, gu, dn);
    const std::vector<Job> slot = slot_order(m, 17u + (unsigned) L);
    std::vector<At> at;
    for (const Job& j : slot)
        at.push_back(At{m.shard[(size_t) j.il], m.gate_off[(size_t) j.il] + (uint64_t) j.e * m.gu});
    const std::vector<size_t> perm = file_order(at);
    std::vector<Job> file;
    size_t moved = 0;
    for (size_t i = 0; i < perm.size(); ++i) {
        file.push_back(slot[perm[i]]);
        if (perm[i] != i) ++moved;
    }
    // a permutation: every expert once, in the tier the slot order put it in (VRAM: a layer's first nv of its order)
    std::vector<int> seen((size_t) (L * E), 0);
    std::vector<char> vram((size_t) (L * E), 0);
    for (const Job& j : slot) vram[(size_t) (j.il * E + j.e)] = j.vram ? 1 : 0;
    bool tiers_kept = true;
    for (const Job& j : file) {
        ++seen[(size_t) (j.il * E + j.e)];
        tiers_kept = tiers_kept && (vram[(size_t) (j.il * E + j.e)] != 0) == j.vram;
    }
    const bool once = std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; });
    check(file.size() == slot.size() && once && tiers_kept,
          std::string(name) + ": the file order holds every expert once, each with the tier the slot order gave it");
    bool by_layer = true;
    for (size_t i = 1; i < file.size(); ++i)
        by_layer = by_layer &&
                   (file[i - 1].il < file[i].il || (file[i - 1].il == file[i].il && file[i - 1].e < file[i].e));
    check(by_layer && moved > 0, std::string(name) + ": layer by layer, each layer's experts in id order (" +
                                     std::to_string(moved) + " of " + std::to_string(file.size()) + " moved)");
    size_t b_slot = 0, b_file = 0;
    const size_t r_slot = batch_runs(m, slot, b_slot), r_file = batch_runs(m, file, b_file);
    check(r_file == 3 * b_file, std::string(name) + ": file order, 3 runs a batch (" + std::to_string(r_file) + " in " +
                                    std::to_string(b_file) + " batches)");
    check(r_slot > 2 * r_file, std::string(name) + ": slot order, over twice the runs even with each batch sorted (" +
                                   std::to_string(r_slot) + ")");
    if (print)
        std::printf("  %s: slot order %.2f runs a batch (24 reads, 8 threads at once), file order %.2f (one thread); "
                    "%zu batches, %.1f MB a run in file order\n", name, (double) r_slot / (double) b_slot,
                    (double) r_file / (double) b_file, b_file,
                    (double) (L * E) * (double) (2 * gu + dn) / (double) r_file / 1e6);
}

void test_warm_order() {
    std::printf("[4. fast_warm's jobs: slot order vs file order]\n");
    warm_case("small (12 x 16, 13 VRAM slots, 1 shard)", 12, 16, 13, 1, 4352, 5120, false);
    warm_case("4 shards (20 x 64, 40 VRAM slots)", 20, 64, 40, 4, 3000, 3100, false);
    // GLM-5.3 UD-IQ1_S as served: 75 trunk MoE layers + the NextN block, 256 experts, ~3.45 MB a slice, 114 slots
    warm_case("GLM-5.3 (76 x 256, 114 VRAM slots, 5 shards)", 76, 256, 114, 5, 3450000, 3450000, true);
}

void test_hash() {
    std::printf("[5. hash_bytes / mix]\n");
    std::vector<uint8_t> a(4099);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t) (i * 131u + 7u);
    const uint64_t h0 = hash_bytes(a.data(), a.size());
    check(hash_bytes(a.data(), a.size()) == h0, "equal bytes, equal hash");
    bool all = true;
    const size_t where[] = {0, 7, 2048, 4095, 4098};
    for (size_t pos : where)
        for (int bit = 0; bit < 8; ++bit) {
            std::vector<uint8_t> b = a;
            b[pos] ^= (uint8_t) (1u << bit);
            all = all && hash_bytes(b.data(), b.size()) != h0;
        }
    check(all, "one flipped bit (bytes 0, 7, 2048, 4095 and the tail's 4098, every bit) changes the hash");
    std::vector<uint8_t> z(a);
    z.push_back(0);
    check(hash_bytes(z.data(), z.size()) != h0 && hash_bytes(a.data(), 8) != hash_bytes(a.data(), 9),
          "a trailing zero byte changes the hash");
    check(mix(mix(1, 2), 3) != mix(mix(1, 3), 2) && mix(0, 0) != 0, "mix depends on the order");
}

void test_counters() {
    std::printf("[6. counters]\n");
    stats_reset();
    note(10, 4, 2, 30, 6);
    note(6, 0, 1, 18, 3);
    const Stats s = stats();
    check(s.warms == 2 && s.jobs == 16 && s.moved == 4 && s.batches == 3 && s.reads == 48 && s.runs == 9,
          "note() adds up over warm-ups");
    busy_enter();
    busy_enter();
    busy_leave();
    busy_leave();
    check(stats().max_busy == 2, "two warm-ups at once: max_busy 2");
    stats_reset();
    busy_enter();
    busy_leave();
    busy_enter();
    busy_leave();
    check(stats().max_busy == 1 && stats().warms == 0, "one after the other (after a reset): max_busy 1");
    Digest a, b;
    a.sum = 11;
    a.vram = 2;
    a.ram = 3;
    a.bytes = 100;
    a.parts = 1;
    b.sum = 22;
    b.vram = 1;
    b.ram = 1;
    b.bytes = 50;
    b.parts = 1;
    stats_reset();
    note_digest(1, b);
    note_digest(0, a);
    const Digest w1 = warm_digest();
    stats_reset();
    note_digest(0, a);
    note_digest(1, b);
    const Digest w2 = warm_digest();
    check(w1.sum == w2.sum && w1.parts == 2 && w1.vram == 3 && w1.ram == 4 && w1.bytes == 150 && w1.ok,
          "warm_digest folds the parts in part order, whatever order they finished in");
    stats_reset();
    note_digest(0, b);
    note_digest(1, a);
    check(warm_digest().sum != w1.sum, "the same part digests on other parts fold differently");
    b.ok = false;
    note_digest(2, b);
    check(!warm_digest().ok, "a part whose device copy failed makes the load's digest not ok");
    stats_reset();
    check(warm_digest().parts == 0, "stats_reset clears the kept digests");
}

}  // namespace

int main() {
    std::printf("glm_warm_seq_test: the file-order expert warm-up (B8.1, warm)\n");
    test_switches();
    test_file_order();
    test_runs();
    test_warm_order();
    test_hash();
    test_counters();
    if (g_fail) {
        std::printf("glm_warm_seq_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_warm_seq_test: ok\n");
    return 0;
}
