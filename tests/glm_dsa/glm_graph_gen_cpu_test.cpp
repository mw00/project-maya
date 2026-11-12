// tests/glm_dsa/glm_graph_gen_cpu_test.cpp - glm-dsa B15.6 on the CPU: the graph generation in the decode-graph key.
//
// The key logic of src/core/glm_dsa_graphs.hpp (DsaGraphs::enqueue through glmd::GraphIndex: an entry per part and
// state arena, a key that differs from the entry's drops it and re-captures) over a mock capture, with glmd::GraphKey and the process-wide generation:
//   - the same key replays; any bump (an arena reallocation) makes the next key differ -> exactly one re-capture per
//     part, also when the reallocated arena came back at the SAME base address (the case the address fields miss);
//   - graph_generation_note bumps only for a new name or a changed (base, bytes), and counts per name;
//   - bumps from several threads are all counted and the generation stays monotonic.
#include "strata/kernels/glmd_graph.hpp"

#include <atomic>
#include <cstdio>
#include <map>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                        \
    do {                                                     \
        if (!(c)) {                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fail;                                        \
        }                                                    \
    } while (0)

// DsaGraphs::enqueue's bookkeeping without CUDA: the same glmd::GraphIndex, a "graph" being its key
struct MockGraphs {
    glmd::GraphIndex idx;
    std::map<int, glmd::GraphKey> graphs;   // by slot
    int captures = 0, replays = 0, invalidations = 0, evictions = 0;
    void enqueue(const glmd::GraphKey& k) {
        const glmd::GraphIndex::Pick pk = idx.pick(k);
        if (pk.drop >= 0) graphs.erase(pk.drop);
        invalidations += pk.invalidated;
        evictions += pk.evicted;
        if (pk.replay) {
            ++replays;
            if (graphs.at(pk.slot) != k) ++bad_replays;
            return;
        }
        graphs[pk.slot] = k;
        idx.commit(pk.slot, k);
        ++captures;
    }
    int bad_replays = 0;
};

// graph_prepare's key (glm_dsa_mixer.cu), with the fields this test varies
glmd::GraphKey key_of(int part, const void* arena, int64_t p, const void* state = (const void*) 0x2000) {
    glmd::GraphKey k;
    k.part = part;
    k.l0 = part;
    k.l1 = part + 26;
    k.arena = arena;
    k.residual = (const void*) 0x1000;
    k.state = state;
    k.regime = glmd::graph_regime(p, 2048, false, 16384, false, 32768);
    k.n_cap = glmd::graph_n_cap(p, 32768);
    k.gen = glmd::graph_generation();
    return k;
}

void key_logic() {
    const uint64_t g0 = glmd::graph_generation();
    CHECK(g0 >= 1, "the generation starts at 1");
    glmd::GraphKey a = key_of(0, (void*) 0xA000, 100), b = a;
    CHECK(a == b, "equal keys");
    b.gen = a.gen + 1;
    CHECK(a != b, "a key differs in its generation alone");
    MockGraphs G;
    char arena_a[64], arena_b[64];
    for (int64_t p = 100; p < 110; ++p)
        for (int part : {0, 26, 54}) G.enqueue(key_of(part, arena_a, p));
    CHECK(G.captures == 3 && G.replays == 27 && G.invalidations == 0, "steady: 3 captures, 27 replays (got %d %d %d)",
          G.captures, G.replays, G.invalidations);
    // a context resize mid-session: the DSA arena is freed and allocated again AT THE SAME BASE (cudaMalloc often
    // hands the freed block back): without the generation the key would be equal and the old graph would replay
    const uint64_t g1 = glmd::graph_generation_bump("test: dsa arena");
    CHECK(g1 == g0 + 1 && glmd::graph_generation() == g1, "bump returns the new generation");
    for (int64_t p = 110; p < 115; ++p)
        for (int part : {0, 26, 54}) G.enqueue(key_of(part, arena_a, p));
    CHECK(G.captures == 6 && G.invalidations == 3 && G.replays == 27 + 12,
          "after the bump: every part re-captures once, then replays (got %d %d %d)", G.captures, G.replays,
          G.invalidations);
    // a move to another base (the address field alone would catch it; the generation agrees)
    glmd::graph_generation_note("test: moe staging", arena_b, 4096);
    for (int part : {0, 26, 54}) G.enqueue(key_of(part, arena_b, 115));
    CHECK(G.captures == 9 && G.invalidations == 6, "moved arena: 3 more captures");
    // a setup that runs again with the same allocation does not force a re-capture
    const uint64_t g2 = glmd::graph_generation();
    CHECK(glmd::graph_generation_note("test: moe staging", arena_b, 4096) == g2, "same note: no bump");
    CHECK(glmd::graph_generation_note("test: moe staging", arena_b, 8192) == g2 + 1, "bytes changed: bump");
    CHECK(glmd::graph_generation_note("test: mtp rows", arena_b, 8192) == g2 + 2, "a new name: bump");
    CHECK(glmd::graph_generation_bumps("test: moe staging") == 2 && glmd::graph_generation_bumps("test: mtp rows") == 1 &&
              glmd::graph_generation_bumps("test: dsa arena") == 1 && glmd::graph_generation_bumps("nothing") == 0,
          "per-name counts");
    // a regime boundary alone still re-captures (unchanged B10.1 behaviour)
    const int before = G.captures;
    G.enqueue(key_of(0, arena_b, 2047));
    G.enqueue(key_of(0, arena_b, 2048));
    CHECK(G.captures == before + 2, "crossing p + 1 > top_k re-captures");
}

// review fix: several sequences (MAYA_GLM_SEQS / --parallel) bind their own state arenas; the cache keeps a graph
// per (part, state) up to the sequence count, so alternating sequences replay instead of re-capturing every token
void sequences() {
    MockGraphs G;
    G.idx.set_per_part(2);
    char arena[64];
    const void* st[2] = {(const void*) 0x10000, (const void*) 0x20000};
    for (int64_t p = 100; p < 120; ++p)
        for (int s = 0; s < 2; ++s)
            for (int part : {0, 26, 54}) G.enqueue(key_of(part, arena, p, st[s]));
    CHECK(G.captures == 6 && G.replays == 114 && G.invalidations == 0 && G.evictions == 0 && G.bad_replays == 0,
          "2 sequences x 3 parts: 6 captures, then every token replays (got %d %d %d %d)", G.captures, G.replays,
          G.invalidations, G.evictions);
    CHECK(G.idx.live() == 6 && G.idx.latest(26) >= 0 && G.idx.latest(7) < 0, "6 live graphs; latest() per part");
    // the old one-entry-per-part rule: every switch re-captured
    MockGraphs one;
    for (int64_t p = 100; p < 120; ++p)
        for (int s = 0; s < 2; ++s) one.enqueue(key_of(0, arena, p, st[s]));
    CHECK(one.captures == 40 && one.replays == 0 && one.evictions == 39,
          "per_part 1 (one sequence's setting) alternating states: a capture per token (got %d)", one.captures);
    // a third state beyond per_part 2: the least recently used graph of that part goes, the others stay
    const void* third = (const void*) 0x30000;
    G.enqueue(key_of(0, arena, 120, st[0]));   // part 0: st[0] used now, st[1] the least recent
    G.enqueue(key_of(0, arena, 120, third));
    CHECK(G.evictions == 1 && G.captures == 7, "a third state evicts one graph");
    const int before = G.captures;
    G.enqueue(key_of(0, arena, 120, st[0]));
    CHECK(G.captures == before && G.bad_replays == 0, "the recently used sequence still replays");
    G.enqueue(key_of(0, arena, 121, st[1]));
    CHECK(G.captures == before + 1, "the evicted one re-captures");
    // a key change on one sequence (a newer generation) invalidates that sequence's graph only
    glmd::graph_generation_bump("test: seqs");
    const int inv = G.invalidations;
    G.enqueue(key_of(26, arena, 130, st[0]));
    CHECK(G.invalidations == inv + 1 && G.bad_replays == 0, "a generation bump: an invalidation, the slot re-captured");
}

void threads() {
    const uint64_t g0 = glmd::graph_generation(), n0 = glmd::graph_generation_bumps(nullptr);
    std::vector<std::thread> th;
    std::atomic<int> bad{0};
    for (int t = 0; t < 4; ++t)
        th.emplace_back([&] {
            uint64_t last = 0;
            for (int i = 0; i < 1000; ++i) {
                const uint64_t g = glmd::graph_generation_bump("test: threads");
                if (g <= last) bad.fetch_add(1);
                last = g;
            }
        });
    for (auto& t : th) t.join();
    CHECK(bad.load() == 0, "per-thread monotonic");
    CHECK(glmd::graph_generation() == g0 + 4000, "4000 bumps counted (got %llu)",
          (unsigned long long) (glmd::graph_generation() - g0));
    CHECK(glmd::graph_generation_bumps(nullptr) == n0 + 4000, "the counters agree");
}

}  // namespace

int main() {
    key_logic();
    sequences();
    threads();
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
