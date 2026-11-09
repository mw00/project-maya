// src/core/glm_warm_seq.cpp - glm-dsa B8.1 (warm): the file-order warm-up's switches, order, run count, counters and
// the kept warm digests (include/strata/core/glm_warm_seq.hpp).  Plain C++: tests/glm_dsa/glm_warm_seq_test.cpp links
// it alone; the engine side (Turn, computing the digests) is src/core/glm_warm_seq_model.cu.
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_warm_seq.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <utility>

namespace strata::core::glmwarm {

// read at each warm-up, not once a process: a test loads the same pack in both orders
bool seq_on(bool glm_dsa) { return strata::glmenv::env_flag("STRATA_GLM_WARM_SEQ", glm_dsa); }

bool sum_on() { return strata::glmenv::env_flag("STRATA_GLM_WARM_SUM", false); }

static bool before(const At& a, const At& b) { return a.shard != b.shard ? a.shard < b.shard : a.off < b.off; }

std::vector<size_t> file_order(const std::vector<At>& at) {
    std::vector<size_t> p(at.size());
    std::iota(p.begin(), p.end(), (size_t) 0);
    std::stable_sort(p.begin(), p.end(), [&](size_t a, size_t b) { return before(at[a], at[b]); });
    return p;
}

void read_order(std::vector<Piece>& pc) {
    std::stable_sort(pc.begin(), pc.end(), [](const Piece& a, const Piece& b) {
        return before(At{a.shard, a.off}, At{b.shard, b.off});
    });
}

size_t runs(const std::vector<Piece>& pc) {
    size_t n = 0;
    for (size_t i = 0; i < pc.size(); ++i)
        if (i == 0 || pc[i].shard != pc[i - 1].shard || pc[i].off != pc[i - 1].off + pc[i - 1].len) ++n;
    return n;
}

namespace {
std::atomic<uint64_t> g_warms{0}, g_jobs{0}, g_moved{0}, g_batches{0}, g_reads{0}, g_runs{0};
std::atomic<int> g_busy{0}, g_max_busy{0};
std::mutex g_digest_mu;
std::vector<std::pair<int, Digest>> g_digests;   // (part, its warm digest) since stats_reset()
}  // namespace

void note_digest(int part, const Digest& d) {
    std::lock_guard<std::mutex> lk(g_digest_mu);
    g_digests.emplace_back(part, d);
}

Digest warm_digest() {
    std::vector<std::pair<int, Digest>> v;
    {
        std::lock_guard<std::mutex> lk(g_digest_mu);
        v = g_digests;
    }
    std::stable_sort(v.begin(), v.end(), [](const std::pair<int, Digest>& a, const std::pair<int, Digest>& b) {
        return a.first < b.first;
    });
    Digest d;
    uint64_t h = 0x5741524D53554D31ull;
    for (size_t i = 0; i < v.size(); ++i) {
        h = mix(mix(h, (uint64_t) v[i].first), v[i].second.sum);
        d.vram += v[i].second.vram;
        d.ram += v[i].second.ram;
        d.bytes += v[i].second.bytes;
        d.parts += v[i].second.parts;
        d.ok = d.ok && v[i].second.ok;
    }
    d.sum = h;
    return d;
}

Stats stats() {
    Stats s;
    s.warms = g_warms.load();
    s.jobs = g_jobs.load();
    s.moved = g_moved.load();
    s.batches = g_batches.load();
    s.reads = g_reads.load();
    s.runs = g_runs.load();
    s.max_busy = g_max_busy.load();
    return s;
}

void stats_reset() {
    g_warms.store(0);
    g_jobs.store(0);
    g_moved.store(0);
    g_batches.store(0);
    g_reads.store(0);
    g_runs.store(0);
    g_max_busy.store(g_busy.load());
    std::lock_guard<std::mutex> lk(g_digest_mu);
    g_digests.clear();
}

void note(size_t jobs, size_t moved, size_t batches, size_t reads, size_t runs) {
    g_warms.fetch_add(1);
    g_jobs.fetch_add(jobs);
    g_moved.fetch_add(moved);
    g_batches.fetch_add(batches);
    g_reads.fetch_add(reads);
    g_runs.fetch_add(runs);
}

void busy_enter() {
    const int b = g_busy.fetch_add(1) + 1;
    int m = g_max_busy.load();
    while (b > m && !g_max_busy.compare_exchange_weak(m, b)) {
    }
}

void busy_leave() { g_busy.fetch_sub(1); }

}  // namespace strata::core::glmwarm
