#include "../src/ds4/ds4_recent_stage.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Stage = strata::ds4::RecentStage;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}

struct Rig {
    Stage stage;
    int allocations = 0, releases = 0, allow_calls = 0, fail_allocation = -1;
    bool allowed = true;
    std::vector<uint8_t*> allocated;
    bool init(size_t bytes = 96) {
        return stage.init(bytes,
            [&](size_t n) -> uint8_t* {
                if (allocations++ == fail_allocation) return nullptr;
                auto* p = new uint8_t[n];
                std::memset(p, 0xcd, n);
                allocated.push_back(p);
                return p;
            },
            [&](uint8_t* p) { release(p); },
            [&] { ++allow_calls; return allowed; });
    }
    void release(uint8_t* p) {
        const auto it = std::find(allocated.begin(), allocated.end(), p);
        check(it != allocated.end(), "release only owned allocation, once");
        allocated.erase(it);
        delete[] p;
        ++releases;
    }
    bool close(bool drained = true) {
        return stage.close([&] { return drained; }, [&](uint8_t* p) { release(p); });
    }
    ~Rig() { close(); check(allocated.empty(), "no fixture allocation leak"); }
};

// Three different matrix layouts exercise reuse across layers with different
// offsets, lengths, and padding. Only payload bytes are significant to kernels.
struct Expert {
    size_t bytes;
    std::array<size_t, 3> off, len;
    unsigned seed;
    uint8_t value(int r, size_t i) const { return uint8_t((seed + unsigned(r) * 37 + unsigned(i) * 13) % 251); }
    void read(uint8_t* p) const {
        for (int r = 0; r < 3; ++r)
            for (size_t i = 0; i < len[(size_t) r]; ++i) p[off[(size_t) r] + i] = value(r, i);
    }
    void verify(const std::vector<uint8_t>& p) const {
        check(p.size() == bytes, "correct expert blob size");
        for (int r = 0; r < 3; ++r)
            for (size_t i = 0; i < len[(size_t) r]; ++i)
                check(p[off[(size_t) r] + i] == value(r, i), "all matrix payload bytes match");
    }
};

struct Copy {
    int stage;
    const uint8_t* source;
    Expert expert;
    std::vector<uint8_t> submitted;
};
struct Backend {
    std::vector<Copy> pending;
    void submit(Stage& stage, int i, const Expert& expert) {
        const auto* p = stage.data(i);
        expert.verify(std::vector<uint8_t>(p, p + expert.bytes));
        pending.push_back({i, p, expert, std::vector<uint8_t>(p, p + expert.bytes)});
    }
    void complete(int i) {
        const auto it = std::find_if(pending.begin(), pending.end(), [&](const Copy& c) { return c.stage == i; });
        check(it != pending.end(), "completion has submitted work");
        const std::vector<uint8_t> copied(it->source, it->source + it->expert.bytes);
        check(copied == it->submitted, "source not overwritten before simulated DMA completes");
        it->expert.verify(copied);
        pending.erase(it);
    }
    bool drain() {
        while (!pending.empty()) complete(pending.front().stage);
        return true;
    }
};

static void allocation_and_fallback() {
    check(!Stage::fits(0) && Stage::fits(1) && Stage::fits(Stage::kMaxBufferBytes) &&
          !Stage::fits(Stage::kMaxBufferBytes + 1), "32 MiB per-buffer limit");
    for (int failure : {0, 1}) {
        Rig r; r.fail_allocation = failure;
        check(!r.init() && !r.stage.enabled(), "allocation failure disables pair");
        check(r.releases == failure && r.allocated.empty() && r.allow_calls == 0, "partial allocation rolled back");
    }
    { Rig r; r.allowed = false;
      check(!r.init() && !r.stage.enabled() && r.releases == 2, "headroom failure rolls back both"); }
    { Rig r;
      check(!r.init(Stage::kMaxBufferBytes + 1) && r.allocations == 0, "oversize allocates nothing"); }
    Rig r;
    int index = 17, reads = 0;
    const auto reader = [&](uint8_t*) { ++reads; return true; };
    check(r.stage.prepare(10, index, reader, reader) == Stage::Prepare::Unavailable && index == -1,
          "disabled pool selects original path");
    check(r.init(), "successful pair");
    for (auto* p : r.allocated)
        check(std::all_of(p, p + 96, [](uint8_t v) { return v == 0; }), "allocations initialized including padding");
    check(r.stage.prepare(97, index, reader, reader) == Stage::Prepare::Unavailable && reads == 0,
          "unsupported blob does not call readers");
    int store_reads = 0, file_reads = 0;
    check(r.stage.prepare(40, index,
          [&](uint8_t*) { ++store_reads; return false; },
          [&](uint8_t* p) { ++file_reads; p[0] = 42; return true; }) == Stage::Prepare::Ready,
          "store miss reads file into owned source");
    check(store_reads == 1 && file_reads == 1 && r.stage.data(index)[0] == 42, "file fallback used exactly once");
    check(r.stage.poll(index, Stage::Completion::Complete, [] { return false; }) == Stage::Completion::Complete,
          "completed transfer releases source without fencing");
    check(r.stage.prepare(40, index, [](uint8_t*) { return false; }, [](uint8_t*) { return false; }) ==
          Stage::Prepare::ReadFailed && index == -1, "read failure leaves no submitted job/source lease");
    check(r.stage.prepare(40, index, [](uint8_t*) { return true; },
          [](uint8_t*) { check(false, "file not read on store hit"); return false; }) == Stage::Prepare::Ready,
          "read-failed buffer is reusable");
}

static void repeated_worker_sources() {
    Rig r; Backend gpu; check(r.init(), "pair for repeated fills");
    const std::array<Expert, 3> experts{{
        {64, {0, 16, 40}, {11, 19, 17}, 5},
        {96, {0, 32, 64}, {25, 23, 29}, 71},
        {48, {0, 8, 24}, {5, 11, 19}, 137}
    }};
    int store_reads = 0, file_reads = 0;
    const auto prepare = [&](const Expert& e, bool hit, int& index) {
        return r.stage.prepare(e.bytes, index,
            [&](uint8_t* p) { ++store_reads; if (hit) e.read(p); return hit; },
            [&](uint8_t* p) { ++file_reads; e.read(p); return true; });
    };
    int a = -1, b = -1, c = -1;
    check(prepare(experts[0], true, a) == Stage::Prepare::Ready && a == 0, "first buffer selected");
    gpu.submit(r.stage, a, experts[0]);
    check(prepare(experts[1], false, b) == Stage::Prepare::Ready && b == 1, "second buffer selected on file miss");
    gpu.submit(r.stage, b, experts[1]);
    int fences = 0;
    r.stage.poll(a, Stage::Completion::Pending, [&] { ++fences; return true; });
    const int reads_before_wait = store_reads + file_reads;
    check(prepare(experts[2], false, c) == Stage::Prepare::Waiting && c == -1 &&
          reads_before_wait == store_reads + file_reads && fences == 0, "busy/pending buffers cause no read or reuse");
    // Deliberately complete the second buffer first: ownership is per source,
    // not dependent on a FIFO assumption about events or job-vector positions.
    gpu.complete(b);
    r.stage.poll(b, Stage::Completion::Complete, [] { return false; });
    check(prepare(experts[2], true, c) == Stage::Prepare::Ready && c == b, "only completed buffer reused");
    gpu.submit(r.stage, c, experts[2]);
    gpu.complete(a);
    r.stage.poll(a, Stage::Completion::Complete, [] { return false; });
    gpu.complete(c);
    r.stage.poll(c, Stage::Completion::Complete, [] { return false; });
    for (int cycle = 0; cycle < 24; ++cycle) {
        const auto& e = experts[(size_t) cycle % experts.size()];
        check(prepare(e, cycle % 2 == 0, a) == Stage::Prepare::Ready, "repeated mixed-geometry fill");
        gpu.submit(r.stage, a, e);
        gpu.complete(a);
        r.stage.poll(a, Stage::Completion::Complete, [] { return false; });
    }
    check(file_reads > 1, "both ready-store and file sources exercised repeatedly");
    check(r.stage.close([&] { return gpu.drain(); }, [&](uint8_t* p) {
        check(gpu.pending.empty(), "drain precedes release"); r.release(p);
    }), "worker shutdown releases pair");
    check(r.releases == 2 && !r.stage.enabled(), "one release per buffer");
}

static void failures_and_shutdown() {
    Rig r; Backend gpu; check(r.init(), "pair for failures");
    const Expert e{48, {0, 16, 32}, {11, 13, 9}, 19};
    const auto prepare = [&](int& i) {
        return r.stage.prepare(e.bytes, i, [&](uint8_t* p) { e.read(p); return true; },
                               [](uint8_t*) { return false; });
    };
    int a = -1, b = -1, next = -1;
    check(prepare(a) == Stage::Prepare::Ready, "source before failed issue");
    gpu.submit(r.stage, a, e); // an issue/event-record failure may leave a partial copy queued
    bool fenced = false;
    r.stage.failed(a, [&] { fenced = true; return gpu.drain(); });
    check(fenced && prepare(next) == Stage::Prepare::Ready && next == a, "failed issue fenced before reuse");
    gpu.submit(r.stage, next, e);
    check(r.stage.poll(next, Stage::Completion::Failed, [&] { return gpu.drain(); }) == Stage::Completion::Failed,
          "event error is a failed fill, not a permanent pending fill");
    check(prepare(a) == Stage::Prepare::Ready, "event-error source reusable after successful fence");
    r.stage.failed(a, [] { return false; });
    check(prepare(b) == Stage::Prepare::Ready && b != a, "failed fence quarantines its source");
    check(prepare(next) == Stage::Prepare::Waiting, "one quarantined and one busy waits for busy source");
    r.stage.failed(b, [] { return false; });
    check(prepare(next) == Stage::Prepare::Unavailable, "both quarantined select original path without reusing sources");
    check(!r.close(false) && r.releases == 0 && r.stage.enabled(), "failed shutdown drain frees nothing");
    check(r.close(true) && r.releases == 2, "successful final drain can release quarantined sources");
    check(r.close(true) && r.releases == 2, "shutdown is idempotent");
    check(r.init(), "pair can be initialized again after close");
    check(prepare(a) == Stage::Prepare::Ready, "new source before shutdown with work outstanding");
    gpu.submit(r.stage, a, e);
    check(r.stage.close([&] { return gpu.drain(); }, [&](uint8_t* p) {
        check(gpu.pending.empty(), "outstanding DMA finishes before host release"); r.release(p);
    }), "normal shutdown drains outstanding work");
}

int main() {
    allocation_and_fallback();
    repeated_worker_sources();
    failures_and_shutdown();
    std::puts("PASS: production recent-stage allocation, store/file sources, two-buffer reuse, mixed geometry, failures, shutdown");
}
