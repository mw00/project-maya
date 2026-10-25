// src/mm3/mm3_cpu.cpp - maya-mm3's CPU lane (mm3_cpu.hpp).
//
// One call: the experts' gate and up row sums in jobs of 16 slices (256 inputs, each job rotating its own two blocks
// of x), then per 128-output band the outputs, swigluoai and the down matrix's rotated input block, then down's row
// sums in jobs of 8 slices, then down's bands - four phases on a pool of spinning workers (a decode step is a few
// milliseconds), the caller one of them.
#include "mm3_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

namespace kc = strata::kernels::cpu;

namespace mm3 {

struct CpuLane::Impl {
    int dim, ff;
    float alpha, limit;
    std::vector<std::thread> th;
    // a phase's jobs in one word: `ticket` holds its generation (high 32 bits), its job count (16) and the next job (low
    // 16); a worker takes job j only by a compare-and-swap of the word it read, j below the count in that same word - so
    // one late from the phase before can neither take a job of the new phase (nor run it with the new function) nor
    // count one there (a count of its own, read apart from the word, let a late worker take job n of the old phase as
    // one of the new phase's before its word was stored: a job run twice, the phase over one job early)
    std::atomic<uint64_t> gen{0}, ticket{0};
    std::atomic<int> done{0};
    static constexpr int kMaxJobs = 0xffff;
    std::atomic<const std::function<void(int)>*> fn{nullptr};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> sleepers{0};
    // scratch, for up to 8 experts
    static constexpr int kMaxE = 8;
    int GQ, GB, DQ, DB, EB, FB;   // gate / up K jobs and bands, down K jobs and bands, x blocks, h blocks
    std::vector<float> xh, xsb, acc, xhd, xsd, accd, yd;
    std::vector<float> mxh, mxsb, macc, mxhd, mxsd, maccd, myd;   // (run_multi's, by entry: allocated on first use)

    void worker(int i) {
        // (pinned to the physical cores, after the caller's: CPUs 1.. here are cores of both sockets)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET((i + 1) % (int) std::thread::hardware_concurrency(), &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        uint64_t seen = 0;
        for (;;) {
            int spins = 0;
            while (gen.load(std::memory_order_acquire) == seen) {
                if (stop.load()) return;
                if (++spins < 200000) {
                    _mm_pause();
                } else {   // idle a while: sleep until the next call
                    std::unique_lock<std::mutex> lk(mu);
                    ++sleepers;
                    cv.wait_for(lk, std::chrono::milliseconds(50),
                                [&] { return gen.load() != seen || stop.load(); });
                    --sleepers;
                    spins = 0;
                }
            }
            seen = gen.load(std::memory_order_acquire);
            work(seen);
        }
    }
    void work(uint64_t g) {
        for (uint64_t t = ticket.load(); (t >> 32) == (g & 0xffffffffull);) {
            const uint32_t j = (uint32_t) (t & 0xffffu), n = (uint32_t) ((t >> 16) & 0xffffu);
            if (j >= n) return;
            if (!ticket.compare_exchange_weak(t, t + 1)) continue;   // (t now holds the current ticket)
            (*fn.load())((int) j);
            done.fetch_add(1, std::memory_order_release);
            t = ticket.load();
        }
    }
    void parallel(int n, const std::function<void(int)>& f) {
        if (n > kMaxJobs) {   // (the word's count: more in parts)
            for (int b = 0; b < n; b += kMaxJobs) {
                const std::function<void(int)> fp = [&](int j) { f(b + j); };
                parallel(std::min(kMaxJobs, n - b), fp);
            }
            return;
        }
        if (n <= 0) return;
        const uint64_t g = gen.load() + 1;
        fn.store(&f);
        done.store(0);
        ticket.store(((g & 0xffffffffull) << 32) | ((uint64_t) n << 16));   // (after the function: a worker that sees
                                                                            //  the word sees it)
        gen.store(g, std::memory_order_release);
        if (sleepers.load()) cv.notify_all();
        work(g);
        while (done.load(std::memory_order_acquire) < n) _mm_pause();
    }
};

CpuLane::CpuLane(int threads, int dim, int ff, float alpha, float limit) : p(new Impl) {
    p->dim = dim;
    p->ff = ff;
    p->alpha = alpha;
    p->limit = limit;
    p->GQ = dim / 256;    // gate / up: jobs of 16 slices (256 inputs)
    p->GB = ff / 128;
    p->DQ = ff / 128;     // down: jobs of 8 slices (128 inputs)
    p->DB = dim / 128;
    p->EB = dim / 128;
    p->FB = ff / 128;
    const int E = Impl::kMaxE;
    p->xh.resize((size_t) E * 2 * dim);
    p->xsb.resize((size_t) E * 2 * p->EB);
    p->acc.resize((size_t) E * 2 * p->GQ * ff);
    p->xhd.resize((size_t) E * ff);
    p->xsd.resize((size_t) E * p->FB);
    p->accd.resize((size_t) E * p->DQ * dim);
    p->yd.resize((size_t) E * dim);
    for (int i = 0; i < threads; ++i) p->th.emplace_back([this, i] { p->worker(i); });
}

CpuLane::~CpuLane() {
    p->stop = true;
    p->cv.notify_all();
    for (auto& t : p->th) t.join();
    delete p;
}

int CpuLane::threads() const { return (int) p->th.size() + 1; }

void CpuLane::run(const std::vector<CpuExpert>& ex, const float* x, float* out) {
    Impl& I = *p;
    const int ne = (int) std::min<size_t>(ex.size(), Impl::kMaxE);
    const int dim = I.dim, ff = I.ff;
    if (ne == 0) {
        std::fill(out, out + dim, 0.0f);
        return;
    }
    auto xh = [&](int e, int r) { return I.xh.data() + ((size_t) e * 2 + r) * dim; };
    auto xsb = [&](int e, int r) { return I.xsb.data() + ((size_t) e * 2 + r) * I.EB; };
    auto acc = [&](int e, int r, int q) { return I.acc.data() + (((size_t) e * 2 + r) * I.GQ + q) * ff; };
    auto xhd = [&](int e) { return I.xhd.data() + (size_t) e * ff; };
    auto xsd = [&](int e) { return I.xsd.data() + (size_t) e * I.FB; };
    auto accd = [&](int e, int q) { return I.accd.data() + ((size_t) e * I.DQ + q) * dim; };
    auto yd = [&](int e) { return I.yd.data() + (size_t) e * dim; };
    // gate / up row sums: job (e, r, q) - x's blocks 2q, 2q + 1 rotated, the slices [16 q, 16 q + 16) for every column
    const std::function<void(int)> fa = [&](int j) {
        const int e = j / (2 * I.GQ), r = (j / I.GQ) % 2, q = j % I.GQ;
        const kc::Exl3Part& m = ex[(size_t) e].p[r];
        for (int b = 2 * q; b < 2 * q + 2; ++b)
            xsb(e, r)[b] = kc::exl3_cpu_prepare_block(m.suh + b * 128, x + b * 128, xh(e, r) + b * 128);
        kc::exl3_cpu_rows_acc(m, xh(e, r), q * 16, (q + 1) * 16, acc(e, r, q));
    };
    I.parallel(ne * 2 * I.GQ, fa);
    // per band: gate / up outputs, swigluoai, down's rotated input block
    const std::function<void(int)> fb = [&](int j) {
        const int e = j / I.GB, b = j % I.GB;
        float y[2][128], a[128], hb[128];
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 128; ++c) a[c] = 0.0f;
            for (int q = 0; q < I.GQ; ++q) {
                const float* aq = acc(e, r, q) + b * 128;
                for (int c = 0; c < 128; ++c) a[c] += aq[c];
            }
            double xs = 0.0;
            for (int k = 0; k < I.EB; ++k) xs += xsb(e, r)[k];
            kc::exl3_cpu_band_out(ex[(size_t) e].p[r], a, (float) xs, b * 128, y[r]);
        }
        for (int c = 0; c < 128; ++c) {
            const float g = std::min(y[0][c], I.limit);
            const float u = std::min(std::max(y[1][c], -I.limit), I.limit);
            hb[c] = g / (1.0f + std::exp(-I.alpha * g)) * (u + 1.0f);
        }
        const kc::Exl3Part& d = ex[(size_t) e].p[2];
        xsd(e)[b] = kc::exl3_cpu_prepare_block(d.suh + b * 128, hb, xhd(e) + b * 128);
    };
    I.parallel(ne * I.GB, fb);
    // down's row sums: job (e, q) - the slices [8 q, 8 q + 8) for every column
    const std::function<void(int)> fc = [&](int j) {
        const int e = j / I.DQ, q = j % I.DQ;
        kc::exl3_cpu_rows_acc(ex[(size_t) e].p[2], xhd(e), q * 8, (q + 1) * 8, accd(e, q));
    };
    I.parallel(ne * I.DQ, fc);
    // down's bands
    const std::function<void(int)> fd = [&](int j) {
        const int e = j / I.DB, b = j % I.DB;
        float a[128];
        for (int c = 0; c < 128; ++c) a[c] = 0.0f;
        for (int q = 0; q < I.DQ; ++q) {
            const float* aq = accd(e, q) + b * 128;
            for (int c = 0; c < 128; ++c) a[c] += aq[c];
        }
        double xs = 0.0;
        for (int k = 0; k < I.FB; ++k) xs += xsd(e)[k];
        kc::exl3_cpu_band_out(ex[(size_t) e].p[2], a, (float) xs, b * 128, yd(e) + b * 128);
    };
    I.parallel(ne * I.DB, fd);
    for (int d = 0; d < dim; ++d) {
        float s = 0.0f;
        for (int e = 0; e < ne; ++e) s += ex[(size_t) e].w * yd(e)[d];
        out[d] = s;
    }
}

void CpuLane::run_multi(const std::vector<CpuExpert>& ex, const std::vector<std::vector<std::pair<int, float>>>& users,
                        const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order) {
    Impl& I = *p;
    const int dim = I.dim, ff = I.ff;
    std::fill(out, out + (size_t) T * dim, 0.0f);
    // the entries (expert, token, weight), and the chunks of at most 8 of one expert's
    struct Ent {
        int e, t;
        float w;
    };
    struct Chunk {
        int e, j0, n;
    };
    std::vector<Ent> ent;
    std::vector<Chunk> ch;
    std::vector<std::vector<int>> ent_of(ex.size());   // expert, user -> its entry
    for (size_t e = 0; e < ex.size(); ++e)
        for (size_t a = 0; a < users[e].size(); a += kc::kExl3CpuMaxN) {
            const int n = (int) std::min<size_t>(kc::kExl3CpuMaxN, users[e].size() - a);
            ch.push_back({(int) e, (int) ent.size(), n});
            for (int i = 0; i < n; ++i) {
                ent_of[e].push_back((int) ent.size());
                ent.push_back({(int) e, users[e][a + (size_t) i].first, users[e][a + (size_t) i].second});
            }
        }
    const int N = (int) ent.size();
    if (N == 0) return;
    if (N > kMaxEntries) {   // (not for a verify window: 16 tokens x 4 experts)
        std::fprintf(stderr, "mm3 cpu lane: %d entries, more than %d - the rest skipped\n", N, kMaxEntries);
        return;
    }
    if (I.macc.empty()) {
        const size_t E = kMaxEntries;
        I.mxh.resize(E * 2 * dim);
        I.mxsb.resize(E * 2 * I.EB);
        I.macc.resize(E * 2 * I.GQ * ff);
        I.mxhd.resize(E * ff);
        I.mxsd.resize(E * I.FB);
        I.maccd.resize(E * I.DQ * dim);
        I.myd.resize(E * dim);
    }
    auto xh = [&](int j, int r) { return I.mxh.data() + ((size_t) j * 2 + r) * dim; };
    auto xsb = [&](int j, int r) { return I.mxsb.data() + ((size_t) j * 2 + r) * I.EB; };
    auto acc = [&](int j, int r, int q) { return I.macc.data() + (((size_t) j * 2 + r) * I.GQ + q) * ff; };
    auto xhd = [&](int j) { return I.mxhd.data() + (size_t) j * ff; };
    auto xsd = [&](int j) { return I.mxsd.data() + (size_t) j * I.FB; };
    auto accd = [&](int j, int q) { return I.maccd.data() + ((size_t) j * I.DQ + q) * dim; };
    auto yd = [&](int j) { return I.myd.data() + (size_t) j * dim; };
    const int C = (int) ch.size();
    const std::function<void(int)> fa = [&](int jb) {
        const Chunk& c = ch[(size_t) (jb / (2 * I.GQ))];
        const int r = (jb / I.GQ) % 2, q = jb % I.GQ;
        const kc::Exl3Part& m = ex[(size_t) c.e].p[r];
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < c.n; ++i) {
            const int j = c.j0 + i;
            const float* xt = x + (size_t) ent[(size_t) j].t * dim;
            for (int b = 2 * q; b < 2 * q + 2; ++b)
                xsb(j, r)[b] = kc::exl3_cpu_prepare_block(m.suh + b * 128, xt + b * 128, xh(j, r) + b * 128);
            xp[i] = xh(j, r);
            ap[i] = acc(j, r, q);
        }
        if (c.n == 1) kc::exl3_cpu_rows_acc(m, xp[0], q * 16, (q + 1) * 16, ap[0]);
        else kc::exl3_cpu_rows_acc_n(m, xp, c.n, q * 16, (q + 1) * 16, ap);
    };
    I.parallel(C * 2 * I.GQ, fa);
    const std::function<void(int)> fb = [&](int jb) {
        const Chunk& c = ch[(size_t) (jb / I.GB)];
        const int b = jb % I.GB;
        const kc::Exl3Part& d = ex[(size_t) c.e].p[2];
        for (int i = 0; i < c.n; ++i) {
            const int j = c.j0 + i;
            float y[2][128], a[128], hb[128];
            for (int r = 0; r < 2; ++r) {
                for (int cc = 0; cc < 128; ++cc) a[cc] = 0.0f;
                for (int q = 0; q < I.GQ; ++q) {
                    const float* aq = acc(j, r, q) + b * 128;
                    for (int cc = 0; cc < 128; ++cc) a[cc] += aq[cc];
                }
                double xs = 0.0;
                for (int k = 0; k < I.EB; ++k) xs += xsb(j, r)[k];
                kc::exl3_cpu_band_out(ex[(size_t) c.e].p[r], a, (float) xs, b * 128, y[r]);
            }
            for (int cc = 0; cc < 128; ++cc) {
                const float g = std::min(y[0][cc], I.limit);
                const float u = std::min(std::max(y[1][cc], -I.limit), I.limit);
                hb[cc] = g / (1.0f + std::exp(-I.alpha * g)) * (u + 1.0f);
            }
            xsd(j)[b] = kc::exl3_cpu_prepare_block(d.suh + b * 128, hb, xhd(j) + b * 128);
        }
    };
    I.parallel(C * I.GB, fb);
    const std::function<void(int)> fc = [&](int jb) {
        const Chunk& c = ch[(size_t) (jb / I.DQ)];
        const int q = jb % I.DQ;
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < c.n; ++i) {
            xp[i] = xhd(c.j0 + i);
            ap[i] = accd(c.j0 + i, q);
        }
        if (c.n == 1) kc::exl3_cpu_rows_acc(ex[(size_t) c.e].p[2], xp[0], q * 8, (q + 1) * 8, ap[0]);
        else kc::exl3_cpu_rows_acc_n(ex[(size_t) c.e].p[2], xp, c.n, q * 8, (q + 1) * 8, ap);
    };
    I.parallel(C * I.DQ, fc);
    const std::function<void(int)> fd = [&](int jb) {
        const Chunk& c = ch[(size_t) (jb / I.DB)];
        const int b = jb % I.DB;
        for (int i = 0; i < c.n; ++i) {
            const int j = c.j0 + i;
            float a[128];
            for (int cc = 0; cc < 128; ++cc) a[cc] = 0.0f;
            for (int q = 0; q < I.DQ; ++q) {
                const float* aq = accd(j, q) + b * 128;
                for (int cc = 0; cc < 128; ++cc) a[cc] += aq[cc];
            }
            double xs = 0.0;
            for (int k = 0; k < I.FB; ++k) xs += xsd(j)[k];
            kc::exl3_cpu_band_out(ex[(size_t) c.e].p[2], a, (float) xs, b * 128, yd(j) + b * 128);
        }
    };
    I.parallel(C * I.DB, fd);
    const auto acc_entry = [&](int j) {
        float* o = out + (size_t) ent[(size_t) j].t * dim;
        const float w = ent[(size_t) j].w;
        const float* y = yd(j);
        for (int d = 0; d < dim; ++d) o[d] += w * y[d];
    };
    if (order != nullptr) {
        for (const auto& [e, u] : *order) acc_entry(ent_of[(size_t) e][(size_t) u]);
    } else {
        for (int j = 0; j < N; ++j) acc_entry(j);
    }
}

}  // namespace mm3
