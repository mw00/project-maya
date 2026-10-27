// src/mm3/mm3_cpu.cpp - maya-mm3's CPU lane (mm3_cpu.hpp).
//
// One call: the experts' gate and up row sums in jobs of 16 slices (256 inputs, each job rotating its own two blocks
// of x), then per 128-output band the outputs, swigluoai and the down matrix's rotated input block, then down's row
// sums in jobs of 8 slices, then down's bands - on a pool of spinning workers (a decode step is a few milliseconds),
// the caller one of them.  run_multi's first three steps are one dispatch whose jobs start as their inputs are done (a
// chunk's bands once its gate / up jobs are, down's jobs of slice q once band q is), the bands of the last a barrier
// later: two barriers a call instead of four, the same arithmetic in the same order.
#include "mm3_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <thread>

#include <immintrin.h>

#include "mm3_os.hpp"
#include "mm3_env.hpp"

#include "strata/kernels/cpu/pool.hpp"

namespace kc = strata::kernels::cpu;

namespace mm3 {

struct CpuLane::Impl {
    int dim, ff, ffmax;   // (ff: the current width, set_width's; the buffers are ffmax's)
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
    int max_entries = 64;
    // (MM3_LANE_PROF: run_multi's time by phase, printed at the end)
    const bool prof = ::mm3::env("MM3_LANE_PROF") != nullptr;
    const bool old_phases = ::mm3::env("MM3_LANE_OLD") != nullptr;   // (the four phases with a barrier each)
    double ph[6] = {0, 0, 0, 0, 0, 0};
    long calls = 0, subs = 0;
    int GQ, GB, DQ, DB, EB, FB;   // gate / up K jobs and bands, down K jobs and bands, x blocks, h blocks
    std::vector<float> xh, xsb, acc, xhd, xsd, accd, yd;
    std::vector<float> mxh, mxsb, macc, mxhd, mxsd, maccd, myd;   // (run_multi's, by entry: allocated on first use)

    std::vector<int> cores;   // (the physical cores but the first, one logical CPU each: the workers' places)
    void worker(int i) {
        // (pinned to the physical cores, after the caller's; more workers than cores share them in turn)
        if (!cores.empty()) os::pin_self(cores[(size_t) i % cores.size()]);
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

CpuLane::CpuLane(int threads, int dim, int ff, float alpha, float limit, int max_entries) : p(new Impl) {
    p->max_entries = max_entries;
    p->dim = dim;
    p->ff = ff;
    p->ffmax = ff;
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
    p->cores = kc::physical_cores(true);
    for (int i = 0; i < threads; ++i) p->th.emplace_back([this, i] { p->worker(i); });
}

int CpuLane::auto_workers() { return (int) kc::physical_cores(true).size(); }

CpuLane::~CpuLane() {
    if (p->prof && p->calls)
        std::fprintf(stderr, "cpu lane run_multi: %ld calls, %.1f experts a call; us a call: setup %.1f gate/up %.1f "
                     "bands %.1f down %.1f down bands %.1f sums %.1f\n", p->calls, (double) p->subs / p->calls,
                     1e6 * p->ph[0] / p->calls, 1e6 * p->ph[1] / p->calls, 1e6 * p->ph[2] / p->calls,
                     1e6 * p->ph[3] / p->calls, 1e6 * p->ph[4] / p->calls, 1e6 * p->ph[5] / p->calls);
    p->stop = true;
    p->cv.notify_all();
    for (auto& t : p->th) t.join();
    delete p;
}

int CpuLane::threads() const { return (int) p->th.size() + 1; }
int CpuLane::max_entries() const { return p->max_entries; }

void CpuLane::parallel_for(int n, const std::function<void(int)>& f) { p->parallel(n, f); }

bool CpuLane::set_width(int ff) {
    if (ff <= 0 || ff > p->ffmax || ff % 128) return false;
    p->ff = ff;
    p->GB = p->DQ = p->FB = ff / 128;
    return true;
}

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
                        const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order,
                        bool accumulate) {
    Impl& I = *p;
    const int dim = I.dim, ff = I.ff;
    if (!accumulate) std::fill(out, out + (size_t) T * dim, 0.0f);
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
    if (N > I.max_entries) {   // (not for a verify window: 16 tokens x 4 experts)
        std::fprintf(stderr, "mm3 cpu lane: %d entries, more than %d - the rest skipped\n", N, I.max_entries);
        return;
    }
    if (I.macc.empty()) {
        const size_t E = (size_t) I.max_entries;
        I.mxh.resize(E * 2 * dim);
        I.mxsb.resize(E * 2 * I.EB);
        I.macc.resize(E * 2 * I.GQ * I.ffmax);
        I.mxhd.resize(E * I.ffmax);
        I.mxsd.resize(E * (I.ffmax / 128));
        I.maccd.resize(E * (I.ffmax / 128) * dim);
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
    using clk = std::chrono::steady_clock;
    auto tq = clk::now();
    auto lap = [&](int i) {
        if (!I.prof) return;
        const auto t2 = clk::now();
        I.ph[i] += std::chrono::duration<double>(t2 - tq).count();
        tq = t2;
    };
    if (I.prof) {
        ++I.calls;
        I.subs += (long) ex.size();
    }
    lap(0);
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
    // (down's jobs also by a quarter of its outputs: a slice's job alone left half the threads idle for one expert)
    constexpr int kDP = 4;
    const int dcols = dim / kDP;
    const std::function<void(int)> fc = [&](int jb) {
        const Chunk& c = ch[(size_t) (jb / (I.DQ * kDP))];
        const int q = (jb / kDP) % I.DQ, p = jb % kDP;
        const float* xp[kc::kExl3CpuMaxN];
        float* ap[kc::kExl3CpuMaxN];
        for (int i = 0; i < c.n; ++i) {
            xp[i] = xhd(c.j0 + i);
            ap[i] = accd(c.j0 + i, q);
        }
        kc::exl3_cpu_rows_acc_cols_n(ex[(size_t) c.e].p[2], xp, c.n, q * 8, (q + 1) * 8, p * dcols, (p + 1) * dcols, ap);
    };
    const int na = C * 2 * I.GQ, nb = C * I.GB, ncj = C * I.DQ * kDP;
    if (I.old_phases) {
        I.parallel(na, fa);
        lap(1);
        I.parallel(nb, fb);
        lap(2);
        I.parallel(ncj, fc);
        lap(3);
    } else {
        // one dispatch, a loop a thread: the next gate / up job (fa) while there is one, else the band jobs (fb, chunk
        // by chunk) once their chunk's gate / up jobs are all done, down's jobs (fc, chunk by chunk, slice by slice)
        // once their band is - each list taken in order, a job's inputs published by its writers' release counts
        std::vector<std::atomic<int>> a_done((size_t) C), b_done((size_t) C * I.GB);
        for (auto& x : a_done) x.store(0, std::memory_order_relaxed);
        for (auto& x : b_done) x.store(0, std::memory_order_relaxed);
        std::atomic<int> a_next{0}, b_next{0}, c_next{0}, c_left{ncj};
        const int per_a = 2 * I.GQ, GB = I.GB, DQ = I.DQ;
        const std::function<void(int)> loop = [&](int) {
            for (int idle = 0;;) {
                if (c_left.load(std::memory_order_acquire) == 0) return;
                if (a_next.load(std::memory_order_relaxed) < na) {   // (gate / up first: every later job waits on it)
                    const int ja = a_next.fetch_add(1, std::memory_order_relaxed);
                    if (ja < na) {
                        fa(ja);
                        a_done[(size_t) (ja / per_a)].fetch_add(1, std::memory_order_release);
                        idle = 0;
                        continue;
                    }
                }
                int jb = b_next.load(std::memory_order_relaxed);
                if (jb < nb && a_done[(size_t) (jb / GB)].load(std::memory_order_acquire) == per_a &&
                    b_next.compare_exchange_weak(jb, jb + 1)) {
                    fb(jb);
                    b_done[(size_t) jb].store(1, std::memory_order_release);
                    idle = 0;
                    continue;
                }
                int jc = c_next.load(std::memory_order_relaxed);
                // (job jc: chunk jc / (DQ kDP), slice (jc / kDP) % DQ - its band (the down input's block) the same)
                if (jc < ncj && b_done[(size_t) ((jc / (DQ * kDP)) * GB + (jc / kDP) % DQ)].load(std::memory_order_acquire) &&
                    c_next.compare_exchange_weak(jc, jc + 1)) {
                    fc(jc);
                    c_left.fetch_sub(1, std::memory_order_acq_rel);
                    idle = 0;
                    continue;
                }
                for (int i = 0; i < (1 << std::min(idle++, 6)); ++i) _mm_pause();   // (nothing ready: back off)
            }
        };
        I.parallel((int) I.th.size() + 1, loop);
        lap(1);
    }
    // down's bands and the tokens' sums, a job a band: its outputs of every entry, then added into the tokens' rows
    // in order's sequence (each value's sum in the same order as entry by entry over the whole rows)
    std::vector<int> seq;
    if (order != nullptr) {
        for (const auto& [e, u] : *order) seq.push_back(ent_of[(size_t) e][(size_t) u]);
    } else {
        for (int j = 0; j < N; ++j) seq.push_back(j);
    }
    const std::function<void(int)> fd = [&](int b) {
        for (const Chunk& c : ch)
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
        for (int j : seq) {
            float* o = out + (size_t) ent[(size_t) j].t * dim + b * 128;
            const float w = ent[(size_t) j].w;
            const float* y = yd(j) + b * 128;
            for (int d = 0; d < 128; ++d) o[d] += w * y[d];
        }
    };
    I.parallel(I.DB, fd);
    lap(4);
    lap(5);
}

}  // namespace mm3
