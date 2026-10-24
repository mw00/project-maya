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
    std::atomic<uint64_t> gen{0};
    std::atomic<int> next{0}, done{0};
    int n_jobs = 0;
    const std::function<void(int)>* fn = nullptr;
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> sleepers{0};
    // scratch, for up to 8 experts
    static constexpr int kMaxE = 8;
    int GQ, GB, DQ, DB, EB, FB;   // gate / up K jobs and bands, down K jobs and bands, x blocks, h blocks
    std::vector<float> xh, xsb, acc, xhd, xsd, accd, yd;

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
            work();
        }
    }
    void work() {
        for (int j; (j = next.fetch_add(1)) < n_jobs;) {
            (*fn)(j);
            done.fetch_add(1, std::memory_order_release);
        }
    }
    void parallel(int n, const std::function<void(int)>& f) {
        fn = &f;
        n_jobs = n;
        done.store(0);
        next.store(0);
        gen.fetch_add(1, std::memory_order_release);
        if (sleepers.load()) cv.notify_all();
        work();
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

}  // namespace mm3
