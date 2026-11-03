// tests/ds4_iq1_bench.cpp - the CPU lane's IQ1_S rows (src/ds4/ds4_iq_cpu.cpp) timed as cpu_experts_iq runs them: a
// DeepSeek-V4.1-Flash expert (gate / up 2048 rows of 4096, down 4096 rows of 2048, all IQ1_S) in jobs of 64 rows, 1 to
// 4 tokens, on one thread and on a pool of N; every form's results checked against ggml-cpu's dot bit for bit first.
// No GPU, no model (random blocks: any bit pattern is a valid IQ1_S block).
//
//   ds4_iq1_bench [threads=12] [experts=8] [reps=40]     (experts: the set cycled through - 8 x 4.9 MB is past L3)
//   DS4_IQ1_BENCH_DUMP=file: the own rows' outputs of the check written there (to compare two builds byte for byte)
#if defined(DS4_IQ1_BENCH_SRC)   // (another version of the source, e.g. the commit before, to compare: -DDS4_IQ1_BENCH_SRC=...)
#include DS4_IQ1_BENCH_SRC
#else
#include "../src/ds4/ds4_iq_cpu.cpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace gg = strata::ds4::gguf;

namespace {

using Fn = std::function<void(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int)>;

std::vector<uint8_t> rand_rows(int n, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size(GGML_TYPE_IQ1_S, n), bb = ggml_type_size(GGML_TYPE_IQ1_S);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[o], &d, sizeof d);
    }
    return w;
}

void ggml_rows(const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) dot(n, &out[t][r - r0], 0, w + (size_t) r * rb, 0, act[t], 0, 1);
}

// a pool of n threads (the caller one of them) that spin between phases, as the engine's pool does on a decode step
struct Pool {
    int n;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0}, next{0};
    std::atomic<bool> quit{false};
    int jobs = 0;
    const std::function<void(int)>* f = nullptr;
    explicit Pool(int n_) : n(n_) {
        for (int k = 1; k < n; ++k)
            th.emplace_back([this] {
                int seen = 0;
                for (;;) {
                    int g;
                    while ((g = gen.load(std::memory_order_acquire)) == seen)
                        if (quit.load(std::memory_order_relaxed)) return;
                    seen = g;
                    work();
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    void work() {
        for (int j; (j = next.fetch_add(1, std::memory_order_relaxed)) < jobs;) (*f)(j);
    }
    void run(int nj, const std::function<void(int)>& fn) {
        jobs = nj;
        f = &fn;
        next.store(0);
        done.store(0);
        gen.fetch_add(1, std::memory_order_acq_rel);
        work();
        while (done.load(std::memory_order_acquire) < n - 1) {
        }
    }
    ~Pool() {
        quit = true;
        for (auto& t : th) t.join();
    }
};

struct Expert {
    std::vector<uint8_t> gate, up, down;
};

constexpr int E = 4096, FF = 2048, RB = 64;

// one expert's rows for nt tokens: gate / up jobs (64 rows of each), then the down's - the pool's two phases
void expert(const Fn& fn, const Expert& x, const void* const* xa, const void* const* ha, int nt, std::vector<float>& yg,
            std::vector<float>& yu, std::vector<float>& yd, Pool* pool) {
    const size_t gr = ggml_row_size(GGML_TYPE_IQ1_S, E), dr = ggml_row_size(GGML_TYPE_IQ1_S, FF);
    std::function<void(int)> fg = [&](int j) {
        float *go[8], *uo[8];
        for (int t = 0; t < nt; ++t) {
            go[t] = yg.data() + (size_t) t * FF + j * RB;
            uo[t] = yu.data() + (size_t) t * FF + j * RB;
        }
        fn(x.gate.data(), gr, E, xa, nt, go, j * RB, j * RB + RB);
        fn(x.up.data(), gr, E, xa, nt, uo, j * RB, j * RB + RB);
    };
    std::function<void(int)> fd = [&](int j) {
        float* o[8];
        for (int t = 0; t < nt; ++t) o[t] = yd.data() + (size_t) t * E + j * RB;
        fn(x.down.data(), dr, FF, ha, nt, o, j * RB, j * RB + RB);
    };
    if (pool == nullptr) {
        for (int j = 0; j < FF / RB; ++j) fg(j);
        for (int j = 0; j < E / RB; ++j) fd(j);
    } else {
        pool->run(FF / RB, fg);
        pool->run(E / RB, fd);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int nth = argc > 1 ? std::atoi(argv[1]) : 12, nex = argc > 2 ? std::atoi(argv[2]) : 8,
              reps = argc > 3 ? std::atoi(argv[3]) : 40;
    ggml_cpu_init();
#if defined(DS4_IQ1_BENCH_PDEP)
    gg::g_iq1_pdep = true;   // (a source from before the vector decode: its pdep form, the one Zen 3 took)
#endif
    for (int m : {3, 0, 1, 2}) {   // (own()'s choice of the float steps' form)
        gg::g_iq1_fma = m;
        if (gg::own_check(GGML_TYPE_IQ1_S)) break;
        gg::g_iq1_fma = -1;
    }
    std::printf("own_check(IQ1_S): %s (float form %d)\n", gg::g_iq1_fma >= 0 ? "OK" : "FAILED", gg::g_iq1_fma);
    if (gg::g_iq1_fma < 0) return 1;

    struct Form {
        const char* name;
        Fn fn;
    };
    auto own_rows_fn = [](const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
        gg::iq1s_rows_n(w, rb, n, act, nt, out, r0, r1);
    };
    std::vector<Form> forms = {{"ggml dot", ggml_rows}, {"own rows", own_rows_fn}};

    std::mt19937 rng(20261028u);
    std::vector<Expert> ex((size_t) nex);
    for (auto& x : ex) {
        x.gate = rand_rows(E, FF, rng);
        x.up = rand_rows(E, FF, rng);
        x.down = rand_rows(FF, E, rng);
    }
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t xb = ggml_row_size(GGML_TYPE_Q8_K, E), hb = ggml_row_size(GGML_TYPE_Q8_K, FF);
    constexpr int MT = 5;
    std::vector<uint8_t> xq(xb * MT), hq(hb * MT);
    std::vector<const void*> xa(MT), ha(MT);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> v((size_t) E);
        for (int t = 0; t < MT; ++t) {
            for (auto& f : v) f = nd(rng) * (t % 3 == 2 ? 30.0f : 1.0f);
            q8->from_float(v.data(), xq.data() + xb * t, E);
            for (int k = 0; k < FF; ++k) v[(size_t) k] = nd(rng) * 0.1f;
            q8->from_float(v.data(), hq.data() + hb * t, FF);
            xa[(size_t) t] = xq.data() + xb * t;
            ha[(size_t) t] = hq.data() + hb * t;
        }
    }
    std::vector<float> yg((size_t) MT * FF), yu((size_t) MT * FF), yd((size_t) MT * E);

    // bit-exactness: every form against ggml's dot, 1-5 tokens, every row of two experts
    FILE* dump = std::getenv("DS4_IQ1_BENCH_DUMP") ? std::fopen(std::getenv("DS4_IQ1_BENCH_DUMP"), "wb") : nullptr;
    bool all_ok = true;
    for (int nt = 1; nt <= MT; ++nt)
        for (int e = 0; e < std::min(nex, 2); ++e) {
            std::vector<float> rg, ru, rd;
            for (size_t k = 0; k < forms.size(); ++k) {
                std::fill(yg.begin(), yg.end(), -7.0f);
                std::fill(yu.begin(), yu.end(), -7.0f);
                std::fill(yd.begin(), yd.end(), -7.0f);
                expert(forms[k].fn, ex[(size_t) e], xa.data(), ha.data(), nt, yg, yu, yd, nullptr);
                if (k == 0) {
                    rg = yg, ru = yu, rd = yd;
                    continue;
                }
                const bool ok = std::memcmp(rg.data(), yg.data(), yg.size() * 4) == 0 &&
                                std::memcmp(ru.data(), yu.data(), yu.size() * 4) == 0 &&
                                std::memcmp(rd.data(), yd.data(), yd.size() * 4) == 0;
                if (!ok) std::printf("MISMATCH: %s nt=%d expert %d\n", forms[k].name, nt, e);
                all_ok = all_ok && ok;
                if (dump != nullptr && k == 1) {   // (the own rows' outputs)
                    std::fwrite(yg.data(), 4, yg.size(), dump);
                    std::fwrite(yu.data(), 4, yu.size(), dump);
                    std::fwrite(yd.data(), 4, yd.size(), dump);
                }
            }
        }
    if (dump != nullptr) std::fclose(dump);
    std::printf("bit-exact vs ggml-cpu's dot (1-5 tokens, %d experts, all rows): %s\n", std::min(nex, 2),
                all_ok ? "OK" : "FAILED");

    bool pass1 = false;
    const unsigned fmask = argc > 4 ? (unsigned) std::strtoul(argv[4], nullptr, 0) : ~0u;   // (forms to time)
    for (int threads : {1, nth}) {
        if (threads == nth && nth == 1 && pass1) break;
        pass1 = true;
        std::unique_ptr<Pool> pool(threads > 1 ? new Pool(threads) : nullptr);   // (no spinning threads on one)
        std::printf("\n%d thread%s, %d experts cycled, us per expert (median of 7 rounds of %d)\n", threads,
                    threads > 1 ? "s" : "", nex, reps);
        std::printf("%-16s", "form");
        for (int nt : {1, 2, 4}) std::printf("  nt=%d   ", nt);
        std::printf("\n");
        for (size_t fi = 0; fi < forms.size(); ++fi) {
            const Form& f = forms[fi];
            if (!((fmask >> fi) & 1)) continue;
            std::printf("%-16s", f.name);
            for (int nt : {1, 2, 4}) {
                const int r = threads == 1 ? std::max(4, reps / 4) : reps;
                std::vector<double> t;
                for (int round = 0; round < 7; ++round) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int k = 0; k < r; ++k)
                        expert(f.fn, ex[(size_t) (k % nex)], xa.data(), ha.data(), nt, yg, yu, yd,
                               pool.get());
                    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / r);
                }
                std::sort(t.begin(), t.end());
                std::printf("  %7.1f", t[3]);
                std::fflush(stdout);
            }
            std::printf("\n");
        }
    }
    return all_ok ? 0 : 1;
}
