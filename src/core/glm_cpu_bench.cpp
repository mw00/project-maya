// src/core/glm_cpu_bench.cpp - how fast does this CPU run one GLM routed expert (ggml-cpu's dots)?
//
//   glm_cpu_bench <pack_dir> <layer> [threads...] [--row4 0|1|both] [--experts n]
//
// glm-dsa B0.11: the expert geometry comes from the pack - n_embd, the routed n_ff and the swiglu clamp from the
// metadata of the GGUF shard native_experts.txt names (glm-dsa / GLM-5.3: 6144 x 2048, no clamp; glm5-next / Flash:
// 4096 x 2048, clamp 10), checked against the layer's blob size there.  Reads the first 64 experts of the layer
// straight from its shard(s) (native_experts.txt gives the offsets) - ~450 MB on Flash, ~620 MB on GLM-5.3's IQ1_S
// layers, far past the L3, so every expert comes from DRAM like a real RAM-tier miss - then, for each thread count
// given (default 1 6 12), on a persistent spinning pool (no thread start per expert):
//  - lane: the engine's CPU lane call (Glm5Model::lane_experts) timed the way its setup calibrates it
//    (fast_cpu_lane_setup): --experts experts a call (default 4, the calibration's), their gate/up rows end to end cut
//    in threads x max(1, min(8, 48 / threads)) contiguous pieces (MAYA_GLM_CPU_SPLIT=<pieces a thread> overrides, as
//    in the engine), each h quantized on the caller, the down rows likewise, the sum; 100 ms of calls for the clocks,
//    then 5 rounds of 16 calls 100 ms apart, and an expert's time is the fastest round's median over the call's
//    experts.  At the engine's thread count on its calibration layer this is the engine log's "CPU lane: N threads
//    ..., an expert X ms on the CPU" (GLM-5.3 UD-IQ1_S on the test PC: 36 threads, an IQ1_S gate/up layer, 0.350 ms).
//    Each count's output is checked bit for bit against one single-thread run with row4 off (exit 1 on a mismatch).
//  - alone: one expert a call, gate/up (2 x n_ff rows of n_embd) and down (n_embd rows of n_ff) timed apart, the rows
//    split evenly over the threads (mean over 3 laps of the 64, and the best).
// MAYA_GLM_LANE_ROW4=1 turns the lane's 4-row GEMV on as in the engine (a type without a 4-row kernel keeps ggml's
// dots; the header line says which); --row4 0 / 1 overrides it, --row4 both runs every count with it off, then on.
// GLM_CPU_BENCH_TOKENS=k (1..16): each row dotted with k activations (a verify window's tokens sharing the expert):
// the same DRAM reads, k times the dots (row4 runs only at k = 1).  The engine interleaves its RAM tier over the NUMA
// nodes and leaves its pool unpinned, so run it the same way, e.g. on the test PC with the first IQ1_S gate/up layer:
//   L=$(awk '!/^#/ && $2 == 19 {print $1; exit}' $P/native_experts.txt)
//   numactl --interleave=all build/glm_cpu_bench $P $L 18 27 36 48 60 72 --row4 both
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/glm_lane_row4.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// (the engine's pool spins with pause too: a spinning thread then leaves its SMT sibling the core's ports)
inline void cpu_relax() {
#if defined(__x86_64__)
    asm volatile("pause" ::: "memory");
#endif
}

// nt - 1 spinning helpers + the caller; run(fn) calls fn(t, nt) once on every thread and returns when all are done
class Spin {
public:
    explicit Spin(int nt) : nt_(nt) {
        for (int t = 1; t < nt; ++t)
            th_.emplace_back([this, t] {
                uint64_t seen = 0;
                for (;;) {
                    uint64_t g;
                    while ((g = gen_.load(std::memory_order_acquire)) == seen) cpu_relax();
                    if (quit_.load(std::memory_order_relaxed)) return;
                    seen = g;
                    (*fn_)(t, nt_);
                    done_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    ~Spin() {
        quit_.store(true);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : th_) t.join();
    }
    void run(const std::function<void(int, int)>& fn) {
        fn_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        fn(0, nt_);
        while (done_.load(std::memory_order_acquire) < nt_ - 1) cpu_relax();
    }

private:
    int nt_;
    std::vector<std::thread> th_;
    const std::function<void(int, int)>* fn_ = nullptr;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};
};

// the pack's expert geometry, from shard 1's metadata (the file glm_pack_geometry reads)
struct Geo {
    std::string arch;
    int64_t n_embd = 0, n_ff = 0;
    float limit = std::numeric_limits<float>::infinity();   // the lane's swiglu clamp (glm-dsa: none)
};

bool pack_geo(const std::string& gguf, Geo& g, std::string& err) {
    try {
        const strata::GgufFile f(gguf);
        const strata::MetaValue* a = f.get("general.architecture");
        if (a == nullptr || a->type != strata::MetaType::STRING) {
            err = gguf + " has no general.architecture";
            return false;
        }
        // a key under the file's own arch prefix (glm-dsa., glm5-next., glm5next.); a per-layer array gives its first
        const auto num = [&](const char* key, double def) -> double {
            const strata::MetaValue* v = f.get(a->s + "." + key);
            if (v != nullptr && v->type == strata::MetaType::ARRAY) v = v->items.empty() ? nullptr : &v->items[0];
            return v != nullptr && v->is_num() ? v->num() : def;
        };
        g.arch = a->s;
        g.n_embd = (int64_t) num("embedding_length", 0.0);
        g.n_ff = (int64_t) num("expert_feed_forward_length", 0.0);
        if (a->s == "glm5-next" || a->s == "glm5next") {
            g.limit = (float) num("swiglu_clamp_exp", 10.0);   // Maya's Flash default when the file has none
        } else if (a->s != "glm-dsa") {
            err = gguf + ": architecture " + a->s + " is not a GLM one (glm-dsa, glm5-next)";
            return false;
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (g.n_embd <= 0 || g.n_ff <= 0) {
        err = gguf + ": no " + g.arch + ".embedding_length / " + g.arch + ".expert_feed_forward_length";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    namespace kc = strata::kernels::cpu;
    std::vector<std::string> pos;
    std::string row4 = "env";
    int n_call = 4;
    bool bad = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--row4" && i + 1 < argc) row4 = argv[++i];
        else if (a == "--experts" && i + 1 < argc) n_call = std::atoi(argv[++i]);
        else if (a.rfind("--", 0) == 0) bad = true;
        else pos.push_back(a);
    }
    std::vector<int> counts;
    for (size_t i = 2; i < pos.size(); ++i) {
        counts.push_back(std::atoi(pos[i].c_str()));
        if (counts.back() < 1) bad = true;
    }
    if (bad || pos.size() < 2 || (row4 != "env" && row4 != "0" && row4 != "1" && row4 != "both") || n_call < 1 ||
        n_call > 8) {
        std::fprintf(stderr, "usage: %s <pack_dir> <layer> [threads...] [--row4 0|1|both] [--experts 1..8]\n"
                             "  (env: MAYA_GLM_LANE_ROW4, MAYA_GLM_CPU_SPLIT, GLM_CPU_BENCH_TOKENS)\n", argv[0]);
        return 2;
    }
    if (counts.empty()) counts = {1, 6, 12};
    const std::string pack = pos[0];
    const int layer = std::atoi(pos[1].c_str());
    std::ifstream ne(pack + "/native_experts.txt");
    std::string line, shard0, shard[3];
    int gu = -1, dt = -1, n_expert = 64;
    uint64_t go = 0, uo = 0, dof = 0, blob = 0;
    while (std::getline(ne, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const std::string key = "absolute offsets in ";
            const size_t a = line.find(key);
            if (a != std::string::npos) {
                const size_t b = line.find(',', a);
                shard0 = line.substr(a + key.size(), b == std::string::npos ? std::string::npos : b - a - key.size());
            }
            const size_t at = line.find("(n_expert ");   // v3+ headers (64 assumed without)
            if (at != std::string::npos) n_expert = std::atoi(line.c_str() + at + 10);
            continue;
        }
        std::istringstream ss(line);
        int l = -1;
        uint64_t off = 0;
        ss >> l >> gu >> dt >> off >> blob >> go >> uo >> dof;
        std::string w[3];
        ss >> w[0] >> w[1] >> w[2];   // v3: the layer's shard (none: shard 1); v4 straddle row: gate, up, down shards
        if (l == layer) {
            for (int r = 0; r < 3; ++r) shard[r] = !w[2].empty() ? w[r] : !w[0].empty() ? w[0] : shard0;
            break;
        }
    }
    if (shard[0].empty()) {
        std::fprintf(stderr, "layer %d not found in %s/native_experts.txt\n", layer, pack.c_str());
        return 1;
    }
    Geo geo;
    std::string err;
    if (!pack_geo(pack + "/../" + shard0, geo, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const int n_embd = (int) geo.n_embd, n_ff = (int) geo.n_ff, n_exp = std::max(1, std::min(64, n_expert));
    kc::NativeFmt f;
    if (!kc::native_fmt(gu, dt, n_embd, n_ff, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (f.bytes != blob) {
        std::fprintf(stderr, "layer %d: native_experts.txt has %llu B an expert, %s's %d x %d makes %zu B\n", layer,
                     (unsigned long long) blob, geo.arch.c_str(), n_embd, n_ff, f.bytes);
        return 1;
    }
    // expert e's gate rows sit at go + e * gub (the GGUF's [expert][row] order), up and down likewise
    const size_t gub = f.gu_row * (size_t) n_ff, dnb = f.d_row * (size_t) n_embd, eb = 2 * gub + dnb;
    const uint64_t base[3] = {go, uo, dof};
    const size_t rb[3] = {gub, gub, dnb}, at[3] = {0, gub, 2 * gub};
    std::vector<uint8_t> blobs(eb * n_exp);
    for (int r = 0; r < 3; ++r) {
        const std::string path = pack + "/../" + shard[r];
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "open %s failed\n", path.c_str());
            return 1;
        }
        for (int e = 0; e < n_exp; ++e)
            if (pread(fd, blobs.data() + eb * e + at[r], rb[r], (off_t) (base[r] + e * rb[r])) != (ssize_t) rb[r]) {
                std::fprintf(stderr, "short read\n");
                return 1;
            }
        close(fd);
    }
    std::vector<const uint8_t*> ex(n_exp);
    for (int e = 0; e < n_exp; ++e) ex[(size_t) e] = blobs.data() + eb * e;
    const int K = std::max(1, std::min(16, getenv("GLM_CPU_BENCH_TOKENS") ? std::atoi(getenv("GLM_CPU_BENCH_TOKENS")) : 1));
    const int E = std::min(n_call, n_exp), n_sets = n_exp / E;
    static const int split = [] {   // the engine's MAYA_GLM_CPU_SPLIT (glm_fast_path.cu lane_experts)
        const char* v = getenv("MAYA_GLM_CPU_SPLIT");
        return v ? std::max(1, std::min(64, std::atoi(v))) : 0;
    }();
    std::vector<float> xs((size_t) K * n_embd);
    std::vector<uint8_t> act(K * kc::kNativeActBytes), hq(K * kc::kNativeHBytes);
    std::vector<float> ff((size_t) K * n_ff), dn((size_t) K * n_embd);
    std::vector<const void*> ap(K), hp(K);
    std::vector<float*> fp(K), dp(K);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n_embd; ++i) xs[(size_t) k * n_embd + i] = std::sin(0.37f * i + 0.11f * k) * 0.5f;
        kc::native_quant_act(f, xs.data() + (size_t) k * n_embd, act.data() + k * kc::kNativeActBytes);
        ap[k] = act.data() + k * kc::kNativeActBytes;
        hp[k] = hq.data() + k * kc::kNativeHBytes;
        fp[k] = ff.data() + (size_t) k * n_ff;
        dp[k] = dn.data() + (size_t) k * n_embd;
    }
    // the lane's rows: expert i, token k at [i * K + k]
    std::vector<uint8_t> lhq((size_t) E * K * kc::kNativeHBytes);
    std::vector<float> lff((size_t) E * K * n_ff), ldn((size_t) E * K * n_embd), lout((size_t) K * n_embd);
    std::vector<const void*> lh((size_t) E * K);
    std::vector<float*> lf((size_t) E * K), ld((size_t) E * K);
    for (int i = 0; i < E * K; ++i) {
        lh[(size_t) i] = lhq.data() + (size_t) i * kc::kNativeHBytes;
        lf[(size_t) i] = lff.data() + (size_t) i * n_ff;
        ld[(size_t) i] = ldn.data() + (size_t) i * n_embd;
    }
    // one lane call on experts e[0..E) (Glm5Model::lane_experts): x quantized, the E experts' gate/up rows end to end
    // in T contiguous pieces, each h quantized on the caller, the down rows likewise, the sum in i order (the route's
    // weights; 1 here, as in the calibration).  The engine's threads claim the pieces in turn; here thread t takes
    // t, t + n, ... - the same split when T = threads (27 threads and up)
    const auto lane = [&](Spin& pool, int T, const uint8_t* const* e) {
        for (int k = 0; k < K; ++k)
            kc::native_quant_act(f, xs.data() + (size_t) k * n_embd, act.data() + k * kc::kNativeActBytes);
        const int gu_tot = E * n_ff, dn_tot = E * n_embd;
        pool.run([&](int t, int n) {
            for (int j = t; j < T; j += n) {
                const int q1 = (int) ((int64_t) gu_tot * (j + 1) / T);
                for (int q = (int) ((int64_t) gu_tot * j / T); q < q1;) {
                    const int i = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
                    kc::native_gu_rows_split(f, e[i], e[i] + gub, ap.data(), K, lf.data() + (size_t) i * K, r0, r1,
                                             geo.limit);
                    q += r1 - r0;
                }
            }
        });
        for (int i = 0; i < E * K; ++i) kc::native_quant_h(f, lf[(size_t) i], (void*) lh[(size_t) i]);
        pool.run([&](int t, int n) {
            for (int j = t; j < T; j += n) {
                const int q1 = (int) ((int64_t) dn_tot * (j + 1) / T);
                for (int q = (int) ((int64_t) dn_tot * j / T); q < q1;) {
                    const int i = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
                    kc::native_down_rows_split(f, e[i] + 2 * gub, lh.data() + (size_t) i * K, K,
                                               ld.data() + (size_t) i * K, r0, r1);
                    q += r1 - r0;
                }
            }
        });
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < n_embd; ++r) {
                float m = 0.0f;
                for (int i = 0; i < E; ++i) m += ld[(size_t) i * K + k][r];
                lout[(size_t) k * n_embd + r] = m;
            }
    };
    // the check's reference: experts 0..E-1 on one thread with row4 off
    std::vector<float> ref;
    {
        Spin one(1);
        kc::lane_row4_override(0);
        lane(one, 1, ex.data());
        ref = lout;
        kc::lane_row4_override(-1);
    }
    std::printf("layer %d (%s, n_embd %d, n_ff %d, swiglu clamp %g) gate/up %s (%d) down %s (%d): expert %.2f MB "
                "(gate/up %.2f, down %.2f), %d experts cycled from DRAM, %d activation(s) a row\n",
                layer, geo.arch.c_str(), n_embd, n_ff, (double) geo.limit, strata::ggml_type_name((uint32_t) gu), gu,
                strata::ggml_type_name((uint32_t) dt), dt, (double) eb / 1e6, 2.0 * gub / 1e6, dnb / 1e6, n_exp, K);
    std::printf("row4: MAYA_GLM_LANE_ROW4 %s, --row4 %s; a 4-row kernel for gate/up %s, down %s%s\n",
                kc::lane_row4_on() ? "on" : "off", row4.c_str(), kc::row4_supported(gu) ? "yes" : "no (ggml's dots)",
                kc::row4_supported(dt) ? "yes" : "no (ggml's dots)", K > 1 ? " (unused: k > 1)" : "");
    std::vector<int> modes;   // lane_row4_override values: -1 follows the env
    if (row4 == "both") modes = {0, 1};
    else if (row4 == "env") modes = {-1};
    else modes = {std::atoi(row4.c_str())};
    std::vector<double> lane_ms(counts.size() * modes.size());
    std::vector<int> lane_on(modes.size());
    int rc = 0;
    for (size_t ci = 0; ci < counts.size(); ++ci) {
        const int nt = counts[ci];
        Spin pool(nt);
        const int T = nt * (split > 0 ? split : std::max(1, std::min(8, 48 / nt)));
        for (size_t mi = 0; mi < modes.size(); ++mi) {
            kc::lane_row4_override(modes[mi]);
            lane_on[mi] = kc::lane_row4_on() ? 1 : 0;
            // ---- the lane, as fast_cpu_lane_setup times it
            int s = 0;
            const auto busy = [&](int ms) {
                for (const auto tw = std::chrono::steady_clock::now();
                     std::chrono::steady_clock::now() - tw < std::chrono::milliseconds(ms);)
                    lane(pool, T, ex.data() + (size_t) (s++ % n_sets) * E);
            };
            busy(100);
            double best = 0.0, sum = 0.0;
            for (int round = 0; round < 5; ++round) {
                if (round > 0) busy(100);
                std::array<double, 16> runs{};
                for (int rep = 0; rep < 16; ++rep) {
                    const uint8_t* const* e = ex.data() + (size_t) (s++ % n_sets) * E;
                    const auto t0 = std::chrono::steady_clock::now();
                    lane(pool, T, e);
                    runs[(size_t) rep] =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    sum += runs[(size_t) rep];
                }
                std::sort(runs.begin(), runs.end());
                const double med = 0.5 * (runs[7] + runs[8]);
                best = round == 0 ? med : std::min(best, med);
            }
            const double c_ms = best / E;
            lane_ms[ci * modes.size() + mi] = c_ms;
            lane(pool, T, ex.data());   // experts 0..E-1 again, for the check
            const bool same = std::memcmp(lout.data(), ref.data(), ref.size() * sizeof(float)) == 0;
            if (!same) rc = 1;
            // ---- one expert alone, gate/up and down timed apart
            const int reps = 4 * n_exp;
            double t_gu = 0, t_dn = 0, best1 = 1e9;
            for (int rep = 0; rep < reps; ++rep) {
                const uint8_t* b = ex[(size_t) (rep % n_exp)];
                const auto t0 = std::chrono::steady_clock::now();
                pool.run([&](int t, int n) {
                    const int r0 = n_ff * t / n, r1 = n_ff * (t + 1) / n;
                    kc::native_gu_rows_split(f, b, b + gub, ap.data(), K, fp.data(), r0, r1, geo.limit);
                });
                const auto t1 = std::chrono::steady_clock::now();
                for (int k = 0; k < K; ++k) kc::native_quant_h(f, fp[k], (void*) hp[k]);
                pool.run([&](int t, int n) {
                    const int r0 = n_embd * t / n, r1 = n_embd * (t + 1) / n;
                    kc::native_down_rows_split(f, b + 2 * gub, hp.data(), K, dp.data(), r0, r1);
                });
                const auto t2 = std::chrono::steady_clock::now();
                if (rep < n_exp) continue;   // the first lap warms the pool and the clocks
                const double g = std::chrono::duration<double, std::micro>(t1 - t0).count();
                const double d = std::chrono::duration<double, std::micro>(t2 - t1).count();
                t_gu += g;
                t_dn += d;
                best1 = std::min(best1, g + d);
            }
            const double n = reps - n_exp;
            std::printf("threads %2d, row4 %s: lane %.3f ms an expert (%.2f GB/s; mean %.3f ms; %d experts a call in "
                        "%d pieces), output %s\n",
                        nt, lane_on[mi] ? "on " : "off", c_ms, (double) eb / c_ms / 1e6, sum / 80.0 / E, E, T,
                        same ? "bit-identical" : "DIFFERS from the one-thread row4-off run");
            std::printf("                    alone: gate/up %7.1f us (%.2f GB/s)  down %7.1f us (%.2f GB/s)  expert "
                        "%7.1f us mean, %7.1f best\n",
                        t_gu / n, 2.0 * gub / (t_gu / n) / 1e3, t_dn / n, dnb / (t_dn / n) / 1e3, (t_gu + t_dn) / n,
                        best1);
        }
    }
    kc::lane_row4_override(-1);
    // the sweep: what the engine's "an expert X ms on the CPU" would be at each count
    std::printf("lane, ms an expert");
    for (size_t mi = 0; mi < modes.size(); ++mi) std::printf(" row4 %s", lane_on[mi] ? "on " : "off");
    std::printf("\n");
    for (size_t ci = 0; ci < counts.size(); ++ci) {
        std::printf("  threads %2d      ", counts[ci]);
        for (size_t mi = 0; mi < modes.size(); ++mi) std::printf("%9.3f", lane_ms[ci * modes.size() + mi]);
        if (modes.size() == 2) std::printf("   on / off %.2fx", lane_ms[ci * 2 + 1] / lane_ms[ci * 2]);
        std::printf("\n");
    }
    if (rc != 0) std::fprintf(stderr, "glm_cpu_bench: a lane output differed from the reference (see above)\n");
    return rc;
}
