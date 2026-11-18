// Decode-only, model-free expert benchmark. Run from the repository root:
// HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/lib build-multi/hip_expert_kernel_bench
// --parity-only also checks sparse plans, CPU contributions and shared experts.
// --sweep exposes rejected tuning variants; --shared times the fused Q6_K shared down.
#include "strata/kernels/glm_expert_bench.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace bench = gf::hip_expert_bench;
using V = bench::Variant;
namespace {
constexpr int NE = 8, EMB = 4096, FF = 2048, POOLS = 5, CALLS = 60, ROUNDS = 7;
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s (no processes will be stopped)\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { check(hipMalloc(&p, n), "hipMalloc"); }
    ~Buffer() { if (p) check(hipFree(p), "hipFree"); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void put(const void* src, size_t n) { check(hipMemcpy(p, src, n, hipMemcpyHostToDevice), "H2D"); }
    void zero(size_t n) { check(hipMemset(p, 0, n), "memset"); }
    std::vector<uint8_t> get(size_t n) const {
        std::vector<uint8_t> v(n);
        check(hipMemcpy(v.data(), p, n, hipMemcpyDeviceToHost), "D2H");
        return v;
    }
};
uint16_t half_bits(float f) {
    return std::bit_cast<uint16_t>(__float2half(f));
}
float half_float(uint16_t b) {
    return __half2float(std::bit_cast<__half>(b));
}
int block_bytes(int t) {
    switch (t) {
        case 14: return sizeof(block_q6_K);
        case 16: return sizeof(block_iq2_xxs);
        case 17: return sizeof(block_iq2_xs);
        case 18: return sizeof(block_iq3_xxs);
        case 21: return sizeof(block_iq3_s);
        case 22: return sizeof(block_iq2_s);
        default: std::abort();
    }
}
void weights(uint8_t* dst, size_t bytes, int t, std::mt19937& rng) {
    const int bs = block_bytes(t);
    for (size_t i = 0; i < bytes; i += bs) {
        for (int j = 0; j < bs; ++j) dst[i + j] = rng() & 255;
        // Finite, signed scales; all other block bits are valid grid/scale/sign indices.
        const float magnitude = (1 + (rng() % 1024)) / 65536.f;
        const uint16_t d = half_bits((rng() & 1) ? magnitude : -magnitude);
        std::memcpy(dst + i + (t == 14 ? bs - 2 : 0), &d, 2);
    }
}
std::vector<uint8_t> activation(int n, std::mt19937& rng, bool zeros = false) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> v(size_t(n / 32) * sizeof(block_q8_1));
    for (int b = 0; b < n / 32; ++b) {
        float x[32], amax = 0, sum = 0;
        for (auto& f : x) { f = zeros ? 0.f : nd(rng); amax = std::max(amax, std::fabs(f)); sum += f; }
        const float d = amax / 127.f;
        uint8_t* p = v.data() + size_t(b) * sizeof(block_q8_1);
        const uint16_t dh = half_bits(d), sh = half_bits(sum);
        std::memcpy(p, &dh, 2); std::memcpy(p + 2, &sh, 2);
        for (int i = 0; i < 32; ++i) p[4 + i] = uint8_t(int8_t(d == 0 ? 0 : std::round(x[i] / d)));
    }
    return v;
}
std::vector<float> dequant(const std::vector<uint8_t>& v) {
    std::vector<float> out(v.size() / sizeof(block_q8_1) * 32);
    for (size_t b = 0; b < out.size() / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, v.data() + b * sizeof(block_q8_1), 2);
        const float d = half_float(dh);
        for (int j = 0; j < 32; ++j) out[32 * b + j] = d * int8_t(v[b * sizeof(block_q8_1) + 4 + j]);
    }
    return out;
}
bool compare(const char* label, const std::vector<float>& a, const std::vector<float>& b, bool exact) {
    double max_abs = 0, num = 0, den = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite &= std::isfinite(a[i]) && std::isfinite(b[i]);
        const double diff = double(a[i]) - b[i];
        max_abs = std::max(max_abs, std::abs(diff)); num += diff * diff; den += double(a[i]) * a[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const bool ok = finite && (exact || (rel <= 1e-6 && max_abs <= 1e-4));
    std::printf("parity %-28s max_abs=%.3e rel_L2=%.3e bit_identical=%s %s\n",
                label, max_abs, rel, exact ? "yes" : "no", ok ? "PASS" : "FAIL");
    return ok;
}
const char* name(V v) {
    switch (v) {
        case V::baseline: return "baseline";
        case V::production: return "production";
        case V::signs: return "signs";
        case V::waves8: return "waves8";
        case V::rows2: return "rows2";
        case V::rows8: return "rows8";
        case V::global_grid: return "global_grid";
        case V::global_plain: return "global_plain";
        case V::global_waves8: return "global_waves8";
        case V::global_rows2: return "global_rows2";
        case V::global_rows8: return "global_rows8";
        case V::split_shared: return "split_shared";
        case V::direct_signs: return "direct_signs";
        case V::vector_load: return "vector_load";
        case V::occupancy8: return "occupancy8";
        case V::direct_lds: return "direct_lds";
        case V::direct_rows4: return "direct_rows4";
        case V::direct_lds_rows4: return "direct_lds_rows4";
    }
    std::abort();
}
struct Graph {
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    Graph(hipStream_t s, const std::function<void(int)>& launch) {
        check(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal), "capture begin");
        for (int i = 0; i < CALLS; ++i) launch(i % POOLS);
        check(hipStreamEndCapture(s, &graph), "capture end");
        check(hipGraphInstantiateWithFlags(&exec, graph, 0), "graph instantiate");
    }
    ~Graph() { check(hipGraphExecDestroy(exec), "graph exec destroy"); check(hipGraphDestroy(graph), "graph destroy"); }
};
// Alternate the two graphs each round so clock/server-load drift affects both.
void timing(const char* kernel, V variant, size_t bytes, hipStream_t s,
            const std::function<void(int)>& old_launch, const std::function<void(int)>& new_launch) {
    Graph old(s, old_launch), fresh(s, new_launch);
    hipEvent_t start, end;
    check(hipEventCreate(&start), "event create"); check(hipEventCreate(&end), "event create");
    for (int j = 0; j < 3; ++j) {
        check(hipGraphLaunch(old.exec, s), "warmup old"); check(hipGraphLaunch(fresh.exec, s), "warmup new");
    }
    std::vector<double> a, b;
    for (int r = 0; r < ROUNDS; ++r) for (int j = 0; j < 2; ++j) {
        const bool baseline = (r + j) % 2 == 0;
        check(hipEventRecord(start, s), "event start");
        check(hipGraphLaunch(baseline ? old.exec : fresh.exec, s), "timed graph");
        check(hipEventRecord(end, s), "event end"); check(hipEventSynchronize(end), "event wait");
        float ms;
        check(hipEventElapsedTime(&ms, start, end), "elapsed");
        (baseline ? a : b).push_back(ms * 1000. / CALLS);
    }
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    const double before = a[ROUNDS / 2], after = b[ROUNDS / 2];
    std::printf("bench %-11s %-12s old=%.3f us %.2f GB/s new=%.3f us %.2f GB/s speedup=%.3fx latency_reduction=%.1f%%\n",
                kernel, name(variant), before, bytes / before / 1000., after, bytes / after / 1000.,
                before / after, 100 * (1 - after / before));
    check(hipEventDestroy(start), "event destroy"); check(hipEventDestroy(end), "event destroy");
}
bool run(int gu_type, int dn_type, bool parity_only, bool shared_mode, bool sweep, hipStream_t s) {
    const int pools = parity_only ? 1 : POOLS;
    const size_t gu_bytes = size_t(2) * FF * (EMB / 256) * block_bytes(gu_type);
    const size_t dn_bytes = size_t(EMB) * (FF / 256) * block_bytes(dn_type);
    const size_t blob_bytes = gu_bytes + dn_bytes;
    const size_t hbytes = size_t(NE) * (FF / 32) * sizeof(block_q8_1);
    std::printf("fixture gu=%d down=%d experts=%d n_embd=%d n_ff=%d pools=%d weights=%.1f MiB shared=%s\n",
                gu_type, dn_type, NE, EMB, FF, pools, pools * NE * blob_bytes / 1048576., shared_mode ? "Q6_K 4096x2048" : "off");
    Buffer blobs(pools * NE * blob_bytes), ptrs(pools * NE * 8), w(NE * 4), flag(4), cpu(EMB * 4);
    Buffer xq(EMB / 32 * sizeof(block_q8_1)), hq(hbytes), old_hq(hbytes), out(EMB * 4), old_out(EMB * 4);
    Buffer shared(EMB * 4), sh_weights(size_t(EMB) * (FF / 256) * sizeof(block_q6_K));
    Buffer sh_hq(FF / 32 * sizeof(block_q8_1)), sh_result(EMB * 4), sh_old(EMB * 4);
    std::mt19937 rng(0x5eed + dn_type);
    std::vector<uint8_t> raw(blob_bytes);
    std::vector<unsigned long long> pointers(pools * NE);
    for (int p = 0; p < pools * NE; ++p) {
        weights(raw.data(), gu_bytes, gu_type, rng); weights(raw.data() + gu_bytes, dn_bytes, dn_type, rng);
        void* dst = blobs.as<uint8_t>() + p * blob_bytes;
        check(hipMemcpy(dst, raw.data(), raw.size(), hipMemcpyHostToDevice), "expert upload");
        pointers[p] = reinterpret_cast<unsigned long long>(dst);
    }
    ptrs.put(pointers.data(), pointers.size() * 8);
    float weights_host[NE];
    for (int i = 0; i < NE; ++i) weights_host[i] = float(i + 1) / 36.f;
    w.put(weights_host, sizeof(weights_host)); flag.zero(4);
    auto x = activation(EMB, rng); xq.put(x.data(), x.size());
    auto sh_x = activation(FF, rng); sh_hq.put(sh_x.data(), sh_x.size());
    std::vector<uint8_t> sw(size_t(EMB) * (FF / 256) * sizeof(block_q6_K));
    weights(sw.data(), sw.size(), 14, rng); sh_weights.put(sw.data(), sw.size());
    std::vector<float> extra(EMB);
    for (int i = 0; i < EMB; ++i) extra[i] = float((i % 31) - 15) / 32.f;
    cpu.put(extra.data(), EMB * 4); shared.put(extra.data(), EMB * 4);
    gf::MoeDev plans[POOLS];
    for (int p = 0; p < pools; ++p) {
        plans[p].plan_ptr = ptrs.as<unsigned long long>() + p * NE;
        plans[p].plan_w = w.as<float>(); plans[p].cpu_flag = flag.as<int>(); plans[p].cpu_part = cpu.as<float>();
    }
    const std::vector<V> gu_variants = sweep ? std::vector<V>{
        V::signs, V::waves8, V::global_grid, V::global_plain, V::global_waves8, V::split_shared,
        V::direct_signs, V::vector_load, V::occupancy8, V::direct_lds, V::production} : std::vector<V>{V::production};
    const std::vector<V> dn_variants = sweep ? std::vector<V>{
        V::signs, V::rows2, V::rows8, V::global_grid, V::global_plain, V::global_rows2, V::global_rows8,
        V::direct_signs, V::vector_load, V::direct_lds, V::direct_rows4, V::direct_lds_rows4,
        V::production} : std::vector<V>{V::production};
    bool ok = true;
    // Include k<8, absent experts, CPU/shared branches, zero activations and unclamped SwiGLU.
    const int cases = parity_only ? 6 : 1;
    for (int c = 0; c < cases; ++c) {
        const int k = c == 1 ? 1 : c == 2 ? 3 : NE;
        const bool use_shared = c == 3 || shared_mode;
        const int cpu_flag = c == 3;
        flag.put(&cpu_flag, 4);
        auto plan = pointers;
        if (c == 2 || c == 3) { plan[1] = 0; plan[6] = 0; }
        ptrs.put(plan.data(), plan.size() * 8);
        x = activation(EMB, rng, c == 4); xq.put(x.data(), x.size());
        const float limit = c == 5 ? 1e9f : 10.f;
        auto gu = [&](V v, int pool, void* dst, float* sh_dst) {
            bench::gate_up(v, gu_type, plans[pool], k, EMB, FF, limit, xq.p, dst,
                          use_shared ? sh_weights.p : nullptr, 14, sh_hq.p, FF, sh_dst, s);
        };
        auto dn = [&](V v, int pool, const void* h, float* dst) {
            bench::down(v, dn_type, plans[pool], k, EMB, FF, gu_bytes, h,
                        use_shared ? shared.as<float>() : nullptr, dst, s);
        };
        old_hq.zero(hbytes); gu(V::baseline, 0, old_hq.p, sh_old.as<float>());
        check(hipStreamSynchronize(s), "baseline gu sync");
        auto ref_h = old_hq.get(hbytes);
        auto ref_sh = use_shared ? sh_old.get(EMB * 4) : std::vector<uint8_t>{};
        for (V v : gu_variants) {
            hq.zero(hbytes); gu(v, 0, hq.p, sh_result.as<float>()); check(hipStreamSynchronize(s), "gu sync");
            auto got = hq.get(hbytes);
            const std::string label = "gu" + std::to_string(gu_type) + " " + name(v) + " case" + std::to_string(c);
            ok &= compare(label.c_str(), dequant(ref_h), dequant(got), ref_h == got);
            if (use_shared) {
                const auto got_sh = sh_result.get(EMB * 4);
                std::vector<float> a(EMB), b(EMB);
                std::memcpy(a.data(), ref_sh.data(), EMB * 4); std::memcpy(b.data(), got_sh.data(), EMB * 4);
                ok &= compare("shared down", a, b, ref_sh == got_sh);
            }
        }
        // Down uses the identical baseline hq so gate/up differences cannot mask errors.
        dn(V::baseline, 0, old_hq.p, old_out.as<float>()); check(hipStreamSynchronize(s), "baseline dn sync");
        auto ref_d = old_out.get(EMB * 4);
        for (V v : dn_variants) {
            dn(v, 0, old_hq.p, out.as<float>()); check(hipStreamSynchronize(s), "dn sync");
            auto got = out.get(EMB * 4);
            std::vector<float> a(EMB), b(EMB);
            std::memcpy(a.data(), ref_d.data(), EMB * 4); std::memcpy(b.data(), got.data(), EMB * 4);
            const std::string label = "down" + std::to_string(dn_type) + " " + name(v) + " case" + std::to_string(c);
            ok &= compare(label.c_str(), a, b, ref_d == got);
        }
        if (!parity_only) {
            // Independent random h activations in all pools; bytes read, not FLOPs, define GB/s.
            auto h = activation(NE * FF, rng); old_hq.put(h.data(), h.size());
            const std::string glabel = "gate_up<" + std::to_string(gu_type) + ">";
            const std::string dlabel = "down<" + std::to_string(dn_type) + ">";
            for (V v : gu_variants)
                timing(glabel.c_str(), v, NE * gu_bytes + (use_shared ? sw.size() : 0), s,
                       [&](int p) { gu(V::baseline, p, hq.p, sh_result.as<float>()); }, [&](int p) { gu(v, p, hq.p, sh_result.as<float>()); });
            for (V v : dn_variants)
                timing(dlabel.c_str(), v, NE * dn_bytes, s,
                       [&](int p) { dn(V::baseline, p, old_hq.p, out.as<float>()); },
                       [&](int p) { dn(v, p, old_hq.p, out.as<float>()); });
        }
    }
    return ok;
}

// --window: a verify window's layer through moe_experts_multi - every token its own plan - with the shared experts
// unpacked once (MoeDev::mt_dedup) against the launch without it and against each token's own moe_gate_up / moe_down,
// bit for bit; then the time of both for several window shapes (fixed cost, per distinct expert, per extra token).
bool run_window(int gu_type, int dn_type, bool parity_only, hipStream_t s) {
    constexpr int WT = 4, NP = 24;   // tokens a window, experts a pool
    const int pools = parity_only ? 1 : POOLS;
    const size_t gu_bytes = size_t(2) * FF * (EMB / 256) * block_bytes(gu_type);
    const size_t dn_bytes = size_t(EMB) * (FF / 256) * block_bytes(dn_type);
    const size_t blob_bytes = gu_bytes + dn_bytes;
    const size_t xqb = size_t(EMB / 32) * sizeof(block_q8_1), hqb = size_t(8) * (FF / 32) * sizeof(block_q8_1);
    const size_t shb = size_t(FF / 32) * sizeof(block_q8_1);
    std::printf("window fixture gu=%d down=%d pools=%d x %d experts\n", gu_type, dn_type, pools, NP);
    Buffer blobs(pools * NP * blob_bytes), ptrs(POOLS * WT * 8 * 8), w(WT * 8 * 4), flag(WT * 4), cpu(WT * EMB * 4);
    Buffer xq(WT * xqb), hq_a(WT * hqb), hq_b(WT * hqb), out_a(WT * EMB * 4), out_b(WT * EMB * 4);
    Buffer sh_weights(size_t(EMB) * (FF / 256) * sizeof(block_q6_K)), sh_hq(WT * shb), sh_a(WT * EMB * 4), sh_b(WT * EMB * 4);
    std::mt19937 rng(0xd00d + 31 * gu_type + dn_type);
    std::vector<uint8_t> raw(blob_bytes);
    std::vector<unsigned long long> pool_ptr(pools * NP);
    for (int p = 0; p < pools * NP; ++p) {
        weights(raw.data(), gu_bytes, gu_type, rng); weights(raw.data() + gu_bytes, dn_bytes, dn_type, rng);
        void* dst = blobs.as<uint8_t>() + p * blob_bytes;
        check(hipMemcpy(dst, raw.data(), raw.size(), hipMemcpyHostToDevice), "expert upload");
        pool_ptr[p] = reinterpret_cast<unsigned long long>(dst);
    }
    std::vector<uint8_t> sw(size_t(EMB) * (FF / 256) * sizeof(block_q6_K));
    weights(sw.data(), sw.size(), 14, rng); sh_weights.put(sw.data(), sw.size());
    std::vector<float> extra(WT * EMB);
    for (int i = 0; i < WT * EMB; ++i) extra[i] = float((i % 37) - 18) / 32.f;
    cpu.put(extra.data(), extra.size() * 4);
    auto fill_x = [&](bool zeros) {
        for (int t = 0; t < WT; ++t) {
            auto x = activation(EMB, rng, zeros && t == 1);
            check(hipMemcpy(xq.as<uint8_t>() + t * xqb, x.data(), x.size(), hipMemcpyHostToDevice), "x");
            auto h = activation(FF, rng);
            check(hipMemcpy(sh_hq.as<uint8_t>() + t * shb, h.data(), h.size(), hipMemcpyHostToDevice), "sh h");
        }
    };
    // a window's plans from pool p: token t takes ne[t] experts, `share` of them (where it can) from the tokens before
    auto make_plans = [&](int p, int T, int k, const int* ne, double share, std::vector<unsigned long long>& plan,
                          std::vector<float>& pw) {
        plan.assign(WT * 8, 0ull); pw.assign(WT * 8, 0.f);
        std::vector<int> used;
        for (int t = 0; t < T; ++t) {
            std::vector<int> mine;
            for (int i = 0; i < ne[t] && i < k; ++i) {
                int e = -1;
                const bool want = !used.empty() && (rng() % 1000) < share * 1000;
                for (int tries = 0; tries < 64 && e < 0; ++tries) {
                    const int c = want && tries < 32 ? used[rng() % used.size()] : int(rng() % NP);
                    if (std::find(mine.begin(), mine.end(), c) == mine.end()) e = c;
                }
                for (int c = 0; c < NP && e < 0; ++c)
                    if (std::find(mine.begin(), mine.end(), c) == mine.end()) e = c;
                mine.push_back(e);
                plan[8 * t + i] = pool_ptr[p * NP + e];
                pw[8 * t + i] = float(1 + (rng() % 64)) / 97.f;
            }
            used.insert(used.end(), mine.begin(), mine.end());
        }
    };
    auto dev = [&](int p, int T, int dedup) {
        gf::MoeDev m{};
        m.plan_ptr = ptrs.as<unsigned long long>() + p * WT * 8; m.plan_w = w.as<float>();
        m.cpu_flag = flag.as<int>(); m.cpu_part = cpu.as<float>();
        m.mt_T = T; m.mt_xq_s = EMB / 32; m.mt_hq_s = 8 * (FF / 32); m.mt_sh_hq_s = FF / 32; m.mt_out_s = EMB; m.mt_cpu_s = EMB;
        m.mt_dedup = dedup;
        return m;
    };
    bool ok = true;
    int n_cases = 0, n_bad = 0;
    for (int c = 0; c < (parity_only ? 48 : 12); ++c) {
        const int T = 2 + c % 3, k = (c / 3) % 2 ? 8 : 5;
        const bool use_shared = c % 4 == 3, zeros = c % 5 == 4;
        const double share = c % 6 == 0 ? 0.0 : c % 6 == 1 ? 1.0 : 0.35;
        int ne[WT] = {k, k, k, k}, cf[WT] = {0, 0, 0, 0};
        if (c % 7 >= 4) for (int t = 0; t < T; ++t) ne[t] = int(rng() % (k + 1));   // narrower plans, also empty ones
        if (c % 8 == 5) cf[1] = 1;
        if (c % 16 == 13) cf[T - 1] = 1;
        flag.put(cf, sizeof(cf));
        std::vector<unsigned long long> plan; std::vector<float> pw;
        make_plans(0, T, k, ne, share, plan, pw);
        check(hipMemcpy(ptrs.p, plan.data(), plan.size() * 8, hipMemcpyHostToDevice), "plan");
        w.put(pw.data(), pw.size() * 4);
        fill_x(zeros);
        const float limit = c % 9 == 8 ? 1e9f : 10.f;
        if (!gf::win_dedup_shape(gu_type, dn_type, T, k)) { std::printf("window: shape not taken (T=%d k=%d)\n", T, k); return false; }
        auto multi = [&](int dedup, Buffer& hq, Buffer& o, Buffer& so) {
            hq.zero(WT * hqb); o.zero(WT * EMB * 4); so.zero(WT * EMB * 4);
            const bool launched = gf::moe_experts_multi(gu_type, dn_type, dev(0, T, dedup), k, EMB, FF, limit, gu_bytes, xq.p, hq.p,
                use_shared ? sh_weights.p : nullptr, 14, use_shared ? sh_hq.p : nullptr, FF,
                use_shared ? so.as<float>() : nullptr, o.as<float>(), s);
            check(hipStreamSynchronize(s), "window sync");
            return launched;
        };
        if (!multi(0, hq_a, out_a, sh_a) || !multi(1, hq_b, out_b, sh_b)) { std::printf("window: not launched\n"); return false; }
        const auto ha = hq_a.get(WT * hqb), hb = hq_b.get(WT * hqb), oa = out_a.get(WT * EMB * 4), ob = out_b.get(WT * EMB * 4);
        const auto sa = sh_a.get(WT * EMB * 4), sb = sh_b.get(WT * EMB * 4);
        // each token on its own launches (the token path), into the "a" buffers
        hq_a.zero(WT * hqb); out_a.zero(WT * EMB * 4);
        for (int t = 0; t < T; ++t) {
            gf::MoeDev m = dev(0, 1, 0);
            m.plan_ptr += 8 * t; m.plan_w += 8 * t; m.cpu_flag += t; m.cpu_part += size_t(t) * EMB;
            float* so = use_shared ? sh_a.as<float>() + size_t(t) * EMB : nullptr;
            bench::gate_up(V::production, gu_type, m, k, EMB, FF, limit, xq.as<uint8_t>() + t * xqb, hq_a.as<uint8_t>() + t * hqb,
                           use_shared ? sh_weights.p : nullptr, 14, use_shared ? sh_hq.as<uint8_t>() + t * shb : nullptr, FF, so, s);
            bench::down(V::production, dn_type, m, k, EMB, FF, gu_bytes, hq_a.as<uint8_t>() + t * hqb, so,
                        out_a.as<float>() + size_t(t) * EMB, s);
        }
        check(hipStreamSynchronize(s), "token sync");
        const auto h1 = hq_a.get(WT * hqb), o1 = out_a.get(WT * EMB * 4);
        // (h: only the slots a token's plan fills are written by any of the three ways)
        const bool same = ha == hb && oa == ob && sa == sb && h1 == hb && o1 == ob;
        ++n_cases; n_bad += !same;
        if (!same || c < 6)
            std::printf("window parity gu%d down%d case%d T=%d k=%d share=%.2f shared=%d: once vs multi h %s out %s, once vs token h %s out %s %s\n",
                        gu_type, dn_type, c, T, k, share, int(use_shared), ha == hb ? "same" : "DIFF", oa == ob ? "same" : "DIFF",
                        h1 == hb ? "same" : "DIFF", o1 == ob ? "same" : "DIFF", same ? "PASS" : "FAIL");
        ok &= same;
    }
    std::printf("window parity gu%d down%d: %d cases, %d differ: %s\n", gu_type, dn_type, n_cases, n_bad, n_bad ? "FAIL" : "PASS");
    if (parity_only) return ok;
    // ---- time: old = the launch without mt_dedup, new = with it
    int zf[WT] = {0, 0, 0, 0};
    flag.put(zf, sizeof(zf));
    fill_x(false);
    struct Shape { const char* name; int T, k; int ne[WT]; double share; };
    const Shape shapes[] = {
        {"T3 15 routed, none shared", 3, 5, {5, 5, 5, 0}, 0.0},
        {"T3 15 routed, some shared", 3, 5, {5, 5, 5, 0}, 0.31},
        {"T3 15 routed, 5 distinct", 3, 5, {5, 5, 5, 0}, 1.0},
        {"T3 5 routed (one token)", 3, 5, {5, 0, 0, 0}, 0.0},
        {"T3 10 routed, 5 distinct", 3, 5, {5, 5, 0, 0}, 1.0},
        {"T3 nothing routed (launch only)", 3, 5, {0, 0, 0, 0}, 0.0},
        {"T2 10 routed, some shared", 2, 5, {5, 5, 0, 0}, 0.4},
        {"T4 20 routed, some shared", 4, 5, {5, 5, 5, 5}, 0.33},
        {"T3 24 routed, some shared (k 8)", 3, 8, {8, 8, 8, 0}, 0.3},
    };
    for (const Shape& sp : shapes) {
        std::vector<unsigned long long> all(POOLS * WT * 8); std::vector<float> pw;
        int distinct = 0, routed = 0;
        for (int p = 0; p < POOLS; ++p) {
            std::vector<unsigned long long> plan;
            make_plans(p, sp.T, sp.k, sp.ne, sp.share, plan, pw);
            std::copy(plan.begin(), plan.end(), all.begin() + p * WT * 8);
            std::vector<unsigned long long> u;
            for (auto v : plan) if (v) { ++routed; if (std::find(u.begin(), u.end(), v) == u.end()) u.push_back(v); }
            distinct += int(u.size());
        }
        ptrs.put(all.data(), all.size() * 8); w.put(pw.data(), pw.size() * 4);
        check(hipStreamSynchronize(s), "plans");
        std::printf("window shape %-32s routed %.1f distinct %.1f a call: ", sp.name, routed / double(POOLS), distinct / double(POOLS));
        auto go = [&](int dedup, int p) {
            gf::moe_experts_multi(gu_type, dn_type, dev(p, sp.T, dedup), sp.k, EMB, FF, 10.f, gu_bytes, xq.p, hq_b.p,
                                  nullptr, 0, nullptr, 0, nullptr, out_b.as<float>(), s);
        };
        timing("window", V::production, size_t(distinct) * blob_bytes / POOLS, s, [&](int p) { go(0, p); }, [&](int p) { go(1, p); });
    }
    return ok;
}
}
int main(int argc, char** argv) {
    bool parity_only = false, shared_mode = false, sweep = false, window = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--parity-only") parity_only = true;
        else if (std::string(argv[i]) == "--shared") shared_mode = true;
        else if (std::string(argv[i]) == "--sweep") sweep = true;
        else if (std::string(argv[i]) == "--window") window = true;
        else { std::fprintf(stderr, "usage: %s [--parity-only] [--shared] [--sweep] [--window]\n", argv[0]); return 2; }
    }
    check(hipSetDevice(0), "GPU 0");
    hipDeviceProp_t prop;
    check(hipGetDeviceProperties(&prop, 0), "device properties");
    size_t free_bytes, total_bytes;
    check(hipMemGetInfo(&free_bytes, &total_bytes), "VRAM info");
    std::printf("GPU 0: %s (%s), free=%.1f MiB, graph_calls=%d rounds=%d (median)\n",
                prop.name, prop.gcnArchName, free_bytes / 1048576., CALLS, ROUNDS);
    hipStream_t stream;
    check(hipStreamCreate(&stream), "stream create");
    if (window) {
        // the window launch with shared experts unpacked once: Maya-S's formats, then the other pairs it takes
        bool wok = run_window(16, 22, parity_only, stream);
        wok &= run_window(16, 18, parity_only, stream);
        if (parity_only) { wok &= run_window(22, 22, true, stream); wok &= run_window(18, 16, true, stream); }
        check(hipStreamDestroy(stream), "stream destroy");
        std::printf("hip_expert_kernel_bench --window: %s\n", wok ? "PASS" : "FAIL");
        return wok ? 0 : 1;
    }
    bool ok = run(16, 22, parity_only, shared_mode, sweep, stream);
    ok &= run(16, 18, parity_only, shared_mode, sweep, stream);
    ok &= run(21, 17, parity_only, shared_mode, sweep, stream);
    if (parity_only) { ok &= run(18, 16, true, shared_mode, sweep, stream); ok &= run(22, 22, true, shared_mode, sweep, stream); }
    check(hipStreamDestroy(stream), "stream destroy");
    std::printf("hip_expert_kernel_bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
