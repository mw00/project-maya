// tests/glm_route_one_test.cu - the decode's one-pass route (moe_route_one_kernel, STRATA_GLM_ROUTE_ONE) against
// moe_route_kernel, bit for bit (needs a GPU, no model).  Random routes over 256 experts: ties (quantized scores),
// non-finite scores, an earlier route's error, VRAM / RAM-tier / disk-only / helper-owned experts, the CPU lane's
// split table, spares free or taken, PROMOTE_MIN, TRUNK_KEEP, DISK_SKIP and MISS_SKIP, k of 1..8, the counts' aging
// (every 4096 routes) and the shared expert's q8_1.  After every route a hash of everything the route writes - the
// plan, the sequences and flags, the whole table (spares included), the route counts, the request's slot and the
// shared expert's h.  The env switches are read once a process, so the test runs itself once a setting
// (STRATA_GLM_ROUTE_ONE 0 / 1, STRATA_GLM_ROUTE_AFFINITY 0 / 0.02) and compares the hashes; it also times both.
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        const cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));              \
            std::exit(2);                                                                                        \
        }                                                                                                        \
    } while (0)

template<class T> T* dev(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc((void**) &p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

template<class T> T* mapped(size_t n, T** d) {
    T* p = nullptr;
    CK(cudaHostAlloc((void**) &p, n * sizeof(T), cudaHostAllocMapped));
    std::memset(p, 0, n * sizeof(T));
    CK(cudaHostGetDevicePointer((void**) d, p, 0));
    return p;
}

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

template<class T> uint64_t fnv_dev(uint64_t h, const T* p, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return fnv(h, v.data(), n * sizeof(T));
}

constexpr int E = 256, L = 4, N = 4096, FF_SH = 1024, ROUTES = 20000, TIMED = 5000;

int child() {
    std::mt19937_64 rng(20261028);
    const auto uni = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    const auto chance = [&](double p) { return uni(0.0, 1.0) < p; };
    const auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };

    const int n_keys = L * E;
    gf::MoeDev d;
    d.n_keys = n_keys;
    const size_t tab_n = 2 * (size_t) n_keys + (size_t) (L + 1) * gf::kSpares;
    d.tab = dev<unsigned long long>(tab_n);
    uint8_t* helper = dev<uint8_t>(n_keys);
    d.scratch = dev<unsigned long long>(8);
    d.plan_ptr = dev<unsigned long long>(8);
    d.plan_w = dev<float>(8);
    d.plan_id = dev<int>(8);
    d.fetch_src = dev<unsigned long long>(8);
    d.pf_src = dev<unsigned long long>(8);
    d.pf_dst = dev<unsigned long long>(8);
    d.pf_n = dev<int>(1);
    d.seq = dev<unsigned int>(1);
    d.wait_seq = dev<unsigned int>(1);
    d.cpu_seq = dev<unsigned int>(1);
    d.cpu_flag = dev<int>(1);
    d.dcnt = dev<unsigned int>(n_keys);
    d.route_err_d = dev<int>(1);
    gf::MoeRequest* ring_d = nullptr;
    gf::MoeRequest* ring = mapped<gf::MoeRequest>(gf::kRingSize, &ring_d);
    d.ring = ring_d;
    int* rerr_d = nullptr;
    volatile int* rerr = mapped<int>(1, &rerr_d);
    d.route_error = rerr_d;
    float* logits = dev<float>(E);
    float* bias = dev<float>(E);
    float* x = dev<float>(N);
    float* sh_g = dev<float>(FF_SH);
    float* sh_u = dev<float>(FF_SH);
    const size_t hq_bytes = (size_t) FF_SH / 32 * 36;
    uint8_t* sh_hq = dev<uint8_t>(hq_bytes);
    {
        std::vector<unsigned long long> scr(8);
        for (int i = 0; i < 8; ++i) scr[i] = 0x5c0000000000ull + 0x1000ull * i;
        CK(cudaMemcpy(d.scratch, scr.data(), 64, cudaMemcpyHostToDevice));
        std::vector<float> v(N);
        for (auto& f : v) f = (float) uni(-2, 2);
        CK(cudaMemcpy(x, v.data(), N * 4, cudaMemcpyHostToDevice));
        std::vector<float> g(FF_SH), u(FF_SH);
        for (int i = 0; i < FF_SH; ++i) g[i] = (float) uni(-9, 9), u[i] = (float) uni(-9, 9);
        CK(cudaMemcpy(sh_g, g.data(), FF_SH * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sh_u, u.data(), FF_SH * 4, cudaMemcpyHostToDevice));
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::vector<float> lg(E), bs(E);
    std::vector<unsigned long long> tab(tab_n);
    std::vector<uint8_t> ho(n_keys);
    std::vector<unsigned int> cnt(n_keys);
    struct Case {
        int layer, k, skip_from, promote_min, disk_skip_from, win, win_n, win_rows;
        bool norm_w, use_bias, use_helper;
        unsigned long long cpu_plan;
        float disk_skip_w, miss_skip_w;
    };
    const auto random_case = [&](bool fresh_tables) {
        Case c;
        c.layer = pick(0, L - 1);
        const int ks[] = {8, 8, 8, 8, 6, 4, 1};
        c.k = ks[pick(0, 6)];
        c.skip_from = chance(0.7) ? 8 : pick(0, 8);
        c.promote_min = chance(0.5) ? 0 : pick(1, 4);
        c.disk_skip_from = chance(0.7) ? 8 : pick(0, 8);
        c.disk_skip_w = chance(0.7) ? 0.0f : (float) uni(0.02, 0.2);
        c.miss_skip_w = chance(0.7) ? 0.0f : (float) uni(0.02, 0.15);
        c.win = pick(0, 3), c.win_n = pick(0, 4), c.win_rows = pick(0, 2);
        c.norm_w = chance(0.8);
        c.use_bias = chance(0.6);
        c.use_helper = chance(0.3);
        c.cpu_plan = 0ull;
        if (chance(0.7))
            for (int f = 1; f <= 8; ++f) c.cpu_plan |= (unsigned long long) pick(0, f) << (4 * f);
        // the scores: normal, or quantized to a few values (ties), now and then one not finite
        const bool ties = chance(0.25);
        for (int e = 0; e < E; ++e) {
            float v = (float) uni(-4, 4);
            if (ties) v = std::round(v * 2.0f) / 2.0f;
            lg[e] = v;
            bs[e] = ties && chance(0.5) ? 0.0f : (float) uni(-0.05, 0.05);
        }
        if (chance(0.02)) lg[pick(0, E - 1)] = chance(0.5) ? NAN : INFINITY;
        CK(cudaMemcpyAsync(logits, lg.data(), E * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(bias, bs.data(), E * 4, cudaMemcpyHostToDevice, s));
        if (fresh_tables) {
            // this layer's tables: resident, RAM tier, helper-owned and its spares, at random densities
            CK(cudaMemcpyAsync(tab.data(), d.tab, tab_n * 8, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(cnt.data(), d.dcnt, n_keys * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            const double pv = uni(0, 1), pr = uni(0, 1), ph = uni(0, 0.2), ps = uni(0, 1);
            for (int e = 0; e < E; ++e) {
                const size_t key = (size_t) c.layer * E + e;
                tab[key] = chance(pv) ? (0x7a0000000000ull | ((uint64_t) rng() & 0xffffff000ull)) + 0x1000ull : 0ull;
                tab[n_keys + key] = chance(pr) ? (0x3b0000000000ull | ((uint64_t) rng() & 0xffffff000ull)) + 0x1000ull : 0ull;
                ho[key] = chance(ph) ? 1 : 0;
                cnt[key] = (unsigned int) pick(0, 12);
            }
            for (int j = 0; j < gf::kSpares; ++j)
                tab[2 * (size_t) n_keys + (size_t) c.layer * gf::kSpares + j] =
                    chance(ps) ? 0x9d0000000000ull + 0x100000ull * (uint64_t) pick(1, 4000) : 0ull;
            CK(cudaMemcpyAsync(d.tab, tab.data(), tab_n * 8, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(d.dcnt, cnt.data(), n_keys * 4, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(helper, ho.data(), n_keys, cudaMemcpyHostToDevice, s));
            // an earlier route's error (sticky until cleared), now and then
            const int e0 = chance(0.02) ? 4 * c.layer + 1 : 0;
            CK(cudaMemcpyAsync(d.route_err_d, &e0, 4, cudaMemcpyHostToDevice, s));
            CK(cudaStreamSynchronize(s));
            *rerr = e0;
        }
        return c;
    };
    const auto run = [&](const Case& c) {
        gf::MoeDev dd = d;
        dd.helper_owned = c.use_helper ? helper : nullptr;
        gf::moe_route(logits, c.use_bias ? bias : nullptr, E, c.k, 2.5f, c.norm_w, c.layer, x, N, dd, sh_g, sh_u, 7.0f,
                      FF_SH, sh_hq, s, nullptr, nullptr, 0, nullptr, nullptr, 0, c.skip_from, c.cpu_plan,
                      c.promote_min, 0, 0, c.win, nullptr, c.win_n, c.disk_skip_from, c.disk_skip_w, c.miss_skip_w,
                      c.win_rows);
    };
    int cov[7] = {};
    for (int r = 0; r < ROUTES; ++r) {
        const Case c = random_case(true);
        run(c);
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        uint64_t h = 1469598103934665603ull;
        h = fnv_dev(h, d.plan_ptr, 8);
        h = fnv_dev(h, d.plan_w, 8);
        h = fnv_dev(h, d.plan_id, 8);
        h = fnv_dev(h, d.fetch_src, 8);
        h = fnv_dev(h, d.pf_n, 1);
        h = fnv_dev(h, d.seq, 1);
        h = fnv_dev(h, d.wait_seq, 1);
        h = fnv_dev(h, d.cpu_seq, 1);
        h = fnv_dev(h, d.cpu_flag, 1);
        h = fnv_dev(h, d.route_err_d, 1);
        h = fnv_dev(h, d.tab, tab_n);
        h = fnv_dev(h, d.dcnt, (size_t) n_keys);
        h = fnv_dev(h, sh_hq, hq_bytes);
        const int re = *rerr;
        h = fnv(h, &re, 4);
        unsigned int seq = 0;
        CK(cudaMemcpy(&seq, d.seq, 4, cudaMemcpyDeviceToHost));
        const gf::MoeRequest& q = ring[seq % gf::kRingSize];
        h = fnv(h, &q, sizeof(gf::MoeRequest));
        std::printf("%d %016llx\n", r, (unsigned long long) h);
        // (what the routes covered, to stderr)
        if (q.error) ++cov[0];
        else {
            cov[1] += q.cpu_mask != 0u, cov[2] += q.promo_mask != 0u, cov[3] += (q.fetch_mask & ~q.promo_mask) != 0u;
            cov[4] += q.miss_mask != 0u, cov[5] += q.skip_mask != 0u, cov[6] += __builtin_popcount(q.promo_mask) > 1;
        }
    }
    std::fprintf(stderr, "covered: %d refused, %d with CPU-lane experts, %d promotions (%d two or more), %d scratch "
                 "fetches, %d disk misses, %d disk skips\n", cov[0], cov[1], cov[2], cov[6], cov[3], cov[4], cov[5]);
    // the time a route takes, back to back on the stream (the tables as the last route left them)
    std::vector<Case> cs;
    for (int r = 0; r < 64; ++r) cs.push_back(random_case(r == 0));
    for (auto& c : cs) c.layer = cs[0].layer;   // (one layer's tables were refreshed; the last case's scores)
    int zero = 0;
    CK(cudaMemcpy(d.route_err_d, &zero, 4, cudaMemcpyHostToDevice));
    *rerr = 0;
    for (int r = 0; r < 200; ++r) run(cs[r % 64]);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, s));
    for (int r = 0; r < TIMED; ++r) run(cs[r % 64]);
    CK(cudaEventRecord(t1, s));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    std::printf("time %.2f\n", 1000.0 * ms / TIMED);
    return 0;
}

std::vector<std::string> run_child(const char* self, const char* one, const char* affinity, double* us) {
    const std::string cmd = std::string("STRATA_GLM_ROUTE_ONE=") + one + " STRATA_GLM_ROUTE_AFFINITY=" + affinity +
                            " '" + self + "' child";
    FILE* f = popen(cmd.c_str(), "r");
    std::vector<std::string> lines;
    if (f == nullptr) return lines;
    char buf[256];
    while (std::fgets(buf, sizeof buf, f)) {
        if (std::strncmp(buf, "time ", 5) == 0) *us = std::atof(buf + 5);
        else lines.emplace_back(buf);
    }
    if (pclose(f) != 0) lines.clear();
    return lines;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "child") == 0) return child();
    int fail = 0;
    for (const char* aff : {"0", "0.02"}) {
        double us0 = 0.0, us1 = 0.0;
        const auto a = run_child(argv[0], "0", aff, &us0);
        const auto b = run_child(argv[0], "1", aff, &us1);
        if (a.size() != (size_t) ROUTES || b.size() != (size_t) ROUTES) {
            std::printf("FAIL affinity %s: a run did not finish (%zu / %zu routes)\n", aff, a.size(), b.size());
            ++fail;
            continue;
        }
        int first = -1, n = 0;
        for (int i = 0; i < ROUTES; ++i)
            if (a[i] != b[i]) {
                if (first < 0) first = i;
                ++n;
            }
        if (n) {
            std::printf("FAIL affinity %s: %d of %d routes differ, the first route %d\n", aff, n, ROUTES, first);
            ++fail;
        } else {
            std::printf("ok   affinity %s: %d routes the same bits; a route %.2f us (moe_route_kernel) -> %.2f us "
                        "(one pass)\n", aff, ROUTES, us0, us1);
        }
    }
    std::printf(fail ? "FAILED\n" : "PASSED\n");
    return fail ? 1 : 0;
}
