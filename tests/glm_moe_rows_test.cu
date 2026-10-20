// tests/glm_moe_rows_test.cu - the batched rows' MoE pieces (glm_rows.cu, STRATA_GLM_SKIP_EMPTY / _GROUP_EXPERTS)
// against what the per-row path runs, on random Q8_0 experts (needs a GPU, no model):
//  - moe_wait_fetch against moe_wait + moe_fetch: a route with only RAM-tier fetches, and one with a disk miss the host
//    has answered (its source, table edits and CPU part) - the same slots, tables and parts;
//  - moe_cpu_wait_rows against a moe_cpu_wait per row (rows with and without a CPU-lane answer);
//  - a group's routed experts each once (moe_win_experts over the rows' plans, the shared expert's down rows by mv_nt)
//    against each row's moe_gate_up (with the fused shared down) + moe_down: bit for bit, with experts shared between
//    rows, left-out entries and a row with a host part.
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

int g_fail = 0;
std::mt19937 g_rng(1234);

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
template<class T> T* mapped(size_t n) {
    T* p = nullptr;
    CK(cudaHostAlloc((void**) &p, n * sizeof(T), cudaHostAllocMapped));
    std::memset((void*) p, 0, n * sizeof(T));
    return p;
}
template<class T> std::vector<T> back(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template<class T> void put(T* d, const std::vector<T>& h) {
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
void same(const char* what, const void* a, const void* b, size_t bytes) {
    if (std::memcmp(a, b, bytes) == 0) return;
    std::fprintf(stderr, "FAIL: %s differs\n", what);
    ++g_fail;
}

// ---- moe_wait_fetch == moe_wait + moe_fetch
struct Route {
    gf::MoeDev d;
    unsigned long long* plan_ptr;
    unsigned long long* fetch_src;
    unsigned long long* tab;
    unsigned int* wait_seq;
    float* cpu_part;
    int* cpu_flag;
    uint8_t* slots;
};
constexpr int kTab = 64;
constexpr size_t kBlob = 40000;   // (not a multiple of 16 bytes x blocks: the copy's tail)

Route route_make(gf::MoeResponse* resp_d) {
    Route r;
    r.plan_ptr = dev<unsigned long long>(8);
    r.fetch_src = dev<unsigned long long>(8);
    r.tab = dev<unsigned long long>(kTab);
    r.wait_seq = dev<unsigned int>(1);
    r.cpu_part = dev<float>(256);
    r.cpu_flag = dev<int>(1);
    r.slots = dev<uint8_t>(8 * kBlob);
    std::vector<unsigned long long> pp(8);
    for (int i = 0; i < 8; ++i) pp[(size_t) i] = (unsigned long long) (r.slots + (size_t) i * kBlob);
    put(r.plan_ptr, pp);
    r.d.tab = r.tab;
    r.d.plan_ptr = r.plan_ptr;
    r.d.fetch_src = r.fetch_src;
    r.d.wait_seq = r.wait_seq;
    r.d.resp = resp_d;
    r.d.cpu_part = r.cpu_part;
    r.d.cpu_flag = r.cpu_flag;
    return r;
}

void test_wait_fetch() {
    auto* resp = mapped<gf::MoeResponse>(1);
    std::vector<uint8_t*> src(8);
    for (auto& p : src) {
        p = mapped<uint8_t>(kBlob);
        for (size_t j = 0; j < kBlob; ++j) p[j] = (uint8_t) g_rng();
    }
    for (int scenario = 0; scenario < 2; ++scenario) {
        Route a = route_make(resp), b = route_make(resp);
        std::vector<unsigned long long> fs(8, 0ull);
        fs[1] = (unsigned long long) src[1];
        fs[4] = (unsigned long long) src[4];
        unsigned int need = 0;
        std::memset((void*) resp, 0, sizeof(gf::MoeResponse));
        if (scenario == 1) {   // a disk miss at entry 2 the host answered: its source, two table edits, a CPU part
            need = 7;
            resp->ptr_mask = 1u << 2;
            resp->ptr[2] = (unsigned long long) src[2];
            resp->n_upd = 2;
            resp->upd_key[0] = 3;
            resp->upd_val[0] = 111;
            resp->upd_key[1] = 9;
            resp->upd_val[1] = 222;
            resp->cpu = 1;
            for (int e = 0; e < 256; ++e) resp->cpu_part[e] = 0.25f * (float) e;
            resp->seq = need;
        }
        for (Route* r : {&a, &b}) {
            put(r->fetch_src, fs);
            put(r->wait_seq, std::vector<unsigned int>{need});
        }
        gf::moe_wait(a.d, 256, 0);
        gf::moe_fetch(a.d, 8, kBlob, 0);
        gf::moe_wait_fetch(b.d, 256, 8, kBlob, 0);
        CK(cudaDeviceSynchronize());
        const auto sa = back(a.slots, 8 * kBlob), sb = back(b.slots, 8 * kBlob);
        same(scenario ? "wait_fetch slots (disk miss)" : "wait_fetch slots", sa.data(), sb.data(), sa.size());
        const auto ta = back(a.tab, kTab), tb = back(b.tab, kTab);
        same("wait_fetch tables", ta.data(), tb.data(), ta.size() * 8);
        const auto pa = back(a.cpu_part, 256), pb = back(b.cpu_part, 256);
        same("wait_fetch cpu part", pa.data(), pb.data(), pa.size() * 4);
        const auto fa = back(a.cpu_flag, 1), fb = back(b.cpu_flag, 1);
        same("wait_fetch cpu flag", fa.data(), fb.data(), 4);
        if (scenario == 1 && std::memcmp(sb.data() + 2 * kBlob, src[2], kBlob) != 0) {
            std::fprintf(stderr, "FAIL: the disk miss's slot does not hold its source\n");
            ++g_fail;
        }
    }
}

// ---- moe_cpu_wait_rows == a moe_cpu_wait per row
void test_cpu_wait_rows() {
    constexpr int E = 256, N = 5;
    std::vector<gf::CpuAnswer*> ans(N);
    std::vector<unsigned int*> seq(N);
    float* out_a = dev<float>((size_t) N * E);
    float* out_b = dev<float>((size_t) N * E);
    std::vector<float> init((size_t) N * E);
    for (auto& v : init) v = (float) ((int) (g_rng() % 2001) - 1000) / 64.0f;
    put(out_a, init);
    put(out_b, init);
    gf::CpuWaitRows w;
    for (int r = 0; r < N; ++r) {
        ans[(size_t) r] = mapped<gf::CpuAnswer>(1);
        seq[(size_t) r] = dev<unsigned int>(1);
        const unsigned int sq = r == 2 ? 0u : 40u + (unsigned) r;   // row 2: no CPU-lane experts
        put(seq[(size_t) r], std::vector<unsigned int>{sq});
        for (int e = 0; e < E; ++e) ans[(size_t) r]->part[e] = (float) (r + 1) * 0.125f + (float) e;
        ans[(size_t) r]->seq = sq == 0 ? 99u : sq;
        w.seq[r] = seq[(size_t) r];
        w.ans[r] = ans[(size_t) r];
        w.out[r] = out_b + (size_t) r * E;
        gf::MoeDev d;
        d.cpu_seq = seq[(size_t) r];
        d.cpu_ans = ans[(size_t) r];
        gf::moe_cpu_wait(d, E, out_a + (size_t) r * E, 0);
    }
    gf::moe_cpu_wait_rows(w, N, E, 0);
    CK(cudaDeviceSynchronize());
    const auto a = back(out_a, (size_t) N * E), b = back(out_b, (size_t) N * E);
    same("cpu_wait_rows", a.data(), b.data(), a.size() * 4);
}

// ---- a group's experts each once == each row's own
std::vector<uint8_t> q8_0_rows(int rows, int n_in) {   // random Q8_0 rows (FP16 scale + 32 int8 a block)
    std::vector<uint8_t> b((size_t) rows * (n_in / 32) * 34);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (size_t at = 0; at < b.size(); at += 34) {
        const __half h = __float2half(sc(g_rng));
        std::memcpy(&b[at], &h, 2);
        for (int j = 0; j < 32; ++j) b[at + 2 + (size_t) j] = (uint8_t) (int8_t) ((int) (g_rng() % 255) - 127);
    }
    return b;
}

void test_grouped_experts() {
    constexpr int E = 512, FF = 256, FFs = 256, K = 8, NE = 12, N = 4, T8 = 8;   // Q8_0 = type 8
    constexpr size_t q8 = 36;
    const size_t rb_e = (size_t) (E / 32) * 34, rb_f = (size_t) (FF / 32) * 34;
    const size_t blob = 2 * (size_t) FF * rb_e + (size_t) E * rb_f, down_off = 2 * (size_t) FF * rb_e;
    uint8_t* experts = dev<uint8_t>((size_t) NE * blob);
    for (int e = 0; e < NE; ++e) {
        std::vector<uint8_t> gu = q8_0_rows(2 * FF, E), dn = q8_0_rows(E, FF);
        CK(cudaMemcpy(experts + (size_t) e * blob, gu.data(), gu.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(experts + (size_t) e * blob + down_off, dn.data(), dn.size(), cudaMemcpyHostToDevice));
    }
    std::vector<uint8_t> shd = q8_0_rows(E, FFs);
    uint8_t* sh_down = dev<uint8_t>(shd.size());
    put(sh_down, shd);
    // the rows' FFN inputs and shared-expert h (q8_1 from random floats)
    std::vector<float> xh((size_t) N * E), hh((size_t) N * FFs);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : xh) v = nd(g_rng);
    for (auto& v : hh) v = 0.3f * nd(g_rng);
    float* xf = dev<float>(xh.size());
    float* hf = dev<float>(hh.size());
    put(xf, xh);
    put(hf, hh);
    uint8_t* xq = dev<uint8_t>((size_t) N * (E / 32) * q8);
    uint8_t* sh_hq = dev<uint8_t>((size_t) N * (FFs / 32) * q8);
    for (int r = 0; r < N; ++r) {
        gf::quantize_q8_1(xf + (size_t) r * E, xq + (size_t) r * (E / 32) * q8, E, 0);
        gf::quantize_q8_1(hf + (size_t) r * FFs, sh_hq + (size_t) r * (FFs / 32) * q8, FFs, 0);
    }
    // the plans: rows share experts (rows 0 / 1 three of them, row 3 two of row 0's), left-out entries (0), row 2
    // has a host part (cpu_flag)
    const int ids[N][K] = {{0, 1, 2, 3, 4, 5, 6, 7}, {2, 1, 0, 8, 9, -1, 10, 11}, {11, 10, 9, 8, -1, -1, 3, 5},
                           {7, 6, 1, 2, 3, 4, 5, 0}};
    std::vector<unsigned long long> pp((size_t) N * T8, 0ull);
    std::vector<float> pw((size_t) N * T8, 0.0f), cpart((size_t) N * E, 0.0f);
    std::vector<int> cflag(N, 0);
    std::uniform_real_distribution<float> wd(0.05f, 0.4f);
    for (int r = 0; r < N; ++r)
        for (int i = 0; i < K; ++i) {
            pp[(size_t) r * T8 + i] = ids[r][i] < 0 ? 0ull : (unsigned long long) (experts + (size_t) ids[r][i] * blob);
            pw[(size_t) r * T8 + i] = wd(g_rng);
        }
    cflag[2] = 1;
    for (int e = 0; e < E; ++e) cpart[(size_t) 2 * E + e] = 0.01f * (float) (e % 17);
    auto* plan_ptr = dev<unsigned long long>(pp.size());
    auto* plan_w = dev<float>(pw.size());
    auto* cpu_part = dev<float>(cpart.size());
    auto* cpu_flag = dev<int>(N);
    put(plan_ptr, pp);
    put(plan_w, pw);
    put(cpu_part, cpart);
    put(cpu_flag, cflag);
    const float limit = 7.0f;
    // per row: moe_gate_up with the fused shared down, then moe_down
    float* out_a = dev<float>((size_t) N * E);
    float* sh_a = dev<float>(E);
    uint8_t* hq_a = dev<uint8_t>((size_t) T8 * (FF / 32) * q8);
    for (int r = 0; r < N; ++r) {
        gf::MoeDev d;
        d.plan_ptr = plan_ptr + (size_t) r * T8;
        d.plan_w = plan_w + (size_t) r * T8;
        d.cpu_part = cpu_part + (size_t) r * E;
        d.cpu_flag = cpu_flag + r;
        gf::moe_gate_up(8, d, K, E, FF, limit, xq + (size_t) r * (E / 32) * q8, hq_a, sh_down, 8,
                        sh_hq + (size_t) r * (FFs / 32) * q8, FFs, sh_a, 0);
        gf::moe_down(8, d, K, E, FF, down_off, hq_a, sh_a, out_a + (size_t) r * E, 0);
    }
    // grouped: the shared down rows by mv_nt, then every distinct expert once
    float* out_b = dev<float>((size_t) N * E);
    float* sh_b = dev<float>((size_t) N * E);
    gf::MvJob sd = {sh_down, sh_hq, nullptr, sh_b, nullptr, 1.0f, 8, FFs, E};
    if (!gf::mv_nt(&sd, 1, N, 0)) {
        std::fprintf(stderr, "FAIL: mv_nt (Q8_0)\n");
        ++g_fail;
    }
    gf::MoeWin mw;
    mw.plan_ptr = plan_ptr;
    mw.plan_w = plan_w;
    mw.cpu_part = cpu_part;
    mw.cpu_flag = cpu_flag;
    mw.xq = xq;
    mw.xq_stride = (size_t) (E / 32) * q8;
    mw.hq = dev<uint8_t>((size_t) N * T8 * (FF / 32) * q8);
    mw.part = dev<float>((size_t) N * T8 * E);
    mw.sh_out = sh_b;
    mw.out = out_b;
    mw.out_stride = E;
    mw.grp_ptr = dev<unsigned long long>((size_t) N * T8);
    mw.grp_start = dev<int>((size_t) N * T8 + 1);
    mw.grp_ent = dev<int>((size_t) N * T8);
    mw.n_groups = dev<int>(1);
    mw.T = N;
    mw.k = K;
    mw.n_embd = E;
    mw.n_ff = FF;
    mw.limit = limit;
    mw.down_off = down_off;
    if (!gf::moe_win_experts(8, 8, mw, 0)) {
        std::fprintf(stderr, "FAIL: moe_win_experts (Q8_0)\n");
        ++g_fail;
    }
    CK(cudaDeviceSynchronize());
    if (gf::launch_errors() > 0) {
        std::fprintf(stderr, "FAIL: %d kernel launch(es) failed\n", gf::launch_errors());
        ++g_fail;
    }
    const auto a = back(out_a, (size_t) N * E), b = back(out_b, (size_t) N * E);
    const auto ng = back(mw.n_groups, 1);
    if (ng[0] != NE) {
        std::fprintf(stderr, "FAIL: %d expert groups, want %d\n", ng[0], NE);
        ++g_fail;
    }
    int diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
    if (diff != 0) {
        std::fprintf(stderr, "FAIL: grouped experts: %d of %zu outputs differ from the per-row kernels' (e.g. %g vs %g)\n",
                     diff, a.size(), (double) a[0], (double) b[0]);
        ++g_fail;
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("glm_moe_rows_test: no GPU, skipped\n");
        return 0;
    }
    test_wait_fetch();
    test_cpu_wait_rows();
    test_grouped_experts();
    if (g_fail == 0) std::printf("glm_moe_rows_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
