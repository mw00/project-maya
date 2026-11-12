// tests/glm_dsa/glm_tier_engine_test.cpp - glm-dsa B8.3 / B8.4 through the engine's own entry points, on the CPU (links
// strata_engine; calls nothing that needs a device, so it runs without a GPU).
//
//   1. glmtier::cpu_experts - the node-split lane as fast_cpu_experts calls it: two NodeLane pools (one per socket,
//      here unpinned), each computing its half of every expert of the call and a partial y, the partials added -
//      against the serial half reference (< 1e-6) and Maya's unsplit lane on the unified blobs (< 1e-5), repeated
//      (the lanes' task hand-off) and from two threads at once (two parts sharing the lanes);
//   2. the fall-throughs that keep Maya's path: no tier, a layer that is not split, a slot outside the class;
//   3. glmtier::read_split - a disk read into a split slot (fast_read_part's chunks) lands in both halves as scatter;
//   5. (W6) socket_lane keeps the per-socket NodeLane for the load; set_lane_threads drives both node-split pools;
//   4. glmtier::lane_pools' mode on this machine (one pool for one part; Maya's shared pool for 3 parts unless two
//      NUMA nodes are online and MAYA_GLM_CPU_SHARED=socket) and its log line.
#include "glm_tier_numa.hpp"

#include "strata/core/glm_arch.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace gt = strata::core::glmtier;
namespace kc = strata::kernels::cpu;
using strata::core::Glm5Model;
using strata::core::Glm5Geometry;
using strata::core::GlmArch;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void fill(ggml_type t, uint8_t* p, size_t bytes, std::mt19937& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bb = ggml_type_size(t);
    const size_t off = t == GGML_TYPE_Q2_K ? 80 : 0;
    for (size_t b = 0; b + bb <= bytes; b += bb) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(0.002f + 0.002f * (float) (rng() % 100) / 100.0f);
        std::memcpy(p + b + off, &h, 2);
    }
}

double max_rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) num = std::max(num, (double) std::fabs(a[i] - b[i])), den = std::max(den, (double) std::fabs(b[i]));
    return den > 0 ? num / den : num;
}

}  // namespace

int main() {
    std::mt19937 rng(8004);
    if (!kc::native_experts_available()) {
        std::printf("glm_tier_engine_test: no ggml-cpu in this build\n");
        return 77;
    }
    const int E = 6144, FF = 2048, NE = 5, SLOTS = 6;
    const ggml_type gu = GGML_TYPE_IQ2_XXS, dn = GGML_TYPE_IQ3_XXS;
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_embd = E;
    g.n_ff_exp = FF;
    g.swiglu_exp = std::numeric_limits<float>::infinity();
    kc::NativeFmt f, hf;
    std::string why;
    gt::SplitLayout L;
    if (!kc::native_fmt(gu, dn, E, FF, f, why) || !gt::half_fmt(f, hf, why) ||
        !gt::split_layout(f.gu_row, f.d_row, FF, E, ggml_blck_size(dn), ggml_type_size(dn), L, why)) {
        std::printf("glm_tier_engine_test: setup failed: %s\n", why.c_str());
        return 1;
    }
    // a fake split RAM class over plain memory: SLOTS slots a half, half 1 at + D (layer 1 split, layer 0 not)
    const size_t stride = gt::half_stride({L}), D = (size_t) SLOTS * stride;
    std::vector<uint8_t> map(2 * D + 64, 0);
    uint8_t* base = map.data();
    Glm5Model::FastState F;
    auto T = std::make_shared<gt::TierState>();
    T->dsa = true;
    T->split_req = true;
    T->nodes = {0, 1};
    T->cls.resize(1);
    T->cls[0].ours = true;
    T->cls[0].split = true;
    T->cls[0].map = base;
    T->cls[0].D = D;
    T->cls[0].stride = stride;
    T->cls[0].n = SLOTS;
    T->L = {gt::SplitLayout{}, L};
    T->half = {kc::NativeFmt{}, hf};
    T->layer_cls = {0, 0};
    T->lane[0] = std::make_shared<gt::NodeLane>(0, std::vector<int>{}, 3);
    T->lane[1] = std::make_shared<gt::NodeLane>(1, std::vector<int>{}, 2);
    T->act.assign(kc::kNativeActBytes, 0);
    for (int h = 0; h < 2; ++h) {
        T->ff[h].assign((size_t) 8 * L.H, 0.0f);
        T->hq[h].assign((size_t) 8 * kc::kNativeHBytes, 0);
        T->dn[h].assign((size_t) 8 * E, 0.0f);
        T->part[h].assign((size_t) E, 0.0f);
    }
    F.tier = T;
    F.cpu_fmt = {f, f};

    // NE experts in slots 5, 0, 3, 1, 4 (unified copies kept for the reference)
    const int slot_of[NE] = {5, 0, 3, 1, 4};
    std::vector<std::vector<uint8_t>> blob((size_t) NE, std::vector<uint8_t>(f.bytes));
    std::vector<const uint8_t*> h0p, up;
    for (int i = 0; i < NE; ++i) {
        fill(gu, blob[(size_t) i].data(), 2 * f.up_off, rng);
        fill(dn, blob[(size_t) i].data() + f.down_off, f.bytes - f.down_off, rng);
        uint8_t* s = base + (size_t) slot_of[i] * stride;
        gt::scatter(L, blob[(size_t) i].data(), s, s + D);
        h0p.push_back(s);
        up.push_back(blob[(size_t) i].data());
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) E);
    for (auto& v : x) v = nd(rng);
    const float w[NE] = {0.3f, 0.25f, 0.2f, 0.15f, 0.1f};

    // references: the serial halves (+ one add) and Maya's unsplit lane
    std::vector<uint8_t> act(kc::kNativeActBytes), hq(kc::kNativeHBytes);
    kc::native_quant_act(f, x.data(), act.data());
    std::vector<float> y0((size_t) E), y1((size_t) E), ys((size_t) E), sff((size_t) FF), sdn((size_t) E);
    std::vector<const uint8_t*> h1p;
    for (auto* p : h0p) h1p.push_back(p + D);
    gt::half_experts_serial(hf, L, NE, h0p.data(), w, act.data(), g.swiglu_exp, y0.data(), sff.data(), hq.data(), sdn.data());
    gt::half_experts_serial(hf, L, NE, h1p.data(), w, act.data(), g.swiglu_exp, y1.data(), sff.data(), hq.data(), sdn.data());
    gt::add_partial(y0.data(), y1.data(), ys.data(), E);
    std::vector<float> yu((size_t) E, 0.0f), ffu((size_t) FF), dnu((size_t) E);
    for (int i = 0; i < NE; ++i) {
        const void* a[1] = {act.data()};
        float* o[1] = {ffu.data()};
        kc::native_gu_rows_split(f, up[(size_t) i], up[(size_t) i] + f.up_off, a, 1, o, 0, FF, g.swiglu_exp);
        kc::native_quant_h(f, ffu.data(), hq.data());
        const void* hh[1] = {hq.data()};
        float* d[1] = {dnu.data()};
        kc::native_down_rows_split(f, up[(size_t) i] + f.down_off, hh, 1, d, 0, E);
        for (int r = 0; r < E; ++r) yu[(size_t) r] += w[i] * dnu[(size_t) r];
    }

    std::printf("B8.4 node-split lane through glmtier::cpu_experts (2 NodeLanes, %d experts)\n", NE);
    std::vector<float> out((size_t) E, -1.0f);
    const bool ran = gt::cpu_experts(F, g, 1, NE, h0p.data(), w, x.data(), out.data());
    check(ran, "a split layer takes the node-split lane");
    char b1[48], b2[48];
    std::snprintf(b1, sizeof b1, "%.1e", max_rel(out.data(), ys.data(), E));
    std::snprintf(b2, sizeof b2, "%.1e", max_rel(out.data(), yu.data(), E));
    check(max_rel(out.data(), ys.data(), E) < 1e-6, std::string("pooled halves vs the serial halves: ") + b1 + " < 1e-6");
    check(max_rel(out.data(), yu.data(), E) < 1e-5, std::string("pooled node-split vs Maya's unsplit lane: ") + b2 + " < 1e-5");
    bool repeat = true;
    for (int k = 0; k < 20; ++k) {
        std::vector<float> o2((size_t) E);
        gt::cpu_experts(F, g, 1, NE - (k % 3), h0p.data(), w, x.data(), o2.data());
        if (k % 3 == 0) repeat = repeat && std::memcmp(o2.data(), out.data(), (size_t) E * 4) == 0;
    }
    check(repeat, "20 calls of 3..5 experts: the lanes hand tasks over, the same experts give the same bits");
    std::vector<float> oa((size_t) E), ob((size_t) E);
    bool par = true;
    for (int k = 0; k < 10; ++k) {
        std::thread ta([&] { gt::cpu_experts(F, g, 1, NE, h0p.data(), w, x.data(), oa.data()); });
        std::thread tb([&] { gt::cpu_experts(F, g, 1, NE, h0p.data(), w, x.data(), ob.data()); });
        ta.join();
        tb.join();
        par = par && std::memcmp(oa.data(), out.data(), (size_t) E * 4) == 0 && std::memcmp(ob.data(), out.data(), (size_t) E * 4) == 0;
    }
    check(par, "two parts calling at once (the shared lanes serialise the calls)");
    check(T->split_calls.load() == 1 + 20 + 20, "split_calls counts every call");

    std::printf("Fall-throughs to Maya's path\n");
    check(!gt::cpu_experts(F, g, 0, NE, h0p.data(), w, x.data(), out.data()), "an unsplit layer: Maya's lane runs");
    check(gt::split_layer(T.get(), 1) && !gt::split_layer(T.get(), 0) && !gt::split_layer(nullptr, 1) &&
              !gt::split_layer(T.get(), 7),
          "split_layer: only layers of a split class");
    Glm5Model::FastState F0;
    check(!gt::cpu_experts(F0, g, 1, NE, h0p.data(), w, x.data(), out.data()), "no tier (glm5-next, segmentation off)");
    std::vector<uint8_t> elsewhere(f.bytes);
    bool called = false;
    check(!gt::read_split(T.get(), 1, elsewhere.data(), 0, 4096, [&](uint8_t*) { called = true; }) && !called,
          "read_split: a destination outside the class (disk staging) reads as Maya does");
    check(!gt::read_split(T.get(), 1, nullptr, 0, 4096, [&](uint8_t*) { called = true; }) && !called,
          "read_split: a VRAM-bound read (no RAM slot) reads as Maya does");

    std::printf("B8.4 disk reads into a split slot (read_split, fast_read_part's chunks)\n");
    {
        uint8_t* s = base + 2 * stride;
        std::memset(s, 0, stride);
        std::memset(s + D, 0, stride);
        const std::vector<uint8_t>& u = blob[2];
        const size_t roles[3][2] = {{0, L.gu_bytes}, {L.gu_bytes, L.gu_bytes}, {L.down_off, L.blob - L.down_off}};
        bool all = true;
        for (const auto& rl : roles) {
            const int n_chunks = 5;
            const size_t len = rl[1];
            const size_t piece = ((len + n_chunks - 1) / n_chunks + 4095) & ~(size_t) 4095;
            for (int c = 0; c < n_chunks; ++c) {
                const size_t c0 = std::min(len, (size_t) c * piece), c1 = std::min(len, c0 + piece);
                if (c1 <= c0) continue;
                all = all && gt::read_split(T.get(), 1, s, rl[0] + c0, c1 - c0,
                                            [&](uint8_t* buf) { std::memcpy(buf, u.data() + rl[0] + c0, c1 - c0); });
            }
        }
        std::vector<uint8_t> r0(L.half_bytes), r1(L.half_bytes);
        gt::scatter(L, u.data(), r0.data(), r1.data());
        check(all && std::memcmp(s, r0.data(), L.half_bytes) == 0 && std::memcmp(s + D, r1.data(), L.half_bytes) == 0,
              "gate, up and down roles in 5 chunks each land in both halves as scatter");
    }

    std::printf("B8.3 lane pool mode on this machine\n");
    {
        Glm5Model::FastState G;
        unsetenv("MAYA_GLM_CPU_SHARED");
        check(!gt::lane_pools(G, g, 1, 0, false, false, 4, -1, 0) && G.cpu_pool == nullptr,
              "one part: Maya makes its pool (mode One)");
        Glm5Model::FastState G3;
        check(!gt::lane_pools(G3, g, 3, 1, false, false, 4, -1, 0) && G3.cpu_pool == nullptr,
              "3 parts, MAYA_GLM_CPU_SHARED unset: Maya's shared pool");
        setenv("MAYA_GLM_CPU_SHARED", "socket", 1);
        Glm5Model::FastState GS;
        const bool socket = gt::lane_pools(GS, g, 3, 1, false, false, 4, -1, 0);
        std::string nodes;
        {
            FILE* fo = std::fopen("/sys/devices/system/node/online", "r");
            char bufn[64] = "";
            if (fo != nullptr) {
                if (std::fgets(bufn, sizeof bufn, fo) == nullptr) bufn[0] = 0;
                std::fclose(fo);
            }
            nodes = bufn;
            nodes.erase(std::remove(nodes.begin(), nodes.end(), '\n'), nodes.end());
        }
        const bool two = nodes.find('-') != std::string::npos || nodes.find(',') != std::string::npos;
        check(socket == two && (!socket || (GS.cpu_pool != nullptr && GS.cpu_node >= 0)),
              "MAYA_GLM_CPU_SHARED=socket: a pinned per-socket pool iff two NUMA nodes (online: " + nodes + ")");
        unsetenv("MAYA_GLM_CPU_SHARED");
    }

    std::printf("W6 runtime fix (B8.3): the per-socket lane is kept for the load and shared by the node's parts\n");
    {
        std::weak_ptr<gt::NodeLane> first;
        {
            Glm5Model::FastState P0, P1;
            auto a = gt::socket_lane(P0, g, 0, 3, 0);
            first = a;
            a.reset();   // lane_pools keeps only the pool: the NodeLane must live on in P0's tier state
            check(!first.expired() && P0.tier != nullptr && P0.tier->socket_lane != nullptr && P0.tier->ready,
                  "part 0's NodeLane outlives the lane_pools call (held by its tier state)");
            auto b = gt::socket_lane(P1, g, 0, 3, 0);
            check(b != nullptr && b == first.lock() && b->pool == P0.tier->socket_lane->pool,
                  "part 1 on the same node gets the same NodeLane and the same Workers pool");
            b.reset();
        }
        check(first.expired(), "the load's teardown (both parts' FastState gone) lets the lane go");
        Glm5Model::FastState P2;
        auto c = gt::socket_lane(P2, g, 0, 3, 0);
        check(c != nullptr && c->pool != nullptr && c->pool->size() == 3, "a new load makes a new lane");
    }

    std::printf("W6 runtime fix (B8.4): the cpu_threads knob on both node-split pools\n");
    {
        const int s0 = T->lane[0]->pool->size(), s1 = T->lane[1]->pool->size();
        T->mode = gt::LanePool::One;
        check(!gt::set_lane_threads(F, 1), "not node-split: false (Maya's set_active on F.cpu_pool)");
        T->mode = gt::LanePool::NodeSplit;
        const bool set2 = gt::set_lane_threads(F, 2);
        const int a0 = T->lane[0]->pool->active(), a1 = T->lane[1]->pool->active();
        check(set2 && a0 == 2 && a1 == gt::lane_threads_scaled(2, s0, s1) && a1 < s1,
              "node-split, n = 2: node 0 gets 2 of " + std::to_string(s0) + ", node 1 its scaled share (" +
                  std::to_string(a1) + " of " + std::to_string(s1) + ")");
        std::vector<float> o2((size_t) E, 0.0f);
        check(gt::cpu_experts(F, g, 1, NE, h0p.data(), w, x.data(), o2.data()) && max_rel(o2.data(), out.data(), E) < 1e-6,
              "cpu_experts at the reduced counts: the same result");
        check(gt::set_lane_threads(F, 0) && T->lane[0]->pool->active() == s0 && T->lane[1]->pool->active() == s1,
              "n = 0: every thread on both nodes");
        T->mode = gt::LanePool::One;
    }

    if (g_fail) {
        std::printf("glm_tier_engine_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_tier_engine_test: all passed\n");
    return 0;
}
