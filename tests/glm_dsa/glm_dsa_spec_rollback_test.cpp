// tests/glm_dsa/glm_dsa_spec_rollback_test.cpp - glm-dsa B11.2: a rejected draft on glm-dsa needs no copy-back, and
// leaves no stale selection (CPU twins; runs anywhere); decode_spec's step order with MAYA_GLM_SPEC_DRAFT_EARLY off
// and on.  The checks are described in glm_dsa_spec_rollback.hpp.
#include <cstdio>
#include <memory>
#include <string>

namespace {
int g_fail = 0;
}
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

#include "glm_dsa_spec_rollback.hpp"

namespace {

struct CpuBe {
    std::vector<std::unique_ptr<uint8_t[]>> mem;
    void* alloc(size_t n) {
        mem.emplace_back(new uint8_t[std::max<size_t>(n, 4)]());
        return mem.back().get();
    }
    void put(void* d, const void* h, size_t n) { std::memcpy(d, h, n); }
    void get(void* h, const void* s, size_t n) { std::memcpy(h, s, n); }
    void sync() {}
    bool gemv(const float* W, const float* x, float* y, int in, int out, float alpha) {
        for (int o = 0; o < out; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < in; ++i) acc += W[(size_t) o * in + i] * x[i];
            y[o] = alpha * acc;
        }
        return true;
    }
    bool prep(const glmd::DsaPrep& a) { glmd::twin::dsa_prep(a); return true; }
    bool rope_q(const glmd::DsaRopeQ& a) { glmd::twin::dsa_rope_q(a); return true; }
    bool score(const glmd::IdxScore& a) { glmd::twin::idx_score(a); return true; }
    bool topk(const float* s, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        glmd::twin::topk(s, ld, n_kv, 1, pos, top_k, sel);
        return true;
    }
    bool mla2(const glmd::Mla2& a) { glmd::twin::mla2(a); return true; }
};

}  // namespace

int main() {
    spec_rb::suite<CpuBe>("SMALL", mixfx::small_dims(64), 40, 0);
    spec_rb::suite<CpuBe>("SMALL", mixfx::small_dims(64), 40, 1);
    spec_rb::suite<CpuBe>("MID", mixfx::mid_dims(64), 36, 0);
    // glm-dsa B11.2: decode_spec's step order, MAYA_GLM_SPEC_DRAFT_EARLY off and on (glm_dsa_spec_rollback.hpp)
    spec_rb::schedule<CpuBe>("SMALL", mixfx::small_dims(64), 40, 0, false);
    spec_rb::schedule<CpuBe>("SMALL", mixfx::small_dims(64), 40, 0, true);
    spec_rb::schedule<CpuBe>("SMALL", mixfx::small_dims(64), 40, 1, true);
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
