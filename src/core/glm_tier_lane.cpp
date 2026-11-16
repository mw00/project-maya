// src/core/glm_tier_lane.cpp - the glm-dsa port B8.4 node-split CPU lane arithmetic (see glm_tier_lane.hpp).
#include "glm_tier_lane.hpp"

namespace strata::core::glmtier {

bool half_fmt(const kc::NativeFmt& full, kc::NativeFmt& half, std::string& why) {
    if (full.n_ff <= 0 || full.n_ff % 2 != 0) {
        why = "n_ff is odd";
        return false;
    }
    if (!kc::native_fmt(full.gu_type, full.d_type, full.n_embd, full.n_ff / 2, half, why)) return false;
    if (half.gu_row != full.gu_row || 2 * half.d_row != full.d_row || 2 * half.h_bytes != full.h_bytes) {
        why = "the down activation or row does not halve on a block boundary";
        return false;
    }
    return true;
}

void half_gu_rows(const kc::NativeFmt& half, const SplitLayout& L, const uint8_t* half_blob, const void* act,
                  float* ff_half, int r0, int r1, float swiglu_limit) {
    const void* a[1] = {act};
    float* o[1] = {ff_half};
    kc::native_gu_rows_split(half, half_blob, half_blob + L.h_gu, a, 1, o, r0, r1, swiglu_limit);
}

void half_down_rows(const kc::NativeFmt& half, const SplitLayout& L, const uint8_t* half_blob, const void* hq_half,
                    float* y, int r0, int r1) {
    const void* h[1] = {hq_half};
    float* o[1] = {y};
    kc::native_down_rows_split(half, half_blob + L.h_down_off, h, 1, o, r0, r1);
}

void half_experts_serial(const kc::NativeFmt& half, const SplitLayout& L, int ne, const uint8_t* const* half_blob,
                         const float* w, const void* act, float swiglu_limit, float* y_h, float* ff, void* hq,
                         float* dn) {
    const int n_embd = (int) L.n_embd;
    for (int r = 0; r < n_embd; ++r) y_h[r] = 0.0f;
    for (int i = 0; i < ne; ++i) {
        half_gu_rows(half, L, half_blob[i], act, ff, 0, (int) L.H, swiglu_limit);
        kc::native_quant_h(half, ff, hq);
        half_down_rows(half, L, half_blob[i], hq, dn, 0, n_embd);
        for (int r = 0; r < n_embd; ++r) y_h[r] += w[i] * dn[r];
    }
}

}  // namespace strata::core::glmtier
