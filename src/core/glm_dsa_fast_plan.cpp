// src/core/glm_dsa_fast_plan.cpp - the host-only half of the glm-dsa fast binder (glm-dsa B2.4); see the header.
#include "glm_dsa_fast_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::core::glmdsa {

const char* src_name(Src s) {
    switch (s) {
        case Src::Native: return "native";
        case Src::Bf16: return "bf16";
        case Src::F32: return "f32";
        default: return "missing";
    }
}

namespace {
Pick make(Src s, int type, const void* w) {
    Pick p;
    p.src = s;
    p.type = type;
    p.w = w;
    return p;
}
}  // namespace

Pick pick_mv(const Tables& t, const std::string& name) {
    int type = 0;
    const void* q = nullptr;
    const void* f = nullptr;
    const bool has_native = t.native && t.native(name, type, q, f);
    if (has_native && type != kTypeF32 && q != nullptr && t.mv_ok && t.mv_ok(type)) return make(Src::Native, type, q);
    if (t.bf16)
        if (const void* b = t.bf16(name)) return make(Src::Bf16, kTypeBF16, b);
    if (has_native && type == kTypeF32 && f != nullptr) return make(Src::F32, kTypeF32, f);
    if (t.f32)
        if (const void* p = t.f32(name)) return make(Src::F32, kTypeF32, p);
    Pick miss;
    miss.why = has_native && type != kTypeF32 ? "type " + std::to_string(type) + " has no mv kernel and no BF16 copy"
                                              : "not in the pack";
    return miss;
}

Pick pick_headwise(const Tables& t, const std::string& name) {
    int type = 0;
    const void* q = nullptr;
    const void* f = nullptr;
    if (t.native && t.native(name, type, q, f) && type == kTypeQ8_0 && q != nullptr) return make(Src::Native, type, q);
    return pick_float(t, name);
}

Pick pick_float(const Tables& t, const std::string& name) {
    if (t.bf16)
        if (const void* b = t.bf16(name)) return make(Src::Bf16, kTypeBF16, b);
    int type = 0;
    const void* q = nullptr;
    const void* f = nullptr;
    const bool has_native = t.native && t.native(name, type, q, f);
    if (has_native && type == kTypeF32 && f != nullptr) return make(Src::F32, kTypeF32, f);
    if (t.f32)
        if (const void* p = t.f32(name)) return make(Src::F32, kTypeF32, p);
    Pick miss;
    miss.why = has_native ? "only native rows (type " + std::to_string(type) + "); a BF16 or F32 copy is required"
                          : "not in the pack";
    return miss;
}

Pick pick_router(const Tables& t, const std::string& name) {
    if (t.f32)
        if (const void* p = t.f32(name)) return make(Src::F32, kTypeF32, p);
    if (t.bf16)
        if (const void* b = t.bf16(name)) return make(Src::Bf16, kTypeBF16, b);
    Pick miss;
    miss.why = "neither an F32 nor a BF16 copy";
    return miss;
}

const char* buf_name(int b) {
    static const char* const n[kNumBufs] = {"q",      "kv_raw", "q_abs", "score", "cells",      "qr_raw", "qr",
                                            "qr_q",   "ik_raw", "iq",    "iw",    "attn_out",   "attn_out_q", "act"};
    return b >= 0 && b < kNumBufs ? n[b] : "?";
}

bool arena_plan(const ArenaDims& d, ArenaPlan& out, std::string& err) {
    out = ArenaPlan{};
    const int64_t qh = d.qk_nope + d.qk_rope, kv_row = d.kv_lora + d.qk_rope, hv = d.n_head * d.v_head;
    if (d.n_embd <= 0 || d.n_head <= 0 || d.q_lora <= 0 || d.kv_lora <= 0 || d.qk_nope <= 0 || d.qk_rope < 0 ||
        d.v_head <= 0 || d.idx_heads <= 0 || d.idx_key <= 0 || d.idx_top_k <= 0 || d.max_ctx <= 0) {
        err = "glm-dsa arena: a non-positive dimension";
        return false;
    }
    if (d.n_embd % kQ8Block != 0 || d.q_lora % kQ8Block != 0 || hv % kQ8Block != 0 || d.n_ff_dense % kQ8Block != 0 ||
        d.n_ff_sh % kQ8Block != 0) {
        err = "glm-dsa arena: a q8_1 width is not a multiple of 32";
        return false;
    }
    // the widest input a q8_1-only call site hands to gf::mv: the dense and shared downs, eh_proj's [enorm | hnorm],
    // attn_output's and q_b's inputs
    out.act_floats = std::max({d.n_ff_dense, d.n_ff_sh, 2 * d.n_embd, hv, d.q_lora, d.n_embd});
    const size_t f = sizeof(float);
    out.bytes[kQ] = (size_t) (d.n_head * qh) * f;
    out.bytes[kKvRaw] = (size_t) kv_row * f;
    out.bytes[kQAbs] = (size_t) (d.n_head * kv_row) * f;
    out.bytes[kScore] = (size_t) d.max_ctx * f;
    out.bytes[kCells] = (size_t) d.idx_top_k * sizeof(int32_t);
    out.bytes[kQrRaw] = (size_t) d.q_lora * f;
    out.bytes[kQr] = (size_t) d.q_lora * f;
    out.bytes[kQrQ] = q8_1_bytes(d.q_lora);
    out.bytes[kIkRaw] = (size_t) d.idx_key * f;
    out.bytes[kIq] = (size_t) (d.idx_heads * d.idx_key) * f;
    out.bytes[kIw] = (size_t) d.idx_heads * f;
    out.bytes[kAttnOut] = (size_t) hv * f;
    out.bytes[kAttnOutQ] = q8_1_bytes(hv);
    out.bytes[kAct] = (size_t) kMaxMvJobs * (size_t) out.act_floats * f;
    size_t at = 0;
    for (int b = 0; b < kNumBufs; ++b) {
        out.off[b] = at;
        at += (out.bytes[b] + 255u) & ~(size_t) 255u;
    }
    out.total = at;
    return true;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const int exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    float v;
    if (exp == 0) v = std::ldexp((float) man, -24);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = std::ldexp((float) (man | 0x400), exp - 25);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits |= sign;
    std::memcpy(&v, &bits, 4);
    return v;
}

uint16_t float_to_half(float f) {
    // round to nearest even, saturating to the largest finite value only through the caller's clamp
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const float a = std::fabs(f);
    if (std::isnan(f)) return (uint16_t) (sign | 0x7e00u);
    if (a >= 65520.0f) return (uint16_t) (sign | 0x7c00u);
    if (a < 6.103515625e-05f) {   // subnormal: units of 2^-24
        const float q = std::nearbyint(a * 16777216.0f);
        return (uint16_t) (sign | (uint32_t) q);
    }
    int e;
    const float m = std::frexp(a, &e);   // a = m * 2^e, m in [0.5, 1)
    float mant = std::nearbyint(m * 2048.0f);   // 11 significant bits
    if (mant >= 2048.0f) {
        mant /= 2.0f;
        ++e;
    }
    const uint32_t exp = (uint32_t) (e - 1 + 15);
    return (uint16_t) (sign | (exp << 10) | ((uint32_t) mant & 0x3ffu));
}

void q8_1_to_f32_host(const void* xq, int64_t n, float* y) {
    const uint8_t* b = (const uint8_t*) xq;
    for (int64_t i = 0; i < n / kQ8Block; ++i) {
        uint16_t dh;
        std::memcpy(&dh, b + (size_t) i * kQ8Bytes, 2);
        const float d = half_to_float(dh);
        const int8_t* qs = (const int8_t*) (b + (size_t) i * kQ8Bytes + 4);
        for (int k = 0; k < kQ8Block; ++k) y[i * kQ8Block + k] = d * (float) qs[k];
    }
}

void q8_1_quantize_host(const float* x, int64_t n, void* xq) {
    uint8_t* b = (uint8_t*) xq;
    for (int64_t i = 0; i < n / kQ8Block; ++i) {
        const float* v = x + i * kQ8Block;
        float amax = 0.0f, sum = 0.0f;
        for (int k = 0; k < kQ8Block; ++k) {
            amax = std::max(amax, std::fabs(v[k]));
            sum += v[k];
        }
        const float d = std::min(amax / 127.0f, 65504.0f);
        const uint16_t dh = float_to_half(d), sh = float_to_half(std::max(-65504.0f, std::min(65504.0f, sum)));
        std::memcpy(b + (size_t) i * kQ8Bytes, &dh, 2);
        std::memcpy(b + (size_t) i * kQ8Bytes + 2, &sh, 2);
        int8_t* qs = (int8_t*) (b + (size_t) i * kQ8Bytes + 4);
        for (int k = 0; k < kQ8Block; ++k)
            qs[k] = amax == 0.0f ? 0 : (int8_t) std::max(-127.0f, std::min(127.0f, std::round(v[k] / d)));
    }
}

}  // namespace strata::core::glmdsa
