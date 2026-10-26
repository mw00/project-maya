// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx512_one.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <vector>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

// iq512_rows / iq256_rows
using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);

// fn's rows against ggml-cpu's own dot, bit for bit: synthetic rows of `type` (random codes, every block's f16 scale
// - and a K-quant's min scale - a small normal value) and Q8_K activations from ggml's quantizer, `rows` rows for each
// token count of `nts` (the multi-token default 2, 5 and 11 - 11: the kernels' groups of 8 / 4 and their remainders).
// 306 results: a form off in a few % of them (Q5_K's fused min term under MSVC: ~7%) cannot pass by chance - 39 did
// in ~6% of starts
bool rows_match_ggml(RowsFn fn, int type, std::initializer_list<int> nts = {2, 5, 11}, int rows = 17) {
    const ggml_type t = (ggml_type) type;
    const ggml_type at = traits(type)->vec_dot_type;
    const int n = 2048;   // eight 256-value blocks
    const size_t rb = ggml_row_size(t, n), bb = ggml_type_size(t), ab = ggml_row_size(at, n);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    const auto next = [&st] {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return st;
    };
    std::vector<uint8_t> w(rb * rows);
    for (auto& b : w) b = (uint8_t) next();
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        if (t == GGML_TYPE_IQ1_M) {
            // Its FP16 scale is embedded in four high nibbles, not at the block's start.
            for (int j = 0; j < 4; ++j) {
                uint16_t word;
                std::memcpy(&word, &w[o + 48 + 2 * j], sizeof word);
                word = (word & 0x0fff) | (((d >> (4 * j)) & 15) << 12);
                std::memcpy(&w[o + 48 + 2 * j], &word, sizeof word);
            }
        } else {
            std::memcpy(&w[o], &d, sizeof d);
        }
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(0.001f + 0.0005f * (float) (next() % 16));
            std::memcpy(&w[o + sizeof d], &m, sizeof m);
        }
    }
    // Check final IQ1_M delta rounding with subnormal accumulators too; other ggml builds may contract differently.
    const int shifts[] = {0, -120, -128};
    const int num_shifts = t == GGML_TYPE_IQ1_M ? 3 : 1;
    for (int shift_index = 0; shift_index < num_shifts; ++shift_index) for (int nt : nts) {
        const int shift = shifts[shift_index];
        std::vector<uint8_t> a(ab * (size_t) nt);
        std::vector<float> x((size_t) n), got((size_t) nt * rows), ref((size_t) nt * rows);
        std::vector<const void*> ap((size_t) nt);
        std::vector<float*> op((size_t) nt);
        for (int k = 0; k < nt; ++k) {
            for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
            traits(at)->from_float(x.data(), a.data() + ab * (size_t) k, n);
            if (shift != 0) {
                for (size_t offset = ab * (size_t) k; offset < ab * (size_t) (k + 1); offset += ggml_type_size(at)) {
                    float scale;
                    std::memcpy(&scale, a.data() + offset, sizeof scale);
                    scale = std::ldexp(scale, shift);
                    std::memcpy(a.data() + offset, &scale, sizeof scale);
                }
            }
            ap[(size_t) k] = a.data() + ab * (size_t) k;
            op[(size_t) k] = got.data() + (size_t) k * rows;
        }
        fn(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows);
        for (int k = 0; k < nt; ++k)
            for (int r = 0; r < rows; ++r)
                traits(type)->vec_dot(n, &ref[(size_t) k * rows + r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) k], 0, 1);
        if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) != 0) return false;
    }
    return true;
}

RowsFn pick_rows(int type) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_CPU_MT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return nullptr;
    init_once();
    // AVX-512 where cpu_avx512_ok() (VNNI and VBMI: Ice Lake / Zen 4 and later, which run 512-bit integer code at
    // full clock); on a Skylake-SP Xeon 6152 the AVX-2 rows were 7-13% faster (glm_cpu_bench, Maya-L, 2-16 tokens)
    const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    RowsFn fn = nullptr;
    void (*set_q5k)(bool) noexcept = nullptr;
    if (avx512 && iq512_supported(type)) {
        fn = iq512_rows;
        set_q5k = iq512_set_q5k_fused;
    } else if (avx2 && iq256_supported(type)) {
        fn = iq256_rows;
        set_q5k = iq256_set_q5k_fused;
    }
    if (fn == nullptr) return nullptr;
    if (rows_match_ggml(fn, type)) return fn;
    if (type == GGML_TYPE_Q5_K) {   // a ggml-cpu built without contraction (MSVC): its min term unfused
        set_q5k(false);
        if (rows_match_ggml(fn, type)) return fn;
        set_q5k(true);
    }
    std::fprintf(stderr, "native experts: the multi-token %s rows differ from ggml-cpu's dot in this build - "
                         "ggml's dots for them\n", ggml_type_name((ggml_type) type));
    return nullptr;
}

// the multi-token rows for a weight type when they are exact here, nullptr otherwise (picked and checked once)
RowsFn exact_rows(int type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return nullptr;
    static std::once_flag once[GGML_TYPE_COUNT];
    static RowsFn fn[GGML_TYPE_COUNT];
    std::call_once(once[type], [type] { fn[type] = pick_rows(type); });
    return fn[type];
}

// The one-token rows (a decode step's CPU experts) for the formats whose ggml-cpu AVX2 dot rebuilds every value from
// lookups: IQ3_XXS's four keven_signs loads per 32 values, IQ3_S's 9-bit indices built in vectors and read back from
// memory.  iq256_rows1 is ggml's dot with two changes - IQ3_XXS's signs built in registers, IQ3_S's
// indices in scalar registers - so it is bit-identical to ggml's dot (checked here once, 306 results).  On every AVX-2
// CPU, like ggml-cpu's own dot (it has no AVX-512 form for these). IQ1_M uses the same exact AVX2 row backend in
// both modes, sharing decode across tokens and avoiding PDEP. IQ2_XXS (Maya-S's gate/up) first tries the
// AVX-512 rows of iq_avx512_one.cpp (VBMI + BMI2: the grid as 2-bit codes in registers, the signs by pdep), checked
// the same way.  STRATA_GLM_CPU_1T=0 keeps ggml's dot, STRATA_NO_IQ512 the AVX-2 rows.
RowsFn pick_rows1(int type) {
    static const bool on = [] {
        const char* mt = std::getenv("STRATA_GLM_CPU_MT");
        const char* v = std::getenv("STRATA_GLM_CPU_1T");
        return (mt == nullptr || std::atoi(mt) != 0) && (v == nullptr || std::atoi(v) != 0);
    }();
    if (!on) return nullptr;
    if (std::getenv("STRATA_NO_IQ512") == nullptr && iq512_one_supported(type)) {
        init_once();
        if (rows_match_ggml(iq512_one_rows, type, {1}, 306)) return iq512_one_rows;
        std::fprintf(stderr, "native experts: the AVX-512 one-token %s rows differ from ggml-cpu's dot in this build - "
                             "not used\n", ggml_type_name((ggml_type) type));
    }
    if (!iq256_rows1_supported(type)) return nullptr;
    init_once();
    if (!cpu_avx2_ok() || std::getenv("STRATA_NO_IQ256") != nullptr) return nullptr;
    if (rows_match_ggml(iq256_rows1, type, {1}, 306)) return iq256_rows1;
    std::fprintf(stderr, "native experts: the one-token %s rows differ from ggml-cpu's dot in this build - "
                         "ggml's dot for them\n", ggml_type_name((ggml_type) type));
    return nullptr;
}

RowsFn exact_rows1(int type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return nullptr;
    static std::once_flag once[GGML_TYPE_COUNT];
    static RowsFn fn[GGML_TYPE_COUNT];
    std::call_once(once[type], [type] { fn[type] = pick_rows1(type); });
    return fn[type];
}

// the exact rows for nt tokens of a weight type (nt >= 1), nullptr: ggml-cpu's dot
RowsFn exact_rows_nt(int type, int nt) { return nt >= 2 ? exact_rows(type) : exact_rows1(type); }

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_multi_rows_exact(int type) { return exact_rows(type) != nullptr; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    if (gu_type < 0 || gu_type >= GGML_TYPE_COUNT || d_type < 0 || d_type >= GGML_TYPE_COUNT) {
        err = "native experts: type " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " is not a ggml type";
        return false;
    }
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;   // pre-AVX2: ggml's vec_dot
    if (nt >= 2 && iq512_supported(f.gu_type)) {   // one token: ggml-cpu is as fast or faster
        if (avx512) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_gu_rows_split(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base,
                          const void* const* act, int nt, float* const* ff, int r0, int r1,
                          float swiglu_limit) {
    const int n = (int) f.n_embd;
    const bool clamp = swiglu_limit != std::numeric_limits<float>::infinity();
    if (r1 > r0) {
        if (const RowsFn fn = exact_rows_nt(f.gu_type, nt)) {
            // every gate row of the range for all tokens (each row decoded once per group of tokens), then every up
            // row, then the same epilogue as below on the same values
            const int nr = r1 - r0;
            thread_local std::vector<float> buf;
            thread_local std::vector<float*> go, uo;
            buf.resize((size_t) 2 * nt * nr);
            go.resize((size_t) nt);
            uo.resize((size_t) nt);
            for (int t = 0; t < nt; ++t) {
                go[(size_t) t] = buf.data() + (size_t) t * nr;
                uo[(size_t) t] = buf.data() + (size_t) (nt + t) * nr;
            }
            fn(f.gu_type, gate_base + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, go.data(), 0, nr);
            fn(f.gu_type, up_base + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, uo.data(), 0, nr);
            for (int t = 0; t < nt; ++t) {
                for (int r = 0; r < nr; ++r) {
                    float g = go[(size_t) t][r], u = uo[(size_t) t][r];
                    if (clamp) {
                        g = std::fmin(g, swiglu_limit);
                        u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
                    }
                    ff[t][r0 + r] = (g / (1.f + std::exp(-g))) * u;
                }
            }
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_base + (size_t) r * f.gu_row;
        const uint8_t* ur = up_base + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            if (clamp) {   // GLM's clamped swiglu (glm_ffn's semantics), matching the device path
                g = std::fmin(g, swiglu_limit);
                u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows_split(const NativeFmt& f, const uint8_t* down_base, const void* const* hq,
                            int nt, float* const* out, int r0, int r1) {
    if (r1 > r0) {
        if (const RowsFn fn = exact_rows_nt(f.d_type, nt)) {
            thread_local std::vector<float*> o;
            o.resize((size_t) nt);
            for (int t = 0; t < nt; ++t) o[(size_t) t] = out[t] + r0;
            fn(f.d_type, down_base + (size_t) r0 * f.d_row, f.d_row, (int) f.n_ff, hq, nt, o.data(), 0, r1 - r0);
            return;
        }
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down_base + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    if (nt >= 2 && f.d_type == 20 && iq4nl_mt) {
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
