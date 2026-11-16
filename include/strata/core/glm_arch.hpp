// include/strata/core/glm_arch.hpp - the glm-dsa port: the GLM architecture dispatch and the geometry superset.
//
// Glm5Geometry (glm_model.hpp) was written for one model, GLM-5.3-Flash (`glm5-next`).  glm-dsa also serves the
// GLM-5.3 744B-A40B (`glm-dsa`), whose GGUF differs in ways the Flash reader would silently mis-read: MLA with a
// 64-dim rope tail (key_length_mla = nope + rope), a token-level lightning indexer present only on the "full"
// layers, a plain residual instead of mHC, no SwiGLU clamp.  This header carries what the two share:
//
//   * GlmArch and the extra fields (arch, qk_rope, rope_theta, idx_rope, idx_ln_eps, idx_kind, n_ctx_train) with
//     Flash-neutral defaults, mixed into Glm5Geometry as a CRTP base so every Maya reader stays bit-identical;
//   * the helpers the engine asks instead of assuming Flash (qk_head, kv_row, attn_scale, mhc, residual_floats,
//     idx_full, boundary_ok, n_moe_layers);
//   * the hooks Maya-owned files call (one line each, listed in STRATA-EDITS.txt): the arch dispatch in
//     Glm5Geometry::from_gguf, the tensor-shape check at load, the pack geometry read before the split bounds, the
//     per-arch NextN tensor lists, the prompt-path gate and the forward gate.
//
// The glm-dsa reader lives in src/core/glm_dsa_geometry.cpp, the shape check in src/core/glm_geometry_check.cpp.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
struct TensorInfo;
}  // namespace strata

namespace strata::core {

struct Glm5Geometry;

enum class GlmArch : int {
    Glm5Next = 0,   // GLM-5.3-Flash: KDA + NoPE MLA with a k-pool indexer, mHC (Maya's reader)
    GlmDsa = 1,     // GLM-5.3 744B: MLA + rope tail, token lightning indexer (full/shared layers), plain residual
};

// how the DSA indexer selects
enum class GlmIdxKind : int {
    KPool = 0,      // Flash: top pools of kpool positions (+ the open tail)
    Token = 1,      // glm-dsa: top-k individual tokens, shared layers reuse the previous full layer's selection
};

const char* glm_arch_name(GlmArch a);

/// The fields and helpers glm-dsa adds to Glm5Geometry.  D is Glm5Geometry (CRTP), so the helpers read its fields
/// without a second copy.  Every default leaves a Flash geometry exactly as Maya's reader builds it.
template <class D> struct GlmArchExt {
    GlmArch arch = GlmArch::Glm5Next;
    int qk_rope = 0;              // rope dims per attention head (MLA q_pe / k_pe); 0 = NoPE MLA
    double rope_theta = 0;        // RoPE base (glm-dsa: rope.freq_base, NORM / interleaved pairs)
    int idx_rope = 0;             // indexer q/k dims that are roped, the first idx_rope of idx_key ([rope|nope])
    float idx_ln_eps = 0;         // the indexer key LayerNorm eps (glm-dsa; HF 1e-6, llama.cpp 0)
    GlmIdxKind idx_kind = GlmIdxKind::KPool;
    int64_t n_ctx_train = 0;      // context_length (selects llama.cpp's indexer-types back-compat rule)

    /// One attention head's q/k width: nope + rope (Flash 256 + 0, glm-dsa 192 + 64).
    int qk_head() const { return d().qk_nope + qk_rope; }
    /// One latent cache row: kv_lora + the shared k_pe (Flash 512, glm-dsa 576).
    int kv_row() const { return d().kv_lora + qk_rope; }
    /// The attention softmax scale 1/sqrt(qk_head) (both 1/16; glm-dsa has no YaRN mscale).
    float attn_scale() const { return 1.0f / std::sqrt((float) qk_head()); }
    /// Multi-stream hyper-connection residual (Flash: 4 streams); false = one plain residual vector.
    bool mhc() const { return d().hc > 1; }
    /// Floats of the residual a part hands to the next one (hc * n_embd with mHC, n_embd plain).
    int residual_floats() const { return (mhc() ? d().hc : 1) * d().n_embd; }
    /// Layer il carries its own indexer (a "full" layer; layers past the trunk are the NextN block, always full).
    bool idx_full(int il) const {
        const auto& v = d().indexer_full;
        if (il >= 0 && il < (int) v.size()) return v[(size_t) il] != 0;
        return il >= d().n_layers;
    }
    /// A layer split may start a part at layer il: inside the trunk, and on glm-dsa a full-indexer layer (the
    /// shared layers reuse the previous full layer's top-k, which the per-token hop does not carry).
    bool boundary_ok(int il) const {
        return il >= 1 && il < d().n_layers && (arch != GlmArch::GlmDsa || idx_full(il));
    }
    /// The trunk's MoE layers (Flash 42, glm-dsa 75); the NextN block is not counted.
    int n_moe_layers() const { return d().n_layers - d().dense_lead; }

private:
    const D& d() const { return static_cast<const D&>(*this); }
};

// ---------------------------------------------------------------- hooks (src/core/glm_dsa_geometry.cpp)

/// Glm5Geometry::from_gguf's first line.  Reads general.architecture: glm5-next / glm5next -> 0 (Maya's body
/// continues); glm-dsa -> from_gguf_glm_dsa, 1 on success, -1 + err on refusal; absent -> a warning and 0 (the
/// legacy key probe); anything else -> -1 + err.
int glm_arch_route(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);

/// The glm-dsa geometry reader (prefix "glm-dsa."): required keys fail closed, every value the engine cannot serve
/// is refused with the key named.  Tensor-dependent facts (n_vocab without vocab_size) are filled by
/// glm_geometry_check.
bool from_gguf_glm_dsa(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);

/// The pack's own geometry (shard 1 named by native_experts.txt), read before anything is sized from it.
bool glm_pack_geometry(const std::string& pack_dir, Glm5Geometry& out, std::string& err);

/// The split boundaries of a glm-dsa pack must be full-indexer layers (Flash: always true, Maya's own check follows).
bool glm_split_bounds_ok(const Glm5Geometry& g, const std::vector<int>& bounds, std::string& err);

/// The glm-dsa forward does not exist yet: a glm-dsa geometry is refused here, after the geometry and the tensor
/// shapes were validated and before any weight is uploaded.  Flash: true.
bool glm_forward_available(const Glm5Geometry& g, std::string& err);

/// The NextN draft block's tensor lists: Maya's Flash lists come in, the arch's go out (glm-dsa: no indexer
/// compressor, the router kept F32).  glm_mtp_fallback: the tensor an optional one falls back to (glm-dsa's
/// nextn.shared_head_norm -> output_norm), nullptr when it has none.
void glm_mtp_lists(GlmArch a, std::vector<std::string>& raw, std::vector<std::string>& b16,
                   std::vector<std::string>& f32);
const char* glm_mtp_fallback(GlmArch a, const std::string& name);

/// glm-dsa: each trunk layer's dense VRAM cost (the non-routed tensors of index.txt, sized from the shards) for
/// the proportional split; leaves `cost` as it is for Flash or when the pack cannot be read.
void glm_dense_layer_cost(const Glm5Geometry& g, const std::string& pack_dir, std::vector<double>& cost);

// ---------------------------------------------------------------- shape check (src/core/glm_geometry_check.cpp)

using GlmFindTensor = std::function<const strata::TensorInfo*(const std::string&)>;

/// Validates every tensor of layers [l0, l1) (plus the globals, plus the NextN block when l1 is the trunk's end)
/// against the geometry: a wrong or missing tensor fails with its name and both shapes.  glm-dsa only (Flash: true,
/// unchanged); fills n_vocab from token_embd when the GGUF has no vocab_size.
bool glm_geometry_check(Glm5Geometry& g, const GlmFindTensor& find, int l0, int l1, std::string& err);

}  // namespace strata::core

// W10: Maya v1.1.3's features against the arch (EXL3, DFlash2, --parallel rows, the expert profile's hint), so every
// file that knows the geometry can ask them without another include hook
#include "strata/core/glm_maya_sync.hpp"
