// include/maya/core/glp.hpp - GLP (GGUF Layer Projection) reader: the container contract of
// https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md, and the one apply this engine implements
// (mode=project, spec_version=1, hook_point=residual_stream_post_layer, rank=1).
//
// A GLP file is a GGUF v3 control vector whose glp.* metadata says HOW the direction applies:
//
//   project:  h <- h - alpha * (h . d_hat) d_hat     per layer, at the post-layer residual stream
//   add:      h <- h + d                             llama.cpp's stock control vector
//
// The operation travels with the data: a reader that does not implement glp.mode must refuse the file, never
// fall back to add.  Loading a projective direction into an additive consumer raises no error and produces wrong
// output - it pushes every token along the axis instead of removing it - so every unimplemented mode, hook,
// spec_version, rank and key here is a named refusal, not a fallback (GLP.md §Reader conformance).
//
// The reader is header-only and has no GPU or model dependency: it validates a file against the model's n_embd
// and n_layers and returns the dense tables the cvec kernel takes (maya/kernels/cvec.hpp).  `direction.N` applies
// at layer N, no offset; `direction.0` is rejected rather than silently skipped (a partial apply is invisible).
#pragma once

#include "maya/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace maya::core::glp {

/// The one hook this reader applies at: the accumulated residual after the layer's writes are folded in
/// (llama.cpp's `build_cvec()` site).  The hc fold site (`hc_fold_pre_engram`, spec_version 3) is a different
/// tensor and is refused; so are the two writer sites (`*_out_pre_residual`).
inline constexpr const char* kHookResidualPostLayer = "residual_stream_post_layer";
inline constexpr const char* kHookFoldPreEngram = "hc_fold_pre_engram";   // spec_version 3: the fold + companion writer pair

struct Vector {
    int64_t n_embd = 0;
    int64_t n_layers = 0;
    float alpha = 1.0f;          ///< glp.alpha_default, or the caller's override
    int rank = 1;
    std::vector<float> dir;      ///< n_layers * n_embd, unit rows; zero rows = unsteered
    std::vector<float> s;        ///< n_layers: alpha (0 = unsteered)
    std::vector<int> layers;     ///< the layer ids the file carries, ascending
    std::string mode, hook, derived_at, spec_version;
    int spec = 1;   ///< 1: residual_stream_post_layer; 3: hc_fold_pre_engram (the fold + writer pair)
    std::string base_model, base_revision, content_sha256;
    bool loaded() const { return !dir.empty(); }
};

struct LoadOptions {
    float alpha = -1.0f;   ///< < 0: the file's glp.alpha_default (or 1.0)
};

namespace detail {

inline bool is_number(MetaType t) {
    return t == MetaType::U8 || t == MetaType::I8 || t == MetaType::U16 || t == MetaType::I16 ||
           t == MetaType::U32 || t == MetaType::I32 || t == MetaType::F32 || t == MetaType::U64 ||
           t == MetaType::I64 || t == MetaType::F64;
}

inline bool meta_str(const GgufFile& g, const std::string& path, const char* key, std::string& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) return true;
    if (v->type != MetaType::STRING) {
        err = path + ": " + key + " must be a string";
        return false;
    }
    out = v->s;
    return true;
}

inline bool meta_num(const GgufFile& g, const std::string& path, const char* key, double& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) return true;
    if (!is_number(v->type)) {
        err = path + ": " + key + " must be a number";
        return false;
    }
    out = v->num();
    return true;
}

inline bool meta_bool(const GgufFile& g, const std::string& path, const char* key, bool& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (v == nullptr) return true;
    if (v->type != MetaType::BOOL) {
        err = path + ": " + key + " must be a bool";
        return false;
    }
    out = v->u != 0;
    return true;
}

/// "3,5,7" -> {3,5,7}; empty -> empty.  A malformed list is an error: the key is redundant with the tensor
/// names, so a disagreement is an exporter bug worth failing on.
inline bool parse_layers(const std::string& s, std::vector<int>& out, std::string& err) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (i >= s.size()) break;
        const size_t a = i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        if (i == a) {
            err = "glp.layer_ids_zero_based: expected comma-separated layer ids, got '" + s + "'";
            return false;
        }
        const long v = std::strtol(s.c_str() + a, nullptr, 10);
        if (v < 0 || v > 100000) {
            err = "glp.layer_ids_zero_based: layer " + std::to_string(v) + " is out of range";
            return false;
        }
        out.push_back((int) v);
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (i < s.size()) {
            if (s[i] != ',') {
                err = "glp.layer_ids_zero_based: expected ',' at '" + s.substr(i) + "'";
                return false;
            }
            ++i;
        }
    }
    return true;
}

/// `direction.N` / `direction.N.j` -> layer N, direction j (0 for the base name).  Strict: decimal digits, no
/// leading zeros (`direction.03` would otherwise alias `direction.3`), and no other suffix.
inline bool parse_tensor_name(const std::string& name, int& layer, int& j, std::string& err) {
    layer = -1;
    j = 0;
    if (name.compare(0, 10, "direction.") != 0) return false;
    size_t i = 10;
    const size_t a = i;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
    if (i == a) {
        err = name + ": expected direction.<layer>[.<j>]";
        return false;
    }
    if (name[a] == '0' && i - a > 1) {
        err = name + ": non-canonical layer id (leading zero)";
        return false;
    }
    const long l = std::strtol(name.c_str() + a, nullptr, 10);
    if (l > 100000) {
        err = name + ": layer id out of range";
        return false;
    }
    layer = (int) l;
    if (i == name.size()) return true;
    if (name[i] != '.') {
        err = name + ": expected direction.<layer>[.<j>]";
        return false;
    }
    ++i;
    const size_t b = i;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
    if (i == b || i != name.size()) {
        err = name + ": expected direction.<layer>[.<j>]";
        return false;
    }
    if (name[b] == '0' && i - b > 1) {
        err = name + ": non-canonical direction index (leading zero)";
        return false;
    }
    j = (int) std::strtol(name.c_str() + b, nullptr, 10);
    return true;
}

}  // namespace detail

/// Reads and validates a GLP file against the model's geometry.  Returns false with a named, actionable error
/// for every refusal.  `warnings` collects the non-fatal notes (a transferred vector, a non-unit direction).
inline bool load(const std::string& path, int64_t n_embd, int64_t n_layers, const LoadOptions& opt, Vector& out,
                 std::string& err, std::vector<std::string>& warnings) {
    out = Vector{};
    warnings.clear();
    if (n_embd < 1 || n_layers < 1) {
        err = "GLP: the model's geometry is empty (n_embd " + std::to_string(n_embd) + ", n_layers " +
              std::to_string(n_layers) + ")";
        return false;
    }
    try {
        GgufFile g(path);
        // general.architecture: the producer writes "controlvector"; absent is accepted (the glp.* keys are the
        // contract), any other value is refused.
        if (const MetaValue* arch = g.get("general.architecture")) {
            if (arch->type != MetaType::STRING || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is " +
                      (arch->type == MetaType::STRING ? "'" + arch->s + "'" : "not a string") +
                      ", expected 'controlvector')";
                return false;
            }
        }
        // ---- glp.mode: the one field a reader may not ignore.  Absence = add (legacy llama.cpp cvec).
        std::string mode = "add";
        if (!detail::meta_str(g, path, "glp.mode", mode, err)) return false;
        if (mode != "project") {
            if (g.get("glp.mode") == nullptr)
                err = path + ": no glp.mode: a legacy additive control vector (absence = add).  This reader "
                      "implements project only - load it with --control-vector (additive) or re-export it with "
                      "glp.mode=project";
            else
                err = path + ": glp.mode='" + mode + "' is not implemented (this reader implements 'project'; "
                      "'add' files use --control-vector)";
            return false;
        }
        // ---- glp.spec_version: required; 1 (residual_stream_post_layer) or 3 (hc_fold_pre_engram).
        // Version 3 files apply at the hc fold site: the projection at the fold consuming the layer's
        // FFN write AND the companion attention output writer (the same stack, the same alpha) - a
        // reader that arms only the fold under-doses and must refuse the file (GLP.md, "The
        // companion writer is part of the site"). This reader returns the tables for BOTH halves;
        // the engine applies them at its two writes.  Version 2 (rank-k) remains refused.
        if (g.get("glp.spec_version") == nullptr) {
            err = path + ": missing glp.spec_version (required; this reader implements 1 and 3)";
            return false;
        }
        double spec = 0.0;
        if (!detail::meta_num(g, path, "glp.spec_version", spec, err)) return false;
        if (spec != 1.0 && spec != 3.0) {
            err = path + ": glp.spec_version=" + std::to_string((long long) spec) + " is not implemented (this "
                  "reader implements 1 and 3" +
                  (spec == 2.0 ? "; rank-k and the alpha-scaling keys need a rank-k reader" : "") + ")";
            return false;
        }
        // ---- glp.hook_point: absent = residual_stream_post_layer (v1's canonical site).  A v3 file
        // must name hc_fold_pre_engram; a v1 file must not.
        std::string hook = kHookResidualPostLayer;
        if (!detail::meta_str(g, path, "glp.hook_point", hook, err)) return false;
        if (spec == 1.0 && hook != kHookResidualPostLayer) {
            err = path + ": glp.hook_point='" + hook + "' with glp.spec_version=1: this reader applies v1 files at '" +
                  kHookResidualPostLayer + "' only";
            if (hook == "hc_fold_pre_engram")
                err += "; the hc fold site needs glp.spec_version 3 and the fold + companion writer pair";
            return false;
        }
        if (spec == 3.0 && hook != kHookFoldPreEngram) {
            err = path + ": glp.hook_point='" + hook + "' with glp.spec_version=3: a v3 file must name '" +
                  kHookFoldPreEngram + "'";
            return false;
        }
        // ---- glp.rank: 1 only.
        double rank = 1.0;
        if (!detail::meta_num(g, path, "glp.rank", rank, err)) return false;
        if (rank != 1.0) {
            err = path + ": glp.rank=" + std::to_string((long long) rank) + " is not implemented (rank 1 only; a "
                  "rank-k file needs spec_version 2 and a rank-k reader)";
            return false;
        }
        // ---- version-2 keys are refused under version 1 (a file that carries them is malformed).
        for (const char* key : {"glp.dir_scales", "glp.layer_scales"}) {
            if (g.get(key) != nullptr) {
                err = path + ": " + key + " requires glp.spec_version 2 (this reader implements 1)";
                return false;
            }
        }
        bool orthonormal = false;
        if (!detail::meta_bool(g, path, "glp.orthonormal", orthonormal, err)) return false;
        (void) orthonormal;   // rank 1: informational; a rank-k file would have to be true and verified
        // ---- alpha: a separate parameter, never folded into the data (projection is quadratic in the norm).
        double alpha = 1.0;
        if (!detail::meta_num(g, path, "glp.alpha_default", alpha, err)) return false;
        if (!std::isfinite(alpha) || alpha < 0.0) {
            err = path + ": glp.alpha_default must be a finite number >= 0";
            return false;
        }
        if (opt.alpha >= 0.0f) alpha = opt.alpha;
        // ---- provenance (parsed, never gated).
        std::string derived_at;
        if (!detail::meta_str(g, path, "glp.derived_at", derived_at, err)) return false;
        if (!derived_at.empty() && derived_at != hook)
            warnings.push_back(path + ": glp.derived_at='" + derived_at + "' differs from glp.hook_point='" + hook +
                               "' (a transferred vector; applied at hook_point)");
        if (!detail::meta_str(g, path, "glp.base_model", out.base_model, err)) return false;
        if (!detail::meta_str(g, path, "glp.base_revision", out.base_revision, err)) return false;
        if (!detail::meta_str(g, path, "glp.content_sha256", out.content_sha256, err)) return false;
        // ---- tensors.
        std::vector<int> have;                 // layer ids in file order
        std::vector<std::vector<float>> rows;  // parallel to have: unit directions
        for (const TensorInfo& t : g.tensors()) {
            if (t.name.rfind("direction.", 0) != 0) continue;
            int layer = -1, j = 0;
            std::string e;
            if (!detail::parse_tensor_name(t.name, layer, j, e)) {
                err = path + ": " + e;
                return false;
            }
            if (j != 0) {
                err = path + ": " + t.name + " is a rank-k tensor (direction.<layer>.<j>), but glp.rank=1";
                return false;
            }
            if (layer == 0) {
                err = path + ": direction.0 is invalid here (layer 0 is producer-gated; this reader rejects it "
                      "rather than silently skipping it)";
                return false;
            }
            // The bound is the container's index space, not a physical-layer count: a looped
            // model's vector names execution steps 1..n_layers (GLP-48 ships direction.1..48
            // against 48 executions), so `N == n_layers` is valid and applied.  Past that is
            // past the model.
            if (layer > n_layers) {
                err = path + ": " + t.name + " is past the model's " + std::to_string(n_layers) +
                      " layers (direction.N applies at layer N)";
                return false;
            }
            if (t.type != 0) {
                err = path + ": " + t.name + " must be F32, is " + t.type_name();
                return false;
            }
            if (t.shape.size() != 1 || t.shape[0] != (uint64_t) n_embd) {
                std::string shape;
                for (uint64_t d : t.shape) shape += (shape.empty() ? "" : ", ") + std::to_string(d);
                err = path + ": " + t.name + " is [" + shape + "], a direction is one row of the model's hidden "
                      "size, [" + std::to_string(n_embd) + "]";
                return false;
            }
            const uint64_t bytes = (uint64_t) n_embd * 4;
            const uint64_t avail = g.file_size() > g.data_start() ? g.file_size() - g.data_start() : 0;
            if (t.offset > avail || bytes > avail - t.offset) {
                err = path + ": " + t.name + " runs past the end of the file";
                return false;
            }
            const float* src = reinterpret_cast<const float*>(g.tensor_data(t));
            double nrm = 0.0;
            for (int64_t k = 0; k < n_embd; ++k) nrm += (double) src[k] * src[k];
            nrm = std::sqrt(nrm);
            if (!(nrm > 0.0)) {
                err = path + ": " + t.name + " is all zeros";
                return false;
            }
            if (std::fabs(nrm - 1.0) > 1e-3)
                warnings.push_back(path + ": " + t.name + " is not unit (norm " + std::to_string(nrm) +
                                   "); normalized to d_hat");
            std::vector<float> row((size_t) n_embd);
            for (int64_t k = 0; k < n_embd; ++k) row[(size_t) k] = (float) (src[k] / nrm);
            have.push_back(layer);
            rows.push_back(std::move(row));
        }
        if (have.empty()) {
            err = path + ": no direction.N tensors";
            return false;
        }
        std::vector<int> sorted = have;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            err = path + ": two tensors for one layer";
            return false;
        }
        // ---- glp.layer_ids_zero_based: redundant with the names; a disagreement is an exporter bug.
        std::string ids;
        if (!detail::meta_str(g, path, "glp.layer_ids_zero_based", ids, err)) return false;
        if (!ids.empty()) {
            std::vector<int> listed;
            if (!detail::parse_layers(ids, listed, err)) {
                err = path + ": " + err;
                return false;
            }
            std::sort(listed.begin(), listed.end());
            if (listed != sorted) {
                std::string a, b;
                for (int l : listed) a += (a.empty() ? "" : ",") + std::to_string(l);
                for (int l : sorted) b += (b.empty() ? "" : ",") + std::to_string(l);
                err = path + ": glp.layer_ids_zero_based says [" + a + "] but the tensors are [" + b + "]";
                return false;
            }
        }
        // ---- the dense tables the cvec kernel takes.
        out.n_embd = n_embd;
        out.n_layers = n_layers;
        out.alpha = (float) alpha;
        out.rank = 1;
        out.mode = mode;
        out.hook = hook;
        out.spec = (int) spec;
        out.derived_at = derived_at;
        out.spec_version = std::to_string((long long) spec);
        out.layers = sorted;
        // One row past n_layers: direction.N == n_layers is valid (the container's index space),
        // so the dense tables carry a (n_layers + 1)-th row the cvec kernel indexes by layer id.
        out.dir.assign((size_t) ((n_layers + 1) * n_embd), 0.0f);
        out.s.assign((size_t) (n_layers + 1), 0.0f);
        for (size_t i = 0; i < have.size(); ++i) {
            const int l = have[i];
            std::copy(rows[i].begin(), rows[i].end(), out.dir.begin() + (size_t) l * n_embd);
            out.s[(size_t) l] = (float) alpha;
        }
        return true;
    } catch (const std::exception& e) {   // GgufFile refuses a malformed header by throwing
        err = path + ": " + e.what();
        return false;
    }
}

}  // namespace maya::core::glp
