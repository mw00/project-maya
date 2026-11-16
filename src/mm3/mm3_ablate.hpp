// src/mm3/mm3_ablate.hpp - maya-mm3's --ablate: a unit direction r ([dim]) projected out of every residual write,
// y <- y - scale * r (r . y) on each layer's attention output and on its FFN output (the dense MLP's, or the MoE's:
// the routed experts and the shared expert together) before it is added to h (k::project_out, mm3_model.cu's
// ablate()).  maya-ds4's --ablate (ds4_ablate.hpp) on this model; the use it is written for is abliteration, r a refusal
// direction.  The layer split only: --tp with --ablate loads the layer split instead (the tensor-parallel path sums
// the residual on the host).
//
// The direction comes from a llama-cvector-generator file (general.architecture "controlvector", direction.<layer>
// F32 [dim]: --ablate FILE --ablate-layer N).  A rank-1 abliteration adapter is maya-ds4's only: its tensors are
// DeepSeek's writers, refused here.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "../ds4/ds4_ablate.hpp"   // (its GGUF checks and the control vector's loader)

namespace mm3::ablate {

/// The direction of `path` (r unit length, `dim` values); `scale` 0: 1; `layer` 0: the file's only direction.N.
/// `what` describes the pick for the start-up log; `engine` names the program in the messages (maya's GLM engine
/// loads the same files through this function).
inline bool load(const std::string& path, int layer, int dim, float& scale, std::vector<float>& r, std::string& what,
                 std::string& err, const char* engine = "maya-mm3") {
    try {
        strata::GgufFile g(path);
        const strata::MetaValue* type = g.get("general.type");
        if (type != nullptr && type->s == "adapter") {
            err = path + ": an abliteration adapter (maya-ds4's: DeepSeek's tensors) - " + engine +
                  " takes a control vector (llama-cvector-generator's direction.N)";
            return false;
        }
        const strata::MetaValue* arch = g.get("general.architecture");
        if (arch == nullptr || arch->s != "controlvector") {
            err = path + ": not a control vector (general.architecture " + (arch ? "\"" + arch->s + "\"" : "missing") +
                  ") - " + std::string(engine) + " takes llama-cvector-generator's direction.N";
            return false;
        }
        // (GgufFile takes general.alignment as given - 0 as its default, 32 - and one that is not a power of two -
        // UINT64_MAX - wraps the data section's start back into the header, whose bytes would then read as the direction)
        if (const strata::MetaValue* a = g.get("general.alignment"))
            if (a->u != 0 && (a->u > (1ull << 20) || (a->u & (a->u - 1)) != 0)) {
                err = path + ": general.alignment " + std::to_string(a->u) + " is not a power of two up to 2^20";
                return false;
            }
        if (layer <= 0) {   // (a file of one direction: whichever direction.N it is)
            int only = 0, n = 0;
            for (const strata::TensorInfo& t : g.tensors())
                if (t.name.compare(0, 10, "direction.") == 0) {
                    ++n;
                    only = std::atoi(t.name.c_str() + 10);
                }
            if (n == 1 && only >= 1 && only <= 4096 && g.find("direction." + std::to_string(only)) != nullptr)
                layer = only;
        }
        if (layer > 0) {
            const std::string tname = "direction." + std::to_string(layer);
            const strata::TensorInfo* t = g.find(tname);
            if (t != nullptr && (t->shape.size() != 1 || t->shape[0] != (uint64_t) dim)) {
                std::string shape;
                for (uint64_t d : t->shape) shape += (shape.empty() ? "" : ", ") + std::to_string(d);
                err = path + ": " + tname + " is [" + shape + "] - a direction is one row of the model's hidden size, [" +
                      std::to_string(dim) + "]";
                return false;
            }
        }
    } catch (const std::exception& e) {   // (GgufFile refuses a malformed header by throwing)
        err = path + ": " + e.what();
        return false;
    }
    std::string w;
    if (!strata::ds4::ablate::load_direction(path, layer, 0, scale, r, w, err)) return false;
    if ((int) r.size() != dim) {
        err = path + ": a direction of " + std::to_string(r.size()) + " values, the model's hidden size is " +
              std::to_string(dim);
        return false;
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " direction.%d, scale %.3g", layer > 0 ? layer : 1, scale);
    what = "abliteration: " + path + buf + ", projected out of every layer's attention and FFN writes";
    return true;
}

}  // namespace mm3::ablate
