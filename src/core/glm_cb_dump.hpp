// src/core/glm_cb_dump.hpp - glm-dsa B7.4 / B7.6: the GLM_CB_DIR seam-dump policy of the glm-dsa paths (the slow
// reference forward glm_dsa_ref.cu and the decode mixer glm_dsa_mixer.cu).  Header-only, host code, no CUDA.
//
//   GLM_CB_DIR=<dir>      dump the seams as <dir>/<name>-<layer>.f32 (the existing convention; ids are int32 bytes)
//   GLM_CB_APPEND=1       append every token's values instead of overwriting, so a token-by-token run keeps every
//                         position (row i = the i-th forward of the process); selection seams (top_k-L) are padded
//                         with -1 to a fixed width (the indexer's top_k) so the rows stay aligned before p = top_k
//   GLM_CB_ONLY=<list>    only these seams: comma-separated seam names ("top_k") or exact dump names ("top_k-26");
//                         an 8K-token run that appends every seam of every layer would write hundreds of GB
// llama.cpp's side is tools/llm_logits_dump with LLM_CB_DIR, LLM_CB_APPEND=1 and LLM_CB_ONLY (same syntax), and
// tools/strata/seam_bisect.py reads both.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace glmcb {

inline const char* dir() {
    static const char* d = std::getenv("GLM_CB_DIR");
    return d != nullptr && d[0] != '\0' ? d : nullptr;
}

inline bool append() {
    static const bool a = [] {
        const char* v = std::getenv("GLM_CB_APPEND");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return a;
}

/// True when `name` ("<seam>-<layer>" or "<seam>") passes the GLM_CB_ONLY filter (always, without one).
inline bool wanted_in(const char* only, const std::string& name) {
    if (only == nullptr || only[0] == '\0') return true;
    const size_t dash = name.rfind('-');
    const bool layered = dash != std::string::npos && dash + 1 < name.size() &&
                         name.find_first_not_of("0123456789", dash + 1) == std::string::npos;
    const std::string seam = layered ? name.substr(0, dash) : name;
    const std::string list = only;
    size_t s = 0;
    while (s <= list.size()) {
        size_t e = list.find(',', s);
        if (e == std::string::npos) e = list.size();
        const std::string item = list.substr(s, e - s);
        if (!item.empty() && (item == name || item == seam)) return true;
        s = e + 1;
    }
    return false;
}

inline bool wanted(const std::string& name) {
    static const char* only = std::getenv("GLM_CB_ONLY");
    return wanted_in(only, name);
}

/// Writes (or appends) `bytes` bytes to <dir>/<name><ext>; in append mode a record shorter than `pad_bytes` is
/// padded with 0xFF bytes (int32 -1) up to it.
inline void write(const std::string& name, const void* data, size_t bytes, const char* ext = ".f32",
                  size_t pad_bytes = 0) {
    if (dir() == nullptr || !wanted(name)) return;
    std::FILE* f = std::fopen((std::string(dir()) + "/" + name + ext).c_str(), append() ? "ab" : "wb");
    if (f == nullptr) return;
    if (bytes > 0) std::fwrite(data, 1, bytes, f);
    if (append() && pad_bytes > bytes) {
        const std::vector<unsigned char> pad(pad_bytes - bytes, 0xFF);
        std::fwrite(pad.data(), 1, pad.size(), f);
    }
    std::fclose(f);
}

}  // namespace glmcb
