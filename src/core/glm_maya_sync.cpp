// src/core/glm_maya_sync.cpp - the glm-dsa port W10: Maya v1.1.3's new features against the glm-dsa arch
// (include/strata/core/glm_maya_sync.hpp, docs/strata/engine/MAYA-SYNC.md).
#include "strata/core/glm_maya_sync.hpp"

#include <cstdio>
#include <cstring>

namespace strata::core {

bool glm_exl3_ok(GlmArch a, std::string& err) {
    if (a != GlmArch::GlmDsa) return true;
    err = "pack: exl3.txt in a glm-dsa pack - EXL3 (exllamav3) quants serve GLM-5.3-Flash only; GLM-5.3 744B runs "
          "from its GGUF (tools/strata/glm_dsa_pack.py)";
    return false;
}

bool glm_dflash_ok(GlmArch a, std::string& err) {
    if (a != GlmArch::GlmDsa) return true;
    err = "glm dflash: the DFlash2 drafter is made for GLM-5.3-Flash (its hidden-state taps and vocabulary); GLM-5.3 "
          "744B (glm-dsa) drafts with its NextN block - unset MAYA_GLM_DFLASH / --dflash";
    return false;
}

bool glm_rows_ok(GlmArch a) { return a != GlmArch::GlmDsa; }

bool glm_profile_header(const std::string& path, uint32_t& version, int64_t& n_layers, int64_t& n_expert,
                        std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    char magic[4] = {};
    uint32_t hdr[3] = {};
    const bool ok = std::fread(magic, 1, 4, f) == 4 && std::fread(hdr, 4, 3, f) == 3;
    std::fclose(f);
    if (!ok || std::memcmp(magic, "STRP", 4) != 0) {
        err = path + " is not an expert profile (no STRP header)";
        return false;
    }
    version = hdr[0];
    n_layers = (int64_t) hdr[1];
    n_expert = (int64_t) hdr[2];
    return true;
}

std::string glm_profile_hint(GlmArch a, int64_t profile_layers, int64_t n_expert) {
    if (a != GlmArch::GlmDsa) return std::string();
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  " (glm-dsa needs a %lld x %lld profile - the decoder layers, then the NextN block; the shipped "
                  "data/expert-profile-glm.bin is GLM-5.3-Flash's: make one with tools/strata/glm_dsa_profile.py "
                  "<pack> [traces / usage / expert_counts.txt])",
                  (long long) profile_layers, (long long) n_expert);
    return buf;
}

}  // namespace strata::core
