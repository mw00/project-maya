// include/strata/core/glm_maya_sync.hpp - the glm-dsa port W10: Maya v1.1.3's new features against the glm-dsa arch.
//
// Maya v1.1.3 (merged into glm-dsa in wave 10, docs/strata/engine/MAYA-SYNC.md) brought five features; this header is
// what the Maya-owned files ask (one line each, STRATA-EDITS.txt) to keep glm-dsa on the paths that serve it:
//
//   * EXL3 packs (exllamav3 trellis experts, tools/exl3_pack.py) exist for GLM-5.3-Flash only: a glm-dsa pack with an
//     exl3.txt is refused (glm_exl3_ok);
//   * the DFlash2 block drafter (--dflash) is a GLM-5.3-Flash drafter (and sm_80+ in Maya's setup): glm-dsa drafts with
//     its NextN block, so --dflash is refused there (glm_dflash_ok);
//   * --parallel's batched rows (glm_rows.cu, multi_rows) run the Flash token path's kernels (mHC, KDA, the k-pool
//     DSA): on glm-dsa the arena is not allocated, so rows_ok() is false and the server's pipelined multi path serves
//     the conversations one after the other (glm_rows_ok);
//   * the expert profile (--expert-profile, glm-dsa's STRP file) is arch-neutral in the engine (n_layers + 1 layers of
//     n_expert pairs), but the profile setup ships (data/expert-profile-glm.bin) is Flash's 46 x 288: a glm-dsa
//     model needs its own (tools/strata/glm_dsa_profile.py), and the refusal says so (glm_profile_hint);
//   * the disk-expert skip and MAYA_GLM_MTP_KEEP apply unchanged (the route kernel and fast_moe are arch-neutral).
//
// Every function is host-only and CUDA-free, so tests/glm_dsa/glm_maya_sync_test.cpp checks them on a CPU box.
#pragma once

#include "strata/core/glm_arch.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// An EXL3 pack (exl3.txt) on this arch: true for GLM-5.3-Flash; false + err on glm-dsa.
bool glm_exl3_ok(GlmArch a, std::string& err);

/// A DFlash2 drafter (--dflash / MAYA_GLM_DFLASH) on this arch: true for GLM-5.3-Flash; false + err on glm-dsa.
bool glm_dflash_ok(GlmArch a, std::string& err);

/// --parallel's batched rows (glm_rows.cu) on this arch: true for GLM-5.3-Flash (Maya), false on glm-dsa (the
/// pipelined multi path serves the conversations instead).
bool glm_rows_ok(GlmArch a);

/// The layers of an expert profile for a model of n_layers decoder layers: the decoder layers, then the NextN block
/// (Glm5Model::profile_layers, glm-dsa's layout).  GLM-5.3-Flash 46, GLM-5.3 744B 79.
inline int64_t glm_profile_layers(int64_t n_layers) { return n_layers + 1; }

/// Reads an expert profile's header (`STRP`, version, n_layers, n_expert): false + err when it is not one.
bool glm_profile_header(const std::string& path, uint32_t& version, int64_t& n_layers, int64_t& n_expert,
                        std::string& err);

/// What to append to a refused --expert-profile's message: on glm-dsa, the layout the model needs and how to make a
/// profile for it ("" on Flash: Maya's message as it is).
std::string glm_profile_hint(GlmArch a, int64_t profile_layers, int64_t n_expert);

}  // namespace strata::core
