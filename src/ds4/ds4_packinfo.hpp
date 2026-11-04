// src/ds4/ds4_packinfo.hpp - `maya-ds4 --pack-info --pack <dir>`: a pack described from its ds4.txt and tensors.txt
// alone - no GPU, no weights read, the model's files only looked at (there, and long enough).  Host code only: the
// kernels' types come in through KernelTypes (ds4_main asks ds4_iq and the EXL3 kernels; tests pass their own).
#pragma once

#include <cstdio>
#include <string>

namespace strata::ds4 {

/// The types the engine's kernels take, asked of the kernels themselves rather than a copy of their lists: a ggml type
/// id's name ("?" or null: none), whether the routed experts' kernels take it, whether the dense GEMVs do, and whether
/// the EXL3 kernels take a bitrate (k2: twice the bits a weight, the mul1 codebook).
struct KernelTypes {
    const char* (*name)(int type) = nullptr;
    bool (*expert)(int type) = nullptr;
    bool (*dense)(int type) = nullptr;
    bool (*exl3_k2)(int k2) = nullptr;
};

/// The pack's report on `out`: its type map (layer x role) with an expert's and a token's bytes, where its parts go
/// (the GPUs, pinned RAM, the files) and how much of the expert profile's routing the most-routed experts take (the
/// tiers), the drafter's and the Engram's metadata, tensors no kernel takes, and every file it points at checked.
/// `profile`: the expert profile to read ("": <dir>/expert-profile.bin).  0: it can load (warnings may be printed);
/// 1: its description cannot be read; 2: it cannot load (a dead pack - files missing or short - or tensors the engine
/// does not read).
int pack_info(const std::string& dir, const std::string& profile, const KernelTypes& kt, std::FILE* out);

}  // namespace strata::ds4
