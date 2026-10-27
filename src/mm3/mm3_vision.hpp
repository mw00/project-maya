// maya-mm3: MiniMax-M3.1-Flash's vision tower - a picture's pixels -> the rows its image tokens carry
//
// The CLIP-style tower (a Conv3d patch embedding over 2 x 14 x 14 patches, pre-LayerNorm, 32 pre-LN encoder layers
// of 1280 with 16 heads of 80 and 3D RoPE over 78 of their dims, exact-GELU MLPs of 5120) and the projector (per
// patch 1280 -> 6144 -> 6144, then each 2 x 2 block's four rows together 24576 -> 6144 -> 6144), as transformers'
// MiniMaxM3VLVisionModel and the checkpoint's multi_modal_projector + patch_merge_mlp.  Weights BF16 as stored (the
// patch embedding F32), activations F32 (BF16 at the GEMMs' inputs, the reference's BF16 matmuls), attention in F32.
// A picture attends within itself only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mm3_pack.hpp"

namespace mm3 {

struct VisionGeo {
    int dim = 1280, heads = 16, hd = 80, ff = 5120, layers = 32;
    int patch = 14, tps = 2, merge = 2;
    int proj = 6144, out_dim = 6144;
    float eps = 1e-5f, theta = 10000.0f;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int max_pixels = 672 * 672, min_pixels = 4 * 28 * 28;
    int image_token = -1, image_start = -1, image_end = -1;
    int factor() const { return patch * merge; }
    int max_patches() const { return max_pixels / (patch * patch); }
};

class Vision {
public:
    Vision();
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    // whether the pack has a vision tower (mm3.txt's vision.* keys and its tensors)
    static bool in_pack(const Pack& pk);
    // its weights in host memory (the BF16 matrices where `pk` maps them: it must outlive the tower), pictures encoded
    // on device `dev` - nothing of the tower stays in its memory between pictures: an encode takes the room it needs
    // (room_bytes() at most), copies the weights in a stage at a time and gives the room back
    bool load(const Pack& pk, int dev, std::string& err);
    // the engine's room for an encode when cudaMalloc has none: lend(bytes) its device memory of at least that many
    // bytes (nullptr: none), give_back() after the encode (the engine's contents there are gone)
    void set_lender(std::function<void*(size_t)> lend, std::function<void()> give_back);
    const VisionGeo& geo() const;
    // a picture's image tokens: (height / 28) * (width / 28); 0 for sides that are not multiples of 28 or too many
    // pixels
    int tokens(int width, int height) const;
    // rgb: height x width x 3 bytes -> rows F32 [tokens][out_dim]
    bool encode(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    size_t bytes() const;        // its device memory now (0 between pictures)
    size_t host_bytes() const;   // its weights in host memory (or the pack's mapping)
    size_t room_bytes() const;   // the device memory an encode of the largest picture takes
    size_t last_bytes() const;   // ... and the last encode took
    int device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace mm3
