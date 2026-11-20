// src/hy4/hy4_vision.hpp - maya-hy4: Hy4-flash's vision tower (the mmproj GGUF) - a picture's pixels -> the rows its
// image tokens carry
//
// A SigLIP-shaped encoder (a 16 x 16 patch embedding with a bias, the learned 128 x 128 position grid resampled
// bilinearly to the picture's patch grid, 27 pre-LayerNorm layers of 1152 with 16 heads of 72 - biased q, k, v and out
// projections, no rotary positions - and tanh-GELU MLPs of 4304), then the projector: an RMSNorm, each 2 x 2 block of
// patches through a strided 2 x 2 convolution (4608), an exact GELU, a 1 x 1 convolution to the model's 3072, and the
// learned newline row after each row of blocks.  As the reference's clip graph (hy_v4_flash): a picture of h x w
// pixels gives (h / 32) * (w / 32 + 1) rows, every one an image token's.  Weights BF16 (the file's F32 convolutions are
// BF16 values), activations F32 with BF16 at the GEMMs' inputs, attention in F32 (TF32 products).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hy4 {

struct Pack;

struct VisionGeo {
    int dim = 1152, heads = 16, hd = 72, ff = 4304, layers = 27;
    int patch = 16, merge = 2, grid = 128, out_dim = 3072;
    float eps = 1e-5f;
    float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    int min_pixels = 4096, max_pixels = 4194304;
    int image_token = -1, image_start = -1, image_end = -1;
    int factor() const { return patch * merge; }
    // a picture's rows (its image tokens): (height / 32) * (width / 32 + 1); 0 for sides that are not multiples of 32
    // or more pixels than max_pixels
    int tokens(int width, int height) const;
};

class Vision {
public:
    Vision();
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    // whether the pack has a vision tower (hy4.txt's vision.* keys, the mmproj file's tensors in tensors.txt)
    static bool in_pack(const Pack& pk);
    // its geometry from hy4.txt (false: one maya-hy4 does not run)
    static bool geometry(const Pack& pk, VisionGeo& g, std::string& err);
    // the device memory an encode of the largest picture takes (nothing is kept free for it: the engine lends it)
    static size_t room_for(const VisionGeo& g);
    // its weights in host memory (the BF16 matrices where `pk` maps them: it must outlive the tower), pictures encoded
    // on device `dev` - nothing of the tower stays in its memory between pictures: an encode takes the room it needs
    // (room_for() at most), copies the weights in a layer at a time and gives the room back
    bool load(const Pack& pk, int dev, std::string& err);
    const VisionGeo& geo() const;
    // the device memory an encode of a picture of these sides takes (0: sides the tower does not read)
    size_t room_of(int width, int height) const;
    // rgb: height x width x 3 bytes -> rows F32 [tokens][out_dim].  `room`: `room_bytes` of device `dev`'s memory
    // lent for the encode (idle memory of the caller's: its contents are lost) - used where the picture's room fits
    // in it, else (and with none) the encode allocates its own and frees it
    bool encode(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err,
                uint8_t* room = nullptr, size_t room_bytes = 0);
    size_t host_bytes() const;   // its weights in host memory (or the pack's mapping)
    size_t room_bytes() const;   // room_for(geo())
    size_t last_bytes() const;   // the device memory the last encode took
    bool last_lent() const;      // ... in the room it was lent (false: its own allocation)
    int device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace hy4
