// src/ling/ling_vision.hpp - Ling-3.2-air (bailing_moe_v3_vl) qwen3_moe_vit + linear_proj CUDA vision tower
//
// Runs directly from a BF16 mmproj GGUF (created by tools/ling/convert_mmproj.py) with ZERO permanent GPU VRAM
// footprint: weights stay memory-mapped on host and stream one block at a time into caller-lent VRAM (e.g. tail of
// the expert cache pool) or a temporary allocation freed right after encoding.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ling {

struct VisionGeo {
    int dim = 1152;
    int heads = 16;
    int hd = 72;
    int ff = 4304;
    int layers = 27;
    int patch = 16;
    int merge = 2;
    int temporal_patch = 2;
    int grid = 48;            // sqrt(2304)
    int out_dim = 2560;
    float eps = 1e-6f;
    float rope_theta = 10000.0f;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int min_pixels = 4096;
    int max_pixels = 4194304;
    int image_token = 157157;
    int image_start = 157158;
    int image_end = 157159;
    int mrope_section[3] = {8, 12, 12};

    int factor() const { return patch * merge; }
    // Number of merged visual tokens emitted for an image of (width, height): (height/32) * (width/32)
    int tokens(int width, int height) const {
        const int f = factor();
        if (width <= 0 || height <= 0 || width % f != 0 || height % f != 0) return 0;
        const int64_t px = (int64_t) width * (int64_t) height;
        if (px < min_pixels || px > max_pixels) return 0;
        return (height / f) * (width / f);
    }
};

class VisionTower {
public:
    VisionTower();
    ~VisionTower();
    VisionTower(const VisionTower&) = delete;
    VisionTower& operator=(const VisionTower&) = delete;

    // Load mmproj GGUF metadata & mmap tensors on host; encodes will run on CUDA device `dev`.
    bool load(const std::string& mmproj_path, int dev, std::string& err);
    bool loaded() const;
    const VisionGeo& geo() const;

    // Device scratch bytes required to encode an image of (width, height) (or worst-case max_pixels if 0,0).
    size_t room_of(int width, int height) const;
    size_t max_room_bytes() const;

    // Encode RGB uint8 image [height][width][3] -> F32 embeddings [tokens][out_dim] on host.
    // If `lent_room` has at least `room_of(width, height)` bytes on `dev`, uses it with zero cudaMalloc.
    bool encode(const uint8_t* rgb, int width, int height, std::vector<float>& out_rows, std::string& err,
                uint8_t* lent_room = nullptr, size_t lent_bytes = 0);

    size_t last_bytes() const;
    bool last_lent() const;
    int device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace ling
