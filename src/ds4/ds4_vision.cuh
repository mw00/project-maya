// src/ds4/ds4_vision.cuh - DeepSeek-V4.1's vision tower on one GPU (DeepSeek's inference/vision.py):
//   patches (14 x 14 RGB, scaled to [-1, 1]) -> Linear(588 -> 1024)
//   32 blocks: x += attn(rmsnorm(x)) (16 heads of 64, full bidirectional attention over the image, a 2D rotary on q
//              and k: the first half of each head's frequencies the patch row, the second its column);
//              x += w2(silu(gate) * up) with gate, up = w1(rmsnorm(x))
//   rmsnorm, then the aligner: each 3 x 3 block of patches (zero-padded at the right and bottom) as one 9216 row,
//   Linear(9216 -> 5120), GELU, Linear(5120 -> 5120): one LLM embedding row per block, in reading order.
// The BF16 weights live in pinned host memory and stream to the GPU a block at a time (two slots, the next block's
// copy under this one's work); the GEMMs are cuBLAS (BF16 in, FP32 accumulation), the residual stream FP32.  Nothing
// of the tower stays on the GPU between pictures: an encode is handed its room (room_bytes(), ~320 MB - the engine's
// idle prompt buffers, expert slots lent for it, or free memory), copies the small weights (norms, biases, the patch
// embedding) in, lays its work buffers out there and is done with it when it returns.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::ds4 {

struct VisionGeo {
    int layers = 0, dim = 1024, heads = 16, inter = 2816, patch = 14, merge = 3, out = 5120;
    float theta = 10000.0f, eps = 1e-6f;
    int max_tokens = 1024;   // an image's span in the prompt at most: n_h * (n_w + 1) + 2 tokens
};

class VisionTower {
public:
    /// read(name, shape, dst): the tensor's shape, and (dst not null) its BF16 data into dst; false: not there
    using Reader = std::function<bool(const std::string& name, std::vector<int64_t>& shape, void* dst)>;
    VisionTower() = default;
    VisionTower(const VisionTower&) = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    ~VisionTower();
    /// The weights checked and read into pinned host memory, the streams and the cuBLAS handle made on `device`:
    /// nothing of the GPU's memory is taken (room_bytes() says what an encode will be handed).
    bool load(const VisionGeo& g, int device, const Reader& read, std::string& err);
    /// Frees everything load took (a load that failed half way too); loaded() is false after it.
    void release();
    bool loaded() const { return dev_ >= 0; }
    int device() const { return dev_; }
    /// The GPU memory an encode lays itself out in: the largest picture's (every encode takes it whole, so a picture's
    /// buffers and its results are the same whatever room it is handed).
    size_t room_bytes() const { return room_; }
    /// Its GPU memory now: room_bytes() while a picture is encoded, else 0.
    size_t device_bytes() const { return cur_; }
    /// Its weights in pinned host memory.
    size_t host_bytes() const { return host_b_; }
    /// The patches an image may have (its aligner grid within max_tokens).
    int max_patches() const { return max_n_; }
    /// rgb: height x width x 3 bytes, both sides multiples of the patch size -> the aligner's rows, BF16
    /// [n_h * n_w][out] in reading order (n_h x n_w: the patch grid over 3, rounded up).  room: room_bytes() of
    /// device() memory, 256-byte aligned, the tower's until this returns (every copy and kernel of it done by then).
    bool encode(const uint8_t* rgb, int width, int height, uint8_t* room, std::vector<uint16_t>& rows, int& n_h,
                int& n_w, std::string& err);

private:
    struct Piece {             // a weight the encode streams in: host bits, its size
        const uint16_t* host = nullptr;
        size_t n = 0;
    };
    // a buffer's place in the room: its byte offset (256-aligned, in the order and sizes the tower's own allocations
    // had before it borrowed its room)
    struct Lay {
        size_t pe_w = 0, pe_b = 0, norm = 0, al1_b = 0, al2_b = 0, inv_freq = 0, pre = 0;   // pre: the small ones' end
        std::vector<size_t> n1, n2, bqkv, bo;
        size_t x = 0, xn = 0, qkv = 0, y = 0, o = 0, mid = 0, s = 0, p = 0, slot0 = 0, slot1 = 0, ws = 0;
        int cells = 0;   // (the largest aligner grid: where the aligner's buffers sit in qkv)
    };
    bool run(const uint8_t* rgb, int width, int height, uint8_t* room, std::vector<uint16_t>& rows, int& n_h,
             int& n_w, std::string& err);
    VisionGeo g_;
    int dev_ = -1, at_ = -1, max_n_ = 0, qb_ = 0, rb_ = 0;   // (at_: the device its streams are on)
    size_t room_ = 0, cur_ = 0, slot_n_ = 0, host_b_ = 0;
    Lay lay_;
    uint16_t* host_ = nullptr;          // pinned: every block's [wqkv, wo, w1, w2], then aligner w1, w2
    uint8_t* pre_host_ = nullptr;       // pinned: the small weights as the room's first lay_.pre bytes hold them
    size_t block_n_ = 0;                // a block's weights (BF16 values)
    size_t al1_off_ = 0, al2_off_ = 0;  // the aligner's weights in host_
    // in the room while an encode runs: the patch embedding, the norms and biases (copied in), the work buffers
    uint16_t *pe_w_ = nullptr, *pe_b_ = nullptr, *norm_ = nullptr, *al1_b_ = nullptr, *al2_b_ = nullptr;
    std::vector<uint16_t*> n1_, n2_, bqkv_, bo_;
    uint8_t* rgb_ = nullptr;
    uint16_t *patch_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *o_ = nullptr, *mid_ = nullptr, *p_ = nullptr,
             *un_ = nullptr, *a16_ = nullptr, *outb_ = nullptr;
    float *x_ = nullptr, *y_ = nullptr, *s_ = nullptr, *inv_freq_ = nullptr;
    uint16_t* slot_[2] = {nullptr, nullptr};
    void* blas_ = nullptr;              // cublasHandle_t
    uint8_t* ws_ = nullptr;
    void *st_ = nullptr, *cs_ = nullptr;   // cudaStream_t: the work, the weight copies
    void *copied_[2] = {nullptr, nullptr}, *used_[2] = {nullptr, nullptr};   // cudaEvent_t per slot
};

}  // namespace strata::ds4
