// src/mm3/mm3_model.hpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on the GPUs.
//
// The layers split over the GPUs by their memory, each GPU holding its layers' dense matrices, KV cache (BF16
// [ctx][4][128] K and V) and index-key cache (BF16 [ctx][128]), then as many of its layers' routed experts as fit
// (in VRAM by expert, a blob each); the others are read from the model's files into a GPU's staging slots when a
// token routes to them.  One forward (the prompt path) runs any number of tokens at once, a decode step included.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mm3_pack.hpp"

namespace mm3 {

struct VisionGeo;
class TpEngine;

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    std::vector<int> layer_split; // the first layer of each GPU's share after the first (empty: by their free memory)
    int max_ctx = 32768;
    int chunk = 1024;             // prompt tokens a forward takes at most
    int staging = 48;             // expert staging slots a GPU (each one expert's blob)
    double reserve_gb = -1;       // VRAM a GPU keeps free beyond the plan (< 0: 1.0, the tensor-parallel path 0.3 -
                                  // it allocates all it needs at the load)
    int resident_cap = -1;        // routed experts kept in VRAM a layer at most (-1: as many as fit)
    std::string profile;          // an expert profile (uint32 [layers][experts] routings): the most used stay in VRAM
    bool pin_pool = true;         // the experts not in VRAM assembled once in pinned host memory (a miss: one DMA)
    bool fast_decode = true;      // one token: the decode GEMVs and the token's fused experts (else the prompt path)
    int cpu_threads = 43;         // the CPU lane's workers besides the caller (-1: no CPU lane, misses are staged)
    bool dspark = false;          // load the DSpark draft (the pack's dspark/) on the head's GPU
    bool vision = false;          // load the vision tower (the pack's vision.*) on gpus[vision_gpu], before the experts
    int vision_gpu = 0;
    bool promote = true;          // the staging slots a cache: an expert the CPU lane keeps computing copied into one
                                  // (from the pinned pool, on a copy stream of its own) while the model runs
    int promote_lag = 2;          // a promoted expert's slot is its own this many forwards after its copy began, the
                                  // copy waited for then (1: the next forward - plain decode then waits ~5 ms a token)
    int tp = 0;                   // > 0: the tensor-parallel path (mm3_tp.hpp) in groups of tp GPUs (1, 2 or 4)
    double cache_gb = 1.5;        // (tp) each GPU's expert cache (promote's slots), taken from its experts in VRAM
    bool verbose = false;
};

struct Stats {
    double load_s = 0;
    long staged = 0;              // expert blobs copied into staging slots
    long resident = 0, experts = 0;
    double pinned_gb = 0;         // the pinned pool of the experts not in VRAM
    long profiled = 0;            // routings counted (save_profile writes them)
    long cpu_experts = 0;         // routed experts the CPU lane computed
    long promoted = 0;            // experts copied into a staging slot because they kept coming back
    double t_promo = 0;           // a forward's start waiting for those copies
    double t_sync = 0, t_lane = 0, t_token = 0;   // decode: waiting on the routing, the CPU lane, all (seconds)
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    /// Runs ids at positions pos0 .. pos0 + n - 1 (the cache holds every position before pos0), in chunks of at most
    /// Options::chunk; logits_out (if given) gets the logits of the last `want` positions of this call, [want][vocab]
    /// F32, row-major.
    bool forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err);
    int vocab() const;
    int max_ctx() const;
    int bos() const;
    int eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    /// The routings counted so far (with the loaded profile's, if any) as a profile file.
    bool save_profile(const std::string& path, std::string& err) const;
    /// The DSpark draft (Options::dspark): its block (the most tokens a proposal has; 0 without the draft), and up to
    /// n tokens proposed after `anchor` at position pos (the cache holds the positions before it) with their
    /// confidences.
    int draft_block() const;
    bool draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err);
    /// The vision tower (Options::vision): its geometry (null without it), a picture's rows (rgb height x width x 3,
    /// sides multiples of 28 -> F32 [tokens][hidden]), and the pictures the next forwards' image tokens take their
    /// rows from (position start .. start + rows / hidden - 1; set_images({}) after the prompt).
    struct Image {
        int start = 0;
        std::vector<float> rows;
    };
    const VisionGeo* vision() const;
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    void set_images(std::vector<Image> images);

private:
    struct Impl;
    Impl* p;
    TpEngine* tpe = nullptr;   // (Options::tp > 0: every call goes to it)
};

}  // namespace mm3
