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

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    int max_ctx = 32768;
    int chunk = 1024;             // prompt tokens a forward takes at most
    int staging = 48;             // expert staging slots a GPU (each one expert's blob)
    double reserve_gb = 1.0;      // VRAM a GPU keeps free beyond the plan
    int resident_cap = -1;        // routed experts kept in VRAM a layer at most (-1: as many as fit)
    bool verbose = false;
};

struct Stats {
    double load_s = 0;
    long staged = 0;              // expert blobs copied into staging slots
    long resident = 0, experts = 0;
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

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
