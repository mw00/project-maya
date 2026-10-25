// src/mm3/mm3_tp.hpp - maya-mm3's tensor-parallel path (Options::tp > 0).
//
// The GPUs in groups of tp (the first GPU alone when the others make whole groups: it also holds the output head, the
// DSpark draft and the vision tower); each group a run of layers, every layer cut over its members - each member its
// query heads (n_head / tp) and their KV heads, its index heads, o_proj's columns of its heads, and 1 / tp of every
// FFN's width (the dense layers', the shared expert's and each routed expert's: gate / up by output, down by input).
// The residual stream stays on the host: per layer the host publishes the normed input, the members' partials come
// back (pinned memory, a counter each - no P2P on these GPUs), the host sums them in member order. It also routes the
// tokens (the router in F32 on the CPU) and runs the routed experts not in VRAM on the CPU lane - each as tp
// sub-experts, the members' slices - while the GPUs run theirs. The members' work for a whole forward is enqueued up
// front (a thread a GPU), every step after the first waiting on the host's counter.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mm3_model.hpp"

namespace mm3 {

class TpEngine {
public:
    TpEngine();
    ~TpEngine();
    TpEngine(const TpEngine&) = delete;
    TpEngine& operator=(const TpEngine&) = delete;

    bool load(const Options& o, std::string& err);
    bool forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err);
    int vocab() const;
    int max_ctx() const;
    int bos() const;
    int eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    std::vector<std::string> vram_lines() const;
    bool save_profile(const std::string& path, std::string& err) const;
    int draft_block() const;
    bool draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err);
    bool forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err);
    void set_progress(std::function<bool(int)> f);
    void set_ppl(bool on);
    const std::vector<double>& ppl_nll() const;
    int done() const;
    bool draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                       std::vector<int>& drafts, std::vector<float>& conf,
                       std::vector<std::vector<std::pair<int, float>>>& q, std::string& err);
    const VisionGeo* vision() const;
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    void set_images(std::vector<Model::Image> images);

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace mm3
