// src/core/glm_mtp_state_model.cu - glm-dsa B15.3: the model side of the NextN (MTP) sequence state
// (include/strata/core/glm_mtp_state.hpp).
//
// Each (model part, sequence) that carries the NextN block owns a device carrier row and its snapshot copy (n_embd
// floats each, allocated on first use on the part's device) plus a glmmtp::CarrierState.  Glm5Model members:
//   mtp_carry_note(p, h, s)        after the trunk head's res_head (forward_fast, spec_tail, the multi-sequence
//                                  issue): row = h (D2D on the head's stream), pos = p.
//   mtp_carry_row(p, fallback, s)  fast_mtp's hnorm input: the row when it holds p, else the row zeroed (never a
//                                  stale row of another sequence, request or position).
//   mtp_carry_reset()              Glm5Model::reset (a sequence starts at pos 0): the row zeroed (#28333).
//   mtp_carry_snapshot(restore)    snapshot_save / snapshot_restore: the copy (valid when it holds h_{n-1}).
//   mtp_slot_put / mtp_slot_get    slot_save / slot_load / slot_load_prefix: the MTP1 trailer after the runs (#30141).
//   state_fp()                     the slot fingerprint with the NextN block's identity and the carrier bytes.
//   mtp_block_id()                 that identity (0 without a block on this chain).
//   mtp_carrier_stats()            counters over the chain's carriers.
// Everything is a no-op on Flash (glm5-next) and on parts without the NextN block: mtp_carry_row returns the
// fallback (FastState::head_x, Maya's behaviour), so Flash stays bit-identical.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_mtp_state.hpp"
#include "strata/core/glm_state_layout.hpp"

#include "glm_fast_state.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <istream>
#include <map>
#include <mutex>
#include <ostream>
#include <vector>

namespace strata::core {

namespace {

struct DevCarrier {
    glmmtp::CarrierState st;
    float* row = nullptr;
    float* snap = nullptr;
};

struct PartCarriers {
    std::map<int, DevCarrier> seq;
    glmmtp::CarrierStats extra;   // restored / trailer_bad (the rest is summed from the states)
    uint32_t block = 0;
    bool block_known = false;
    bool warned = false;
};

std::mutex g_mu;
std::map<const void*, PartCarriers> g_reg;

}  // namespace

static bool mtp_state_on(const Glm5Geometry& g, int mtp_il, const void* fast) {
    return g.arch == GlmArch::GlmDsa && mtp_il >= 0 && fast != nullptr;
}

// the bound sequence's carrier on this part (rows allocated and zeroed on first use, on this part's device)
static DevCarrier* carrier_of(const void* model, int seq, int n_embd, int dev, cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    DevCarrier& c = g_reg[model].seq[seq];
    if (c.row == nullptr) {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != dev) cudaSetDevice(dev);
        const size_t b = (size_t) n_embd * sizeof(float);
        if (cudaMalloc(&c.row, b) != cudaSuccess || cudaMalloc(&c.snap, b) != cudaSuccess) {
            cudaGetLastError();
            if (c.row) cudaFree(c.row);
            c.row = c.snap = nullptr;
            if (cur != dev) cudaSetDevice(cur);
            return nullptr;
        }
        cudaMemsetAsync(c.row, 0, b, s);
        cudaMemsetAsync(c.snap, 0, b, s);
        if (cur != dev) cudaSetDevice(cur);
    }
    return &c;
}

void Glm5Model::mtp_carry_note(int64_t p, const float* h, cudaStream_t s) {
    if (!mtp_state_on(g_, mtp_il_, fast_) || h == nullptr) return;
    DevCarrier* c = carrier_of(this, cur_seq_, g_.n_embd, dev_, s);
    if (c == nullptr) return;
    cudaMemcpyAsync(c->row, h, (size_t) g_.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
    c->st.note(p);
}

const float* Glm5Model::mtp_carry_row(int64_t p, const float* fallback, cudaStream_t s) {
    if (!mtp_state_on(g_, mtp_il_, fast_)) return fallback;
    DevCarrier* c = carrier_of(this, cur_seq_, g_.n_embd, dev_, s);
    if (c == nullptr) return fallback;
    const int64_t had = c->st.pos;
    if (!c->st.use(p)) {
        cudaMemsetAsync(c->row, 0, (size_t) g_.n_embd * sizeof(float), s);
        std::lock_guard<std::mutex> lk(g_mu);
        PartCarriers& pc = g_reg[this];
        if (had >= 0 && !pc.warned && getenv("MAYA_GLM_SPEC_LOG") != nullptr) {
            pc.warned = true;
            std::fprintf(stderr, "glm mtp: the draft at position %lld found the carrier of position %lld (sequence %d) "
                                 "- zeros instead (B15.3)\n", (long long) p, (long long) had, cur_seq_);
        }
    }
    return c->row;
}

void Glm5Model::mtp_carry_reset() {
    if (!mtp_state_on(g_, mtp_il_, fast_)) return;
    cudaStream_t s = fast_->cs;
    DevCarrier* c = carrier_of(this, cur_seq_, g_.n_embd, dev_, s);
    if (c == nullptr) return;
    c->st.begin();
    c->st.snap_pos = -1;
    cudaMemsetAsync(c->row, 0, (size_t) g_.n_embd * sizeof(float), s);
    cudaMemsetAsync(c->snap, 0, (size_t) g_.n_embd * sizeof(float), s);
}

void Glm5Model::mtp_carry_snapshot(bool restore) {
    if (!mtp_state_on(g_, mtp_il_, fast_)) return;
    cudaStream_t s = fast_->cs;
    DevCarrier* c = carrier_of(this, cur_seq_, g_.n_embd, dev_, s);
    if (c == nullptr) return;
    const size_t b = (size_t) g_.n_embd * sizeof(float);
    if (restore) {
        c->st.snapshot_restore();
        cudaMemcpyAsync(c->row, c->snap, b, cudaMemcpyDeviceToDevice, s);
    } else {
        c->st.snapshot_save(snap_pos_);
        if (c->st.snap_pos >= 0) cudaMemcpyAsync(c->snap, c->row, b, cudaMemcpyDeviceToDevice, s);
        else cudaMemsetAsync(c->snap, 0, b, s);
    }
}

void Glm5Model::mtp_carry_forget() {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_reg.find(this);
    if (it == g_reg.end()) return;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(dev_);
    for (auto& kv : it->second.seq) {
        if (kv.second.row) cudaFree(kv.second.row);
        if (kv.second.snap) cudaFree(kv.second.snap);
    }
    cudaSetDevice(cur);
    g_reg.erase(it);
}

uint32_t Glm5Model::mtp_block_id() const {
    const Glm5Model* t = nullptr;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (mtp_state_on(m->g_, m->mtp_il_, m->fast_)) t = m;
    if (t == nullptr) return 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_reg.find(t);
        if (it != g_reg.end() && it->second.block_known) return it->second.block;
    }
    const std::string P = "blk." + std::to_string(t->mtp_il_) + ".";
    std::vector<glmmtp::BlockTensor> ts;
    for (const auto& kv : t->ws_map_)
        if (kv.first.compare(0, P.size(), P) == 0)
            ts.push_back({kv.first, kv.second.type, (uint64_t) kv.second.n_in * (uint64_t) kv.second.n_out});
    for (const auto& kv : t->w16_)
        if (kv.first.compare(0, P.size(), P) == 0) ts.push_back({kv.first + "#bf16", 30, 0});
    if (t->fast_ != nullptr && (size_t) t->mtp_il_ < t->fast_->L.size()) {
        const auto& Ly = t->fast_->L[(size_t) t->mtp_il_];
        ts.push_back({P + "#experts", Ly.gu_type * 256 + Ly.d_type, (uint64_t) Ly.blob});
    }
    std::string src = "own";
    if (t->mtp_src_ >= 0) {
        const char* mg = getenv("MAYA_GLM_MTP_GGUF");
        std::error_code ec;
        const std::filesystem::path fp(mg != nullptr ? mg : "");
        src = "gguf:" + fp.filename().string() + ":" + std::to_string((unsigned long long) std::filesystem::file_size(fp, ec));
    }
    const uint32_t id = glmmtp::block_id(ts, src);
    std::lock_guard<std::mutex> lk(g_mu);
    PartCarriers& pc = g_reg[t];
    pc.block = id;
    pc.block_known = true;
    return id;
}

GlmStateFingerprint Glm5Model::state_fp() const {
    GlmStateFingerprint fp = glm_state_fingerprint(g_, lay_);
    if (g_.arch == GlmArch::GlmDsa) {
        const uint32_t id = mtp_block_id();
        fp[12] = glmmtp::fp_block_word(id);
        fp[13] = glmmtp::fp_carrier_word(true, id, g_.n_embd);
    }
    return fp;
}

glmmtp::CarrierStats Glm5Model::mtp_carrier_stats() const {
    glmmtp::CarrierStats r;
    std::lock_guard<std::mutex> lk(g_mu);
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const auto it = g_reg.find(m);
        if (it == g_reg.end()) continue;
        for (const auto& kv : it->second.seq) {
            const glmmtp::CarrierState& st = kv.second.st;
            r.notes += st.notes;
            r.reads += st.reads;
            r.zero_reads += st.zero_reads;
            r.stale_reads += st.stale_reads;
            r.resets += st.resets;
        }
        r.restored += it->second.extra.restored;
        r.trailer_bad += it->second.extra.trailer_bad;
    }
    return r;
}

bool Glm5Model::mtp_slot_put(std::ostream& f, std::string& err) {
    Glm5Model* t = nullptr;   // the chain's part with the NextN block
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (mtp_state_on(m->g_, m->mtp_il_, m->fast_)) t = m;
    if (t == nullptr) return true;   // no NextN block on this chain: no trailer (Flash never has one)
    const int E = g_.n_embd;
    std::vector<float> row((size_t) E, 0.0f);
    int64_t pos = -1;
    cudaSetDevice(t->dev_);
    DevCarrier* c = carrier_of(t, t->cur_seq_, E, t->dev_, t->fast_->cs);
    if (c != nullptr) {
        cudaStreamSynchronize(t->fast_->cs);
        // the slot holds the snapshot (positions 0 .. n-1): its copy, valid when it is h_{n-1}
        if (c->st.snap_pos >= 0 && c->st.snap_pos == snap_pos_ - 1 &&
            cudaMemcpy(row.data(), c->snap, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess)
            pos = c->st.snap_pos;
        else
            std::fill(row.begin(), row.end(), 0.0f);
    }
    cudaSetDevice(dev_);
    if (!glmmtp::write_trailer(f, E, pos, mtp_block_id(), row.data())) {
        err = "writing the MTP1 trailer failed";
        return false;
    }
    return true;
}

void Glm5Model::mtp_slot_get(std::istream& f, int64_t n, bool whole) {
    Glm5Model* t = nullptr;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (mtp_state_on(m->g_, m->mtp_il_, m->fast_)) t = m;
    if (t == nullptr) return;
    const int E = g_.n_embd;
    int64_t pos = -1;
    std::vector<float> row;
    std::string why;
    const glmmtp::Trailer r = glmmtp::read_trailer(f, E, n, mtp_block_id(), pos, row, why);
    cudaSetDevice(t->dev_);
    cudaStream_t s = t->fast_->cs;
    DevCarrier* c = carrier_of(t, t->cur_seq_, E, t->dev_, s);
    if (c != nullptr) {
        const size_t b = (size_t) E * sizeof(float);
        if (r == glmmtp::Trailer::Ok && whole && pos >= 0) {
            cudaStreamSynchronize(s);
            cudaMemcpy(c->row, row.data(), b, cudaMemcpyHostToDevice);
            cudaMemcpy(c->snap, row.data(), b, cudaMemcpyHostToDevice);
            c->st.pos = c->st.snap_pos = pos;
        } else {
            c->st.begin();
            c->st.snap_pos = -1;
            cudaMemsetAsync(c->row, 0, b, s);
            cudaMemsetAsync(c->snap, 0, b, s);
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        PartCarriers& pc = g_reg[t];
        if (r == glmmtp::Trailer::Ok && whole && pos >= 0) ++pc.extra.restored;
        if (r == glmmtp::Trailer::Bad) ++pc.extra.trailer_bad;
    }
    if (r == glmmtp::Trailer::Bad)
        std::fprintf(stderr, "glm mtp: %s - the slot loads, the draft carrier starts at zero (B15.3)\n", why.c_str());
    cudaSetDevice(dev_);
}

}  // namespace strata::core
