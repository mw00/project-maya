#pragma once
// The batched rows' state (glm_rows.cu, glm_mtp_rows.cu): each part's row activations and one row's binding.
#include "glm_fast_state.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

namespace glmrows {
constexpr int kQ8 = 36;   // bytes per q8_1 block
inline size_t up256(size_t b) { return (b + 255u) & ~(size_t) 255u; }
inline float* row_at(float* b, int64_t per, int i) { return b + (size_t) i * (size_t) per; }
inline uint8_t* rowq_at(uint8_t* b, int64_t n_in, int i) { return b + (size_t) i * (size_t) (n_in / 32) * kQ8; }
inline const float* image_row_in(const std::vector<int64_t>& pos, const std::vector<float>& rows, int64_t p,
                                 int n_embd) {
    const auto it = std::lower_bound(pos.begin(), pos.end(), p);
    if (it == pos.end() || *it != p) return nullptr;
    return rows.data() + (size_t) (it - pos.begin()) * (size_t) n_embd;
}
}  // namespace glmrows

// rows_max() rows of each activation, one row after the other (mv_nt reads and writes them so)
struct Glm5Model::RowsState {
    void* arena = nullptr;
    size_t bytes = 0;
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    uint8_t* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr;
    uint8_t* gated_q = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr;
    uint8_t *qr_q = nullptr, *attn_q = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *dg = nullptr, *du = nullptr;
    uint8_t* dhq = nullptr;
    float* head_x = nullptr;
    uint8_t* head_xq = nullptr;
    float* logits = nullptr;
    int* tok = nullptr;
    // STRATA_GLM_ROWS_BATCH: each row's own copies of the token path's KDA scratch (F->conv, F->g1, F->g2) and hc
    // partials / barrier counter (F->part, F->counter)
    float* kconv[3] = {nullptr, nullptr, nullptr};
    float *kg1 = nullptr, *kg2 = nullptr, *hc_part = nullptr;
    uint8_t* hc_ctr = nullptr;
    int64_t hc_part_n = 0;
    // STRATA_GLM_ROWS_ATTN: the DSA attention of all the rows in one launch (gf::mla_attend_rows) - each row's q_abs and
    // ctx (n_head x kv_lora), its own selection and streamed slot cells (n_sel_max), and the launch's partials
    float *aq = nullptr, *actx = nullptr, *apart = nullptr;
    int *acells = nullptr, *aslot = nullptr;
    int apart_tokens = 0;
    float* ascore = nullptr;   // ... and each row's indexer scores (ascore_n a row), the rows' selection in one launch
    int64_t ascore_n = 0;
    int* tok_h = nullptr;   // pinned
    cudaEvent_t ev_done = nullptr;
};
// one row's sequence on this part: state, position, caches, KV stream and image row
struct Glm5Model::RowSeq {
    int seq = 0;
    float* state = nullptr;
    float *Rc = nullptr, *Ro = nullptr;   // the residual pair (the token path's convention, per row)
    int64_t p = 0;
    int ring = 0;
    const std::vector<int64_t> *kda_S = nullptr, *kda_conv = nullptr, *dsa_lat = nullptr, *dsa_ik = nullptr,
                               *dsa_ig = nullptr, *dsa_pool = nullptr;
    const float* img = nullptr;
    const KvStream* kv = nullptr;   // KV streaming: that sequence's stream (null: not streamed)
    // the residual pair's base: the sequence's arena start (rows_bind), a verify window's row its own pair
    // (rows_bind_win, STRATA_GLM_MULTI_MTP) - the embedding goes in there, the last layer leaves the residual there
    float* res = nullptr;
    float* kin = nullptr;   // a verify window's row: its conversation's KDA input records (null: none kept)
    int kin_t = 0;          // ... the row's index in them
};

}  // namespace strata::core
