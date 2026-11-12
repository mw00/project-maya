// src/core/glm_state_bind.cpp - the glm-dsa port B4.1: Glm5Model's state arena laid out by the descriptor
// (include/strata/core/glm_state_layout.hpp).  load(), load_pack() and seq_layout() call this instead of carving by
// hand; the per-layer offset vectors Maya's code reads are copied from the descriptor, which the part keeps (lay_)
// for the snapshot, the slots and the glm-dsa kernels (dsa_kpe, the index keys).
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

// glm-dsa reaches state_carve through load_pack (fast: Fp16 / Int8, slow: SlowF32) since the B4.6 integration opened
// glm_forward_available; the mixer (src/core/glm_dsa_mixer.cu) reads lay_ for every cache it writes.
namespace strata::core {

bool Glm5Model::state_carve(int64_t ctx, GlmStateMode mode, int ik_ring, int mtp_il, std::string& err,
                            int64_t lat_rows) {
    GlmStateLayout L;
    if (!glm_state_carve(g_, l0_, l1_, mtp_il, ctx, mode, ik_ring, L, err, lat_rows)) return false;
    kda_S_ = L.kda_S;
    kda_conv_ = L.kda_conv;
    dsa_lat_ = L.dsa_lat;
    dsa_ik_ = L.dsa_ik;
    dsa_ig_ = L.dsa_ig;
    dsa_pool_ = L.dsa_pool;
    ik_ring_ = L.ik_ring;
    state_bytes_ = L.bytes;
    lay_ = std::move(L);
    return true;
}

}  // namespace strata::core
