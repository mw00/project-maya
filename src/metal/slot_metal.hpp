#pragma once
// Build the snapshot's exact GGML state layout. Include after ggml declarations.
#include "slot_state.hpp"
#include <cstring>

namespace maya::metal {
template<class Model>
bool slot_metal_blocks(Model& m, int64_t pos, bool restoring, std::vector<SlotBlock>& out) {
    if (pos <= 0 || pos > m.n_ctx || m.g.idx_kpool <= 0 || m.g.n_embd <= 0) return false;
    if (!restoring && (m.snap_pos != pos || m.hx_snap != pos - 1 ||
                       m.hid_snap.size() != (size_t)m.g.n_embd)) return false;
    if (restoring) m.hid_snap.resize((size_t)m.g.n_embd);
    out.clear();
    auto tensor = [&](ggml_tensor* t, uint64_t n) {
        if (!t || n > ggml_nbytes(t)) return false;
        out.push_back({n,
            [t](uint64_t off, void* p, size_t len) { ggml_backend_tensor_get(t, p, (size_t)off, len); return true; },
            [t](uint64_t off, const void* p, size_t len) { ggml_backend_tensor_set(t, p, (size_t)off, len); return true; }});
        return true;
    };
    auto rows = [&](ggml_tensor* t, uint64_t n) {
        if (!t || (n && t->nb[1] > std::numeric_limits<uint64_t>::max() / n)) return false;
        return tensor(t, n * t->nb[1]);
    };
    for (auto& s : m.ls) {
        if (s.conv) {
            if (!s.conv_snap || !s.ssm || !s.ssm_snap ||
                ggml_nbytes(s.conv) != ggml_nbytes(s.conv_snap) ||
                ggml_nbytes(s.ssm) != ggml_nbytes(s.ssm_snap) ||
                !tensor(s.conv_snap, ggml_nbytes(s.conv_snap)) ||
                !tensor(s.ssm_snap, ggml_nbytes(s.ssm_snap))) return false;
        } else if (!rows(s.lat, (uint64_t)pos) || !rows(s.ikg, (uint64_t)pos) ||
                   !rows(s.pooled, (uint64_t)pos / (uint64_t)m.g.idx_kpool)) return false;
    }
    out.push_back({(uint64_t)m.hid_snap.size() * sizeof(float),
        [&m](uint64_t off, void* p, size_t n) { std::memcpy(p, (const char*)m.hid_snap.data() + off, n); return true; },
        [&m](uint64_t off, const void* p, size_t n) { std::memcpy((char*)m.hid_snap.data() + off, p, n); return true; }});
    return true;
}
}
