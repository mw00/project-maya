// src/core/glm_prompt_cache_model.cu - glm-dsa B14.14: the model side of the NVMe prompt cache
// (include/strata/core/glm_prompt_cache.hpp).
//
//   Glm5Model::slot_load_prefix(path, n_file, n_use, err): Maya's slot_load for a PREFIX of a saved state.  The file is
//   Maya's MAYALST3 slot file (slot_save) of n_file positions; this reads the first n_use positions of every cache run
//   and skips the rest.  Valid on glm-dsa only, where every run is per-position rows of an append-only cache (latents,
//   k_pe, the full layers' and the NextN block's index keys; GlmStateLayout::slot_runs) and there is no recurrent
//   snapshot or open pool; anything else is refused.  The header, fingerprint and per-part layout checks are
//   slot_load's.  Then the snapshot position is n_use and snapshot_restore() puts the model there.
//   Glm5Model::pcache_identity(): what a cache entry must match to be restored - the slot fingerprint words (geometry,
//   latent mode, k_pe / index-key dtype, the B14.11 rotation bit, latent row size), each part's layer range, the
//   NextN block, and the pack's files (names and sizes, as Maya's kept slots check).
//   B15.3: the entry file ends with slot_save's MTP1 carrier trailer; the fingerprint (state_fp) carries the NextN
//   block's identity (MAYA_GLM_MTP_GGUF or the model's own, by tensor types) and the carrier bytes, so an entry
//   written with another draft block is refused.  A prefix restore (n_use < n_file) zeroes the carrier.
//
// Portable runtime calls only (cudaMemcpy, cudaSetDevice, cudaStreamSynchronize): a HIP build compiles this file
// through hip_compat like every src/*.cu.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include "glm_fast_state.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace strata::core {

bool Glm5Model::slot_load_prefix(const std::string& path, int64_t n_file, int64_t n_use, std::string& err) {
    if (!lay_.glm_dsa()) {
        err = "a prefix restore needs glm-dsa's append-only caches";
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    const auto get = [&](void* p, size_t bytes) { return (bool) f.read((char*) p, (std::streamsize) bytes); };
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) halves.push_back(m);
    uint64_t magic = 0;
    uint32_t nh = 0;
    int64_t n = 0;
    GlmStateFingerprint fp{};
    if (!get(&magic, 8) || magic != kGlmSlotMagicV3 || !get(fp.data(), sizeof(fp)) ||
        fp != state_fp() || !get(&nh, 4) || !get(&n, 8) || nh != halves.size() || n != n_file ||
        n_use <= 0 || n_use > n || n_use > max_ctx_) {
        err = "not a prompt-cache state of this model";
        return false;
    }
    std::vector<float> buf;
    for (Glm5Model* m : halves) {
        if (m->fast_ == nullptr) {
            err = "no fast path";
            return false;
        }
        const Glm5Geometry& g = m->g_;
        const int64_t run = m->lay_.kda_run();
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const auto runs = m->lay_.slot_runs(n), use = m->lay_.slot_runs(n_use);
        int32_t hdr[4] = {};
        int64_t frun = 0;
        if (n_rec != 0 || m->lay_.snap_pool_floats() != 0 || runs.size() != use.size() || !get(hdr, sizeof(hdr)) ||
            !get(&frun, 8) || hdr[0] != m->l0_ || hdr[1] != m->l1_ || hdr[2] != n_rec ||
            hdr[3] != (int32_t) runs.size() || frun != run) {
            err = "the state's layout differs from this model's";
            return false;
        }
        for (size_t i = 0; i < runs.size(); ++i) {
            int64_t fl = 0;
            if (!get(&fl, 8) || fl != runs[i].second || use[i].first != runs[i].first || use[i].second > fl) {
                err = "the state's layout differs from this model's";
                return false;
            }
        }
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        for (size_t i = 0; i < runs.size(); ++i) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;   // floats
            for (int64_t at = 0; at < use[i].second; at += kChunk) {
                const int64_t c = std::min(kChunk, use[i].second - at);
                buf.resize((size_t) c);
                if (!get(buf.data(), (size_t) c * sizeof(float)) ||
                    cudaMemcpy(dsa_run_ptr(m->lay_, m->kv_, m->state_, use[i].first) + at, buf.data(),
                               (size_t) c * sizeof(float), cudaMemcpyDefault) != cudaSuccess) {   // (L1-6: or a host copy)
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the state did not read back";
                    return false;
                }
            }
            f.seekg((std::streamoff) ((runs[i].second - use[i].second) * (int64_t) sizeof(float)), std::ios::cur);
            if (!f) {
                cudaSetDevice(dev_);
                err = "the state file is short";
                return false;
            }
        }
        m->kv_stream_reset_maps();   // L1-6 KV streaming: the slots held the other conversation's rows
        m->snap_pos_ = n_use;
    }
    // B15.3: the MTP1 carrier trailer holds h_{n_file - 1}: it stands only for a whole-entry restore
    mtp_slot_get(f, n_file, n_use == n_file);
    cudaSetDevice(dev_);
    if (!snapshot_restore()) {
        err = "the snapshot did not restore";
        return false;
    }
    return true;
}

std::string Glm5Model::pcache_identity() const {
    std::ostringstream id;
    id << "strata-pcache-1";
#ifdef STRATA_VERSION
    id << " engine=" << STRATA_VERSION;
#endif
    id << " fp=";
    const GlmStateFingerprint fp = state_fp();   // B15.3: words 12-13 name the NextN block and its carrier
    for (size_t i = 0; i < fp.size(); ++i) id << (i ? "," : "") << fp[i];
    id << " parts=";
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) id << m->l0_ << "-" << m->l1_ << ";";
    id << " mtp=" << (has_mtp() ? 1 : 0);
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const char* fn : {"index.txt", "dense.bin", "native_experts.txt"})
        id << " " << fn << ":" << (unsigned long long) fs::file_size(fs::path(pack_dir_) / fn, ec);
    std::vector<std::string> shards;
    for (const auto& de : fs::directory_iterator(fs::path(pack_dir_).parent_path(), ec))
        if (de.path().extension() == ".gguf")
            shards.push_back(de.path().filename().string() + ":" + std::to_string(de.file_size(ec)));
    std::sort(shards.begin(), shards.end());
    for (const auto& s : shards) id << " " << s;
    return id.str();
}

}  // namespace strata::core
