// src/core/glm_pack_identity.cu - the identity of the bytes a pack load actually mapped.
//
//   Glm5Model::pack_content_identity(): a digest of every file pack_shard_mmap mapped for this part (the GGUF shards,
//   MAYA_GLM_MTP_GGUF's draft block, an EXL3 pack's safetensors), read through the mapping itself rather than by
//   reopening paths - a path re-read could see another file than the load did.  Slots and prompt-cache entries carry
//   it, so a retained state written over one set of weights is refused by a later load of different bytes under the
//   same names and sizes. The loader calls this only when persistent caching is enabled; default and RAM-only loads
//   do not read every shard for identity.
//   Glm5Model::pack_identity() = pack_content_identity() + what the state layout holds (geometry, mode, k_pe and
//   index-key dtype, rotation, latent row size, the NextN block and its carrier): the same words state_fp() carries,
//   read from lay_ directly because this file is built without CUDA and so without state_fp's translation unit.
//
// Portable runtime calls only: a HIP build compiles this file through hip_compat like every src/*.cu.  The digest is
// SHA-256 (FIPS 180-4) over each shard's mapping, in pack_shards_ order with its path and size; it is not a signature
// and says nothing about who wrote the file.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace strata::core {

// Every part's identity: this one's layer range and its own mapped bytes, then each split part's in order.
std::string Glm5Model::pack_identity_of(const Glm5Model& part) {
    std::ostringstream id;
    id << "strata-pack-1 engine=";
#ifdef STRATA_VERSION
    id << STRATA_VERSION;
#else
    id << 0;
#endif
    id << " fp=";
    const Glm5Geometry& g = part.g_;
    const GlmStateLayout& lay = part.lay_;
    int n_full = 0;
    for (int il = 0; il < g.n_layers; ++il) n_full += g.arch == GlmArch::GlmDsa ? (int) g.idx_full(il) : 0;
    const int32_t format = kGlmFpFormat;   // random routing marks a slot through state_fp(); not a weight property
    const GlmStateFingerprint fp{format,
                                 (int32_t) g.arch,
                                 g.n_layers,
                                 g.n_embd,
                                 g.kv_lora,
                                 g.qk_rope,
                                 g.idx_key,
                                 n_full,
                                 (int32_t) lay.mode,
                                 (int32_t) (lay.kpe_pp * 4),
                                 lay.glm_dsa() ? lay.ik_bytes | (lay.ik_had ? kGlmFpIdxHad : 0) : 0,
                                 lay.glm_dsa() ? (int32_t) (lay.lat_pp * 4) : 0,
                                 lay.mtp_il >= 0 ? 1 : 0,   // the NextN block: its content is in the digest
                                 (int32_t) (lay.mtp_il >= 0 ? g.n_embd * 4 : 0)};
    for (int32_t word : fp) id << word << ',';
    id << " parts=";
    for (const Glm5Model* m = &part; m != nullptr; m = m->split_next_.get()) id << m->l0_ << "-" << m->l1_ << ';';
    id << " content=";
    for (const Glm5Model* m = &part; m != nullptr; m = m->split_next_.get()) {
        id << m->l0_ << ":" << m->pack_content_digest_ << ";";
    }
    return id.str();
}

std::string Glm5Model::pack_content_identity(PackShardDigests& digests) const {
    Sha256 s;
    for (const Shard& sh : pack_shards_) {
        if (sh.base == nullptr || sh.size > SIZE_MAX) return {};   // never accept a partial or path-based digest
        s.feed("shard:", 6);
        s.feed(sh.path.data(), sh.path.size());
        s.feed(":", 1);
        const uint64_t size = sh.size;
        s.feed(&size, sizeof size);
        const std::string bytes = digests.digest(sh.file_id, sh.size, sh.base);
        if (bytes.empty()) return {};   // never accept an invalid mapping or an empty digest
        s.feed(bytes.data(), bytes.size());
    }
    feed_pack_metadata(s, pack_metadata_bytes_);
    // Bind the parsed routing rather than reopening metadata paths: a replacement of native_experts.txt
    // or exl3.txt after parsing must not give retained state from another expert layout an identity match.
    std::ostringstream routing;
    routing << "routing-v1:" << pack_layers_.size() << ':';
    for (const NativeLayer& layer : pack_layers_) {
        routing << layer.layer << ',' << layer.fmt.gu_type << ',' << layer.fmt.d_type << ','
                << layer.gate_shard << ',' << layer.up_shard << ',' << layer.down_shard << ','
                << layer.gate_off << ',' << layer.up_off << ',' << layer.down_off << ',' << layer.x_cb << ';';
        for (int r = 0; r < 3; ++r)
            routing << layer.x_k2[r] << ',' << layer.x_k[r] << ',' << layer.x_n[r] << ';';
        routing << layer.x_shard.size() << ':';
        for (int index : layer.x_shard) routing << index << ',';
        routing << layer.x_off.size() << ':';
        for (uint64_t offset : layer.x_off) routing << offset << ',';
    }
    routing << "matrices:" << exl3_mats_.size() << ':';
    for (const auto& entry : exl3_mats_) {
        const auto& mat = entry.second;
        routing << entry.first.size() << ':' << entry.first << ',' << mat.k << ',' << mat.n << ','
                << mat.k2 << ',' << mat.cb << ',' << mat.ld << ',' << mat.fmt << ',' << mat.gg_type << ';';
    }
    const std::string parsed = routing.str();
    s.feed(parsed.data(), parsed.size());
    return s.hex();
}

std::string Glm5Model::pack_identity() const { return Glm5Model::pack_identity_of(*this); }

}  // namespace strata::core
