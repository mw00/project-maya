// src/core/glm_pack_identity.cu - the identity of the bytes a pack load actually mapped.
//
//   Glm5Model::pack_content_identity(): a digest of every file pack_shard_mmap mapped for this part (the GGUF shards,
//   MAYA_GLM_MTP_GGUF's draft block, an EXL3 pack's safetensors), read through the mapping itself rather than by
//   reopening paths - a path re-read could see another file than the load did.  Slots and prompt-cache entries carry
//   it, so a retained state written over one set of weights is refused by a later load of different bytes under the
//   same names and sizes.
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

namespace {

// SHA-256, FIPS 180-4.  Written out here rather than taken from a library so the identity is the same bytes on every
// platform this engine builds for (no OpenSSL or BCrypt dependency in the build).
struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t length = 0;              // bytes fed so far
    unsigned char block[64]{};
    size_t used = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const unsigned char* p) {
        static constexpr uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t) p[4 * i] << 24) | ((uint32_t) p[4 * i + 1] << 16) | ((uint32_t) p[4 * i + 2] << 8) |
                   (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void feed(const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        length += n;
        while (n > 0) {
            const size_t take = n < sizeof block - used ? n : sizeof block - used;
            std::memcpy(block + used, p, take);
            used += take; p += take; n -= take;
            if (used == sizeof block) {
                compress(block);
                used = 0;
            }
        }
    }

    std::string hex() {
        const uint64_t bits = length * 8;
        const unsigned char pad = 0x80;
        feed(&pad, 1);
        const unsigned char zero = 0;
        while (used != 56) feed(&zero, 1);
        unsigned char tail[8];
        for (int i = 0; i < 8; ++i) tail[i] = (unsigned char) (bits >> (56 - 8 * i));
        feed(tail, sizeof tail);
        static constexpr char digits[] = "0123456789abcdef";
        std::string out(64, '0');
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) {
                const unsigned char byte = (unsigned char) (h[i] >> (24 - 8 * j));
                out[(size_t) (i * 8 + 2 * j)] = digits[byte >> 4];
                out[(size_t) (i * 8 + 2 * j + 1)] = digits[byte & 15];
            }
        return out;
    }
};

static void feed_pack_metadata(Sha256& s, const std::string& bytes) {
    s.feed("metadata:", 9);
    const uint64_t size = bytes.size();
    s.feed(&size, sizeof size);
    s.feed(bytes.data(), bytes.size());
}

}  // namespace

std::string PackShardDigests::digest(const std::string& file_id, uint64_t size, const void* base) {
    if (file_id.empty() || base == nullptr || size > SIZE_MAX) return {};
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_pair(file_id, size);
    const auto existing = entries.find(key);
    if (existing != entries.end()) return existing->second;
    Sha256 shard;
    shard.feed(base, (size_t) size);
    ++hash_count;
    return entries.emplace(key, shard.hex()).first->second;
}

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
