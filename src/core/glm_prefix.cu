// glm_prefix.cu -- P3 prefix cache: block hashing + registry (Task 3.1).
// Host-side SHA-256 (the hash runs at intake, before the GPU work) plus the
// refcounted block registry the pool's eviction will consult in Task 3.2.
#include "strata/core/glm_prefix.hpp"
#include <cstring>
#include <cmath>

namespace strata { namespace prefix {

namespace {

inline uint32_t rotr32(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

// The 64 SHA-256 round constants: fractional parts of the cube roots of the
// first 64 primes, scaled by 2^32 (FIPS 180-4). Computed once at startup;
// double precision sits far inside the 32-bit grid for primes below 311.
struct KTable {
  uint32_t k[64];
  KTable() {
    int n = 0;
    for (int p = 2; n < 64; ++p) {
      bool prime = true;
      for (int d = 2; d * d <= p; ++d)
        if (p % d == 0) { prime = false; break; }
      if (!prime) continue;
      double c = std::cbrt((double)p);
      double f = c - (double)(long long)c;
      k[n++] = (uint32_t)(f * 4294967296.0);
    }
  }
};
const KTable g_k;

struct Sha256 {
  uint32_t st[8];
  uint8_t buf[64];
  size_t buflen = 0;
  uint64_t total = 0;
  Sha256() { reset(); }
  void reset() {
    st[0] = 0x6a09e667; st[1] = 0xbb67ae85; st[2] = 0x3c6ef372; st[3] = 0xa54ff53a;
    st[4] = 0x510e527f; st[5] = 0x9b05688c; st[6] = 0x1f83d9ab; st[7] = 0x5be0cd19;
    buflen = 0; total = 0;
  }
  void compress(const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[4*i + 0] << 24) | ((uint32_t)p[4*i + 1] << 16)
           | ((uint32_t)p[4*i + 2] << 8)  |  (uint32_t)p[4*i + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
      uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + g_k.k[i] + w[i];
      uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
  }
  void update(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    total += len;
    while (len) {
      size_t take = 64 - buflen; if (take > len) take = len;
      memcpy(buf + buflen, p, take);
      buflen += take; p += take; len -= take;
      if (buflen == 64) { compress(buf); buflen = 0; }
    }
  }
  void finalize(uint8_t out[32]) {
    uint64_t bits = total * 8;   // captured before the padding mutates total
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t z = 0;
    while (buflen != 56) update(&z, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; ++i) lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lenb, 8);
    for (int i = 0; i < 8; ++i) {
      out[4*i + 0] = (uint8_t)(st[i] >> 24);
      out[4*i + 1] = (uint8_t)(st[i] >> 16);
      out[4*i + 2] = (uint8_t)(st[i] >> 8);
      out[4*i + 3] = (uint8_t)(st[i]);
    }
  }
};

uint64_t be64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
  return v;
}

} // namespace

void sha256(const void* data, size_t len, uint8_t out[32]) {
  Sha256 s;
  if (len) s.update(data, len);
  s.finalize(out);
}

BlockId hash_block(const int32_t* tokens, size_t n_tokens,
                   const void* const* layer_pages, size_t n_layers,
                   size_t page_bytes) {
  Sha256 s;
  // frame the inputs so different layouts can never collide
  uint8_t hdr[8];
  hdr[0] = (uint8_t)n_tokens;  hdr[1] = (uint8_t)(n_tokens >> 8);
  hdr[2] = (uint8_t)n_layers;  hdr[3] = (uint8_t)(n_layers >> 8);
  hdr[4] = (uint8_t)page_bytes; hdr[5] = (uint8_t)(page_bytes >> 8);
  hdr[6] = (uint8_t)(page_bytes >> 16); hdr[7] = (uint8_t)(page_bytes >> 24);
  s.update(hdr, 8);
  if (n_tokens) s.update(tokens, n_tokens * sizeof(int32_t));
  for (size_t l = 0; l < n_layers; ++l)
    if (page_bytes) s.update(layer_pages[l], page_bytes);
  uint8_t d[32];
  s.finalize(d);
  BlockId id;                  // 128-bit truncation: first two digest words
  id.hi = be64(d);
  id.lo = be64(d + 8);
  return id;
}

size_t BlockRegistry::acquire(const BlockId& id) {
  auto it = slots_.find(id);
  if (it == slots_.end()) {
    Entry e; e.refs = 1; e.slot = slots_.size();
    slots_.emplace(id, e);
    return 1;
  }
  return ++it->second.refs;
}

size_t BlockRegistry::release(const BlockId& id) {
  auto it = slots_.find(id);
  if (it == slots_.end() || it->second.refs == 0) return 0;
  return --it->second.refs;
}

size_t BlockRegistry::refcount(const BlockId& id) const {
  auto it = slots_.find(id);
  return it == slots_.end() ? 0 : it->second.refs;
}

size_t BlockRegistry::find_slot(const BlockId& id) const {
  auto it = slots_.find(id);
  return it == slots_.end() ? (size_t)-1 : it->second.slot;
}

size_t BlockRegistry::register_block(const BlockId& id, uint64_t token_key) {
  auto it = slots_.find(id);
  if (it == slots_.end()) {
    Entry e; e.refs = 1; e.slot = slots_.size();
    slots_.emplace(id, e);
    by_tokens_[token_key].push_back(id);
    return 1;
  }
  return ++it->second.refs;
}

std::vector<std::pair<BlockId, size_t>> BlockRegistry::candidates(uint64_t token_key) const {
  std::vector<std::pair<BlockId, size_t>> out;
  auto it = by_tokens_.find(token_key);
  if (it == by_tokens_.end()) return out;
  for (const BlockId& b : it->second) {
    auto e = slots_.find(b);
    if (e != slots_.end()) out.push_back({b, e->second.slot});
  }
  return out;
}

void BlockRegistry::forget(const BlockId& id) {
  auto it = slots_.find(id);
  if (it == slots_.end()) return;
  for (auto& kv : by_tokens_) {
    std::vector<BlockId>& v = kv.second;
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] == id) { v.erase(v.begin() + (long) i); break; }
  }
  slots_.erase(it);
}

uint64_t BlockRegistry::token_key(const int32_t* tokens, size_t n_tokens) {
  // FNV-1a over the token IDs + the count: a lookup key only (the registrar's
  // full BlockId covered the bytes); a collision only costs a boundary mismatch.
  uint64_t h = 0xcbf29ce484222325ull;
  h ^= (uint64_t) n_tokens;
  h *= 0x100000001b3ull;
  for (size_t i = 0; i < n_tokens; ++i) {
    const uint8_t* p = (const uint8_t*) (tokens + i);
    for (int b = 0; b < 4; ++b) { h ^= p[b]; h *= 0x100000001b3ull; }
  }
  return h;
}

}} // namespace strata::prefix
