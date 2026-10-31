// glm_prefix.hpp -- P3 prefix cache: block hashing + registry (Task 3.1).
// A block ID is a 128-bit truncation of a SHA-256 over the block's token IDs
// followed by every DSA layer's page bytes in layer order, so two prompts
// share a block only when the tokens AND all 11 layers' bytes match.
#ifndef STRATA_GLM_PREFIX_HPP
#define STRATA_GLM_PREFIX_HPP

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata { namespace prefix {

struct BlockId {
  uint64_t hi = 0, lo = 0;
  bool operator==(const BlockId& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const BlockId& o) const { return !(*this == o); }
};

struct BlockIdHash {
  size_t operator()(const BlockId& b) const {
    return (size_t)(b.hi * 0x9e3779b97f4a7c15ULL ^ b.lo);
  }
};

// Plain SHA-256 (host): out gets the 32-byte big-endian digest.
void sha256(const void* data, size_t len, uint8_t out[32]);

// Block hash: SHA-256 over the block's token IDs then each layer's page
// bytes in layer order, truncated to 128 bits (the first two digest words).
BlockId hash_block(const int32_t* tokens, size_t n_tokens,
                   const void* const* layer_pages, size_t n_layers,
                   size_t page_bytes);

// Task 3.3: the boundary snapshot's carriage (D6).  A stateful tile's KDA
// runs + open key pool ride EITHER the latent pool's GPU slab pages (the
// default: the store's entry holds the page refs, so the pool's eviction
// sweep cannot take a refcounted page) OR the tile's host bytes (the
// fallback when the pool cannot serve).  Both mediums hold the SAME flat
// stream, in the SAME order (the runs in recr-layer order, then the pool
// rows per DSA layer, 2 rings each), so the apply reads one medium or the
// other with identical offsets.
struct SnapPages {
  std::vector<int32_t> pages;   // the slab pages, in stream order
  int64_t bytes = 0;            // the stream's total bytes
  bool live() const { return !pages.empty() && bytes > 0; }
};

// Registry: block ID -> refcount + slot (insertion order; the slot indexes the
// owning store's block vector). Refcount 0 means evictable; the pool's eviction
// (Task 3.2) only takes refcount-0 pages. The token-key index lets an intake
// find candidates from the prompt's tokens alone (the registrar hashed the full
// content - tokens + every layer's page bytes - at registration).
class BlockRegistry {
public:
  size_t acquire(const BlockId& id);   // creates at 0, returns the count after
  size_t release(const BlockId& id);   // decrements, floors at 0
  size_t refcount(const BlockId& id) const;
  size_t size() const { return slots_.size(); }
  size_t find_slot(const BlockId& id) const;  // insertion index, SIZE_MAX if absent
  // Task 3.2: the owning store's eviction drops the entry (the block left the store).
  void forget(const BlockId& id);
  // Task 3.2: register a block (refs -> 1, indexed under its tokens' key). The
  // canonical rows + the boundary snapshot live in the owning store's block.
  size_t register_block(const BlockId& id, uint64_t token_key);
  // The candidates a token-key lookup may hit: (id, slot) pairs, empty when the
  // key is unknown. A token-key collision only costs a boundary mismatch.
  std::vector<std::pair<BlockId, size_t>> candidates(uint64_t token_key) const;
  // The token hash of a page's tokens (FNV-1a over the IDs + the count): the
  // intake's lookup key.
  static uint64_t token_key(const int32_t* tokens, size_t n_tokens);
private:
  struct Entry { size_t refs = 0; size_t slot = 0; };
  std::unordered_map<BlockId, Entry, BlockIdHash> slots_;
  std::unordered_map<uint64_t, std::vector<BlockId>> by_tokens_;
};

}} // namespace strata::prefix

#endif
