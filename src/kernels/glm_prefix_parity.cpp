// glm_prefix_parity.cpp --selftest
// Task 3.1 (TDD): block hashing + registry parity checks.
// identical prompts -> identical block IDs (page-aligned, D3 granularity)
// one token changed at pos k -> block IDs equal below k/P, different from k/P on
// two layers' bytes swapped -> different ID (hash covers all 11 layers' page bytes)
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/glm_prefix.hpp"

using strata::prefix::BlockId;
using strata::prefix::BlockRegistry;
using strata::prefix::hash_block;
using strata::prefix::sha256;

static int g_failures = 0;

static void require(const char* name, bool ok, int line) {
  printf("%s %s (line %d)\n", ok ? "PASS" : "FAIL", name, line);
  if (!ok) ++g_failures;
}

#define REQUIRE(name, cond, line) require((name), (cond), (line))

static const size_t P = 256;          // D3 granularity: tokens per page
static const size_t N_LAYERS = 11;    // the 11 DSA layers
static const size_t TOK_BYTES = 64;   // per-token per-layer page bytes (test layout)

struct Prompt {
  std::vector<int32_t> tokens;
  std::vector<std::vector<uint8_t>> pages;  // N_LAYERS x n_blocks*P*TOK_BYTES
};

static Prompt make_prompt(size_t n_tokens, uint64_t seed) {
  Prompt pr;
  pr.tokens.resize(n_tokens);
  for (size_t i = 0; i < n_tokens; ++i)
    pr.tokens[i] = (int32_t)((i * 2654435761u + (uint32_t)seed) ^ (uint32_t)(i >> 7));
  size_t n_blocks = (n_tokens + P - 1) / P;
  pr.pages.resize(N_LAYERS);
  for (size_t l = 0; l < N_LAYERS; ++l) {
    pr.pages[l].resize(n_blocks * P * TOK_BYTES);
    for (size_t i = 0; i < pr.pages[l].size(); ++i)
      pr.pages[l][i] = (uint8_t)((i * 131u + l * 7u + (uint32_t)seed) ^ (uint8_t)(i >> 5));
  }
  return pr;
}

static std::vector<BlockId> block_ids(const Prompt& pr) {
  std::vector<BlockId> ids;
  size_t n_blocks = (pr.tokens.size() + P - 1) / P;
  const void* layers[N_LAYERS];
  for (size_t b = 0; b < n_blocks; ++b) {
    for (size_t l = 0; l < N_LAYERS; ++l)
      layers[l] = pr.pages[l].data() + b * P * TOK_BYTES;
    size_t rem = pr.tokens.size() - b * P;
    size_t tok = (P < rem) ? P : rem;
    ids.push_back(hash_block(pr.tokens.data() + b * P, tok,
                             layers, N_LAYERS, P * TOK_BYTES));
  }
  return ids;
}

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) != "--selftest") {
    printf("usage: glm_prefix_parity --selftest\n");
    return 1;
  }

  // The hash itself is pinned first: SHA-256 of the empty input.
  uint8_t d[32];
  sha256("", 0, d);
  bool vec_ok = true;
  const uint8_t want[16] = {0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                            0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24};
  for (int i = 0; i < 16; ++i) vec_ok = vec_ok && (d[i] == want[i]);
  REQUIRE("sha256 empty vector", vec_ok, 0);

  // identical prompts -> identical block IDs (page-aligned, D3 granularity)
  Prompt a = make_prompt(3 * P, 7);
  Prompt b = make_prompt(3 * P, 7);
  std::vector<BlockId> ids_a = block_ids(a);
  std::vector<BlockId> ids_b = block_ids(b);
  bool same = (ids_a.size() == ids_b.size());
  for (size_t i = 0; same && i < ids_a.size(); ++i) same = (ids_a[i] == ids_b[i]);
  REQUIRE("same prompt -> same block IDs", same, 0);

  // one token changed at pos k -> blocks equal below k/P, different from k/P on
  Prompt c = make_prompt(3 * P, 7);
  size_t k = P + 5;  // an edit inside block 1
  c.tokens[k] ^= 0x2a;
  std::vector<BlockId> ids_c = block_ids(c);
  size_t diverges_from = ids_a.size();
  for (size_t i = 0; i < ids_a.size(); ++i)
    if (ids_a[i] != ids_c[i]) { diverges_from = i; break; }
  REQUIRE("edit at k diverges from floor(k/P) on", diverges_from == k / P, 0);

  // two layers' bytes swapped -> different ID (hash covers all 11 layers' bytes)
  Prompt e = make_prompt(3 * P, 7);
  std::swap(e.pages[2], e.pages[7]);
  std::vector<BlockId> ids_e = block_ids(e);
  REQUIRE("layer bytes covered by the hash", ids_a[0] != ids_e[0], 0);

  // registry refcounts: acquire/release, floor at 0, per-ID independence
  BlockRegistry reg;
  BlockId x = ids_a[0];
  BlockId y = ids_a[1];
  bool reg_ok = true;
  reg_ok = reg_ok && (reg.acquire(x) == 1);
  reg_ok = reg_ok && (reg.acquire(x) == 2);
  reg_ok = reg_ok && (reg.acquire(y) == 1);
  reg_ok = reg_ok && (reg.refcount(x) == 2) && (reg.refcount(y) == 1);
  reg_ok = reg_ok && (reg.release(x) == 1) && (reg.release(x) == 0);
  reg_ok = reg_ok && (reg.release(x) == 0);   // floors at 0
  reg_ok = reg_ok && (reg.refcount(y) == 1);  // y untouched
  reg_ok = reg_ok && (reg.size() == 2);
  REQUIRE("registry refcounts acquire/release", reg_ok, 0);

  // Task 3.3: the boundary snapshot's carriage (D6): the handle's semantics.
  strata::prefix::SnapPages sp;
  REQUIRE("snap carriage: the default handle is the host fallback", !sp.live(), 0);
  sp.pages.push_back(7);
  sp.bytes = 1;
  REQUIRE("snap carriage: pages + bytes make the handle live", sp.live(), 0);
  REQUIRE("snap carriage: pages without bytes stays the fallback", !([] {
            strata::prefix::SnapPages s2; s2.pages.push_back(7); return s2.live(); }()), 0);

  // Task 3.3: the boundary stream's capacity math (D6): the released
  // GLM-5.3-Flash geometry (glm_model.hpp:72-92: d_inner 8192 = kda_head_dim
  // 128 x n_head 64, d_conv 4, idx_kpool 4, idx_key 128), one boundary =
  // 34 KDA layers' runs + 11 DSA caches' open key pool (2 rings x
  // (idx_kpool-1) x idx_key rows each); the measured D6 row
  // (152.67 MB per sequence per boundary). The snapshot rides the latent
  // pool's slab pages (page_size 256 rows x 1024 B = 256 KB/page) and the
  // consequence row must still hold: 13 boundaries' snapshot + canonical
  // pages fit the 2 GB pool's 8192 pages.
  const size_t run_f = 8192u * 128u + 3u * 8192u * 3u;         // one KDA layer's run
  const size_t pool_f = 2u * 3u * 128u;                        // one DSA cache's 2 rings
  const int64_t boundary_bytes = (int64_t) (34 * run_f + 11 * pool_f) * 4;
  REQUIRE("D6: one boundary stream is the measured 152.67 MB",
          boundary_bytes == 152667136, 0);
  const int64_t page_bytes = 262144;                           // 256 rows x 1024 B
  const size_t pages_per_boundary = (size_t) ((boundary_bytes + page_bytes - 1) / page_bytes);
  REQUIRE("D6: one boundary stream spans the computed slab pages",
          pages_per_boundary == 583, 0);
  REQUIRE("D6: 13 boundaries' snapshot + canonical pages fit the pool",
          (size_t) 13 * (pages_per_boundary + 22) <= 8192, 0);

  printf("glm_prefix_parity: %d failures\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
