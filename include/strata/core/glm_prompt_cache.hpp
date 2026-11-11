// include/strata/core/glm_prompt_cache.hpp - glm-dsa B14.14: the NVMe prompt cache across conversations (glm-dsa).
//
// Maya's conversation slots (src/program/generate.cpp, Glm5Model::slot_save / slot_load) set ONE conversation aside
// and take it back when a prompt extends it exactly.  On glm-dsa every cache is append-only and per position (the
// latent, k_pe and index-key rows; no recurrent state), so a saved state of n tokens also holds the state of every
// prefix of it.  This store keeps such states under a directory, content-addressed by their tokens, and restores the
// LONGEST COMMON PREFIX of a new prompt with any of them - a new chat that shares a 20K-token system prompt and tool
// list with an earlier one reads only what differs (BUILD-PLAN B14.14, ik_llama.cpp's --prompt-cache across
// conversations, docs/parameters.md:162-164).
//
// MAYA_GLM_PROMPT_CACHE_DIR=<dir> turns it on (default off).  MAYA_GLM_PROMPT_CACHE_GB caps the bytes on disk
// (default 64), MAYA_GLM_PROMPT_CACHE_MIN (default 512) is the shortest state stored and the shortest prefix
// restored, MAYA_GLM_PROMPT_CACHE_MAX the entry count (default 64).
//
// Files per entry: pc-<16 hex>.bin (Maya's MAYALST3 slot file, written by Glm5Model::slot_save) and pc-<16 hex>.meta
// (this store's record).  The name is entry_key(): the 64-bit hash of the entry's tokens (FNV-1a over the int32 ids,
// length mixed in) with the model identity mixed in, so storing the same conversation twice touches one entry.  The .meta holds: magic "GMPCACH1", the
// model identity (the slot fingerprint words, the split's layer ranges and the pack's files: Glm5Model::
// pcache_identity), the tokens, the .bin's size, the magic again.  Both files are written under .tmp names and
// renamed, the .meta last.
//
// Lookup: every entry keeps a chain of block hashes (h_b = the hash of tokens [0, (b + 1) * kBlock)); a prompt's chain
// is compared block by block (no token compare for entries that diverge in their first block), then the common
// prefix is extended token by token inside the next block.  The restore length is restore_len(): the common prefix,
// except that a restore shorter than the entry stops one token before the prompt diverges (the NextN block's row
// t pairs h_t with token t + 1, so the row before the divergence holds the entry's next token, not the prompt's), and
// it always leaves one prompt token to forward (the first sample's logits).
//
// LRU: the .meta's modification time is the entry's last use (a restore or a re-store touches it), so the order
// survives restarts.  put() evicts least recently used entries until the total is under the cap and the count under
// the maximum; a new entry also supersedes every entry whose tokens are a prefix of its own (it holds their states).
//
// Fingerprint refusal: an entry whose identity differs from this model's (another model, quantization, split, key
// dtype, B14.11 rotation, an engine version, ...) is never restored.  It stays on disk until room is needed, but it
// counts toward the caps, which cover the whole directory: put() removes such entries first (oldest .meta first),
// before any of this model's, so identity changes (bring-up A/Bs, a Maya sync) never strand more than the cap.  The
// identity is mixed into the entry name (entry_key), so two identities never write the same file.  put() also keeps
// a free-space floor (kMinFreeBytes, the slots' 8 GB): below it, other identities' entries are removed, then the
// state is not stored.
// Damaged entries (bad magic, a .bin of the wrong size, a .bin without its .meta, .tmp leftovers) are removed.
//
// Plain C++17 (std::filesystem), no CUDA: the save and load are callbacks.  The engine glue is
// src/core/glm_prompt_cache_engine.cpp; tests/glm_dsa/glm_prompt_cache_test.cpp tests the store with fake payloads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {
class Glm5Model;
}

namespace strata::core::glmpc {

constexpr size_t kBlock = 256;                     // tokens per chained block hash
constexpr uint64_t kMetaMagic = 0x3148434143504d47ull;   // "GMPCACH1"

/// FNV-1a 64 over the token ids (little-endian bytes), then the count.
uint64_t hash_tokens(const int32_t* t, size_t n);
/// An entry's key (its file name): hash_tokens with the model identity's bytes and length mixed in.
uint64_t entry_key(const std::vector<int32_t>& t, const std::string& identity);
/// put()'s free-space floor in config_env (the conversation slots' 8 GB).
constexpr uint64_t kMinFreeBytes = 8ull << 30;
/// h_b for every complete block: the hash of tokens [0, (b + 1) * kBlock), chained (h_b = mix(h_{b-1}, block b)).
std::vector<uint64_t> block_chain(const std::vector<int32_t>& t);
/// "pc-<16 hex digits>".
std::string entry_stem(uint64_t key);
/// How many tokens a restore of an entry of n_entry tokens takes for a prompt of n_prompt tokens sharing lcp with it.
size_t restore_len(size_t lcp, size_t n_entry, size_t n_prompt);

struct StoreConfig {
    std::string dir;                   // empty: off
    uint64_t cap_bytes = 64ull << 30;  // MAYA_GLM_PROMPT_CACHE_GB
    size_t max_entries = 64;           // MAYA_GLM_PROMPT_CACHE_MAX
    size_t min_tokens = 512;           // MAYA_GLM_PROMPT_CACHE_MIN
    uint64_t min_free_bytes = 0;       // put() stores nothing below this free space (config_env: kMinFreeBytes)
    std::string identity;              // the model identity an entry must carry
};
/// The env switches, with the caller's identity.
StoreConfig config_env(const std::string& identity);

struct Entry {
    std::string stem;                  // <dir>/pc-<hex> without the extension
    uint64_t key = 0, bytes = 0;
    std::vector<int32_t> tokens;
    std::vector<uint64_t> chain;
    uint64_t used = 0;                 // LRU clock (larger: more recent)
    std::string bin() const { return stem + ".bin"; }
    std::string meta() const { return stem + ".meta"; }
};

class Store {
public:
    /// Writes the slot file of the model's current state (its first n positions) to path; returns its bytes, 0 + err.
    using SaveFn = std::function<uint64_t(const std::string& path, std::string& err)>;
    /// Restores the first n_use positions from the slot file at path (written for n_file positions); false + err.
    using LoadFn = std::function<bool(const std::string& path, int64_t n_file, int64_t n_use, std::string& err)>;

    /// Creates the directory, indexes this model's entries, removes damaged ones.  false: off or not writable.
    bool open(const StoreConfig& c, std::string& log);
    bool on() const { return on_; }
    const StoreConfig& config() const { return cfg_; }

    struct Match {
        int idx = -1;        // the entry
        size_t lcp = 0;      // tokens it shares with the prompt
        size_t restore = 0;  // restore_len(lcp, its size, the prompt's size)
    };
    /// The entry with the longest restore for this prompt (ties: the most recently used).  idx -1: none at least
    /// min_tokens long.
    Match lookup(const std::vector<int32_t>& prompt) const;
    /// lookup + load + touch.  The restored token count (0: nothing restored; err says why when a load failed, and
    /// that entry is removed).
    size_t take(const std::vector<int32_t>& prompt, const LoadFn& load, std::string& err);
    /// Stores the state of `tokens` (the model stands at tokens.size()): skipped (true, *stored false) when shorter
    /// than min_tokens or already held by an entry that has them as a prefix (that entry is touched); otherwise
    /// written through `save`, superseded entries removed, then LRU eviction to the caps.
    bool put(const std::vector<int32_t>& tokens, const SaveFn& save, std::string& err, bool* stored = nullptr);
    /// Marks entry i used now (and its .meta's time).
    void touch(size_t i);

    const std::vector<Entry>& entries() const { return e_; }
    uint64_t total_bytes() const;                       // this model's entries
    uint64_t foreign_bytes() const;                     // other identities' entries still on disk (.bin + .meta)
    size_t foreign_entries() const { return other_.size(); }
    int refused_foreign() const { return foreign_; }   // entries of another identity seen at open
    int reclaimed_foreign() const { return reclaimed_; }   // ... removed since open (caps, free space)
    int removed_damaged() const { return damaged_; }   // damaged files removed at open
    int evicted() const { return evicted_; }           // entries removed by the caps since open

private:
    struct Foreign {
        std::string stem;
        uint64_t bytes = 0;
        int64_t age = 0;   // the .meta's modification time (file clock ticks)
    };
    void drop(size_t i);
    bool reclaim_oldest_foreign();
    bool room_on_disk() const;
    bool write_meta(const Entry& e, std::string& err) const;
    bool read_meta(const std::string& path, Entry& e, bool& foreign) const;
    StoreConfig cfg_;
    bool on_ = false;
    std::vector<Entry> e_;
    std::vector<Foreign> other_;
    uint64_t clock_ = 0;
    int foreign_ = 0, damaged_ = 0, evicted_ = 0, reclaimed_ = 0;
};

// ---- the engine glue (src/core/glm_prompt_cache_engine.cpp; one-line hooks in generate.cpp: the slots' setup,
// prep_prompt and the end of the run, every context's with several conversations).  All do nothing unless
// MAYA_GLM_PROMPT_CACHE_DIR is set, MAYA_GLM_NO_REUSE is not, and the model is glm-dsa on the fast path; the store
// opens at the first call (its line goes to stderr).

/// The store is on for this model (MAYA_GLM_PROMPT_CACHE_DIR set and writable, glm-dsa on the fast path, no
/// MAYA_GLM_NO_REUSE).  generate.cpp then turns Maya's conversation slots off: the store restores every prefix a slot
/// would, and both would write the same multi-GB state at each conversation switch.
bool engine_on(Glm5Model& model);
/// The conversation in the model (snap_tokens: the tokens of its snapshot, which stands at their count) into the
/// store, when text only and at least min_tokens long.  Call before the model's state is replaced.
void engine_put(Glm5Model& model, const std::vector<int32_t>& snap_tokens, bool text_only);
/// The longest stored prefix of `prompt` back in the model (nothing reused yet, text only): the restored token count
/// (0: none).  On a restore snap_tokens becomes the restored prefix, snap_img_hash no_images, snap_in_slot false.
size_t engine_take(Glm5Model& model, const std::vector<int32_t>& prompt, bool text_only,
                   std::vector<int32_t>& snap_tokens, uint64_t& snap_img_hash, bool& snap_in_slot, uint64_t no_images);

}  // namespace strata::core::glmpc
