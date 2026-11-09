// src/core/glm_prompt_cache.cpp - glm-dsa B14.14: the content-addressed prompt-cache store
// (include/strata/core/glm_prompt_cache.hpp).
#include "strata/core/glm_prompt_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace strata::core::glmpc {

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kFnvBasis = 1469598103934665603ull, kFnvPrime = 1099511628211ull;

uint64_t fnv_tokens(uint64_t h, const int32_t* t, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = (uint32_t) t[i];
        for (int b = 0; b < 4; ++b) h = (h ^ ((v >> (8 * b)) & 0xffu)) * kFnvPrime;
    }
    return h;
}
uint64_t fnv_u64(uint64_t h, uint64_t v) {
    for (int b = 0; b < 8; ++b) h = (h ^ ((v >> (8 * b)) & 0xffu)) * kFnvPrime;
    return h;
}

double env_num(const char* k, double d) {
    const char* v = std::getenv(k);
    return v && v[0] ? std::atof(v) : d;
}

bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::char_traits<char>::length(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

}  // namespace

uint64_t hash_tokens(const int32_t* t, size_t n) { return fnv_u64(fnv_tokens(kFnvBasis, t, n), (uint64_t) n); }

uint64_t entry_key(const std::vector<int32_t>& t, const std::string& identity) {
    uint64_t h = hash_tokens(t.data(), t.size());
    for (unsigned char c : identity) h = (h ^ c) * kFnvPrime;
    return fnv_u64(h, (uint64_t) identity.size());
}

std::vector<uint64_t> block_chain(const std::vector<int32_t>& t) {
    std::vector<uint64_t> c;
    uint64_t h = kFnvBasis;
    for (size_t b = 0; (b + 1) * kBlock <= t.size(); ++b) {
        h = fnv_tokens(h, t.data() + b * kBlock, kBlock);
        c.push_back(fnv_u64(h, (uint64_t) (b + 1)));
    }
    return c;
}

std::string entry_stem(uint64_t key) {
    char b[32];
    std::snprintf(b, sizeof b, "pc-%016llx", (unsigned long long) key);
    return b;
}

size_t restore_len(size_t lcp, size_t n_entry, size_t n_prompt) {
    if (n_prompt == 0 || n_entry == 0) return 0;
    const size_t cap = std::min({lcp, n_entry, n_prompt - 1});
    if (cap == n_entry) return cap;              // the whole entry: the prompt continues it (Maya's slot semantics)
    if (cap == lcp && cap > 0) return cap - 1;   // the prompt diverges at lcp: the NextN row lcp - 1 is the entry's
    return cap;
}

StoreConfig config_env(const std::string& identity) {
    StoreConfig c;
    if (const char* d = std::getenv("STRATA_GLM_PROMPT_CACHE_DIR")) c.dir = d;
    c.cap_bytes = (uint64_t) (std::max(0.0, env_num("STRATA_GLM_PROMPT_CACHE_GB", 64.0)) * 1073741824.0);
    c.max_entries = (size_t) std::max(1.0, env_num("STRATA_GLM_PROMPT_CACHE_MAX", 64.0));
    c.min_tokens = (size_t) std::max(1.0, env_num("STRATA_GLM_PROMPT_CACHE_MIN", 512.0));
    c.min_free_bytes = kMinFreeBytes;
    c.identity = identity;
    return c;
}

// ---------------------------------------------------------------- the .meta record
bool Store::write_meta(const Entry& e, std::string& err) const {
    const std::string tmp = e.meta() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const uint64_t il = cfg_.identity.size(), nt = e.tokens.size();
        f.write((const char*) &kMetaMagic, 8);
        f.write((const char*) &il, 8);
        f.write(cfg_.identity.data(), (std::streamsize) il);
        f.write((const char*) &nt, 8);
        f.write((const char*) e.tokens.data(), (std::streamsize) (nt * sizeof(int32_t)));
        f.write((const char*) &e.bytes, 8);
        f.write((const char*) &kMetaMagic, 8);
        f.flush();
        if (!f) {
            err = "writing " + tmp + " failed";
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, e.meta(), ec);
    if (ec) err = "renaming " + tmp + ": " + ec.message();
    return !ec;
}

bool Store::read_meta(const std::string& path, Entry& e, bool& foreign) const {
    foreign = false;
    std::ifstream f(path, std::ios::binary);
    uint64_t m0 = 0, il = 0, nt = 0, m1 = 0;
    if (!f.read((char*) &m0, 8) || m0 != kMetaMagic || !f.read((char*) &il, 8) || il > (1u << 20)) return false;
    std::string id(il, '\0');
    if (!f.read(id.data(), (std::streamsize) il)) return false;
    if (!f.read((char*) &nt, 8) || nt == 0 || nt > (1ull << 26)) return false;
    e.tokens.resize(nt);
    if (!f.read((char*) e.tokens.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &e.bytes, 8) ||
        !f.read((char*) &m1, 8) || m1 != kMetaMagic)
        return false;
    if (id != cfg_.identity) {   // whole and readable, but another model's: refused, left alone
        foreign = true;
        return false;
    }
    e.key = entry_key(e.tokens, cfg_.identity);
    e.chain = block_chain(e.tokens);
    return true;
}

// ---------------------------------------------------------------- open
bool Store::open(const StoreConfig& c, std::string& log) {
    cfg_ = c;
    on_ = false;
    e_.clear();
    other_.clear();
    foreign_ = damaged_ = evicted_ = reclaimed_ = 0;
    if (cfg_.dir.empty()) return false;
    std::error_code ec;
    fs::create_directories(cfg_.dir, ec);
    if (!fs::is_directory(cfg_.dir, ec)) {
        log = "prompt cache: " + cfg_.dir + " is not a writable directory - off";
        return false;
    }
    struct Found {
        Entry e;
        fs::file_time_type t;
    };
    std::vector<Found> found;
    std::vector<fs::path> rm;
    for (const auto& de : fs::directory_iterator(cfg_.dir, ec)) {
        const std::string fn = de.path().filename().string(), p = de.path().string();
        if (fn.rfind("pc-", 0) != 0) continue;
        if (ends_with(fn, ".tmp")) {   // an interrupted write
            rm.push_back(de.path());
            continue;
        }
        if (ends_with(fn, ".bin")) {
            if (!fs::exists(p.substr(0, p.size() - 4) + ".meta", ec)) rm.push_back(de.path());   // never completed
            continue;
        }
        if (!ends_with(fn, ".meta")) continue;
        Found fd;
        fd.e.stem = p.substr(0, p.size() - 5);
        bool foreign = false;
        const bool ok = read_meta(p, fd.e, foreign);
        if (foreign) {   // kept, but counted toward the directory's caps and reclaimed first (put)
            ++foreign_;
            Foreign o;
            o.stem = fd.e.stem;
            o.bytes = (uint64_t) fs::file_size(p, ec);
            if (fs::exists(fd.e.bin(), ec)) o.bytes += (uint64_t) fs::file_size(fd.e.bin(), ec);
            o.age = (int64_t) fs::last_write_time(p, ec).time_since_epoch().count();
            other_.push_back(std::move(o));
            continue;
        }
        const uint64_t have = fs::exists(fd.e.bin(), ec) ? (uint64_t) fs::file_size(fd.e.bin(), ec) : 0;
        if (!ok || have == 0 || have != fd.e.bytes || fd.e.stem != (fs::path(cfg_.dir) / entry_stem(fd.e.key)).string()) {
            rm.push_back(de.path());
            rm.push_back(fd.e.bin());
            continue;
        }
        fd.t = fs::last_write_time(p, ec);
        found.push_back(std::move(fd));
    }
    for (const auto& p : rm) {
        if (fs::remove(p, ec)) ++damaged_;
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
    for (auto& fd : found) {
        fd.e.used = ++clock_;
        e_.push_back(std::move(fd.e));
    }
    on_ = true;
    char b[320];
    std::snprintf(b, sizeof b, "prompt cache: %s holds %zu entr%s of this model (%.2f GB, cap %.1f GB)", cfg_.dir.c_str(),
                  e_.size(), e_.size() == 1 ? "y" : "ies", (double) total_bytes() / 1e9, (double) cfg_.cap_bytes / 1e9);
    log = b;
    if (foreign_) {
        char f[160];
        std::snprintf(f, sizeof f, ", %d of another model or setting (%.2f GB: counted toward the caps, removed first "
                      "when room is needed)", foreign_, (double) foreign_bytes() / 1e9);
        log += f;
    }
    if (damaged_) log += ", " + std::to_string(damaged_) + " damaged file(s) removed";
    return true;
}

uint64_t Store::total_bytes() const {
    uint64_t t = 0;
    for (const Entry& e : e_) t += e.bytes;
    return t;
}

uint64_t Store::foreign_bytes() const {
    uint64_t t = 0;
    for (const Foreign& o : other_) t += o.bytes;
    return t;
}

bool Store::reclaim_oldest_foreign() {
    if (other_.empty()) return false;
    size_t old = 0;
    for (size_t i = 1; i < other_.size(); ++i)
        if (other_[i].age < other_[old].age) old = i;
    std::error_code ec;
    fs::remove(other_[old].stem + ".meta", ec);   // the .meta first, as drop()
    fs::remove(other_[old].stem + ".bin", ec);
    other_.erase(other_.begin() + (long) old);
    ++reclaimed_;
    return true;
}

bool Store::room_on_disk() const {
    if (cfg_.min_free_bytes == 0) return true;
    std::error_code ec;
    const auto sp = fs::space(cfg_.dir, ec);
    return ec || sp.available >= cfg_.min_free_bytes;
}

void Store::touch(size_t i) {
    if (i >= e_.size()) return;
    e_[i].used = ++clock_;
    std::error_code ec;
    fs::last_write_time(e_[i].meta(), fs::file_time_type::clock::now(), ec);
}

void Store::drop(size_t i) {
    std::error_code ec;
    fs::remove(e_[i].meta(), ec);   // the .meta first: a crash between leaves a .bin that open() removes
    fs::remove(e_[i].bin(), ec);
    e_.erase(e_.begin() + (long) i);
}

// ---------------------------------------------------------------- lookup / take
Store::Match Store::lookup(const std::vector<int32_t>& prompt) const {
    Match best;
    if (!on_ || prompt.empty()) return best;
    const std::vector<uint64_t> pc = block_chain(prompt);
    for (size_t i = 0; i < e_.size(); ++i) {
        const Entry& e = e_[i];
        size_t b = 0;   // whole blocks in common (chained hashes), then the tokens inside the next block
        while (b < pc.size() && b < e.chain.size() && pc[b] == e.chain[b]) ++b;
        size_t lcp = b * kBlock;
        // a matching chain is a hash claim: confirm it on the tokens (a collision only shortens the match)
        if (lcp > 0 && !std::equal(prompt.begin(), prompt.begin() + (long) lcp, e.tokens.begin())) {
            lcp = 0;
            while (lcp < prompt.size() && lcp < e.tokens.size() && prompt[lcp] == e.tokens[lcp]) ++lcp;
        }
        while (lcp < prompt.size() && lcp < e.tokens.size() && prompt[lcp] == e.tokens[lcp]) ++lcp;
        const size_t r = restore_len(lcp, e.tokens.size(), prompt.size());
        if (r < cfg_.min_tokens) continue;
        if (r > best.restore || (r == best.restore && best.idx >= 0 && e.used > e_[(size_t) best.idx].used)) {
            best.idx = (int) i;
            best.lcp = lcp;
            best.restore = r;
        }
    }
    return best;
}

size_t Store::take(const std::vector<int32_t>& prompt, const LoadFn& load, std::string& err) {
    const Match m = lookup(prompt);
    if (m.idx < 0) return 0;
    const Entry& e = e_[(size_t) m.idx];
    if (!load(e.bin(), (int64_t) e.tokens.size(), (int64_t) m.restore, err)) {
        drop((size_t) m.idx);
        return 0;
    }
    touch((size_t) m.idx);
    return m.restore;
}

// ---------------------------------------------------------------- put
bool Store::put(const std::vector<int32_t>& tokens, const SaveFn& save, std::string& err, bool* stored) {
    if (stored) *stored = false;
    if (!on_ || tokens.size() < cfg_.min_tokens) return true;
    // already held: an entry with these tokens as a prefix (the same conversation, or a longer one)
    for (size_t i = 0; i < e_.size(); ++i)
        if (e_[i].tokens.size() >= tokens.size() && std::equal(tokens.begin(), tokens.end(), e_[i].tokens.begin())) {
            touch(i);
            return true;
        }
    // the free-space floor (the slots' 8 GB): entries of another identity go first, then the state is not stored
    while (!room_on_disk() && reclaim_oldest_foreign()) {
    }
    if (!room_on_disk()) {
        char b[160];
        std::snprintf(b, sizeof b, "under %.0f GB free in the directory - not stored", (double) cfg_.min_free_bytes / 1e9);
        err = b;
        return false;
    }
    Entry e;
    e.key = entry_key(tokens, cfg_.identity);
    e.stem = (fs::path(cfg_.dir) / entry_stem(e.key)).string();
    e.tokens = tokens;
    e.chain = block_chain(tokens);
    for (size_t i = e_.size(); i-- > 0;)   // a hash collision with a different conversation: the new one replaces it
        if (e_[i].key == e.key) drop(i);
    std::error_code ec;
    const std::string tmp = e.bin() + ".tmp";
    e.bytes = save(tmp, err);
    if (e.bytes == 0 || (fs::rename(tmp, e.bin(), ec), ec) || !write_meta(e, err)) {
        if (err.empty()) err = ec ? ec.message() : "the state was not written";
        fs::remove(tmp, ec);
        fs::remove(e.bin(), ec);
        return false;
    }
    // superseded: entries whose tokens are a prefix of the new one's
    for (size_t i = e_.size(); i-- > 0;)
        if (e_[i].tokens.size() < tokens.size() &&
            std::equal(e_[i].tokens.begin(), e_[i].tokens.end(), tokens.begin()))
            drop(i);
    e.used = ++clock_;
    e_.push_back(std::move(e));
    if (stored) *stored = true;
    // the caps cover the whole directory: entries of another identity (never usable here) go first, oldest first;
    // then this model's least recently used, never the new entry
    uint64_t total = total_bytes() + foreign_bytes();
    const auto over = [&] { return total > cfg_.cap_bytes || e_.size() + other_.size() > cfg_.max_entries; };
    while (over() && !other_.empty()) {
        reclaim_oldest_foreign();
        total = total_bytes() + foreign_bytes();
    }
    while (e_.size() > 1 && over()) {
        size_t lru = 0;
        for (size_t i = 1; i + 1 < e_.size(); ++i)
            if (e_[i].used < e_[lru].used) lru = i;
        total -= e_[lru].bytes;
        drop(lru);
        ++evicted_;
    }
    return true;
}

}  // namespace strata::core::glmpc
