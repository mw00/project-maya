// tests/glm_dsa/glm_prompt_cache_test.cpp - glm-dsa B14.14 (CPU only): the content-addressed prompt-cache store
// (include/strata/core/glm_prompt_cache.hpp) with fake state payloads (a file of the tokens; the "load" checks the
// requested prefix against them), in a scratch directory under the build tree (or $TMPDIR).
//   1. hashing: token-sensitive and length-sensitive keys, the block chain, entry names;
//   2. restore_len: the whole entry, a divergence (one token before it), a prompt that ends inside the entry;
//   3. put / lookup / take: longest common prefix across entries, the min length, dedupe (a stored prefix is
//      touched, not rewritten), supersession of entries that are prefixes, a hash-chain claim confirmed on tokens;
//   4. LRU: the byte cap and the entry cap evict the least recently used, a take refreshes recency, and the order
//      survives a re-open (the .meta times);
//   5. fingerprint refusal: a store opened with another identity sees none of the entries and leaves them on disk;
//      damaged files (.tmp, a .bin without .meta, a .bin of the wrong size, a bad magic) are removed at open;
//   6. a failed load removes that entry and restores nothing.
#include "strata/core/glm_prompt_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)
#include <vector>

using namespace strata::core::glmpc;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

std::vector<int32_t> seq(size_t n, int32_t base, int32_t step = 1) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + (int32_t) i * step;
    return v;
}
std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// the fake model: its "state" is the token list; saving writes it padded to `pad` bytes per token
struct Fake {
    std::vector<int32_t> state;
    size_t pad = 64;
    int saves = 0, loads = 0;
    bool fail_load = false;
    std::vector<int32_t> restored;
    Store::SaveFn save() {
        return [this](const std::string& path, std::string& err) -> uint64_t {
            ++saves;
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            const uint64_t n = state.size();
            f.write((const char*) &n, 8);
            for (int32_t t : state) {
                f.write((const char*) &t, 4);
                const std::string z(pad - 4, '\0');
                f.write(z.data(), (std::streamsize) z.size());
            }
            f.flush();
            if (!f) {
                err = "write";
                return 0;
            }
            return (uint64_t) f.tellp();
        };
    }
    Store::LoadFn load() {
        return [this](const std::string& path, int64_t n_file, int64_t n_use, std::string& err) -> bool {
            ++loads;
            if (fail_load) {
                err = "simulated read error";
                return false;
            }
            std::ifstream f(path, std::ios::binary);
            uint64_t n = 0;
            if (!f.read((char*) &n, 8) || (int64_t) n != n_file || n_use > n_file) {
                err = "header";
                return false;
            }
            restored.clear();
            for (int64_t i = 0; i < n_use; ++i) {
                int32_t t = 0;
                f.read((char*) &t, 4);
                f.seekg((std::streamoff) (pad - 4), std::ios::cur);
                restored.push_back(t);
            }
            return (bool) f;
        };
    }
};

StoreConfig conf(const std::string& dir, const std::string& id = "model-A fp=1,2,3") {
    StoreConfig c;
    c.dir = dir;
    c.cap_bytes = 1ull << 30;
    c.max_entries = 16;
    c.min_tokens = 64;
    c.identity = id;
    return c;
}

void test_hash() {
    std::printf("[1. hashing]\n");
    const std::vector<int32_t> a = seq(1000, 7), b = [&] { auto v = a; v[999] += 1; return v; }();
    check(hash_tokens(a.data(), a.size()) != hash_tokens(b.data(), b.size()), "one token changed: another key");
    check(hash_tokens(a.data(), 999) != hash_tokens(a.data(), 1000), "length matters");
    const std::vector<int32_t> z1(3, 0), z2(4, 0);
    check(hash_tokens(z1.data(), 3) != hash_tokens(z2.data(), 4), "zero tokens of different counts differ");
    const auto ca = block_chain(a), cb = block_chain(b);
    check(ca.size() == 1000 / kBlock && ca == cb, "block chain: 3 whole blocks of 256, the last partial block not hashed");
    std::vector<int32_t> c = a;
    c[300] = -1;
    const auto cc = block_chain(c);
    check(cc[0] == ca[0] && cc[1] != ca[1] && cc[2] != ca[2], "a change in block 1 changes blocks 1.. (chained)");
    check(entry_stem(0x1234abcdull) == "pc-000000001234abcd", "entry name: pc-<16 hex>");
    check(entry_key(a, "model-A") != entry_key(a, "model-B") && entry_key(a, "x") != hash_tokens(a.data(), a.size()),
          "the entry key mixes in the identity: two identities never share a file");
}

void test_restore_len() {
    std::printf("[2. restore_len]\n");
    check(restore_len(1000, 1000, 1200) == 1000, "the prompt extends the whole entry: 1000");
    check(restore_len(700, 1000, 1200) == 699, "diverges at 700: 699 (the NextN row 699 holds the entry's token 700)");
    check(restore_len(1000, 1500, 1000) == 999, "the prompt ends inside the entry: 999 (one token left to forward)");
    check(restore_len(1000, 1500, 1001) == 999, "a 1001-token prompt diverging at 1000 inside an entry of 1500: 999");
    check(restore_len(1000, 1500, 1200) == 999 && restore_len(1001, 1500, 1001) == 1000,
          "the same divergence for a longer prompt; a prompt that is all prefix: n_prompt - 1");
    check(restore_len(0, 1000, 10) == 0 && restore_len(5, 0, 10) == 0 && restore_len(5, 5, 0) == 0, "degenerate: 0");
}

void test_store(const std::string& dir) {
    std::printf("[3. put / lookup / take]\n");
    fs::remove_all(dir);
    Store s;
    std::string log, err;
    check(s.open(conf(dir), log) && s.on() && s.entries().empty(), "opens an empty directory: " + log);
    Fake m;
    const std::vector<int32_t> sys = seq(2000, 100);              // a shared system prompt
    const std::vector<int32_t> A = cat(sys, seq(600, 50000)), B = cat(sys, seq(900, 60000));
    m.state = A;
    bool stored = false;
    check(s.put(A, m.save(), err, &stored) && stored && s.entries().size() == 1 && m.saves == 1, "stores conversation A");
    m.state = B;
    check(s.put(B, m.save(), err, &stored) && stored && s.entries().size() == 2, "stores conversation B");
    // a new chat with the same system prompt: the longest prefix is the system prompt (from either; most recent B)
    const std::vector<int32_t> C = cat(sys, seq(50, 70000));
    Store::Match mc = s.lookup(C);
    check(mc.idx >= 0 && mc.lcp == 2000 && mc.restore == 1999 && s.entries()[(size_t) mc.idx].tokens == B,
          "a new chat sharing the 2000-token system prompt: restore 1999, from the most recent entry");
    // continuing A: the whole of A
    const std::vector<int32_t> A2 = cat(A, seq(30, 80000));
    m.restored.clear();
    const size_t got = s.take(A2, m.load(), err);
    check(got == A.size() && m.restored == A, "A's next turn: all " + std::to_string(A.size()) + " tokens restored");
    check(s.lookup(seq(63, 100)).idx < 0, "a prompt sharing under min_tokens (64): nothing");
    check(s.lookup(seq(3000, 900000)).idx < 0, "an unrelated prompt: nothing");
    // dedupe: storing A again, or a prefix of A, touches A
    m.saves = 0;
    m.state = A;
    check(s.put(A, m.save(), err, &stored) && !stored && m.saves == 0, "storing A again: no write");
    const std::vector<int32_t> A_pre(A.begin(), A.begin() + 2300);
    check(s.put(A_pre, m.save(), err, &stored) && !stored && m.saves == 0, "storing a prefix of A: no write");
    // supersession: a continuation of B replaces B
    m.state = cat(B, seq(400, 61000));
    check(s.put(m.state, m.save(), err, &stored) && stored && s.entries().size() == 2,
          "B continued: the new entry supersedes B (2 entries)");
    bool b_gone = true;
    for (const Entry& e : s.entries()) b_gone = b_gone && e.tokens != B;
    check(b_gone && !fs::exists(fs::path(dir) / (entry_stem(entry_key(B, conf(dir).identity)) + ".bin")),
          "B's files are removed");
    // a prompt that diverges inside the first block
    std::vector<int32_t> D = A;
    D[40] = -7;
    check(s.lookup(D).idx < 0, "diverging at token 40 (restore 39, under min_tokens 64): nothing");
    std::vector<int32_t> E2 = A;
    E2[1500] = -7;
    const Store::Match me = s.lookup(E2);
    check(me.lcp == 1500 && me.restore == 1499, "diverging at 1500 inside block 5: lcp 1500, restore 1499");
}

void test_lru(const std::string& dir) {
    std::printf("[4. LRU and the caps]\n");
    fs::remove_all(dir);
    StoreConfig c = conf(dir);
    c.max_entries = 3;
    Store s;
    std::string log, err;
    s.open(c, log);
    Fake m;
    std::vector<std::vector<int32_t>> conv;
    for (int i = 0; i < 3; ++i) {
        conv.push_back(seq(200, 1000 * (i + 1)));
        m.state = conv.back();
        s.put(conv.back(), m.save(), err);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // distinct .meta times
    }
    // use conv[0]: then conv[1] is the least recently used
    check(s.take(cat(conv[0], {1, 2, 3}), m.load(), err) == 200, "conv 0 restored (now the most recent)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    conv.push_back(seq(200, 9000));
    m.state = conv.back();
    s.put(conv.back(), m.save(), err);
    std::vector<std::vector<int32_t>> have;
    for (const Entry& e : s.entries()) have.push_back(e.tokens);
    const auto holds = [&](const std::vector<int32_t>& t) {
        for (const auto& h : have)
            if (h == t) return true;
        return false;
    };
    check(s.entries().size() == 3 && !holds(conv[1]) && holds(conv[0]) && holds(conv[2]) && holds(conv[3]) &&
              s.evicted() == 1,
          "entry cap 3: the 4th store evicts conv 1 (least recently used), not conv 0");
    // re-open: the order survives (the .meta times): the next eviction takes conv 2
    Store s2;
    s2.open(c, log);
    conv.push_back(seq(200, 12000));
    m.state = conv.back();
    s2.put(conv.back(), m.save(), err);
    have.clear();
    for (const Entry& e : s2.entries()) have.push_back(e.tokens);
    check(s2.entries().size() == 3 && !holds(conv[2]) && holds(conv[0]) && holds(conv[3]) && holds(conv[4]),
          "after a re-open the least recently used is still known (conv 2 evicted)");
    // the byte cap
    StoreConfig cb = conf(dir);
    const uint64_t one = s2.entries()[0].bytes;
    cb.cap_bytes = one * 2 + one / 2;   // room for two
    Store s3;
    s3.open(cb, log);
    conv.push_back(seq(200, 15000));
    m.state = conv.back();
    s3.put(conv.back(), m.save(), err);
    check(s3.entries().size() == 2 && s3.total_bytes() <= cb.cap_bytes, "byte cap of 2.5 entries: 2 kept");
}

void test_refusal(const std::string& dir) {
    std::printf("[5. fingerprint refusal and damaged files]\n");
    fs::remove_all(dir);
    std::string log, err;
    Fake m;
    {
        Store s;
        s.open(conf(dir), log);
        m.state = seq(300, 5);
        s.put(m.state, m.save(), err);
    }
    Store other;
    other.open(conf(dir, "model-A fp=1,2,3 | rotated index keys"), log);
    check(other.entries().empty() && other.refused_foreign() == 1 && other.lookup(cat(seq(300, 5), {9})).idx < 0,
          "another identity (e.g. the B14.11 rotation bit): the entry is refused: " + log);
    size_t files = 0;
    for (const auto& de : fs::directory_iterator(dir)) files += de.path().filename().string().rfind("pc-", 0) == 0;
    check(files == 2, "and left on disk (.bin + .meta)");
    // damaged files
    {
        Store s;
        s.open(conf(dir), log);
        m.state = seq(300, 77);
        s.put(m.state, m.save(), err);
        const std::string bin = s.entries().back().bin();
        std::ofstream(bin, std::ios::app) << "trailing";   // the .bin no longer matches its .meta's size
        std::ofstream(fs::path(dir) / "pc-00000000000000cc.meta") << "not a meta";
    }
    std::ofstream(fs::path(dir) / "pc-00000000000000aa.bin.tmp") << "x";
    std::ofstream(fs::path(dir) / "pc-00000000000000bb.bin") << "orphan";
    Store s;
    s.open(conf(dir), log);
    check(s.entries().size() == 1 && s.entries()[0].tokens == seq(300, 5) && s.removed_damaged() == 5,
          "damaged files removed at open (.tmp, orphan .bin, size mismatch pair, bad magic): " + log);
}

// review fix: entries of another identity count toward the caps (directory-wide) and are reclaimed first, the
// identity is part of the entry name, and put() keeps a free-space floor
void test_foreign_caps(const std::string& dir) {
    std::printf("[5b. other identities: the directory-wide caps, the free-space floor]\n");
    fs::remove_all(dir);
    std::string log, err;
    Fake m;
    const std::vector<int32_t> same = seq(300, 5);
    {
        Store a;   // identity A: two entries, the first one older
        a.open(conf(dir), log);
        m.state = same;
        a.put(m.state, m.save(), err);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        m.state = seq(300, 40000);
        a.put(m.state, m.save(), err);
    }
    StoreConfig cb = conf(dir, "model-A | another engine version");
    cb.max_entries = 3;
    Store b;
    b.open(cb, log);
    check(b.foreign_entries() == 2 && b.foreign_bytes() > 0 && b.entries().empty(),
          "identity B sees A's 2 entries as foreign and counts their bytes: " + log);
    m.state = same;   // the same tokens under B: another file, A's stays
    bool stored = false;
    check(b.put(m.state, m.save(), err, &stored) && stored && b.foreign_entries() == 2 && b.reclaimed_foreign() == 0,
          "B stores the same conversation as A under its own name (A's entry untouched)");
    size_t files = 0;
    for (const auto& de : fs::directory_iterator(dir)) files += de.path().filename().string().rfind("pc-", 0) == 0;
    check(files == 6, "3 entries on disk (A's two and B's)");
    m.state = seq(300, 70000);   // a 4th entry with an entry cap of 3: A's OLDEST goes, not B's
    check(b.put(m.state, m.save(), err, &stored) && stored && b.entries().size() == 2 && b.foreign_entries() == 1 &&
              b.reclaimed_foreign() == 1 && b.evicted() == 0,
          "the entry cap is directory-wide: a foreign entry is removed first (B keeps both of its own)");
    Store a2;
    a2.open(conf(dir), log);
    check(a2.entries().size() == 1 && a2.entries()[0].tokens == seq(300, 40000),
          "A re-opened: its older entry was the one reclaimed, its newer one is intact");
    // the byte cap: room for B's two entries and the new one only, A's last entry must go
    cb.cap_bytes = b.total_bytes() / 2 * 3 + b.total_bytes() / 4;
    cb.max_entries = 16;
    Store b2;
    b2.open(cb, log);
    m.state = seq(300, 90000);
    check(b2.put(m.state, m.save(), err, &stored) && stored && b2.foreign_entries() == 0 && b2.entries().size() == 3 &&
              b2.total_bytes() <= cb.cap_bytes,
          "the byte cap counts foreign bytes: A's last entry is reclaimed before any of B's");
    // the free-space floor: nothing is stored below it
    StoreConfig cf = conf(dir, "model-C");
    cf.min_free_bytes = ~0ull;
    Store c;
    c.open(cf, log);
    m.state = seq(300, 110000);
    err.clear();
    check(!c.put(m.state, m.save(), err, &stored) && !stored && c.entries().empty() && !err.empty() &&
              c.foreign_entries() == 0,
          "under the free-space floor: other identities' entries removed, then nothing stored (an error)");
    check(config_env("x").min_free_bytes == kMinFreeBytes, "config_env: the 8 GB floor of the slots");
}

void test_failed_load(const std::string& dir) {
    std::printf("[6. a failed load]\n");
    fs::remove_all(dir);
    std::string log, err;
    Store s;
    s.open(conf(dir), log);
    Fake m;
    m.state = seq(500, 3);
    s.put(m.state, m.save(), err);
    m.fail_load = true;
    err.clear();
    const size_t n = s.take(cat(m.state, {1}), m.load(), err);
    check(n == 0 && !err.empty() && s.entries().empty(), "a failed load: nothing restored, the entry removed (" + err + ")");
    Store off;
    check(!off.open(conf(""), log) && !off.on() && off.lookup(seq(1000, 1)).idx < 0, "an empty directory setting: off");
}

}  // namespace

int main() {
    const char* tmp = std::getenv("TMPDIR");
    const std::string base = (fs::path(tmp && tmp[0] ? tmp : fs::current_path().string()) /
                              ("glm_prompt_cache_test." + std::to_string((long long) ::getpid())))
                                 .string();
    test_hash();
    test_restore_len();
    test_store(base + "/a");
    test_lru(base + "/b");
    test_refusal(base + "/c");
    test_foreign_caps(base + "/e");
    test_failed_load(base + "/d");
    std::error_code ec;
    fs::remove_all(base, ec);
    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
