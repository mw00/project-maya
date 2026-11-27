// src/ds4/ds4_keep.hpp - maya-ds4's incremental GEN line (DS4_GEN_KEEP=1, INFO gen_keep=1; the server's MAYA_GEN_KEEP).
// A client re-sends its whole conversation every turn, and the server sent all of it to the engine as text: over 1 MB
// at 220K tokens, read in 4 KB pieces, copied and parsed, to be compared with the tokens the state already holds.  With
// `keep=<n> h=<hash>` the line carries only the ids from n on: the engine takes the first n from its own state - if
// the state holds at least n tokens and the hash of its first n is the line's - and the request's ids are then the
// very ids the whole line would have carried, so everything after the parse is as before.  Anything else (a fresh
// engine, another conversation in the model, a state cut short by a STOP or an unwind, a server out of step) is
// answered with a RESEND line and no request: the server writes the whole line.  The hash is BLAKE2b-128 (RFC 7693,
// no key; the server's hashlib.blake2b(digest_size=16)) over the ids as little-endian int32, as 32 hex digits: equal
// hashes of equal counts are equal ids for every purpose, so no prompt other than the server's is ever read.  The
// engine hashes its own tokens (never something the server said it holds), by a running state: the tokens the hash
// covers are kept beside it and compared with the state's first (memcmp), and only the tokens past them are hashed -
// O(new) a request; a state that no longer begins with them starts the hash over.  Pure logic, no GPU
// (tests/ds4_keep_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace maya::ds4 {

// BLAKE2b with a 16-byte digest and no key, incremental; a copy is the state so far (the running hash's snapshot)
struct Blake2b128 {
    uint64_t h[8];
    uint64_t t = 0;   // (bytes compressed so far: under 2^64, the high counter word stays 0)
    uint8_t buf[128];
    size_t n = 0;
    Blake2b128() {
        static const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                                       0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                                       0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
        for (int i = 0; i < 8; ++i) h[i] = iv[i];
        h[0] ^= 0x01010000ull ^ 16ull;   // (the parameter block: digest length 16, no key, fanout 1, depth 1)
    }
    void update(const uint8_t* p, size_t len) {
        while (len > 0) {
            if (n == 128) {   // (a full block is compressed only when more follows: the last one is final()'s)
                t += 128;
                compress(false);
                n = 0;
            }
            const size_t k = std::min(len, (size_t) 128 - n);
            std::memcpy(buf + n, p, k);
            n += k, p += k, len -= k;
        }
    }
    // the digest of everything so far as 32 hex digits; the state itself goes on unchanged (a copy is finished)
    std::string hex() const {
        Blake2b128 f = *this;
        f.t += f.n;
        std::memset(f.buf + f.n, 0, 128 - f.n);
        f.compress(true);
        static const char* d = "0123456789abcdef";
        std::string s(32, '0');
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = (uint8_t) (f.h[i / 8] >> (8 * (i % 8)));
            s[(size_t) (2 * i)] = d[b >> 4], s[(size_t) (2 * i + 1)] = d[b & 15];
        }
        return s;
    }

  private:
    static uint64_t rotr(uint64_t x, int r) { return (x >> r) | (x << (64 - r)); }
    void compress(bool last) {
        static const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                                       0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                                       0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
        static const uint8_t sigma[12][16] = {
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
            {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
            {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
            {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
            {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
        uint64_t m[16], v[16];
        for (int i = 0; i < 16; ++i) {   // (the block's words, little endian whatever the host's order)
            uint64_t w = 0;
            for (int b = 7; b >= 0; --b) w = (w << 8) | buf[8 * i + b];
            m[i] = w;
        }
        for (int i = 0; i < 8; ++i) v[i] = h[i], v[i + 8] = iv[i];
        v[12] ^= t;
        if (last) v[14] = ~v[14];
        const auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
            v[a] = v[a] + v[b] + x, v[d] = rotr(v[d] ^ v[a], 32);
            v[c] = v[c] + v[d], v[b] = rotr(v[b] ^ v[c], 24);
            v[a] = v[a] + v[b] + y, v[d] = rotr(v[d] ^ v[a], 16);
            v[c] = v[c] + v[d], v[b] = rotr(v[b] ^ v[c], 63);
        };
        for (int r = 0; r < 12; ++r) {
            const uint8_t* s = sigma[r];
            g(0, 4, 8, 12, m[s[0]], m[s[1]]), g(1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(2, 6, 10, 14, m[s[4]], m[s[5]]), g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(0, 5, 10, 15, m[s[8]], m[s[9]]), g(1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(2, 7, 8, 13, m[s[12]], m[s[13]]), g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
    }
};

// The running hash of the state's first tokens: `seen` is what `run` covers.  hex(st, n): the hash of st[0 .. n) - the
// state's own tokens, n at most its size - going on from `seen` when st still begins with it and n is not short of
// it (the usual turn: the conversation grew), from the start otherwise (the state was cut, went back, or another
// conversation took the model).  Never trusts `seen` beyond the comparison with st itself.
struct KeepHash {
    std::vector<int32_t> seen;
    Blake2b128 run;
    std::string hex(const int32_t* st, size_t n) {
        if (seen.size() > n || (!seen.empty() && std::memcmp(seen.data(), st, seen.size() * sizeof(int32_t)) != 0)) {
            seen.clear();
            run = Blake2b128();
        }
        uint8_t le[4 * 256];
        for (size_t i = seen.size(); i < n;) {
            const size_t k = std::min(n - i, (size_t) 256);
            for (size_t j = 0; j < k; ++j) {
                const uint32_t x = (uint32_t) st[i + j];
                le[4 * j] = (uint8_t) x, le[4 * j + 1] = (uint8_t) (x >> 8);
                le[4 * j + 2] = (uint8_t) (x >> 16), le[4 * j + 3] = (uint8_t) (x >> 24);
            }
            run.update(le, 4 * k);
            i += k;
        }
        seen.insert(seen.end(), st + seen.size(), st + n);
        return run.hex();
    }
};

// A GEN line's keep=<n> h=<hash> against the state's tokens `st`: true, and `ids` (the line's ids, from n on) become
// the whole prompt - st[0 .. n) ahead of them - when the state holds n tokens whose hash is `h`; false, `ids`
// untouched, otherwise (the caller answers RESEND).  n = 0 keeps nothing and is never sent: refused, like a hash of
// another length.
inline bool keep_ids(KeepHash& kh, const std::vector<int32_t>& st, long long n, const std::string& h,
                     std::vector<int>& ids) {
    if (n <= 0 || (unsigned long long) n > st.size() || h.size() != 32 || ids.empty()) return false;
    if (kh.hex(st.data(), (size_t) n) != h) return false;
    std::vector<int> whole;
    whole.reserve((size_t) n + ids.size());
    whole.insert(whole.end(), st.begin(), st.begin() + (long) n);
    whole.insert(whole.end(), ids.begin(), ids.end());
    ids.swap(whole);
    return true;
}

}  // namespace maya::ds4
