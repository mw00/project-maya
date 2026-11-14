#include "strata/core/state_cache.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>

#ifdef STRATA_STATE_OPENSSL
#include <openssl/evp.h>
#endif

namespace strata::core::statecache {
std::string content_digest(const void* data, size_t bytes) {
    std::array<uint8_t, 32> digest{};
#ifdef STRATA_STATE_OPENSSL
    unsigned size = 0;
    if (EVP_Digest(data, bytes, digest.data(), &size, EVP_sha256(), nullptr) != 1 || size != digest.size())
        throw std::runtime_error("state content digest failed");
#else
    constexpr std::array<uint32_t, 64> constants = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t, 8> state = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const auto block = [&](const uint8_t* input) {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | input[4 * i + j];
        for (int i = 16; i < 64; ++i) {
            const uint32_t a = w[i - 15], b = w[i - 2];
            w[i] = w[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) + w[i - 7] +
                (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
        }
        auto h = state;
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h[7] + (std::rotr(h[4], 6) ^ std::rotr(h[4], 11) ^ std::rotr(h[4], 25)) +
                ((h[4] & h[5]) ^ (~h[4] & h[6])) + constants[i] + w[i];
            const uint32_t t2 = (std::rotr(h[0], 2) ^ std::rotr(h[0], 13) ^ std::rotr(h[0], 22)) +
                ((h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]));
            for (int j = 7; j > 0; --j) h[j] = h[j - 1];
            h[4] += t1; h[0] = t1 + t2;
        }
        for (int i = 0; i < 8; ++i) state[i] += h[i];
    };
    const auto* input = static_cast<const uint8_t*>(data);
    const size_t whole = bytes / 64;
    for (size_t i = 0; i < whole; ++i) block(input + 64 * i);
    std::array<uint8_t, 128> tail{};
    const size_t remaining = bytes % 64, blocks = remaining < 56 ? 1 : 2;
    if (remaining) std::memcpy(tail.data(), input + whole * 64, remaining);
    tail[remaining] = 0x80;
    const uint64_t bits = uint64_t(bytes) * 8;
    for (int i = 0; i < 8; ++i) tail[blocks * 64 - 1 - i] = uint8_t(bits >> (8 * i));
    for (size_t i = 0; i < blocks; ++i) block(tail.data() + i * 64);
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) digest[4 * i + j] = uint8_t(state[i] >> (24 - 8 * j));
#endif
    constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        result[2 * i] = hex[digest[i] >> 4]; result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}
}  // namespace strata::core::statecache
