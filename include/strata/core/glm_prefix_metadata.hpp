#pragma once

#include "strata/core/glm_prefix_admission.hpp"

#include <array>
#include <istream>
#include <ostream>
#include <string>

namespace strata::core::glm_prefix {

// Separate file names and metadata magic keep these files out of older slot readers.
inline constexpr uint64_t metadata_magic = 0x3148434e52424d47ull; // GMBRNCH1
inline constexpr uint64_t hash_start = 14695981039346656037ull;
inline constexpr size_t max_identity_bytes = 16u << 10;

inline uint64_t hash_bytes(uint64_t hash, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

struct Metadata {
    std::vector<int32_t> tokens;
    uint64_t bytes = 0, checksum = 0;
};

inline bool write_metadata(std::ostream& out, const std::string& identity, std::span<const int32_t> tokens,
                           uint64_t bytes, uint64_t checksum) {
    const uint64_t fields[] = {metadata_magic, identity.size(), bytes, tokens.size(), checksum};
    if (identity.empty() || identity.size() > max_identity_bytes ||
        !bytes || bytes > state_budget || tokens.size() < min_tokens || tokens.size() > max_state_tokens) return false;
    const uint64_t tag = hash_bytes(hash_bytes(hash_bytes(hash_start, fields, sizeof(fields)),
                                                identity.data(), identity.size()),
                                    tokens.data(), tokens.size() * sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(fields), sizeof(fields));
    out.write(identity.data(), identity.size());
    out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    return bool(out);
}

inline bool read_metadata(std::istream& in, const std::string& identity, size_t token_limit, Metadata& entry) {
    uint64_t fields[5]{};
    if (!in.read(reinterpret_cast<char*>(fields), sizeof(fields)) || fields[0] != metadata_magic ||
        !fields[1] || fields[1] != identity.size() || fields[1] > max_identity_bytes ||
        !fields[2] || fields[2] > state_budget || fields[3] < min_tokens ||
        fields[3] > std::min(token_limit, max_state_tokens)) return false;
    std::string stored(identity.size(), '\0');
    if (!in.read(stored.data(), stored.size()) || stored != identity) return false;
    Metadata candidate;
    candidate.tokens.resize(fields[3]);
    const size_t bytes = candidate.tokens.size() * sizeof(int32_t);
    uint64_t stored_tag = 0;
    if (!in.read(reinterpret_cast<char*>(candidate.tokens.data()), bytes) ||
        !in.read(reinterpret_cast<char*>(&stored_tag), sizeof(stored_tag)) ||
        in.peek() != std::char_traits<char>::eof() || in.bad()) return false;
    const auto tag = hash_bytes(hash_bytes(hash_bytes(hash_start, fields, sizeof(fields)),
                                           identity.data(), identity.size()), candidate.tokens.data(), bytes);
    if (tag != stored_tag) return false;
    candidate.bytes = fields[2];
    candidate.checksum = fields[4];
    entry = std::move(candidate);
    return true;
}

inline bool check_payload(std::istream& in, uint64_t size, uint64_t checksum) {
    std::array<char, 64u << 10> buffer;
    uint64_t hash = hash_start;
    while (size) {
        const auto count = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
        if (!in.read(buffer.data(), count)) return false;
        hash = hash_bytes(hash, buffer.data(), count);
        size -= count;
    }
    return hash == checksum && in.peek() == std::char_traits<char>::eof() && !in.bad();
}

}  // namespace strata::core::glm_prefix
