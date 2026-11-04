#pragma once
#include <stdexcept>
#include <string>

// The line protocol separates fields with spaces. Encode UTF-8 path bytes rather
// than quoting them, so filenames containing spaces need no shell-style parser.
inline std::string ds4_decode_grammar_path(const std::string & hex) {
    if (hex.empty() || hex.size() % 2) throw std::runtime_error("empty or odd grammar_hex");
    const auto nibble = [](unsigned char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error("invalid grammar_hex digit");
    };
    std::string path;
    path.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        const char c = char((nibble(hex[i]) << 4) | nibble(hex[i + 1]));
        if (!c) throw std::runtime_error("NUL in grammar_hex");
        path.push_back(c);
    }
    // Reject malformed UTF-8, overlong forms, surrogates and values past U+10FFFF.
    for (size_t i = 0; i < path.size();) {
        const auto c = (unsigned char) path[i++];
        if (c < 0x80) continue;
        int n; unsigned cp, min;
        if (c >= 0xc2 && c <= 0xdf) { n = 1; cp = c & 31; min = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { n = 2; cp = c & 15; min = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { n = 3; cp = c & 7; min = 0x10000; }
        else throw std::runtime_error("invalid UTF-8 grammar_hex");
        if (path.size() - i < (size_t) n) throw std::runtime_error("truncated UTF-8 grammar_hex");
        while (n--) {
            const auto b = (unsigned char) path[i++];
            if ((b & 0xc0) != 0x80) throw std::runtime_error("invalid UTF-8 grammar_hex continuation");
            cp = (cp << 6) | (b & 63);
        }
        if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
            throw std::runtime_error("invalid UTF-8 grammar_hex scalar");
    }
    return path;
}
