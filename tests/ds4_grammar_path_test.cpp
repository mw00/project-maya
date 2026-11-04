#include "ds4_grammar_path.hpp"
#include <cstdio>
int main() {
    const std::string path = "/Users/Andrey Name/грамматика.gbnf";
    std::string hex;
    const char digits[] = "0123456789abcdef";
    for (unsigned char c : path) { hex += digits[c >> 4]; hex += digits[c & 15]; }
    if (ds4_decode_grammar_path(hex) != path) return 1;
    for (const std::string bad : {"", "0", "xx", "00", "410042", "c080", "eda080", "f4908080", "e282"}) {
        bool rejected = false;
        try { ds4_decode_grammar_path(bad); } catch (const std::exception &) { rejected = true; }
        if (!rejected) { std::fprintf(stderr, "accepted invalid hex: %s\n", bad.c_str()); return 1; }
    }
    std::puts("grammar path: PASS spaces Unicode malformed NUL UTF-8");
}
