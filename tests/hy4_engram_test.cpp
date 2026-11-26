// maya-hy4's Engram lookup switches (src/hy4/hy4_engram.hpp), without a GPU or a model file.
#include "../src/hy4/hy4_engram.hpp"

#include <cstdio>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    namespace e = hy4::engram;
    // MADV_RANDOM: on by default and for any value but an explicit 0
    CHECK(e::random_enabled(nullptr));
    CHECK(e::random_enabled(""));
    CHECK(e::random_enabled("1"));
    CHECK(e::random_enabled("2"));
    CHECK(!e::random_enabled("0"));
    // the crew: 8 by default, clamped to 1..64
    CHECK(e::threads(nullptr) == 8);
    CHECK(e::threads("") == 8);
    CHECK(e::threads("1") == 1);
    CHECK(e::threads("32") == 32);
    CHECK(e::threads("64") == 64);
    CHECK(e::threads("100") == 64);
    CHECK(e::threads("0") == 1);
    CHECK(e::threads("-9") == 1);
    return 0;
}
