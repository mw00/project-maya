// tests/ds4_prompt_mem_test.cpp - the DeepSeek engine's prompt-path and memory rules (src/ds4/ds4_prompt_mem.hpp),
// each against the cases it was written for.  CPU only, no model.
#include "../src/ds4/ds4_prompt_mem.hpp"

#include <cstdio>

namespace d = strata::ds4;

static int fails = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                   \
        }                                                              \
    } while (0)

int main() {
    {   // switches: unset / empty -1, else 0 or 1
        CHECK(d::tri_of(nullptr) == -1 && d::tri_of("") == -1);
        CHECK(d::tri_of("0") == 0 && d::tri_of("1") == 1 && d::tri_of("2") == 1);
    }
    {   // the path by length: off unless asked, 96 by default, length:N moves it, never below one token
        CHECK(d::prompt_pick_length(nullptr) == 0);
        CHECK(d::prompt_pick_length("") == 0);
        CHECK(d::prompt_pick_length("adaptive") == 0);
        CHECK(d::prompt_pick_length("lengthy") == 0);
        CHECK(d::prompt_pick_length("length:") == 0);
        CHECK(d::prompt_pick_length("length") == 96);
        CHECK(d::prompt_pick_length("length:400") == 400);
        CHECK(d::prompt_pick_length("length:0") == 1 && d::prompt_pick_length("length:-5") == 1);
        const int m = d::prompt_pick_length("length");
        for (int rep = 0; rep < 3; ++rep) {   // (the same prompt, the same path, every time)
            CHECK(!d::prompt_by_length(2, m));
            CHECK(!d::prompt_by_length(28, m));
            CHECK(!d::prompt_by_length(95, m));
            CHECK(d::prompt_by_length(96, m));
            CHECK(d::prompt_by_length(8192, m));
        }
    }
    if (fails == 0) std::printf("ds4 prompt / memory rules: ok\n");
    return fails == 0 ? 0 : 1;
}
