// tests/mm3_slots_test.cpp - which idle conversation a request takes under --parallel (mm3_slots.hpp)
#include <cstdio>
#include <vector>

#include "mm3_slots.hpp"

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    using mm3::SlotView;
    using mm3::pick_slot;
    // {busy, from, size, used}
    // a short text question shares the chat template (186 tokens) with a long conversation about a picture: the
    // empty conversation takes it, the picture's cache stays
    CHECK(pick_slot({{false, 186, 3400, 5}, {false, 0, 0, 0}}) == 1);
    // ... and with no empty one, the least recently used goes - the template alone decides nothing
    CHECK(pick_slot({{false, 186, 3400, 5}, {false, 186, 900, 7}}) == 0);
    CHECK(pick_slot({{false, 186, 3400, 9}, {false, 186, 900, 7}}) == 1);
    // the conversation continued: taken whatever else is free
    CHECK(pick_slot({{false, 0, 0, 0}, {false, 3400, 3400, 5}}) == 1);
    CHECK(pick_slot({{false, 0, 0, 0}, {false, 3399, 3400, 5}}) == 1);   // (its last token read again)
    // a long shared start (an agent's system prompt) still counts, though it drops more
    CHECK(pick_slot({{false, 0, 0, 0}, {false, 9000, 30000, 5}}) == 1);
    // a short shared start that drops little (a short conversation) is still reused
    CHECK(pick_slot({{false, 0, 0, 0}, {false, 186, 300, 5}}) == 1);
    // busy ones are never taken; all busy: -1
    CHECK(pick_slot({{true, 3400, 3400, 1}, {false, 0, 0, 0}}) == 1);
    CHECK(pick_slot({{true, 0, 0, 0}, {true, 0, 0, 0}}) == -1);
    // empty ones: the least recently used first (as before)
    CHECK(pick_slot({{false, 0, 0, 4}, {false, 0, 0, 2}, {false, 0, 0, 3}}) == 1);
    // the longest reuse wins among the counted ones
    CHECK(pick_slot({{false, 600, 700, 1}, {false, 2000, 2100, 9}}) == 1);
    std::printf("mm3_slots_test: ok\n");
    return 0;
}
