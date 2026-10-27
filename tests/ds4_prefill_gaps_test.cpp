#include "../src/ds4/ds4_prefill_gaps.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

using strata::ds4::PrefillGaps;
using strata::ds4::prefill_gap_budget_ms;
using strata::ds4::prefill_gap_due;
using strata::ds4::prefill_gaps_parse;

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    // Unset or malformed is off; 0 is off; the share is clamped to 90 and the cap may be given.
    CHECK(prefill_gaps_parse(nullptr).percent == 0);
    CHECK(prefill_gaps_parse("").percent == 0);
    CHECK(prefill_gaps_parse("0").percent == 0);
    CHECK(prefill_gaps_parse("nonsense").percent == 0);
    CHECK(prefill_gaps_parse("5").percent == 5 && prefill_gaps_parse("5").max_ms == 4000.0);
    CHECK(prefill_gaps_parse("50,1500").percent == 50 && prefill_gaps_parse("50,1500").max_ms == 1500.0);
    CHECK(prefill_gaps_parse("95").percent == 90);
    CHECK(prefill_gaps_parse("20,0").percent == 20 && prefill_gaps_parse("20,0").max_ms == 4000.0);

    // The gap is percent / (100 - percent) of the stage, capped, and one decode round at least is the credit.
    PrefillGaps g50 = prefill_gaps_parse("50");
    CHECK(prefill_gap_budget_ms(g50, 4000.0) == 4000.0);
    CHECK(prefill_gap_budget_ms(g50, 2000.0) == 2000.0);
    PrefillGaps g20 = prefill_gaps_parse("20,1500");
    CHECK(prefill_gap_budget_ms(g20, 4000.0) == 1000.0);
    PrefillGaps g90 = prefill_gaps_parse("90,4000");
    CHECK(prefill_gap_budget_ms(g90, 1000.0) == 4000.0);
    CHECK(prefill_gap_budget_ms(g90, 100.0) == 900.0);

    CHECK(prefill_gap_due(g50, 4000.0, 0.0));
    CHECK(prefill_gap_due(g50, 4000.0, 8000.0));      // 4000 >= half of 8000
    CHECK(!prefill_gap_due(g50, 4000.0, 8100.0));     // 4000 < 4050
    CHECK(!prefill_gap_due(g50, 1000.0, 4000.0));     // 1000 < 2000: the next boundary earns more
    CHECK(prefill_gap_due(g50, 2000.0, 4000.0));      // 2000 >= 2000
    CHECK(!prefill_gap_due(PrefillGaps{}, 4000.0, 0.0));

    // Per-lane buffer arithmetic: a writer and a reader that take lane `s` get the same pointer, and a
    // different lane gets a different one (the layout is [lane][kWin rows of compressed KV][kWin rows of keys]).
    const size_t kWin = 16, ckv = 128, ik = 64, bounce = 256, eg = 1024;
    std::vector<uint8_t> x_stage_l((size_t) 2 * kWin * (ckv + ik));
    std::vector<unsigned> x_flag_l((size_t) 2 * 32);
    std::vector<uint8_t> hand_l((size_t) 2 * bounce);
    std::vector<unsigned> hand_flag_l((size_t) 2 * 32);
    std::vector<uint8_t> h_eg_l((size_t) 2 * eg);
    auto stage = [&](int lane) { return x_stage_l.data() + (size_t) lane * kWin * (ckv + ik); };
    auto sflag = [&](int lane) { return x_flag_l.data() + (size_t) lane * 32; };
    auto hand = [&](int lane) { return hand_l.data() + (size_t) lane * bounce; };
    auto hflag = [&](int lane) { return hand_flag_l.data() + (size_t) lane * 32; };
    auto eng = [&](int lane) { return h_eg_l.data() + (size_t) lane * eg; };
    CHECK(stage(0) == x_stage_l.data() && stage(1) == x_stage_l.data() + 16 * 192);
    CHECK(stage(0) != stage(1));
    CHECK(sflag(0) == x_flag_l.data() && sflag(1) == x_flag_l.data() + 32);
    CHECK(hand(0) == hand_l.data() && hand(1) == hand_l.data() + 256);
    CHECK(hflag(0) == hand_flag_l.data() && hflag(1) == hand_flag_l.data() + 32);
    CHECK(eng(0) == h_eg_l.data() && eng(1) == h_eg_l.data() + 1024);

    std::puts("prefill gaps: parse, budget, credit and per-lane buffer offsets passed");
    return 0;
}
