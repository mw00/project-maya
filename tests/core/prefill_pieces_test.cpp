// prefill_pieces (STRATA_GLM_PREFILL_INTERLEAVE): the pieces a prompt is read in - every token once, in order, no piece
// but a whole segment shorter than the minimum.  CPU only.
#include "strata/core/prefill_pieces.hpp"

#include <cstdio>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
// the call ends a prompt [from, end) is read in, marks kept (as generate.cpp's prep_prompt walks them)
std::vector<size_t> walk(size_t from, size_t end, const std::vector<size_t>& marks, size_t piece) {
    std::vector<size_t> out;
    size_t at = from, mi = 0;
    while (at < end) {
        const size_t seg = mi < marks.size() ? marks[mi] : end;
        const size_t e = strata::core::prefill_piece_end(at, seg, piece);
        out.push_back(e);
        if (e == seg && mi < marks.size()) ++mi;
        at = e;
    }
    return out;
}
}  // namespace

int main() {
    using strata::core::prefill_piece_end;
    // off: the whole segment
    expect(prefill_piece_end(0, 10000, 0) == 10000, "off: one call");
    expect(prefill_piece_end(100, 1000, 2048) == 1000, "shorter than a piece: one call");
    // pieces
    expect(prefill_piece_end(0, 10000, 2048) == 2048, "first piece");
    expect(prefill_piece_end(8192, 10000, 2048) == 10000, "last piece takes the rest (1808)");
    expect(prefill_piece_end(0, 2100, 2048) == 2100, "a 52-token rest goes with the piece");
    expect(prefill_piece_end(0, 2112, 2048) == 2048, "a 64-token rest is its own piece");
    // a walk: every call ends past the last one, the marks are call ends, the last end is the prompt's
    {
        const std::vector<size_t> marks = {3000, 20000};
        const auto ends = walk(17, 30001, marks, 2048);
        bool ok = !ends.empty() && ends.back() == 30001;
        size_t prev = 17;
        for (const size_t e : ends) {
            ok = ok && e > prev && (e - prev >= 64 || e == 30001);
            prev = e;
        }
        for (const size_t m : marks) {
            bool hit = false;
            for (const size_t e : ends) hit = hit || e == m;
            ok = ok && hit;
        }
        expect(ok, "walk with marks: ordered, marks kept, pieces >= 64");
        const auto off = walk(17, 30001, marks, 0);
        expect(off == std::vector<size_t>({3000, 20000, 30001}), "walk off: the marks and the end, as before");
    }
    // the setting
    using strata::core::prefill_interleave_parse;
    expect(prefill_interleave_parse(nullptr).tokens == 0, "unset: off");
    expect(prefill_interleave_parse("0").tokens == 0, "0: off");
    expect(prefill_interleave_parse("2048").tokens == 2048 && prefill_interleave_parse("2048").rounds == 1,
           "2048: one round");
    expect(prefill_interleave_parse("1024,4").rounds == 4, "1024,4: four rounds");
    expect(prefill_interleave_parse("16").tokens == 256, "a piece is 256 at least");
    expect(prefill_interleave_parse("2048").exact && !prefill_interleave_parse("2048,1,cut").exact, "exact unless ,cut");
    using strata::core::prefill_aligned_piece;
    expect(prefill_aligned_piece(2048, 8128) == 8128, "exact: one whole chunk at least");
    expect(prefill_aligned_piece(16384, 8128) == 16256, "exact: two chunks");
    expect(prefill_aligned_piece(2048, 0) == 2048, "no chunk: as asked");
    if (fails == 0) std::printf("prefill_pieces_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
