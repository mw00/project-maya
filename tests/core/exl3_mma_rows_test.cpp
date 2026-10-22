// exl3_mma_rows (STRATA_GLM_EXL3_MMA_ROWS_SHORT): the parse, and the rows an expert takes through the tensor-core
// kernels by chunk length - the base (STRATA_GLM_EXL3_MMA_ROWS) unless the chunk is short and the setting is on.
// CPU only.
#include "strata/core/exl3_mma_rows.hpp"

#include <cstdio>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
}  // namespace

int main() {
    using namespace strata::core;
    const MmaRowsShort off = mma_rows_short_parse(nullptr), on = mma_rows_short_parse("192"),
                       at = mma_rows_short_parse("256,2048");
    expect(off.rows == 0 && mma_rows_short_parse("").rows == 0 && mma_rows_short_parse("0").rows == 0 &&
               mma_rows_short_parse("-5").rows == 0,
           "unset, empty, 0 and negative: off");
    expect(on.rows == 192 && on.max_tokens == 4096, "rows alone: chunks up to 4096 tokens");
    expect(at.rows == 256 && at.max_tokens == 2048, "rows and the longest chunk");
    expect(mma_rows_short_parse("192,0").max_tokens == 4096, "a chunk length of 0 keeps 4096");
    expect(mma_rows_for(1024, 64, off) == 64, "off: the base in every chunk");
    expect(mma_rows_for(1024, 64, on) == 192 && mma_rows_for(4096, 64, on) == 192, "a short chunk: the setting's rows");
    expect(mma_rows_for(4097, 64, on) == 64, "a longer chunk: the base");
    expect(mma_rows_for(0, 64, on) == 64, "an unknown chunk length: the base");
    expect(mma_rows_for(3000, 64, at) == 64 && mma_rows_for(2048, 64, at) == 256, "the chunk limit given");
    expect(mma_rows_for(1024, 0, off) == 0, "base 0 (no tensor-core path) stays 0 when off");
    if (fails == 0) std::printf("exl3_mma_rows_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
