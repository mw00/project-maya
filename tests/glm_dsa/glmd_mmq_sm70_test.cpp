// tests/glm_dsa/glmd_mmq_sm70_test.cpp - B14.2 / B9.8 on the host: for every type the prompt path sends to MMQ
// (strata::prefill::mmq::supported), every tile width the host switch can pick on a V100 (cc 700, 96 KB opt-in
// shared memory) has a real sm_70 instance in this build (llama.cpp 1163af96: a host-picked mmq_x whose device code
// was NO_DEVICE_CODE gave silent zeros), for chunks of 1..8192 columns; and the B14.1 clamp sees the scale layout.
// CPU only: no device is touched.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_mmq_strata.hpp"

#include <cstdio>
#include <string>

int main() {
    namespace ma = strata::prefill::mmq_strata;
    int fail = 0, n = 0;
    for (int t = 0; t < 64; ++t) {
        if (!strata::prefill::mmq::supported(t)) continue;
        ++n;
        std::string why;
        const bool ok = ma::sm70_check(t, 700, 98304, 8192, why);
        std::printf("type %2d: sm_70 instances for every host-picked J, 1..8192 columns: %s %s (scale halves %d)\n", t,
                    ok ? "ok" : "FAIL", why.c_str(), ma::ds_halves(t));
        fail += !ok;
    }
    const bool blk = ma::block_bytes() == 144 && ma::blocks(1, 6144) == 48 && ma::blocks(3, 100) == 12;
    std::printf("block_q8_1_mmq 144 B, 128 values a block, rows padded to 512: %s\n", blk ? "ok" : "FAIL");
    fail += !blk;
    std::printf("%s (%d types, %d failures)\n", fail ? "FAILED" : "PASSED", n, fail);
    return fail ? 1 : 0;
}
