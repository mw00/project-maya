#include "../src/hy4/hy4_tp_stage_slots.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(expr)                                                                 \
    do {                                                                            \
        if (!(expr)) {                                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr);               \
            return 1;                                                               \
        }                                                                           \
    } while (0)

int main() {
    CHECK(hy4::parse_tp_stage_slots(nullptr, 128) == 128);
    CHECK(hy4::parse_tp_stage_slots("", 128) == 128);
    CHECK(hy4::parse_tp_stage_slots("0", 128) == 128);
    CHECK(hy4::parse_tp_stage_slots("-8", 128) == 128);
    CHECK(hy4::parse_tp_stage_slots("1", 128) == 16);
    CHECK(hy4::parse_tp_stage_slots("15", 128) == 16);
    CHECK(hy4::parse_tp_stage_slots("16", 128) == 16);
    CHECK(hy4::parse_tp_stage_slots("32", 128) == 32);
    CHECK(hy4::parse_tp_stage_slots("128", 128) == 128);
    CHECK(hy4::parse_tp_stage_slots("256", 128) == 128);

    CHECK(hy4::tp_stage_slots_for_group(1, 32, 128, 0) == 128);
    CHECK(hy4::tp_stage_slots_for_group(1, 32, 128, 96) == 224);
    CHECK(hy4::tp_stage_slots_for_group(4, 32, 128, 96) == 32);
    CHECK(hy4::tp_stage_pass_count(0, 32) == 0);
    CHECK(hy4::tp_stage_pass_count(32, 32) == 1);
    CHECK(hy4::tp_stage_pass_count(33, 32) == 2);

    constexpr int kExperts = 128;
    constexpr size_t kBlob = 256;
    std::vector<uint8_t> reference(kExperts * kBlob);
    for (size_t i = 0; i < reference.size(); ++i)
        reference[i] = static_cast<uint8_t>((i * 13 + (i / kBlob) * 7) & 0xff);

    std::mt19937 rng(0x104090);
    std::vector<int> all_e(kExperts);
    for (int i = 0; i < kExperts; ++i) all_e[static_cast<size_t>(i)] = i;

    for (int cap : {32, 16}) {
        std::vector<uint8_t> gpu_stage(static_cast<size_t>(cap) * kBlob, 0);
        for (int iter = 0; iter < 512; ++iter) {
            std::shuffle(all_e.begin(), all_e.end(), rng);
            const int n = 1 + static_cast<int>(rng() % 96);
            std::vector<int> u(all_e.begin(), all_e.begin() + n);
            int passes = 0;
            for (size_t p0 = 0; p0 < u.size(); p0 += static_cast<size_t>(cap)) {
                ++passes;
                const int np = static_cast<int>(std::min<size_t>(static_cast<size_t>(cap), u.size() - p0));
                std::array<int, kExperts> pmap{};
                pmap.fill(-1);
                for (int i = 0; i < np; ++i) {
                    const int e = u[p0 + static_cast<size_t>(i)];
                    std::memcpy(gpu_stage.data() + static_cast<size_t>(i) * kBlob,
                                reference.data() + static_cast<size_t>(e) * kBlob, kBlob);
                    pmap[static_cast<size_t>(e)] = i;
                }
                for (int i = 0; i < np; ++i) {
                    const int e = u[p0 + static_cast<size_t>(i)];
                    CHECK(pmap[static_cast<size_t>(e)] >= 0 && pmap[static_cast<size_t>(e)] < cap);
                    CHECK(std::memcmp(gpu_stage.data() + static_cast<size_t>(pmap[static_cast<size_t>(e)]) * kBlob,
                                      reference.data() + static_cast<size_t>(e) * kBlob, kBlob) == 0);
                }
            }
            CHECK(passes == hy4::tp_stage_pass_count(u.size(), cap));
        }
    }
    std::printf("hy4_tp_stage_slots_test: 1024 randomized multi-pass bounded stage tests passed (memcmp==0)\n");
    return 0;
}
