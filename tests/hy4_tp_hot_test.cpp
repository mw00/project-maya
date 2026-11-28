#include "../src/hy4/hy4_tp_hot.hpp"

#include <algorithm>
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
    constexpr int kTP = 4;
    constexpr int kExperts = 64;
    constexpr int kPerGpuSlots = 12;
    constexpr size_t kBlob = 192;

    std::vector<uint8_t> reference(kExperts * kBlob);
    for (size_t i = 0; i < reference.size(); ++i)
        reference[i] = static_cast<uint8_t>((i * 19 + (i / kBlob) * 29) & 0xff);

    std::vector<std::vector<int>> slot_of(kTP, std::vector<int>(kExperts, -1));
    std::vector<std::vector<uint8_t>> vram(kTP, std::vector<uint8_t>(kPerGpuSlots * kBlob, 0));
    std::vector<char> lane_e(kExperts, 0);
    std::vector<std::vector<char>> stream_e(kTP, std::vector<char>(kExperts, 0));

    for (int e = 0; e < kTP * kPerGpuSlots; ++e) {
        const int m = e / kPerGpuSlots, sl = e % kPerGpuSlots;
        slot_of[static_cast<size_t>(m)][static_cast<size_t>(e)] = sl;
        std::memcpy(vram[static_cast<size_t>(m)].data() + static_cast<size_t>(sl) * kBlob,
                    reference.data() + static_cast<size_t>(e) * kBlob, kBlob);
    }
    for (int e = kTP * kPerGpuSlots; e < kExperts; ++e) {
        const int k = e % kTP;
        lane_e[static_cast<size_t>(e)] = 1;
        stream_e[static_cast<size_t>(k)][static_cast<size_t>(e)] = 1;
    }

    // Threshold boundary check: equal scores below ratio*res + margin must produce zero swaps.
    std::vector<float> flat_scores(kExperts, 10.0f);
    CHECK(hy4::plan_tp_hot_layer(0, flat_scores.data(), kExperts, slot_of, lane_e, stream_e, 1.25, 2.0).empty());

    std::mt19937 rng(0x4102026);
    std::vector<float> scores(kExperts, 0.0f);
    for (int round = 0; round < 500; ++round) {
        for (int e = 0; e < kExperts; ++e) scores[static_cast<size_t>(e)] = static_cast<float>(rng() % 1000);
        auto plan = hy4::plan_tp_hot_layer(5, scores.data(), kExperts, slot_of, lane_e, stream_e, 1.0, 0.0);
        const size_t n_apply = std::min<size_t>(plan.size(), 4);
        for (size_t i = 0; i < n_apply; ++i) {
            const auto& sw = plan[i];
            const int sl = hy4::apply_tp_hot_layer_swap(
                sw.out_e, sw.in_e, slot_of[static_cast<size_t>(sw.gpu_rank)],
                lane_e, stream_e[static_cast<size_t>(sw.cpu_rank)]);
            CHECK(sl >= 0 && sl < kPerGpuSlots);
            std::memcpy(vram[static_cast<size_t>(sw.gpu_rank)].data() + static_cast<size_t>(sl) * kBlob,
                        reference.data() + static_cast<size_t>(sw.in_e) * kBlob, kBlob);
        }

        for (int e = 0; e < kExperts; ++e) {
            int gcnt = 0, ccnt = 0;
            for (int m = 0; m < kTP; ++m) {
                const int sl = slot_of[static_cast<size_t>(m)][static_cast<size_t>(e)];
                if (sl >= 0) {
                    ++gcnt;
                    CHECK(std::memcmp(vram[static_cast<size_t>(m)].data() + static_cast<size_t>(sl) * kBlob,
                                      reference.data() + static_cast<size_t>(e) * kBlob, kBlob) == 0);
                }
                if (stream_e[static_cast<size_t>(m)][static_cast<size_t>(e)]) ++ccnt;
            }
            CHECK((gcnt == 1 && ccnt == 0 && lane_e[static_cast<size_t>(e)] == 0) ||
                  (gcnt == 0 && ccnt == 1 && lane_e[static_cast<size_t>(e)] == 1));
        }
    }
    std::printf("hy4_tp_hot_test: 500 randomized multi-GPU swaps (~2000 swaps) passed invariants & memcmp==0\n");
    return 0;
}
