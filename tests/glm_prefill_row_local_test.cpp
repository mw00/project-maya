#include "glm_prefill_row_local.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#ifdef MAYA_ROW_LOCAL_NATIVE_TEST
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"

#include <cmath>
#include <cstring>
#endif

namespace glmfast = strata::core::glmfast;

static void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

static thread_local int worker_node = 0;

struct Pool {
    std::vector<int> nodes;

    template<class Job>
    void run(int jobs, const Job& job) const {
        if (nodes.size() == 1) {
            worker_node = nodes.front();
            for (int index = 0; index < jobs; ++index) job(index);
            return;
        }
        std::atomic<int> next{0};
        std::vector<std::thread> threads;
        for (int node : nodes) {
            threads.emplace_back([&, node] {
                worker_node = node;
                for (int index = next.fetch_add(1); index < jobs; index = next.fetch_add(1)) job(index);
            });
        }
        for (auto& thread : threads) thread.join();
    }
};

static void check_dispatch(const Pool& pool, int rows, int block, int low, int begin, int end) {
    const std::vector<int> counts{1, 4, 2, 7, 3};
    std::vector<int> offsets{0};
    for (int count : counts) offsets.push_back(offsets.back() + count);
    const int experts = end - begin;
    const int jobs = experts * (rows / block);
    constexpr int sentinel = -1;
    std::vector<std::atomic<int>> output(offsets.back() * rows + 2);
    std::vector<std::atomic<int>> visits(jobs);
    for (auto& v : output) v.store(sentinel);
    for (auto& v : visits) v.store(0);
    glmfast::run_prefill_row_blocks(pool, experts, rows, block, low, [] { return worker_node; }, [&](int job) {
        check(job >= 0 && job < jobs && visits[job].fetch_add(1) == 0, "duplicate or invalid job");
        const int expert = begin + job % experts;
        const int row = job / experts * block;
        check(row + block <= rows, "row block crosses end");
        for (int token = offsets[expert]; token < offsets[expert + 1]; ++token) {
            for (int column = row; column < row + block; ++column) {
                check(output[1 + token * rows + column].exchange(expert) == sentinel, "overlapping tile writes");
            }
        }
    });
    for (auto& count : visits) check(count.load() == 1, "missing job");
    check(output.front().load() == sentinel && output.back().load() == sentinel, "guard overwritten");
    for (int expert = 0; expert < static_cast<int>(counts.size()); ++expert) {
        for (int token = offsets[expert]; token < offsets[expert + 1]; ++token) {
            for (int row = 0; row < rows; ++row) {
                const int expected = begin <= expert && expert < end && row < rows / block * block ? expert : sentinel;
                check(output[1 + token * rows + row].load() == expected, "unexpected row coverage");
            }
        }
    }
}

#ifdef MAYA_ROW_LOCAL_NATIVE_TEST
namespace cpu = strata::kernels::cpu;

static void check_native_parity(ggml_type gu_type, ggml_type down_type, bool packed) {
    constexpr int width = 256, gate_block = 32, down_block = 64, experts = 3;
    constexpr float sentinel = -123456.0f, limit = 7.0f;
    constexpr unsigned char byte_sentinel = 0xA5;
    const int offsets[]{0, 1, 13, 30};  // experts with 1, 12 and 17 token rows
    const Pool pool{{2, 7, 2, 7}};
    cpu::NativeFmt format;
    std::string error;
    check(cpu::native_fmt(gu_type, down_type, width, width, format, error), "native format unavailable");
    check(!packed || (cpu::kq_type_ok(gu_type) && cpu::kq_type_ok(down_type) &&
                      format.gu_act == 15 && format.d_act == 15), "packed format unavailable");
    check(!ggml_quantize_requires_imatrix(gu_type) && !ggml_quantize_requires_imatrix(down_type),
          "fixture does not supply an importance matrix");
    std::vector<std::vector<unsigned char>> blobs(experts, std::vector<unsigned char>(format.bytes));
    std::vector<float> weights(width * width);
    for (int expert = 0; expert < experts; ++expert) {
        for (std::size_t offset : {std::size_t{0}, format.up_off, format.down_off}) {
            const ggml_type weight_type = offset == format.down_off ? down_type : gu_type;
            for (std::size_t i = 0; i < weights.size(); ++i)
                weights[i] = (static_cast<int>((i * 13 + offset / 64 + expert * 5) % 127) - 63) * 0.013f;
            // The public quantizer initializes the selected format before producing the weight blocks.
            const std::size_t written = ggml_quantize_chunk(weight_type, weights.data(), blobs[expert].data() + offset,
                                                            0, width, width, nullptr);
            check(written == width * ggml_row_size(weight_type, width), "unexpected quantized weight size");
        }
    }
    const int total_rows = offsets[experts];
    std::vector<unsigned char> act(total_rows * format.act_bytes);
    std::vector<float> input(width);
    for (int row = 0; row < total_rows; ++row) {
        for (int i = 0; i < width; ++i) input[i] = ((i * 17 + row * 7) % 109 - 54) * 0.017f;
        cpu::native_quant_act(format, input.data(), act.data() + row * format.act_bytes);
    }

    struct Buffers {
        std::vector<float> ff, out, first_batch;
        std::vector<unsigned char> hq;
    };
    const auto run = [&](int low_node) {
        Buffers b{std::vector<float>(total_rows * width + 2, sentinel),
                  std::vector<float>(total_rows * width + 2, sentinel), {},
                  std::vector<unsigned char>(total_rows * format.h_bytes + 128, byte_sentinel)};
        auto* hq = b.hq.data() + 64;
        const auto dispatch = [&](int begin, int end, int block, bool down) {
            const int count = end - begin;
            std::vector<std::vector<unsigned char>> packs(experts);
            std::vector<const void*> pack_ptr(experts, nullptr);
            const std::size_t stride = down ? format.h_bytes : format.act_bytes;
            const unsigned char* quantized = down ? hq : act.data();
            for (int expert = begin; packed && expert < end; ++expert) {
                const int tokens = offsets[expert + 1] - offsets[expert];
                if (tokens < 12) continue;
                packs[expert].resize(cpu::kq_act_bytes(tokens, width) + 63);
                auto* aligned = reinterpret_cast<unsigned char*>(
                    (reinterpret_cast<std::uintptr_t>(packs[expert].data()) + 63u) & ~std::uintptr_t{63});
                std::vector<const void*> rows(tokens);
                for (int t = 0; t < tokens; ++t) rows[t] = quantized + (offsets[expert] + t) * stride;
                cpu::kq_pack_act(rows.data(), tokens, width, aligned);
                pack_ptr[expert] = aligned;
            }
            glmfast::run_prefill_row_blocks(pool, count, width, block, low_node, [] { return worker_node; }, [&](int job) {
                const int expert = begin + job % count;
                const int first = job / count * block;
                const int tokens = offsets[expert + 1] - offsets[expert];
                const auto* blob = blobs[expert].data();
                std::vector<const void*> rows(tokens);
                std::vector<float*> output(tokens);
                for (int t = 0; t < tokens; ++t) {
                    rows[t] = quantized + (offsets[expert] + t) * stride;
                    output[t] = (down ? b.out : b.ff).data() + 1 + (offsets[expert] + t) * width;
                }
                if (pack_ptr[expert] != nullptr && down) {
                    for (auto*& ptr : output) ptr += first;
                    cpu::kq_rows_packed(down_type, blob + format.down_off + first * format.d_row, format.d_row, width,
                                        pack_ptr[expert], tokens, output.data(), 0, block);
                } else if (pack_ptr[expert] != nullptr) {
                    std::vector<float> values(2 * tokens * block);
                    std::vector<float*> gate(tokens), up(tokens);
                    for (int t = 0; t < tokens; ++t) {
                        gate[t] = values.data() + t * block;
                        up[t] = values.data() + (tokens + t) * block;
                    }
                    cpu::kq_rows_packed(gu_type, blob + first * format.gu_row, format.gu_row, width,
                                        pack_ptr[expert], tokens, gate.data(), 0, block);
                    cpu::kq_rows_packed(gu_type, blob + format.up_off + first * format.gu_row, format.gu_row, width,
                                        pack_ptr[expert], tokens, up.data(), 0, block);
                    for (int t = 0; t < tokens; ++t) {
                        for (int r = 0; r < block; ++r) {
                            const float g = std::fmin(gate[t][r], limit);
                            const float u = std::fmin(std::fmax(up[t][r], -limit), limit);
                            output[t][first + r] = (g / (1.f + std::exp(-g))) * u;
                        }
                    }
                } else if (down) {
                    cpu::native_down_rows_split(format, blob + format.down_off, rows.data(), tokens,
                                                output.data(), first, first + block);
                } else {
                    cpu::native_gu_rows_split(format, blob, blob + format.up_off, rows.data(), tokens,
                                              output.data(), first, first + block, limit);
                }
            });
        };
        dispatch(0, experts, gate_block, false);
        for (int row = 0; row < total_rows; ++row)
            cpu::native_quant_h(format, b.ff.data() + 1 + row * width, hq + row * format.h_bytes);
        dispatch(0, 2, down_block, true);
        check(std::all_of(b.out.begin() + 1 + offsets[2] * width, b.out.end(),
                          [sentinel](float value) { return value == sentinel; }), "first down batch touched second batch");
        b.first_batch.assign(b.out.begin() + 1, b.out.begin() + 1 + offsets[2] * width);
        dispatch(2, experts, down_block, true);
        check(std::equal(b.first_batch.begin(), b.first_batch.end(), b.out.begin() + 1),
              "second down batch touched first batch");
        for (const auto* floats : {&b.ff, &b.out}) {
            check(floats->front() == sentinel && floats->back() == sentinel, "float sentinel overwritten");
            check(std::all_of(floats->begin() + 1, floats->end() - 1,
                              [sentinel](float value) { return std::isfinite(value) && value != sentinel; }), "invalid output");
        }
        check(std::all_of(b.hq.begin(), b.hq.begin() + 64, [byte_sentinel](auto v) { return v == byte_sentinel; }) &&
              std::all_of(b.hq.end() - 64, b.hq.end(), [byte_sentinel](auto v) { return v == byte_sentinel; }), "quant sentinel overwritten");
        return b;
    };
    const Buffers ordinary = run(-1), local = run(2);
    check(std::memcmp(ordinary.ff.data(), local.ff.data(), ordinary.ff.size() * sizeof(float)) == 0 &&
          ordinary.hq == local.hq &&
          std::memcmp(ordinary.out.data(), local.out.data(), ordinary.out.size() * sizeof(float)) == 0,
          "row-local arithmetic differs bitwise");
}
#endif

int main() {
    using glmfast::row_halves_has_slot;
    unsigned char slots[128]{};
    const glmfast::RowHalvesPlacement placement{64, 128, 256, 2, 7};
    check(placement.matches(64, 128, 256), "live placement rejected");
    check(!placement.matches(64, 128, 288) && !glmfast::RowHalvesPlacement{}.matches(64, 128, 256),
          "mismatched or missing placement accepted");
    check(!glmfast::RowHalvesPlacement{64, 128, 256, 2, 2}.matches(64, 128, 256), "same node twice");
    check(row_halves_has_slot(slots + 32, slots, 32, 4), "slot rejected");
    check(!row_halves_has_slot(slots + 1, slots, 32, 4) &&
          !row_halves_has_slot(slots + 128, slots, 32, 4) &&
          !row_halves_has_slot(slots, slots + 32, 32, 3) &&
          !row_halves_has_slot(nullptr, slots, 32, 4), "foreign or non-slot address accepted");

    const Pool one_high{{7}};
    std::vector<int> order;
    glmfast::run_prefill_row_blocks(one_high, 3, 128, 32, 2, [] { return worker_node; },
                                    [&](int job) { order.push_back(job); });
    check(order == std::vector<int>{6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5}, "no local-first or steal");
    order.clear();
    glmfast::run_prefill_row_blocks(one_high, 3, 129, 32, 2, [] {
        check(false, "node query on unaligned fallback");
        return 7;
    }, [&](int job) { order.push_back(job); });
    std::vector<int> ordinary(12);
    std::iota(ordinary.begin(), ordinary.end(), 0);
    check(order == ordinary, "tail fallback changed old job order");
    order.clear();
    glmfast::run_prefill_row_blocks(one_high, 3, 128, 32, -1, [] {
        check(false, "node query on disabled path");
        return 7;
    }, [&](int job) { order.push_back(job); });
    check(order == ordinary, "disabled path changed old job order");

    const Pool mixed{{2, 7, 7, 7, -1, 2}};
    check_dispatch(mixed, 2048, 32, 2, 0, 5);
    check_dispatch(mixed, 4096, 64, 2, 0, 3);
    check_dispatch(mixed, 4096, 64, 2, 3, 5);
    check_dispatch(mixed, 4096, 64, 2, 3, 3);
    check_dispatch(mixed, 129, 32, 2, 0, 1);
    check_dispatch(mixed, 192, 64, 2, 0, 3);
    check_dispatch(Pool{{7, 7, 7, 7}}, 128, 32, 2, 0, 5);

    std::thread a([&] { check_dispatch(mixed, 128, 32, 2, 0, 3); });
    std::thread b([&] { check_dispatch(mixed, 256, 64, 2, 2, 5); });
    a.join();
    b.join();
    std::cout << "row-local host scheduling: PASS\n";
#ifdef MAYA_ROW_LOCAL_NATIVE_TEST
    check_native_parity(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false);
    check_native_parity(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, false);
    if (cpu::kq_avx512_ok()) {
        check_native_parity(GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, true);
        std::cout << "row-local native GGML and packed parity: PASS\n";
    } else {
        std::cout << "row-local native GGML parity: PASS; packed AVX-512 unavailable\n";
    }
    check_native_parity(GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K, false);
    std::cout << "row-local IQ3_S gate/up, Q5_K down: PASS\n";
    check_native_parity(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, false);
    std::cout << "row-local IQ3_S gate/up, IQ4_XS down: PASS\n";
#endif
}
