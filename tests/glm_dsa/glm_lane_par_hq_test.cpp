// CPU-only unit tests for include/maya/core/glm_lane_par_hq.hpp (no CUDA or ggml needed).
#include "maya/core/glm_lane_par_hq.hpp"

#undef NDEBUG
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using maya::core::glmlane::ParHqTracker;
using maya::core::glmlane::scale_single_rows;

static uint64_t hash_block_256(const float* src) {
    uint64_t h = 1469598103934665603ull;
    const auto* b = reinterpret_cast<const unsigned char*>(src);
    for (size_t i = 0; i < 256 * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static void test_eligibility() {
    assert(ParHqTracker::eligible(1, 2048, 8 * 292));
    assert(ParHqTracker::eligible(8, 4096, 16 * 292));
    assert(ParHqTracker::eligible(3, 256, 292));
    assert(!ParHqTracker::eligible(0, 2048, 8 * 292));
    assert(!ParHqTracker::eligible(9, 2048, 8 * 292));
    assert(!ParHqTracker::eligible(1, 0, 292));
    assert(!ParHqTracker::eligible(1, 300, 292));
    assert(!ParHqTracker::eligible(1, 8192, 32 * 292));   // n_hb = 32 > 16
    assert(!ParHqTracker::eligible(1, 2048, 8 * 292 + 1)); // not divisible by n_hb
    assert(!ParHqTracker::eligible(1, 2048, 0));
}

static void test_default_and_row_halves_multithreaded(int ne, int n_ff, int T, bool row_halves) {
    const int n_hb = n_ff / ParHqTracker::kBlockSize;
    std::vector<float> ff((size_t) ne * (size_t) n_ff, -999.0f);
    std::vector<uint64_t> ref((size_t) ne * (size_t) n_hb, 0);
    std::vector<uint64_t> got((size_t) ne * (size_t) n_hb, 0);
    std::vector<std::atomic<int>> calls((size_t) ne * (size_t) n_hb);
    for (auto& c : calls) c.store(0, std::memory_order_relaxed);

    // Reference values and block hashes
    std::vector<float> ref_ff((size_t) ne * (size_t) n_ff);
    for (int i = 0; i < ne; ++i)
        for (int r = 0; r < n_ff; ++r)
            ref_ff[(size_t) i * (size_t) n_ff + (size_t) r] = std::sin(0.13f * r + 0.7f * i);
    for (int i = 0; i < ne; ++i)
        for (int b = 0; b < n_hb; ++b)
            ref[(size_t) i * (size_t) n_hb + (size_t) b] =
                hash_block_256(ref_ff.data() + (size_t) i * (size_t) n_ff + (size_t) b * 256);

    ParHqTracker tracker;
    tracker.reset(ne, n_ff);

    const auto quant_cb = [&](int ei, int b) {
        calls[(size_t) ei * (size_t) n_hb + (size_t) b].fetch_add(1, std::memory_order_relaxed);
        got[(size_t) ei * (size_t) n_hb + (size_t) b] =
            hash_block_256(ff.data() + (size_t) ei * (size_t) n_ff + (size_t) b * 256);
    };

    const int n_threads = std::min(T, 8);
    if (!row_halves) {
        const int tot = ne * n_ff;
        std::atomic<int> next_job{0};
        std::vector<std::thread> th;
        th.reserve((size_t) n_threads);
        for (int w = 0; w < n_threads; ++w) {
            th.emplace_back([&] {
                for (;;) {
                    const int job = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (job >= T) break;
                    const int q0 = (int) ((int64_t) tot * job / T);
                    const int q1 = (int) ((int64_t) tot * (job + 1) / T);
                    for (int q = q0; q < q1;) {
                        const int i = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
                        for (int r = r0; r < r1; ++r)
                            ff[(size_t) i * (size_t) n_ff + (size_t) r] =
                                ref_ff[(size_t) i * (size_t) n_ff + (size_t) r];
                        tracker.mark_rows(i, r0, r1, quant_cb);
                        q += r1 - r0;
                    }
                }
            });
        }
        for (auto& t : th) t.join();
    } else {
        const int J = std::max(1, T / 2);
        const int half = n_ff / 2;
        const int tot = ne * half;
        std::atomic<int> qp[2];
        qp[0].store(0, std::memory_order_relaxed);
        qp[1].store(0, std::memory_order_relaxed);
        std::vector<std::thread> th;
        th.reserve((size_t) n_threads);
        for (int w = 0; w < n_threads; ++w) {
            th.emplace_back([&, w] {
                const int me = w & 1;
                for (int t = 0; t < 2; ++t) {
                    const int h = t == 0 ? me : 1 - me;
                    for (;;) {
                        const int j = qp[h].fetch_add(1, std::memory_order_acq_rel);
                        if (j >= J) break;
                        const int q0 = (int) ((int64_t) tot * j / J);
                        const int q1 = (int) ((int64_t) tot * (j + 1) / J);
                        for (int q = q0; q < q1;) {
                            const int i = q / half, r0 = q % half, r1 = std::min(half, r0 + (q1 - q));
                            const int gr0 = h * half + r0, gr1 = h * half + r1;
                            for (int r = gr0; r < gr1; ++r)
                                ff[(size_t) i * (size_t) n_ff + (size_t) r] =
                                    ref_ff[(size_t) i * (size_t) n_ff + (size_t) r];
                            tracker.mark_rows(i, gr0, gr1, quant_cb);
                            q += r1 - r0;
                        }
                    }
                }
            });
        }
        for (auto& t : th) t.join();
    }

    for (size_t idx = 0; idx < calls.size(); ++idx) {
        assert(calls[idx].load() == 1);
        assert(got[idx] == ref[idx]);
    }
}

static void test_scale_single_rows_bit_exact() {
    const int n_embd = 4096;
    std::vector<float> dst((size_t) n_embd), out_ref((size_t) n_embd), out_got((size_t) n_embd);
    for (int r = 0; r < n_embd; ++r) dst[(size_t) r] = (r % 7 == 0) ? -0.0f : std::cos(0.21f * r);
    const float w0 = 0.375f;
    for (int r = 0; r < n_embd; ++r) {
        float m = 0.0f;
        m += w0 * dst[(size_t) r];
        out_ref[(size_t) r] = m;
    }
    scale_single_rows(out_got.data(), dst.data(), w0, 0, 1024);
    scale_single_rows(out_got.data(), dst.data(), w0, 1024, 2500);
    scale_single_rows(out_got.data(), dst.data(), w0, 2500, n_embd);
    assert(std::memcmp(out_ref.data(), out_got.data(), (size_t) n_embd * sizeof(float)) == 0);
}

int main() {
    test_eligibility();
    for (int n_ff : {256, 1024, 2048, 4096}) {
        for (int ne : {1, 2, 3, 5, 8}) {
            for (int T : {1, 2, 3, 7, 16, 64}) {
                test_default_and_row_halves_multithreaded(ne, n_ff, T, false);
                if (n_ff % 512 == 0) test_default_and_row_halves_multithreaded(ne, n_ff, T, true);
            }
        }
    }
    test_scale_single_rows_bit_exact();
    std::printf("glm_lane_par_hq_test: PASSED\n");
    return 0;
}
