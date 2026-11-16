// tests/glm_dsa/glm_mmq_pad_cpu_test.cpp - glm-dsa B15.2 on the CPU: the padding arithmetic of the MMQ expert banks.
//
// A host model of the prompt path's staging ring (NG groups of GE slots, gbuf) and its prestage buffer (pbuf): the
// memory starts as garbage (NaN bytes), experts of several size classes are copied in as glm_prefill.cu does (blob
// bytes at slot starts, the layer's stride apart), mmq_zero_gaps' runs are applied, and then for every group / buffer:
//   - the 512 bytes after the last written slot are zero and inside the group's own room (never the next group's),
//   - every written slot's gap (blob .. stride) is zero,
//   - nothing outside the group's room was written by the zeroing (the copy stream may be refilling the next group),
//   - the expert bytes themselves are untouched.
// Plus the allocation and capacity arithmetic: mmq_padded, mmq_slots_fit keeps 512 bytes free, the ring's group
// offsets at any layer stride fit the ring sized with the largest stride.
#include "strata/kernels/glmd_mmq_pad.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                        \
    do {                                                     \
        if (!(c)) {                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fail;                                        \
        }                                                    \
    } while (0)

constexpr uint8_t kNaN = 0xFF;   // 0xFFFF as FP16 / 0xFFFFFFFF as F32: NaN in every scale field
constexpr int NG = 3, GE = 4;    // glm_prefill.cu PrefillState::NG / GE

// glmfast::expert_stride's rounding (the lcm of 16 and the two block sizes), for a blob of gate+up+down bytes
size_t stride_of(size_t blob, size_t a) { return (blob + a - 1) / a * a; }

void arithmetic() {
    CHECK(glmd::mmq_padded(1000) == 1512, "padded");
    CHECK(glmd::mmq_usable(1512) == 1000 && glmd::mmq_usable(100) == 0, "usable");
    CHECK(glmd::mmq_slots_fit(10 * 4096, 4096) == 9, "10 slots of room keep 512 B free: 9 fit (got %lld)",
          (long long) glmd::mmq_slots_fit(10 * 4096, 4096));
    CHECK(glmd::mmq_slots_fit(10 * 4096 + 512, 4096) == 10, "exactly 10 + the tail");
    CHECK(glmd::mmq_slots_fit(4096, 0) == 0, "stride 0");
    // a prestage buffer of NP * gstride bytes: a layer at the largest stride gets NP - 1 slots, a smaller one more,
    // and n * stride + 512 <= bytes always
    const size_t gstride = 13'107'200, bytes = (size_t) 96 * gstride;
    for (size_t st : {gstride, gstride - 528, gstride / 2 + 16, (size_t) 4'194'304}) {
        const int64_t n = glmd::mmq_slots_fit(bytes, st);
        CHECK((size_t) n * st + glmd::kMmqTailPad <= bytes, "prestage at stride %zu: %lld slots overrun", st, (long long) n);
        CHECK((size_t) (n + 1) * st + glmd::kMmqTailPad > bytes, "prestage at stride %zu: one more would fit", st);
    }
    // the ring: groups at any stride <= gstride stay inside the ring sized at gstride
    const size_t ring = glmd::mmq_group_ring_bytes(NG, GE, gstride);
    CHECK(ring == (size_t) NG * GE * gstride + NG * glmd::kMmqTailPad, "ring bytes");
    for (size_t st : {gstride, gstride / 3 * 2, (size_t) 528})
        for (int b = 0; b < NG; ++b)
            CHECK(glmd::mmq_group_offset(b, GE, st) + glmd::mmq_group_room(GE, st) <= ring, "group %d at %zu", b, st);
    // runs: shapes
    auto r = glmd::mmq_gap_runs(2, 1024, 1000, glmd::mmq_group_room(4, 1024));
    CHECK(r.size() == 2 && r[0].off == 1000 && r[0].len == 24 && r[1].off == 2024 && r[1].len == 24 + 512,
          "two slots: gap, then the last gap merged with the tail");
    r = glmd::mmq_gap_runs(4, 1024, 1024, glmd::mmq_group_room(4, 1024));
    CHECK(r.size() == 1 && r[0].off == 4096 && r[0].len == 512, "a full group of gapless slots: its own tail only");
    r = glmd::mmq_gap_runs(3, 1024, 1024, 3 * 1024 + 100);
    CHECK(r.size() == 1 && r[0].len == 100, "the tail is clipped to the room");
    r = glmd::mmq_gap_runs(0, 1024, 1000, glmd::mmq_group_room(4, 1024));
    CHECK(r.size() == 1 && r[0].off == 0 && r[0].len == 512, "an empty group: its first 512 bytes");
    CHECK(glmd::mmq_gap_runs(2, 1000, 1024, 1u << 20).empty(), "blob > stride is refused");
}

struct Layer {
    size_t blob, stride;
};

void apply(std::vector<uint8_t>& mem, size_t base, const std::vector<glmd::MmqRun>& runs) {
    for (const auto& r : runs) std::memset(mem.data() + base + r.off, 0, r.len);
}

void ring_sim() {
    // three size classes (an IQ2_XXS / IQ2_S / IQ3_XXS-like mix: blocks of 66 / 82 / 98 bytes), small to keep it fast
    const std::vector<Layer> layers = {{66 * 300, stride_of(66 * 300, 528)},
                                       {82 * 300, stride_of(82 * 300, 656)},
                                       {98 * 300, stride_of(98 * 300, 784)}};
    size_t gstride = 0;
    for (const auto& L : layers) gstride = std::max(gstride, L.stride);
    const size_t ring = glmd::mmq_group_ring_bytes(NG, GE, gstride);
    std::vector<uint8_t> mem(ring + 4096, kNaN);   // + whatever follows the ring in the carve (pbuf): garbage
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> nexp(0, 11), pick(0, 2);
    int checks = 0;
    for (int step = 0; step < 60; ++step) {
        const Layer& L = layers[(size_t) pick(rng)];
        const int n_staged = 1 + nexp(rng);
        for (int i = 0, gi = 0; i < n_staged; i += GE, ++gi) {
            const int b = gi % NG, n = std::min(GE, n_staged - i);
            const size_t base = glmd::mmq_group_offset(b, GE, L.stride), room = glmd::mmq_group_room(GE, L.stride);
            std::vector<uint8_t> before = mem;
            for (int j = 0; j < n; ++j)   // the copies: blob bytes of a distinct pattern at each slot start
                std::memset(mem.data() + base + (size_t) j * L.stride, 0x11 + j, L.blob);
            apply(mem, base, glmd::mmq_gap_runs(n, L.stride, L.blob, room));
            // checks
            for (int j = 0; j < n; ++j) {
                const uint8_t* s = mem.data() + base + (size_t) j * L.stride;
                bool blob_ok = true, gap_ok = true;
                for (size_t k = 0; k < L.blob; ++k) blob_ok = blob_ok && s[k] == 0x11 + j;
                for (size_t k = L.blob; k < L.stride; ++k) gap_ok = gap_ok && s[k] == 0;
                CHECK(blob_ok, "step %d group %d slot %d: expert bytes changed", step, b, j);
                CHECK(gap_ok, "step %d group %d slot %d: gap not zero", step, b, j);
            }
            const size_t tail = base + (size_t) n * L.stride;
            bool tail_ok = true;
            for (size_t k = 0; k < glmd::kMmqTailPad; ++k) tail_ok = tail_ok && mem[tail + k] == 0;
            CHECK(tail_ok, "step %d group %d: the 512 bytes after slot %d are not zero", step, b, n - 1);
            CHECK(tail + glmd::kMmqTailPad <= base + room, "step %d group %d: the tail leaves the group's room", step, b);
            bool outside_ok = true;
            for (size_t k = 0; k < mem.size(); ++k)
                if ((k < base || k >= base + room) && mem[k] != before[k]) outside_ok = false;
            CHECK(outside_ok, "step %d group %d: bytes outside the group's room changed", step, b);
            ++checks;
        }
    }
    // the prestage buffer: NP slots of gstride, n <= mmq_slots_fit at the layer's stride, the tail zeroed
    const int NP = 16;
    const size_t pbytes = (size_t) NP * gstride;
    for (const auto& L : layers) {
        std::vector<uint8_t> pb(pbytes + 4096, kNaN);
        const int n = (int) glmd::mmq_slots_fit(pbytes, L.stride);
        for (int j = 0; j < n; ++j) std::memset(pb.data() + (size_t) j * L.stride, 0x40, L.blob);
        apply(pb, 0, glmd::mmq_gap_runs(n, L.stride, L.blob, pbytes));
        bool ok = true;
        for (size_t k = 0; k < glmd::kMmqTailPad; ++k) ok = ok && pb[(size_t) n * L.stride + k] == 0;
        CHECK(ok && (size_t) n * L.stride + glmd::kMmqTailPad <= pbytes, "prestage at stride %zu: tail", L.stride);
        bool past = true;
        for (size_t k = pbytes; k < pb.size(); ++k) past = past && pb[k] == kNaN;
        CHECK(past, "prestage: nothing written past the buffer");
        ++checks;
    }
    std::printf("ring / prestage simulation: %d groups and buffers checked (ring %zu bytes, largest stride %zu)\n", checks,
                ring, gstride);
}

}  // namespace

int main() {
    glmd::mmq_pad_enable(true);   // (load_pack does this for a glm-dsa model)
    arithmetic();
    ring_sim();
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
