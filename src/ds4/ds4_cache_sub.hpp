#pragma once
#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_SUB_HD __host__ __device__
#else
#define DS4_SUB_HD
#endif

namespace strata::ds4::cache_sub {
// Approximate cache-aware selection. No model, allocations, device work or cache
// changes. Caller supplies authoritative candidate rank order and a residency
// bit (primary OR healthy helper). Selected values are candidate-rank indices.
//
// Budget units are normalized router coefficient mass, NOT error probability.
// Expert similarity is unknown. FP32 cost is src/mass + alt/mass, evaluated as
// two divisions then an addition; this also avoids numerator overflow. Native
// mass and spent budget use sequential FP32 sums. Compile without fast-math.
//
// With valid k, invalid input restores all native choices before returning
// false. Invalid k is rejected before touching the caller's output buffer.
DS4_SUB_HD inline bool choose(const int* ids, const float* scores,
                               const std::uint8_t* resident, int k, int n,
                               float ratio, bool gated, float budget,
                               int* selected, float* spent_out = nullptr) {
    if (spent_out) *spent_out = 0;
    if (k < 1 || k > 8) return false;
    for (int i = 0; i < k; ++i) selected[i] = i;
    if (n < k || n > 32 || !(ratio >= 0 && ratio <= 1)) return false;
    if (gated && !(budget >= 0 && budget <= FLT_MAX)) return false;
    float mass = 0;
    int strongest = 0;
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || !(scores[i] >= 0 && scores[i] <= FLT_MAX) || resident[i] > 1) return false;
        for (int j = 0; j < i; ++j) if (ids[i] == ids[j]) return false;
        if (i < k) {
            mass += scores[i];
            if (scores[i] > scores[strongest]) strongest = i;
        }
    }
    if (!(mass > 0 && mass <= FLT_MAX)) return false;
    // Preserve the existing native flag convention: ratio zero disables swaps.
    if (ratio == 0) return true;
    int slots[8], alternatives[8], count = 0;
    float costs[8];
    for (int i = 0; i < k; ++i) {
        if (i == 0 || i == strongest || resident[i] || scores[i] == 0) continue;
        for (int j = k; j < n; ++j) {
            if (!resident[j] || scores[j] < ratio * scores[i]) continue;
            bool used = false;
            for (int q = 0; q < count; ++q) used = used || alternatives[q] == j;
            if (used) continue;
            slots[count] = i; alternatives[count] = j;
            costs[count] = scores[i] / mass + scores[j] / mass;
            ++count;
            break;
        }
    }
    float spent = 0;
    if (!gated) {
        for (int q = 0; q < count; ++q) {
            selected[slots[q]] = alternatives[q];
            spent += costs[q];
        }
        if (spent_out) *spent_out = spent;
        return true;
    }
    // Stable small-k insertion sort. Candidate assignments are fixed before
    // admission; rejected alternatives are never recycled to another slot.
    for (int i = 1; i < count; ++i) {
        const int slot = slots[i], alt = alternatives[i];
        const float cost = costs[i];
        int j = i;
        while (j > 0 && (cost < costs[j-1] ||
               (cost == costs[j-1] && (slot < slots[j-1] ||
                (slot == slots[j-1] && ids[alt] < ids[alternatives[j-1]]))))) {
            slots[j] = slots[j-1]; alternatives[j] = alternatives[j-1]; costs[j] = costs[j-1]; --j;
        }
        slots[j] = slot; alternatives[j] = alt; costs[j] = cost;
    }
    for (int q = 0; q < count; ++q) {
        const float next = spent + costs[q];
        if (next <= budget) {
            selected[slots[q]] = alternatives[q];
            spent = next;
        }
    }
    if (spent_out) *spent_out = spent;
    return true;
}
}
#undef DS4_SUB_HD
