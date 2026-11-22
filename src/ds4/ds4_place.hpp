// src/ds4/ds4_place.hpp - the DeepSeek engine's trunk expert placement as pure host arithmetic: how many VRAM slots
// each layer's experts get (the per-byte queue with a profile, the even split without), and how a layer's slots split
// into statics, recent slots and the expert cache's spare - shared by ds4_model.cu (place_experts) and
// tests/ds4_place_test.cpp.
//
// fit 0 is the engine's own rule at 3589d6f (place_experts, ds4_model.cu: the queue, the even split, the split); the
// engine keeps its inline code for it, and these functions at fit 0 are what the test checks against a real load and
// what DS4_RECENT_FIT's load line counts against.  DS4_RECENT_FIT (MAYA_DS4_RECENT_FIT, off by default):
//   1: the split only - a layer whose slots cannot hold the whole recent set beside a static and the spare keeps as
//      many recent slots as leave one static and the spare (today: none, so a layer under DS4_RECENT + 2 slots has no
//      recency cache at all - the 2.5-bit layers of the 2.25bpw EXL3 pack, 18-31 slots, a quarter of the picks);
//   2: also the queue - each layer first takes its recent set, a static and the spare (the floor; over the budget:
//      the queue as at 1), then each further slot is priced at the share of the static it adds (the split keeps
//      recent_n + 1 of a layer's slots for the cache, so its k-th slot is static k - 1 - recent_n), not the k-th
//      expert's share.
// Only where the slots go changes: the CPU lane and the GPU compute the same experts (rounding-level, as #644).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4::place {

// DS4_RECENT_FIT's value: unset, empty, 0, negative or not a number -> 0 (off); 1; 2 and above -> 2
inline int recent_fit_of(const char* v) {
    if (v == nullptr || *v == 0) return 0;
    return std::clamp(std::atoi(v), 0, 2);
}

// a layer's routing shares, largest first: its profile counts smoothed a tenth toward even (a layer the profile barely
// saw is not left without) - the queue's prices, a share a byte
inline std::vector<double> shares(const uint32_t* p, int n_exp) {
    double sum = 0;
    for (int e = 0; e < n_exp; ++e) sum += p[e];
    std::vector<double> s;
    s.reserve((size_t) n_exp);
    for (int e = 0; e < n_exp; ++e) s.push_back(0.9 * (sum > 0 ? p[e] / sum : 1.0 / n_exp) + 0.1 / n_exp);
    std::sort(s.begin(), s.end(), std::greater<double>());
    return s;
}

// a layer the queue gives slots to (one of a group's trunk layers with one expert layout)
struct QLayer {
    int il = 0;                  // the layer's index (a tie goes to the larger, as the engine's queue)
    int n_exp = 0;
    double unit = 0;             // bytes a slot (a member's mean part, a group's)
    std::vector<double> share;   // shares(), largest first
};

// the slots the recent set's floor takes in a layer (fit 2): the set, a static and the spare, at most every expert
inline int fit_floor(int n_exp, int recent_n) { return std::min(n_exp, std::max(0, recent_n) + 2); }

// the per-byte queue: one slot at a time to the layer whose next slot's share a byte is the largest, while the budget
// holds it (a layer whose next one does not fit drops out; smaller ones may still).  slots[i] for q[i].  fit < 2:
// the engine's queue.  fit 2: the floor first, then the split's prices; false (slots untouched) when the floor's
// bytes are over the budget - the caller then places as at fit 1
inline bool queue(const std::vector<QLayer>& q, double budget, int recent_n, int fit, std::vector<int>& slots) {
    std::vector<int> s(q.size(), 0);
    double used = 0;
    if (fit >= 2) {
        for (size_t i = 0; i < q.size(); ++i) {
            s[i] = fit_floor(q[i].n_exp, recent_n);
            used += q[i].unit * s[i];
        }
        if (used > budget) return false;
    }
    int il_max = -1;
    for (const QLayer& x : q) il_max = std::max(il_max, x.il);
    std::vector<int> at((size_t) (il_max + 1), -1);   // layer -> its place in q
    for (size_t i = 0; i < q.size(); ++i) at[(size_t) q[i].il] = (int) i;
    // the price of layer i's slot k + 1 (it holds k): the k-th expert's share, or (fit 2) the share of the static it
    // adds, k - 1 - recent_n (a layer at its floor or above holds at least one)
    auto price = [&](size_t i, int k) {
        const int j = fit >= 2 ? std::clamp(k - 1 - std::max(0, recent_n), 0, q[i].n_exp - 1) : k;
        return q[i].share[(size_t) j] / q[i].unit;
    };
    using Top = std::pair<double, int>;   // (the next slot's share a byte, layer)
    std::priority_queue<Top> pq;
    for (size_t i = 0; i < q.size(); ++i)
        if (fit < 2 || s[i] < q[i].n_exp) pq.push({price(i, s[i]), q[i].il});
    while (!pq.empty()) {
        const size_t i = (size_t) at[(size_t) pq.top().second];
        pq.pop();
        if (used + q[i].unit > budget) continue;   // (this layer's experts no longer fit; smaller ones may)
        used += q[i].unit;
        if (++s[i] < q[i].n_exp) pq.push({price(i, s[i]), q[i].il});
    }
    slots = std::move(s);
    return true;
}

// without a profile (or DS4_PLACE_UNIFORM): each layer the same share of its experts, then what the floors left, whole
// experts layer by layer
inline void even(const std::vector<QLayer>& q, double budget, std::vector<int>& slots) {
    double want = 0;
    for (const QLayer& x : q) want += x.unit * x.n_exp;
    const double frac = want > 0 ? std::min(1.0, budget / want) : 0.0;
    std::vector<int> s(q.size(), 0);
    for (size_t i = 0; i < q.size(); ++i) {
        s[i] = std::min(q[i].n_exp, (int) std::floor(frac * q[i].n_exp));
        while (s[i] > 0 && q[i].unit * s[i] > budget) --s[i];   // (the floors can still round the total up)
    }
    double used = 0;
    for (size_t i = 0; i < q.size(); ++i) used += q[i].unit * s[i];
    for (size_t i = 0; i < q.size(); ++i)
        while (s[i] < q[i].n_exp && used + q[i].unit <= budget) {
            ++s[i];
            used += q[i].unit;
        }
    slots = std::move(s);
}

// a layer's recent slots out of its n_slots (2 or more, fewer than its experts, the expert cache on - a layer with a
// spare): the whole set where a static is left beside it and the spare, else none (fit 0, the engine's rule) or (fit
// 1 or 2, 3 slots or more) as many as leave one static and the spare.  eligible: a trunk layer of a load whose recent
// slots can fill (the engine's !mtp && (!file_experts || (gguf && DS4_LAND)))
inline int recent_slots(int n_slots, int recent_n, bool eligible, int fit) {
    if (recent_n <= 0 || !eligible) return 0;
    if (n_slots - 1 - recent_n >= 1) return recent_n;
    return fit >= 1 && n_slots >= 3 ? std::min(recent_n, n_slots - 2) : 0;
}

// a layer's slots: statics [0, statics), recent slots [statics, statics + recent), the spare last
struct Split {
    int slots = 0, statics = 0, recent = 0, spare = 0;
};

// the split of a layer the queue gave n_vram slots: every expert in VRAM, the cache off or one slot -> every slot a
// static; else the recent slots and the spare out of them
inline Split split(int n_vram, int n_exp, bool cache, int recent_n, bool eligible, int fit) {
    Split s;
    s.slots = std::min(n_vram, n_exp);
    if (cache && s.slots >= 2 && s.slots < n_exp) {
        s.recent = recent_slots(s.slots, recent_n, eligible, fit);
        s.spare = 1;
        s.statics = s.slots - 1 - s.recent;
    } else {
        s.statics = s.slots;
    }
    return s;
}

// ---- DS4_SPLIT: the group that runs each trunk layer
// "a,b,..": the layers each group takes, in the groups' order - a count past the last group is left out, the layers
// past the last count go to the last group (both as ever).  "n@g": the next n layers on group g, whichever its turn.
// False: an n@g that names no group.
inline bool layer_groups(const char* text, int n_groups, int n_layers, std::vector<int>& grp, std::string& err) {
    grp.assign((size_t) std::max(0, n_layers), std::max(0, n_groups - 1));
    int il = 0, gi = 0;
    for (const char* q = text; q != nullptr && *q;) {
        const int n = std::atoi(q);
        int at = -1;
        while (*q && *q != ',' && *q != '@') ++q;
        if (*q == '@') {
            char* e = nullptr;
            const long v = std::strtol(q + 1, &e, 10);
            if (e == q + 1 || (*e != 0 && *e != ',') || v < 0 || v >= n_groups) {
                err = "DS4_SPLIT: n@g names a group, 0.." + std::to_string(n_groups - 1);
                return false;
            }
            at = (int) v;
            q = e;
        }
        if (*q == ',') ++q;
        if (at < 0 && gi >= n_groups) continue;
        const int to = at >= 0 ? at : gi++;
        for (int j = 0; j < n && il < n_layers; ++j) grp[(size_t) il++] = to;
    }
    return true;
}

// The layers that go back to a group that ran earlier ones (an n@g): `at` their first, -1 where every group's layers
// are one run.  A device's thread puts its layers on its stream in order and hands a window (or a prompt's chunk) on
// once, so what runs is one group in two runs of layers, the second the trunk's last layers.  False: anything else.
inline bool layers_back(const std::vector<int>& grp, int n_groups, int& at, std::string& err) {
    std::vector<int> runs((size_t) std::max(0, n_groups), 0);
    at = -1;
    for (size_t il = 0; il < grp.size(); ++il) {
        const int gi = grp[il];
        if (il > 0 && grp[il - 1] == gi) continue;
        if (++runs[(size_t) gi] == 1) continue;
        if (at >= 0) {
            err = "DS4_SPLIT: one group takes layers a second time, and only the trunk's last ones (a,b,..,n@g)";
            return false;
        }
        at = (int) il;
    }
    for (size_t il = at < 0 ? grp.size() : (size_t) at; il < grp.size(); ++il)
        if (grp[il] != grp[(size_t) at]) {
            err = "DS4_SPLIT: the layers that go back to a group (n@g) are the trunk's last ones";
            return false;
        }
    return true;
}

}  // namespace strata::ds4::place
