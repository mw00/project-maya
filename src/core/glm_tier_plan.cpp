// src/core/glm_tier_plan.cpp - the glm-dsa port B8.2-B8.4 / B8.7 host-side rules (see glm_tier_plan.hpp).  Plain C++.
#include "strata/core/glm_env.hpp"
#include "glm_tier_plan.hpp"

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace strata::core::glmtier {

// ---------------------------------------------------------------- B8.2

size_t seg_cap_bytes(bool glm_dsa, const char* env) {
    if (env != nullptr && env[0] != '\0') {
        const double gb = std::atof(env);
        if (!(gb > 0.0)) return 0;
        return (size_t) (gb * 1073741824.0);
    }
    return glm_dsa ? kSegCapDefault : 0;
}

int64_t SegPlan::total() const {
    int64_t t = 0;
    for (int s : slots) t += s;
    return t;
}

SegPlan seg_plan(int64_t n, size_t slot_bytes, size_t cap) {
    SegPlan p;
    p.slot_bytes = slot_bytes;
    if (n <= 0 || slot_bytes == 0) return p;
    const int64_t per = cap == 0 ? n : std::max<int64_t>(1, (int64_t) (cap / slot_bytes));
    p.seg_slots = (int) std::min<int64_t>(per, n);
    for (int64_t left = n; left > 0; left -= p.seg_slots) p.slots.push_back((int) std::min<int64_t>(left, p.seg_slots));
    return p;
}

int64_t shrink_step(int64_t n, int tries) { return tries < 4 ? n * 31 / 32 : n * 7 / 8; }

int64_t seg_pin(const SegPlan& plan, int64_t least, const std::function<void(std::vector<char>&)>& first_round,
                const std::function<bool(int, int)>& pin, const std::function<void(int, int)>& release,
                std::vector<int>& pinned) {
    const int ns = (int) plan.slots.size();
    pinned.clear();
    if (ns == 0) return least <= 0 ? 0 : -1;
    std::vector<char> ok((size_t) ns, 0);
    first_round(ok);
    int first_bad = ns;
    for (int k = 0; k < ns; ++k)
        if (!ok[(size_t) k]) {
            first_bad = k;
            break;
        }
    // every segment after the first failure goes (the class keeps a prefix of whole segments plus one shrunk last)
    for (int k = first_bad + 1; k < ns; ++k)
        if (ok[(size_t) k]) release(k, plan.slots[(size_t) k]);
    int64_t prefix = 0;
    for (int k = 0; k < first_bad; ++k) {
        pinned.push_back(plan.slots[(size_t) k]);
        prefix += plan.slots[(size_t) k];
    }
    if (first_bad == ns) {
        if (prefix >= least) return prefix;
        for (int k = 0; k < ns; ++k) release(k, pinned[(size_t) k]);
        pinned.clear();
        return -1;
    }
    // the last segment shrinks with the class total, Maya's steps (31/32 four times, then 7/8)
    int64_t total = prefix + plan.slots[(size_t) first_bad];
    for (int tries = 0;; ++tries) {
        const int64_t next = shrink_step(total, tries);
        if (next >= total) break;   // (no progress: a tiny class)
        total = next;
        const int64_t s = total - prefix;
        if (s <= 0 || total < least) break;
        if (pin(first_bad, (int) s)) {
            pinned.push_back((int) s);
            return total;
        }
    }
    if (prefix >= least && prefix > 0) return prefix;
    for (int k = 0; k < (int) pinned.size(); ++k) release(k, pinned[(size_t) k]);
    pinned.clear();
    return -1;
}

int64_t node_mem_avail(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    int64_t free_kb = -1, inact_file = 0, srecl = 0;
    while (std::getline(in, line)) {
        // "Node 0 MemFree:        12345 kB"
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const size_t sp = line.rfind(' ', colon);
        const std::string key = line.substr(sp == std::string::npos ? 0 : sp + 1, colon - (sp == std::string::npos ? 0 : sp + 1));
        const long long v = std::atoll(line.c_str() + colon + 1);
        if (key == "MemFree") free_kb = v;
        else if (key == "Inactive(file)") inact_file = v;
        else if (key == "SReclaimable") srecl = v;
    }
    if (free_kb < 0) return -1;
    return (free_kb + inact_file + srecl) * 1024;
}

int64_t node_reserve_bytes(const char* env) {
    double gb = 4.0;
    if (env != nullptr && env[0] != '\0') gb = std::max(0.0, std::atof(env));
    return (int64_t) (gb * 1073741824.0);
}

namespace {
bool node_fits(int64_t n, size_t stride, size_t cap, bool split, const std::vector<int>& nodes,
               const std::vector<int64_t>& budget) {
    std::vector<int64_t> need(nodes.size(), 0);
    const auto index = [&](int node) {
        for (size_t j = 0; j < nodes.size(); ++j)
            if (nodes[j] == node) return (int) j;
        return -1;
    };
    if (split) {
        for (int h = 0; h < 2; ++h) {
            const int j = index(seg_node(0, h, true, nodes));
            if (j >= 0) need[(size_t) j] += n * (int64_t) stride;
        }
    } else {
        const SegPlan p = seg_plan(n, stride, cap);
        for (size_t k = 0; k < p.slots.size(); ++k) {
            const int j = index(seg_node((int) k, 0, false, nodes));
            if (j >= 0) need[(size_t) j] += (int64_t) p.slots[k] * (int64_t) stride;
        }
    }
    for (size_t j = 0; j < nodes.size(); ++j)
        if (j < budget.size() && budget[j] >= 0 && need[j] > budget[j]) return false;
    return true;
}
}  // namespace

int64_t node_fit_slots(int64_t n, size_t stride, size_t cap, bool split, const std::vector<int>& nodes,
                       const std::vector<int64_t>& budget) {
    if (n <= 0 || stride == 0 || nodes.empty()) return std::max<int64_t>(0, n);
    if (node_fits(n, stride, cap, split, nodes, budget)) return n;
    // the need per node grows with n (whole segments fill in order), so the largest fit is a binary search
    int64_t lo = 0, hi = n;   // fits(lo) (0 slots), !fits(hi)
    while (hi - lo > 1) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (node_fits(mid, stride, cap, split, nodes, budget)) lo = mid;
        else hi = mid;
    }
    return lo;
}

int node_fit_action(int64_t n, int64_t fit, int64_t least) {
    if (fit >= n) return 0;
    if (fit >= std::max<int64_t>(1, least) && fit * 4 >= n * 3) return 1;
    return 2;
}

int seg_node(int k, int half, bool split, const std::vector<int>& nodes) {
    if (nodes.empty()) return -1;
    if (split) return nodes[(size_t) std::min<size_t>((size_t) (half != 0), nodes.size() - 1)];
    return nodes[(size_t) k % nodes.size()];
}

// ---------------------------------------------------------------- B8.7

int thp_mode(const char* env) {
    return strata::glmenv::switch_value("STRATA_GLM_THP", env, strata::glmenv::kAuto, true);
}

int64_t buddy_huge_free(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    int64_t total = 0;
    bool any = false;
    while (std::getline(in, line)) {
        const char* z = std::strstr(line.c_str(), "zone");
        if (z == nullptr) continue;
        z += 4;
        while (*z == ' ') ++z;
        while (*z != ' ' && *z != '\0') ++z;   // the zone's name
        unsigned long long c[16] = {};
        int k = 0;
        char* end = nullptr;
        for (const char* q = z; k < 16; ++k, q = end) {
            c[k] = std::strtoull(q, &end, 10);
            if (end == q) break;
        }
        for (int o = 9; o < k; ++o) total += (int64_t) (c[o] << (o + 12));
        any = true;
    }
    return any ? total : -1;
}

std::vector<int64_t> buddy_huge_free_nodes(const std::string& text, const std::vector<int>& nodes) {
    std::vector<int64_t> out(nodes.size(), -1);
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        int node = -1;
        if (std::sscanf(line.c_str(), " Node %d,", &node) != 1) continue;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (nodes[j] != node) continue;
            const int64_t b = buddy_huge_free(line);
            if (b >= 0) out[j] = std::max<int64_t>(0, out[j]) + b;
        }
    }
    return out;
}

std::vector<char> thp_plan_nodes(const std::vector<size_t>& unit_bytes, const std::vector<int>& unit_node,
                                 const std::vector<int64_t>& node_free, int64_t all_free, int mode) {
    std::vector<char> out(unit_bytes.size(), 0);
    // group the units by their budget (-1: the machine-wide one), keep each group's order, plan each group alone
    std::vector<int> keys;
    for (size_t k = 0; k < unit_bytes.size(); ++k) {
        const int key = k < unit_node.size() && unit_node[k] >= 0 && unit_node[k] < (int) node_free.size() ? unit_node[k] : -1;
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) keys.push_back(key);
    }
    for (int key : keys) {
        std::vector<size_t> idx, bytes;
        for (size_t k = 0; k < unit_bytes.size(); ++k) {
            const int kk = k < unit_node.size() && unit_node[k] >= 0 && unit_node[k] < (int) node_free.size() ? unit_node[k] : -1;
            if (kk != key) continue;
            idx.push_back(k);
            bytes.push_back(unit_bytes[k]);
        }
        const std::vector<char> p = thp_plan(bytes, key < 0 ? all_free : node_free[(size_t) key], mode);
        for (size_t i = 0; i < idx.size(); ++i) out[idx[i]] = p[i];
    }
    return out;
}

std::vector<char> thp_plan(const std::vector<size_t>& seg_bytes, int64_t huge_free, int mode) {
    std::vector<char> out(seg_bytes.size(), 0);
    int64_t left = huge_free;
    for (size_t k = 0; k < seg_bytes.size(); ++k) {
        if (mode == 0) continue;
        if (mode == 1 || huge_free < 0) {   // forced, or unknown (Maya: madvise when buddyinfo is unreadable)
            out[k] = 1;
            continue;
        }
        if (left >= (int64_t) seg_bytes[k] + ((int64_t) 1 << 30)) {
            out[k] = 1;
            left -= (int64_t) seg_bytes[k];
        }
    }
    return out;
}

SmapsSum smaps_sum(const std::string& smaps, const std::vector<std::pair<uintptr_t, size_t>>& ranges) {
    SmapsSum s;
    std::istringstream in(smaps);
    std::string line;
    bool inside = false;
    while (std::getline(in, line)) {
        unsigned long long a = 0, b = 0;
        char dash = 0;
        // a mapping's header: "start-end perms offset dev inode [path]" (hex addresses)
        if (!line.empty() && std::isxdigit((unsigned char) line[0]) && std::sscanf(line.c_str(), "%llx%c%llx", &a, &dash, &b) == 3 &&
            dash == '-') {
            inside = false;
            for (const auto& r : ranges)
                if (a >= r.first && a < r.first + r.second) inside = true;
            if (inside) s.size += (int64_t) (b - a);
            continue;
        }
        if (!inside) continue;
        long long kb = 0;
        if (std::sscanf(line.c_str(), "AnonHugePages: %lld kB", &kb) == 1) s.anon_huge += kb * 1024;
        else if (std::sscanf(line.c_str(), "Rss: %lld kB", &kb) == 1) s.rss += kb * 1024;
    }
    return s;
}

std::vector<std::pair<size_t, size_t>> fault_plan(size_t bytes, bool huge, int threads, int t) {
    std::vector<std::pair<size_t, size_t>> out;
    constexpr size_t k2M = (size_t) 2 << 20;
    if (!huge || threads <= 1) {
        if (t == 0 && bytes > 0) out.push_back({0, bytes});
        return out;
    }
    for (size_t o = (size_t) t * k2M; o < bytes; o += (size_t) threads * k2M) out.push_back({o, std::min(bytes, o + k2M)});
    return out;
}

// ---------------------------------------------------------------- B8.3

int lane_threads_scaled(int n, int ref_size, int size) {
    if (size <= 0) return 1;
    if (n <= 0 || ref_size <= 0) return size;
    const int64_t v = ((int64_t) n * size * 2 + ref_size) / ((int64_t) ref_size * 2);   // n * size / ref_size, rounded
    return (int) std::max<int64_t>(1, std::min<int64_t>(size, v));
}

const char* lane_pool_name(LanePool m) {
    switch (m) {
        case LanePool::One: return "one pool";
        case LanePool::PerPart: return "a pool per part";
        case LanePool::Shared: return "one shared pool";
        case LanePool::Socket: return "a pinned pool per socket";
        case LanePool::NodeSplit: return "node-split: a pinned pool per socket, each on its half of every expert";
    }
    return "?";
}

// ---------------------------------------------------------------- B8.4

bool numa_split_requested(bool glm_dsa, const char* env) {
    return glm_dsa && strata::glmenv::switch_value("STRATA_GLM_NUMA_SPLIT", env, strata::glmenv::kOff) == strata::glmenv::kOn;
}

bool split_layout(size_t gu_row, size_t d_row, int64_t n_ff, int64_t n_embd, int64_t d_blck, size_t d_blck_bytes,
                  SplitLayout& L, std::string& why) {
    L = SplitLayout{};
    if (n_ff <= 0 || n_embd <= 0 || gu_row == 0 || d_row == 0 || d_blck <= 0 || d_blck_bytes == 0) {
        why = "empty geometry";
        return false;
    }
    if (n_ff % 2 != 0) {
        why = "n_ff is odd";
        return false;
    }
    const int64_t H = n_ff / 2;
    if (H % d_blck != 0) {
        why = "n_ff / 2 is not a whole number of down blocks";
        return false;
    }
    if ((size_t) (n_ff / d_blck) * d_blck_bytes != d_row) {
        why = "the down row is not whole blocks";
        return false;
    }
    if (gu_row % 4 != 0 || (d_row / 2) % 4 != 0) {
        why = "row bytes not a multiple of 4";
        return false;
    }
    L.gu_row = gu_row;
    L.d_row = d_row;
    L.n_ff = n_ff;
    L.n_embd = n_embd;
    L.H = H;
    L.gu_bytes = (size_t) n_ff * gu_row;
    L.down_off = 2 * L.gu_bytes;
    L.blob = L.down_off + (size_t) n_embd * d_row;
    L.h_gu = (size_t) H * gu_row;
    L.h_down_off = 2 * L.h_gu;
    L.hd_row = d_row / 2;
    L.half_bytes = L.h_down_off + (size_t) n_embd * L.hd_row;
    return true;
}

std::vector<CopyOp> split_ops(const SplitLayout& L) {
    std::vector<CopyOp> ops;
    for (int h = 0; h < 2; ++h) {
        ops.push_back(CopyOp{h, (size_t) h * L.h_gu, 0, L.h_gu, 1, L.h_gu, L.h_gu});                          // gate
        ops.push_back(CopyOp{h, L.gu_bytes + (size_t) h * L.h_gu, L.h_gu, L.h_gu, 1, L.h_gu, L.h_gu});        // up
        ops.push_back(CopyOp{h, L.down_off + (size_t) h * L.hd_row, L.h_down_off, L.hd_row, (size_t) L.n_embd,
                             L.d_row, L.hd_row});                                                               // down
    }
    return ops;
}

void scatter(const SplitLayout& L, const uint8_t* u, uint8_t* h0, uint8_t* h1) {
    for (const CopyOp& op : split_ops(L)) {
        uint8_t* h = op.half ? h1 : h0;
        for (size_t r = 0; r < op.rows; ++r)
            std::memcpy(h + op.h_off + r * op.h_pitch, u + op.u_off + r * op.u_pitch, op.width);
    }
}

void gather(const SplitLayout& L, const uint8_t* h0, const uint8_t* h1, uint8_t* u) {
    for (const CopyOp& op : split_ops(L)) {
        const uint8_t* h = op.half ? h1 : h0;
        for (size_t r = 0; r < op.rows; ++r)
            std::memcpy(u + op.u_off + r * op.u_pitch, h + op.h_off + r * op.h_pitch, op.width);
    }
}

void scatter_range(const SplitLayout& L, size_t u0, const uint8_t* src, size_t len, uint8_t* h0, uint8_t* h1) {
    const size_t u1 = std::min(L.blob, u0 + len);
    size_t u = u0;
    while (u < u1) {
        // the run from u to the end of its row piece (a gate/up half section, or a down row half) is contiguous on both
        // sides
        size_t run_end;
        if (u < L.down_off) {
            const size_t sec = u < L.gu_bytes ? 0 : 1;
            const size_t base = sec * L.gu_bytes, mid = base + (size_t) L.H * L.gu_row;
            run_end = u < mid ? mid : base + L.gu_bytes;
        } else {
            const size_t v = u - L.down_off, r = v / L.d_row, c = v - r * L.d_row;
            run_end = L.down_off + r * L.d_row + (c < L.hd_row ? L.hd_row : L.d_row);
        }
        run_end = std::min(run_end, u1);
        const SplitLoc at = split_loc(L, u);
        std::memcpy((at.half ? h1 : h0) + at.off, src + (u - u0), run_end - u);
        u = run_end;
    }
}

size_t half_stride(const std::vector<SplitLayout>& layers) {
    size_t m = 0;
    for (const auto& L : layers) m = std::max(m, L.half_bytes);
    return (m + 4095) & ~(size_t) 4095;
}

void add_partial(const float* y0, const float* y1, float* y, int64_t n) {
    for (int64_t r = 0; r < n; ++r) y[r] = y0[r] + y1[r];
}

}  // namespace strata::core::glmtier
