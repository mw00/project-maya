// tests/glm_dsa/glm_kv_groups_test.cpp - KV streaming on glm-dsa (GLM-5.3 744B), CPU only: the residency-map groups
// and the identity window as a protocol, on a host twin of the device kernels.
//
// 1. glm_kv_leads on GLM-5.3's indexer schedule (full layers 0, 1, 2, 6, 10, ... 74) for one part, the 4-part split
//    and a part that carries the NextN block: 22 maps for the 79 caches, every shared layer led by the full layer
//    before it on its own part, the NextN block alone; grouped = false and a short per_set fall back to more maps.
// 2. The protocol.  A twin of one part with a few indexer groups: per layer a host copy (the truth a position was last
//    written with) and VRAM slot rows, per group one residency map with glm_kv_stream.cu's resolve (hits stamped, each
//    miss claimed once, a CLOCK sweep in steps that never takes a slot of this call, the selection rewritten in slot
//    rows) and the group copy (each missed block for every layer of the group).  The engine's own rules drive it
//    (glm_kv_leads, glm_kv_window_serves / _chunk / _load) in the order the engine issues them:
//      decode token: per layer the prep (host row always, the slot row when its block is resident), then the read -
//                    the window's rows as they are, the full layer's resolve, or the shared layers' reuse of it;
//      a rejected draft: the same position written again with other values;
//      a prompt: chunks inside the window on the slots (+ host), others on the stage (host only, maps evicted);
//      a checkpoint restore: the position goes back; a reset; a slot save / load (kv_stream_prime) into the window
//      or, for a conversation longer than it, with every block evicted.
//    After EVERY read, each row the attention would read must hold what its position was last written with - for
//    the full layer and for every shared layer - and the map must stay a bijection between resident blocks and slots.
//    Random schedules over small and large windows, pages of 1 (glm-dsa) and 4.
#include "strata/core/glm_kv_groups.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using strata::core::glm_kv_leads;
using strata::core::glm_kv_window_chunk;
using strata::core::glm_kv_window_load;
using strata::core::glm_kv_window_serves;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok || getenv("VERBOSE")) std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool glm53_full(int il) { return il < 3 || (il >= 6 && (il - 2) % 4 == 0); }

void test_leads() {
    const int NL = 78;
    const auto full = [](int il) { return glm53_full(il); };
    const auto layers = [&](int l0, int l1, bool mtp) {
        std::vector<int> ls;
        if (mtp) ls.push_back(NL);
        for (int il = l0; il < l1; ++il) ls.push_back(il);
        return ls;
    };
    size_t n = 0, total = 0;
    // the 4-part split of the reference box: [0,26) [26,42) [42,58) [58,78) + NextN
    const int b[5] = {0, 26, 42, 58, 78};
    const size_t want[4] = {8, 4, 4, 6};
    for (int part = 0; part < 4; ++part) {
        const std::vector<int> ls = layers(b[part], b[part + 1], part == 3);
        const std::vector<int> lead = glm_kv_leads(ls, NL, full, true, 8, n);
        total += n;
        check(n == want[part], "part " + std::to_string(part) + ": " + std::to_string(n) + " maps");
        bool ok = true;
        for (int il : ls) {
            const int ld = lead[(size_t) il];
            ok = ok && ld >= 0 && ld <= il && (il == NL ? ld == NL : ld >= b[part] && ld < NL);
            if (il < NL && full(il)) ok = ok && ld == il;
            if (il < NL && !full(il)) {   // the nearest full layer below it, on this part
                int f = il;
                while (f > b[part] && !full(f)) --f;
                ok = ok && ld == f && full(f);
            }
        }
        check(ok, "part " + std::to_string(part) + ": every layer led by its group's full layer");
        for (int il = 0; il <= NL; ++il)
            if (std::find(ls.begin(), ls.end(), il) == ls.end()) ok = ok && lead[(size_t) il] == -1;
        check(ok, "part " + std::to_string(part) + ": layers of other parts lead nothing");
    }
    check(total == 22, "22 maps for the 79 caches of the 4-part split (" + std::to_string(total) + ")");
    {
        const std::vector<int> ls = layers(0, NL, true);
        glm_kv_leads(ls, NL, full, true, 8, n);
        check(n == 22, "one part: 22 maps");
        const std::vector<int> lead = glm_kv_leads(ls, NL, full, false, 8, n);
        bool ok = n == 79;
        for (int il : ls) ok = ok && lead[(size_t) il] == il;
        check(ok, "grouped off: a map a layer");
        const std::vector<int> l2 = glm_kv_leads(ls, NL, full, true, 2, n);
        ok = true;
        for (int il : ls) {   // no map serves more than 2 layers; a continued group's lead resolves for itself
            int members = 0;
            for (int j : ls) members += l2[(size_t) j] == l2[(size_t) il];
            ok = ok && members <= 2 && l2[(size_t) il] <= il;
        }
        check(ok && n > 22, "per_set 2: groups continue on further maps");
    }
    {   // a part that starts on a shared layer (a split the engine refuses; the rule stays safe): it leads itself
        const std::vector<int> ls = layers(4, 10, false);
        const std::vector<int> lead = glm_kv_leads(ls, NL, full, true, 8, n);
        check(lead[4] == 4 && lead[5] == 4 && lead[6] == 6 && lead[9] == 6 && n == 2, "a part from a shared layer");
    }
}

// ---------------------------------------------------------------- the twin
struct Map {
    std::vector<int> page_table, slot_block, slot_stamp, slot_ref, miss_block, miss_slot;
    int epoch = 0, hand = 0, n_miss = 0;
    bool overflow = false;
    uint64_t misses = 0, lookups = 0;
};

struct Twin {
    int page = 1, n_slots = 0, n_blocks = 0, RT = 8;   // RT: the sweep's step (the kernel's block of 1,024)
    int64_t ctx = 0;
    std::vector<int> ls, lead;                         // this part's layers, their maps' leads
    std::vector<Map> map;                              // by layer (a lead's is used)
    std::vector<std::vector<uint64_t>> host, slots, stage;   // by layer: a value a position (0: never written)
    bool window = true, ident = false;
    int64_t pos = 0;                                   // positions written
    // Mixer::kv_sel_*
    int sel_lead = -1;
    int64_t sel_pos = -1;
    std::vector<int> slot_cells;
    uint64_t resolves = 0, reuses = 0, idents = 0;

    int cells() const { return n_slots * page; }
    Map& M(int il) { return map[(size_t) lead[(size_t) il]]; }

    void reset_maps(bool identity) {
        const bool id = identity && window;
        for (int il : ls) {
            if (lead[(size_t) il] != il) continue;
            Map& m = map[(size_t) il];
            m = Map{};
            m.page_table.assign((size_t) n_blocks, -1);
            m.slot_block.assign((size_t) n_slots, -1);
            m.slot_stamp.assign((size_t) n_slots, -1);
            m.slot_ref.assign((size_t) n_slots, 0);
            m.miss_block.assign((size_t) n_slots, -1);
            m.miss_slot.assign((size_t) n_slots, -1);
            if (id)
                for (int b = 0; b < std::min(n_slots, n_blocks); ++b) m.page_table[(size_t) b] = m.slot_block[(size_t) b] = b;
        }
        ident = id;
    }
    void init(int n_layers, const std::vector<int>& layers, const std::vector<int>& leads, int pg, int slots_n,
              int64_t context) {
        page = pg;
        n_slots = slots_n;
        ctx = context;
        n_blocks = (int) ((context + pg - 1) / pg);
        ls = layers;
        lead = leads;
        map.assign((size_t) n_layers + 1, Map{});
        host.assign((size_t) n_layers + 1, {});
        slots.assign((size_t) n_layers + 1, {});
        stage.assign((size_t) n_layers + 1, {});
        for (int il : ls) {
            host[(size_t) il].assign((size_t) n_blocks * pg, 0);
            slots[(size_t) il].assign((size_t) n_slots * pg, 0);
        }
        slot_cells.assign(4096, -1);
        reset_maps(true);
        pos = 0;
    }
    // Glm5Model::reset: the arena zeroed, the maps start over
    void reset() {
        for (int il : ls) std::fill(slots[(size_t) il].begin(), slots[(size_t) il].end(), 0);
        reset_maps(true);
        pos = 0;
        sel_lead = -1;
    }
    // dsa_prep: the host row always, the slot row while the block is resident
    void prep(int il, int64_t p, uint64_t v) {
        host[(size_t) il][(size_t) p] = v;
        const int sl = M(il).page_table[(size_t) (p / page)];
        if (sl >= 0) slots[(size_t) il][(size_t) sl * page + p % page] = v;
    }
    // glm_kv_stream.cu resolve_block + copy_set_kernel, one thread
    void resolve(int il, const std::vector<int>& cells) {
        Map& m = M(il);
        const int epoch = m.epoch + 1, n_sel = (int) cells.size();
        int n_miss = 0, lookups = 0;
        for (int i = 0; i < n_sel; ++i) {
            const int c = cells[(size_t) i];
            if (c < 0) continue;
            const int b = c / page;
            if (i > 0 && cells[(size_t) i - 1] >= 0 && cells[(size_t) i - 1] / page == b) continue;
            ++lookups;
            const int sl = m.page_table[(size_t) b];
            if (sl >= 0) {
                m.slot_stamp[(size_t) sl] = epoch;
                m.slot_ref[(size_t) sl] = 1;
            } else if (sl == -1) {
                m.page_table[(size_t) b] = -2;
                m.miss_block[(size_t) n_miss++] = b;
            }
        }
        const int need = n_miss, n = n_slots;
        int hand = m.hand, got = 0;
        for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
            const int want = need - got;
            std::vector<int> js((size_t) RT);
            std::vector<char> mine((size_t) RT), cand((size_t) RT);
            int total = 0, cut = RT;
            for (int t = 0; t < RT; ++t) {
                const int j = (int) (((long long) hand + t) % n);
                js[(size_t) t] = j;
                mine[(size_t) t] = m.slot_stamp[(size_t) j] == epoch;
                cand[(size_t) t] = !mine[(size_t) t] && (m.slot_block[(size_t) j] < 0 || m.slot_ref[(size_t) j] == 0);
            }
            int rank = 0;
            std::vector<int> ranks((size_t) RT);
            for (int t = 0; t < RT; ++t) {
                ranks[(size_t) t] = rank;
                if (cand[(size_t) t]) {
                    if (rank == want - 1) cut = t + 1;
                    ++rank;
                }
            }
            total = rank;
            for (int t = 0; t < RT; ++t) {
                const int j = js[(size_t) t];
                if (cand[(size_t) t] && ranks[(size_t) t] < want) {
                    m.miss_slot[(size_t) (got + ranks[(size_t) t])] = j;
                    m.slot_stamp[(size_t) j] = epoch;
                } else if (t < cut && !mine[(size_t) t]) {
                    m.slot_ref[(size_t) j] = 0;
                }
            }
            got += total < want ? total : want;
            hand = (int) (((long long) hand + cut) % n);
        }
        const int placed = got < need ? got : need;
        for (int k = 0; k < need; ++k) {
            const int b = m.miss_block[(size_t) k];
            if (k >= placed) {
                m.page_table[(size_t) b] = -1;
                continue;
            }
            const int sl = m.miss_slot[(size_t) k], old = m.slot_block[(size_t) sl];
            if (old >= 0) m.page_table[(size_t) old] = -1;
            m.slot_block[(size_t) sl] = b;
            m.slot_stamp[(size_t) sl] = epoch;
            m.slot_ref[(size_t) sl] = 1;
            m.page_table[(size_t) b] = sl;
        }
        for (int i = 0; i < n_sel; ++i) {
            const int c = cells[(size_t) i];
            const int sl = c < 0 ? -1 : m.page_table[(size_t) (c / page)];
            slot_cells[(size_t) i] = sl < 0 ? -1 : sl * page + c % page;
        }
        m.epoch = epoch;
        m.hand = hand;
        m.n_miss = placed;
        if (placed < need) m.overflow = true;
        m.misses += (uint64_t) placed;
        m.lookups += (uint64_t) lookups;
        // the group copy: every missed block, for every layer on this map
        for (int k = 0; k < placed; ++k)
            for (int jl : ls)
                if (lead[(size_t) jl] == lead[(size_t) il])
                    for (int r = 0; r < page; ++r)
                        slots[(size_t) jl][(size_t) m.miss_slot[(size_t) k] * page + r] =
                            host[(size_t) jl][(size_t) m.miss_block[(size_t) k] * page + r];
    }
    // dsa_mixer_cache + dsa_kv_resolve: the rows layer il's attention reads for `cells` at position p
    std::vector<int> attend_rows(int il, int64_t p, const std::vector<int>& cells, bool group) {
        const bool id = glm_kv_window_serves(ident, p, cells_n());
        if (!id) ident = false;   // kv_window_hit
        if (id) {
            ++idents;
            return cells;
        }
        const int ld = lead[(size_t) il];
        if (group && ld != il && sel_lead == ld && sel_pos == p) {
            ++reuses;
        } else {
            resolve(il, cells);
            ++resolves;
            if (group) {
                sel_lead = ld;
                sel_pos = p;
            } else {
                sel_lead = -1;
            }
        }
        return std::vector<int>(slot_cells.begin(), slot_cells.begin() + (long) cells.size());
    }
    int cells_n() const { return cells(); }
    // the prompt path's chunk [p0, p0 + T): inside the window on the slots, else on the stage
    void prompt_chunk(int64_t p0, int T, const std::vector<std::vector<uint64_t>>& vals, bool& staged) {
        const bool win = glm_kv_window_chunk(ident, p0, T, cells_n());
        staged = !win;
        if (!win) ident = false;
        for (int il : ls) {
            if (win) {   // lat = the arena's slots; kv_stage_out copies the chunk's rows to the host copy
                for (int t = 0; t < T; ++t) slots[(size_t) il][(size_t) (p0 + t)] = vals[(size_t) il][(size_t) t];
                for (int t = 0; t < T; ++t) host[(size_t) il][(size_t) (p0 + t)] = slots[(size_t) il][(size_t) (p0 + t)];
            } else {     // the stage: the prefix from the host copy, the chunk's rows, back to the host copy; map reset
                std::vector<uint64_t>& st = stage[(size_t) il];
                st.assign(host[(size_t) il].begin(), host[(size_t) il].begin() + (long) p0);
                for (int t = 0; t < T; ++t) st.push_back(vals[(size_t) il][(size_t) t]);
                for (int t = 0; t < T; ++t) host[(size_t) il][(size_t) (p0 + t)] = st[(size_t) (p0 + t)];
                Map& m = M(il);   // kv_stream_reset (a group's map: once a layer, harmless)
                std::fill(m.page_table.begin(), m.page_table.end(), -1);
                std::fill(m.slot_block.begin(), m.slot_block.end(), -1);
                std::fill(m.slot_stamp.begin(), m.slot_stamp.end(), -1);
                std::fill(m.slot_ref.begin(), m.slot_ref.end(), 0);
                m.epoch = m.hand = 0;
            }
        }
        sel_lead = -1;
    }
    // kv_stream_prime after a slot / prompt-cache load of n positions into the host copies
    void prime(int64_t n) {
        const bool id = glm_kv_window_load(window, n, cells_n());
        reset_maps(id);
        if (id)
            for (int il : ls)
                for (int64_t r = 0; r < n; ++r) slots[(size_t) il][(size_t) r] = host[(size_t) il][(size_t) r];
        sel_lead = -1;
    }
    bool map_ok() const {
        for (int il : ls) {
            if (lead[(size_t) il] != il) continue;
            const Map& m = map[(size_t) il];
            if (m.overflow) return false;
            for (int b = 0; b < n_blocks; ++b) {
                const int sl = m.page_table[(size_t) b];
                if (sl < -1 || sl >= n_slots || (sl >= 0 && m.slot_block[(size_t) sl] != b)) return false;
            }
            for (int sl = 0; sl < n_slots; ++sl) {
                const int b = m.slot_block[(size_t) sl];
                if (b >= n_blocks || (b >= 0 && m.page_table[(size_t) b] != sl)) return false;
            }
        }
        return true;
    }
};

// a selection like the indexer's: up to top_k distinct cells of 0..p, ascending within a page run, recent-heavy
std::vector<int> select(std::mt19937& rng, int64_t p, int top_k, int cap) {
    std::vector<int> c;
    if (p + 1 <= top_k) {
        for (int64_t i = 0; i <= p; ++i) c.push_back((int) i);
    } else {
        std::set<int> s;
        s.insert((int) p);
        while ((int) s.size() < top_k) {
            const int64_t back = (rng() & 3) ? (int64_t) (rng() % (uint32_t) std::min<int64_t>(p + 1, 4 * top_k)) : (int64_t) (rng() % (uint32_t) (p + 1));
            s.insert((int) (p - back));
        }
        c.assign(s.begin(), s.end());
        if (rng() & 1) std::reverse(c.begin(), c.end());   // (any order: the resolve only skips adjacent repeats)
    }
    while ((int) c.size() < cap) c.push_back(-1);
    return c;
}

struct Scenario {
    int page, n_slots, top_k;
    int64_t ctx;
    bool grouped, window;
    uint32_t seed;
    int steps;
};

void run(const Scenario& sc) {
    std::mt19937 rng(sc.seed);
    // a part with the NextN block and layers [2, 12): groups {2..5} {6..9} {10, 11}, NextN alone
    const int NL = 12;
    std::vector<int> ls = {NL};
    for (int il = 2; il < NL; ++il) ls.push_back(il);
    size_t n_maps = 0;
    const std::vector<int> lead = glm_kv_leads(ls, NL, glm53_full, sc.grouped, 8, n_maps);
    Twin T, saved;
    T.window = sc.window;
    T.init(NL, ls, lead, sc.page, sc.n_slots, sc.ctx);
    if (!sc.window) T.reset_maps(false);
    uint64_t next_val = 1, reads = 0;
    bool have_saved = false;
    int64_t saved_n = 0;
    std::string tag = "page " + std::to_string(sc.page) + ", " + std::to_string(sc.n_slots) + " slots, top-k " +
                      std::to_string(sc.top_k) + (sc.grouped ? ", grouped" : ", a map a layer") +
                      (sc.window ? ", window" : ", no window") + ", seed " + std::to_string(sc.seed);
    int bad = 0;
    std::string first_bad;
    const auto read_check = [&](int il, int64_t p, const std::vector<int>& cells, const std::vector<int>& rows) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i] < 0) continue;
            ++reads;
            const bool ok = rows[i] >= 0 && T.slots[(size_t) il][(size_t) rows[i]] == T.host[(size_t) il][(size_t) cells[i]] &&
                            T.host[(size_t) il][(size_t) cells[i]] != 0;
            if (!ok && bad++ == 0)
                first_bad = "layer " + std::to_string(il) + " at position " + std::to_string(p) + " reads cell " +
                            std::to_string(cells[i]) + " from row " + std::to_string(rows[i]);
        }
    };
    // one decode token at position p (every layer of the part in order, then the NextN block), as fast_layers_plain
    const auto token = [&](int64_t p) {
        std::vector<int> cells;
        for (size_t i = 1; i <= ls.size(); ++i) {
            const int il = ls[i % ls.size()];   // trunk layers ascending, the NextN block last
            T.prep(il, p, next_val++);
            if (il == NL || glm53_full(il)) cells = select(rng, p, sc.top_k, sc.top_k);
            read_check(il, p, cells, T.attend_rows(il, p, cells, true));
        }
    };
    // ... and a window's rows (mixer_rows: layers outer, rows inner, each row its own resolve)
    const auto rows_tokens = [&](int64_t p0, int n) {
        std::vector<std::vector<int>> cells((size_t) n);
        for (size_t i = 1; i <= ls.size(); ++i) {
            const int il = ls[i % ls.size()];
            for (int r = 0; r < n; ++r) {
                T.prep(il, p0 + r, next_val++);
                if (il == NL || glm53_full(il)) cells[(size_t) r] = select(rng, p0 + r, sc.top_k, sc.top_k);
                const bool id = glm_kv_window_serves(T.ident, p0 + n - 1, T.cells_n());
                std::vector<int> rows;
                if (id) {
                    rows = cells[(size_t) r];
                } else {
                    T.ident = false;
                    T.resolve(il, cells[(size_t) r]);
                    T.sel_lead = -1;
                    rows.assign(T.slot_cells.begin(), T.slot_cells.begin() + (long) cells[(size_t) r].size());
                }
                read_check(il, p0 + r, cells[(size_t) r], rows);
            }
        }
    };
    uint64_t staged_chunks = 0, window_chunks = 0, loads_in = 0, loads_out = 0;
    for (int step = 0; step < sc.steps; ++step) {
        const uint32_t dice = rng() % 100;
        if (dice < 55) {   // decode a few tokens, some of them rejected drafts written again
            const int n = 1 + (int) (rng() % 12);
            for (int i = 0; i < n && T.pos < sc.ctx; ++i) {
                token(T.pos);
                if (rng() % 5 == 0) token(T.pos);   // the draft was wrong: the position again
                ++T.pos;
            }
        } else if (dice < 62) {   // a verify window's rows
            const int n = 1 + (int) (rng() % 4);
            if (T.pos + n <= sc.ctx) {
                rows_tokens(T.pos, n);
                T.pos += n;
            }
        } else if (dice < 80) {   // a prompt in chunks
            int64_t n = 1 + (int64_t) (rng() % (uint32_t) (2 * T.cells_n()));
            n = std::min<int64_t>(n, sc.ctx - T.pos);
            const int Tc = 1 + (int) (rng() % 257);
            while (n > 0) {
                const int tn = (int) std::min<int64_t>(n, Tc);
                std::vector<std::vector<uint64_t>> vals((size_t) NL + 1);
                for (int il : ls)
                    for (int t = 0; t < tn; ++t) vals[(size_t) il].push_back(next_val++);
                bool staged = false;
                T.prompt_chunk(T.pos, tn, vals, staged);
                (staged ? staged_chunks : window_chunks) += 1;
                T.pos += tn;
                n -= tn;
            }
        } else if (dice < 86) {   // an edited turn: back to an earlier position
            if (T.pos > 1) T.pos = 1 + (int64_t) (rng() % (uint32_t) (T.pos - 1));
        } else if (dice < 91) {   // save the conversation (the host copies' rows 0..pos)
            saved = T;
            saved_n = T.pos;
            have_saved = T.pos > 0;
        } else if (dice < 97) {   // another conversation ran here; the saved one comes back from its slot
            if (have_saved) {
                for (int il : ls) {   // (the slots hold whatever the other conversation left)
                    for (auto& v : T.slots[(size_t) il]) v = next_val++;
                    for (int64_t r = 0; r < saved_n; ++r) T.host[(size_t) il][(size_t) r] = saved.host[(size_t) il][(size_t) r];
                }
                T.prime(saved_n);
                (T.ident ? loads_in : loads_out) += 1;
                T.pos = saved_n;
            }
        } else {
            T.reset();
        }
        if (!T.map_ok()) {
            check(false, tag + ": the residency map is inconsistent at step " + std::to_string(step));
            return;
        }
    }
    check(bad == 0, tag + ": " + std::to_string(reads) + " rows read, " + std::to_string(bad) + " wrong" +
                        (bad ? " (first: " + first_bad + ")" : "") + " | resolves " + std::to_string(T.resolves) +
                        ", reuses " + std::to_string(T.reuses) + ", window reads " + std::to_string(T.idents) +
                        ", prompt chunks " + std::to_string(window_chunks) + " in the window / " +
                        std::to_string(staged_chunks) + " staged, loads " + std::to_string(loads_in) + " / " +
                        std::to_string(loads_out));
    // the protocol's point: a group resolves once a token, and the window resolves nothing
    if (sc.grouped && T.resolves + T.reuses > 0) check(T.reuses > 0, tag + ": shared layers reuse their group's resolve");
    if (!sc.window) check(T.idents == 0, tag + ": no window reads with the window off");
}

}  // namespace

int main() {
    std::printf("glm_kv_groups_test: residency-map groups and the identity window (CPU twin)\n");
    test_leads();
    uint32_t seed = 1;
    for (const int page : {1, 4})
        for (const bool grouped : {true, false})
            for (const bool window : {true, false})
                for (const int n_slots : {24, 64, 400}) {
                    const int top_k = page == 1 ? std::min(16, n_slots / 2) : std::min(16, n_slots * page / 4);
                    run({page, n_slots, top_k, (int64_t) n_slots * page * 6, grouped, window, seed++, 1500});
                    run({page, n_slots, top_k, (int64_t) n_slots * page * 2, grouped, window, seed++, 1500});
                }
    if (g_fail) {
        std::printf("glm_kv_groups_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_kv_groups_test: ok\n");
    return 0;
}
