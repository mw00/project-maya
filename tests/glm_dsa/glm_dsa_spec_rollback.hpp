// tests/glm_dsa/glm_dsa_spec_rollback.hpp - glm-dsa B11.2: the rejected-draft rollback check on any mixer backend
// (the CPU twins in glm_dsa_spec_rollback_test.cpp, the device kernels in glm_dsa_spec_gpu_test.cu).
//
// The pipelined speculative decode's head runs position q+2 with the draft token; when the tail's token differs, the
// head redoes q+2 with it (glm_fast_path.cu decode_spec).  Flash restores its KDA states first; glm-dsa has none, and
// its DSA caches (latent, k_pe, index keys; cell s = position s) are rewritten in place by the redo.  suite<Be> runs
// glmdsa::mixer_layer (the code Glm5Model::fast_dsa_glm runs) over a random layer stack with full and shared layers
// interleaved, twice:
//   A  token by token with the real inputs;
//   B  the same, except that at every rejected position the whole stack first runs with DRAFT inputs, then
//      glmdsa::sel_rollback (what mixer_rollback does to the part's selection state), then the real inputs.
// Checked, selection active (p + 1 > top_k):
//   * the draft run writes ONLY row p of each cache (latent, k_pe, and index keys on full layers): the rollback set
//     glmspec::spec_rows names;
//   * every redo output equals run A's bit for bit, and so do all caches at the end (no stale row survives);
//   * stale selection: after a draft run WITHOUT the rollback, a shared layer run at the same position accepts the
//     rejected run's selection (and its output differs from A's at some position: the hazard is real); WITH the
//     rollback it is refused, so a redo can only proceed through its full layer's fresh selection.
// schedule<Be> (glm-dsa B11.2, STRATA_GLM_SPEC_DRAFT_EARLY): decode_spec's step order on a head part, a two-part tail
// group and the NextN block on the tail's last part, each part its own harness (its own caches and selection state,
// as each part has its own arena), today's order and the draft-early one, against the token-at-a-time run.
// The caller defines `void check(bool, const std::string&)` before including this header.
#pragma once

#include "glm_dsa_mixer_fixture.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace spec_rb {

// one cache region of one layer: base, bytes per position
struct Region {
    const uint8_t* base;
    size_t row;
};
template <class H>
std::vector<Region> regions(const H& h, const mixfx::Model& m, int il) {
    const mixfx::MixerDims& d = m.d;
    const auto& c = h.cache[(size_t) il];
    const size_t lat_b = h.lat_q8 ? (size_t) glmd::lat8_rec_bytes(d.kv_lora) : (size_t) d.kv_lora * 2;
    std::vector<Region> r = {{(const uint8_t*) c.lat, lat_b}, {(const uint8_t*) c.k_pe, (size_t) d.qk_rope * 2}};
    if (m.L[(size_t) il].full) r.push_back({(const uint8_t*) c.ik, (size_t) d.idx_key * 4});
    return r;
}
template <class H>
std::vector<std::vector<uint8_t>> snapshot(const H& h, const mixfx::Model& m) {
    std::vector<std::vector<uint8_t>> s;
    for (int il = 0; il < (int) m.L.size(); ++il)
        for (const Region& r : regions(h, m, il)) {
            s.emplace_back(r.row * (size_t) m.d.max_ctx);
            h.be.get(s.back().data(), r.base, s.back().size());
        }
    return s;
}

template <class Be>
void suite(const char* name, const mixfx::MixerDims& dims, int T, int lat_q8) {
    // full / shared interleave as GLM-5.3 (layer 0 full); two full layers so a shared layer follows each
    const std::vector<bool> pattern = {true, false, false, true, false};
    mixfx::Model m = mixfx::make_model(dims, pattern, T, 17);
    const mixfx::Model draft = mixfx::make_model(dims, pattern, T, 91);   // other inputs at every (layer, position)
    std::printf("[%s, %s latents: %zu layers, %d positions, top-%d]\n", name, lat_q8 ? "INT8" : "FP16", m.L.size(), T,
                dims.top_k);
    using H = mixfx::Harness<Be>;
    Be ba, bb;
    H A(ba, m, mixfx::Mut::None, false, lat_q8);
    H B(bb, m, mixfx::Mut::None, false, lat_q8);
    const int E = dims.n_embd, NL = (int) m.L.size();
    const auto reject = [&](int p) { return p >= 2 && p % 3 != 0; };   // two of three positions are redone
    std::vector<float> oa, ob, stale;
    std::string err;
    bool out_equal = true, write_set_ok = true, draft_differs = true, probe_accepted = true, probe_refused = true;
    int redone = 0, stale_wrong = 0, probes = 0;
    for (int p = 0; p < T; ++p) {
        std::vector<std::vector<float>> outA((size_t) NL);
        for (int il = 0; il < NL; ++il) {
            if (!A.step(il, p, outA[(size_t) il], err)) { check(false, "run A: " + err); return; }
        }
        if (reject(p)) {
            ++redone;
            const auto before = snapshot(B, m);
            // the speculative position with the draft's inputs
            const size_t row = (size_t) p * NL * E;
            std::vector<float> keep(m.X.begin() + (long) row, m.X.begin() + (long) (row + (size_t) NL * E));
            std::copy(draft.X.begin() + (long) row, draft.X.begin() + (long) (row + (size_t) NL * E), m.X.begin() + (long) row);
            for (int il = 0; il < NL; ++il)
                if (!B.step(il, p, ob, err)) { check(false, "draft run: " + err); return; }
            std::copy(keep.begin(), keep.end(), m.X.begin() + (long) row);
            // what the draft wrote: row p of each cache, nothing else
            const auto after = snapshot(B, m);
            size_t k = 0;
            bool any = false;
            for (int il = 0; il < NL; ++il)
                for (const Region& r : regions(B, m, il)) {
                    const auto& x0 = before[k];
                    const auto& x1 = after[k++];
                    for (size_t b = 0; b < x0.size(); ++b)
                        if (x0[b] != x1[b]) {
                            any = true;
                            if (b / r.row != (size_t) p) write_set_ok = false;
                        }
                }
            draft_differs = draft_differs && any;
            // stale selection probe (selection active): the last shared layer (src = layer 3) at p
            if (p + 1 > dims.top_k) {
                ++probes;
                mixfx::SelState saved = B.st;
                std::string e1, e2;
                const bool acc = B.step(4, p, stale, e1);   // no rollback: the rejected run's selection is accepted
                probe_accepted = probe_accepted && acc;
                if (acc && std::memcmp(stale.data(), outA[4].data(), (size_t) E * 4) != 0) ++stale_wrong;
                B.st = saved;
                strata::core::glmdsa::sel_rollback(B.st);
                const bool ref = !B.step(4, p, stale, e2) && e2.find("not made at this position") != std::string::npos;
                probe_refused = probe_refused && ref;
            }
            strata::core::glmdsa::sel_rollback(B.st);   // glmdsa::mixer_rollback on the part
        }
        for (int il = 0; il < NL; ++il) {
            if (!B.step(il, p, ob, err)) { check(false, "run B: " + err); return; }
            if (std::memcmp(ob.data(), outA[(size_t) il].data(), (size_t) E * 4) != 0) out_equal = false;
        }
    }
    check(redone > 0 && draft_differs, std::to_string(redone) + " rejected positions, each draft run changed the caches");
    check(write_set_ok, "a draft run writes only row p of each latent / k_pe / index-key cache (the spec_rows set)");
    check(out_equal, "every redo output == the token-at-a-time output, bit for bit");
    check(snapshot(A, m) == snapshot(B, m), "all caches equal at the end: no stale row survives a redo");
    check(probes > 0 && probe_accepted, "without the rollback a shared layer accepts the rejected selection (" +
                                            std::to_string(probes) + " probes)");
    check(stale_wrong > 0, "... and reads the wrong cells: its output differs at " + std::to_string(stale_wrong) + " / " +
                               std::to_string(probes) + " positions");
    check(probe_refused, "with the rollback the same shared layer is refused (no stale selection is ever read)");
}

// ---------------------------------------------------------------- the step order (glm-dsa B11.2)
// The layers: head [0, 2), tail part 1 [2, 4), tail part 2 [4, 6), each part starting with a full layer (as
// glm_split_bounds_ok demands), and the NextN block as layer 6 (full) on part 2.  The block's input at position p is
// the carrier: part 2's last trunk output, noted when part 2 finishes p (B15.3: a draft that finds the row of another
// position reads zeros).  Per position s the tail takes (the head ran s with the draft's inputs one step before when
// s is rejected, else with the real ones):
//   Order::Today  the draft at s-1, the head's verdict at s (rejected: rollback + redo), part 1, part 2, the head at s+1;
//   Order::Early  the verdict, part 1, the draft at s-1, part 2, the head at s+1 (decode_spec with
//                 STRATA_GLM_SPEC_DRAFT_EARLY=1: spec_tail_front, the draft, spec_tail_back);
//   Order::Wrong  the verdict, part 1, part 2, the draft at s-1 - part 2 notes s before the draft read s-1 (the order
//                 spec_tail_back must not take; the control).
enum class Order { Today, Early, Wrong };

inline const char* order_name(Order o) {
    return o == Order::Today   ? "today's order (the draft first)"
           : o == Order::Early ? "STRATA_GLM_SPEC_DRAFT_EARLY's order (the draft after part 1)"
                               : "the wrong order (part 2 notes s before the draft reads s-1)";
}

inline bool rejected(int p) { return p >= 2 && p % 3 != 0; }   // two of three positions redone, as suite()

template <class H>
std::vector<std::vector<uint8_t>> layer_rows(const H& h, const mixfx::Model& m, int il) {
    std::vector<std::vector<uint8_t>> s;
    for (const Region& r : regions(h, m, il)) {
        s.emplace_back(r.row * (size_t) m.d.max_ctx);
        h.be.get(s.back().data(), r.base, s.back().size());
    }
    return s;
}

struct OrderRun {
    std::vector<std::vector<float>> out;                  // [p * layers + il]: the last output at (il, p)
    std::vector<std::vector<std::vector<uint8_t>>> rows;  // [il]: the layer's cache rows at the end (its part's)
    int stale = 0, redone = 0;                            // drafts that found another position's carrier; redos
    std::string err;
};

// one run of decode_spec's steps in order o; m.X holds the real inputs on entry (X0) and is used for the inputs of
// the moment (the draft's inputs of a rejected head position, the carrier of the NextN block)
template <class Be>
OrderRun run_order(mixfx::Model& m, const mixfx::Model& draft, const std::vector<float>& X0, int lat_q8, Order o) {
    using H = mixfx::Harness<Be>;
    const int NL = (int) m.L.size(), E = m.d.n_embd, T = m.T;
    const int kP1 = 2, kP2 = 4, kNext = NL - 1;
    OrderRun r;
    r.out.assign((size_t) T * NL, std::vector<float>());
    Be b0, b1, b2;
    H h0(b0, m, mixfx::Mut::None, false, lat_q8);
    H h1(b1, m, mixfx::Mut::None, false, lat_q8);
    H h2(b2, m, mixfx::Mut::None, false, lat_q8);
    std::vector<float> carrier((size_t) E, 0.0f);
    const std::vector<float> zeros((size_t) E, 0.0f);
    int carrier_pos = -1;
    const auto set_x = [&](int il, int p, const float* v) {
        std::copy(v, v + E, m.X.begin() + (long) (((size_t) p * NL + il) * E));
    };
    const auto run = [&](H& h, int l0, int l1, int p) -> bool {
        for (int il = l0; il < l1; ++il)
            if (!h.step(il, p, r.out[(size_t) p * NL + il], r.err)) return false;
        return true;
    };
    const auto head = [&](int p, bool draft_inputs) -> bool {   // the head part at p
        const float* src = draft_inputs ? draft.X.data() : X0.data();
        for (int il = 0; il < kP1; ++il) set_x(il, p, src + ((size_t) p * NL + il) * E);
        const bool ok = run(h0, 0, kP1, p);
        for (int il = 0; il < kP1; ++il) set_x(il, p, X0.data() + ((size_t) p * NL + il) * E);
        return ok;
    };
    const auto part2 = [&](int p) -> bool {   // part 2's trunk layers at p, then the carrier note (after res_head)
        if (!run(h2, kP2, kNext, p)) return false;
        carrier = r.out[(size_t) p * NL + kNext - 1];
        carrier_pos = p;
        return true;
    };
    const auto next_block = [&](int p) -> bool {   // the NextN draft at p on part 2: the carrier when it holds p
        if (carrier_pos != p) ++r.stale;
        set_x(kNext, p, carrier_pos == p ? carrier.data() : zeros.data());
        return h2.step(kNext, p, r.out[(size_t) p * NL + kNext], r.err);
    };
    if (!head(0, false)) return r;
    for (int s = 0; s < T; ++s) {
        if (o == Order::Today && s > 0 && !next_block(s - 1)) return r;
        if (rejected(s)) {   // the verdict: restore (glm-dsa: the selection state) and redo with the real inputs
            ++r.redone;
            strata::core::glmdsa::sel_rollback(h0.st);
            if (!head(s, false)) return r;
        }
        if (!run(h1, kP1, kP2, s)) return r;
        if (o == Order::Early && s > 0 && !next_block(s - 1)) return r;
        if (!part2(s)) return r;
        if (o == Order::Wrong && s > 0 && !next_block(s - 1)) return r;
        if (s + 1 < T && !head(s + 1, rejected(s + 1))) return r;
    }
    if (!next_block(T - 1)) return r;
    for (int il = 0; il < NL; ++il) r.rows.push_back(layer_rows(il < kP1 ? h0 : il < kP2 ? h1 : h2, m, il));
    return r;
}

// early = false: today's order; true: the draft-early order, plus the wrong-order control
template <class Be>
void schedule(const char* name, const mixfx::MixerDims& dims, int T, int lat_q8, bool early) {
    const std::vector<bool> pattern = {true, false, true, false, true, false, true};
    mixfx::Model m = mixfx::make_model(dims, pattern, T, 23);
    const mixfx::Model draft = mixfx::make_model(dims, pattern, T, 97);   // the head's inputs at a rejected draft
    const std::vector<float> X0 = m.X;
    const int NL = (int) m.L.size(), E = dims.n_embd, kNext = NL - 1;
    const Order o = early ? Order::Early : Order::Today;
    std::printf("[%s, %s latents, decode_spec's step: head / tail part 1 / tail part 2 + NextN, %d positions, %s]\n",
                name, lat_q8 ? "INT8" : "FP16", T, order_name(o));
    // run A: one harness, token by token, the NextN block right after each position on that position's carrier
    std::vector<std::vector<float>> outA((size_t) T * NL);
    std::vector<std::vector<std::vector<uint8_t>>> rowsA;
    {
        Be be;
        mixfx::Harness<Be> A(be, m, mixfx::Mut::None, false, lat_q8);
        std::string err;
        for (int p = 0; p < T; ++p) {
            for (int il = 0; il < kNext; ++il)
                if (!A.step(il, p, outA[(size_t) p * NL + il], err)) { check(false, "run A: " + err); return; }
            const std::vector<float>& c = outA[(size_t) p * NL + kNext - 1];
            std::copy(c.begin(), c.end(), m.X.begin() + (long) (((size_t) p * NL + kNext) * E));
            if (!A.step(kNext, p, outA[(size_t) p * NL + kNext], err)) { check(false, "run A: " + err); return; }
        }
        for (int il = 0; il < NL; ++il) rowsA.push_back(layer_rows(A, m, il));
    }
    const auto same = [&](const std::vector<float>& a, const std::vector<float>& b) {
        return a.size() == b.size() && !a.empty() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    };
    const OrderRun r = run_order<Be>(m, draft, X0, lat_q8, o);
    if (!r.err.empty()) {
        check(false, std::string(order_name(o)) + ": " + r.err);
        return;
    }
    bool trunk = true, drafts = true;
    for (int p = 0; p < T; ++p)
        for (int il = 0; il < NL; ++il)
            if (!same(r.out[(size_t) p * NL + il], outA[(size_t) p * NL + il])) (il == kNext ? drafts : trunk) = false;
    check(r.redone > 0 && r.stale == 0, std::to_string(r.redone) + " head positions redone; every draft read the "
                                        "carrier of its own position");
    check(trunk, "every part's output == the token-at-a-time run, bit for bit (head redos included)");
    check(drafts, "every NextN draft output == the token-at-a-time run's, bit for bit");
    check(r.rows == rowsA, "every cache equal at the end (each layer on its part)");
    if (!early) return;
    // the control: part 2 notes s before the draft read s-1 - the draft finds the next position's carrier (zeros),
    // its output differs (the last draft too: it attends over the rows the stale ones wrote); the trunk, and so every
    // token, still equals run A (the draft only steers acceptance)
    const OrderRun w = run_order<Be>(m, draft, X0, lat_q8, Order::Wrong);
    if (!w.err.empty()) {
        check(false, std::string(order_name(Order::Wrong)) + ": " + w.err);
        return;
    }
    int differ = 0;
    bool w_trunk = true;
    for (int p = 0; p < T; ++p)
        for (int il = 0; il < NL; ++il) {
            const bool eq = same(w.out[(size_t) p * NL + il], outA[(size_t) p * NL + il]);
            if (il == kNext) differ += eq ? 0 : 1;
            else if (!eq) w_trunk = false;
        }
    check(w.stale == T - 1 && differ >= T - 1,
          std::string("control, ") + order_name(Order::Wrong) + ": " + std::to_string(w.stale) + " drafts found the next "
          "position's carrier, " + std::to_string(differ) + " of " + std::to_string(T) + " drafts differ");
    check(w_trunk, "control: the trunk still equals the token-at-a-time run (a wrong draft never changes a token)");
}

}  // namespace spec_rb
