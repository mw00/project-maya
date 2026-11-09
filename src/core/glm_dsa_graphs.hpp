// src/core/glm_dsa_graphs.hpp - glm-dsa B10.1: the per-part decode graph cache of the glm-dsa fast path.
//
// A captured graph per part (all of its layers) and per state arena - one per sequence with STRATA_GLM_SEQS /
// --parallel, up to set_per_part() a part (glmd::GraphIndex) - keyed by glmd::GraphKey (include/strata/kernels/glmd_graph.hpp:
// the B14.4 rules every captured kernel follows).  The same request accounting as Maya's glmfast::LayerGraphs
// (src/core/glm_layer_graphs.hpp): the body runs once at capture and does its own accounting (the MoE CPU-lane
// requests it posts); a replay runs no host code, so enqueue() adds `requests` to `expected` itself, and the
// CPU-lane waits stay device-side spins inside the graph.  A key change (a regime / capacity boundary, a switch, a
// moved buffer) drops that graph and captures a new one; the caller changes plans only at token boundaries.
//
// Wiring (B4.6 / the mixer): per token, before the part,
//     glmd::set_dev_pos(dev_pos, p, top_k, stream);            // outside the capture: the position is DATA
//     GraphKey k{part, l0, l1, cpu_plan, R, arena, state, bits, regime, n_cap};
//     graphs.enqueue(k, stream, expected, requests, body, replayed, err);
// where body() issues every layer of the part with p_dev / pos = &dev_pos->p, n_kv = k.n_cap, n_sel = top_k.
#pragma once

#include "strata/core/graph.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace strata::core::glmdsa {

class DsaGraphs {
public:
    bool enqueue(const glmd::GraphKey& key, cudaStream_t stream, uint64_t& expected, uint64_t requests,
                 const std::function<bool()>& body, bool& replayed, std::string& err) {
        replayed = false;
        const glmd::GraphIndex::Pick pk = idx_.pick(key);
        if (pk.drop >= 0) graphs_.erase(pk.drop);
        invalidations += pk.invalidated ? 1 : 0;
        evictions += pk.evicted ? 1 : 0;
        if (pk.replay) {
            if (!graphs_[pk.slot].launch(stream, err)) return false;
            expected += requests;
            replayed = true;
            ++replays;
            return true;
        }
        const uint64_t before = expected;
        CapturedGraph graph;
        if (!graph.begin(stream, err)) return false;
        const bool ok = body();
        const cudaError_t launch_error = cudaGetLastError();
        if (!ok || launch_error != cudaSuccess || expected != before + requests) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            expected = before;
            if (err.empty())
                err = launch_error != cudaSuccess ? std::string("glm-dsa graph capture: ") + cudaGetErrorString(launch_error)
                                                  : "glm-dsa graph capture: body or request accounting failed";
            return false;
        }
        if (!graph.end(stream, err) || !graph.launch(stream, err)) {
            expected = before;
            return false;
        }
        last_nodes = graph.nodes();
        graphs_[pk.slot] = std::move(graph);
        idx_.commit(pk.slot, key);
        ++captures;
        return true;
    }
    /// Graphs kept per part: the sequence count (STRATA_GLM_SEQS / --parallel), so each sequence's state keeps its own.
    void set_per_part(int n) { idx_.set_per_part(n); }
    /// Kernel / memcpy nodes of a part's most recently used graph (the launches a replay saves), 0 when none.
    size_t nodes(int part) const {
        const int i = idx_.latest(part);
        const auto it = graphs_.find(i);
        return i < 0 || it == graphs_.end() ? 0 : it->second.nodes();
    }
    void clear() {
        graphs_.clear();
        idx_.clear();
    }
    uint64_t captures = 0, replays = 0, invalidations = 0, evictions = 0;
    size_t last_nodes = 0;

private:
    glmd::GraphIndex idx_;
    std::map<int, CapturedGraph> graphs_;   // by slot
};

}  // namespace strata::core::glmdsa
