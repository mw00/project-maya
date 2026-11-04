// Opt-in GLM layer capture: no per-layer wait, explicit host request accounting.
#pragma once

#include "strata/core/graph.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace strata::core::glmfast {

class LayerGraphs {
public:
    struct Key {
        unsigned long long cpu_plan = 0;
        const float* residual = nullptr;
        const float* other = nullptr;
        bool all_resident = false;   // fast_moe leaves its wait / fetch launches out then: another graph
        int pipe_t = -1;
        bool operator==(const Key& b) const {
            return cpu_plan == b.cpu_plan && residual == b.residual && other == b.other &&
                   all_resident == b.all_resident && pipe_t == b.pipe_t;
        }
    };

    struct GraphId {
        int layer = 0;
        unsigned long long cpu_plan = 0;
        const float* residual = nullptr;
        const float* other = nullptr;
        bool all_resident = false;
        int pipe_t = -1;

        bool operator<(const GraphId& b) const {
            if (layer != b.layer) return layer < b.layer;
            if (pipe_t != b.pipe_t) return pipe_t < b.pipe_t;
            if (cpu_plan != b.cpu_plan) return cpu_plan < b.cpu_plan;
            if (residual != b.residual) return residual < b.residual;
            if (other != b.other) return other < b.other;
            return all_resident < b.all_resident;
        }
    };

    // The body retains direct-enqueue request accounting. Capture invokes it once;
    // replay does not invoke host code and must account for the requests itself.
    bool enqueue(int layer, Key key, cudaStream_t stream, uint64_t& expected, uint64_t requests,
                 const std::function<bool()>& body, bool& replayed, std::string& err) {
        replayed = false;
        const GraphId gid{layer, key.cpu_plan, key.residual, key.other, key.all_resident, key.pipe_t};
        auto it = entries_.find(gid);
        if (it != entries_.end()) {
            if (!it->second.graph.launch(stream, err)) return false;
            expected += requests;
            replayed = true;
            ++replays;
            return true;
        }

        const uint64_t before = expected;
        Entry entry;
        entry.key = key;
        if (!entry.graph.begin(stream, err)) return false;
        const bool ok = body();
        const cudaError_t launch_error = cudaGetLastError();
        if (!ok || launch_error != cudaSuccess || expected != before + requests) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            expected = before;   // none of the captured work was executed
            if (err.empty()) err = launch_error != cudaSuccess
                ? std::string("GLM graph capture: ") + cudaGetErrorString(launch_error)
                : "GLM graph capture: body/request accounting failed";
            return false;
        }
        if (!entry.graph.end(stream, err) || !entry.graph.launch(stream, err)) {
            expected = before;
            return false;
        }
        entries_.emplace(gid, std::move(entry));
        ++captures;
        return true;
    }

    void clear() { entries_.clear(); }
    uint64_t captures = 0, replays = 0, invalidations = 0;

private:
    struct Entry {
        Key key;
        CapturedGraph graph;
    };
    std::map<GraphId, Entry> entries_;
};

}  // namespace strata::core::glmfast
