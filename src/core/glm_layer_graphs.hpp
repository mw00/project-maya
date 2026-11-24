// GLM decode layer capture (MAYA_GLM_KDA_GRAPH): no per-layer wait, explicit host request accounting.
#pragma once

#include "maya/core/graph.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace maya::core::glmfast {

class LayerGraphs {
public:
    struct Key {
        unsigned long long cpu_plan = 0;
        const float* residual = nullptr;
        const float* other = nullptr;
        bool all_resident = false;   // fast_moe leaves its wait / fetch launches out then: another graph
        int pipe_t = -1;
        bool abl = false;            // --ablate: the projecting conversation's layers carry its launches
        // the route kernel's per-request arguments (a captured graph holds them): GEN promote_min= (-1: not named)
        // and miss_skip_w= / miss_skip_from= (-1: not set) - a request that names one gets graphs of its own
        int promote_min = -1;
        float miss_w = -1.0f;
        int miss_from = -1;
        bool operator==(const Key& b) const {
            return cpu_plan == b.cpu_plan && residual == b.residual && other == b.other &&
                   all_resident == b.all_resident && pipe_t == b.pipe_t && abl == b.abl &&
                   promote_min == b.promote_min && miss_w == b.miss_w && miss_from == b.miss_from;
        }
    };

    struct GraphId {
        int layer = 0;
        unsigned long long cpu_plan = 0;
        const float* residual = nullptr;
        const float* other = nullptr;
        bool all_resident = false;
        int pipe_t = -1;
        bool abl = false;
        int promote_min = -1;
        float miss_w = -1.0f;
        int miss_from = -1;

        bool operator<(const GraphId& b) const {
            if (layer != b.layer) return layer < b.layer;
            if (abl != b.abl) return b.abl;
            if (pipe_t != b.pipe_t) return pipe_t < b.pipe_t;
            if (cpu_plan != b.cpu_plan) return cpu_plan < b.cpu_plan;
            if (residual != b.residual) return residual < b.residual;
            if (other != b.other) return other < b.other;
            if (promote_min != b.promote_min) return promote_min < b.promote_min;
            if (miss_w != b.miss_w) return miss_w < b.miss_w;
            if (miss_from != b.miss_from) return miss_from < b.miss_from;
            return all_resident < b.all_resident;
        }
    };

    // The body retains direct-enqueue request accounting. Capture invokes it once;
    // replay does not invoke host code and must account for the requests itself.
    // A capture that fails executes nothing on the device and leaves `expected` as it was: capture_failed is then
    // set, and the caller may run the body directly instead (a replay that fails is an error, not a capture failure).
    bool enqueue(int layer, Key key, cudaStream_t stream, uint64_t& expected, uint64_t requests,
                 const std::function<bool()>& body, bool& replayed, std::string& err) {
        replayed = false;
        const GraphId gid{layer, key.cpu_plan, key.residual, key.other, key.all_resident, key.pipe_t, key.abl,
                          key.promote_min, key.miss_w, key.miss_from};
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
        if (!entry.graph.begin(stream, err)) {
            (void) cudaGetLastError();
            capture_failed = true;
            return false;
        }
        const bool ok = body();
        const cudaError_t launch_error = cudaGetLastError();
        if (!ok || launch_error != cudaSuccess || expected != before + requests) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            (void) cudaGetLastError();
            expected = before;   // none of the captured work was executed
            if (err.empty()) err = launch_error != cudaSuccess
                ? std::string("GLM graph capture: ") + cudaGetErrorString(launch_error)
                : "GLM graph capture: body/request accounting failed";
            capture_failed = true;
            return false;
        }
        if (!entry.graph.end(stream, err)) {
            (void) cudaGetLastError();
            expected = before;
            capture_failed = true;
            return false;
        }
        if (!entry.graph.launch(stream, err)) {
            expected = before;
            return false;
        }
        entries_.emplace(gid, std::move(entry));
        ++captures;
        return true;
    }

    void clear() { entries_.clear(); }
    uint64_t captures = 0, replays = 0, invalidations = 0;
    bool capture_failed = false;   // set by a failed capture (see enqueue); the caller clears it

private:
    struct Entry {
        Key key;
        CapturedGraph graph;
    };
    std::map<GraphId, Entry> entries_;
};

}  // namespace maya::core::glmfast
