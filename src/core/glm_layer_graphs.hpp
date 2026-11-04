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
        bool operator==(const Key& b) const {
            return cpu_plan == b.cpu_plan && residual == b.residual && other == b.other &&
                   all_resident == b.all_resident;
        }
    };

    // The body retains direct-enqueue request accounting. Capture invokes it once;
    // replay does not invoke host code and must account for the requests itself.
    bool enqueue(int layer, Key key, cudaStream_t stream, uint64_t& expected, uint64_t requests,
                 const std::function<bool()>& body, bool& replayed, std::string& err) {
        replayed = false;
        auto it = entries_.find(layer);
        if (it != entries_.end() && !(it->second.key == key)) {
            entries_.erase(it);   // caller changes plans only at a completed token boundary
            it = entries_.end();
            ++invalidations;
        }
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
        entries_.emplace(layer, std::move(entry));
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
    std::map<int, Entry> entries_;
};

}  // namespace strata::core::glmfast
