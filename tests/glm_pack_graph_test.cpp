// Actual-pack logits and recurrent snapshot parity; not a speed benchmark.
#include "strata/core/glm_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void graph_mode(bool on) {
#ifdef _WIN32
    _putenv_s("MAYA_GLM_KDA_GRAPH", on ? "1" : "0");
#else
    setenv("MAYA_GLM_KDA_GRAPH", on ? "1" : "0", 1);
#endif
}
static void require(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "glm_pack_graph_test: %s\n", msg.c_str()); std::exit(1); }
}
int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: glm_pack_graph_test PACK CAPACITY\n"); return 2; }
    strata::core::Glm5Model model;
    std::string err;
    graph_mode(false);
    require(model.load_pack_env(argv[1], std::atoll(argv[2]), err), err);
    require(model.fast() && model.n_parts_ == 1, "requires a single-device fast pack");
    require(!model.has_mtp(), "MTP must be absent: the graph path deliberately bypasses MTP");
    require(model.set_pcie_share(1.0), "CPU lane unavailable");
    const size_t V = (size_t) model.geometry().n_vocab;
    std::vector<float> logits, direct;
    for (int t = 0; t < 16; ++t) require(model.forward({1 + t}, logits, err), err);
    require(model.snapshot_save(), "snapshot save failed");
    for (int t = 0; t < 48; ++t) {
        require(model.forward({31 + t % 29}, logits, err), err);
        direct.insert(direct.end(), logits.begin(), logits.end());
    }
    uint64_t comparisons = 0;
    // Qualify the ordinary snapshot control before attributing a difference to graphs.
    require(model.snapshot_restore() && model.position() == 16, "control snapshot restore failed");
    for (int t = 0; t < 48; ++t) {
        require(model.forward({31 + t % 29}, logits, err), err);
        require(logits.size() == V && std::memcmp(logits.data(), direct.data() + (size_t) t * V, V * 4) == 0,
                "ordinary snapshot control is unstable; graph parity not scored");
    }
    for (int round = 0; round < 3; ++round) {
        require(model.snapshot_restore() && model.position() == 16, "snapshot restore failed");
        graph_mode(true);
        for (int t = 0; t < 48; ++t) {
            require(model.forward({31 + t % 29}, logits, err), err);
            require(logits.size() == V && std::memcmp(logits.data(), direct.data() + (size_t) t * V, V * 4) == 0,
                    "GPU-only graph/direct logits differ at round/step " + std::to_string(round) + "/" + std::to_string(t));
            for (float x : logits) require(std::isfinite(x), "nonfinite logits");
            ++comparisons;
        }
    }
    // Exercise real request accounting and graph invalidation at completed token
    // boundaries. CPU-vs-GPU dots have different rounding; this is a liveness and
    // finite-output check, not a numerical-equivalence claim for mixed tiers.
    uint64_t mixed_steps = 0;
    for (double share : {0.80, 1.0, 0.65, 1.0}) {
        require(model.set_pcie_share(share), "CPU plan change failed");
        for (int t = 0; t < 12; ++t) {
            require(model.forward({701 + (int) mixed_steps}, logits, err), err);
            for (float x : logits) require(std::isfinite(x), "mixed-plan nonfinite logits");
            ++mixed_steps;
        }
    }
    graph_mode(false);
    model.reset();
    require(model.set_pcie_share(1.0), "restore plan failed");
    for (int t = 0; t < 16; ++t) require(model.forward({1 + t}, logits, err), err);
    require(model.snapshot_save(), "reset control snapshot save failed");
    for (int t = 0; t < 48; ++t) {
        require(model.forward({31 + t % 29}, logits, err), err);
        require(logits.size() == V && std::memcmp(logits.data(), direct.data() + (size_t) t * V, V * 4) == 0,
                "ordinary reset control is unstable; graph parity not scored");
    }
    require(model.snapshot_restore() && model.position() == 16, "reset control snapshot restore failed");
    graph_mode(true);
    for (int t = 0; t < 48; ++t) {
        require(model.forward({31 + t % 29}, logits, err), err);
        require(std::memcmp(logits.data(), direct.data() + (size_t) t * V, V * 4) == 0,
                "reset/residual graph parity differs at step " + std::to_string(t));
        ++comparisons;
    }
    std::printf("glm_pack_graph_test: PASS (%llu actual-pack full-vocabulary bitwise comparisons, snapshots/reset; %llu mixed-plan finite/liveness steps)\n",
                (unsigned long long) comparisons, (unsigned long long) mixed_steps);
    return 0;
}
