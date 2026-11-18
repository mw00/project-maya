#include "strata/core/glm_model.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace sc = strata::core::statecache;
void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: glm_state_cache_test PACK\n"); return 2; }
    try {
        strata::core::Glm5Model model;
        std::string error;
        need(model.load_pack_env(argv[1], 256, error), error);
        need(model.fast(), "native fast path required");
        model.set_pcie_share(1.0);  // no CPU pool is already entirely GPU arithmetic
        std::vector<float> logits, expected;
        sc::Prefix prefix{"fixture", {}, {}};
        for (int32_t t = 1; t <= 12; ++t) { need(model.forward({t}, logits, error), error); prefix.tokens.push_back(t); }
        need(model.snapshot_save(), "save native endpoint");
        sc::Capture capture;
        need(model.state_capture(capture, prefix, -1, error), error);
        need(model.forward({13}, expected, error), error);
        const auto directory = std::filesystem::temp_directory_path() / ("maya-glm-state-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        sc::Config config; config.directory = directory.string();
        config.ram_bytes = 128 << 20; config.disk_bytes = 128 << 20;
        {
            sc::StateCache cache(config);
            need(cache.publish(std::move(capture), error), error);
            cache.flush();
            need(cache.stats().write_failures == 0, "storage write completed");
        }
        // An unrelated request overwrites every attention cell, recurrence and
        // pool. Restore after restart must reconstitute all of them independently.
        model.reset();
        for (int32_t t = 31; t <= 54; ++t) need(model.forward({t}, logits, error), error);
        {
            sc::StateCache cache(config);
            need(!cache.find_resume(prefix), "full prompt leaves a token for fresh logits");
            auto prompt = prefix; prompt.tokens.push_back(13);
            auto lease = cache.find_resume(prompt);
            std::vector<sc::Region> regions;
            need(lease && cache.restore(lease, regions, error), error);
            need(model.state_restore(lease.prefix(), regions, error) && model.position() == 12, error);
            cache.release(lease);
            need(model.forward({13}, logits, error), error);
            need(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * 4) == 0,
                 "full-vocabulary logits changed after unrelated request and SSD restore");
            for (float value : logits) need(std::isfinite(value), "non-finite logits");
        }
        std::filesystem::remove_all(directory);
        // Collect an interior checkpoint without cutting this prefill call.
        model.reset();
        std::vector<int32_t> first, second;
        for (int32_t token = 1; token <= 32; ++token) first.push_back(token);
        for (int32_t token = 33; token <= 64; ++token) second.push_back(token);
        need(model.prefill(first, error, 33), "initialize prefill: " + error);
        need(model.prefill(second, error, 65), "baseline prefill: " + error);
        need(model.forward({65}, expected, error), error);
        model.reset();
        need(model.prefill(first, error, 33), "initialize prefill: " + error);
        need(model.prefill_export_prepare(0, 39, error), error);
        need(model.prefill(second, error, 65), "exporting prefill: " + error);
        need(model.forward({65}, logits, error), error);
        need(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * 4) == 0,
             "interior export changed full-call target logits");
        need(model.ckpt_save_at(0, 39, error), error);
        prefix.tokens.clear();
        for (int32_t token = 1; token <= 39; ++token) prefix.tokens.push_back(token);
        need(model.state_capture(capture, prefix, 0, error), error);
        need(model.state_restore(prefix, capture.regions, error) && model.position() == 39, error);
        need(model.forward({40}, logits, error), error);
        for (float value : logits) need(std::isfinite(value), "restored interior state produced non-finite logits");
        if (model.has_mtp()) {
            // At an aligned trunk endpoint, NextN still needs the preceding
            // pool's rows: its context ends at n-1, not n.
            constexpr int64_t mark = 40;
            // Equal microbatch shapes keep the oracle independent of CUDA's
            // numerical differences between eight-row and sixteen-row products.
            model.prefill_force_chunk = 8;
            model.reset();
            need(model.prefill(first, error, 33), error);
            need(model.prefill_export_prepare(0, mark, error), error);
            need(model.prefill(second, error, 65), error);
            need(model.ckpt_save_at(0, mark, error), error);
            prefix.tokens.resize(size_t(mark));
            for (int32_t i = 0; i < mark; ++i) prefix.tokens[size_t(i)] = i+1;
            sc::Capture exported, exact, restored;
            need(model.state_capture(exported, prefix, 0, error), error);
            model.reset();
            need(model.prefill(first, error, 33), error);
            const std::vector<int32_t> tail(second.begin(), second.begin()+8);
            need(model.prefill(tail, error, 41), error);
            need(model.snapshot_save(), "exact aligned endpoint snapshot");
            need(model.state_capture(exact, prefix, -1, error), error);
            need(model.state_restore(prefix, exported.regions, error), error);
            need(model.snapshot_save(), "restored aligned endpoint snapshot");
            need(model.state_capture(restored, prefix, -1, error), error);
            const auto pool = [](const sc::Capture& state) -> const sc::Bytes& {
                for (const auto& region : state.regions) if (region.name == "part-0/open-pool") return region.bytes;
                throw std::runtime_error("NextN open pool absent");
            };
            const auto& expected_pool = pool(exact);
            const auto& restored_pool = pool(restored);
            const size_t count = size_t(model.geometry().idx_kpool-1)*model.geometry().idx_key*2;
            need(expected_pool.size() >= count*4 && restored_pool.size() >= count*4, "NextN pool shape");
            size_t written = 0, actual_written = 0;
            float max_difference = 0;
            for (size_t i = 0; i < count; ++i) {
                float expected_value, actual_value;
                std::memcpy(&expected_value, expected_pool.data()+i*4, 4);
                std::memcpy(&actual_value, restored_pool.data()+i*4, 4);
                written += expected_value != 0;
                need(std::isfinite(actual_value), "aligned interior NextN pool is non-finite");
                actual_written += actual_value != 0;
                max_difference = std::max(max_difference, std::abs(actual_value-expected_value));
            }
            std::printf("NextN pool: expected nonzero %zu, restored nonzero %zu, max difference %.9g\n", written, actual_written, max_difference);
            need(written > 0, "NextN oracle did not produce any pool rows");
            need(max_difference < 1e-5f, "aligned interior export changed NextN open-pool state");
            model.prefill_force_chunk = 0;
            std::puts("GLM NextN: aligned interior checkpoint matches exact-endpoint restore");
        }
        std::puts("GLM state cache: intervening request, SSD restart, identical-prompt logits and interior restore passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
