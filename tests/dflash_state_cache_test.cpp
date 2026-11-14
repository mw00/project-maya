#include "strata/core/glm_dflash.hpp"
#include "strata/core/glm_model.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace sc = strata::core::statecache;
void need(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::map<std::string, std::string> digests(const std::vector<sc::RegionSource>& sources) {
    std::map<std::string, std::string> result;
    for (const auto& source : sources) {
        sc::Bytes bytes(size_t(source.bytes)); std::string error;
        need(source.read(0, bytes.data(), bytes.size(), error), error);
        result[source.name] = sc::content_digest(bytes.data(), bytes.size());
    }
    return result;
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: dflash_state_cache_test NATIVE_SMALL_DFLASH_GGUF\n"); return 2; }
    try {
        strata::core::GlmDflash draft; std::string error;
        constexpr int E = 1024, V = 512;
        need(draft.load(argv[1], 0, error) && draft.bind(E, V, 2, error), error);
        const int64_t n = draft.ring_size()+17;
        const auto stage = [&](int seed) {
            for (int64_t p = 0; p < n; ++p) for (size_t k = 0; k < draft.target_layers().size(); ++k) {
                auto* row = draft.stage_row(p, int(k));
                for (int e = 0; e < E; ++e) row[e] = float((p*17+int64_t(k)*11+e*7+seed)%113-56)/64;
                draft.staged(p, int(k));
            }
            need(draft.inject(0, n, draft.stream(), error), error);
            need(cudaDeviceSynchronize() == cudaSuccess, "DFlash producer failed");
        };
        stage(0);
        const auto dir = std::filesystem::temp_directory_path()/
            ("maya-dflash-state-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        sc::Config config; config.ram_bytes = 1024; config.disk_bytes = 1 << 20; config.directory = dir.string();
        sc::Prefix prefix{"dflash-fixture", std::vector<int32_t>(size_t(n), 1), {}};
        std::map<std::string, std::string> expected;
        {
            sc::StateCache cache(config);
            auto transfer = sc::memory_transfer(); std::vector<sc::RegionSource> sources;
            need(draft.state_sources(n, transfer, sources, error), error);
            expected = digests(sources);
            sc::Capture capture; capture.prefix = prefix;
            need(cache.publish(std::move(capture), sources, error), error); cache.flush();
            need(cache.stats().ram_bytes == 0, "DFlash unit must spill beyond the RAM budget");
        }
        draft.reset(); stage(41);
        {
            sc::StateCache cache(config); auto prompt = prefix; prompt.tokens.push_back(2);
            auto lease = cache.find_resume(prompt); need(bool(lease), "DFlash SSD restart missed");
            {
                auto transfer = sc::memory_transfer(); std::vector<sc::RegionTarget> targets;
                need(draft.state_targets(n, cache, lease, transfer, targets, error), error);
                need(cache.restore(lease, targets, error), error);
            }
            auto transfer = sc::memory_transfer(); std::vector<sc::RegionSource> sources;
            need(draft.state_sources(n, transfer, sources, error), error);
            need(digests(sources) == expected, "written DFlash keys, values or position tags changed");
            cache.release(lease);
        }
        std::filesystem::remove_all(dir);
        std::puts("DFlash state: wrapped ring, overwritten context, bounded SSD capture and exact restore passed");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
