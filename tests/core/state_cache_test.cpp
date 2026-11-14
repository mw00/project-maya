#include "strata/core/state_cache.hpp"

#include <chrono>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstring>
#include <thread>

namespace sc = strata::core::statecache;
using sc::Bytes;

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
sc::Prefix prefix(std::initializer_list<int32_t> tokens, std::string identity = "model-a") {
    return {std::move(identity), tokens, {}};
}
sc::Capture capture(sc::Prefix p, Bytes bytes, bool ready = false) {
    return {std::move(p), {{"kv", std::move(bytes)}}, ready, false};
}
sc::Config config() {
    sc::Config c; c.chunk_bytes = 4; c.ram_bytes = 128; c.disk_bytes = 128;
    c.leaf_last = false; return c;
}
sc::Capture metadata(sc::Prefix p) { sc::Capture m; m.prefix = std::move(p); return m; }
std::filesystem::path scratch(const char* name) {
    return std::filesystem::temp_directory_path() /
           (std::string(name) + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

// A native region over test bytes that counts what the cache reads from it.
struct Native {
    Bytes bytes;
    size_t read = 0;
    sc::RegionSource source(std::string name, bool rows, bool prints, size_t chunk) {
        sc::RegionSource s;
        s.name = std::move(name); s.bytes = bytes.size(); s.rows = rows;
        s.read = [this](uint64_t at, uint8_t* out, size_t n, std::string& e) {
            if (at > bytes.size() || n > bytes.size() - at) { e = "test read out of range"; return false; }
            read += n; std::memcpy(out, bytes.data() + at, n); return true;
        };
        if (prints) s.fingerprints = [this, chunk](size_t step, std::vector<uint64_t>& out, std::string& e) {
            if (step != chunk) { e = "unexpected fingerprint chunk"; return false; }
            out.clear();
            for (size_t at = 0; at < bytes.size(); at += step)
                out.push_back(sc::fingerprint(bytes.data() + at, std::min(step, bytes.size() - at)));
            return true;
        };
        return s;
    }
};
Bytes restore(sc::StateCache& cache, const sc::Prefix& prompt, const std::string& name, std::string& error) {
    auto lease = cache.find_resume(prompt);
    std::vector<sc::Region> out;
    Bytes result;
    if (lease && cache.restore(lease, out, error))
        for (const auto& r : out) if (r.name == name) result = r.bytes;
    cache.release(lease);
    return result;
}

class FailingStorage final : public sc::Storage {
public:
    std::atomic<int> writes{0};
    bool put(const std::string&, const Bytes&, std::string&) override { ++writes; throw std::runtime_error("disk full"); }
    bool get(const std::string&, Bytes&, std::string&) override { return false; }
    bool erase(const std::string&, std::string&) override { return true; }
    std::vector<std::string> manifests(std::string&) override { return {}; }
};

int main() {
    try {
        std::string error;
        check(sc::content_digest("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA256");
        check(sc::content_digest("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA256 block padding");
        const std::string digest_fixture(1000, 'a');
        check(sc::content_digest(digest_fixture.data(), digest_fixture.size()) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3", "SHA256 multiple blocks");
        {
            Bytes a(37); for (size_t i = 0; i < a.size(); ++i) a[i] = uint8_t(i * 7);
            const auto base = sc::fingerprint(a.data(), a.size());
            check(base == sc::fingerprint(a.data(), a.size()), "fingerprint is a function of the bytes");
            check(sc::fingerprint(a.data(), 36) != base && sc::fingerprint(a.data(), 32) != sc::fingerprint(a.data(), 33),
                  "fingerprint covers the length and a partial last word");
            for (size_t i = 0; i < a.size(); ++i) {
                auto b = a; b[i] ^= 1;
                check(sc::fingerprint(b.data(), b.size()) != base, "one changed byte changes the fingerprint");
            }
            auto swapped = a; std::swap_ranges(swapped.begin(), swapped.begin() + 8, swapped.begin() + 8);
            check(sc::fingerprint(swapped.data(), swapped.size()) != base, "fingerprint depends on word order");
            check(base == 0x29c143108d934c7full && sc::fingerprint(nullptr, 0) == 0,
                  "fingerprint golden values (tools: an independent Python version of the same function)");
        }
        {
            sc::StateCache cache(config());
            check(cache.publish(capture(prefix({1, 2}), {1, 2, 3, 4}), error), "publish prefix");
            check(cache.publish(capture(prefix({1, 2}), {1, 2, 3, 4}), error) && cache.stats().units == 1,
                  "duplicate complete publication keeps one unit");
            check(!cache.find_resume(prefix({1, 2})), "identical prompt must leave a token for logits");
            auto lease = cache.find_resume(prefix({1, 2, 3}));
            check(lease && lease.tokens() == 2, "restore deepest complete prefix");
            std::vector<sc::Region> restored;
            check(cache.restore(lease, restored, error) && restored.size() == 1 && restored[0].bytes == Bytes({1, 2, 3, 4}), "restore real payload");
            cache.release(lease);
            check(!cache.find_resume(prefix({1, 9, 3})), "different prefix never hits");
            check(!cache.find_resume(prefix({1, 2, 3}, "model-b")), "different model never hits");
            check(cache.publish(capture(prefix({1, 2}), {9, 8, 7, 6}, true), error), "publish logits-ready endpoint");
            check(bool(cache.find_resume(prefix({1, 2}))), "full prompt may hit with explicitly ready logits");
        }
        {
            sc::StateCache cache(config());
            auto p = prefix({1, 2}); p.extra = {{0, 1, "image-a"}};
            check(cache.publish(capture(p, {1, 2, 3, 4}), error), "publish multimodal state");
            auto query = p; query.tokens.push_back(3);
            check(bool(cache.find_resume(query)), "same image matches");
            query.extra[0].identity = "image-b";
            check(!cache.find_resume(query), "different image refuses reused tokens");
        }
        for (const bool prints : {false, true}) {
            // A child reuses its parent's append-only rows below the parent's end; everything else is copied.
            sc::StateCache cache(config());
            Native kv{{1, 2, 3, 4, 5, 6, 7, 8}}, rec{{9, 9, 9, 9}};
            check(cache.publish(metadata(prefix({1, 2})), {kv.source("kv", true, prints, 4), rec.source("rec", false, prints, 4)},
                                error), "publish parent rows and recurrent state");
            kv.bytes.insert(kv.bytes.end(), {10, 11, 12, 13}); rec.bytes = {7, 7, 7, 7};
            kv.read = rec.read = 0;
            check(cache.publish(metadata(prefix({1, 2, 3})), {kv.source("kv", true, prints, 4), rec.source("rec", false, prints, 4)},
                                error), "publish child");
            check(kv.read == (prints ? 4u : 12u) && rec.read == 4,
                  prints ? "fingerprinted reuse reads only the rows the parent lacks"
                         : "compared reuse reads the shared rows once and copies only new rows");
            const auto stats = cache.stats();
            check(stats.reused_bytes == 8 && stats.ram_bytes == 20 && stats.logical_bytes == 28 && stats.shared_bytes == 8,
                  "reused rows consume physical bytes once");
            check(restore(cache, prefix({1, 2, 3, 4}), "kv", error) == kv.bytes &&
                  restore(cache, prefix({1, 2, 3, 4}), "rec", error) == rec.bytes, "child restores its own bytes");
            check(restore(cache, prefix({1, 2, 9}), "kv", error) == Bytes({1, 2, 3, 4, 5, 6, 7, 8}) &&
                  restore(cache, prefix({1, 2, 9}), "rec", error) == Bytes({9, 9, 9, 9}), "parent keeps its bytes");
        }
        for (const bool prints : {false, true}) {
            // Rows computed apart from the parent (another context, a recomputation) are never substituted.
            sc::StateCache cache(config());
            Native kv{{1, 2, 3, 4, 5, 6, 7, 8}};
            check(cache.publish(metadata(prefix({1, 2})), {kv.source("kv", true, prints, 4)}, error), "publish parent");
            kv.bytes = {1, 2, 3, 5, 5, 6, 7, 8, 10, 11};
            check(cache.publish(metadata(prefix({1, 2, 3})), {kv.source("kv", true, prints, 4)}, error), "publish child");
            check(cache.stats().reused_bytes == 4, "only the matching chunk is reused");
            check(restore(cache, prefix({1, 2, 3, 4}), "kv", error) == kv.bytes, "a changed row is copied, not reused");
            check(restore(cache, prefix({1, 2, 9}), "kv", error) == Bytes({1, 2, 3, 4, 5, 6, 7, 8}), "parent unaffected");
        }
        {
            sc::StateCache cache(config());
            Native kv{{1, 2, 3, 4, 5, 6}};
            check(cache.publish(metadata(prefix({1, 2})), {kv.source("kv", true, true, 4)}, error), "parent with a partial chunk");
            kv.bytes.insert(kv.bytes.end(), {7, 8, 9, 10}); kv.read = 0;
            check(cache.publish(metadata(prefix({1, 2, 3})), {kv.source("kv", true, true, 4)}, error) && kv.read == 6 &&
                  cache.stats().reused_bytes == 4, "the parent's partial last chunk is copied again, not reused");
            check(restore(cache, prefix({1, 2, 3, 4}), "kv", error) == kv.bytes, "child with a regrown chunk restores exactly");
            Native rec{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
            rec.read = 0;
            check(cache.publish(metadata(prefix({1, 2, 3, 5})), {rec.source("kv", false, true, 4)}, error) && rec.read == 10 &&
                  cache.stats().reused_bytes == 4, "a region not marked as rows is never reused");
        }
        {
            auto c = config(); c.ram_bytes = 8; c.chunk_bytes = 8;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({1}), Bytes(8, 1)), error), "publish cheap state");
            check(cache.publish(capture(prefix({2, 3, 4, 5}), Bytes(8, 2)), error), "admit higher cost density");
            check(!cache.find_resume(prefix({1, 9})), "evict lower density state");
            Native low{Bytes(8, 4)};
            check(!cache.publish(metadata(prefix({5})), {low.source("kv", false, false, 8)}, error) && low.read == 0,
                  "a state the policy would evict first is rejected before any copy");
            auto lease = cache.find_resume(prefix({2, 3, 4, 5, 9}));
            check(bool(lease), "higher density retained");
            check(!cache.publish(capture(prefix({7}), Bytes(8, 3)), error), "active reader pins its entire recovery unit");
            std::vector<sc::Region> out;
            check(cache.restore(lease, out, error) && out[0].bytes == Bytes(8, 2), "pinned payload remains intact");
            cache.release(lease);
            check(cache.stats().ram_bytes <= 8 && std::isfinite(cache.stats().watermark), "bounded bytes and finite clock");
            Native again{Bytes(8, 2)};
            check(cache.publish(metadata(prefix({2, 3, 4, 5})), {again.source("kv", false, false, 8)}, error) && again.read == 0,
                  "publishing a held prefix reads nothing");
        }
        {
            const auto dir = scratch("maya-state-test-");
            auto c = config(); c.directory = dir.string();
            {
                sc::StateCache cache(c);
                check(cache.publish(metadata(prefix({4, 5})), {sc::bytes_source("kv", {8,7,6,5,4,3,2,1})}, error),
                      "publish persistent streamed state");
                cache.flush();
                check(cache.stats().disk_bytes == 8 && cache.stats().write_failures == 0, "write-through finished");
                bool locked = false;
                try { sc::StateCache other(c); } catch (const std::exception&) { locked = true; }
                check(locked, "a second engine cannot race local recovery and orphan cleanup");
            }
            std::ofstream(dir / "chunk-interrupted") << "partial";
            std::ofstream(dir / "unit-interrupted.state.tmp") << "partial";
            {
                sc::StateCache cache(c);
                check(!std::filesystem::exists(dir / "chunk-interrupted") &&
                      !std::filesystem::exists(dir / "unit-interrupted.state.tmp"), "interrupted publication orphans removed at restart");
                auto lease = cache.find_resume(prefix({4, 5, 6}));
                Bytes native(8);
                sc::RegionTarget target{"kv", native.size(), [&](uint64_t at, const uint8_t* data, size_t n, std::string&) {
                    std::memcpy(native.data()+at, data, n); return true;
                }};
                check(cache.restore(lease, {target}, error) && native == Bytes({8,7,6,5,4,3,2,1}) && cache.stats().promoted == 1,
                      "bounded native restore promotes a fitting SSD unit back to RAM");
                std::vector<sc::Region> out;
                check(lease && cache.restore(lease, out, error) && out[0].bytes == Bytes({8, 7, 6, 5, 4, 3, 2, 1}), "restart restores a complete state");
                cache.release(lease);
                for (const auto& f : std::filesystem::directory_iterator(dir)) {
                    if (f.path().filename().string().starts_with("chunk-")) { std::ofstream(f.path(), std::ios::binary) << "bad!"; break; }
                }
                lease = cache.find_resume(prefix({4, 5, 6}));
                check(cache.restore(lease, out, error), "resident chunks remain reusable without a disk read");
                cache.release(lease);
            }
            {
                sc::StateCache cache(c);
                auto lease = cache.find_resume(prefix({4, 5, 6}));
                std::vector<sc::Region> out;
                check(!cache.restore(lease, out, error) && out.empty(), "corrupt chunk fails whole load without partial payload");
                cache.release(lease);
                check(!cache.find_resume(prefix({4, 5, 6})), "corrupt unit is not reusable");
            }
            std::filesystem::remove_all(dir);
        }
        {
            // Chunks a child shares with its parent are stored once and survive a restart for both.
            const auto dir = scratch("maya-state-shared-");
            auto c = config(); c.directory = dir.string();
            Native kv{{1, 2, 3, 4, 5, 6, 7, 8}};
            {
                sc::StateCache cache(c);
                check(cache.publish(metadata(prefix({1, 2})), {kv.source("kv", true, true, 4)}, error), "persist parent");
                kv.bytes.insert(kv.bytes.end(), {9, 10, 11, 12});
                check(cache.publish(metadata(prefix({1, 2, 3})), {kv.source("kv", true, true, 4)}, error), "persist child");
                cache.flush();
                check(cache.stats().disk_bytes == 12 && cache.stats().write_failures == 0, "shared chunks written once");
            }
            {
                sc::StateCache cache(c);
                check(cache.stats().units == 2 && cache.stats().disk_bytes == 12, "both states recovered with shared chunks");
                check(restore(cache, prefix({1, 2, 3, 4}), "kv", error) == kv.bytes &&
                      restore(cache, prefix({1, 2, 9}), "kv", error) == Bytes({1, 2, 3, 4, 5, 6, 7, 8}),
                      "recovered states restore exactly");
            }
            std::filesystem::remove_all(dir);
        }
        {
            // A state bigger than RAM that reuses a RAM-only ancestor's rows, where admission evicts that ancestor.
            const auto dir = scratch("maya-state-evicted-parent-");
            auto c = config(); c.directory = dir.string(); c.ram_bytes = 8; c.max_units = 1; c.leaf_last = true;
            c.write_through = false;
            {
                sc::StateCache cache(c);
                Native kv{{1, 2, 3, 4, 5, 6, 7, 8}};
                check(cache.publish(metadata(prefix({1, 2})), {kv.source("kv", true, false, 4)}, error), "RAM-only parent");
                kv.bytes.insert(kv.bytes.end(), {9, 10, 11, 12, 13, 14, 15, 16});
                const bool kept = cache.publish(metadata(prefix({1, 2, 3})), {kv.source("kv", true, false, 4)}, error);
                check(cache.stats().units <= 1, "the state limit holds");
                if (kept) check(restore(cache, prefix({1, 2, 3, 4}), "kv", error) == kv.bytes, "a kept child restores exactly");
                else check(!cache.find_resume(prefix({1, 2, 3, 4})) || bool(cache.find_resume(prefix({1, 2, 9}))),
                           "a refused child leaves no partial state");
                Native other{{7, 7, 7, 7}};
                check(cache.publish(metadata(prefix({5})), {other.source("kv", false, false, 4)}, error) &&
                      restore(cache, prefix({5, 6}), "kv", error) == other.bytes, "the cache stays usable");
            }
            std::filesystem::remove_all(dir);
        }
        {
            auto storage = std::make_shared<FailingStorage>();
            sc::StateCache cache(config(), storage);
            check(cache.publish(metadata(prefix({7})), {sc::bytes_source("kv", {1,2,3,4})}, error),
                  "streamed RAM admission before asynchronous storage failure");
            cache.flush();
            check(storage->writes > 0 && cache.stats().write_failures > 0, "worker reports exceptions and completes its job");
            check(restore(cache, prefix({7, 8}), "kv", error) == Bytes({1, 2, 3, 4}), "storage failure preserves complete RAM state");
        }
        {
            sc::StateCache cache(config());
            cache.observe_prefill(0, 100, 3);
            cache.observe_prefill(100, 200, 7);
            const auto stats = cache.stats();
            check(std::abs(stats.cost_per_token - .01) < 1e-12 && std::abs(stats.cost_per_token_per_100k - 40) < 1e-8,
                  "runtime calibration fits actual linear and attention terms");
            auto state = capture(prefix({1}), {1, 2, 3, 4});
            state.min_prompt_tokens = 4;
            check(cache.publish(std::move(state), error), "publish window-qualified checkpoint");
            check(!cache.find_resume(prefix({1, 2, 3})), "shorter prompt cannot reuse missing draft context");
            auto lease = cache.find_resume(prefix({1, 2, 3, 4}));
            check(bool(lease), "qualified prompt matches");
            cache.invalidate(prefix({1, 2, 3}));
            std::vector<sc::Region> out;
            check(!cache.restore(lease, out, error), "invalidated unit cannot be read through an old lease");
            check(cache.stats().units == 1, "invalidated reader pins storage through completion");
            auto again = capture(prefix({1}), {1, 2, 3, 4}); again.min_prompt_tokens = 4;
            check(!cache.publish(std::move(again), error) && cache.stats().units == 1,
                  "a duplicate cannot revive an invalidated pinned unit");
            cache.release(lease);
            check(cache.stats().units == 0 && cache.stats().ram_bytes == 0, "dead unit and its chunks reclaimed on release");
        }
        {
            const std::vector<uint64_t> marks = {8, 16};
            auto p = sc::plan_export(0, 24, marks, 8, true);
            check(p.end == 16 && p.checkpoint == 8, "one in-chunk export before next required mark");
            p = sc::plan_export(0, 24, marks, 16, true);
            check(p.end == 8 && p.checkpoint == 0, "unsupported offset cuts exactly");
            p = sc::plan_export(0, 24, marks, 1, false);
            check(p.end == 8 && p.checkpoint == 0, "backend fallback is explicit");
            uint64_t endpoint = 0;
            check(sc::Commit{10, 8, 3, 3, true}.exact(endpoint) && endpoint == 13, "accepted state endpoint");
            check(!sc::Commit{10, 8, 3, 2, true}.exact(endpoint), "stop inside accepted window must resolve before publication");
            check(!sc::Commit{10, 8, 3, 3, false}.exact(endpoint), "in-flight state is not exact");
        }
        {
            auto c = config(); c.supersede_lag = 1;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({1,2}), Bytes(4,1)), error), "interior fallback");
            for (int n = 3; n <= 5; ++n) {
                auto state = capture(prefix({1,2,3,4,5}), Bytes(4, uint8_t(n)));
                state.prefix.tokens.resize(size_t(n)); state.frontier = true;
                check(cache.publish(std::move(state), error), "complete frontier chain");
            }
            auto lease = cache.find_resume(prefix({1,2,3,9}));
            check(lease && lease.tokens() == 2, "supersession retires old frontier and preserves interior fallback");
            cache.release(lease);
            lease = cache.find_resume(prefix({1,2,3,4,9}));
            check(lease && lease.tokens() == 4, "lag retains previous complete answer for rewrites");
            cache.release(lease);
        }
        {
            auto c = config(); c.supersede_lag = 0;
            sc::StateCache cache(c);
            auto root = capture(prefix({1,2}), Bytes(4,1)); root.frontier = true;
            check(cache.publish(std::move(root), error), "branch root");
            check(cache.publish(capture(prefix({1,2,9}), Bytes(4,2)), error), "sibling continuation");
            auto end = capture(prefix({1,2,3,4}), Bytes(4,3)); end.frontier = true;
            check(cache.publish(std::move(end), error), "other branch frontier");
            auto lease = cache.find_resume(prefix({1,2,8}));
            check(lease && lease.tokens() == 2, "shared branch point survives frontier supersession");
            cache.release(lease);
        }
        {
            auto c = config(); c.chunk_bytes = 64 << 10;
            Native kv{Bytes(64*c.chunk_bytes+17, 201)}, aux{Bytes(2*c.chunk_bytes+9, 202)};
            c.ram_bytes = kv.bytes.size()+aux.bytes.size();
            sc::StateCache cache(c);
            auto m = metadata(prefix({1,2})); m.prefix.extra = {{0,1,"large-image"}};
            m.sampling_ready = true; m.anchor = true; m.frontier = true; m.min_prompt_tokens = 4;
            check(cache.publish(m, {kv.source("kv", true, false, c.chunk_bytes), aux.source("aux", false, false, c.chunk_bytes)}, error) &&
                  kv.read == kv.bytes.size() && aux.read == aux.bytes.size(),
                  "large streamed capture reads each multi-region byte once with partial chunks");
            const auto stats = cache.stats();
            check(stats.units == 1 && stats.admitted == 1 && stats.logical_bytes == kv.bytes.size()+aux.bytes.size() &&
                  stats.ram_bytes == stats.logical_bytes, "large streamed capture accounts RAM and partial chunks exactly");
            auto prompt = m.prefix; prompt.tokens.push_back(3);
            check(!cache.find_resume(prompt), "large capture preserves minimum prompt length");
            prompt.tokens.push_back(4);
            auto lease = cache.find_resume(prompt);
            std::vector<sc::Region> out;
            check(lease && lease.sampling_ready() && cache.restore(lease, out, error) && out.size() == 2 &&
                  out[0].name == "kv" && out[0].bytes == kv.bytes && out[1].name == "aux" && out[1].bytes == aux.bytes,
                  "large capture restores exact region bytes and sampling metadata");
            cache.release(lease);
        }
        {
            auto c = config(); c.chunk_bytes = 64 << 10; c.ram_bytes = 16ull << 20;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({9}), Bytes(c.chunk_bytes, 9)), error), "seed large capture rollback");
            const auto before = cache.stats();
            const auto caller = std::this_thread::get_id();
            size_t reads = 0;
            sc::RegionSource source;
            source.name = "kv"; source.bytes = (12ull << 20)+3;
            source.read = [&](uint64_t, uint8_t* out, size_t n, std::string& e) {
                check(std::this_thread::get_id() == caller, "native reads stay on the publishing thread");
                check(n <= (8u << 20), "native reads are bounded batches");
                if (++reads == 2) { e = "injected native copy failure"; return false; }
                std::fill_n(out, n, uint8_t(1)); return true;
            };
            check(!cache.publish(metadata(prefix({1,2})), {source}, error) && reads == 2 && error == "injected native copy failure",
                  "a source failure after an earlier batch is reported");
            const auto after = cache.stats();
            check(after.units == before.units && after.ram_bytes == before.ram_bytes &&
                  after.logical_bytes == before.logical_bytes && !cache.find_resume(prefix({1,2,3})),
                  "a failed capture leaves existing units and RAM accounting intact");
            check(restore(cache, prefix({9,10}), "kv", error) == Bytes(c.chunk_bytes, 9), "a failed capture preserves existing bytes");
        }
        {
            auto c = config(); c.max_units = 2; c.leaf_last = true;
            sc::StateCache cache(c);
            check(cache.publish(capture(prefix({1}), {1,2,3,4}), error), "seed pinned unit");
            auto pinned = cache.find_resume(prefix({1,9}));
            check(cache.publish(capture(prefix({8}), {8,8,8,8}), error), "seed eviction candidate");
            Native kv{{1,2,3,4,5,6,7,8}};
            check(cache.publish(metadata(prefix({8,9,10})), {kv.source("kv", false, false, 4)}, error) && kv.read == 8 &&
                  cache.stats().units == 2, "count trimming respects a pinned unit");
            check(restore(cache, prefix({8,9,10,11}), "kv", error) == kv.bytes, "trimmed capture keeps its chunks");
            std::vector<sc::Region> out;
            check(cache.restore(pinned, out, error) && out[0].bytes == Bytes({1,2,3,4}), "pinned unit remains intact");
            cache.release(pinned);
        }
        {
            const auto dir = scratch("maya-state-stream-");
            auto c = config(); c.directory = dir.string(); c.ram_bytes = 8; c.disk_bytes = 128;
            Native kv{Bytes(64)};
            for (size_t i = 0; i < kv.bytes.size(); ++i) kv.bytes[i] = uint8_t(i);
            {
                sc::StateCache cache(c);
                check(cache.publish(metadata(prefix({1,2})), {kv.source("kv", true, false, 4)}, error), "capture larger than RAM directly into SSD");
                cache.flush();
                check(kv.read == kv.bytes.size(), "a capture larger than RAM reads each byte once");
                check(cache.stats().ram_bytes == 0 && cache.stats().disk_bytes == kv.bytes.size(), "large state never materializes in RAM tier");
            }
            {
                sc::StateCache cache(c);
                auto lease = cache.find_resume(prefix({1,2,3}));
                check(bool(lease), "large persisted unit recovered after restart");
                Bytes restored(kv.bytes.size(), 0);
                size_t writes = 0;
                sc::RegionTarget target{"kv", restored.size(), [&](uint64_t at, const uint8_t* data, size_t n, std::string&) {
                    check(at+n <= restored.size(), "restore stays in bounds");
                    ++writes; std::memcpy(restored.data()+at, data, n); return true;
                }};
                auto broken = target; ++broken.bytes;
                check(!cache.restore(lease, {broken}, error) && writes == 0, "large restore validates all layouts before writes");
                check(cache.restore(lease, {target}, error) && restored == kv.bytes, "large SSD unit restores with bounded native writes");
                cache.restored(lease); cache.restored(lease);
                check(cache.stats().restored_tokens == 2, "native completion acknowledged once per lease");
                cache.release(lease);
                check(cache.stats().ram_bytes == 0, "large restore does not promote a full unit beyond host cap");
            }
            std::filesystem::remove_all(dir);
        }
        {
            const auto dir = scratch("maya-state-stream-fail-");
            auto c = config(); c.directory = dir.string(); c.ram_bytes = 4;
            {
                sc::StateCache cache(c);
                size_t reads = 0;
                sc::RegionSource source;
                source.name = "kv"; source.bytes = 32;
                source.read = [&](uint64_t at, uint8_t* out, size_t n, std::string& e) {
                    if (++reads == 1) { e = "injected native copy failure"; return false; }
                    for (size_t i = 0; i < n; ++i) out[i] = uint8_t(at+i);
                    return true;
                };
                check(!cache.publish(metadata(prefix({1,2})), {source}, error), "failed streaming copy is rejected");
                check(cache.stats().units == 0 && cache.stats().disk_bytes == 0 && !cache.find_resume(prefix({1,2,3})),
                      "partial streamed payload never publishes or leaks its tier reservation");
            }
            {
                sc::StateCache cache(c);
                check(cache.stats().units == 0 && cache.stats().disk_bytes == 0, "restart finds no partial state");
            }
            for (const auto& file : std::filesystem::directory_iterator(dir))
                check(file.path().filename() == "owner.lock", "failed stream leaves no published chunks after restart");
            std::filesystem::remove_all(dir);
        }
        std::cout << "state cache: all behavioral checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
