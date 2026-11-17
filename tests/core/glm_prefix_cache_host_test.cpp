// Candidate helper checks. The Linux parser is compiled on Windows only against
// fresh fixture files; this does not start Linux or inspect a real cgroup.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <cstdio>
#include <cstring>
#include <vector>
#include "strata/core/glm_prefix_admission.hpp"
#include "strata/core/glm_prefix_metadata.hpp"
#include "strata/core/glm_cache_memory.hpp"

namespace fs = std::filesystem;
namespace pm = strata::core::glm_cache_memory;
namespace pa = strata::core::glm_prefix;
static int checks = 0, failures = 0;
static void expect(bool ok, const char* name) {
    ++checks;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}
static std::vector<int32_t> prompt(size_t n, int32_t fill = 17) { return std::vector<int32_t>(n, fill); }
static void put(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path); f << value;
    if (!f) throw std::runtime_error("fixture write failed");
}
struct CgroupFixture {
    fs::path dir, root;
    explicit CgroupFixture(const fs::path& p) : dir(p), root(p / "cgroup") { reset(); }
    void reset() {
        put(dir / "meminfo", "MemTotal: 9999999 kB\nMemAvailable: 1024 kB\n");
        put(dir / "membership", "0::/parent/leaf\n");
        put(root / "cgroup.controllers", "memory\n");
        for (const auto& p : {root / "parent", root / "parent/leaf"}) {
            put(p / "memory.max", "max\n"); put(p / "memory.high", "max\n");
            put(p / "memory.current", "0\n");
        }
        for (const char* n : {"memory.max", "memory.high", "memory.current"}) fs::remove(root / n);
    }
    std::optional<uint64_t> read() const { return pm::linux_available(dir / "meminfo", dir / "membership", root); }
};
int main(int argc, char** argv) {
    if (argc != 2) return 64; // The runner owns this fresh temporary directory.
    const fs::path root(argv[1]);
    {
        pa::Admission a;
        auto first = prompt(4000), second = first, third = first;
        second[1883] = 19; third[1883] = 23;
        expect(a.observe(first, 0) == 0, "H1 first path has no learned boundary");
        expect(a.observe(second, 0) == 1883, "H1 second path learns exact unaligned 1883");
        expect(a.observe(third, 1883) == 0, "H1 materialized ancestor suppresses duplicate boundary");
        // Admission only offers a save boundary. It has no restored-state API;
        // the actual third-request hit is covered by the extracted slot test.
        expect(!pa::matches(first, second), "H1 full old prompt is not a compatible shorter state");
        expect(pa::matches(std::span(first).first(1883), third), "H1 exact saved prefix matches third path");
        third[1882] = 99;
        expect(!pa::matches(std::span(first).first(1883), third), "H1 one token mismatch rejects state");
        expect(!pa::matches({}, first) && !pa::matches(first, first) && !pa::matches(first, std::span(first).first(100)),
               "H1 empty equal and longer state are not strict prefixes");
    }
    {
        pa::Admission a;
        auto first = prompt(4000), second = first, adjacent = first;
        second[1883] = 19; adjacent[1889] = 23;
        a.observe(first, 0); a.observe(second, 0);
        expect(a.observe(adjacent, 1883) == 0, "H1 1889 is too near materialized 1883");
        auto exact = first; exact[2907] = 31;
        expect(a.observe(exact, 1883) == 2907, "H1 exact 1024 spacing is allowed");
    }
    {
        pa::Admission a;
        auto first = prompt(4000), extend = first; extend.push_back(88);
        a.observe(first, 0);
        expect(a.observe(extend, 0) == 0, "H1 pure extension does not learn a fork");
        auto shorter = prompt(3000);
        expect(a.observe(shorter, 0) == 0, "H1 pure truncation does not learn a fork");
        auto below_minimum = first; below_minimum[1023] = 31;
        expect(a.observe(below_minimum, 0) == 0, "H1 fork below minimum rejected");
    }
    {
        pa::Admission a;
        auto first = prompt(4000); a.observe(first, 0);
        // Nine other distinct families rotate out the first exact witness.
        for (int32_t i = 0; i < 9; ++i) a.observe(prompt(4000, 100 + i), 0);
        auto branch = first; branch[1883] = 77;
        expect(a.observe(branch, 0) == 0, "H5 witness count is bounded and old family expires");
        auto huge = prompt(pa::max_path_tokens + 1); huge[1883] = 99;
        expect(a.observe(huge, 0) == 0, "H5 oversized witness rejected");
    }
    expect(pm::permits(100, 32, 68), "H4 exact reserve boundary admitted");
    expect(!pm::permits(100, 32, 69), "H4 one byte past reserve rejected");
    expect(!pm::permits(31, 32, 0), "H4 reserve itself cannot underflow");
    expect(!pm::permits(UINT64_MAX, UINT64_MAX, 1), "H4 uint64 total does not wrap");
    expect(pm::permits(UINT64_MAX, 0, UINT64_MAX), "H4 maximum exact quantity admitted");
    expect(!pm::quantity("-1") && !pm::quantity("12x") && !pm::quantity("") && !pm::quantity("18446744073709551616"),
           "H4 corrupt or overflowing byte quantities rejected");
    expect(pm::quantity("18446744073709551615") == UINT64_MAX, "H4 full uint64 quantity parsed");
    CgroupFixture c(root);
    expect(c.read() == 1048576, "H4 infinite groups use MemAvailable not MemTotal");
    put(c.root / "parent/leaf/memory.max", "900000\n");
    put(c.root / "parent/leaf/memory.current", "300000\n");
    expect(c.read() == 600000, "H4 finite leaf limits available bytes");
    put(c.root / "parent/memory.max", "2000000\n");
    put(c.root / "parent/memory.current", "1800000\n");
    expect(c.read() == 200000, "H4 tighter parent wins over leaf");
    put(c.root / "parent/memory.high", "1900000\n");
    expect(c.read() == 100000, "H4 memory.high protects against pressure");
    put(c.root / "parent/memory.current", "2100000\n");
    expect(c.read() == 0, "H4 over limit clamps to zero");
    c.reset();
    put(c.root / "memory.max", "500000\n"); put(c.root / "memory.current", "100000\n");
    expect(c.read() == 400000, "H4 finite namespace root included");
    c.reset(); fs::remove(c.root / "parent/leaf/memory.max");
    expect(!c.read(), "H4 missing child limit fails closed");
    c.reset(); put(c.root / "parent/leaf/memory.max", "bad\n");
    expect(!c.read(), "H4 malformed child limit fails closed");
    c.reset(); put(c.root / "parent/leaf/memory.max", "900000\n"); fs::remove(c.root / "parent/leaf/memory.current");
    expect(!c.read(), "H4 missing finite usage fails closed");
    c.reset(); put(c.root / "parent/leaf/memory.max", "900000\n"); put(c.root / "parent/leaf/memory.current", "broken\n");
    expect(!c.read(), "H4 corrupt usage fails closed");
    c.reset(); put(c.dir / "meminfo", "MemTotal: 99999 kB\n");
    expect(!c.read(), "H4 missing MemAvailable fails closed");
    put(c.dir / "meminfo", "MemAvailable: 18446744073709551615 kB\n");
    expect(!c.read(), "H4 physical byte conversion cannot overflow");
    c.reset(); put(c.dir / "membership", "3:memory:/parent/leaf\n");
    expect(!c.read(), "H4 cgroup v1 is conservatively unsupported");
    put(c.dir / "membership", "0::/../escape\n");
    expect(!c.read(), "H4 parent traversal rejected");
    c.reset(); fs::remove(c.root / "cgroup.controllers");
    expect(!c.read(), "H4 missing root controller fails closed");
    c.reset(); put(c.root / "memory.max", "");
    expect(!c.read(), "H4 existing unreadable root limit is not an unlimited root");
    {
        const std::string identity = "host-model-layout-A";
        const auto tokens = prompt(1883);
        const std::string payload(100001, 's');
        const uint64_t checksum = pa::hash_bytes(pa::hash_start, payload.data(), payload.size());
        std::ostringstream out(std::ios::binary);
        expect(pa::write_metadata(out, identity, tokens, payload.size(), checksum), "H3 valid learned metadata written");
        const std::string good = out.str();
        pa::Metadata entry;
        std::istringstream in(good, std::ios::binary);
        expect(pa::read_metadata(in, identity, 4096, entry) && entry.tokens == tokens &&
                   entry.bytes == payload.size() && entry.checksum == checksum, "H3 learned metadata roundtrip");
        const auto rejected = [&](std::string data, const std::string& id = "host-model-layout-A", size_t limit = 4096) {
            pa::Metadata sentinel{{88}, 99, 77};
            std::istringstream stream(data, std::ios::binary);
            return !pa::read_metadata(stream, id, limit, sentinel) && sentinel.tokens == std::vector<int32_t>{88} &&
                sentinel.bytes == 99 && sentinel.checksum == 77;
        };
        expect(rejected(good, "host-model-layout-B"), "H3 model identity mismatch leaves old entry intact");
        expect(rejected(good, identity, 1882), "H3 context token bound checked before allocation");
        bool truncations = true;
        for (size_t n : {size_t{0}, size_t{8}, size_t{39}, size_t{40}, good.size() - 8, good.size() - 1})
            truncations = truncations && rejected(good.substr(0, n));
        expect(truncations, "H3 truncated metadata rejected atomically");
        expect(rejected(good + "x"), "H3 trailing metadata bytes rejected");
        auto corrupt = good; corrupt[40 + identity.size()] ^= 1;
        expect(rejected(corrupt), "H3 token checksum corruption rejected");
        corrupt = good; corrupt[32] ^= 1;
        expect(rejected(corrupt), "H3 payload checksum metadata corruption rejected");
        corrupt = good; corrupt[0] ^= 1;
        expect(rejected(corrupt), "H3 foreign or old magic rejected by learned reader");
        corrupt = good;
        uint64_t excess = UINT64_MAX;
        std::memcpy(corrupt.data() + 24, &excess, 8);
        expect(rejected(corrupt), "H5 oversized metadata token count rejected before resize");
        corrupt = good; excess = pa::state_budget + 1;
        std::memcpy(corrupt.data() + 16, &excess, 8);
        expect(rejected(corrupt), "H5 state byte subbudget enforced by metadata reader");
        std::ostringstream invalid;
        expect(!pa::write_metadata(invalid, "", tokens, payload.size(), checksum) && invalid.str().empty(),
               "H3 invalid identity rejected before writing");
        std::ostringstream short_tokens;
        expect(!pa::write_metadata(short_tokens, identity, std::span(tokens).first(1023), payload.size(), checksum),
               "H5 short learned prefix cannot be persisted");
        std::istringstream data(payload, std::ios::binary);
        expect(pa::check_payload(data, payload.size(), checksum), "H3 payload stream checksum crosses 64KiB chunk");
        std::istringstream bad(payload, std::ios::binary);
        expect(!pa::check_payload(bad, payload.size(), checksum ^ 1), "H3 wrong payload checksum rejected");
        std::istringstream tail(payload + "x", std::ios::binary);
        expect(!pa::check_payload(tail, payload.size(), checksum), "H3 trailing payload bytes rejected");
        std::istringstream truncated(payload.substr(0, payload.size() - 1), std::ios::binary);
        expect(!pa::check_payload(truncated, payload.size(), checksum), "H3 truncated payload rejected");
    }
    std::printf("RESULT %d/%d\n", checks - failures, checks);
    return failures ? 1 : 0;
}
