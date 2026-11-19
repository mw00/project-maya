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
#include "strata/core/glm_slot_directory_scan.hpp"
#include "strata/core/regular_file_stream.hpp"

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
static void regular_stream_checks(const fs::path& root) {
    using File = strata::core::RegularFileStream;
    fs::create_directories(root);
    const auto path = root / "regular-roundtrip.bin";
    std::error_code ec;
    auto writer = File::create_exclusive(path, 8, ec);
    expect(writer && !ec, "REGULAR_STREAM exclusive create owns a new normal file");
    if (!writer) return;
    writer->output().write("abcdefgh", 8);
    expect(writer->output().good() && writer->output().tellp() == 8,
           "REGULAR_STREAM writing exactly the limit preserves the full payload");
    writer->output().seekp(9);
    expect(!writer->output() && !writer->last_error(),
           "REGULAR_STREAM out-of-range output seek is recoverable without native I/O failure");
    writer->output().clear();
    writer->output().seekp(2);
    writer->output().write("XY", 2);
    expect(writer->output().good() && writer->size() == 8 && writer->flush(ec) && !ec,
           "REGULAR_STREAM bounded output seek overwrites without truncation and flush succeeds");
    expect(writer->close(ec) && !ec && writer->size() == 8 && fs::file_size(path) == 8,
           "REGULAR_STREAM explicit close succeeds and retains measured size");
    auto existing = File::create_exclusive(path, 8, ec);
    expect(!existing && bool(ec) && fs::file_size(path) == 8,
           "REGULAR_STREAM failed exclusive create cannot truncate or remove an existing object");
    auto reader = File::open_read(path, 8, ec);
    expect(reader && !ec && reader->size() == 8, "REGULAR_STREAM normal file opens for bounded input");
    if (!reader) return;
    auto& input = reader->input();
    expect(input.peek() == 'a' && input.tellg() == 0 && input.get() == 'a',
           "REGULAR_STREAM peek and tell do not consume input bytes");
    input.unget();
    char bytes[8]{};
    input.read(bytes, 8);
    expect(input.good() && std::string(bytes, 8) == "abXYefgh" && input.get() == std::char_traits<char>::eof(),
           "REGULAR_STREAM complete normal payload roundtrips and reaches EOF");
    input.clear();
    input.seekg(0);
    input.read(bytes, 8);
    expect(input.good() && std::string(bytes, 8) == "abXYefgh",
           "REGULAR_STREAM same owned handle rewinds after reading to EOF");
    input.seekg(9);
    expect(!input && !reader->last_error(), "REGULAR_STREAM out-of-range input seek only sets stream fail state");
    input.clear();
    input.seekg(2);
    expect(input.get() == 'X', "REGULAR_STREAM clear plus valid seek recovers the same owned input handle");
    input.seekg(-2, std::ios::end);
    expect(input.get() == 'g' && input.get() == 'h' && reader->close(ec) && !ec,
           "REGULAR_STREAM input end-relative seek and explicit close succeed");
    auto too_small = File::open_read(path, 7, ec);
    expect(!too_small && bool(ec) && fs::file_size(path) == 8,
           "REGULAR_STREAM input size limit rejects an ordinary larger file without modifying it");
    const auto bounded_path = root / "regular-bound.bin";
    auto bounded = File::create_exclusive(bounded_path, 8, ec);
    expect(bounded && !ec, "REGULAR_STREAM small bounded writer is created");
    if (bounded) {
        bounded->output().write("12345678", 8);
        bounded->output().put('9');
        expect(!bounded->output() && bool(bounded->last_error()) && bounded->size() == 8,
               "REGULAR_STREAM extra byte is rejected without changing the full payload");
        expect(!bounded->close(ec) && bool(ec) && fs::file_size(bounded_path) == 8,
               "REGULAR_STREAM explicit close reports prior output failure while closing ownership");
        auto intact = File::open_read(bounded_path, 8, ec);
        expect(bool(intact) && !ec, "REGULAR_STREAM bound failure leaves the ordinary payload readable");
        if (intact) {
            intact->input().read(bytes, 8);
            expect(std::string(bytes, 8) == "12345678" && intact->close(ec) && !ec,
                   "REGULAR_STREAM rejected extra output never truncates existing bytes");
        }
    }
    const auto directory = root / "ordinary-directory";
    fs::create_directory(directory);
    expect(!File::open_read(directory, 8, ec) && bool(ec) && fs::is_directory(directory),
           "REGULAR_STREAM ordinary directory is rejected before stream use");
    expect(!File::create_exclusive(directory, 8, ec) && bool(ec) && fs::is_directory(directory),
           "REGULAR_STREAM create collision leaves an ordinary directory untouched");
    const auto automatic_path = root / "regular-owned-close.bin";
    {
        auto automatic = File::create_exclusive(automatic_path, uint64_t{1} << 40, ec);
        expect(automatic && !ec, "REGULAR_STREAM large byte ceiling does not preallocate ceiling-sized memory");
        if (automatic) automatic->output().write("end", 3);
    }
    auto automatic_read = File::open_read(automatic_path, 3, ec);
    expect(automatic_read && !ec && automatic_read->size() == 3,
           "REGULAR_STREAM RAII closes a tiny normal file when explicit close is omitted");
    if (automatic_read) expect(automatic_read->close(ec) && !ec, "REGULAR_STREAM RAII roundtrip input closes cleanly");
}
static void scan_checks(const fs::path& root) {
    using Scan = strata::core::SlotDirectoryScan;
    static_assert(Scan::batch_capacity == 64);
    // These are names only, not filesystem objects: classification requires no attributes.
    expect(!Scan::accepts_name("notes.meta", true) && !Scan::accepts_name("unrelated.bin", false),
           "SCAN_NAME_CONTRACT unrelated names are rejected without filesystem attributes");
    expect(Scan::accepts_name("slot-session.bin.tmp", false) && Scan::accepts_name("slot-incomplete", true),
           "SCAN_NAME_CONTRACT incomplete ordinary names remain available for cleanup");
    expect(Scan::accepts_name("branch-session.meta.tmp", true) && !Scan::accepts_name("branch-session.meta", false),
           "SCAN_NAME_CONTRACT learned names follow enabled policy including cleanup items");
    std::error_code type_error;
    expect(Scan::accepts_type(fs::file_type::regular, type_error) && !type_error,
           "SCAN_TYPE regular file is an accepted candidate");
    for (const auto type : {fs::file_type::directory, fs::file_type::symlink, fs::file_type::block,
                            fs::file_type::character, fs::file_type::fifo, fs::file_type::socket}) {
        type_error = std::make_error_code(std::errc::io_error);
        expect(!Scan::accepts_type(type, type_error) && !type_error,
               "SCAN_TYPE known unsupported type is skipped without following or opening it");
    }
    for (const auto type : {fs::file_type::none, fs::file_type::not_found, fs::file_type::unknown}) {
        expect(!Scan::accepts_type(type, type_error) && type_error == std::errc::io_error,
               "SCAN_TYPE unknown or disappeared enumerated type fails closed");
    }
    for (const size_t count : {size_t{0}, size_t{1}, size_t{64}, size_t{65}, size_t{197}}) {
        for (const bool deleting : {false, true}) {
            const auto dir = root / ("scan-" + std::to_string(count) + (deleting ? "-delete" : "-keep"));
            fs::create_directories(dir);
            std::vector<Scan::Entry> expected, visited;
            const auto now = fs::file_time_type::clock::now();
            for (size_t i = 0; i < count; ++i) {
                const char* prefix = i % 3 == 0 ? "slot-" : i % 3 == 1 ? "branch-" : "unrelated-";
                const auto path = dir / (prefix + std::to_string(i) + (i % 2 ? ".meta" : ".bin"));
                if (i % 67 == 0) fs::create_directory(path);
                else put(path, "fixture");
                fs::last_write_time(path, now - std::chrono::seconds(i % 3));
                if (i % 3 != 2 && i % 67 != 0) expected.emplace_back(fs::last_write_time(path), path);
            }
            std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            Scan scan(dir, true);
            std::error_code ec;
            size_t batches = 0, peak = 0;
            expect(scan.all_entries().empty(), "SCAN_ATOMIC no catalogue is public before enumeration");
            auto first = scan.next(ec);
            expect(scan.visited_entries() == count && scan.candidate_entries() == expected.size(),
                   "SCAN_SINGLE_PASS first next completes exact visit and candidate counts");
            expect(scan.all_entries().size() == expected.size() &&
                   std::equal(scan.all_entries().begin(), scan.all_entries().end(), expected.begin()),
                   "SCAN_ATOMIC first next exposes only the complete sorted catalogue");
            for (auto batch = first; !batch.empty(); batch = scan.next(ec)) {
                ++batches;
                peak = std::max(peak, batch.size());
                for (const auto& item : batch) {
                    visited.push_back(item);
                    if (deleting) fs::remove(item.second);
                }
            }
            std::printf("SCAN_CASE count=%zu candidates=%zu delete=%d batches=%zu peak=%zu\n",
                        count, expected.size(), int(deleting), batches, peak);
            expect(!ec, "scan normal mixed directory has no filesystem errors");
            expect(scan.visited_entries() == count && scan.candidate_entries() == expected.size(),
                   "SCAN_SINGLE_PASS returning later windows never enumerates again");
            expect(peak <= 64, "scan batch never exceeds 64 entries");
            expect(batches == (expected.size() + 63) / 64, "scan crosses all candidate batch boundaries without empty gaps");
            expect(visited == expected, "scan preserves all timestamp/path ordering including deleted cursors");
            expect(!deleting || std::all_of(expected.begin(), expected.end(), [](const auto& item) {
                return !fs::exists(item.second);
            }), "scan visits and removes every candidate item exactly once");
            expect(count < 3 || fs::exists(dir / "unrelated-2.bin"), "scan leaves unrelated mixed items untouched");
            expect(count == 0 || fs::is_directory(dir / "slot-0.bin"),
                   "SCAN_TYPE candidate-named ordinary directory is left untouched");
        }
    }
    const auto budget_dir = root / "scan-small-budgets";
    for (const char* name : {"slot-a.meta", "branch-b.bin", "slot-c.tmp"}) put(budget_dir / name, "fixture");
    Scan exact(budget_dir, true, 3, 3);
    std::error_code ec;
    expect(exact.next(ec).size() == 3 && !ec && exact.visited_entries() == 3 && exact.candidate_entries() == 3,
           "SCAN_BUDGET exact small visit and candidate limits complete at end");
    put(budget_dir / "slot-later.meta", "fixture");
    expect(exact.next(ec).empty() && !ec && exact.all_entries().size() == 3 && exact.visited_entries() == 3,
           "SCAN_SINGLE_PASS entries added after first next do not trigger another enumeration");
    fs::remove(budget_dir / "slot-later.meta");
    const auto mixed_type_dir = root / "scan-small-types";
    fs::create_directories(mixed_type_dir / "slot-directory.meta");
    fs::create_directory(mixed_type_dir / "branch-directory.bin");
    put(mixed_type_dir / "slot-valid.meta", "fixture");
    expect(!Scan::is_regular_entry(mixed_type_dir / "slot-directory.meta", ec) && !ec &&
           Scan::is_regular_entry(mixed_type_dir / "slot-valid.meta", ec) && !ec,
           "SCAN_TYPE real ordinary directory is skipped and regular file accepted without following links");
    Scan mixed_types(mixed_type_dir, true, 3, 1);
    const auto mixed_batch = mixed_types.next(ec);
    expect(!ec && mixed_batch.size() == 1 && mixed_batch.front().second.filename() == "slot-valid.meta" &&
           mixed_types.visited_entries() == 3 && mixed_types.candidate_entries() == 1,
           "SCAN_TYPE ordinary directories cannot consume candidate allowance or block valid files");
    expect(mixed_types.next(ec).empty() && !ec && mixed_types.all_entries().size() == 1,
           "SCAN_TYPE exact regular candidate limit succeeds despite unsupported mixed directories");
    const auto directory_only = root / "scan-directory-only";
    fs::create_directories(directory_only / "slot-directory.tmp");
    Scan unsupported_only(directory_only, true, 1, 0);
    expect(unsupported_only.next(ec).empty() && !ec && unsupported_only.visited_entries() == 1 &&
           unsupported_only.candidate_entries() == 0,
           "SCAN_TYPE known directory succeeds with zero regular candidate allowance");
    Scan visit_limited(budget_dir, true, 2, 3);
    expect(visit_limited.next(ec).empty() && ec == std::errc::no_buffer_space &&
           visit_limited.visited_entries() == 2 && visit_limited.candidate_entries() == 2 &&
           visit_limited.all_entries().empty(),
           "SCAN_BUDGET small visit limit fails atomically before processing an extra item");
    ec.clear();
    expect(visit_limited.next(ec).empty() && ec == std::errc::no_buffer_space &&
           visit_limited.visited_entries() == 2 && visit_limited.all_entries().empty(),
           "SCAN_BUDGET failed enumeration is sticky and never exposes partial items");
    Scan candidate_limited(budget_dir, true, 3, 2);
    expect(candidate_limited.next(ec).empty() && ec == std::errc::no_buffer_space &&
           candidate_limited.visited_entries() == 3 && candidate_limited.candidate_entries() == 2 &&
           candidate_limited.all_entries().empty(),
           "SCAN_BUDGET small candidate limit fails before reading or retaining an extra candidate");
    Scan zero_visits(budget_dir, true, 0, 3);
    expect(zero_visits.next(ec).empty() && ec == std::errc::no_buffer_space &&
           zero_visits.visited_entries() == 0 && zero_visits.candidate_entries() == 0,
           "SCAN_BUDGET zero visits rejects a nonempty directory without processing items");
    const auto irrelevant_dir = root / "scan-small-unrelated";
    put(irrelevant_dir / "notes.txt", "fixture");
    put(irrelevant_dir / "other.bin", "fixture");
    Scan no_candidates(irrelevant_dir, true, 2, 0);
    expect(no_candidates.next(ec).empty() && !ec && no_candidates.visited_entries() == 2 &&
           no_candidates.candidate_entries() == 0 && no_candidates.all_entries().empty(),
           "SCAN_BUDGET unrelated entries count as visits but require no candidate allowance");
    const auto empty_dir = root / "scan-small-empty";
    fs::create_directory(empty_dir);
    Scan empty(empty_dir, true, 0, 0);
    expect(empty.next(ec).empty() && !ec && empty.visited_entries() == 0 && empty.candidate_entries() == 0,
           "SCAN_BUDGET empty directory succeeds with zero limits and no error");
    Scan clamped(empty_dir, true, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
    expect(clamped.visit_limit() == Scan::max_visits && clamped.candidate_limit() == Scan::max_candidates,
           "SCAN_BUDGET explicit limits cannot enlarge production bounds");
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
    scan_checks(root);
    regular_stream_checks(root / "regular-streams");
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
        expect(pa::write_metadata(short_tokens, identity, std::span(tokens).first(12), payload.size(), checksum),
               "H5 explicit low threshold short key can be persisted");
        std::istringstream short_input(short_tokens.str(), std::ios::binary);
        pa::Metadata short_entry;
        expect(pa::read_metadata(short_input, identity, 12, short_entry) && short_entry.tokens.size() == 12,
               "H5 short key roundtrips independently of default admission threshold");
        std::ostringstream empty_tokens;
        expect(!pa::write_metadata(empty_tokens, identity, {}, payload.size(), checksum),
               "H5 empty state key cannot be persisted");
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
