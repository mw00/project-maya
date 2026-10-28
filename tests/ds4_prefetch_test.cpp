// Linux read-ahead skips warm ranges and retains hints when residency samples detect cold pages.
#include "../src/ds4/ds4_os.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace {
int prefetch_calls = 0;

class PrefetchMapping {
public:
    explicit PrefetchMapping(size_t bytes) : bytes_(bytes) {
        data_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data_ == MAP_FAILED) {
            std::perror("ds4 prefetch test: mmap");
            std::exit(2);
        }
    }
    ~PrefetchMapping() { munmap(data_, bytes_); }
    PrefetchMapping(const PrefetchMapping&) = delete;
    PrefetchMapping& operator=(const PrefetchMapping&) = delete;
    unsigned char* data() const { return static_cast<unsigned char*>(data_); }
    size_t size() const { return bytes_; }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
};

bool check_file_demand_read(size_t page) {
    char name[] = "ds4-prefetch-fixture-XXXXXX";
    const int fd = mkstemp(name);
    if (fd < 0) return false;
    const size_t bytes = 16 * page;
    std::vector<unsigned char> expected(bytes);
    for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>(i * 17 + 43);
    bool ok = write(fd, expected.data(), bytes) == static_cast<ssize_t>(bytes) && fsync(fd) == 0;
    // Evict only this newly created private fixture, never a model mapping or the host cache.
    ok = ok && posix_fadvise(fd, 0, bytes, POSIX_FADV_DONTNEED) == 0;
    void* mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    unlink(name);
    close(fd);
    if (mapping == MAP_FAILED) return false;
    ok = ok && madvise(mapping, bytes, MADV_RANDOM) == 0;
    const auto* data = static_cast<const unsigned char*>(mapping);
    for (const size_t index : {0u, 5u, 10u, 15u}) ok = (data[index * page] == expected[index * page]) && ok;
    unsigned char residency = 0;
    ok = mincore(const_cast<unsigned char*>(data + 7 * page), page, &residency) == 0 && !(residency & 1) && ok;
    prefetch_calls = 0;
    strata::ds4::os::prefetch_cold({{data, bytes}});
    ok = prefetch_calls == 0 && ok;
    ok = std::memcmp(data, expected.data(), bytes) == 0 && ok;
    munmap(mapping, bytes);
    if (!ok) std::fprintf(stderr, "ds4 prefetch test: private file miss must skip the hint and load exact bytes on demand\n");
    return ok;
}

bool expect_prefetch_calls(int expected, const char* scenario) {
    if (prefetch_calls == expected) return true;
    std::fprintf(stderr, "ds4 prefetch test: %s: expected %d hints, got %d\n", scenario, expected, prefetch_calls);
    return false;
}
}  // namespace

extern "C" int __real_madvise(void* address, size_t length, int advice);
extern "C" int __wrap_madvise(void* address, size_t length, int advice) {
    if (advice == MADV_WILLNEED) ++prefetch_calls;
    return __real_madvise(address, length, advice);
}

int main(int argc, char** argv) {
    const bool legacy_all = argc == 2 && std::strcmp(argv[1], "--all") == 0;
    const bool legacy_zero = argc == 2 && std::strcmp(argv[1], "--legacy-zero") == 0;
    const bool override_all = argc == 2 && std::strcmp(argv[1], "--override") == 0;
    const bool default_mode = argc == 2 && std::strcmp(argv[1], "--default") == 0;
    const int samples = argc == 3 ? std::atoi(argv[2]) : 4;
    const bool all = legacy_all || legacy_zero || override_all || default_mode || samples == 0;
    unsetenv("DS4_PF_ALL");
    unsetenv("DS4_PF_SAMPLES");
    if (legacy_all || override_all) setenv("DS4_PF_ALL", "1", 1);
    if (legacy_zero) setenv("DS4_PF_ALL", "0", 1);
    if (!legacy_all && !legacy_zero && !default_mode)
        setenv("DS4_PF_SAMPLES", argc == 3 ? argv[2] : "4", 1);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    PrefetchMapping warm(3 * page), cold(2 * page);
    namespace os = strata::ds4::os;
    bool ok = true;

    os::prefetch_cold({});
    ok = expect_prefetch_calls(0, "empty request") && ok;

    os::prefetch_cold({{cold.data(), cold.size()}});
    ok = expect_prefetch_calls(1, "cold range keeps its hint") && ok;

    for (size_t i = 0; i < warm.size(); i += page) warm.data()[i] = 42;
    if (!os::resident(warm.data(), warm.size())) {
        std::fprintf(stderr, "ds4 prefetch test: touched fixture is not resident\n");
        return 2;
    }
    prefetch_calls = 0;
    os::prefetch_cold({{warm.data(), warm.size()}});
    ok = expect_prefetch_calls(all ? 1 : 0, "fully resident range") && ok;

    prefetch_calls = 0;
    os::prefetch_cold({{warm.data() + 1, page}});
    ok = expect_prefetch_calls(all ? 1 : 0, "unaligned fully resident range") && ok;

    // One sample deliberately misses the absent second page. Larger counts must find it.
    cold.data()[0] = 42;
    prefetch_calls = 0;
    os::prefetch_cold({{warm.data(), warm.size()}, {cold.data(), cold.size()}});
    ok = expect_prefetch_calls(all ? 2 : (samples == 1 ? 0 : 1), "mixed warm and partially cold ranges") && ok;

    // Cached edges alone are not enough: the interior samples must still keep a cold range's hint.
    PrefetchMapping sparse(16 * page);
    sparse.data()[0] = 42;
    sparse.data()[15 * page] = 42;
    prefetch_calls = 0;
    os::prefetch_cold({{sparse.data(), sparse.size()}});
    ok = expect_prefetch_calls(all || samples > 2 ? 1 : 0, "cached edges with cold interior") && ok;

    // For four samples on this 16-page range, the literal page indices are 0, 5, 10, 15.
    sparse.data()[5 * page] = 42;
    sparse.data()[10 * page] = 42;
    prefetch_calls = 0;
    os::prefetch_cold({{sparse.data(), sparse.size()}});
    ok = expect_prefetch_calls(all || samples > 4 ? 1 : 0, "unsampled cold interior") && ok;
    if (sparse.data()[7 * page] != 0) return 2;  // A missed anonymous page still loads its zero-fill on demand.
    sparse.data()[7 * page] = 43;
    if (sparse.data()[7 * page] != 43) return 2;

    prefetch_calls = 0;
    os::prefetch_cold({{warm.data() + 1, 1}});
    ok = expect_prefetch_calls(all ? 1 : 0, "one-byte unaligned range") && ok;
    prefetch_calls = 0;
    os::prefetch_cold({{reinterpret_cast<const void*>(1), page}});
    ok = expect_prefetch_calls(1, "failed residency query retains hint") && ok;

    if (!all && samples == 4) ok = check_file_demand_read(page) && ok;
    if (ok) std::printf("ds4_prefetch_test: all checks passed, force-all=%d samples=%d\n", all, samples);
    return ok ? 0 : 1;
}
