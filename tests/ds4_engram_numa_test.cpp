// Linux CPU-only failure/cleanup regression. NUMA syscalls are simulated; no privilege or two-node host needed.
// The real warm shared-file migration is a separate operator check, not a CI topology assumption.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct BindCall { size_t bytes; int mode; unsigned long mask; unsigned flags; };
static std::vector<BindCall> calls;
static std::vector<int> errors;
static long page_size = 4096;
static std::array<unsigned long, 4> allowed{3, 0, 0, 0};
static int query_error = 0;

static long fake_sysconf(int name) {
    check(name == _SC_PAGESIZE, "unexpected sysconf query");
    return page_size;
}
static long fake_syscall(long number, std::nullptr_t, unsigned long* mask, unsigned long maxnode,
                         std::nullptr_t, int flags) {
    check(number == SYS_get_mempolicy && maxnode == 256 && flags == MPOL_F_MEMS_ALLOWED, "allowed-node query");
    if (query_error) { errno = query_error; return -1; }
    std::copy(allowed.begin(), allowed.end(), mask);
    return 0;
}
static long fake_syscall(long number, void*, size_t bytes, int mode, unsigned long* mask,
                         unsigned long maxnode, unsigned flags) {
    check(number == SYS_mbind && maxnode == sizeof(*mask) * 8, "mbind syscall shape");
    calls.push_back({bytes, mode, *mask, flags});
    const int error = calls.size() <= errors.size() ? errors[calls.size() - 1] : 0;
    if (error) { errno = error; return -1; }
    return 0;
}

// Include only this header with a simulated OS boundary; production code remains byte-identical.
#define syscall fake_syscall
#define sysconf fake_sysconf
#include "../src/ds4/ds4_engram_numa.hpp"
#undef sysconf
#undef syscall
#include "../src/ds4/ds4_engram_io.hpp"

using strata::ds4::EngramNuma;
static void reset(std::vector<int> injected = {}) { calls.clear(); errors = std::move(injected); }
static int mappings(const char* path) {
    std::ifstream maps("/proc/self/maps");
    check((bool) maps, "read own mappings");
    int n = 0;
    for (std::string line; std::getline(maps, line);) n += line.find(path) != std::string::npos;
    return n;
}
static void clear_flags() {
    for (const char* key : {"STRATA_DS4_ENGRAM_RAM_NUMA", "MAYA_DS4_ENGRAM_RAM_NUMA", "DS4_ENGRAM_RAM_NUMA"})
        check(::unsetenv(key) == 0, "clear setting");
}

int main() {
    void* memory = MAP_FAILED;
    int fd = -1;
    char path[] = "/tmp/maya-engram-numa-XXXXXX";
    try {
        clear_flags();
        check(!strata::ds4::engram_numa_on(), "default off");
        ::setenv("DS4_ENGRAM_RAM_NUMA", "1x", 1);
        check(!strata::ds4::engram_numa_on(), "exact opt-in only");
        ::setenv("DS4_ENGRAM_RAM_NUMA", "1", 1);
        check(strata::ds4::engram_numa_on(), "short alias");
        ::setenv("MAYA_DS4_ENGRAM_RAM_NUMA", "0", 1);
        check(!strata::ds4::engram_numa_on(), "MAYA over short alias");
        ::setenv("STRATA_DS4_ENGRAM_RAM_NUMA", "1", 1);
        check(strata::ds4::engram_numa_on(), "direct header STRATA precedence");
        clear_flags();

        std::string err;
        EngramNuma numa;
        page_size = 3000; check(!numa.init(err), "unsupported page size refused");
        page_size = 4096;
        query_error = EPERM; check(!numa.init(err), "allowed-node query failure refused");
        query_error = 0;
        for (auto mask : {std::array<unsigned long, 4>{1, 0, 0, 0}, {2, 0, 0, 0}, {7, 0, 0, 0}, {3, 1, 0, 0}}) {
            allowed = mask; check(!numa.init(err), "only nodes exactly 0 and 1 accepted");
        }
        allowed = {3, 0, 0, 0}; check(numa.init(err), "two-node init");
        constexpr size_t bytes = 2 * EngramNuma::kStripe;
        memory = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(memory != MAP_FAILED, "owned anonymous fixture");
        auto* data = static_cast<uint8_t*>(memory);
        std::memset(data, 37, bytes);
        reset(); check(numa.balance(data, bytes, 0, err), "already-correct pages");
        check(calls.size() == 6 && numa.verified[0] == 512 && numa.verified[1] == 512, "two strict scans per stripe");
        check(calls[0].mask == 1 && calls[3].mask == 2 && calls[2].mode == MPOL_DEFAULT && calls[5].mode == MPOL_DEFAULT,
              "single-node alternating stripes and immediate DEFAULT restoration");

        EngramNuma warm; check(warm.init(err), "warm init");
        reset({EIO, 0, 0, 0}); check(warm.balance(data, 4096, EngramNuma::kStripe, err), "warm migration");
        check(calls.size() == 4 && calls[1].flags == (MPOL_MF_MOVE_ALL | MPOL_MF_STRICT) &&
              calls[2].flags == MPOL_MF_STRICT && calls[3].mode == MPOL_DEFAULT && warm.relocated_batches == 1,
              "MOVE_ALL, independent strict verification, DEFAULT");
        EngramNuma offset; check(offset.init(err), "offset init");
        reset(); check(offset.balance(data, EngramNuma::kStripe + 8192, 4096, err), "unaligned stripe origin");
        check(calls[0].bytes == EngramNuma::kStripe - 4096 && calls[3].bytes == 12288 && offset.verified[1] == 3,
              "stripe boundaries follow file offset");

        for (const auto& injected : {std::vector<int>{EPERM, 0}, {EIO, EPERM, 0}, {EIO, EIO, 0}, {0, EIO, 0}}) {
            EngramNuma bad; check(bad.init(err), "failure init");
            reset(injected); check(!bad.balance(data, 4096, 0, err), "failed operation must refuse startup");
            check(calls.back().mode == MPOL_DEFAULT && err.find("moves can persist") != std::string::npos &&
                  bad.verified[0] == 0, "failure restores policy, reports persistent moves, does not count verified pages");
        }
        reset({0, 0, EPERM}); check(!numa.balance(data, 4096, 0, err), "failed DEFAULT reset refused");
        check(err.find("restore mapping policy; candidate refused") != std::string::npos, "reset error message");
        reset({EIO, EPERM, EPERM}); check(!numa.balance(data, 4096, 0, err), "failed cleanup refused");
        check(err.find("policy reset also failed") != std::string::npos, "cleanup failure preserved");
        reset(); check(!numa.balance(data + 1, 4096, 0, err) && calls.empty(), "unaligned range refused before syscall");
        check(!numa.balance(data, 8192, UINT64_MAX - 4095, err) && calls.empty(), "overflow refused before syscall");
        for (size_t i = 0; i < bytes; ++i) check(data[i] == 37, "placement never writes tensor bytes");

        // Exercise actual pin_range error cleanup on a private 4 KiB file, before mlock; no real migration.
        fd = ::mkstemp(path); check(fd >= 0 && ::ftruncate(fd, 4096) == 0, "owned tiny file");
        ::setenv("DS4_ENGRAM_RAM_NUMA", "1", 1);
        reset({EIO, EPERM, 0});
        const int before = mappings(path);
        { strata::ds4::EgReader reader; check(!reader.pin_range(path, 0, 0, 4096, err), "pin_range propagates NUMA failure"); }
        check(mappings(path) == before && err.find("strict migration incomplete") != std::string::npos,
              "pin_range unmaps failed range and preserves error");
        clear_flags();
        ::close(fd); fd = -1; ::unlink(path); ::munmap(memory, bytes); memory = MAP_FAILED;
        std::puts("ds4_engram_numa: opt-in, topology, strict stripes, failure propagation and mapping cleanup OK (simulated NUMA)");
        return 0;
    } catch (const std::exception& e) {
        if (fd >= 0) { ::close(fd); ::unlink(path); }
        if (memory != MAP_FAILED) ::munmap(memory, 2 * EngramNuma::kStripe);
        clear_flags();
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
