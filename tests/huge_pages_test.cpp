// tests/huge_pages_test.cpp - reserved and on-demand huge-page arithmetic, then a real mapping when a pool exists.
// CPU only; no pool on a CI runner is fine, but a pool that advertises room must deliver explicit 2 MB pages.
#include "strata/platform/huge_pages.hpp"

#include <cstdio>
#include <cerrno>

#ifdef __linux__
#include <cstring>
#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#endif

namespace p = strata::platform;

static int fails = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                   \
        }                                                              \
    } while (0)

#ifdef __linux__
static int open_fds() {
    DIR* dir = ::opendir("/proc/self/fd");
    CHECK(dir != nullptr);
    if (dir == nullptr) return -1;
    int count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(dir)) {
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    }
    CHECK(errno == 0);
    CHECK(::closedir(dir) == 0);
    return count - 1;  // Do not count the directory handle used for this inspection.
}

static int huge_file_maps() {
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    CHECK(f != nullptr);
    if (f == nullptr) return -1;
    int count = 0;
    char line[1024];
    while (std::fgets(line, sizeof line, f))
        if (std::strstr(line, "memfd:strata-huge") != nullptr) ++count;
    CHECK(!std::ferror(f));
    std::fclose(f);
    return count;
}

static int fill_calls = 0;

static int record_fill(int, off_t, off_t) {
    ++fill_calls;
    errno = ENOMEM;
    return -1;
}

static int fill_first_piece(int fd, off_t off, off_t len) {
    if (++fill_calls == 1) {
        int result;
        do {
            result = ::fallocate(fd, 0, off, len);
        } while (result != 0 && errno == EINTR);
        return result;
    }
    errno = ENOMEM;
    return -1;
}

static int fill_after_eintr(int fd, off_t off, off_t len) {
    if (++fill_calls <= 2) {
        errno = EINTR;
        return -1;
    }
    return ::fallocate(fd, 0, off, len);
}

static bool kernel_2m(const void* p) {
    std::FILE* f = std::fopen("/proc/self/smaps", "r");
    if (f == nullptr) return false;
    const auto addr = (uintptr_t) p;
    bool in_mapping = false, found = false;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0, kb = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) == 2) in_mapping = addr >= lo && addr < hi;
        else if (in_mapping && std::sscanf(line, "KernelPageSize: %llu kB", &kb) == 1) {
            found = kb == 2048;
            break;
        }
    }
    std::fclose(f);
    return found;
}

static void check_policy(void* addr, int expected) {
    int mode = -1;
    unsigned long mask[4] = {0, 0, 0, 0};
    const long err = syscall(SYS_get_mempolicy, &mode, mask, sizeof mask * 8, addr, 2 /* MPOL_F_ADDR */);
    if (err != 0 && (errno == ENOSYS || errno == EPERM)) {
        std::printf("skipped: get_mempolicy unavailable for policy %d (errno %d)\n", expected, errno);
        return;
    }
    CHECK(err == 0);
    if (err == 0) {
        CHECK(mode == expected);
        CHECK((mask[0] & 1ul) != 0);
    }
}

static bool thread_policy(int* mode, unsigned long (&mask)[4]) {
    const long err = syscall(SYS_get_mempolicy, mode, mask, sizeof mask * 8, nullptr, 0);
    if (err != 0 && (errno == ENOSYS || errno == EPERM)) {
        std::printf("skipped: get_mempolicy unavailable for thread policy (errno %d)\n", errno);
        return false;
    }
    CHECK(err == 0);
    return err == 0;
}

static void check_thread_policy(int expected_mode, const unsigned long (&expected_mask)[4]) {
    int mode = -1;
    unsigned long mask[4] = {0, 0, 0, 0};
    if (!thread_policy(&mode, mask)) return;
    CHECK(mode == expected_mode);
    for (size_t i = 0; i < 4; ++i) CHECK(mask[i] == expected_mask[i]);
}
#endif

int main() {
    constexpr uint64_t page = p::kHugePage2M, unknown = ~uint64_t{0};
    {
        const auto pool = p::huge_pool_from(0, 0, 0, 0, unknown);
        CHECK(pool.reserved_free == 0 && pool.on_demand == 0 && pool.total() == 0);
    }
    {
        const auto pool = p::huge_pool_from(7, 2, 0, 0, 0);
        CHECK(pool.reserved_free == 5 * page && pool.on_demand == 0 && pool.total() == 5 * page);
    }
    {
        const auto pool = p::huge_pool_from(0, 0, 9, 2, unknown);
        CHECK(pool.reserved_free == 0 && pool.on_demand == 7 * page && pool.total() == 7 * page);
    }
    {
        const auto pool = p::huge_pool_from(7, 2, 9, 2, unknown);
        CHECK(pool.reserved_free == 5 * page && pool.on_demand == 7 * page && pool.total() == 12 * page);
    }
    CHECK(p::huge_pool_from(1, 2, 3, 4, unknown).total() == 0);
    CHECK(p::huge_pool_from(1, 2, 3, 0, unknown).reserved_free == 0);
    CHECK(p::huge_pool_from(3, 0, 1, 2, unknown).on_demand == 0);
    {   // MemAvailable caps only on-demand room, rounded down, never the reserved pool.
        const auto pool = p::huge_pool_from(7, 2, 9, 2, 3 * page + page / 2);
        CHECK(pool.reserved_free == 5 * page && pool.on_demand == 3 * page && pool.total() == 8 * page);
        CHECK(p::huge_pool_from(0, 0, 9, 2, page - 1).on_demand == 0);
        CHECK(p::huge_pool_from(0, 0, 9, 2, 7 * page).on_demand == 7 * page);
        CHECK(p::huge_pool_from(0, 0, 9, 2, 100 * page).on_demand == 7 * page);
        CHECK(p::huge_pool_from(0, 0, 9, 2, 0).on_demand == 0);
    }
    CHECK(p::huge_round_up(0) == 0);
    CHECK(p::huge_round_up(1) == page);
    CHECK(p::huge_round_up(page - 1) == page);
    CHECK(p::huge_round_up(page) == page);
    CHECK(p::huge_round_up(page + 1) == 2 * page);
    CHECK(p::huge_round_up(3 * page / 2) == 2 * page);
    {   // Automatic needs the whole rounded allocation; explicit on still needs a nonempty allocation.
        const p::HugePool pool{page, page};
        CHECK(p::huge_on(-1, pool, page + 1));
        CHECK(p::huge_on(-1, pool, 2 * page));
        CHECK(!p::huge_on(-1, pool, 2 * page + 1));
        CHECK(!p::huge_on(-1, p::HugePool{page, 0}, page + 1));
        CHECK(!p::huge_on(-1, {}, 1));
        CHECK(!p::huge_on(0, pool, 1));
        CHECK(!p::huge_on(0, pool, 3 * page));
        CHECK(p::huge_on(1, {}, 1));
        CHECK(p::huge_on(1, pool, 3 * page));
        CHECK(!p::huge_on(-1, pool, 0));
        CHECK(!p::huge_on(0, pool, 0));
        CHECK(!p::huge_on(1, pool, 0));
    }
    {   // The up-front pool check refuses an absurd block without building any pieces.
#ifdef __linux__
        const int fds = open_fds(), maps = huge_file_maps();
        fill_calls = 0;
        int fd = -2;
        size_t got = 123;
        void* m = p::huge_alloc_pieces((size_t) (uint64_t{1} << 46), -1, 1, &got,
                                     p::kHugePiece, 1, 0, record_fill, &fd);
        CHECK(fd == -1);
        CHECK(fill_calls == 0);
        if (fd >= 0) CHECK(::close(fd) == 0);
#else
        size_t got = 123;
        void* m = p::huge_alloc_pieces((size_t) (uint64_t{1} << 46), -1, 1, &got, p::kHugePiece, 1, 0);
#endif
        CHECK(m == nullptr);
        CHECK(got == 0);
        if (m != nullptr) p::huge_free(m, got);
#ifdef __linux__
        CHECK(open_fds() == fds);
        CHECK(huge_file_maps() == maps);
#endif
    }
#ifdef __linux__
    if (p::huge_pool_2m().total() >= 8 * page) {
        size_t got = 0;
        constexpr size_t bytes = 3 * 1024 * 1024;
        void* m = p::huge_alloc(bytes, -1, 1, &got);
        CHECK(m != nullptr);
        if (m != nullptr) {
            CHECK(got == 2 * page);
            auto* bytes = (volatile unsigned char*) m;
            bytes[0] = 0x5a;
            bytes[got - 1] = 0xa5;
            CHECK(bytes[0] == 0x5a && bytes[got - 1] == 0xa5);
            CHECK(kernel_2m(m));
            p::huge_free(m, got);
            std::printf("huge pages: allocated 3 MiB on 4 MiB of explicit 2 MB pages\n");
        }
        // Even on a one-node host, both requested policies must leave the calling thread unchanged.
        int saved_mode = -1;
        unsigned long saved_mask[4] = {0, 0, 0, 0};
        const bool have_policy = thread_policy(&saved_mode, saved_mask);
        m = p::huge_alloc(bytes, 0, 2, &got);
        if (have_policy) check_thread_policy(saved_mode, saved_mask);
        CHECK(m != nullptr);
        if (m != nullptr) {
            check_policy(m, 1 /* MPOL_PREFERRED */);
            p::huge_free(m, got);
        }
        m = p::huge_alloc(bytes, p::kHugeInterleave, 2, &got);
        if (have_policy) check_thread_policy(saved_mode, saved_mask);
        CHECK(m != nullptr);
        if (m != nullptr) {
            check_policy(m, 3 /* MPOL_INTERLEAVE */);
            p::huge_free(m, got);
        }
    } else std::printf("skipped: no 2 MB pool\n");
    const auto pool = p::huge_pool_2m();
    if (pool.total() >= 8 * page) {
        constexpr size_t bytes = 10 * 1024 * 1024, piece = 4 * 1024 * 1024;
        const int fds = open_fds(), maps = huge_file_maps();
        int fd = -2;
        size_t got = 0;
        void* m = p::huge_alloc_pieces(bytes, -1, 1, &got, piece, 2, 0, nullptr, &fd);
        CHECK(m != nullptr);
        if (m != nullptr) {
            CHECK(got == bytes);
            if (fd >= 0) {
                struct stat info{};
                CHECK(::fstat(fd, &info) == 0);
                CHECK((uint64_t) info.st_blocks * 512 == got);  // Before any write could fault pages in.
            } else std::printf("skipped: pages-present-before-return check (anonymous fallback)\n");
            auto* data = (volatile unsigned char*) m;
            for (size_t i = 0; i < bytes / page; ++i) data[i * page] = (unsigned char) (0x40 + i);
            for (size_t i = 0; i < bytes / page; ++i) CHECK(data[i * page] == (unsigned char) (0x40 + i));
            p::huge_free(m, got);
        }
        if (fd >= 0) CHECK(::close(fd) == 0);
        CHECK(open_fds() == fds);
        CHECK(huge_file_maps() == maps);

        // A built first piece must be released when every try of the second piece is refused.
        fill_calls = 0;
        fd = -2;
        got = 123;
        m = p::huge_alloc_pieces(bytes, -1, 1, &got, piece, 2, 0, fill_first_piece, &fd);
        CHECK(m == nullptr);
        CHECK(got == 0);
        CHECK(fill_calls >= 3);
        CHECK(fd == -1);
        if (m != nullptr) p::huge_free(m, got);
        if (fd >= 0) CHECK(::close(fd) == 0);
        CHECK(open_fds() == fds);
        CHECK(huge_file_maps() == maps);

        // Interrupted fills do not spend the one available try.
        fill_calls = 0;
        fd = -2;
        got = 0;
        m = p::huge_alloc_pieces((size_t) page, -1, 1, &got, piece, 1, 0, fill_after_eintr, &fd);
        CHECK(m != nullptr);
        CHECK(got == page);
        CHECK(fill_calls >= 3);
        if (m != nullptr) p::huge_free(m, got);
        if (fd >= 0) CHECK(::close(fd) == 0);
        CHECK(open_fds() == fds);
        CHECK(huge_file_maps() == maps);

        const pid_t child = ::fork();
        CHECK(child >= 0);
        if (child == 0) {
            const struct rlimit limit{1024 * 1024, 1024 * 1024};
            if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(1);
            size_t child_got = 0;
            void* block = p::huge_alloc(4 * 1024 * 1024, -1, 1, &child_got);
            if (block != nullptr) {
                if (child_got != 4 * 1024 * 1024) _exit(2);
                auto* data = (volatile unsigned char*) block;
                data[0] = 0x5a;
                data[child_got - 1] = 0xa5;
                if (data[0] != 0x5a || data[child_got - 1] != 0xa5) _exit(3);
                p::huge_free(block, child_got);
            } else if (child_got != 0) _exit(4);
            _exit(0);
        }
        if (child > 0) {
            int status = 0;
            pid_t waited;
            do {
                waited = ::waitpid(child, &status, 0);
            } while (waited < 0 && errno == EINTR);
            CHECK(waited == child);
            CHECK(WIFEXITED(status));
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
    } else std::printf("skipped: piecewise tests need at least 8 huge pages\n");
    {   // A representable block above the pool's room is cheap to refuse for any pool size (the margin is wide so a
        // pool that grows between the test's read and the allocator's cannot turn this into a real block).
        const int fds = open_fds(), maps = huge_file_maps();
        fill_calls = 0;
        int fd = -2;
        size_t got = 123;
        void* m = p::huge_alloc_pieces((size_t) (p::huge_pool_2m().total() + (uint64_t{64} << 30)),
                                     -1, 1, &got, p::kHugePiece, 1, 0, record_fill, &fd);
        CHECK(m == nullptr);
        CHECK(got == 0);
        CHECK(fd == -1);
        CHECK(fill_calls == 0);
        if (m != nullptr) p::huge_free(m, got);
        if (fd >= 0) CHECK(::close(fd) == 0);
        CHECK(open_fds() == fds);
        CHECK(huge_file_maps() == maps);
    }
#else
    CHECK(p::huge_pool_2m().total() == 0);
    size_t got = 0;
    CHECK(p::huge_alloc((size_t) page, -1, 1, &got) == nullptr);
    std::printf("skipped: no 2 MB pool (not Linux)\n");
#endif
    if (fails == 0) std::printf("huge page rules: ok\n");
    return fails == 0 ? 0 : 1;
}
