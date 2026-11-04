// Windows regression: copied source pages leave the process working set without losing mapped bytes.
#if !defined(_WIN32)
#error This test is Windows-only.
#endif
#include "../src/ds4/ds4_os.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct WSX {
    void* address;
    size_t attributes;
};

static size_t resident_pages(const uint8_t* base, size_t pages) {
    std::vector<WSX> rows(pages);
    for (size_t i = 0; i < pages; ++i) rows[i] = {(void*) (base + i * 4096), 0};
    check(K32QueryWorkingSetEx(GetCurrentProcess(), rows.data(), (DWORD) (rows.size() * sizeof(WSX))) != 0,
          "query working set");
    size_t n = 0;
    for (const WSX& row : rows) n += row.attributes & 1;
    return n;
}

int main() {
    using namespace strata::ds4::os;
    const std::string path = "ds4-copy-cache-windows-test.bin";
    constexpr size_t page = 4096, pages = 4096, bytes = page * pages;
    std::vector<uint8_t> original(bytes), copy(bytes);
    for (size_t i = 0; i < bytes; ++i) original[i] = (uint8_t) ((i * 37 + i / 503) % 251);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        check(out.good(), "create fixture");
        out.write((const char*) original.data(), original.size());
        check(out.good(), "write fixture");
    }
    fd_t fd = kBadFd;
    void* view = nullptr;
    try {
        fd = open_ro(path);
        check(fd != kBadFd, "open fixture");
        view = map_ro(fd, bytes);
        check(view != nullptr, "map fixture");
        auto* base = static_cast<const uint8_t*>(view);
        std::memcpy(copy.data(), view, bytes);
        size_t checksum = 0;
        for (size_t i = 0; i < pages; ++i) checksum += base[i * page];
        check(checksum != 0, "touch every page");
        const size_t before = resident_pages(base, pages);
        check(before >= pages * 95 / 100, "fixture resident before release");
        const uint64_t released = release_copied_cache(fd, view, 1, bytes - 2);
        check(released == bytes - 2 * page, "only inward-aligned whole pages released");
        const size_t after = resident_pages(base, pages);
        check(after <= 2, "interior pages removed from working set");
        check(std::memcmp(view, original.data(), bytes) == 0, "mapped contents survive re-fault");
        check(copy == original, "independent expert copy unchanged");
        check(release_copied_cache(fd, view, 1, page - 2) == 0, "partial-page range ignored");
        check(release_copied_cache(fd, view, 0, 0) == 0, "zero range ignored");
        check(release_copied_cache(fd, nullptr, 0, page) == 0, "missing mapping rejected");
        check(release_copied_cache(kBadFd, view, 0, page) == 0, "invalid fd rejected");
        check(release_copied_cache(fd, view, UINT64_MAX - 1, 16) == 0, "overflow rejected");
        // unlock_range: the same demotion straight from a mapped pointer, interior pages only
        for (size_t i = 0; i < pages; ++i) checksum += base[i * page];   // (resident again after the release above)
        check(resident_pages(base, pages) >= pages * 95 / 100, "fixture resident before unlock_range");
        check(unlock_range(base + 1, bytes - 2) == bytes - 2 * page, "unlock_range releases interior pages");
        check(resident_pages(base, pages) <= 2, "unlock_range leaves the edge pages");
        check(std::memcmp(view, original.data(), bytes) == 0, "contents identical after unlock_range re-fault");
        check(unlock_range(view, page - 2) == 0, "unlock_range ignores a sub-page range");
        check(unlock_range(nullptr, bytes) == 0, "unlock_range ignores a null range");
        // advise(!need): demotes through the fd -> view registry map_ro keeps (map_ro registered this fd's view)
        for (size_t i = 0; i < pages; ++i) checksum += base[i * page];
        check(resident_pages(base, pages) >= pages * 95 / 100, "fixture resident before advise");
        advise(fd, 1, bytes - 2, false);
        check(resident_pages(base, pages) <= 2, "advise(!need) demotes the mapped range");
        check(std::memcmp(view, original.data(), bytes) == 0, "contents identical after advise re-fault");
        advise(fd, 1, bytes - 2, true);   // (a need hint: nothing to do on Windows, and nothing breaks)
        advise(kBadFd, 0, page, false);   // (an fd with no view: a harmless no-op)
        check(working_set_bytes() > 0, "working_set_bytes reports this process's resident set");
        unmap(view, bytes);
        view = nullptr;
        close_fd(fd);
        fd = kBadFd;
        std::remove(path.c_str());
        std::cout << "PASS: resident pages " << before << " -> " << after
                  << "; edge pages retained; mapped bytes identical after re-fault\n";
        return 0;
    } catch (...) {
        if (view != nullptr) unmap(view, bytes);
        if (fd != kBadFd) close_fd(fd);
        std::remove(path.c_str());
        throw;
    }
}
