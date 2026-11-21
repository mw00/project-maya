// maya-hy4's HY4_EG_POPULATE (#1069): src/mm3/mm3_os.hpp's populate_mapped walks a read-only file mapping's pages on
// every platform (it was compiled out off Linux, so the setting did nothing on Windows) - the whole range, an
// unaligned start, a range not a whole number of chunks, and nothing once stop is set.  No model, no GPU.
#include "../src/mm3/mm3_os.hpp"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "maya-hy4-populate-test.bin";
    const size_t n = ((size_t) 5 << 20) + 12345;   // (5 MiB and a bit: not a whole number of chunks or pages)
    {
        std::vector<char> data(n);
        for (size_t i = 0; i < n; ++i) data[i] = (char) ((i * 37 + i / 503) % 251);
        std::ofstream f(path, std::ios::binary);
        f.write(data.data(), (std::streamsize) n);
        CHECK(f.good());
    }
    size_t bytes = 0;
    const uint8_t* m = mm3::os::map_file(path.string(), bytes);
    CHECK(m != nullptr);
    CHECK(bytes == n);

    std::atomic<bool> stop{false};
    // the whole mapping, in 1 MiB chunks and in the default 256 MiB one
    CHECK(mm3::os::populate_mapped(m, n, stop, (size_t) 1 << 20) == n);
    CHECK(mm3::os::populate_mapped(m, n, stop) == n);
    // a start inside a page (the table's tensor need not begin on a page boundary)
    CHECK(mm3::os::populate_mapped(m + 100, n - 100, stop, (size_t) 1 << 20) == n - 100);
    // the bytes are the file's after the walk (it only reads)
    CHECK(m[0] == 0 && m[n - 1] == (uint8_t) (((n - 1) * 37 + (n - 1) / 503) % 251));
    // stop set before the start: nothing walked (the engine's unload sets it, the thread ends at once)
    stop.store(true);
    CHECK(mm3::os::populate_mapped(m, n, stop) == 0);
    CHECK(mm3::os::populate_mapped(m + 100, n - 100, stop) == 0);
    // nothing to walk
    stop.store(false);
    CHECK(mm3::os::populate_mapped(nullptr, n, stop) == 0);
    CHECK(mm3::os::populate_mapped(m, 0, stop) == 0);

    mm3::os::sys::unmap((void*) m, (uint64_t) n);
    std::error_code ec;
    fs::remove(path, ec);
    std::printf("hy4_populate_test: ok\n");
    return 0;
}
