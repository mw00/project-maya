// maya-ds4's Engram reader (src/ds4/ds4_engram_io.hpp): a batch read ahead (submit low) waits behind the others - a
// prompt's next part's rows behind the decode's - and every read gives the file's bytes either way.  A file in the
// temp folder, the reader's own threads; no model, no GPU.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_engram_io.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <vector>

using strata::ds4::EgBatch;
using strata::ds4::EgRead;
using strata::ds4::EgReader;

namespace {

constexpr uint32_t kRow = 64;
std::vector<uint8_t> data;

// n rows of the file into dst, row i from a hashed offset
void fill(EgBatch& b, std::vector<uint8_t>& dst, int n, uint64_t salt) {
    dst.assign((size_t) n * kRow, 0);
    b.clear();
    for (int i = 0; i < n; ++i) {
        EgRead r;
        r.file = 0;
        r.off = ((uint64_t) i * 2654435761u + salt) % (data.size() - kRow);
        r.n = kRow;
        r.dst = dst.data() + (size_t) i * kRow;
        b.reads.push_back(r);
    }
}
bool same(const EgBatch& b) {
    for (size_t i = 0; i < b.reads.size(); ++i)
        if (!b.ok[i] || std::memcmp(b.reads[i].dst, data.data() + b.reads[i].off, kRow) != 0) return false;
    return true;
}

}  // namespace

int main() {
    char name[] = "/tmp/maya-engram-reader-XXXXXX";
    const int fd = mkstemp(name);
    assert(fd >= 0);
    close(fd);
    data.resize(4 << 20);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) ((i * 131) ^ (i >> 9));
    {
        std::ofstream f(name, std::ios::binary);
        f.write((const char*) data.data(), (std::streamsize) data.size());
    }
    // one thread: a large batch read ahead, then a small one asked for - the small one is done first (it waits for at
    // most the job the thread is on), the large one after it; both the file's bytes
    {
        EgReader rd;
        rd.start({name}, 1);
        auto ahead = std::make_unique<EgBatch>();
        auto now = std::make_unique<EgBatch>();
        std::vector<uint8_t> a, b;
        fill(*ahead, a, 20000, 7);
        fill(*now, b, 64, 99);
        rd.submit(*ahead, true);
        rd.submit(*now);
        rd.wait(*now);
        rd.wait(*ahead);
        assert(same(*now) && same(*ahead));
        assert(now->t1.load() <= ahead->t1.load());
        rd.stop();
    }
    // several threads: batches read ahead alone (nothing else queued) are read at once; mixed ones all land
    {
        EgReader rd;
        rd.start({name}, 4);
        std::vector<std::unique_ptr<EgBatch>> bs;
        std::vector<std::vector<uint8_t>> dst(6);
        for (int i = 0; i < 6; ++i) {
            bs.emplace_back(new EgBatch());
            fill(*bs.back(), dst[(size_t) i], 3000 + 500 * i, (uint64_t) i * 1000003u);
            rd.submit(*bs.back(), i % 2 == 0);
        }
        for (auto& b : bs) rd.wait(*b);
        for (auto& b : bs) assert(same(*b));
        auto alone = std::make_unique<EgBatch>();
        std::vector<uint8_t> c;
        fill(*alone, c, 5000, 12345);
        rd.submit(*alone, true);
        rd.wait(*alone);
        assert(same(*alone));
        auto none = std::make_unique<EgBatch>();   // (an empty batch: done at once, either queue)
        rd.submit(*none, true);
        rd.wait(*none);
        rd.stop();
    }
    std::remove(name);
    std::printf("ds4_engram_reader_test: all checks passed\n");
    return 0;
}
