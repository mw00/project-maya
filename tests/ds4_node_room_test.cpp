// ds4_node_room_test: the room maya-ds4 makes on a NUMA node for its part of the CPU lane's copies
// (src/ds4/ds4_os.hpp: node_room_short's arithmetic, node_mem's read of the node's meminfo, node_make_room's no-ops).
// Host code only, no model, no GPU.  Build: g++ -std=c++17 -O2 -pthread -I include tests/ds4_node_room_test.cpp
#include "../src/ds4/ds4_os.hpp"

#include <cstdint>
#include <cstdio>

namespace os = maya::ds4::os;

namespace {

int failures = 0;
#define EXPECT(c)                                                            \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

constexpr uint64_t GB = (uint64_t) 1 << 30, MB = (uint64_t) 1 << 20;

}  // namespace

int main() {
    const uint64_t slack = 1 * GB, least = 256 * MB;
    os::NodeMem m;
    // the node holds the bytes already: nothing
    m.free = 80 * GB, m.file = 100 * GB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 0);
    m.free = 72 * GB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 0);   // (need + slack exactly)
    // short, the cache holds more than is missing: what is missing
    m.free = 2 * GB, m.file = 120 * GB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 70 * GB);
    // short, the cache holds less: nine tenths of it, never all
    m.free = 2 * GB, m.file = 40 * GB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 36 * GB);
    // nothing reclaimable (the node full of memory a reclaim cannot take, #507): nothing asked of it
    m.free = 2 * GB, m.file = 0;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 0);
    // less than a pass is worth
    m.free = 71 * GB + 900 * MB, m.file = 50 * GB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 0);
    m.free = 2 * GB, m.file = 200 * MB;
    EXPECT(os::node_room_short(m, 71 * GB, slack, least) == 0);

#if !defined(_WIN32)
    // this host's node 0, where the kernel says what it holds: free memory, and a no-op for bytes it has free
    os::NodeMem n0;
    if (os::node_mem(0, n0)) {
        EXPECT(n0.free > 0);
        if (n0.free > 2 * GB) EXPECT(os::node_make_room(0, 16 * MB, 2) == 0);
        std::printf("node 0: %.1f GB free, %.1f GB of reclaimable file cache\n", (double) n0.free / GB, (double) n0.file / GB);
    } else {
        std::printf("node 0's meminfo cannot be read here: the read is skipped\n");
    }
    EXPECT(!os::node_mem(100000, n0));              // no such node
    EXPECT(os::node_make_room(-1, 64 * GB, 2) == 0);   // no node
    EXPECT(os::node_make_room(100000, 64 * GB, 2) == 0);
#endif
    if (failures != 0) {
        std::fprintf(stderr, "ds4_node_room_test: %d failed\n", failures);
        return 1;
    }
    std::printf("ds4_node_room_test: ok\n");
    return 0;
}
