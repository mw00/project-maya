#include "slot_state.hpp"
#include <cassert>
#include <cstring>
#include <limits>
#include <unistd.h>
using namespace maya::metal;
int main() {
    char path[] = "/tmp/glm-slot-state-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); close(fd);
    std::vector<unsigned char> a(1100000, 17), b{1,2,3}, dst(a.size(), 0), dst2(3, 0);
    auto block = [](auto& src, auto& out) -> SlotBlock {
        return {(uint64_t) src.size(),
            [&](uint64_t off, void* p, size_t n) { std::memcpy(p, src.data() + off, n); return true; },
            [&](uint64_t off, const void* p, size_t n) { std::memcpy(out.data() + off, p, n); return true; }};
    };
    std::vector<SlotBlock> blocks{block(a, dst), block(b, dst2)};
    assert(slot_state_save(path, 120, blocks, 2000000));
    assert(!slot_state_load(path, 121, blocks, 2000000));
    assert(slot_state_load(path, 120, blocks, 2000000));
    assert(a == dst && b == dst2);
    assert(!slot_state_save(path, 120, blocks, 100));
    FILE* f = std::fopen(path, "r+b"); assert(f); assert(!std::fseek(f, 64, SEEK_SET)); assert(std::fputc(99, f) != EOF); assert(!std::fclose(f));
    dst.assign(dst.size(), 0); dst2.assign(dst2.size(), 0);
    assert(!slot_state_load(path, 120, blocks, 2000000));
    assert(dst[0] == 0 && dst2[0] == 0);
    assert(slot_state_save(path, 120, blocks, 2000000));
    assert(!truncate(path, 100));
    assert(!slot_state_load(path, 120, blocks, 2000000));
    assert(dst[0] == 0 && dst2[0] == 0);
    assert(slot_state_save(path, 120, blocks, 2000000));
    auto failing = blocks;
    failing[1].write = [](uint64_t, const void*, size_t) { return false; };
    assert(!slot_state_load(path, 120, failing, 2000000)); // caller must reset after apply failure
    std::vector<SlotBlock> huge{{std::numeric_limits<uint64_t>::max(), {}, {}}};
    assert(!slot_state_save(path, 1, huge, std::numeric_limits<uint64_t>::max()));
    assert(!unlink(path));
}
