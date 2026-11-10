#include "slot_index.hpp"
#include <cassert>
#include <fstream>
#include <unistd.h>
using namespace maya::metal;
int main() {
    char dir[] = "/tmp/slot-index-XXXXXXXX";
    assert(mkdtemp(dir));
    {
        SlotIndex slots(dir, 2, 6);
        auto save = [](const char* s) { return [s](const auto& p) { std::ofstream f(p, std::ios::binary); f << s; f.close(); return !f.fail(); }; };
        auto ih = [](size_t n) { return uint64_t(n); };
        size_t restored_pos = 0;
        auto ok = [&](const auto& p, size_t pos) { restored_pos = pos; std::ifstream f(p); return f.good(); };
        assert(slots.put({1,2}, 2, 3, save("abc")));
        assert(slots.put({1,2,3}, 3, 3, save("def")));
        assert(slots.restore({1,2,3,4}, 4, ih, ok) == 3);
        assert(restored_pos == 3);
        assert(slots.restore({1,2,3,4}, 2, ih, ok) == 2); // refresh LRU
        assert(slots.put({4,5}, 2, 3, save("ghi")));
        assert(slots.restore({1,2,3,4}, 4, ih, ok) == 2); // longest evicted
        assert(slots.restore({4,5,6}, 3, [](size_t){return 0;}, ok) == 0); // image mismatch
        assert(!slots.put({9}, 1, 7, save("1234567")));
        assert(slots.restore({4,5,6}, 3, ih, [](const auto&, size_t){return false;}) == 0);
        assert(slots.restore({4,5,6}, 3, ih, ok) == 0); // failed restore dropped
    }
    assert(std::filesystem::is_empty(dir));
    std::filesystem::remove(dir);
}
