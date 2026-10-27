#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_ram_slice.hpp"
#include <cassert>
#include <cstring>
#include <random>
#include <vector>
#include <cstdio>

int main() {
    std::mt19937 rng(20261019);
    for (int parts : {1, 2}) {
        for (int members : {1, 2, 4}) {
            for (int trial = 0; trial < 100; ++trial) {
                const size_t rows = 128 * (1 + rng() % 16), stride = 18 * (1 + rng() % 16);
                std::vector<unsigned char> matrix(rows * stride);
                for (auto& byte : matrix) byte = (unsigned char) rng();
                const size_t pb = matrix.size() / parts;
                for (int member = 0; member < members; ++member) {
                    // Gate/up select contiguous rows. Down requires the whole matrix
                    // in staging before the GPU's strided column-slice copy.
                    for (bool down : {false, true}) {
                        const size_t length = down ? matrix.size() : matrix.size() / members;
                        const size_t offset = down ? 0 : member * length;
                        std::vector<unsigned char> staged(length + 32, 0xa5);
                        std::array<strata::ds4::RamSlice, 2> slices;
                        int n = strata::ds4::ram_slices(pb, parts, offset, length, slices);
                        assert(n > 0 && n <= parts);
                        size_t copied = 0;
                        for (int i = 0; i < n; ++i) {
                            const auto& s = slices[i];
                            assert(s.source + s.bytes <= pb);
                            assert(s.destination + s.bytes <= length);
                            std::memcpy(staged.data() + 16 + s.destination,
                                        matrix.data() + s.part * pb + s.source, s.bytes);
                            copied += s.bytes;
                        }
                        assert(copied == length);
                        assert(std::memcmp(staged.data() + 16, matrix.data() + offset, length) == 0);
                        for (int i = 0; i < 16; ++i)
                            assert(staged[i] == 0xa5 && staged[length + 16 + i] == 0xa5);
                    }
                }
            }
        }
    }
    std::array<strata::ds4::RamSlice, 2> slices;
    assert(strata::ds4::ram_slices(32, 2, 60, 8, slices) == -1);
    assert(strata::ds4::ram_slices(32, 2, 0, 0, slices) == 0);
    std::puts("PASS: packed gate/up and full down staging, TP 1/2/4, NUMA 1/2, guards and bounds");
}
