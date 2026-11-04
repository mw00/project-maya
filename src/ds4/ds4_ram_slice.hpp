#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace strata::ds4 {
struct RamSlice {
    int part;
    size_t source, destination, bytes;
};

// The RAM copy is split into equal contiguous row ranges on NH NUMA nodes.
// Reassemble only [offset, offset + bytes) into a packed staging region.
inline int ram_slices(size_t part_bytes, int parts, size_t offset, size_t bytes,
                      std::array<RamSlice, 2>& out) {
    if (parts < 1 || parts > 2 || offset > part_bytes * parts || bytes > part_bytes * parts - offset)
        return -1;
    int count = 0;
    for (int part = 0; part < parts; ++part) {
        const size_t begin = std::max(offset, (size_t) part * part_bytes);
        const size_t end = std::min(offset + bytes, (size_t) (part + 1) * part_bytes);
        if (begin < end)
            out[count++] = {part, begin - (size_t) part * part_bytes, begin - offset, end - begin};
    }
    return count;
}
}  // namespace strata::ds4
