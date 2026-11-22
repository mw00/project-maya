// src/hy4/hy4_engram.hpp - the Engram table's two lookup switches (HY4_EG_RANDOM, HY4_EG_THREADS).  No CUDA here:
// tests/hy4_engram_test.cpp checks the choices without a GPU.
#pragma once

#include <algorithm>
#include <cstdlib>

namespace hy4::engram {

// on unless explicitly off: a row is small and at a hashed offset, so the kernel's readahead window is mostly bytes
// nobody asked for (the DS4 reader's MAYA_DS4_EG_RANDOM is opt-in because its page cache is usually large enough).
inline bool random_enabled(const char* v) { return v == nullptr || *v == 0 || std::atoi(v) != 0; }

// the lookups' queue depth: small random reads, so more threads help until the disk or the mapping is the limit.
inline int threads(const char* v) {
    return std::min(64, std::max(1, v == nullptr || *v == 0 ? 8 : std::atoi(v)));
}

}  // namespace hy4::engram
