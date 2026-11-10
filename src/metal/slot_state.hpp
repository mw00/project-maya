#pragma once
// Bounded, checksummed streaming state transfer, independent of the tensor backend.
// Each block is a contiguous logical state region; callbacks transfer a slice.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

namespace maya::metal {
struct SlotBlock {
    uint64_t bytes;
    std::function<bool(uint64_t, void*, size_t)> read;
    std::function<bool(uint64_t, const void*, size_t)> write;
};
namespace slot_detail {
constexpr uint64_t magic = 0x31544154534d4c47ull;
inline uint64_t hash(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    while (n--) h = (h ^ *p++) * 1099511628211ull;
    return h;
}
inline bool total(const std::vector<SlotBlock>& blocks, uint64_t& n) {
    if (blocks.size() > 65536) return false;
    n = 4 * sizeof(uint64_t) + blocks.size() * sizeof(uint64_t);
    for (const auto& b : blocks) {
        if (b.bytes > std::numeric_limits<uint64_t>::max() - n) return false;
        n += b.bytes;
    }
    return true;
}
inline bool io(FILE* f, void* p, size_t n, bool writing, uint64_t& h) {
    if ((writing ? std::fwrite(p, 1, n, f) : std::fread(p, 1, n, f)) != n) return false;
    h = hash(h, p, n);
    return true;
}
// First pass validates every byte before the second pass modifies the backend.
inline bool transfer(FILE* f, uint64_t pos, const std::vector<SlotBlock>& blocks, bool saving, bool apply) {
    uint64_t h = 1469598103934665603ull;
    uint64_t header[] = {magic, pos, (uint64_t) blocks.size()};
    for (auto& expected : header) {
        uint64_t v = expected;
        if (!io(f, &v, sizeof v, saving, h) || v != expected) return false;
    }
    for (const auto& b : blocks) {
        uint64_t n = b.bytes;
        if (!io(f, &n, sizeof n, saving, h) || n != b.bytes) return false;
    }
    std::vector<unsigned char> buf(1024 * 1024);
    for (const auto& b : blocks) {
        for (uint64_t off = 0; off < b.bytes;) {
            size_t n = (size_t) std::min<uint64_t>(buf.size(), b.bytes - off);
            if (saving && (!b.read || !b.read(off, buf.data(), n))) return false;
            if (!io(f, buf.data(), n, saving, h)) return false;
            if (apply && (!b.write || !b.write(off, buf.data(), n))) return false;
            off += n;
        }
    }
    uint64_t check = h;
    if (saving) return std::fwrite(&check, 1, sizeof check, f) == sizeof check;
    return std::fread(&check, 1, sizeof check, f) == sizeof check && check == h && std::fgetc(f) == EOF && !std::ferror(f);
}
}
// Temporary path is private to SlotIndex; the caller removes a failed file.
inline bool slot_state_size(const std::vector<SlotBlock>& blocks, uint64_t& bytes) {
    return slot_detail::total(blocks, bytes);
}
inline bool slot_state_save(const char* path, uint64_t pos, const std::vector<SlotBlock>& blocks, uint64_t max_bytes) {
    uint64_t n;
    if (!pos || !slot_detail::total(blocks, n) || n > max_bytes) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = slot_detail::transfer(f, pos, blocks, true, false);
    if (std::fclose(f) != 0) ok = false;
    return ok;
}
// On I/O failure during the application pass, the caller MUST reset the model.
inline bool slot_state_load(const char* path, uint64_t pos, const std::vector<SlotBlock>& blocks, uint64_t max_bytes) {
    uint64_t n;
    if (!pos || !slot_detail::total(blocks, n) || n > max_bytes) return false;
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = slot_detail::transfer(f, pos, blocks, false, false) && std::fseek(f, 0, SEEK_SET) == 0 &&
              slot_detail::transfer(f, pos, blocks, false, true);
    if (std::fclose(f) != 0) ok = false;
    return ok;
}
}
