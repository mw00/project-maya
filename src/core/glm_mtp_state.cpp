// src/core/glm_mtp_state.cpp - glm-dsa B15.3: the NextN carrier's bookkeeping, the MTP1 slot trailer, the block
// identity and the draft budget (include/strata/core/glm_mtp_state.hpp).  Plain C++.
#include "strata/core/glm_mtp_state.hpp"

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>

namespace strata::core::glmmtp {

int draft_budget(int n_max, int64_t remaining_predict, int64_t remaining_ctx) {
    const int64_t b = std::min<int64_t>({(int64_t) n_max, remaining_predict, remaining_ctx});
    return b > 0 ? (int) b : 0;
}

bool CarrierState::use(int64_t p) {
    ++reads;
    if (pos == p && p >= 0) return true;
    ++zero_reads;
    if (pos >= 0) ++stale_reads;
    pos = -1;   // the row is zeroed by the caller (a stale row never survives a read)
    return false;
}

void HostCarrier::begin() {
    st.begin();
    std::fill(row.begin(), row.end(), 0.0f);
}

void HostCarrier::note(int64_t p, const float* h) {
    st.note(p);
    std::copy(h, h + row.size(), row.begin());
}

const float* HostCarrier::read(int64_t p) {
    if (!st.use(p)) std::fill(row.begin(), row.end(), 0.0f);
    return row.data();
}

void HostCarrier::snapshot_save(int64_t n) {
    st.snapshot_save(n);
    if (st.snap_pos >= 0) snap = row;
    else std::fill(snap.begin(), snap.end(), 0.0f);
}

void HostCarrier::snapshot_restore() {
    st.snapshot_restore();
    row = snap;
}

namespace {

constexpr uint64_t kFnvBasis = 1469598103934665603ull, kFnvPrime = 1099511628211ull;

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

}  // namespace

uint32_t block_id(std::vector<BlockTensor> tensors, const std::string& source) {
    std::sort(tensors.begin(), tensors.end(), [](const BlockTensor& a, const BlockTensor& b) { return a.name < b.name; });
    uint64_t h = kFnvBasis;
    for (const BlockTensor& t : tensors) {
        h = fnv(h, t.name.data(), t.name.size());
        h = fnv(h, "\0", 1);
        const int32_t ty = t.type;
        h = fnv(h, &ty, 4);
        h = fnv(h, &t.bytes, 8);
    }
    h = fnv(h, source.data(), source.size());
    uint32_t w = (uint32_t) ((h ^ (h >> 31) ^ (h >> 62)) & 0x7fffffffu);
    return w == 0 ? 1u : w;
}

size_t trailer_bytes(int n_embd) { return 4 + 4 + 8 + 4 + 4 + (size_t) n_embd * 4 + 8; }

bool write_trailer(std::ostream& f, int n_embd, int64_t pos, uint32_t block, const float* row) {
    const uint32_t magic = kTrailerMagic, zero = 0;
    const int32_t ne = n_embd;
    f.write((const char*) &magic, 4);
    f.write((const char*) &ne, 4);
    f.write((const char*) &pos, 8);
    f.write((const char*) &block, 4);
    f.write((const char*) &zero, 4);
    f.write((const char*) row, (std::streamsize) n_embd * 4);
    const uint64_t sum = fnv(kFnvBasis, row, (size_t) n_embd * 4);
    f.write((const char*) &sum, 8);
    return (bool) f;
}

Trailer read_trailer(std::istream& f, int n_embd, int64_t n, uint32_t block, int64_t& pos, std::vector<float>& row,
                     std::string& why) {
    uint32_t magic = 0;
    if (!f.read((char*) &magic, 4)) {
        if (f.gcount() == 0) {
            f.clear();
            return Trailer::Absent;
        }
        why = "a truncated MTP1 trailer";
        return Trailer::Bad;
    }
    int32_t ne = 0;
    int64_t p = 0;
    uint32_t blk = 0, z = 0;
    if (magic != kTrailerMagic) {
        why = "not an MTP1 trailer";
        return Trailer::Bad;
    }
    if (!f.read((char*) &ne, 4) || !f.read((char*) &p, 8) || !f.read((char*) &blk, 4) || !f.read((char*) &z, 4)) {
        why = "a truncated MTP1 trailer";
        return Trailer::Bad;
    }
    if (ne != n_embd || n_embd <= 0) {
        why = "the MTP1 trailer's row is " + std::to_string(ne) + " wide, the model's " + std::to_string(n_embd);
        return Trailer::Bad;
    }
    if (blk != block) {
        why = "the MTP1 trailer names another NextN block";
        return Trailer::Bad;
    }
    if (z != 0 || !(p == -1 || p == n - 1)) {
        why = "the MTP1 trailer's position " + std::to_string(p) + " is not -1 or " + std::to_string(n - 1);
        return Trailer::Bad;
    }
    std::vector<float> r((size_t) n_embd);
    uint64_t sum = 0;
    if (!f.read((char*) r.data(), (std::streamsize) n_embd * 4) || !f.read((char*) &sum, 8)) {
        why = "a truncated MTP1 trailer";
        return Trailer::Bad;
    }
    if (sum != fnv(kFnvBasis, r.data(), (size_t) n_embd * 4)) {
        why = "the MTP1 trailer's checksum differs";
        return Trailer::Bad;
    }
    pos = p;
    row.swap(r);
    return Trailer::Ok;
}

}  // namespace strata::core::glmmtp
