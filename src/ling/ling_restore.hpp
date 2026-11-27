// src/ling/ling_restore.hpp - Validated, model-keyed, size-capped disk restore points (".lingrp" v2).
// Pure C++20 header (zero CUDA dependency) so every validation, model-key, and cap-eviction path is unit-tested
// directly without a GPU or GGUF model file.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ling::sys {

static constexpr char kRestoreMagic[4] = {'L', 'R', 'P', '2'};
static constexpr uint32_t kRestoreFormatVersion = 2;
static constexpr const char* kRestoreEngineVersion = "maya-ling-v2";
static constexpr uint32_t kRestoreMaxStrLen = 256;
static constexpr uint32_t kRestoreMaxPics = 1024;

struct RestoreKey {
    std::string model_id;           // GGUF basename / model identifier
    std::string quant_tag;          // tensor quant signature (e.g. "etg=12,etd=12,emb=8,out=8")
    std::string engine_version;     // engine/format compatibility tag
    uint64_t gguf_fingerprint = 0;  // FNV-1a of every shard's size + first/last 64 KiB + 128 spaced 4 KiB samples, then
                                    // the tensor table (names, types, offsets, sizes - mix_tensor_table)
    uint32_t ctx = 0;               // configured max context length
    uint32_t kv_q8 = 0;             // 0 = fp16 KV, 1 = q8 KV
};

struct RestoreExpect {
    RestoreKey key;
    uint32_t n_kda_layers = 0;
    uint32_t n_mla_layers = 0;
    uint32_t vocab = 0;
    uint64_t kda_state_bytes = 0;
    uint64_t kda_conv_bytes = 0;
    uint64_t mla_lat_row_bytes = 0;
    uint64_t mla_kpe_row_bytes = 0;
};

struct RestoreSnapshotData {
    std::vector<int> tokens;
    std::vector<std::pair<int64_t, uint64_t>> pics;
    int rope_delta = 0;
    bool kv_q8 = false;
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> convs;
    std::vector<std::vector<uint8_t>> mla_lat;
    std::vector<std::vector<uint16_t>> mla_kpe;
};

inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    out = a + b;
    return true;
}

inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

// What a shard's fingerprint reads: its size, the first and last 64 KiB, and kFpSamples windows of kFpSample bytes
// spread evenly over the file (two quants of one model with the same size and the same first 64 KiB differ in
// the tensors' bytes).  A file of at most kFpWhole bytes is read whole.  About 1 MiB a shard, 130 reads.
static constexpr size_t kFpEdge = 65536, kFpSample = 4096, kFpSamples = 128;
static constexpr uint64_t kFpWhole = 1ULL << 20;

inline uint64_t compute_gguf_set_fingerprint(const std::vector<std::string>& paths) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const void* p, size_t n) {
        const auto* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    };
    const uint64_t nfiles = (uint64_t) paths.size();
    mix(&nfiles, sizeof nfiles);
    std::vector<uint8_t> buf(kFpEdge);
    for (const std::string& path : paths) {
        std::error_code ec;
        const uint64_t fsz = std::filesystem::file_size(path, ec);
        if (!ec) mix(&fsz, sizeof fsz);
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        auto window = [&](uint64_t off, size_t n) {   // (the offset is mixed in too: a window moved is another file)
            f.clear();
            f.seekg((std::streamoff) off);
            if (!f) return;
            f.read((char*) buf.data(), (std::streamsize) n);
            const auto got = (size_t) std::max<std::streamsize>(0, f.gcount());
            mix(&off, sizeof off);
            if (got > 0) mix(buf.data(), got);
        };
        if (ec || fsz <= kFpWhole) {                  // (small, or of unknown size: from the start, as before)
            window(0, kFpEdge);
            if (!ec)
                for (uint64_t off = kFpEdge; off < fsz; off += kFpEdge) window(off, kFpEdge);
            continue;
        }
        window(0, kFpEdge);
        window(fsz - kFpEdge, kFpEdge);
        const uint64_t step = (fsz - kFpSample) / (kFpSamples - 1);
        for (size_t i = 0; i < kFpSamples; ++i) window(step * i, kFpSample);
    }
    return h;
}

// One tensor of a GGUF set as the fingerprint reads it.
struct FingerprintTensor {
    std::string name;
    uint32_t type = 0;
    uint64_t offset = 0;      // in its shard's data section
    uint64_t elements = 0;
    uint32_t shard = 0;
};

/// The file fingerprint mixed with the tensor table (names, types, offsets, element counts, shards): two files that
/// differ in a tensor's quant, or in where it sits, never share a fingerprint.  Sorted by name, so the order the
/// shards were read in does not matter.
inline uint64_t mix_tensor_table(uint64_t fp, std::vector<FingerprintTensor> tensors) {
    std::sort(tensors.begin(), tensors.end(),
              [](const FingerprintTensor& a, const FingerprintTensor& b) { return a.name < b.name; });
    uint64_t h = fp ^ 0x9e3779b97f4a7c15ULL;
    auto mix = [&](const void* p, size_t n) {
        const auto* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    };
    const uint64_t n = (uint64_t) tensors.size();
    mix(&n, sizeof n);
    for (const FingerprintTensor& t : tensors) {
        const uint64_t len = (uint64_t) t.name.size();
        mix(&len, sizeof len);
        mix(t.name.data(), t.name.size());
        mix(&t.type, sizeof t.type);
        mix(&t.offset, sizeof t.offset);
        mix(&t.elements, sizeof t.elements);
        mix(&t.shard, sizeof t.shard);
    }
    return h;
}

inline uint64_t compute_gguf_fingerprint(const std::string& path) {
    return compute_gguf_set_fingerprint(std::vector<std::string>{path});
}

inline uint64_t restore_snapshot_hash(const RestoreSnapshotData& snap, const RestoreKey& key) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const void* p, size_t n) {
        const auto* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    };
    mix(key.model_id.data(), key.model_id.size());
    mix(key.quant_tag.data(), key.quant_tag.size());
    mix(key.engine_version.data(), key.engine_version.size());
    mix(&key.gguf_fingerprint, sizeof key.gguf_fingerprint);
    mix(&key.ctx, sizeof key.ctx);
    mix(&key.kv_q8, sizeof key.kv_q8);
    if (!snap.tokens.empty()) mix(snap.tokens.data(), snap.tokens.size() * sizeof(int));
    for (const auto& pr : snap.pics) {
        mix(&pr.first, sizeof pr.first);
        mix(&pr.second, sizeof pr.second);
    }
    return h;
}

// Fixed binary header size before variable-length key strings:
//   magic(4) + ver(4) + gguf_fp(8) + ctx(4) + kv_q8(4) + rope_delta(4) +
//   nt(4) + np(4) + nk(4) + nm(4) + len_model(4) + len_quant(4) + len_eng(4) = 56 bytes.
static constexpr uint64_t kRestoreFixedHeaderBytes = 56;

inline bool compute_restore_file_bytes(const RestoreExpect& ex, uint32_t nt, uint32_t np, uint64_t key_str_bytes,
                                       uint64_t& total_out) {
    uint64_t total = 0;
    if (!checked_add_u64(kRestoreFixedHeaderBytes, key_str_bytes, total)) return false;

    uint64_t tok_b = 0, pic_b = 0;
    if (!checked_mul_u64((uint64_t) nt, (uint64_t) sizeof(int32_t), tok_b)) return false;
    if (!checked_mul_u64((uint64_t) np, (uint64_t) (sizeof(int64_t) + sizeof(uint64_t)), pic_b)) return false;
    if (!checked_add_u64(total, tok_b, total)) return false;
    if (!checked_add_u64(total, pic_b, total)) return false;

    uint64_t kda_one = 0, kda_all = 0;
    if (!checked_add_u64(16ULL, ex.kda_state_bytes, kda_one)) return false;
    if (!checked_add_u64(kda_one, ex.kda_conv_bytes, kda_one)) return false;
    if (!checked_mul_u64((uint64_t) ex.n_kda_layers, kda_one, kda_all)) return false;
    if (!checked_add_u64(total, kda_all, total)) return false;

    uint64_t lat_b = 0, kpe_b = 0, mla_one = 0, mla_all = 0;
    if (!checked_mul_u64((uint64_t) nt, ex.mla_lat_row_bytes, lat_b)) return false;
    if (!checked_mul_u64((uint64_t) nt, ex.mla_kpe_row_bytes, kpe_b)) return false;
    if (!checked_add_u64(16ULL, lat_b, mla_one)) return false;
    if (!checked_add_u64(mla_one, kpe_b, mla_one)) return false;
    if (!checked_mul_u64((uint64_t) ex.n_mla_layers, mla_one, mla_all)) return false;
    if (!checked_add_u64(total, mla_all, total)) return false;

    total_out = total;
    return true;
}

inline bool validate_snapshot_against_expect(const RestoreSnapshotData& snap, const RestoreExpect& ex,
                                             std::string& reason) {
    if (snap.tokens.empty() || snap.tokens.size() >= (size_t) ex.key.ctx) {
        reason = "token count " + std::to_string(snap.tokens.size()) + " out of range (0, " +
                 std::to_string(ex.key.ctx) + ")";
        return false;
    }
    if ((snap.kv_q8 ? 1u : 0u) != ex.key.kv_q8) {
        reason = "kv format mismatch";
        return false;
    }
    if (snap.pics.size() > snap.tokens.size() || snap.pics.size() > kRestoreMaxPics) {
        reason = "picture count out of range";
        return false;
    }
    for (size_t i = 0; i < snap.tokens.size(); ++i) {
        if (snap.tokens[i] < 0 || (uint32_t) snap.tokens[i] >= ex.vocab) {
            reason = "token id out of range [0, " + std::to_string(ex.vocab) + ")";
            return false;
        }
    }
    for (const auto& pr : snap.pics) {
        if (pr.first < 0 || (uint64_t) pr.first >= snap.tokens.size()) {
            reason = "picture offset out of token range";
            return false;
        }
    }
    if (snap.states.size() != ex.n_kda_layers || snap.convs.size() != ex.n_kda_layers) {
        reason = "KDA layer count mismatch";
        return false;
    }
    for (uint32_t i = 0; i < ex.n_kda_layers; ++i) {
        if ((uint64_t) snap.states[i].size() * sizeof(float) != ex.kda_state_bytes ||
            (uint64_t) snap.convs[i].size() * sizeof(float) != ex.kda_conv_bytes) {
            reason = "KDA layer " + std::to_string(i) + " buffer size mismatch";
            return false;
        }
    }
    if (snap.mla_lat.size() != ex.n_mla_layers || snap.mla_kpe.size() != ex.n_mla_layers) {
        reason = "MLA layer count mismatch";
        return false;
    }
    uint64_t exp_lat = 0, exp_kpe = 0;
    if (!checked_mul_u64((uint64_t) snap.tokens.size(), ex.mla_lat_row_bytes, exp_lat) ||
        !checked_mul_u64((uint64_t) snap.tokens.size(), ex.mla_kpe_row_bytes, exp_kpe)) {
        reason = "MLA byte size overflow";
        return false;
    }
    for (uint32_t i = 0; i < ex.n_mla_layers; ++i) {
        if ((uint64_t) snap.mla_lat[i].size() != exp_lat ||
            (uint64_t) snap.mla_kpe[i].size() * sizeof(uint16_t) != exp_kpe) {
            reason = "MLA layer " + std::to_string(i) + " buffer size mismatch";
            return false;
        }
    }
    return true;
}

inline bool enforce_restore_dir_cap(const std::string& dir, uint64_t max_bytes, uint64_t reserve_bytes,
                                    size_t* evicted_out = nullptr, std::string* err = nullptr) {
    if (evicted_out) *evicted_out = 0;
    if (dir.empty()) return false;
    if (max_bytes == 0 || reserve_bytes > max_bytes) {
        if (err) {
            *err = "snapshot size (" + std::to_string(reserve_bytes >> 20) + " MiB) exceeds restore-dir cap (" +
                   std::to_string(max_bytes >> 20) + " MiB)";
        }
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return true;

    struct Entry {
        std::filesystem::path path;
        std::filesystem::file_time_type mtime;
        uint64_t size = 0;
    };
    std::vector<Entry> files;
    uint64_t total = 0;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!de.is_regular_file(ec) || ec) continue;
        if (de.path().extension() != ".lingrp") continue;
        const uint64_t sz = de.file_size(ec);
        if (ec) continue;
        const auto mt = de.last_write_time(ec);
        if (ec) continue;
        files.push_back({de.path(), mt, sz});
        total += sz;
    }
    if (total + reserve_bytes <= max_bytes) return true;

    std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) {
        if (a.mtime != b.mtime) return a.mtime < b.mtime;
        return a.path.filename().string() < b.path.filename().string();
    });
    size_t evicted = 0;
    for (const auto& fe : files) {
        if (total + reserve_bytes <= max_bytes) break;
        std::error_code rec;
        if (std::filesystem::remove(fe.path, rec) && !rec) {
            total = (total >= fe.size) ? (total - fe.size) : 0;
            ++evicted;
        }
    }
    if (evicted_out) *evicted_out = evicted;
    if (total + reserve_bytes > max_bytes) {
        if (err) *err = "unable to free enough room under restore-dir cap";
        return false;
    }
    return true;
}

inline bool save_restore_file(const RestoreSnapshotData& snap, const RestoreExpect& ex, const std::string& dir,
                              uint64_t max_dir_bytes, std::string& err) {
    if (dir.empty()) {
        err = "restore directory not configured";
        return false;
    }
    if (ex.key.model_id.size() > kRestoreMaxStrLen || ex.key.quant_tag.size() > kRestoreMaxStrLen ||
        ex.key.engine_version.size() > kRestoreMaxStrLen) {
        err = "restore model key string too long";
        return false;
    }
    if (!validate_snapshot_against_expect(snap, ex, err)) return false;

    const uint32_t nt = (uint32_t) snap.tokens.size();
    const uint32_t np = (uint32_t) snap.pics.size();
    const uint64_t key_str_b = (uint64_t) ex.key.model_id.size() + (uint64_t) ex.key.quant_tag.size() +
                               (uint64_t) ex.key.engine_version.size();
    uint64_t out_bytes = 0;
    if (!compute_restore_file_bytes(ex, nt, np, key_str_b, out_bytes)) {
        err = "restore snapshot byte size overflow";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        err = "cannot create restore directory " + dir + ": " + ec.message();
        return false;
    }

    size_t evicted = 0;
    if (!enforce_restore_dir_cap(dir, max_dir_bytes, out_bytes, &evicted, &err)) return false;

    char fname[64];
    std::snprintf(fname, sizeof fname, "%016llx.lingrp", (unsigned long long) restore_snapshot_hash(snap, ex.key));
    const std::string final_p = (std::filesystem::path(dir) / fname).string();
    const std::string tmp_p = final_p + ".tmp";

    std::ofstream f(tmp_p, std::ios::binary);
    if (!f) {
        err = "cannot open restore point file " + tmp_p;
        return false;
    }
    const uint32_t ver = kRestoreFormatVersion;
    const uint32_t len_m = (uint32_t) ex.key.model_id.size();
    const uint32_t len_q = (uint32_t) ex.key.quant_tag.size();
    const uint32_t len_e = (uint32_t) ex.key.engine_version.size();
    const int32_t rdelta = (int32_t) snap.rope_delta;
    const uint32_t nk = ex.n_kda_layers;
    const uint32_t nm = ex.n_mla_layers;

    f.write(kRestoreMagic, 4);
    f.write((const char*) &ver, 4);
    f.write((const char*) &ex.key.gguf_fingerprint, 8);
    f.write((const char*) &ex.key.ctx, 4);
    f.write((const char*) &ex.key.kv_q8, 4);
    f.write((const char*) &rdelta, 4);
    f.write((const char*) &nt, 4);
    f.write((const char*) &np, 4);
    f.write((const char*) &nk, 4);
    f.write((const char*) &nm, 4);
    f.write((const char*) &len_m, 4);
    f.write((const char*) &len_q, 4);
    f.write((const char*) &len_e, 4);
    if (len_m) f.write(ex.key.model_id.data(), (std::streamsize) len_m);
    if (len_q) f.write(ex.key.quant_tag.data(), (std::streamsize) len_q);
    if (len_e) f.write(ex.key.engine_version.data(), (std::streamsize) len_e);

    for (uint32_t i = 0; i < nt; ++i) {
        const int32_t tok = (int32_t) snap.tokens[i];
        f.write((const char*) &tok, 4);
    }
    for (const auto& pr : snap.pics) {
        f.write((const char*) &pr.first, sizeof pr.first);
        f.write((const char*) &pr.second, sizeof pr.second);
    }
    for (uint32_t i = 0; i < nk; ++i) {
        const uint64_t sb = (uint64_t) snap.states[i].size() * sizeof(float);
        const uint64_t cb = (uint64_t) snap.convs[i].size() * sizeof(float);
        f.write((const char*) &sb, 8);
        if (sb) f.write((const char*) snap.states[i].data(), (std::streamsize) sb);
        f.write((const char*) &cb, 8);
        if (cb) f.write((const char*) snap.convs[i].data(), (std::streamsize) cb);
    }
    for (uint32_t i = 0; i < nm; ++i) {
        const uint64_t lb = (uint64_t) snap.mla_lat[i].size();
        const uint64_t kb = (uint64_t) snap.mla_kpe[i].size() * sizeof(uint16_t);
        f.write((const char*) &lb, 8);
        if (lb) f.write((const char*) snap.mla_lat[i].data(), (std::streamsize) lb);
        f.write((const char*) &kb, 8);
        if (kb) f.write((const char*) snap.mla_kpe[i].data(), (std::streamsize) kb);
    }
    f.close();
    if (!f) {
        std::filesystem::remove(tmp_p, ec);
        err = "write failed on " + tmp_p;
        return false;
    }
    std::filesystem::rename(tmp_p, final_p, ec);
    if (ec) {
        std::filesystem::remove(tmp_p, ec);
        err = "rename failed on " + final_p + ": " + ec.message();
        return false;
    }
    (void) enforce_restore_dir_cap(dir, max_dir_bytes, 0, nullptr, nullptr);
    return true;
}

inline bool load_restore_file(const std::filesystem::path& path, const RestoreExpect& ex, RestoreSnapshotData& out,
                              std::string& reason) {
    out = RestoreSnapshotData{};
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
        reason = "not a regular file";
        return false;
    }
    const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec || file_size < kRestoreFixedHeaderBytes) {
        reason = "truncated header (" + std::to_string(file_size) + " B)";
        return false;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        reason = "cannot open file";
        return false;
    }

    char magic[4] = {};
    uint32_t ver = 0, ctx = 0, kvq8 = 0, nt = 0, np = 0, nk = 0, nm = 0;
    uint32_t len_m = 0, len_q = 0, len_e = 0;
    uint64_t fp = 0;
    int32_t rdelta = 0;
    f.read(magic, 4);
    f.read((char*) &ver, 4);
    f.read((char*) &fp, 8);
    f.read((char*) &ctx, 4);
    f.read((char*) &kvq8, 4);
    f.read((char*) &rdelta, 4);
    f.read((char*) &nt, 4);
    f.read((char*) &np, 4);
    f.read((char*) &nk, 4);
    f.read((char*) &nm, 4);
    f.read((char*) &len_m, 4);
    f.read((char*) &len_q, 4);
    f.read((char*) &len_e, 4);
    if (!f) {
        reason = "short header read";
        return false;
    }
    if (std::memcmp(magic, kRestoreMagic, 4) != 0) {
        reason = "bad magic (expected LRP2)";
        return false;
    }
    if (ver != kRestoreFormatVersion) {
        reason = "format version " + std::to_string(ver) + " != " + std::to_string(kRestoreFormatVersion);
        return false;
    }
    if (len_m > kRestoreMaxStrLen || len_q > kRestoreMaxStrLen || len_e > kRestoreMaxStrLen) {
        reason = "oversized model-key string length";
        return false;
    }
    const uint64_t key_str_b = (uint64_t) len_m + (uint64_t) len_q + (uint64_t) len_e;
    if (kRestoreFixedHeaderBytes + key_str_b > file_size) {
        reason = "truncated model key";
        return false;
    }

    std::string f_model(len_m, '\0'), f_quant(len_q, '\0'), f_eng(len_e, '\0');
    if (len_m) f.read(&f_model[0], (std::streamsize) len_m);
    if (len_q) f.read(&f_quant[0], (std::streamsize) len_q);
    if (len_e) f.read(&f_eng[0], (std::streamsize) len_e);
    if (!f) {
        reason = "short model-key read";
        return false;
    }

    if (f_eng != ex.key.engine_version) {
        reason = "engine version mismatch (" + f_eng + " != " + ex.key.engine_version + ")";
        return false;
    }
    if (f_model != ex.key.model_id) {
        reason = "model mismatch (" + f_model + " != " + ex.key.model_id + ")";
        return false;
    }
    if (f_quant != ex.key.quant_tag) {
        reason = "quant mismatch (" + f_quant + " != " + ex.key.quant_tag + ")";
        return false;
    }
    if (fp != ex.key.gguf_fingerprint) {
        reason = "GGUF fingerprint mismatch";
        return false;
    }
    if (kvq8 > 1 || kvq8 != ex.key.kv_q8) {
        reason = std::string("KV format mismatch (file=") + (kvq8 ? "q8" : "f16") + " live=" +
                 (ex.key.kv_q8 ? "q8" : "f16") + ")";
        return false;
    }
    if (ctx != ex.key.ctx) {
        reason = "context mismatch (file=" + std::to_string(ctx) + " live=" + std::to_string(ex.key.ctx) + ")";
        return false;
    }

    if (nk != ex.n_kda_layers) {
        reason = "KDA layer count mismatch (file=" + std::to_string(nk) + " live=" +
                 std::to_string(ex.n_kda_layers) + ")";
        return false;
    }
    if (nm != ex.n_mla_layers) {
        reason = "MLA layer count mismatch (file=" + std::to_string(nm) + " live=" +
                 std::to_string(ex.n_mla_layers) + ")";
        return false;
    }
    if (nt == 0 || nt >= ex.key.ctx) {
        reason = "token count " + std::to_string(nt) + " out of range (0, " + std::to_string(ex.key.ctx) + ")";
        return false;
    }
    if (np > nt || np > kRestoreMaxPics) {
        reason = "picture count " + std::to_string(np) + " out of range";
        return false;
    }

    uint64_t expected_file_bytes = 0;
    if (!compute_restore_file_bytes(ex, nt, np, key_str_b, expected_file_bytes)) {
        reason = "byte-size overflow in header counts";
        return false;
    }
    if (file_size != expected_file_bytes) {
        reason = "file size mismatch (file=" + std::to_string(file_size) + " expected=" +
                 std::to_string(expected_file_bytes) + ")";
        return false;
    }

    uint64_t expected_lat_b = 0, expected_kpe_b = 0;
    if (!checked_mul_u64((uint64_t) nt, ex.mla_lat_row_bytes, expected_lat_b) ||
        !checked_mul_u64((uint64_t) nt, ex.mla_kpe_row_bytes, expected_kpe_b)) {
        reason = "MLA buffer size overflow";
        return false;
    }
    if ((ex.kda_state_bytes % sizeof(float)) != 0 || (ex.kda_conv_bytes % sizeof(float)) != 0 ||
        (expected_kpe_b % sizeof(uint16_t)) != 0) {
        reason = "misaligned expected layer buffer size";
        return false;
    }

    RestoreSnapshotData sp;
    sp.kv_q8 = (kvq8 != 0);
    sp.rope_delta = (int) rdelta;

    sp.tokens.resize(nt);
    for (uint32_t i = 0; i < nt; ++i) {
        int32_t tok = -1;
        f.read((char*) &tok, 4);
        if (!f) {
            reason = "truncated token array";
            return false;
        }
        if (tok < 0 || (uint32_t) tok >= ex.vocab) {
            reason = "token id " + std::to_string(tok) + " at index " + std::to_string(i) + " out of range [0, " +
                     std::to_string(ex.vocab) + ")";
            return false;
        }
        sp.tokens[i] = (int) tok;
    }

    sp.pics.resize(np);
    for (uint32_t i = 0; i < np; ++i) {
        f.read((char*) &sp.pics[i].first, sizeof(int64_t));
        f.read((char*) &sp.pics[i].second, sizeof(uint64_t));
        if (!f) {
            reason = "truncated picture metadata";
            return false;
        }
        if (sp.pics[i].first < 0 || (uint64_t) sp.pics[i].first >= (uint64_t) nt) {
            reason = "picture start offset out of token range";
            return false;
        }
    }

    sp.states.resize(nk);
    sp.convs.resize(nk);
    for (uint32_t i = 0; i < nk; ++i) {
        uint64_t sb = 0, cb = 0;
        f.read((char*) &sb, 8);
        if (!f || sb != ex.kda_state_bytes) {
            reason = "KDA layer " + std::to_string(i) + " state size mismatch (file=" + std::to_string(sb) +
                     " expected=" + std::to_string(ex.kda_state_bytes) + ")";
            return false;
        }
        sp.states[i].resize((size_t) (sb / sizeof(float)));
        if (sb) f.read((char*) sp.states[i].data(), (std::streamsize) sb);
        f.read((char*) &cb, 8);
        if (!f || cb != ex.kda_conv_bytes) {
            reason = "KDA layer " + std::to_string(i) + " conv size mismatch (file=" + std::to_string(cb) +
                     " expected=" + std::to_string(ex.kda_conv_bytes) + ")";
            return false;
        }
        sp.convs[i].resize((size_t) (cb / sizeof(float)));
        if (cb) f.read((char*) sp.convs[i].data(), (std::streamsize) cb);
        if (!f) {
            reason = "truncated KDA layer " + std::to_string(i);
            return false;
        }
    }

    sp.mla_lat.resize(nm);
    sp.mla_kpe.resize(nm);
    for (uint32_t i = 0; i < nm; ++i) {
        uint64_t lb = 0, kb = 0;
        f.read((char*) &lb, 8);
        if (!f || lb != expected_lat_b) {
            reason = "MLA layer " + std::to_string(i) + " latent size mismatch (file=" + std::to_string(lb) +
                     " expected=" + std::to_string(expected_lat_b) + ")";
            return false;
        }
        sp.mla_lat[i].resize((size_t) lb);
        if (lb) f.read((char*) sp.mla_lat[i].data(), (std::streamsize) lb);
        f.read((char*) &kb, 8);
        if (!f || kb != expected_kpe_b) {
            reason = "MLA layer " + std::to_string(i) + " k_pe size mismatch (file=" + std::to_string(kb) +
                     " expected=" + std::to_string(expected_kpe_b) + ")";
            return false;
        }
        sp.mla_kpe[i].resize((size_t) (kb / sizeof(uint16_t)));
        if (kb) f.read((char*) sp.mla_kpe[i].data(), (std::streamsize) kb);
        if (!f) {
            reason = "truncated MLA layer " + std::to_string(i);
            return false;
        }
    }

    out = std::move(sp);
    return true;
}

inline size_t load_restore_dir(const std::string& dir, const RestoreExpect& ex, uint64_t max_dir_bytes,
                               std::vector<RestoreSnapshotData>& out,
                               std::vector<std::string>* skipped_logs = nullptr) {
    if (dir.empty()) return 0;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return 0;
    if (max_dir_bytes > 0) (void) enforce_restore_dir_cap(dir, max_dir_bytes, 0, nullptr, nullptr);

    struct FileRec {
        std::filesystem::path path;
        std::filesystem::file_time_type mtime;
    };
    std::vector<FileRec> files;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (de.path().extension() != ".lingrp") continue;
        const auto mt = de.last_write_time(ec);
        files.push_back({de.path(), ec ? std::filesystem::file_time_type::min() : mt});
    }
    std::sort(files.begin(), files.end(), [](const FileRec& a, const FileRec& b) {
        if (a.mtime != b.mtime) return a.mtime > b.mtime;
        return a.path.filename().string() > b.path.filename().string();
    });

    size_t loaded = 0;
    for (const auto& fr : files) {
        RestoreSnapshotData sp;
        std::string reason;
        if (load_restore_file(fr.path, ex, sp, reason)) {
            out.push_back(std::move(sp));
            ++loaded;
        } else if (skipped_logs != nullptr) {
            skipped_logs->push_back("ling restore: skipping " + fr.path.filename().string() + ": " + reason);
        }
    }
    return loaded;
}

}  // namespace ling::sys
