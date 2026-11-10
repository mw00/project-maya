#pragma once

// Process-local index for SSD snapshots. The engine owns snapshot serialization;
// this class owns only its private files, prefix matching and capacity accounting.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>

namespace maya::metal {
class SlotIndex {
    struct Entry {
        std::vector<int32_t> tokens;
        uint64_t image_hash, bytes, age;
        std::filesystem::path path;
    };
    std::filesystem::path dir_;
    std::vector<Entry> entries_;
    size_t limit_;
    uint64_t budget_, used_ = 0, age_ = 0, next_ = 0;
    void erase(size_t i) {
        std::error_code ec;
        if (!std::filesystem::remove(entries_[i].path, ec) && ec) return;
        used_ -= entries_[i].bytes;
        entries_.erase(entries_.begin() + (long) i);
    }
public:
    // base must be a trusted NVM directory; a private random subdirectory is
    // used so a restart never restores another model's unvalidated snapshots.
    SlotIndex(const std::filesystem::path& base, size_t count, uint64_t budget)
        : limit_(count), budget_(budget) {
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        if (ec || !count || !budget) return;
        std::string pattern = (base / "metal-slots-XXXXXX").string();
        std::vector<char> p(pattern.begin(), pattern.end()); p.push_back('\0');
        if (char* d = mkdtemp(p.data())) dir_ = d;
    }
    SlotIndex(const SlotIndex&) = delete;
    SlotIndex& operator=(const SlotIndex&) = delete;
    ~SlotIndex() { if (!dir_.empty()) { std::error_code ec; std::filesystem::remove_all(dir_, ec); } }

    // save writes exactly one file, returning false on failure. Only completed
    // files are published. Reserve their exact size first so even the pending
    // file and the existing entries together never exceed the disk quota; save
    // must itself refuse to write more than the declared size.
    bool put(std::vector<int32_t> tokens, uint64_t image_hash, uint64_t bytes,
             const std::function<bool(const std::filesystem::path&)>& save) {
        if (dir_.empty() || tokens.empty() || !bytes || bytes > budget_) return false;
        while (entries_.size() >= limit_ || used_ > budget_ - bytes) {
            if (entries_.empty()) return false;
            size_t i = (size_t) (std::min_element(entries_.begin(), entries_.end(),
                [](const Entry& a, const Entry& b) { return a.age < b.age; }) - entries_.begin());
            size_t old = entries_.size(); erase(i);
            if (old == entries_.size()) return false;
        }
        const auto tmp = dir_ / ("pending-" + std::to_string(next_++));
        std::error_code ec;
        if (!save(tmp)) { std::filesystem::remove(tmp, ec); return false; }
        auto status = std::filesystem::symlink_status(tmp, ec);
        if (ec || !std::filesystem::is_regular_file(status)) { std::filesystem::remove(tmp, ec); return false; }
        if (std::filesystem::file_size(tmp, ec) != bytes || ec) { std::filesystem::remove(tmp, ec); return false; }
        auto path = dir_ / ("slot-" + std::to_string(next_++));
        std::filesystem::rename(tmp, path, ec);
        if (ec) { std::filesystem::remove(tmp, ec); return false; }
        entries_.push_back({std::move(tokens), image_hash, bytes, ++age_, path});
        used_ += bytes;
        return true;
    }
    // image_hash is computed for each candidate's prefix, not for the full prompt.
    // Try longest first; a failed restore is removed and never treated as a hit.
    // load must validate before changing model state or reset it on failure.
    size_t restore(const std::vector<int32_t>& prompt, size_t target,
                   const std::function<uint64_t(size_t)>& image_hash,
                   const std::function<bool(const std::filesystem::path&, size_t)>& load) {
        for (;;) {
            size_t best = entries_.size(), n = 0;
            for (size_t i = 0; i < entries_.size(); ++i) {
                const auto& e = entries_[i];
                if (e.tokens.size() <= target && e.tokens.size() <= prompt.size() && e.tokens.size() > n &&
                    std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin()) &&
                    e.image_hash == image_hash(e.tokens.size())) { best = i; n = e.tokens.size(); }
            }
            if (best == entries_.size()) return 0;
            if (load(entries_[best].path, n)) { entries_[best].age = ++age_; return n; }
            size_t old = entries_.size(); erase(best);
            if (old == entries_.size()) return 0;
        }
    }
};
}
