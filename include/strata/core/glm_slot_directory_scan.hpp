#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

namespace strata::core {

// One bounded enumeration, then immutable sorted windows. This bounds the temporary
// directory catalogue, not token keys or the caller's accepted slot catalogue.
class SlotDirectoryScan {
public:
    using Entry = std::pair<std::filesystem::file_time_type, std::filesystem::path>;
    static constexpr std::size_t batch_capacity = 64;
    static constexpr std::size_t max_visits = 8192;
    static constexpr std::size_t max_candidates = 1024;

    explicit SlotDirectoryScan(std::filesystem::path directory, bool include_learned,
                               std::size_t visit_budget = max_visits,
                               std::size_t candidate_budget = max_candidates)
        : directory_(std::move(directory)), include_learned_(include_learned),
          visit_limit_(std::min(visit_budget, max_visits)),
          candidate_limit_(std::min(candidate_budget, max_candidates)) {}

    // Name-only eligibility must precede attribute reads. Keep temporary and
    // incomplete slot names too: the caller, not this ordering helper, cleans them.
    static bool accepts_name(const std::filesystem::path& path, bool include_learned) {
        const auto name = path.filename().string();
        return name.rfind("slot-", 0) == 0 || (include_learned && name.rfind("branch-", 0) == 0);
    }

    // Only regular files participate. Known unsupported kinds are harmless skips;
    // an absent or indeterminate enumerated item cannot establish a complete scan.
    static bool accepts_type(std::filesystem::file_type type, std::error_code& ec) {
        ec.clear();
        switch (type) {
        case std::filesystem::file_type::regular:
            return true;
        case std::filesystem::file_type::directory:
        case std::filesystem::file_type::symlink:
        case std::filesystem::file_type::block:
        case std::filesystem::file_type::character:
        case std::filesystem::file_type::fifo:
        case std::filesystem::file_type::socket:
            return false;
        default:
            ec = std::make_error_code(std::errc::io_error);
            return false;
        }
    }

    static bool is_regular_entry(const std::filesystem::path& path, std::error_code& ec) {
        const auto status = std::filesystem::symlink_status(path, ec);
        if (ec) return false;
        return accepts_type(status.type(), ec);
    }

    // The first call must reach directory end successfully before any view is
    // published. Later calls only slice memory; deleting visited files is safe.
    // Every view remains valid for this object's lifetime. This is not a snapshot
    // against concurrent filesystem changes during the initial enumeration.
    std::span<const Entry> next(std::error_code& ec) {
        if (!initialized_) enumerate();
        ec = error_;
        if (ec || offset_ == entries_.size()) return {};
        const auto count = std::min(batch_capacity, entries_.size() - offset_);
        const auto result = std::span<const Entry>(entries_).subspan(offset_, count);
        offset_ += count;
        return result;
    }

    // Uninitialized and failed scans expose nothing, even if entries were read.
    std::span<const Entry> all_entries() const {
        return initialized_ && !error_ ? std::span<const Entry>(entries_) : std::span<const Entry>{};
    }

    // Counts describe processed directory entries and admitted candidates, not
    // an extra iterator position that establishes that a budget was exceeded.
    std::size_t visited_entries() const { return visited_; }
    std::size_t candidate_entries() const { return candidates_; }
    std::size_t visit_limit() const { return visit_limit_; }
    std::size_t candidate_limit() const { return candidate_limit_; }

private:
    void enumerate() {
        initialized_ = true;
        entries_.reserve(std::min(visit_limit_, candidate_limit_));
        std::filesystem::directory_iterator it(directory_, error_), end;
        if (error_) return;
        for (; it != end; it.increment(error_)) {
            if (error_) break;
            if (visited_ == visit_limit_) {
                error_ = std::make_error_code(std::errc::no_buffer_space);
                break;
            }
            ++visited_;
            if (!accepts_name(it->path(), include_learned_)) continue;
            if (!is_regular_entry(it->path(), error_)) {
                if (error_) break;
                continue;
            }
            if (candidates_ == candidate_limit_) {
                error_ = std::make_error_code(std::errc::no_buffer_space);
                break;
            }
            const auto stamp = std::filesystem::last_write_time(it->path(), error_);
            if (error_) break;
            entries_.emplace_back(stamp, it->path());
            ++candidates_;
        }
        // The increment after the final permitted item must reach end without
        // error. Another item is a budget failure, not a successful truncation.
        if (error_) {
            entries_.clear();
            return;
        }
        std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
    }

    std::filesystem::path directory_;
    bool include_learned_;
    std::size_t visit_limit_, candidate_limit_;
    std::vector<Entry> entries_;
    std::error_code error_;
    std::size_t visited_ = 0, candidates_ = 0, offset_ = 0;
    bool initialized_ = false;
};

} // namespace strata::core
