#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <system_error>
#include <utility>

namespace strata::core {

// Bounded temporary directory ordering, not a bound on the accepted slot catalogue.
// Rescans trade worst-case O(N^2 / batch_capacity) directory visits for fixed storage.
class SlotDirectoryScan {
public:
    using Entry = std::pair<std::filesystem::file_time_type, std::filesystem::path>;
    static constexpr std::size_t batch_capacity = 64;

    explicit SlotDirectoryScan(std::filesystem::path directory, bool include_learned)
        : directory_(std::move(directory)), include_learned_(include_learned) {}

    // Name-only eligibility must precede attribute reads. Keep temporary and
    // incomplete slot names too: the caller, not this ordering helper, cleans them.
    static bool accepts_name(const std::filesystem::path& path, bool include_learned) {
        const auto name = path.filename().string();
        return name.rfind("slot-", 0) == 0 || (include_learned && name.rfind("branch-", 0) == 0);
    }

    // The returned view lasts until next(). Removing visited files, including the
    // cursor file, is safe; concurrent additions or timestamp changes are not a snapshot.
    std::span<const Entry> next(std::error_code& ec) {
        ec.clear();
        std::size_t count = 0;
        std::filesystem::directory_iterator it(directory_, ec), end;
        if (ec) return {};
        for (; it != end; it.increment(ec)) {
            if (ec) return {};
            if (!accepts_name(it->path(), include_learned_)) continue;
            Entry candidate{std::filesystem::last_write_time(it->path(), ec), it->path()};
            if (ec) return {};
            if (cursor_ && !newer(*cursor_, candidate)) continue;
            if (count < batch_capacity) {
                entries_[count++] = std::move(candidate);
                std::push_heap(entries_.begin(), entries_.begin() + count, newer);
            } else if (newer(candidate, entries_.front())) {
                std::pop_heap(entries_.begin(), entries_.end(), newer);
                entries_.back() = std::move(candidate);
                std::push_heap(entries_.begin(), entries_.end(), newer);
            }
        }
        if (ec || !count) return {};
        std::sort(entries_.begin(), entries_.begin() + count, newer);
        cursor_ = entries_[count - 1];
        return {entries_.data(), count};
    }

private:
    static bool newer(const Entry& a, const Entry& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    }

    std::filesystem::path directory_;
    bool include_learned_;
    std::array<Entry, batch_capacity> entries_;
    std::optional<Entry> cursor_;
};

} // namespace strata::core
