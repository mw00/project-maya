// Host-side ownership for the recent-cache worker's two pinned copy sources.
// CUDA allocation, submission, and event queries stay with the worker; the same
// source selection and lifetime transitions are exercised without a GPU in tests.
#pragma once
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::ds4 {
class RecentStage {
public:
    static constexpr int kBuffers = 2;
    static constexpr size_t kMaxBufferBytes = size_t{32} << 20;
    enum class Prepare { Unavailable, Waiting, Ready, ReadFailed };
    enum class Completion { Pending, Complete, Failed };

    RecentStage() = default;
    RecentStage(const RecentStage&) = delete;
    RecentStage& operator=(const RecentStage&) = delete;
    static bool fits(size_t bytes) { return bytes > 0 && bytes <= kMaxBufferBytes; }

    // Allocate an all-or-nothing pair. close() must precede destruction;
    // a failed drain intentionally keeps its sources alive until process exit.
    template<class Allocate, class Release, class Allow>
    bool init(size_t bytes, Allocate allocate, Release release, Allow allow) {
        assert(!enabled());
        if (!fits(bytes)) return false;
        std::array<uint8_t*, kBuffers> pair{nullptr, nullptr};
        for (auto*& p : pair) {
            p = allocate(bytes);
            if (p) continue;
            for (auto* q : pair) if (q) release(q);
            return false;
        }
        if (!allow()) {
            for (auto* p : pair) release(p);
            return false;
        }
        for (auto* p : pair) std::memset(p, 0, bytes);
        buffers_ = pair;
        bytes_ = bytes;
        return true;
    }
    bool enabled() const { return buffers_[0] != nullptr; }
    const uint8_t* data(int i) const {
        assert(i >= 0 && i < kBuffers && states_[(size_t) i] == State::Busy);
        return buffers_[(size_t) i];
    }

    // No read while both buffers are in use. A miss uses the file reader, not a
    // model mapping. The caller submits only Ready and retires ReadFailed jobs.
    template<class StoreRead, class FileRead>
    Prepare prepare(size_t bytes, int& index, StoreRead store, FileRead file) {
        index = -1;
        if (!enabled() || bytes == 0 || bytes > bytes_) return Prepare::Unavailable;
        bool busy = false;
        for (int i = 0; i < kBuffers; ++i) {
            if (states_[(size_t) i] == State::Busy) busy = true;
            if (states_[(size_t) i] != State::Free) continue;
            states_[(size_t) i] = State::Busy;
            if (!store(buffers_[(size_t) i]) && !file(buffers_[(size_t) i])) {
                states_[(size_t) i] = State::Free;
                return Prepare::ReadFailed;
            }
            index = i;
            return Prepare::Ready;
        }
        // Quarantined sources must never be reused. If all are quarantined,
        // allow the original source path rather than waiting forever for one.
        return busy ? Prepare::Waiting : Prepare::Unavailable;
    }

    template<class Fence>
    void failed(int i, Fence fence) {
        assert(i >= 0 && i < kBuffers && states_[(size_t) i] == State::Busy);
        states_[(size_t) i] = fence() ? State::Free : State::Quarantined;
    }
    template<class Fence>
    Completion poll(int i, Completion result, Fence fence) {
        assert(i >= 0 && i < kBuffers && states_[(size_t) i] == State::Busy);
        if (result == Completion::Complete) states_[(size_t) i] = State::Free;
        else if (result == Completion::Failed) failed(i, fence);
        return result;
    }
    template<class Drain, class Release>
    bool close(Drain drain, Release release) {
        if (!enabled()) return true;
        if (!drain()) return false;
        for (auto*& p : buffers_) { release(p); p = nullptr; }
        states_.fill(State::Free);
        bytes_ = 0;
        return true;
    }

private:
    enum class State { Free, Busy, Quarantined };
    std::array<uint8_t*, kBuffers> buffers_{nullptr, nullptr};
    std::array<State, kBuffers> states_{State::Free, State::Free};
    size_t bytes_ = 0;
};
} // namespace strata::ds4
