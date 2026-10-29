#pragma once
// The serve loop's events (rig's evt, ported): closures in (priority, id) order - the highest priority first, then the
// order they were added in - run one at a time by the one thread that owns the engine's state (run()), added from any
// thread (add(): the reader's lines, a GPU stream's host callback for a token done).  Nothing polls: the consumer waits
// on a condition until an event comes, or for `tick` when the queue is empty.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4::evt {

using Closure = std::function<void()>;

struct Event {
    uint64_t id = 0;
    int priority = 0;
    Closure closure;
};

// a 4-ary heap by (priority desc, id asc), an event's position kept so that update() can move it
class Queue {
public:
    explicit Queue(size_t capacity = kMinCap) { evts_.reserve(capacity < kMinCap ? kMinCap : capacity); }

    size_t size() const { return evts_.size(); }
    bool empty() const { return evts_.empty(); }

    void push(Event e) {
        if (pos_.count(e.id) != 0) return;
        evts_.push_back(std::move(e));
        const size_t i = evts_.size() - 1;
        pos_[evts_[i].id] = i;
        up(i);
    }
    bool pop(Event& out) {
        if (evts_.empty()) return false;
        swap(0, evts_.size() - 1);
        out = std::move(evts_.back());
        evts_.pop_back();
        pos_.erase(out.id);
        if (!evts_.empty()) down(0);
        return true;
    }
    const Event* peek() const { return evts_.empty() ? nullptr : &evts_[0]; }
    bool update(uint64_t id, int priority) {
        const auto it = pos_.find(id);
        if (it == pos_.end()) return false;
        const size_t i = it->second;
        evts_[i].priority = priority;
        up(i);
        down(pos_[id]);
        return true;
    }

private:
    static constexpr size_t kArity = 4;
    static constexpr size_t kMinCap = 8;
    std::vector<Event> evts_;
    std::unordered_map<uint64_t, size_t> pos_;

    static bool less(const Event& a, const Event& b) {
        return a.priority != b.priority ? a.priority > b.priority : a.id < b.id;
    }
    void swap(size_t i, size_t j) {
        std::swap(evts_[i], evts_[j]);
        pos_[evts_[i].id] = i;
        pos_[evts_[j].id] = j;
    }
    void up(size_t i) {
        while (i > 0) {
            const size_t parent = (i - 1) / kArity;
            if (!less(evts_[i], evts_[parent])) return;
            swap(i, parent);
            i = parent;
        }
    }
    void down(size_t i) {
        const size_t n = evts_.size();
        for (;;) {
            const size_t first = i * kArity + 1;
            if (first >= n) return;
            size_t best = first;
            const size_t last = first + kArity - 1 < n ? first + kArity - 1 : n - 1;
            for (size_t c = first + 1; c <= last; ++c)
                if (less(evts_[c], evts_[best])) best = c;
            if (!less(evts_[best], evts_[i])) return;
            swap(i, best);
            i = best;
        }
    }
};

// the consumer: run() pops and runs until stop(); add() from any thread (0: stopped).  `tick` (optional) runs when no
// event came for `idle_ms` (0: at once, the queue empty) - the consumer's own work between events.  pump() runs one
// event, waiting up to `ms` for it: the consumer's own wait inside an event (a gap of a prompt's read: the answers'
// tokens come back as events while the read is paused)
class Engine {
public:
    explicit Engine(std::function<void()> tick = nullptr, int idle_ms = 0) : tick_(std::move(tick)), idle_ms_(idle_ms) {}

    uint64_t add(Closure c, int priority) {
        if (!c) return 0;
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) return 0;
        const uint64_t id = ++clock_;
        q_.push(Event{id, priority < 0 ? 0 : priority, std::move(c)});
        cv_.notify_one();
        return id;
    }
    bool update(uint64_t id, int priority) {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.update(id, priority < 0 ? 0 : priority);
    }
    void run() {
        while (!stopped()) {
            if (pump(tick_ ? idle_ms_ : -1)) continue;
            if (tick_ && !stopped()) tick_();
        }
    }
    // one event, waiting up to `ms` for it (< 0: until one comes or the engine stops); false: none came
    bool pump(int ms) {
        Event e;
        {
            std::unique_lock<std::mutex> lk(mu_);
            const auto ready = [&] { return stopped_ || !q_.empty(); };
            if (ms < 0) cv_.wait(lk, ready);
            else if (ms > 0) cv_.wait_for(lk, std::chrono::milliseconds(ms), ready);
            if (stopped_ || !q_.pop(e)) return false;
        }
        e.closure();
        return true;
    }
    bool stopped() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stopped_;
    }
    void stop() {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
        cv_.notify_all();
    }
    size_t pending() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    Queue q_;
    uint64_t clock_ = 0;
    bool stopped_ = false;
    std::function<void()> tick_;
    int idle_ms_ = 0;
};

}  // namespace strata::ds4::evt
