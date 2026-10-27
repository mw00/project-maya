// ds4_evt.hpp (rig's evt, ported): the queue's (priority, id) order, update, and the engine - events added from
// several threads each run once, on the one consumer, in order; stop ends the run
#include "../src/ds4/ds4_evt.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace strata::ds4::evt;

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

static void queue_order() {
    Queue q;
    const int prio[] = {1, 5, 3, 5, 0, 3, 9, 1, 5, 2};
    for (int i = 0; i < 10; ++i) q.push(Event{(uint64_t) i + 1, prio[i], nullptr});
    q.push(Event{2, 100, nullptr});   // (an id already queued: ignored)
    CHECK(q.size() == 10);
    std::vector<uint64_t> got;
    Event e;
    while (q.pop(e)) got.push_back(e.id);
    const std::vector<uint64_t> want = {7, 2, 4, 9, 3, 6, 10, 1, 8, 5};
    CHECK(got == want);
}

static void queue_update() {
    Queue q;
    for (uint64_t i = 1; i <= 6; ++i) q.push(Event{i, 1, nullptr});
    CHECK(q.update(5, 10));
    CHECK(q.update(1, 0));
    CHECK(!q.update(42, 3));
    Event e;
    q.pop(e);
    CHECK(e.id == 5);
    std::vector<uint64_t> rest;
    while (q.pop(e)) rest.push_back(e.id);
    CHECK((rest == std::vector<uint64_t>{2, 3, 4, 6, 1}));
}

static void engine_threads() {
    Engine eng;
    constexpr int kThreads = 8, kEach = 20000;
    std::atomic<int> ran{0};
    std::vector<int> seen((size_t) kThreads * kEach, 0);   // (written by the consumer only)
    std::thread consumer([&] { eng.run(); });
    std::vector<std::thread> ps;
    for (int t = 0; t < kThreads; ++t)
        ps.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i) {
                const size_t k = (size_t) t * kEach + i;
                eng.add([&, k] {
                    ++seen[k];
                    ++ran;
                }, i % 4);
            }
        });
    for (auto& p : ps) p.join();
    std::atomic<bool> last{false};
    eng.add([&] { last = true; }, 0);
    while (!last.load()) std::this_thread::yield();
    eng.stop();
    consumer.join();
    CHECK(ran.load() == kThreads * kEach);
    bool once = true;
    for (int v : seen) once = once && v == 1;
    CHECK(once);
    CHECK(eng.add([] {}, 1) == 0);   // (stopped: refused)
}

static void engine_priority() {
    // events queued before the consumer starts run highest first, then in the order added
    Engine eng;
    std::vector<int> order;
    eng.add([&] { order.push_back(1); }, 0);
    eng.add([&] { order.push_back(2); }, 3);
    eng.add([&] { order.push_back(3); }, 3);
    eng.add([&] { order.push_back(4); }, 7);
    eng.add([&] { eng.stop(); }, -5);   // (below 0: 0, after the 0 added first)
    eng.run();
    CHECK((order == std::vector<int>{4, 2, 3, 1}));
}

static void engine_tick() {
    // an empty queue runs the tick instead of waiting; the tick may add
    int ticks = 0;
    Engine* ep = nullptr;
    Engine eng([&] {
        if (++ticks == 3) ep->add([&] { ep->stop(); }, 1);
    });
    ep = &eng;
    eng.run();
    CHECK(ticks == 3);
}

static void engine_pump() {
    // pump: one event, or none within the wait; an idle tick after idle_ms without one
    Engine eng;
    CHECK(!eng.pump(0));
    std::vector<int> order;
    eng.add([&] {
        order.push_back(1);
        // (inside an event: the consumer runs the ones that come meanwhile - a read's gap)
        while (eng.pump(0)) {}
    }, 1);
    eng.add([&] { order.push_back(2); }, 0);
    CHECK(eng.pump(-1));
    CHECK((order == std::vector<int>{1, 2}));
    std::thread later([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        eng.add([&] { order.push_back(3); }, 0);
    });
    CHECK(eng.pump(2000));
    later.join();
    CHECK(order.back() == 3);
    int ticks = 0;
    Engine idle([&] { ++ticks; }, 20);
    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(110));
        idle.stop();
    });
    idle.run();
    stopper.join();
    CHECK(ticks >= 3 && ticks <= 6);
}

int main() {
    queue_order();
    queue_update();
    engine_threads();
    engine_priority();
    engine_tick();
    engine_pump();
    if (fails != 0) {
        std::fprintf(stderr, "ds4 evt: %d check(s) failed\n", fails);
        return 1;
    }
    std::printf("ds4 evt: queue order and update, engine threads, priority, tick and pump passed\n");
    return 0;
}
