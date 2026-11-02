// Deterministic host-only regression for sampling expiry with a collective in flight.
#include "../src/ds4/ds4_peer_poll.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>

using namespace strata::ds4;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct FakeStreams {
    Clock::time_point time{};
    std::array<Clock::time_point, 2> finish{};
    bool error = false;
    int queries = 0;
    PeerPoll query() {
        ++queries;
        if (error) return PeerPoll::error;
        return time >= finish[0] && time >= finish[1] ? PeerPoll::ready : PeerPoll::pending;
    }
    bool drain(Clock::duration allowance = 2500ms) {
        return drain_peer_probe([&] { return query(); }, [&] { return time; }, [&] { time += 1ms; }, allowance);
    }
};

int main() {
    // Sampling expired already. Both ranks must still drain, including the slower
    // partner, before the caller may inspect results, reset flags or free scratch.
    FakeStreams crossed;
    const auto sampling_deadline = crossed.time + 3ms;
    crossed.finish = {crossed.time + 7ms, crossed.time + 12ms};
    crossed.time = sampling_deadline;
    REQUIRE(crossed.drain());
    REQUIRE(crossed.time == Clock::time_point{} + 12ms);
    REQUIRE(crossed.queries == 10);
    REQUIRE(crossed.time >= sampling_deadline);

    // No sampling allowance left, but pending work can complete during the
    // separate drain grace. A completed stream is safe even at a zero allowance.
    FakeStreams ready;
    REQUIRE(ready.drain(0ms));
    REQUIRE(ready.queries == 1);
    FakeStreams pending;
    pending.finish = {pending.time + 1ms, pending.time + 1ms};
    REQUIRE(!pending.drain(0ms));
    REQUIRE(pending.time == Clock::time_point{});

    // A missing partner remains unsafe: bounded polling must not turn it into
    // a successful fallback or allow scratch to be released.
    FakeStreams stuck;
    stuck.finish = {stuck.time, Clock::time_point::max()};
    REQUIRE(!stuck.drain());
    REQUIRE(stuck.time == Clock::time_point{} + 2500ms);

    FakeStreams failed;
    failed.error = true;
    REQUIRE(!failed.drain());
    REQUIRE(failed.queries == 1 && failed.time == Clock::time_point{});

    FakeStreams edge;
    edge.finish = {edge.time + 2500ms, edge.time + 2500ms};
    REQUIRE(edge.drain()); // readiness is authoritative, even at the deadline
    REQUIRE(edge.time == Clock::time_point{} + 2500ms);
    std::puts("peer probe drain: expired sampling, both ranks, ready, timeout and query error PASS");
}
