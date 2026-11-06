// Regression checks for the transport precedence used at peer-prefill startup (no GPU or model needed).
#include "../src/ds4/ds4_peer_policy.hpp"
#include <cstdio>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    using strata::ds4::peer_ar_group_eligible;
    // NVIDIA's implicit AR2=1 must be eligible for peer auto or an explicit peer request.
    CHECK(peer_ar_group_eligible(2, 1, true, false, false));
    // AMD's implicit host default and a user explicitly setting AR2=0 are eligible too.
    CHECK(peer_ar_group_eligible(2, 0, true, false, false));
    CHECK(peer_ar_group_eligible(2, 0, false, false, false));
    // Explicit AR2=1 or 2 wins over peer, independent of the requested peer mode.
    CHECK(!peer_ar_group_eligible(2, 1, false, false, false));
    CHECK(!peer_ar_group_eligible(2, 2, false, false, false));
    CHECK(!peer_ar_group_eligible(2, 2, true, false, false));
    CHECK(!peer_ar_group_eligible(1, 1, true, false, false));
    CHECK(!peer_ar_group_eligible(3, 1, true, false, false));
    CHECK(!peer_ar_group_eligible(2, 1, true, true, false));
    CHECK(!peer_ar_group_eligible(2, 1, true, false, true));
    std::puts("peer prefill policy: implicit default, explicit choices, group size and flow passed");
}
