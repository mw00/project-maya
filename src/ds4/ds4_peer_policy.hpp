#pragma once

namespace strata::ds4 {

// Only the implicit exact AR2 default may compete with peer selection. Explicit AR2=1/2
// remains authoritative, and flow groups cannot use the peer prefill transport.
inline bool peer_ar_group_eligible(int members, int ar2, bool implicit_ar2, bool flow0, bool flow1) {
    return members == 2 && (ar2 == 0 || (implicit_ar2 && ar2 == 1)) && !flow0 && !flow1;
}

} // namespace strata::ds4
