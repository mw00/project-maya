// The RAM a peer daemon may take: MemAvailable bounded by the tightest cgroup-v2 ancestor limit.
// Copied from src/core/expert_source.cpp's detail::available_memory_bytes (same loop, same comments'
// spirit) rather than linked: strata_core pulls the GPU runtime headers, and the daemon must stay a
// plain host program.  The LXC lesson lives on in both copies: /proc/meminfo inside a container is
// the host's - without the cgroup walk a "free" figure is fiction and the slab OOMs the box.
#pragma once

#include <cstdint>

namespace strata::peer {

/// The bytes free now for a slab the machine may safely pin (0 when nothing can be read).
uint64_t available_memory_bytes();

/// A whole figure capped by what is available (the daemon's --ram-gb 0 probe: a quarter of what is
/// free now, at most `cap` bytes).
uint64_t budget_probe(uint64_t cap);

}   // namespace strata::peer
