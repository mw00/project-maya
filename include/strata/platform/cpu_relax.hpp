// include/strata/platform/cpu_relax.hpp - the spin-wait hint and the store fence, per architecture.
//
// x86: `_mm_pause` and `_mm_sfence` (the fence orders the write-combining stores into pinned host memory before
// the doorbell the GPU polls).  aarch64: `yield` and `dmb oshst` - the outer-shareable store barrier is what
// makes host stores visible to a coherent device (GB10 / Grace NVLink-C2C) before a later store.
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define STRATA_X86 1
inline void strata_cpu_pause() { _mm_pause(); }
inline void strata_store_fence() { _mm_sfence(); }
#elif defined(__aarch64__)
inline void strata_cpu_pause() { __asm__ __volatile__("yield" ::: "memory"); }
inline void strata_store_fence() { __asm__ __volatile__("dmb oshst" ::: "memory"); }
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
// POWER9 (IBM AC922): `or 27,27,27` is the architected low-priority spin hint (the same one Linux's
// cpu_relax() uses there); `lwsync` orders the stores into pinned host memory before the doorbell the GPU polls,
// and is the PowerPC barrier for a coherent NVLink2 link (seq_cst stays the fallback for other architectures).
inline void strata_cpu_pause() { __asm__ __volatile__("or 27,27,27" ::: "memory"); }
inline void strata_store_fence() { __asm__ __volatile__("lwsync" ::: "memory"); }
#else
#include <atomic>
inline void strata_cpu_pause() {}
inline void strata_store_fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
#endif
