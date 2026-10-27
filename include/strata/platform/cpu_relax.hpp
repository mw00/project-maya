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
#else
#include <atomic>
inline void strata_cpu_pause() {}
inline void strata_store_fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
#endif
