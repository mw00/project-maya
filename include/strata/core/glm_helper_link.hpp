#pragma once
// MAYA_GLM_EXPERT_HELPER=link: the GLM expert helper in a second program (Linux).  The lead (this engine, e.g. a CUDA
// build on an NVIDIA card) keeps the trunk; its RAM tier lives in shared memory (memfd) that the helper program (e.g.
// a HIP build on an AMD APU, maya-glm-helper) maps and registers with its own GPU, so the helper reads the experts it
// owns in place - the same ownership, routing and answers as the in-process integrated helper, across two GPU
// runtimes.  Requests and answers go through one shared control block: the input row, the plan (keys, weights) and
// the result row, a sequence number each way (spin-wait).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::hlink {

inline bool wanted() {
#ifdef __linux__
    const char* h = std::getenv("MAYA_GLM_EXPERT_HELPER");
    return h != nullptr && std::strcmp(h, "link") == 0;
#else
    return false;
#endif
}

constexpr uint32_t kMagic = 0x4d474c4bu;   // "MGLK"
constexpr int kMaxEmbd = 8192;
constexpr size_t kStage = (size_t) 64 << 20;   // a replacement's bytes when they are not in the shared tier

enum Op : int32_t { kOpen = 1, kReserve = 2, kInsert = 3, kReplace = 4, kCompute = 5, kQuit = 6, kPfOpen = 7, kPrompt = 8, kComputeX3 = 9, kPromptX3 = 10, kComputeWin = 11 };

// The prompt exchange (a second memfd): one MoE layer of a prompt chunk at a time.  The lead writes the chunk's FFN
// input x (T rows), the sorted rows' positions and weights (pos/rw, T x K) and, for the rows of the experts the helper
// owns, each row's token and each expert's key and row count; the helper computes those experts in place and writes
// the weighted sum of their rows per token into out (T rows), which the lead adds into its FFN output.
constexpr int kPfMaxK = 16, kPfMaxExperts = 2048;
// Ctl::pf_types: what the helper's prompt products do with a ggml type - MMQ multiplies it; the fused kernel decodes
// it as a gate/up, as a down matrix.  A layer's pair: 1 fused (gate/up and down), 2 MMQ's gate/up and the fused
// down, 0 MMQ, -1 no product (the lead keeps such a layer).
constexpr int kPfMmq = 1, kPfFusedGu = 2, kPfFusedDown = 4;
inline int pf_pair_mode(int gu_bits, int d_bits) {
    if ((gu_bits & kPfFusedGu) && (d_bits & kPfFusedDown)) return 1;
    if ((gu_bits & kPfMmq) && (d_bits & kPfFusedDown)) return 2;
    if ((gu_bits & kPfMmq) && (d_bits & kPfMmq)) return 0;
    return -1;
}
struct PfLayout {
    size_t x = 0, out = 0, pos = 0, rw = 0, rtok = 0, keys = 0, cnts = 0, bytes = 0;
    PfLayout(size_t cap_tokens, size_t n_embd) {
        const auto up = [](size_t b) { return (b + 4095) & ~(size_t) 4095; };
        size_t at = 0;
        x = at;    at += up(cap_tokens * n_embd * sizeof(float));
        out = at;  at += up(cap_tokens * n_embd * sizeof(float));
        pos = at;  at += up(cap_tokens * kPfMaxK * sizeof(int32_t));
        rw = at;   at += up(cap_tokens * kPfMaxK * sizeof(float));
        rtok = at; at += up((cap_tokens * kPfMaxK + 512) * sizeof(int32_t));
        keys = at; at += up(kPfMaxExperts * sizeof(uint64_t));   // each expert's blob: a LEAD address (shared tier)
        cnts = at; at += up(kPfMaxExperts * sizeof(int32_t));
        bytes = at;
    }
};
// The exchange's slots: slot 0 = one layout for chunks up to cap tokens; slots 1 and 2 = two layouts for up to cap / 2
// tokens each, side by side in the same bytes (the staggered prompt pipeline keeps one chunk half's layer in each
// while the other's is computed).  base: the slot's first byte in the exchange.
inline PfLayout pf_slot_layout(size_t cap_tokens, size_t n_embd, int slot, size_t* base) {
    if (slot <= 0) {
        *base = 0;
        return PfLayout(cap_tokens, n_embd);
    }
    const PfLayout h(cap_tokens / 2, n_embd);
    *base = (size_t) (slot - 1) * h.bytes;
    return h;
}
inline size_t pf_exchange_bytes(size_t cap_tokens, size_t n_embd) {
    const size_t one = PfLayout(cap_tokens, n_embd).bytes, two = 2 * PfLayout(cap_tokens / 2, n_embd).bytes;
    return one > two ? one : two;
}

// kComputeWin: a verify window's layer in ONE call - every token's own plan (its keys and weights, in its route's
// order), input row and result row; the helper runs the token path's kernels for all of them in one gate/up and one
// down launch (kernels::glmf::moe_experts_multi), so each token's sum is kCompute's for it, bit for bit.  The call
// lives in Ctl::stage (which only a kReplace uses otherwise, never beside a call): the header at 0, the input rows at
// win_x_off(), the result rows at win_out_off(), a row every kMaxEmbd floats.  Ctl::caps says whether the helper
// program has it (kCapWin, kOpen's answer; a helper built before it leaves 0 there).
constexpr uint32_t kCapWin = 1u;
constexpr int kWinTokens = 16;
struct WinCall {
    int32_t T, k;                    // the tokens; the widest plan among them (ne[t] <= k <= 8)
    int32_t ne[kWinTokens];
    int32_t keys[kWinTokens][8];
    float w[kWinTokens][8];
    double gpu_ms;                   // the helper's answer: its time for the call (host clock, ms)
};
constexpr size_t win_x_off() { return 4096; }
constexpr size_t win_out_off() { return win_x_off() + (size_t) kWinTokens * kMaxEmbd * sizeof(float); }

struct alignas(64) Ctl {
    uint32_t magic;
    uint32_t ready;                 // the helper program set it once its regions are mapped and registered
    char pad0[56];
    std::atomic<uint32_t> req;      // the lead: the request's sequence (written last)
    char pad1[60];
    std::atomic<uint32_t> rep;      // the helper: the sequence it answered (written last)
    int32_t status;                 // 0: done; else err holds why
    uint32_t caps;                  // kOpen's answer: what this helper program can do beyond the first calls (kCapWin)
    char pad2[52];
    int32_t op, ne, gu_type, d_type;
    uint64_t down_off;
    float limit;
    int32_t keys[8];
    float w[8];
    int64_t a, b, c, d;
    uint64_t host;                  // a pointer in the LEAD's address space (translated by the helper)
    int32_t pf_T, pf_K, pf_n, pf_rows, pf_r0, pf_dsa, pf_nff, pf_slot;   // kPrompt: the layer's chunk (PfLayout; pf_slot: pf_slot_layout)
    double pf_gpu_ms;               // kPrompt: the helper's time for it (host clock, ms)
    uint8_t pf_types[64];           // kOpen's answer: the helper's prompt products by ggml type (kPfMmq | kPfFusedGu | kPfFusedDown)
    // kOpen: x3_on = 1 for an EXL3 pack's experts (the helper makes its EXL3 kernels' scratch), its answer x3_pf = 1
    // when the helper's prompt products cover EXL3 experts; kComputeX3 / kPromptX3: the layer's blob layout
    // (kernels::exl3::ExpertLayout, field by field) - the call is kCompute's / kPrompt's otherwise
    int32_t x3_on, x3_cb, x3_fmt, x3_k2[3], x3_gg[3], x3_pf;
    uint64_t x3_suh[3], x3_svh[3], x3_trellis[3];
    char err[512];
    float x[kMaxEmbd];
    float out[kMaxEmbd];
    uint8_t stage[kStage];
};

/// lead side: a shared, mapped block of `bytes` (memfd + mmap MAP_SHARED, every page faulted in); nullptr on failure.
/// The caller registers it with its own runtime.
void* shared_alloc(size_t bytes);
/// the shared blocks made so far: base, bytes, memfd
struct Block { void* base; size_t bytes; int fd; };
const std::vector<Block>& blocks();
/// a fresh control block (memfd); its fd in fd
Ctl* ctl_alloc(int& fd);
/// whether [p, p + bytes) lies in one of the shared blocks (the helper program can read it in place)
bool in_shared(const void* p, size_t bytes);
/// lead side: the prompt exchange block (memfd, mapped, faulted in; not in blocks()); nullptr on failure
void* xchg_alloc(size_t bytes, int& fd);
/// helper side: [host, host + bytes) is a block this process's GPU reads at its own address `dev` (the block's pages
/// imported as buffers, not a registration of the host range); device_view: the GPU address standing for host
/// pointer p, nullptr when p lies in no such block (the caller then asks its runtime, as before)
void add_device_view(const void* host, size_t bytes, void* dev);
void* device_view(const void* p);

}  // namespace strata::core::hlink
