#include "strata/core/remote_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <thread>
extern char** environ;
#endif

namespace strata::core {
namespace glmpf {   // glm_prefill_dsa.cu (B14.1: the glm-dsa clamp after MMQ's quantizer, HIP only)
void mmq_clamp(bool on, void* xq, int64_t rows, int64_t cols, int type, cudaStream_t s);
}
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

// GlmExpertHelper::pending_ held for one use of the helper's stream and buffers (a call from enqueue to wait, or a
// replace): released at scope exit only when the stream is known to be idle - nothing was queued (idle from the
// start), or a cudaStreamSynchronize returned cudaSuccess.  Otherwise it stays set (the stream may still read the slots
// or write the caller's buffer): no enqueue or replace runs until close() has synchronized the stream.  keep: the
// claim outlives the scope (a queued call, until its wait).
struct Claim {
    std::atomic<bool>& flag;
    bool idle = true, keep = false;
    ~Claim() {
        if (idle && !keep) flag.store(false, std::memory_order_release);
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace

// (nothing may leave a destructor: caught here, so GCC 14+ emits no __cxa_call_terminate for it - a build whose
// link finds an older libstdc++ first, as CUDA 12.0 with GCC 13 as nvcc's host compiler beside a GCC 15 default,
// lacked that symbol)
GlmExpertHelper::~GlmExpertHelper() {
    try {
        close();
    } catch (...) {
    }
}

bool GlmExpertHelper::open(int device, int n_embd, int n_ff,
                           const std::vector<std::pair<void*, size_t>>& regions, std::string& err, bool exl3) {
    close();
    if (device == kLinkDevice) {
        if (exl3) { err = "GLM expert helper link: native GGUF experts only"; return false; }
        return open_link(n_embd, n_ff, regions, err);
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    int integrated = 0;
    if (!check(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device), "integrated memory", err, device))
        return false;
    if (n_embd <= 0 || n_embd > 8192 || n_embd % 32 || n_ff <= 0 || n_ff % 32) {
        err = "GLM expert helper: unsupported expert dimensions";
        return false;
    }
    device_ = device;
    integrated_ = integrated != 0;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    if (integrated_) for (const auto& [host, bytes] : regions) {
        void* mapped = nullptr;
        if (!check(cudaHostGetDevicePointer(&mapped, host, 0), "map RAM tier", err, device)) return false;
        regions_.push_back({(const uint8_t*) host, (uint8_t*) mapped, bytes});
    }
    const size_t need = 2 * (size_t) n_embd * sizeof(float) + (size_t) (n_embd + 8 * n_ff) / 32 * 36 + 1024;
    if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "helper stream", err, device) ||
        !check(cudaMalloc((void**) &arena_, need), "helper scratch", err, device)) return false;
    // EXL3: the stream's kernel scratch now (the first launch would make it), before reserve() sizes the cache
    if (exl3 && !strata::kernels::exl3::prepare(stream_)) {
        err = "CUDA" + std::to_string(device) + " experts: the EXL3 kernels' scratch did not allocate";
        return false;
    }
    uint8_t* at = arena_;
    const auto take = [&](size_t bytes) {
        void* p = at;
        at += (bytes + 63u) & ~(size_t) 63u;
        return p;
    };
    x_ = (float*) take((size_t) n_embd * sizeof(float));
    out_ = (float*) take((size_t) n_embd * sizeof(float));
    xq_ = take((size_t) n_embd / 32 * 36);
    hq_ = take((size_t) 8 * n_ff / 32 * 36);
    plan_.plan_ptr = (unsigned long long*) take(8 * sizeof(unsigned long long));
    plan_.plan_w = (float*) take(8 * sizeof(float));
    plan_.cpu_flag = (int*) take(64);
    if (!check(cudaMemset(plan_.cpu_flag, 0, 64), "helper flags", err, device)) return false;
    // zero-copy on an integrated helper (the Qwen helper's MAYA_REMOTE_ZEROCOPY pattern): its device reads the host's
    // memory as its own, so the input, the plan and the result stay in one mapped pinned block
    const char* zc = std::getenv("MAYA_GLM_HELPER_ZEROCOPY");
    if (integrated_ && !(zc != nullptr && zc[0] == '0')) {
        const size_t xb = (size_t) n_embd * sizeof(float);
        void* block = nullptr;
        if (!check(cudaHostAlloc(&block, 2 * xb + 8 * sizeof(unsigned long long) + 8 * sizeof(float),
                                 cudaHostAllocPortable | cudaHostAllocMapped), "helper zero-copy block", err, device))
            return false;
        zc_host_ = (uint8_t*) block;
        x_h_ = (float*) zc_host_;
        out_h_ = (float*) (zc_host_ + xb);
        plan_ptr_h_ = (unsigned long long*) (zc_host_ + 2 * xb);
        plan_w_h_ = (float*) (zc_host_ + 2 * xb + 8 * sizeof(unsigned long long));
        uint8_t* z = nullptr;
        if (!check(cudaHostGetDevicePointer((void**) &z, block, 0), "helper zero-copy mapping", err, device)) return false;
        x_z_ = (float*) z;
        out_z_ = (float*) (z + xb);
        zplan_ = plan_;
        zplan_.plan_ptr = (unsigned long long*) (z + 2 * xb);
        zplan_.plan_w = (float*) (z + 2 * xb + 8 * sizeof(unsigned long long));
        zero_copy_ = true;
    }
    prof_ = std::getenv("MAYA_GLM_PROF") != nullptr;
    for (auto& e : ev_)
        if (prof_ && !check(cudaEventCreate(&e), "helper profile event", err, device)) return false;
    return true;
}

bool GlmExpertHelper::reserve(int keys, size_t requested_bytes, std::string& err) {
    if (link_) {
        ctl_->op = hlink::kReserve;
        ctl_->a = keys;
        ctl_->b = (int64_t) requested_bytes;
        if (!link_post(err) || !link_wait(err)) return false;
        slots_.assign((size_t) keys, {});
        owned_.assign((size_t) keys, 0);
        capacity_ = requested_bytes;
        return true;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    slots_.assign((size_t) keys, {});
    owned_.assign((size_t) keys, 0);
    capacity_ = requested_bytes;
    if (!integrated_) {
        size_t free = 0, total = 0;
        if (!check(cudaMemGetInfo(&free, &total), "helper capacity", err, device_)) return false;
        constexpr size_t headroom = 512ull << 20;   // PR #578's automatic VRAM budget
        capacity_ = std::min(capacity_, free > headroom ? free - headroom : 0);
        capacity_ &= ~(size_t) 255;
        if (capacity_ && !check(cudaMalloc((void**) &pool_, capacity_), "helper cache", err, device_)) return false;
    }
    return true;
}

bool GlmExpertHelper::insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err) {
    if (key < 0 || (size_t) key >= slots_.size() || owns(key) || bytes > stride || stride > available()) {
        err = "GLM expert helper: invalid cache insertion"; return false;
    }
    Slot s{const_cast<uint8_t*>(host), nullptr, bytes};
    if (link_) {
        ctl_->op = hlink::kInsert;
        ctl_->a = key;
        ctl_->b = (int64_t) bytes;
        ctl_->c = (int64_t) stride;
        ctl_->host = (uint64_t) (uintptr_t) host;
        if (!link_post(err) || !link_wait(err)) return false;
        slots_[(size_t) key] = s;
        owned_[(size_t) key] = 1;
        used_ += stride;
        return true;
    }
    if (integrated_) {
        for (const auto& r : regions_) {
            const uintptr_t offset = (uintptr_t) host - (uintptr_t) r.host;
            if ((uintptr_t) host >= (uintptr_t) r.host && offset <= r.bytes && bytes <= r.bytes - offset) {
                s.device = r.mapped + offset; break;
            }
        }
        if (!s.device) { err = "GLM expert helper: expert outside the mapped RAM tier"; return false; }
    } else {
        DeviceScope scope(device_);
        if (!scope.ok) { err = scope.error(device_); return false; }
        s.device = pool_ + used_;
        if (!check(cudaMemcpyAsync(s.device, host, bytes, cudaMemcpyHostToDevice, stream_), "cache upload", err, device_) ||
            !check(cudaStreamSynchronize(stream_), "cache upload finish", err, device_)) return false;
    }
    slots_[(size_t) key] = s;
    owned_[(size_t) key] = 1;
    used_ += stride;
    return true;
}

bool GlmExpertHelper::replace(int out, int in, const uint8_t* host, std::string& err) {
    // (the slots a pending call reads, and the helper's stream: wait first; held until this copy has landed)
    bool free_now = false;
    if (!pending_.compare_exchange_strong(free_now, true, std::memory_order_acq_rel)) {
        err = "GLM expert helper: cache replacement while a call is pending"; return false;
    }
    Claim claim{pending_};
    if (out < 0 || (size_t) out >= slots_.size() || in < 0 || (size_t) in >= slots_.size() || !owns(out) || owns(in)) {
        err = "GLM expert helper: invalid cache replacement"; return false;
    }
    Slot s = slots_[(size_t) out];
    if (link_) {
        // the helper program swaps its own slots; the bytes travel only when they are not in the shared tier
        bool shared = false;
        for (const auto& b : hlink::blocks()) {
            const uintptr_t off = (uintptr_t) host - (uintptr_t) b.base;
            if ((uintptr_t) host >= (uintptr_t) b.base && off <= b.bytes && s.bytes <= b.bytes - off) { shared = true; break; }
        }
        if (!shared && s.bytes > hlink::kStage) { err = "GLM expert helper link: expert larger than the stage"; return false; }
        if (!shared) std::memcpy(ctl_->stage, host, s.bytes);
        ctl_->op = hlink::kReplace;
        ctl_->a = out;
        ctl_->b = in;
        ctl_->d = shared ? 0 : 1;
        ctl_->host = (uint64_t) (uintptr_t) host;
        if (!link_post(err) || !link_wait(err)) return false;
        slots_[(size_t) in] = {const_cast<uint8_t*>(host), nullptr, s.bytes};
        slots_[(size_t) out] = {};
        owned_[(size_t) out] = 0;
        owned_[(size_t) in] = 1;
        return true;
    }
    if (integrated_) {
        uint8_t* mapped = nullptr;
        for (const auto& r : regions_) {
            const uintptr_t offset = (uintptr_t) host - (uintptr_t) r.host;
            if ((uintptr_t) host >= (uintptr_t) r.host && offset <= r.bytes && s.bytes <= r.bytes - offset) {
                mapped = r.mapped + offset; break;
            }
        }
        // Already in shared RAM: exchange ownership, not bytes. The old expert
        // stays in the host tier, including in --glm-ram-resident mode.
        if (mapped) s = {const_cast<uint8_t*>(host), mapped, s.bytes};
        else std::memcpy(s.host, host, s.bytes);   // a disk-only candidate uses the existing slot
    }
    else {
        DeviceScope scope(device_);
        if (!scope.ok) { err = scope.error(device_); return false; }
        if (!check(cudaMemcpyAsync(s.device, host, s.bytes, cudaMemcpyHostToDevice, stream_), "cache replace", err, device_))
            return false;
        claim.idle = false;
        const cudaError_t done = cudaStreamSynchronize(stream_);
        claim.idle = done == cudaSuccess;
        if (!check(done, "cache replace finish", err, device_)) return false;
    }
    slots_[(size_t) in] = s;
    slots_[(size_t) out] = {};
    owned_[(size_t) out] = 0;
    owned_[(size_t) in] = 1;
    return true;
}

// PR #578: complement the primary cache, replacing a cold helper expert only
// with a hotter unowned expert of the SAME layer. Pending primary admissions
// are included in `primary` by the caller before this idle-boundary operation.
std::vector<GlmExpertHelper::Swap> GlmExpertHelper::swaps(const std::vector<uint32_t>& usage,
        const std::vector<int>& primary, int ne, int max_swaps) const {
    std::vector<Swap> result;
    for (size_t base = 0; base < owned_.size(); base += ne) {
        std::vector<std::pair<uint32_t, int>> candidates, victims;
        uint32_t primary_cold = UINT32_MAX;
        for (int e = 0; e < ne; ++e) {
            const int key = (int) base + e;
            if (primary[(size_t) key] >= 0) primary_cold = std::min(primary_cold, usage[(size_t) key]);
            if (owns(key)) victims.emplace_back(usage[(size_t) key], key);
            else if (primary[(size_t) key] < 0 && usage[(size_t) key] >= 2) candidates.emplace_back(usage[(size_t) key], key);
        }
        std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.first > b.first; });
        std::sort(victims.begin(), victims.end());
        for (size_t i = 0; i < std::min(candidates.size(), victims.size()); ++i) {
            if (victims[i].first > primary_cold || candidates[i].first < victims[i].first + 2u) break;
            result.push_back({candidates[i].first - victims[i].first, candidates[i].second, victims[i].second});
        }
    }
    std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.gain > b.gain; });
    if ((int) result.size() > max_swaps) result.resize((size_t) max_swaps);
    return result;
}

bool GlmExpertHelper::compute(int gu_type, int d_type, size_t down_off, float limit, int ne,
                              const int* keys, const float* weights, const float* x,
                              float* out, std::string& err) {
    return start(nullptr, gu_type, d_type, down_off, limit, ne, keys, weights, x, out, err) && finish(err, false);
}

bool GlmExpertHelper::compute_exl3(const strata::kernels::exl3::ExpertLayout& lay, float limit, int ne,
                                   const int* keys, const float* weights, const float* x,
                                   float* out, std::string& err) {
    return start(&lay, 0, 0, 0, limit, ne, keys, weights, x, out, err) && finish(err, false);
}

bool GlmExpertHelper::enqueue(int gu_type, int d_type, size_t down_off, float limit, int ne,
                              const int* keys, const float* weights, const float* x,
                              float* out, std::string& err) {
    return start(nullptr, gu_type, d_type, down_off, limit, ne, keys, weights, x, out, err);
}

bool GlmExpertHelper::enqueue_exl3(const strata::kernels::exl3::ExpertLayout& lay, float limit, int ne,
                                   const int* keys, const float* weights, const float* x,
                                   float* out, std::string& err) {
    return start(&lay, 0, 0, 0, limit, ne, keys, weights, x, out, err);
}

bool GlmExpertHelper::wait(std::string& err) { return finish(err, true); }

// The first half of a call - xl: an EXL3 pack's layout (compute_exl3), null: native GGUF experts of gu_type / d_type
// (compute): the input and the plan up, the kernels, and the result's copy into out, all on the helper's stream.  The
// call holds pending_ from here to its wait (Claim), so no replace or other call overlaps it.  A failure after anything
// was queued waits for the stream (nothing may land in out later; on the helper's device, current here) and leaves
// nothing pending - unless that wait failed too (the claim then stays until close()).
bool GlmExpertHelper::start(const strata::kernels::exl3::ExpertLayout* xl, int gu_type, int d_type, size_t down_off,
                            float limit, int ne, const int* keys, const float* weights, const float* x,
                            float* out, std::string& err) {
    namespace gf = strata::kernels::glmf;
    namespace x3 = strata::kernels::exl3;
    bool free_now = false;
    if (!pending_.compare_exchange_strong(free_now, true, std::memory_order_acq_rel)) {
        err = "GLM expert helper: a call before this one was not waited for, or a cache replacement runs";
        return false;
    }
    Claim claim{pending_};
    if (device_ < 0 || ne < 1 || ne > 8 ||
        (xl == nullptr && (!gf::moe_supported(gu_type) || !gf::moe_supported(d_type)))) {
        err = "GLM expert helper: unsupported expert types or route width";
        return false;
    }
    unsigned long long ptr[8] = {};
    for (int i = 0; i < ne; ++i) {
        if (keys[i] < 0 || (size_t) keys[i] >= slots_.size() || !owns(keys[i])) {
            err = "GLM expert helper: expert not owned by helper"; return false;
        }
        ptr[i] = (unsigned long long) slots_[(size_t) keys[i]].device;
    }
    size_t bytes = 0;
    for (int i = 0; i < ne; ++i) bytes += slots_[(size_t) keys[i]].bytes;
    if (link_) {
        if (xl != nullptr) { err = "GLM expert helper link: native GGUF experts only"; return false; }
        const auto t0 = std::chrono::steady_clock::now();
        ctl_->op = hlink::kCompute;
        ctl_->ne = ne;
        ctl_->gu_type = gu_type;
        ctl_->d_type = d_type;
        ctl_->down_off = down_off;
        ctl_->limit = limit;
        for (int i = 0; i < ne; ++i) ctl_->keys[i] = keys[i], ctl_->w[i] = weights[i];
        std::memcpy(ctl_->x, x, (size_t) n_embd_ * sizeof(float));
        if (!link_post(err)) return false;
        pend_ = {};
        pend_.ne = ne;
        pend_.bytes = bytes;
        pend_.out = out;
        pend_.t0 = t0;
        pend_.t1 = std::chrono::steady_clock::now();
        claim.keep = true;
        return true;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto t0 = std::chrono::steady_clock::now();
    // (EXL3: the family's launch failures only print and count - DS4_HELPER's check, around this call's kernels)
    const int x3_errors = xl != nullptr ? x3::launch_errors() : 0;
    const auto failed = [&] {   // (err is set)
        claim.idle = cudaStreamSynchronize(stream_) == cudaSuccess;
        return false;
    };
    if (prof_ && !check(cudaEventRecord(ev_[0], stream_), "helper profile event", err, device_)) return failed();
    if (zero_copy_) {
        std::memcpy(x_h_, x, (size_t) n_embd_ * sizeof(float));
        std::memcpy(plan_ptr_h_, ptr, (size_t) ne * sizeof(ptr[0]));
        std::memcpy(plan_w_h_, weights, (size_t) ne * sizeof(float));
    } else if (!check(cudaMemcpyAsync(x_, x, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                      "helper input", err, device_) ||
               !check(cudaMemcpyAsync(plan_.plan_ptr, ptr, (size_t) ne * sizeof(ptr[0]), cudaMemcpyHostToDevice, stream_),
                      "helper pointers", err, device_) ||
               !check(cudaMemcpyAsync(plan_.plan_w, weights, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, stream_),
                      "helper weights", err, device_)) return failed();
    const gf::MoeDev& plan = zero_copy_ ? zplan_ : plan_;
    if (prof_ && !check(cudaEventRecord(ev_[1], stream_), "helper profile event", err, device_)) return failed();
    if (xl != nullptr) {
        // EXL3: gate/up/swiglu and down over the F32 input (no quantize: its stage times 0), the weighted sum in plan
        // order - the trunk's own decode call (glm_fast_path.cu's fast_moe_experts) without the CPU part and the
        // shared expert; h in the stream's scratch
        x3::MoeArgs xa;
        xa.plan_ptr = plan.plan_ptr;
        xa.plan_w = plan.plan_w;
        xa.lay = *xl;
        xa.k = ne;
        xa.n_embd = n_embd_;
        xa.n_ff = n_ff_;
        xa.limit = limit;
        if (prof_ && !check(cudaEventRecord(ev_[2], stream_), "helper profile event", err, device_)) return failed();
        x3::moe_gate_up(xa, zero_copy_ ? x_z_ : x_, nullptr, stream_);
        if (prof_ && !check(cudaEventRecord(ev_[3], stream_), "helper profile event", err, device_)) return failed();
        x3::moe_down(xa, nullptr, nullptr, zero_copy_ ? out_z_ : out_, stream_);
    } else {
        gf::quantize_q8_1(zero_copy_ ? x_z_ : x_, xq_, n_embd_, stream_);
        if (prof_ && !check(cudaEventRecord(ev_[2], stream_), "helper profile event", err, device_)) return failed();
        gf::moe_gate_up(gu_type, plan, ne, n_embd_, n_ff_, limit, xq_, hq_, nullptr, 0, nullptr, 0, nullptr, stream_);
        if (prof_ && !check(cudaEventRecord(ev_[3], stream_), "helper profile event", err, device_)) return failed();
        gf::moe_down(d_type, plan, ne, n_embd_, n_ff_, down_off, hq_, nullptr, zero_copy_ ? out_z_ : out_, stream_);
    }
    if (prof_ && !check(cudaEventRecord(ev_[4], stream_), "helper profile event", err, device_)) return failed();
    if (!zero_copy_ && !check(cudaMemcpyAsync(out, out_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                              "helper result", err, device_)) return failed();
    if (prof_ && !check(cudaEventRecord(ev_[5], stream_), "helper profile event", err, device_)) return failed();
    pend_ = Pending{xl != nullptr, x3_errors, ne, bytes, out, t0, std::chrono::steady_clock::now()};
    claim.keep = true;   // (finish() releases it)
    return true;
}

// The second half: wait for the stream, then the EXL3 launch check, the zero-copy result's copy and the profile.
// split: a caller's own wait (enqueue / wait) - the time between the two halves counts apart (the caller's work
// beside the helper), not as waiting.  The call's claim on pending_ is released on the way out, only once the stream
// has been synchronized: nothing is pending afterwards, whatever the outcome - unless no synchronize succeeded (the
// claim then stays until close(), and out may still be written).
bool GlmExpertHelper::finish(std::string& err, bool split) {
    namespace x3 = strata::kernels::exl3;
    if (!pending_.load(std::memory_order_acquire)) {
        err = "GLM expert helper: nothing to wait for";
        return false;
    }
    Claim claim{pending_, false};   // (this call's, taken in start(): released when the stream is idle)
    const Pending p = pend_;
    if (link_) {
        const auto tw = split ? std::chrono::steady_clock::now() : p.t1;
        const bool ok = link_wait(err);
        claim.idle = ok;   // (a lost helper program keeps the claim: nothing more goes to it)
        if (!ok) return false;
        std::memcpy(p.out, ctl_->out, (size_t) n_embd_ * sizeof(float));
        if (prof_) {
            const auto t3 = std::chrono::steady_clock::now();
            enqueue_ms_ += std::chrono::duration<double, std::milli>(p.t1 - p.t0).count();
            wait_ms_ += std::chrono::duration<double, std::milli>(t3 - tw).count();
            ++calls_;
            experts_ += (unsigned long long) p.ne;
            bytes_ += (unsigned long long) p.bytes;
            if (split) {
                ++split_calls_;
                beside_ms_ += std::chrono::duration<double, std::milli>(tw - p.t1).count();
            }
        }
        return true;
    }
    DeviceScope scope(device_);
    if (!scope.ok) {
        // The helper's device could not be made current, yet nothing of this call may land in out (or read a slot)
        // once the caller has seen it fail: synchronize the stream from the device that is current.  A stream belongs
        // to the device (context) it was made on, and synchronizing it waits on that context whichever is current -
        // the multi-device rules restrict kernel launches and cudaEventRecord to the stream's device, not waits (cf.
        // cudaEventSynchronize / cudaStreamWaitEvent across devices; cuStreamSynchronize takes no context).  Should
        // that wait fail all the same, the claim stays: no enqueue or replace touches the stream until close().
        claim.idle = cudaStreamSynchronize(stream_) == cudaSuccess;
        err = scope.error(device_);
        return false;
    }
    const auto tw = split ? std::chrono::steady_clock::now() : p.t1;
    const cudaError_t done = cudaStreamSynchronize(stream_);
    claim.idle = done == cudaSuccess;
    if (!check(done, "helper finish", err, device_)) return false;
    if (p.exl3 && x3::launch_errors() != p.x3_errors) {
        err = "CUDA" + std::to_string(device_) + " experts: the helper's EXL3 kernels did not run (exl3 launch errors " +
              std::to_string(p.x3_errors) + " -> " + std::to_string(x3::launch_errors()) + ")";
        return false;
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (zero_copy_) std::memcpy(p.out, out_h_, (size_t) n_embd_ * sizeof(float));   // (the result's copy, on the host)
    const auto t3 = std::chrono::steady_clock::now();
    last_gpu_ms_ = -1.0;
    if (prof_) {
        float ms[5] = {};
        for (int k = 0; k < 5; ++k)
            if (!check(cudaEventElapsedTime(&ms[k], ev_[k], ev_[k + 1]), "helper profile timing", err, device_)) return false;
        last_gpu_ms_ = 0.0;
        for (int k = 0; k < 5; ++k) stage_ms_[k] += ms[k], last_gpu_ms_ += ms[k];
        enqueue_ms_ += std::chrono::duration<double, std::milli>(p.t1 - p.t0).count();
        wait_ms_ += std::chrono::duration<double, std::milli>(t2 - tw).count();
        out_ms_ += std::chrono::duration<double, std::milli>(t3 - t2).count();
        ++calls_;
        experts_ += (unsigned long long) p.ne;
        bytes_ += (unsigned long long) p.bytes;
        if (split) {
            ++split_calls_;
            beside_ms_ += std::chrono::duration<double, std::milli>(tw - p.t1).count();
        }
    }
    return true;
}

void GlmExpertHelper::close() {
    if (device_ < 0) return;
    if (link_) {
        if (prof_ && calls_ > 0)
            std::fprintf(stderr, "glm helper link prof: %llu calls, %.2f experts a call: host %.3f ms a call (post %.3f, "
                                 "wait %.3f), %llu split calls, %.3f ms a split call between the halves\n", calls_,
                         (double) experts_ / (double) calls_, (enqueue_ms_ + wait_ms_) / (double) calls_,
                         enqueue_ms_ / (double) calls_, wait_ms_ / (double) calls_, split_calls_,
                         split_calls_ ? beside_ms_ / (double) split_calls_ : 0.0);
        close_link();
        calls_ = experts_ = bytes_ = split_calls_ = 0;
        enqueue_ms_ = wait_ms_ = out_ms_ = beside_ms_ = 0;
        pending_.store(false, std::memory_order_release);
        pend_ = {};
        device_ = -1;
        capacity_ = used_ = 0;
        slots_.clear();
        owned_.clear();
        return;
    }
    {   // the helper program's prompt scratch
        DeviceScope scope(device_);
        if (stream_ != nullptr && pf_scratch_ != nullptr) cudaStreamSynchronize(stream_);
        if (pf_scratch_ != nullptr) cudaFree(pf_scratch_);
        if (pf_bnd_h_ != nullptr) cudaFreeHost(pf_bnd_h_);
        delete (strata::prefill::mmq::Context*) pf_mq_;
        pf_scratch_ = nullptr;
        pf_bnd_h_ = pf_bnd_d_ = nullptr;
        pf_mq_ = nullptr;
        pf_scratch_bytes_ = 0;
        pf_win_ = 0;
        pf_ = pf_dev_ = nullptr;
        pf_cap_ = 0;
    }
    if (prof_ && calls_ > 0) {
        const double n = (double) calls_, kernels_ms = stage_ms_[2] + stage_ms_[3];
        std::fprintf(stderr, "glm helper prof CUDA%d: %llu calls, %.2f experts a call, zero-copy %s: host %.3f ms a call "
                             "(enqueue %.3f, wait %.3f, result copy %.3f); GPU copy-in %.3f, quantize %.3f, gate_up %.3f, down "
                             "%.3f, copy-out %.3f ms a call; the experts' bytes over gate_up + down at %.1f GB/s\n",
                     device_, calls_, (double) experts_ / n, zero_copy_ ? "on" : "off", (enqueue_ms_ + wait_ms_ + out_ms_) / n,
                     enqueue_ms_ / n, wait_ms_ / n, out_ms_ / n, stage_ms_[0] / n, stage_ms_[1] / n, stage_ms_[2] / n,
                     stage_ms_[3] / n, stage_ms_[4] / n, kernels_ms > 0.0 ? (double) bytes_ / (kernels_ms * 1e6) : 0.0);
        if (split_calls_ > 0)   // (enqueue / wait: the caller's own work ran between the two halves)
            std::fprintf(stderr, "glm helper prof CUDA%d: %llu of the calls split (enqueue, the caller's work, wait): %.3f ms "
                                 "a split call between the two halves\n",
                         device_, split_calls_, beside_ms_ / (double) split_calls_);
    }
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        if (arena_) cudaFree(arena_);
        if (pool_) cudaFree(pool_);
        if (stream_) cudaStreamDestroy(stream_);
        for (auto& e : ev_)
            if (e) cudaEventDestroy(e);
        if (zc_host_) cudaFreeHost(zc_host_);
    }
    zero_copy_ = prof_ = false;
    zc_host_ = nullptr;
    x_h_ = out_h_ = x_z_ = out_z_ = plan_w_h_ = nullptr;
    plan_ptr_h_ = nullptr;
    zplan_ = {};
    for (auto& e : ev_) e = nullptr;
    calls_ = experts_ = bytes_ = split_calls_ = 0;
    enqueue_ms_ = wait_ms_ = out_ms_ = beside_ms_ = 0;
    last_gpu_ms_ = -1.0;
    pending_.store(false, std::memory_order_release);
    pend_ = {};
    for (auto& v : stage_ms_) v = 0;
    device_ = -1;
    arena_ = nullptr;
    pool_ = nullptr;
    capacity_ = used_ = 0;
    stream_ = nullptr;
    regions_.clear();
    slots_.clear();
    owned_.clear();
    plan_ = {};
}

// ---------------------------------------------------------------- the helper in a second program (Linux)
bool GlmExpertHelper::open_link(int n_embd, int n_ff, const std::vector<std::pair<void*, size_t>>& regions,
                                std::string& err) {
#ifdef __linux__
    if (n_embd <= 0 || n_embd > hlink::kMaxEmbd || n_ff <= 0) {
        err = "GLM expert helper link: unsupported dimensions";
        return false;
    }
    const char* exe = std::getenv("MAYA_GLM_HELPER_EXE");
    if (exe == nullptr || exe[0] == '\0') {
        err = "GLM expert helper link: MAYA_GLM_HELPER_EXE names no helper program (maya-glm-helper of a HIP build)";
        return false;
    }
    // every region must lie in a shared block (the RAM tier the link allocated): the helper reads them in place
    const auto& blocks = hlink::blocks();
    std::vector<int> used(blocks.size(), 0);
    for (const auto& [host, bytes] : regions) {
        bool in = false;
        for (size_t i = 0; i < blocks.size() && !in; ++i) {
            const uintptr_t off = (uintptr_t) host - (uintptr_t) blocks[i].base;
            if ((uintptr_t) host >= (uintptr_t) blocks[i].base && off <= blocks[i].bytes && bytes <= blocks[i].bytes - off)
                in = true, used[i] = 1;
        }
        if (!in) {
            err = "GLM expert helper link: a RAM-tier region outside the shared memory";
            return false;
        }
    }
    ctl_ = hlink::ctl_alloc(ctl_fd_);
    if (ctl_ == nullptr) {
        err = "GLM expert helper link: the control block did not allocate";
        return false;
    }
    std::vector<std::string> args = {exe, "--ctl", std::to_string(ctl_fd_)};
    double gb = 0;
    for (size_t i = 0; i < used.size(); ++i)
        if (used[i]) {
            args.push_back("--region");
            args.push_back(std::to_string(blocks[i].fd) + ":" + std::to_string(blocks[i].bytes) + ":" +
                           std::to_string((uintptr_t) blocks[i].base));
            gb += blocks[i].bytes / 1073741824.0;
        }
    // the prompt exchange (MAYA_GLM_LINK_PROMPT=0: none - prompts then stream the experts the lead lacks, as before):
    // MAYA_GLM_LINK_PF_TOKENS chunk tokens at most (default 32768; a longer chunk's layers stream as before)
    {
        const char* pv = std::getenv("MAYA_GLM_LINK_PROMPT");
        const char* tv = std::getenv("MAYA_GLM_LINK_PF_TOKENS");
        const int cap = tv != nullptr ? std::atoi(tv) : 32768;
        if (!(pv != nullptr && std::strcmp(pv, "0") == 0) && cap > 0) {
            pf_lay_ = hlink::PfLayout((size_t) cap, (size_t) n_embd);
            const size_t xb = hlink::pf_exchange_bytes((size_t) cap, (size_t) n_embd);   // (slot 0, or slots 1 + 2)
            pf_ = (uint8_t*) hlink::xchg_alloc(xb, pf_fd_);
            if (pf_ != nullptr) {
                pf_bytes_ = xb;
                pf_cap_ = cap;
                args.push_back("--xchg");
                args.push_back(std::to_string(pf_fd_) + ":" + std::to_string(pf_bytes_) + ":" + std::to_string(cap));
            }
        }
    }
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid = -1;
    if (posix_spawn(&pid, exe, nullptr, nullptr, argv.data(), environ) != 0) {
        err = std::string("GLM expert helper link: could not start ") + exe;
        close_link();
        return false;
    }
    child_ = (int) pid;
    link_ = true;
    device_ = kLinkDevice;
    integrated_ = true;   // (it reads the shared RAM tier in place: the integrated helper's ownership rules)
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    prof_ = std::getenv("MAYA_GLM_PROF") != nullptr;
    std::fprintf(stderr, "glm helper link: started %s (pid %d) over %.2f GB of shared RAM tier\n", exe, child_, gb);
    // its registration of the shared tier takes a while (tens of GB): wait for ready, or its end
    for (;;) {
        if (__atomic_load_n(&ctl_->ready, __ATOMIC_ACQUIRE) != 0) break;
        int st = 0;
        if (waitpid(child_, &st, WNOHANG) == child_) {
            child_ = -1;
            err = "GLM expert helper link: the helper program ended while starting";
            close();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ctl_->op = hlink::kOpen;
    ctl_->a = n_embd;
    ctl_->b = n_ff;
    if (!link_post(err) || !link_wait(err)) {
        close();
        return false;
    }
    // the exchange as this GPU reads it (out is added into the FFN in place; x and the plan are copied down into it)
    if (pf_ != nullptr) {
        void* dp = nullptr;
        if (cudaHostRegister(pf_, pf_bytes_, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess &&
            cudaHostGetDevicePointer(&dp, pf_, 0) == cudaSuccess) {
            pf_registered_ = true;
            pf_dev_ = (uint8_t*) dp;
            std::fprintf(stderr, "glm helper link: prompt exchange %.2f GB (chunks up to %d tokens): the helper computes "
                                 "its own experts' prompt rows\n", pf_bytes_ / 1073741824.0, pf_cap_);
        } else {
            (void) cudaGetLastError();
            pf_cap_ = 0;
            std::fprintf(stderr, "glm helper link: the prompt exchange did not register - prompts stream as before\n");
        }
    }
    return true;
#else
    (void) n_embd;
    (void) n_ff;
    (void) regions;
    err = "GLM expert helper link: Linux only";
    return false;
#endif
}

bool GlmExpertHelper::link_post(std::string& err) {
    if (ctl_ == nullptr || child_ < 0) {
        err = "GLM expert helper link: the helper program is gone";
        return false;
    }
    ctl_->req.store(++seq_, std::memory_order_release);
    return true;
}

bool GlmExpertHelper::link_wait(std::string& err) {
#ifdef __linux__
    if (ctl_ == nullptr) {
        err = "GLM expert helper link: not open";
        return false;
    }
    for (uint64_t spin = 0;; ++spin) {
        if (ctl_->rep.load(std::memory_order_acquire) == seq_) break;
        if ((spin & 0xffff) == 0xffff) {
            int st = 0;
            if (child_ < 0 || waitpid(child_, &st, WNOHANG) == child_) {
                child_ = -1;
                err = "GLM expert helper link: the helper program ended";
                return false;
            }
            if (spin > (1ull << 22)) std::this_thread::yield();
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    if (ctl_->status != 0) {
        ctl_->err[sizeof ctl_->err - 1] = '\0';
        err = std::string("GLM expert helper link: ") + ctl_->err;
        return false;
    }
    return true;
#else
    err = "GLM expert helper link: Linux only";
    return false;
#endif
}

void GlmExpertHelper::close_link() {
#ifdef __linux__
    if (ctl_ != nullptr && child_ >= 0) {
        ctl_->op = hlink::kQuit;
        ctl_->req.store(++seq_, std::memory_order_release);
        for (int i = 0; i < 250; ++i) {   // (5 s, then it is stopped)
            int st = 0;
            if (waitpid(child_, &st, WNOHANG) == child_) {
                child_ = -1;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (child_ >= 0) {
            kill(child_, SIGKILL);
            int st = 0;
            waitpid(child_, &st, 0);
            child_ = -1;
        }
    }
    if (ctl_ != nullptr) munmap(ctl_, sizeof(hlink::Ctl));
    if (ctl_fd_ >= 0) ::close(ctl_fd_);
    if (pf_registered_) cudaHostUnregister(pf_);
    if (pf_ != nullptr) munmap(pf_, pf_bytes_);
    if (pf_fd_ >= 0) ::close(pf_fd_);
#endif
    pf_ = pf_dev_ = nullptr;
    pf_bytes_ = 0;
    pf_cap_ = 0;
    pf_fd_ = -1;
    pf_registered_ = false;
    ctl_ = nullptr;
    ctl_fd_ = -1;
    child_ = -1;
    link_ = false;
    integrated_ = false;
}

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; MAYA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP)
    const char* spin = std::getenv("MAYA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_meta_) cudaFree(d_meta_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    device_ = -1;
    stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err) {
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts) {
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) slots);
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            selected.push_back(pair);
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) { err = "CUDA" + std::to_string(device) + " experts: no unclaimed experts remain"; close(); return false; }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    for (const auto& pair : selected)
        needed += lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, sizeof(RemoteMeta), cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, sizeof(RemoteMeta)), "group metadata", err, device);
    if (!allocated) { close(); return false; }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  MAYA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("MAYA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta), cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(zero_copy_ ? z_x_ : d_x_, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t) layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        if (!strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s)) {
            err = "CUDA" + std::to_string(device_) + " experts: no device kernel for layer " + std::to_string(layer) +
                  "'s expert types";   // glm-dsa B3.5: was std::exit in the kernel
            return false;
        }
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) dst_.size() * H * sizeof(float);
    if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (group_id_.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

// ---- the linked helper's prompt exchange (glm_helper_link.hpp PfLayout) ----

bool GlmExpertHelper::pf_post(int gu_type, int d_type, size_t down_off, float limit, bool dsa, int T, int K, int n,
                              int rows, int r0, std::string& err, int slot) {
    if (!link_ || ctl_ == nullptr || pf_cap_ <= 0 || T > (slot > 0 ? pf_slot_cap() : pf_cap_) || slot < 0 || slot > 2 || K > hlink::kPfMaxK || n > hlink::kPfMaxExperts) {
        err = "GLM expert helper link: no prompt exchange for this layer";
        return false;
    }
    ctl_->op = hlink::kPrompt;
    ctl_->gu_type = gu_type;
    ctl_->d_type = d_type;
    ctl_->down_off = down_off;
    ctl_->limit = limit;
    ctl_->pf_T = T;
    ctl_->pf_K = K;
    ctl_->pf_n = n;
    ctl_->pf_rows = rows;
    ctl_->pf_r0 = r0;
    ctl_->pf_dsa = dsa ? 1 : 0;
    ctl_->pf_slot = slot;   // (the exchange slot: glm_helper_link.hpp pf_slot_layout)
    ctl_->pf_gpu_ms = 0.0;
    return link_post(err);
}

bool GlmExpertHelper::pf_wait(std::string& err, double* helper_ms) {
    if (!link_wait(err)) return false;
    if (helper_ms != nullptr) *helper_ms = ctl_->pf_gpu_ms;
    return true;
}

bool GlmExpertHelper::pf_attach(uint8_t* host, size_t bytes, int cap_tokens, std::string& err) {
    if (!integrated_ || device_ < 0 || cap_tokens <= 0) {
        err = "GLM expert helper: the prompt exchange needs an open integrated helper";
        return false;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    void* dp = nullptr;
    if (!check(cudaHostGetDevicePointer(&dp, host, 0), "map the prompt exchange", err, device_)) return false;
    pf_lay_ = hlink::PfLayout((size_t) cap_tokens, (size_t) n_embd_);
    if (hlink::pf_exchange_bytes((size_t) cap_tokens, (size_t) n_embd_) > bytes) {
        err = "GLM expert helper: the prompt exchange is smaller than its layout";
        return false;
    }
    pf_ = host;
    pf_dev_ = (uint8_t*) dp;
    pf_cap_ = cap_tokens;
    return true;
}

bool GlmExpertHelper::pf_run(int gu_type, int d_type, size_t down_off, float limit, bool dsa, int T, int K, int n,
                             int rows, int r0, const uint8_t* const* blobs, std::string& err, int slot) {
#if defined(STRATA_USE_HIP)
    namespace mmq = strata::prefill::mmq;
    namespace gb = strata::kernels::glmb;
    if (pf_ == nullptr || T <= 0 || slot < 0 || slot > 2 || T > (slot > 0 ? pf_slot_cap() : pf_cap_) || K <= 0 || K > hlink::kPfMaxK || n < 0 || n > hlink::kPfMaxExperts ||
        rows < 0 || rows > T * K || r0 < 0) {
        err = "GLM expert helper: invalid prompt call";
        return false;
    }
    if (!mmq::built() || !mmq::supported(gu_type) || !mmq::supported(d_type)) {
        err = "GLM expert helper: this build's MMQ does not cover the layer's expert types";
        return false;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const int E = n_embd_, nff = n_ff_;
    const auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    // the scratch for a window of W rows (MAYA_GLM_LINK_PF_WIN, default 8192): q8 x, gate/up, q8 h, the rows' out
    if (pf_win_ == 0) {
        const char* wv = std::getenv("MAYA_GLM_LINK_PF_WIN");
        int W = wv != nullptr ? std::atoi(wv) : 8192;
        W = (std::max(W, 256) + 255) & ~255;
        const size_t need = al(mmq::q8_bytes(W, E)) + al((size_t) W * 2 * nff * sizeof(float)) +
                            al(mmq::q8_bytes(W, nff)) + al((size_t) W * E * sizeof(float)) +
                            al(((size_t) W + mmq::IDS_PAD) * sizeof(int32_t));
        if (!check(cudaMalloc(&pf_scratch_, need), "prompt scratch", err, device_)) return false;
        pf_scratch_bytes_ = need;
        void* bh = nullptr;
        void* bd = nullptr;
        if (!check(cudaHostAlloc(&bh, (size_t) 2 * (hlink::kPfMaxExperts + 8192) * sizeof(int32_t),
                                 cudaHostAllocPortable | cudaHostAllocMapped), "prompt bounds", err, device_) ||
            !check(cudaHostGetDevicePointer(&bd, bh, 0), "prompt bounds mapping", err, device_)) return false;
        pf_bnd_h_ = (int32_t*) bh;
        pf_bnd_d_ = (int32_t*) bd;
        pf_mq_ = new mmq::Context();
        pf_win_ = W;
        uint8_t* p = (uint8_t*) pf_scratch_;
        p += al(mmq::q8_bytes(W, E)) + al((size_t) W * 2 * nff * sizeof(float)) + al(mmq::q8_bytes(W, nff)) +
             al((size_t) W * E * sizeof(float));
        mmq::iota((int32_t*) p, (int64_t) W + mmq::IDS_PAD, stream_);
        std::fprintf(stderr, "maya-glm-helper: prompt windows of %d rows (%.0f MB scratch)\n", W, need / 1048576.0);
    }
    const int W = pf_win_;
    uint8_t* p = (uint8_t*) pf_scratch_;
    void* Xq = p;                 p += al(mmq::q8_bytes(W, E));
    float* GU = (float*) p;       p += al((size_t) W * 2 * nff * sizeof(float));
    void* Hq = p;                 p += al(mmq::q8_bytes(W, nff));
    float* OUTP = (float*) p;     p += al((size_t) W * E * sizeof(float));
    const int32_t* iota = (const int32_t*) p;
    // each expert's blob as this GPU reads it (the registered shared tier, in place)
    std::vector<const uint8_t*> dev_of((size_t) n, nullptr);
    for (int j = 0; j < n; ++j)
        for (const auto& r : regions_) {
            const uintptr_t off = (uintptr_t) blobs[j] - (uintptr_t) r.host;
            if (blobs[j] != nullptr && (uintptr_t) blobs[j] >= (uintptr_t) r.host && off < r.bytes) {
                dev_of[(size_t) j] = r.mapped + off;
                break;
            }
        }
    const int32_t* cnts = pf_cnts(slot);
    const float* x_d = (const float*) pf_at(slot, &hlink::PfLayout::x, pf_dev_);
    float* out_d = (float*) pf_at(slot, &hlink::PfLayout::out, pf_dev_);
    const int32_t* pos_d = (const int32_t*) pf_at(slot, &hlink::PfLayout::pos, pf_dev_);
    const float* rw_d = (const float*) pf_at(slot, &hlink::PfLayout::rw, pf_dev_);
    const int32_t* rtok_d = (const int32_t*) pf_at(slot, &hlink::PfLayout::rtok, pf_dev_);
    int64_t sum = 0;
    for (int j = 0; j < n; ++j) {
        if (cnts[j] < 0 || dev_of[(size_t) j] == nullptr) {
            err = "GLM expert helper: a prompt expert outside the shared RAM tier";
            return false;
        }
        sum += cnts[j];
    }
    if (sum != rows) {
        err = "GLM expert helper: the prompt experts' rows do not add up";
        return false;
    }
    const int errs0 = mmq::launch_errors();
    auto& mq = *(mmq::Context*) pf_mq_;
    cudaStream_t s = stream_;
    cudaMemsetAsync(out_d, 0, (size_t) T * E * sizeof(float), s);
    const size_t gu_bytes = mmq::matrix_bytes(gu_type, 2 * (int64_t) nff, E);
    const size_t dn_bytes = mmq::matrix_bytes(d_type, E, nff);
    struct Piece { int j, lo, hi, b; };
    std::vector<Piece> ps;
    int bp = 0, j = 0, jr = 0;
    for (int a = 0; a < rows;) {
        const int b = std::min(rows, a + W), nr = b - a;
        ps.clear();
        for (int at = a; at < b;) {
            const int take = std::min(cnts[j] - jr, b - at);
            if (take > 0) {
                pf_bnd_h_[2 * bp] = at - a;
                pf_bnd_h_[2 * bp + 1] = at - a + take;
                ps.push_back({j, at - a, at - a + take, bp++});
            }
            at += take;
            jr += take;
            if (jr == cnts[j]) { ++j; jr = 0; }
        }
        mmq::quantize(x_d, rtok_d + a, Xq, gu_type, E, E, nr, s);
        glmpf::mmq_clamp(dsa, Xq, nr, E, gu_type, s);
        for (const auto& pc : ps) {
            mmq::Product P;
            P.w = dev_of[(size_t) pc.j];
            P.type = gu_type;
            P.w_rows = 2 * (int64_t) nff;
            P.w_cols = E;
            P.expert_bytes = gu_bytes;
            P.n = 1;
            P.xq = Xq;
            P.bounds = pf_bnd_d_ + 2 * pc.b;
            P.ids = iota;
            P.total_rows = nr;
            P.max_rows = pc.hi - pc.lo;
            P.dst = GU;
            P.ld_dst = 2 * (int64_t) nff;
            mq.run(P, s);
        }
        gb::swiglu_rows(GU, GU, nr, nff, limit, s, 2 * nff);
        mmq::quantize(GU, nullptr, Hq, d_type, nff, 2 * (int64_t) nff, nr, s);
        glmpf::mmq_clamp(dsa, Hq, nr, nff, d_type, s);
        for (const auto& pc : ps) {
            mmq::Product D;
            D.w = dev_of[(size_t) pc.j] + down_off;
            D.type = d_type;
            D.w_rows = E;
            D.w_cols = nff;
            D.expert_bytes = dn_bytes;
            D.n = 1;
            D.xq = Hq;
            D.bounds = pf_bnd_d_ + 2 * pc.b;
            D.ids = iota;
            D.total_rows = nr;
            D.max_rows = pc.hi - pc.lo;
            D.dst = OUTP;
            D.ld_dst = E;
            mq.run(D, s);
        }
        gb::moe_combine_add(OUTP, r0 + a, r0 + a, r0 + b, pos_d, rw_d, T, K, E, out_d, s);
        a = b;
        if (2 * (bp + n + 2) > 2 * (hlink::kPfMaxExperts + 8192)) {   // (the bounds table is reused from its start)
            if (!check(cudaStreamSynchronize(s), "prompt window", err, device_)) return false;
            bp = 0;
        }
    }
    if (!check(cudaStreamSynchronize(s), "prompt layer", err, device_)) return false;
    if (mmq::launch_errors() != errs0) {
        err = "GLM expert helper: MMQ refused a prompt product";
        return false;
    }
    return true;
#else
    (void) gu_type; (void) d_type; (void) down_off; (void) limit; (void) dsa; (void) T; (void) K; (void) n;
    (void) rows; (void) r0; (void) blobs; (void) slot;
    err = "GLM expert helper: prompt products need the helper program of a HIP build";
    return false;
#endif
}

} // namespace strata::core
