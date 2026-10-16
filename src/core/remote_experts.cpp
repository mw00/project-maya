#include "strata/core/remote_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace strata::core {
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
                           const std::vector<std::pair<void*, size_t>>& regions, std::string& err) {
    close();
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    int integrated = 0;
    if (!check(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device), "integrated memory", err, device))
        return false;
    if (n_embd <= 0 || n_embd > 4096 || n_embd % 32 || n_ff <= 0 || n_ff % 32) {
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
    return check(cudaMemset(plan_.cpu_flag, 0, 64), "helper flags", err, device);
}

bool GlmExpertHelper::reserve(int keys, size_t requested_bytes, std::string& err) {
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
    if (out < 0 || (size_t) out >= slots_.size() || in < 0 || (size_t) in >= slots_.size() || !owns(out) || owns(in)) {
        err = "GLM expert helper: invalid cache replacement"; return false;
    }
    Slot s = slots_[(size_t) out];
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
        if (!check(cudaMemcpyAsync(s.device, host, s.bytes, cudaMemcpyHostToDevice, stream_), "cache replace", err, device_) ||
            !check(cudaStreamSynchronize(stream_), "cache replace finish", err, device_)) return false;
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
    namespace gf = strata::kernels::glmf;
    if (device_ < 0 || ne < 1 || ne > 8 || !gf::moe_supported(gu_type) || !gf::moe_supported(d_type)) {
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
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (!check(cudaMemcpyAsync(x_, x, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "helper input", err, device_) ||
        !check(cudaMemcpyAsync(plan_.plan_ptr, ptr, (size_t) ne * sizeof(ptr[0]), cudaMemcpyHostToDevice, stream_),
               "helper pointers", err, device_) ||
        !check(cudaMemcpyAsync(plan_.plan_w, weights, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "helper weights", err, device_)) return false;
    gf::quantize_q8_1(x_, xq_, n_embd_, stream_);
    gf::moe_gate_up(gu_type, plan_, ne, n_embd_, n_ff_, limit, xq_, hq_, nullptr, 0, nullptr, 0, nullptr, stream_);
    gf::moe_down(d_type, plan_, ne, n_embd_, n_ff_, down_off, hq_, nullptr, out_, stream_);
    return check(cudaMemcpyAsync(out, out_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                 "helper result", err, device_) &&
           check(cudaStreamSynchronize(stream_), "helper finish", err, device_);
}

void GlmExpertHelper::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        if (arena_) cudaFree(arena_);
        if (pool_) cudaFree(pool_);
        if (stream_) cudaStreamDestroy(stream_);
    }
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
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
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
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
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
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s);
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

} // namespace strata::core
