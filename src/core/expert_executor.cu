#include "strata/core/expert_executor.h"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gf = strata::kernels::glmf;
struct GpuError : std::runtime_error { using std::runtime_error::runtime_error; };
void ck(cudaError_t e) { if (e != cudaSuccess) throw GpuError(cudaGetErrorString(e)); }
void require(bool b, const char *msg) { if (!b) throw std::runtime_error(msg); }
int message(const char *s, char *err, size_t n) { if (err && n) std::snprintf(err,n,"%s",s); return 1; }
struct Device {
    int old = -1;
    bool restored = false;
    explicit Device(int dev) { ck(cudaGetDevice(&old)); ck(cudaSetDevice(dev)); }
    void restore() { ck(cudaSetDevice(old)); restored = true; }
    ~Device() { if (!restored && old >= 0) cudaSetDevice(old); }
};
struct Context {
    maya_context_config cfg{};
    maya_context_stats counters{};
    std::mutex mutex;
    bool poisoned = false;
    cudaStream_t stream = nullptr;
    unsigned char *arena = nullptr, *pool = nullptr;
    size_t stride = 0, down_offset = 0;
    float *x = nullptr, *out = nullptr;
    void *xq = nullptr, *hq = nullptr;
    gf::MoeDev plan;
    unsigned char *rows_arena = nullptr;
    float *rows_x = nullptr;
    gf::MoeWin window;
    unsigned long long row_ptr[16*8]{};
    float row_weights[16*8]{}, row_result[16*4096]{};
    std::vector<uint64_t> generation;
    // Persistent host plan/output storage survives every submitted read/write.
    unsigned long long pointers[8]{};
    float weights[8]{}, result[4096]{};
    // Same small mapped-block layout as GlmExpertHelper; allocated by this runtime,
    // never by the primary GPU's runtime. Only single-token execution uses it.
    struct alignas(64) Mapped {
        float input[4096], output[4096];
        unsigned long long pointers[8];
        float weights[8];
    };
    Mapped *mapped = nullptr, *mapped_device = nullptr;
    bool prof = false;
    cudaEvent_t events[6]{};
    uint64_t prof_calls = 0;
    double stage_ms[5]{}, enqueue_ms = 0, wait_ms = 0, publish_ms = 0;
    void mark(int i) { if (prof) ck(cudaEventRecord(events[i],stream)); }
    int cleanup() {
        cudaError_t first = cudaSuccess;
        auto save = [&](cudaError_t e) { if (first == cudaSuccess) first = e; };
        if (stream) save(cudaStreamSynchronize(stream));
        if (prof_calls) {
            const double n=double(prof_calls);
            std::fprintf(stderr,"glm plugin prof: device %d types %d/%d mapped %d calls %llu; "
                "host enqueue/wait/publish %.6f/%.6f/%.6f ms/call; GPU input/quantize/gate/down/result "
                "%.6f/%.6f/%.6f/%.6f/%.6f ms/call\n",cfg.device,cfg.gu_type,cfg.down_type,
                mapped!=nullptr,(unsigned long long)prof_calls,enqueue_ms/n,wait_ms/n,publish_ms/n,
                stage_ms[0]/n,stage_ms[1]/n,stage_ms[2]/n,stage_ms[3]/n,stage_ms[4]/n);
            prof_calls=0;
        }
        for (auto &e:events) if (e) { save(cudaEventDestroy(e)); e=nullptr; }
        if (mapped) { save(cudaFreeHost(mapped)); mapped=nullptr; mapped_device=nullptr; }
        if (rows_arena) { save(cudaFree(rows_arena)); rows_arena = nullptr; }
        if (arena) { save(cudaFree(arena)); arena = nullptr; }
        if (stream) { save(cudaStreamDestroy(stream)); stream = nullptr; }
        if (first != cudaSuccess) throw GpuError(cudaGetErrorString(first));
        return 0;
    }
};
template<class F> int invoke(void *opaque, char *error, size_t n, F fn) {
    if (!opaque) return message("null context",error,n);
    auto *c = static_cast<Context *>(opaque);
    std::unique_lock<std::mutex> lock(c->mutex, std::defer_lock);
    try {
        lock.lock();
        require(!c->poisoned,"context poisoned by runtime failure");
        Device dev(c->cfg.device);
        try { fn(*c); }
        catch (...) {
            // Drain even when submission fails; don't return host buffer ownership early.
            if (cudaStreamSynchronize(c->stream) != cudaSuccess) c->poisoned = true;
            dev.restore();
            throw;
        }
        dev.restore();
        return 0;
    } catch (const GpuError &e) { c->poisoned = true; return message(e.what(),error,n); }
    catch (const std::exception &e) { return message(e.what(),error,n); }
    catch (...) { c->poisoned = true; return message("unknown context error",error,n); }
}
int create(const maya_context_config *cfg, void **opaque, char *error, size_t n) {
    if (!opaque) return message("null output context",error,n);
    *opaque = nullptr;
    Context *c = nullptr;
    try {
        require(cfg && cfg->size == sizeof(*cfg) && cfg->version == 1,"invalid config ABI");
        require(cfg->width == 4096 && cfg->intermediate == 2048 && cfg->slots > 0,
                "unsupported shape or slot count");
        require(gf::moe_supported(cfg->gu_type) && gf::moe_supported(cfg->down_type),"unsupported quantization");
        c = new Context;
        c->cfg = *cfg;
        c->down_offset = 2 * size_t(cfg->intermediate) * gf::row_bytes(cfg->gu_type,cfg->width);
        c->stride = c->down_offset + size_t(cfg->width) * gf::row_bytes(cfg->down_type,cfg->intermediate);
        require(c->stride && cfg->slots<=std::numeric_limits<size_t>::max()/c->stride,"expert pool size overflow");
        size_t used = 0;
        auto reserve = [&](size_t bytes) {
            require(bytes<=std::numeric_limits<size_t>::max()-255,"arena size overflow");
            const size_t aligned=(bytes+255)&~size_t(255);
            require(used<=cfg->byte_budget && aligned<=cfg->byte_budget-used,"memory budget exceeded");
            const size_t start=used; used+=aligned; return start;
        };
        size_t pool = reserve(c->stride * cfg->slots), x = reserve(4096*4), out = reserve(4096*4);
        size_t xq = reserve(4096/32*36), hq = reserve(8*2048/32*36);
        size_t ptr = reserve(8*8), weights = reserve(8*4), flag = reserve(4);
        require(used <= cfg->byte_budget,"memory budget exceeded");
        c->generation.resize(cfg->slots,0);
        Device dev(cfg->device);
        try {
            ck(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
            ck(cudaMalloc(reinterpret_cast<void **>(&c->arena),used));
            c->pool=c->arena+pool; c->x=reinterpret_cast<float *>(c->arena+x); c->out=reinterpret_cast<float *>(c->arena+out);
            c->xq=c->arena+xq; c->hq=c->arena+hq;
            c->plan.plan_ptr=reinterpret_cast<unsigned long long *>(c->arena+ptr);
            c->plan.plan_w=reinterpret_cast<float *>(c->arena+weights);
            c->plan.cpu_flag=reinterpret_cast<int *>(c->arena+flag);
            ck(cudaMemsetAsync(c->plan.cpu_flag,0,4,c->stream)); ck(cudaStreamSynchronize(c->stream));
            c->counters.allocated_bytes=used;
            // Opt in with the existing native-helper setting; retain copying on
            // discrete devices, insufficient budget or unavailable host mapping.
            const char *zc=std::getenv("STRATA_GLM_HELPER_ZEROCOPY");
            if (zc && std::strcmp(zc,"1")==0) {
                int integrated=0;
                ck(cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,cfg->device));
                cudaError_t status=cudaSuccess;
                if (integrated && sizeof(Context::Mapped)<=cfg->byte_budget-used) {
                    status=cudaHostAlloc(reinterpret_cast<void **>(&c->mapped),sizeof(Context::Mapped),
                                         cudaHostAllocPortable|cudaHostAllocMapped);
                    if (status==cudaSuccess)
                        status=cudaHostGetDevicePointer(reinterpret_cast<void **>(&c->mapped_device),c->mapped,0);
                    if (status!=cudaSuccess) {
                        if (c->mapped) { ck(cudaFreeHost(c->mapped)); c->mapped=nullptr; }
                        c->mapped_device=nullptr;
                        cudaGetLastError();
                    } else c->counters.allocated_bytes+=sizeof(Context::Mapped);
                }
                std::fprintf(stderr,"glm plugin buffers: device %d types %d/%d %s, %zu host bytes%s%s\n",
                    cfg->device,cfg->gu_type,cfg->down_type,c->mapped?"mapped":"copy fallback",
                    c->mapped?sizeof(Context::Mapped):size_t(0),status==cudaSuccess?"":"; ",
                    status==cudaSuccess?"":cudaGetErrorString(status));
            }
            c->prof=std::getenv("STRATA_GLM_PROF")!=nullptr;
            if (c->prof) for (auto &e:c->events) ck(cudaEventCreate(&e));
        } catch (...) { try { c->cleanup(); } catch (...) {} dev.restore(); throw; }
        try { dev.restore(); } catch (...) { try { c->cleanup(); } catch (...) {} throw; }
        *opaque = c; return 0;
    } catch (const std::exception &e) { delete c; return message(e.what(),error,n); }
    catch (...) { delete c; return message("unknown create error",error,n); }
}
int upload(void *ctx, uint32_t slot, uint64_t expected, const void *bytes, size_t count,
           uint64_t *next, char *error, size_t n) {
    return invoke(ctx,error,n,[&](Context &c) {
        require(slot<c.generation.size() && bytes && next && count==c.stride,"invalid upload");
        require(c.generation[slot]==expected && expected<std::numeric_limits<uint64_t>::max(),"stale upload generation");
        ck(cudaMemcpyAsync(c.pool+size_t(slot)*c.stride,bytes,count,cudaMemcpyHostToDevice,c.stream));
        ck(cudaStreamSynchronize(c.stream));
        c.generation[slot]=expected+1; *next=expected+1;
        c.counters.uploads++; c.counters.upload_bytes+=count;
    });
}
int compute(void *ctx, uint32_t count, const uint32_t *slots, const uint64_t *gens,
            const float *weights, const float *input, float clamp, float *output, char *error, size_t n) {
    return invoke(ctx,error,n,[&](Context &c) {
        require(count>=1 && count<=8 && slots && gens && weights && input && output && std::isfinite(clamp) && clamp>0,
                "invalid compute descriptor");
        for (uint32_t i=0;i<count;i++) {
            require(slots[i]<c.generation.size() && gens[i]!=0 && c.generation[slots[i]]==gens[i],"stale or absent expert");
            require(std::isfinite(weights[i]),"nonfinite routing");
        }
        for (int i=0;i<4096;i++) require(std::isfinite(input[i]),"nonfinite input");
        for (uint32_t i=0;i<count;i++) {
            c.pointers[i]=reinterpret_cast<unsigned long long>(c.pool+size_t(slots[i])*c.stride);
            c.weights[i]=weights[i];
        }
        using Clock=std::chrono::steady_clock;
        const auto t0=c.prof?Clock::now():Clock::time_point{};
        gf::MoeDev plan=c.plan;
        float *device_x=c.x, *device_out=c.out, *host_out=c.result;
        c.mark(0);
        if (c.mapped) {
            std::memcpy(c.mapped->input,input,4096*4);
            std::memcpy(c.mapped->pointers,c.pointers,count*8);
            std::memcpy(c.mapped->weights,c.weights,count*4);
            device_x=c.mapped_device->input; device_out=c.mapped_device->output;
            host_out=c.mapped->output;
            plan.plan_ptr=c.mapped_device->pointers; plan.plan_w=c.mapped_device->weights;
        } else {
            ck(cudaMemcpyAsync(c.x,input,4096*4,cudaMemcpyHostToDevice,c.stream));
            ck(cudaMemcpyAsync(c.plan.plan_ptr,c.pointers,count*8,cudaMemcpyHostToDevice,c.stream));
            ck(cudaMemcpyAsync(c.plan.plan_w,c.weights,count*4,cudaMemcpyHostToDevice,c.stream));
        }
        ck(cudaMemsetAsync(device_out,0xff,4096*4,c.stream)); // NaNs expose missing writes.
        c.mark(1);
        gf::quantize_q8_1(device_x,c.xq,4096,c.stream); c.mark(2);
        gf::moe_gate_up(c.cfg.gu_type,plan,count,4096,2048,clamp,c.xq,c.hq,nullptr,0,nullptr,0,nullptr,c.stream);
        c.mark(3);
        gf::moe_down(c.cfg.down_type,plan,count,4096,2048,c.down_offset,c.hq,nullptr,device_out,c.stream);
        c.mark(4);
        ck(cudaGetLastError());
        if (!c.mapped) ck(cudaMemcpyAsync(c.result,c.out,4096*4,cudaMemcpyDeviceToHost,c.stream));
        c.mark(5);
        const auto t1=c.prof?Clock::now():Clock::time_point{};
        ck(cudaStreamSynchronize(c.stream));
        const auto t2=c.prof?Clock::now():Clock::time_point{};
        for (int i=0;i<4096;i++) if (!std::isfinite(host_out[i])) { c.poisoned=true; throw std::runtime_error("nonfinite expert output"); }
        if (c.prof) {
            for (int i=0;i<5;i++) { float ms=0; ck(cudaEventElapsedTime(&ms,c.events[i],c.events[i+1])); c.stage_ms[i]+=ms; }
        }
        std::memcpy(output,host_out,sizeof(c.result)); c.counters.computes++;
        if (c.prof) {
            c.enqueue_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
            c.wait_ms+=std::chrono::duration<double,std::milli>(t2-t1).count();
            c.publish_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t2).count();
            ++c.prof_calls;
        }
    });
}
int stats(void *ctx, maya_context_stats *out, char *error, size_t n) {
    return invoke(ctx,error,n,[&](Context &c) { require(out,"null stats"); *out=c.counters; });
}
int destroy(void **opaque, char *error, size_t n) {
    if (!opaque) return message("null context pointer",error,n);
    auto *c=static_cast<Context *>(*opaque); if (!c) return 0;
    *opaque=nullptr;
    int rc=0;
    try { Device dev(c->cfg.device); try { c->cleanup(); } catch (...) { dev.restore(); throw; } dev.restore(); }
    catch (const std::exception &e) { rc=message(e.what(),error,n); }
    catch (...) { rc=message("unknown destroy error",error,n); }
    delete c; return rc;
}
const maya_context_api api{sizeof(maya_context_api),1,create,upload,compute,stats,destroy};
}
extern "C" __attribute__((visibility("default")))
const maya_context_api *maya_expert_context_api(uint32_t version) { return version==1 ? &api : nullptr; }

extern "C" __attribute__((visibility("default")))
int maya_expert_context_read(void *ctx, uint32_t slot, uint64_t generation,
                             size_t offset, void *output, size_t bytes, char *error, size_t n) {
    return invoke(ctx,error,n,[&](Context &c) {
        require(slot<c.generation.size() && generation && c.generation[slot]==generation,
                "stale or absent readback expert");
        require(output && bytes && offset<=c.stride && bytes<=c.stride-offset,"invalid readback range");
        ck(cudaMemcpyAsync(output,c.pool+size_t(slot)*c.stride+offset,bytes,cudaMemcpyDeviceToHost,c.stream));
        ck(cudaStreamSynchronize(c.stream));
    });
}

extern "C" __attribute__((visibility("default")))
int maya_expert_context_rows(void *ctx, uint32_t count, const uint32_t *slots,
                             const uint64_t *gens, const float *input, float clamp,
                             float *output, char *error, size_t n) {
    return invoke(ctx,error,n,[&](Context &c) {
        require(count>=1 && count<=16 && slots && gens && input && output &&
                std::isfinite(clamp) && clamp>0,"invalid expert-row descriptor");
        for (uint32_t i=0;i<count;i++)
            require(slots[i]<c.generation.size() && gens[i] && c.generation[slots[i]]==gens[i],
                    "stale or absent row expert");
        for (size_t i=0;i<size_t(count)*4096;i++) require(std::isfinite(input[i]),"nonfinite row input");
        auto &w=c.window;
        if (!c.rows_arena) {
            constexpr size_t bytes=3*1024*1024;
            require(c.counters.allocated_bytes<=c.cfg.byte_budget &&
                    bytes<=c.cfg.byte_budget-c.counters.allocated_bytes,"expert-row scratch exceeds budget");
            ck(cudaMalloc(reinterpret_cast<void **>(&c.rows_arena),bytes));
            c.counters.allocated_bytes+=bytes;
            size_t used=0;
            auto take=[&](size_t size) {
                const size_t offset=used; used+=(size+255)&~size_t(255);
                require(used<=bytes,"expert-row scratch overflow"); return c.rows_arena+offset;
            };
            c.rows_x=reinterpret_cast<float *>(take(16*4096*4));
            w.out=reinterpret_cast<float *>(take(16*4096*4));
            w.xq=take(16*4096/32*36); w.hq=take(16*8*2048/32*36);
            w.part=reinterpret_cast<float *>(take(16*8*4096*4));
            w.plan_ptr=reinterpret_cast<unsigned long long *>(take(16*8*8));
            w.plan_w=reinterpret_cast<float *>(take(16*8*4));
            w.cpu_flag=reinterpret_cast<int *>(take(16*4));
            w.grp_ptr=reinterpret_cast<unsigned long long *>(take(16*8*8));
            w.grp_start=reinterpret_cast<int *>(take((16*8+1)*4));
            w.grp_ent=reinterpret_cast<int *>(take(16*8*4));
            w.n_groups=reinterpret_cast<int *>(take(4));
            ck(cudaMemsetAsync(const_cast<int *>(w.cpu_flag),0,16*4,c.stream));
            w.xq_stride=4096/32*36; w.out_stride=4096;
            w.k=1; w.n_embd=4096; w.n_ff=2048; w.down_off=c.down_offset;
        }
        w.T=int(count); w.limit=clamp;
        for (uint32_t i=0;i<count;i++) {
            c.row_ptr[i*8]=reinterpret_cast<unsigned long long>(c.pool+size_t(slots[i])*c.stride);
            c.row_weights[i*8]=1.0f;
        }
        ck(cudaMemcpyAsync(c.rows_x,input,size_t(count)*4096*4,cudaMemcpyHostToDevice,c.stream));
        ck(cudaMemcpyAsync(const_cast<unsigned long long *>(w.plan_ptr),c.row_ptr,count*8*8,cudaMemcpyHostToDevice,c.stream));
        ck(cudaMemcpyAsync(const_cast<float *>(w.plan_w),c.row_weights,count*8*4,cudaMemcpyHostToDevice,c.stream));
        ck(cudaMemsetAsync(w.out,0xff,size_t(count)*4096*4,c.stream));
        gf::quantize_q8_1(c.rows_x,const_cast<void *>(w.xq),count*4096,c.stream);
        require(gf::moe_win_experts(c.cfg.gu_type,c.cfg.down_type,w,c.stream),"unsupported expert-row format");
        ck(cudaGetLastError());
        ck(cudaMemcpyAsync(c.row_result,w.out,size_t(count)*4096*4,cudaMemcpyDeviceToHost,c.stream));
        ck(cudaStreamSynchronize(c.stream));
        for (size_t i=0;i<size_t(count)*4096;i++) if (!std::isfinite(c.row_result[i])) {
            c.poisoned=true; throw std::runtime_error("nonfinite expert-row output");
        }
        std::memcpy(output,c.row_result,size_t(count)*4096*4); c.counters.computes++;
    });
}
