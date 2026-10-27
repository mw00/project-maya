// src/mm3/mm3_tp.cu - maya-mm3's tensor-parallel path (see mm3_tp.hpp)
#include "mm3_tp.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#include "mm3_cands.hpp"
#include "mm3_cpu.hpp"
#include "mm3_env.hpp"
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_geo.hpp"
#include "mm3_host.hpp"
#include "mm3_kernels.cuh"
#include "mm3_os.hpp"
#include "mm3_vision.hpp"

namespace mm3 {
namespace x3 = strata::mm3::exl3;

#define TK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("tp: ") + #call + ": " + cudaGetErrorString(e_);         \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

constexpr int kWin = 16;          // a decode window's tokens at most (the decode kernels, enqueued up front)
constexpr int kDraftRows = 256;   // the draft's context rows a call (a prompt chunk's in runs of these)
constexpr size_t kAlign = 256;
constexpr int kLine = 16;         // a counter's stride in the mapped words (its own 64-byte line)
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
const std::string P = "language_model.model.";

float f16_to_f32(uint16_t b) {
    const uint32_t s = (uint32_t) (b & 0x8000) << 16, e = (b >> 10) & 0x1f, m = b & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            float f = (float) m / 1024.0f / 16384.0f;
            std::memcpy(&u, &f, 4);
            u |= s;
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112) << 23) | (m << 13);
    }
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// an F32 / BF16 / F16 tensor as F32 on the host (+ add to every value)
bool host_f32(const Pack& pk, const std::string& name, float add, std::vector<float>& out, std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = "tp: " + name + " missing";
        return false;
    }
    const size_t es = t->dtype == "F32" ? 4 : 2;
    const size_t n = t->bytes / es;
    out.resize(n);
    const uint8_t* d = pk.data(*t);
    for (size_t i = 0; i < n; ++i) {
        float v;
        if (t->dtype == "F32") {
            std::memcpy(&v, d + 4 * i, 4);
        } else {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            if (t->dtype == "BF16") {
                const uint32_t u = (uint32_t) b << 16;
                std::memcpy(&v, &u, 4);
            } else {
                v = f16_to_f32(b);
            }
        }
        out[i] = v + add;
    }
    return true;
}

// an EXL3 matrix's tensors in the mapped files
struct Ex3 {
    const uint8_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
    int K = 0, N = 0, tb = 0, k2 = 0;   // inputs, outputs, a 16 x 16 tile's bytes, twice its bits
};

bool ex3(const Pack& pk, const std::string& pre, Ex3& x, std::string& err) {
    const Tensor* tr = pk.get(pre + ".trellis");
    const Tensor* su = pk.get(pre + ".suh");
    const Tensor* sv = pk.get(pre + ".svh");
    if (!tr || !su || !sv || tr->shape.size() != 3) {
        err = "tp: " + pre + ": not an EXL3 matrix";
        return false;
    }
    x.tr = pk.data(*tr);
    x.suh = pk.data(*su);
    x.svh = pk.data(*sv);
    x.K = (int) (su->bytes / 2);
    x.N = (int) (sv->bytes / 2);
    x.tb = (int) tr->shape[2] * 2;
    x.k2 = (int) tr->shape[2] / 8;
    return true;
}

// a slice of it: outputs [lo, hi) (by 'n': gate / up, a head's rows) or inputs [lo, hi) (by 'k': down, o_proj's
// columns) - trellis, suh, svh bytes
size_t tr_bytes(const Ex3& x, char by, int lo, int hi) {
    return by == 'n' ? (size_t) (x.K / 16) * ((hi - lo) / 16) * x.tb : (size_t) ((hi - lo) / 16) * (x.N / 16) * x.tb;
}
size_t suh_bytes(const Ex3& x, char by, int lo, int hi) { return by == 'n' ? (size_t) x.K * 2 : (size_t) (hi - lo) * 2; }
size_t svh_bytes(const Ex3& x, char by, int lo, int hi) { return by == 'n' ? (size_t) (hi - lo) * 2 : (size_t) x.N * 2; }
void cut(const Ex3& x, char by, int lo, int hi, uint8_t* tr, uint8_t* suh, uint8_t* svh) {
    if (by == 'n') {
        const size_t row = (size_t) (x.N / 16) * x.tb, part = (size_t) ((hi - lo) / 16) * x.tb;
        for (int kt = 0; kt < x.K / 16; ++kt)
            std::memcpy(tr + (size_t) kt * part, x.tr + (size_t) kt * row + (size_t) (lo / 16) * x.tb, part);
        std::memcpy(suh, x.suh, (size_t) x.K * 2);
        std::memcpy(svh, x.svh + (size_t) lo * 2, (size_t) (hi - lo) * 2);
    } else {
        const size_t row = (size_t) (x.N / 16) * x.tb;
        std::memcpy(tr, x.tr + (size_t) (lo / 16) * row, (size_t) ((hi - lo) / 16) * row);
        std::memcpy(suh, x.suh + (size_t) lo * 2, (size_t) (hi - lo) * 2);
        std::memcpy(svh, x.svh, (size_t) x.N * 2);
    }
}

// a thread that runs jobs one at a time (a member's enqueueing)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    void stop() {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        th.join();
    }
};

// (the host's AVX2 sums and norms: mm3_host.cpp - nvcc cannot read <immintrin.h> with every host compiler)
using host::dot;
using host::rms_rows;
using host::sum_into;

}  // namespace

struct TpEngine::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    std::vector<std::string> info;
    std::vector<std::string> vram;           // the Monitor's VRAM lines (serve/gpu_map.py), measured at the load's end

    struct Member {
        int id = 0;
        cudaStream_t s = nullptr;
        cublasHandle_t blas = nullptr;
        size_t free0 = 0;
        float *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr, *bs = nullptr;
        float *att = nullptr, *gg = nullptr, *uu = nullptr, *sh = nullptr, *out = nullptr, *win_h = nullptr;
        float *win_part = nullptr, *inv = nullptr, *rlog_d = nullptr, *attp = nullptr;
        int *pos = nullptr, *kept = nullptr, *pos_h = nullptr;
        uint8_t* plan_d = nullptr;
        unsigned* counter = nullptr;
        // the prompt path's scratch (chunk rows)
        float *p_xn = nullptr, *p_q = nullptr, *p_k = nullptr, *p_v = nullptr, *p_iq = nullptr, *p_ik = nullptr;
        float *p_bs = nullptr, *p_att = nullptr, *p_gg = nullptr, *p_uu = nullptr, *p_sh = nullptr;
        float *p_y = nullptr, *p_w = nullptr, *p_rlog = nullptr;
        uint16_t *p_xn16 = nullptr, *p_att16 = nullptr, *p_h16 = nullptr;
        int *p_pos = nullptr, *p_kept = nullptr, *p_rowidx = nullptr, *p_invrow = nullptr, *p_bres = nullptr,
            *p_bstg = nullptr;
        std::unique_ptr<Worker> w = std::make_unique<Worker>();
        std::string err;   // (its last job's)
    };
    struct Group {
        std::vector<int> mem;   // the members (indices into M)
        int l0 = 0, l1 = 0;     // its layers
        float *xa = nullptr, *xb = nullptr, *part = nullptr;   // mapped: the attention's and the FFN's normed input,
        uint8_t* plans = nullptr;                             // the members' partials [tp][kWin][dim], their plans
        unsigned *fin = nullptr, *fout = nullptr;              // the host's counter, the members' (kLine apart)
        unsigned seq_in = 0;
        std::vector<unsigned> seq_out;
        float* rlog = nullptr;        // mapped: member 0's router logits [kWin][n_exp]
        // the prompt path's (chunk rows): the normed inputs, the partials, the router's logits, the MoE plan
        float *p_xa = nullptr, *p_xb = nullptr, *p_part = nullptr, *p_rlog = nullptr;
        uint8_t* p_plan = nullptr;
        unsigned* frout = nullptr;    // its counter
        unsigned seq_rout = 0;
        // the expert cache (Options::promote): n_slots slice blobs on every member - the same expert in slot s on all -
        // filled from the pinned pool on each member's copy stream after a window, its own a fixed number of windows
        // after (promote_lag): which experts a window finds in VRAM follows from the tokens alone
        int n_slots = 0;
        size_t slot_bytes = 0;
        std::vector<uint8_t*> slot_base;                  // [member]
        std::vector<cudaStream_t> cs;                     // [member]
        std::vector<std::vector<cudaEvent_t>> slot_ev;    // [member][slot]
        std::vector<long long> slot_owner, slot_pend, slot_iss, slot_used;
        int n_pending = 0;
        std::vector<std::pair<int, long long>> promo_q;   // (slot, key) to copy after the window
        // the decode's promotions hold at most prom_cap slots (slot_prom): the rest stay free for a prompt's staged
        // experts, which take no promoted slot while another is free - a prompt leaves the decode's cache as it was
        std::vector<char> slot_prom;
        int n_prom = 0, prom_cap = 0;
        std::vector<cudaEvent_t> p_ev;   // [member] a prompt layer's staged copies done (on cs)
        std::vector<float> p_cpu;        // the prompt's CPU lane sums [chunk][dim]
    };
    struct Slice {   // a layer on one member
        x3::Mat q, k, v, o, mg, mu, md, sg, su, sd;
        float *qn1 = nullptr, *kn1 = nullptr, *iqn1 = nullptr, *ikn1 = nullptr, *wiq = nullptr, *wik = nullptr;
        uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
        k::KvQ kvq;   // (--kv-cache: K's / V's quantized caches - a format other than BF16 instead of kc / vc)
        k::KvHost kvh;   // (--kv-streaming: the caches above a ring, every position in host memory too)
        uint8_t* res_base = nullptr;
        float* gate = nullptr;   // (member 0: the router, F32 [n_exp][dim])
    };
    struct Layer {
        int group = 0;
        bool moe = false;
        std::vector<float> in1, post1, gate, bias;
        std::vector<Slice> sl;
        x3::ExpertLayout lay;
        size_t blob = 0;                          // a member's slice of an expert
        std::vector<std::array<Ex3, 3>> ex;       // the experts' w1, w3, w2 in the files
        std::vector<int> res_slot, res_ids;
        std::vector<const uint8_t*> host;         // expert -> its tp slices in the pinned pool (blob apart)
        std::vector<int> slot_of;                 // expert -> its slot in the group's cache (-1: none)
    };

    std::vector<Member> M;
    std::vector<Group> G;
    std::vector<Layer> L;
    int tp = 1;
    int chunk = 1024;                        // a prompt chunk's tokens at most (Options::tp_chunk)
    int head_m = 0;                          // the member holding the head, the draft and the vision tower
    std::vector<void*> owned;
    std::vector<int> owned_dev;
    std::vector<void*> hosted;               // cudaHostAlloc'd
    // the Monitor's map of each GPU (vram_lines): dalloc's bytes by what they hold, a device id's
    enum { kAW, kAKV, kAAct, kAPrompt, kAPool, kADraft, kAN };
    int acat = kAAct;
    std::map<int, std::array<size_t, kAN>> acct;
    std::map<int, size_t> acct_ic, acct_kvh;   // (the index-key caches, every member's whole; the host KV bytes)
    uint8_t* pool = nullptr;
    size_t plan_bytes = 0;
    std::vector<float> inv_freq;
    // the head
    x3::Mat head;
    std::vector<float> norm1;
    float *xf = nullptr, *lg = nullptr, *lgh = nullptr;
    // (forward_cands) this forward's head rows as candidates: topk_rows into cand_h (mapped), cand_ex the extra ids
    bool cand_on = false;
    int cand_kc = 1, cand_ne = 0;
    k::CandRow* cand_h = nullptr;
    int* cand_ex = nullptr;
    unsigned* fhead = nullptr;
    unsigned seq_head = 0;
    cudaEvent_t head_ev = nullptr;
    // the host's side of a forward
    std::vector<float> h, cpu_out, rl;
    std::vector<uint32_t> prof;
    std::unique_ptr<CpuLane> lane;
    std::unique_ptr<Draft> draftm;
    std::vector<int> tap_of;
    float* taps_h = nullptr;            // (pinned: a window's taps as its layers pass them)
    float* taps_d = nullptr;
    unsigned* ftap = nullptr;           // (mapped) the host's counter of taps written: the draft's steps wait on it
    unsigned seq_tap = 0;
    std::unique_ptr<Vision> vis;
    std::vector<Model::Image> imgs;
    std::atomic<bool> abort_{false};
    bool in_prompt = false;   // (the pipeline's host threads at work: the shared tallies are theirs to add after)
    // (MM3_TP_PROF: where a window's host time goes, printed at the end)
    const bool prof_on = ::mm3::env("MM3_TP_PROF") != nullptr;
    double tp_norm = 0, tp_sum = 0, tp_route = 0, tp_plan = 0, tp_lane = 0, tp_wait = 0, tp_head = 0, tp_end = 0,
           tp_draft = 0, tp_start = 0;
    long tp_windows = 0;

    ~Impl() {
        if (prof_on && tp_windows)
            std::fprintf(stderr, "tp host, ms a window (%ld): start %.2f norms %.2f sums %.2f routing %.2f plans %.2f "
                         "lane %.2f waits %.2f head %.2f end %.2f draft ctx %.2f\n", tp_windows,
                         1e3 * tp_start / tp_windows, 1e3 * tp_norm / tp_windows, 1e3 * tp_sum / tp_windows,
                         1e3 * tp_route / tp_windows, 1e3 * tp_plan / tp_windows, 1e3 * tp_lane / tp_windows,
                         1e3 * tp_wait / tp_windows, 1e3 * tp_head / tp_windows, 1e3 * tp_end / tp_windows,
                         1e3 * tp_draft / tp_windows);
        abort_.store(true);
        release_all();
        for (Member& m : M) m.w->stop();
        for (Member& m : M) {
            cudaSetDevice(m.id);
            if (m.s) cudaStreamSynchronize(m.s);
        }
        draftm.reset();
        vis.reset();
        for (size_t i = 0; i < owned.size(); ++i) {
            cudaSetDevice(owned_dev[i]);
            cudaFree(owned[i]);
        }
        if (head_ev) cudaEventDestroy(head_ev);
        for (Group& gr : G)
            for (size_t j = 0; j < gr.cs.size(); ++j) {
                cudaSetDevice(M[(size_t) gr.mem[j]].id);
                if (gr.cs[j]) cudaStreamSynchronize(gr.cs[j]);
                for (cudaEvent_t e : gr.slot_ev[j]) cudaEventDestroy(e);
                if (j < gr.p_ev.size() && gr.p_ev[j]) cudaEventDestroy(gr.p_ev[j]);
                if (gr.cs[j]) cudaStreamDestroy(gr.cs[j]);
            }
        for (void* x : hosted) cudaFreeHost(x);
        if (pool) cudaFreeHost(pool);
        for (Member& m : M) {
            cudaSetDevice(m.id);
            if (m.blas) cublasDestroy(m.blas);
            if (m.s) cudaStreamDestroy(m.s);
        }
    }

    // (an error: every counter far ahead, so no device waits for the host for good)
    void release_all() {
        if (ftap) *(volatile unsigned*) ftap = seq_tap + 0x40000000u;
        for (Group& gr : G)
            if (gr.fin) *(volatile unsigned*) gr.fin = gr.seq_in + 0x40000000u;
        if (fhead) *(volatile unsigned*) fhead = seq_head + 0x40000000u;
    }

    template <class T>
    bool dalloc(int dev, T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaSetDevice(dev) != cudaSuccess || cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "tp: CUDA" + std::to_string(dev) + ": out of device memory (" + std::to_string((n * sizeof(T)) >> 20) +
                  " MB)";
            return false;
        }
        owned.push_back(v);
        owned_dev.push_back(dev);
        acct[dev][(size_t) acat] += std::max<size_t>(n * sizeof(T), 16);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, unsigned flags, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, std::max<size_t>(n * sizeof(T), 64), flags) != cudaSuccess) {
            err = "tp: pinned host memory (" + std::to_string((n * sizeof(T)) >> 20) + " MB) not allocated";
            return false;
        }
        std::memset(v, 0, std::max<size_t>(n * sizeof(T), 64));
        hosted.push_back(v);
        p = (T*) v;
        return true;
    }

    // an EXL3 matrix's slice on device dev
    bool upload_slice(int dev, const std::string& pre, char by, int lo, int hi, x3::Mat& m, std::string& err) {
        Ex3 x;
        if (!ex3(pk, pre, x, err)) return false;
        const int full = by == 'n' ? x.N : x.K;
        if (lo % 128 || hi % 128 || lo < 0 || hi > full || lo >= hi) {
            err = "tp: " + pre + ": a slice [" + std::to_string(lo) + ", " + std::to_string(hi) + ") of " +
                  std::to_string(full);
            return false;
        }
        const size_t a = tr_bytes(x, by, lo, hi), b = suh_bytes(x, by, lo, hi), c = svh_bytes(x, by, lo, hi);
        std::vector<uint8_t> hb(a + b + c);
        cut(x, by, lo, hi, hb.data(), hb.data() + a, hb.data() + a + b);
        uint8_t* d = nullptr;
        if (!dalloc(dev, d, up(a) + up(b) + up(c), err)) return false;
        TK(cudaMemcpy(d, hb.data(), a, cudaMemcpyHostToDevice));
        TK(cudaMemcpy(d + up(a), hb.data() + a, b, cudaMemcpyHostToDevice));
        TK(cudaMemcpy(d + up(a) + up(b), hb.data() + a + b, c, cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.trellis = (const uint32_t*) d;
        m.suh = (const uint16_t*) (d + up(a));
        m.svh = (const uint16_t*) (d + up(a) + up(b));
        m.k = by == 'k' ? hi - lo : x.K;
        m.n = by == 'n' ? hi - lo : x.N;
        m.k2 = x.k2;
        m.cb = 2;
        if (!x3::supported(m)) {
            err = "tp: " + pre + ": a slice the EXL3 kernels do not cover";
            return false;
        }
        return true;
    }

    bool upload_f32(int dev, const std::vector<float>& v, float*& out, std::string& err) {
        if (!dalloc(dev, out, v.size(), err)) return false;
        TK(cudaMemcpy(out, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return true;
    }

    // expert e of layer l, member m's slice (of tm) -> dst (l.lay)
    void assemble(const Layer& l, int e, int m, int tm, uint8_t* dst) const {
        const int f0 = m * (g.ff / tm), f1 = f0 + g.ff / tm;
        for (int r = 0; r < 3; ++r) {
            const Ex3& x = l.ex[(size_t) e][(size_t) r];
            cut(x, r < 2 ? 'n' : 'k', f0, f1, dst + l.lay.trellis[r], dst + l.lay.suh[r], dst + l.lay.svh[r]);
        }
    }

    bool load(const Options& o_, std::string& err) {
        o = o_;
        const double t0 = now();
        if (!pk.load(o.pack, err) || !g.read(pk, err)) return false;
        if (const char* pr = ::mm3::env("MM3_PROMOTE"))
            std::sscanf(pr, "%f,%d,%f", &kPromote, &kMaxPending, &kDecay);
        tp = o.tp;
        chunk = std::max(kWin + 1, std::min(o.tp_chunk, 8192));
        ffn_f = o.tp_ffn_rows > 0 ? std::max(o.tp_ffn_rows, (chunk + kMaxSlices - 1) / kMaxSlices) : chunk;
        if (const char* sv = ::mm3::env("MM3_ATTN_SPLIT")) attn_splits = std::max(1, std::atoi(sv));
        attn_splits = std::min(attn_splits, g.topk);
        if (tp != 1 && tp != 2 && tp != 4) {
            err = "--tp " + std::to_string(tp) + ": 1, 2 or 4 (a member: 16 query heads a KV head)";
            return false;
        }
        if ((g.ff / tp) % 128 || (g.ff_dense / tp) % 128 || (g.ff_shared / tp) % 128) {
            err = "--tp " + std::to_string(tp) + ": the FFN widths do not cut in 128s";
            return false;
        }
        // the GPUs
        int n = 0;
        TK(cudaGetDeviceCount(&n));
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < n; ++i) ids.push_back(i);
        for (int id : ids) {
            if (id < 0 || id >= n) {
                err = "tp: no CUDA" + std::to_string(id);
                return false;
            }
            Member m;
            m.id = id;
            TK(cudaSetDevice(id));
            TK(cudaStreamCreateWithFlags(&m.s, cudaStreamNonBlocking));
            if (cublasCreate(&m.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(m.blas, m.s) != CUBLAS_STATUS_SUCCESS ||
                cublasSetMathMode(m.blas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
                err = "tp: cuBLAS on CUDA" + std::to_string(id);
                return false;
            }
            M.push_back(std::move(m));
        }
        // the groups: the first GPU alone when the others make whole groups (it keeps the head), else all in groups
        const int ng = (int) M.size();
        if (tp > 1 && (ng - 1) % tp == 0 && ng > 1) {
            G.push_back(Group{{0}});
            for (int i = 1; i < ng; i += tp) {
                Group gr;
                for (int j = 0; j < tp; ++j) gr.mem.push_back(i + j);
                G.push_back(gr);
            }
        } else if (ng % tp == 0) {
            for (int i = 0; i < ng; i += tp) {
                Group gr;
                for (int j = 0; j < tp; ++j) gr.mem.push_back(i + j);
                G.push_back(gr);
            }
        } else {
            err = "--tp " + std::to_string(tp) + ": " + std::to_string(ng) + " GPUs make no groups of it (all, or all "
                  "but the first)";
            return false;
        }
        // (a group of one GPU: 1 member taking the whole layer)
        head_m = 0;
        // the head, the draft and the vision tower on the first GPU, before the layers take the memory
        {
            Member& hm = M[(size_t) head_m];
            TK(cudaSetDevice(hm.id));
            if (!host_f32(pk, P + "norm.weight", 1.0f, norm1, err)) return false;
            Ex3 x;
            acat = kAW;
            if (!ex3(pk, "language_model.lm_head", x, err) ||
                !upload_slice(hm.id, "language_model.lm_head", 'n', 0, x.N, head, err))
                return false;
            acat = kAAct;
            if (!dalloc(hm.id, lg, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(lgh, (size_t) kWin * g.vocab, cudaHostAllocPortable, err)) return false;
            if (!halloc(cand_h, (size_t) kWin, cudaHostAllocMapped | cudaHostAllocPortable, err) ||
                !halloc(cand_ex, (size_t) k::kCandExtra, cudaHostAllocMapped | cudaHostAllocPortable, err))
                return false;
            if (!halloc(xf, (size_t) kWin * g.dim, cudaHostAllocMapped | cudaHostAllocPortable, err) ||
                !halloc(fhead, kLine, cudaHostAllocMapped | cudaHostAllocPortable, err))
                return false;
            TK(cudaEventCreateWithFlags(&head_ev, cudaEventDisableTiming));
            tap_of.assign((size_t) g.n_layers, -1);
            if (o.dspark) {
                draftm = std::make_unique<Draft>();
                const Tensor* et = pk.get(P + "embed_tokens.weight");
                // (on the head member's own stream: a device's work that waits on the host goes on one stream, in
                //  order - the draft's context steps on a second stream, parked behind a wait, held up the first)
                if (!draftm->load(pk, hm.id, hm.s, hm.blas, &head, (const uint16_t*) pk.data(*et), kDraftRows, err))
                    return false;
                const auto& tl = draftm->geo().taps;
                for (size_t j = 0; j < tl.size(); ++j) {
                    if (tl[j] < 0 || tl[j] >= g.n_layers) {
                        err = "dspark: a tap at layer " + std::to_string(tl[j]);
                        return false;
                    }
                    tap_of[(size_t) tl[j]] = (int) j;
                }
                acat = kADraft;
                // (taps_d / taps_h a window's; a prompt's in tap_ring, the last tap_rows positions - the draft's window
                //  and more - read where they are by its context)
                const bool tok = dalloc(hm.id, taps_d, tl.size() * (size_t) kWin * g.dim, err);
                acat = kAAct;
                tap_rows = draftm->geo().window + 64;
                if (!tok ||
                    !halloc(taps_h, tl.size() * (size_t) kWin * g.dim, cudaHostAllocMapped | cudaHostAllocPortable, err) ||
                    !halloc(tap_ring, tl.size() * (size_t) tap_rows * g.dim, cudaHostAllocMapped | cudaHostAllocPortable,
                            err) ||
                    !halloc(ftap, kLine, cudaHostAllocMapped | cudaHostAllocPortable, err))
                    return false;
                // (every window size once now: cuBLAS takes its workspace and picks its kernels here, not during a
                //  forward - the devices then wait on the host, which must not wait on them)
                TK(cudaMemset(taps_d, 0, tl.size() * (size_t) kWin * g.dim * 4));
                for (int t = 1; t <= kWin; ++t) {
                    for (size_t j = 0; j < tl.size(); ++j)
                        if (!draftm->context_tap((int) j, taps_d + j * kWin * g.dim, t, err)) return false;
                    if (!draftm->context_finish(t, 0, err)) return false;
                }
                info.push_back("DSpark draft: " + std::to_string(draftm->bytes() >> 20) + " MB on CUDA" +
                               std::to_string(hm.id));
            }
            if (o.vision && o.vision_auto && !Vision::in_pack(pk)) o.vision = false;   // (by default: text only then)
            if (o.vision) {
                if (!Vision::in_pack(pk)) {
                    err = "--vision: the pack has no vision tower (pack it again with tools/mm3/pack.py)";
                    return false;
                }
                vis = std::make_unique<Vision>();
                const int vd = M[(size_t) std::clamp(o.vision_gpu, 0, (int) M.size() - 1)].id;
                if (!vis->load(pk, vd, err)) {   // (before the layers: what fails here is this GPU's own memory)
                    if (o.vision_must) return false;
                    info.push_back("NOTE: vision off: the tower did not load on CUDA" + std::to_string(vd) + " (" + err +
                                   ") - text only");
                    std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
                    err.clear();
                    cudaGetLastError();
                    vis.reset();
                    o.vision = false;
                }
                if (vis) {
                    vis->set_lender([this, vd](size_t need) { return lend_slots(vd, need); },
                                    [this]() { drop_slots(); });
                    info.push_back("vision tower: on demand - its " + std::to_string(vis->host_bytes() >> 20) +
                                   " MB in host memory (the pack's mapping), a picture at a time on CUDA" +
                                   std::to_string(vd) + " (up to " + std::to_string(vis->room_bytes() >> 20) +
                                   " MB, given back after it)");
                    info.push_back("NOTE: pictures: the vision tower is not kept in VRAM - each picture copies it in, "
                                   "the first one slower (its weights read from the SSD)");
                }
            }
        }
        for (Member& m : M) {
            TK(cudaSetDevice(m.id));
            size_t fr = 0, tot = 0;
            TK(cudaMemGetInfo(&fr, &tot));
            m.free0 = fr;
        }
        // the layers over the groups by their room (a group: its tightest member's free memory, tp times)
        {
            std::vector<double> room(G.size());
            double total = 0;
            for (size_t gi = 0; gi < G.size(); ++gi) {
                size_t mn = SIZE_MAX;
                for (int mi : G[gi].mem) mn = std::min(mn, M[(size_t) mi].free0);
                room[gi] = (double) mn * G[gi].mem.size();
                total += room[gi];
            }
            L.assign((size_t) g.n_layers, Layer{});
            double acc = 0;
            size_t gi = 0;
            for (int li = 0; li < g.n_layers; ++li) {
                const double mid = (li + 0.5) / g.n_layers * total;
                while (gi + 1 < G.size() && acc + room[gi] < mid) acc += room[gi++];
                L[(size_t) li].group = (int) gi;
            }
            for (size_t i = 0; i < G.size(); ++i) {
                G[i].l0 = g.n_layers;
                G[i].l1 = 0;
            }
            for (int li = 0; li < g.n_layers; ++li) {
                Group& gr = G[(size_t) L[(size_t) li].group];
                gr.l0 = std::min(gr.l0, li);
                gr.l1 = std::max(gr.l1, li + 1);
            }
            G.erase(std::remove_if(G.begin(), G.end(), [](const Group& x) { return x.l1 <= x.l0; }), G.end());
            for (int li = 0; li < g.n_layers; ++li)
                for (size_t i = 0; i < G.size(); ++i)
                    if (li >= G[i].l0 && li < G[i].l1) L[(size_t) li].group = (int) i;
        }
        inv_freq.resize((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_freq.data());
        plan_bytes = (size_t) kWin * 8 * 8 * 2 + (size_t) kWin * 8 * 4 * 3 + 64;
        const size_t nblk = (size_t) (o.max_ctx + g.blk - 1) / g.blk;
        // KV streaming (--kv-resident): the VRAM caches a ring of the last kv_ring positions (whole blocks, 4096 at
        // least - more than a prompt chunk), the rest of the context read from host memory; 0: all in VRAM
        int kv_ring = 0;
        if (o.kv_resident > 0) {
            kv_ring = std::max(4096, (o.kv_resident + g.blk - 1) / g.blk * g.blk);
            if (kv_ring >= o.max_ctx) kv_ring = 0;
        }
        // the members' scratch, the groups' mapped buffers
        for (Group& gr : G) {
            const int t = (int) gr.mem.size();
            const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
            const int mfs = g.ff / t, mfd = g.ff_dense / t, mfss = g.ff_shared / t;
            for (int mi : gr.mem) {
                Member& m = M[(size_t) mi];
                TK(cudaSetDevice(m.id));
                const int dv = m.id;
                auto A = [&](auto*& p, size_t c) { return dalloc(dv, p, c, err); };
                const int ffmax = std::max({mfs, mfd, mfss});
                if (!A(m.xn, (size_t) kWin * g.dim) || !A(m.q, (size_t) kWin * mh * g.hd) ||
                    !A(m.k, (size_t) kWin * mkv * g.hd) || !A(m.v, (size_t) kWin * mkv * g.hd) ||
                    !A(m.iq, (size_t) kWin * mix * g.ix_dim) || !A(m.ik, (size_t) kWin * g.ix_dim) ||
                    !A(m.bs, (size_t) mix * kWin * nblk) || !A(m.att, (size_t) kWin * mh * g.hd) ||
                    !A(m.gg, (size_t) kWin * ffmax) || !A(m.uu, (size_t) kWin * ffmax) || !A(m.sh, (size_t) kWin * g.dim) ||
                    !A(m.out, (size_t) kWin * g.dim) || !A(m.win_h, (size_t) kWin * 8 * mfs) ||
                    !A(m.win_part, (size_t) kWin * 8 * g.dim) || !A(m.inv, inv_freq.size()) || !A(m.pos, (size_t) kWin) ||
                    !A(m.kept, (size_t) mix * kWin * g.topk) || !A(m.plan_d, plan_bytes) || !A(m.counter, (size_t) 16) ||
                    !A(m.rlog_d, (size_t) kWin * g.n_exp) || !A(m.attp, k::attn_part_floats(kWin, mkv, g.topk)))
                    return false;
                TK(cudaMemset(m.counter, 0, 64));
                acat = kAPrompt;
                {   // the prompt path's scratch
                    // (the attention's a chunk's rows; the FFN's a slice's - F tokens, FK = F K routed rows)
                    const size_t C = chunk, rows = C * g.k_exp, F = (size_t) ffn_rows(), FK = F * g.k_exp;
                    const size_t ns = (size_t) n_slices_max();
                    const int ffd = std::max(mfd, mfss);
                    if (!A(m.p_xn, C * g.dim) || !A(m.p_xn16, C * g.dim) || !A(m.p_q, C * mh * g.hd) ||
                        !A(m.p_k, C * mkv * g.hd) || !A(m.p_v, C * mkv * g.hd) || !A(m.p_iq, C * mix * g.ix_dim) ||
                        !A(m.p_ik, C * g.ix_dim) || !A(m.p_bs, (size_t) mix * C * nblk) || !A(m.p_kept, (size_t) mix * C * g.topk) ||
                        !A(m.p_att, C * mh * g.hd) || !A(m.p_att16, C * mh * g.hd) || !A(m.p_sh, F * g.dim) ||
                        !A(m.p_h16, std::max(FK * mfs, F * ffd)) || !A(m.p_y, (FK + 1) * g.dim) ||
                        !A(m.p_w, rows) || !A(m.p_rlog, C * g.n_exp) || !A(m.p_pos, C) || !A(m.p_rowidx, rows) ||
                        !A(m.p_invrow, rows) || !A(m.p_bres, ns * ((size_t) g.n_exp + 1)) || !A(m.p_bstg, ns * 4097))
                        return false;
                    // (the dense FFN's and the shared expert's gate / up rows in p_y, which the routed experts use
                    //  only after them; the outputs into p_xn, whose readers are all before them on the stream)
                    if (2 * F * ffd <= FK * g.dim) {
                        m.p_gg = m.p_y;
                        m.p_uu = m.p_y + F * ffd;
                    } else if (!A(m.p_gg, F * ffd) || !A(m.p_uu, F * ffd)) {
                        return false;
                    }
                    TK(cudaMemset(m.p_y + FK * g.dim, 0, (size_t) g.dim * 4));   // (the CPU lane's entries' row)
                }
                acat = kAAct;
                TK(cudaMemcpy(m.inv, inv_freq.data(), inv_freq.size() * 4, cudaMemcpyHostToDevice));
                if (!halloc(m.pos_h, (size_t) chunk, cudaHostAllocPortable, err)) return false;
                if (!x3::prepare(m.s)) {
                    err = "tp: the EXL3 scratch on CUDA" + std::to_string(m.id);
                    return false;
                }
            }
            const unsigned mapped = cudaHostAllocMapped | cudaHostAllocPortable;
            if (!halloc(gr.p_xa, (size_t) chunk * g.dim, mapped, err) || !halloc(gr.p_xb, (size_t) chunk * g.dim, mapped, err) ||
                !halloc(gr.p_part, (size_t) t * chunk * g.dim, mapped, err) ||
                !halloc(gr.p_rlog, (size_t) chunk * g.n_exp, mapped, err) || !halloc(gr.p_plan, plan_p_bytes(), mapped, err))
                return false;
            if (!halloc(gr.xa, (size_t) kWin * g.dim, mapped, err) || !halloc(gr.xb, (size_t) kWin * g.dim, mapped, err) ||
                !halloc(gr.part, (size_t) t * kWin * g.dim, mapped, err) || !halloc(gr.plans, (size_t) t * plan_bytes, mapped, err) ||
                !halloc(gr.fin, kLine, mapped, err) || !halloc(gr.fout, (size_t) t * kLine, mapped, err) ||
                !halloc(gr.rlog, (size_t) kWin * g.n_exp, mapped, err) || !halloc(gr.frout, kLine, mapped, err))
                return false;
            gr.seq_out.assign((size_t) t, 0);
        }
        // the layers' slices, member by member
        acat = kAW;
        for (int li = 0; li < g.n_layers; ++li) {
            Layer& l = L[(size_t) li];
            Group& gr = G[(size_t) l.group];
            const int t = (int) gr.mem.size();
            const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
            const int mfs = g.ff / t, mfd = g.ff_dense / t, mfss = g.ff_shared / t;
            const std::string N = P + "layers." + std::to_string(li) + ".";
            const std::string AT = N + "self_attn.";
            l.moe = g.moe[(size_t) li] != 0;
            std::vector<float> qn1, kn1, iqn1, ikn1, wiq, wik;
            if (!host_f32(pk, N + "input_layernorm.weight", 1.0f, l.in1, err) ||
                !host_f32(pk, N + "post_attention_layernorm.weight", 1.0f, l.post1, err) ||
                !host_f32(pk, AT + "q_norm.weight", 1.0f, qn1, err) || !host_f32(pk, AT + "k_norm.weight", 1.0f, kn1, err) ||
                !host_f32(pk, AT + "index_q_norm.weight", 1.0f, iqn1, err) ||
                !host_f32(pk, AT + "index_k_norm.weight", 1.0f, ikn1, err) ||
                !host_f32(pk, AT + "index_q_proj.weight", 0.0f, wiq, err) ||
                !host_f32(pk, AT + "index_k_proj.weight", 0.0f, wik, err))
                return false;
            if (l.moe) {
                const std::string Mo = N + "block_sparse_moe.";
                if (!host_f32(pk, Mo + "gate.weight", 0.0f, l.gate, err) ||
                    !host_f32(pk, Mo + "e_score_correction_bias", 0.0f, l.bias, err))
                    return false;
                // the experts in the files, and a member's slice's layout
                l.ex.resize((size_t) g.n_exp);
                const char* parts[3] = {"w1", "w3", "w2"};
                for (int e = 0; e < g.n_exp; ++e)
                    for (int r = 0; r < 3; ++r)
                        if (!ex3(pk, Mo + "experts." + std::to_string(e) + "." + parts[r], l.ex[(size_t) e][(size_t) r], err))
                            return false;
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const Ex3& x = l.ex[0][(size_t) r];
                    const char by = r < 2 ? 'n' : 'k';
                    l.lay.trellis[r] = off;
                    off += up(tr_bytes(x, by, 0, mfs));
                    l.lay.suh[r] = off;
                    off += up(suh_bytes(x, by, 0, mfs));
                    l.lay.svh[r] = off;
                    off += up(svh_bytes(x, by, 0, mfs));
                    l.lay.k2[r] = x.k2;
                }
                l.lay.cb = 2;
                l.blob = off;
                if (l.lay.k2[0] != l.lay.k2[1]) {
                    err = "tp: layer " + std::to_string(li) + ": gate and up at different bits";
                    return false;
                }
                for (int e = 1; e < g.n_exp; ++e)
                    for (int r = 0; r < 3; ++r)
                        if (l.ex[(size_t) e][(size_t) r].k2 != l.lay.k2[r] || l.ex[(size_t) e][(size_t) r].K != l.ex[0][(size_t) r].K) {
                            err = "tp: layer " + std::to_string(li) + ": experts of different shapes or bits";
                            return false;
                        }
            }
            l.sl.resize((size_t) t);
            for (int j = 0; j < t; ++j) {
                Member& m = M[(size_t) gr.mem[(size_t) j]];
                Slice& S = l.sl[(size_t) j];
                TK(cudaSetDevice(m.id));
                const int dv = m.id;
                // attention: its query heads, KV heads, index heads; o_proj's columns of its heads
                if (!upload_slice(dv, AT + "q_proj", 'n', j * mh * g.hd, (j + 1) * mh * g.hd, S.q, err) ||
                    !upload_slice(dv, AT + "k_proj", 'n', j * mkv * g.hd, (j + 1) * mkv * g.hd, S.k, err) ||
                    !upload_slice(dv, AT + "v_proj", 'n', j * mkv * g.hd, (j + 1) * mkv * g.hd, S.v, err) ||
                    !upload_slice(dv, AT + "o_proj", 'k', j * mh * g.hd, (j + 1) * mh * g.hd, S.o, err))
                    return false;
                std::vector<float> wq(wiq.begin() + (size_t) j * mix * g.ix_dim * g.dim,
                                      wiq.begin() + (size_t) (j + 1) * mix * g.ix_dim * g.dim);
                if (!upload_f32(dv, qn1, S.qn1, err) || !upload_f32(dv, kn1, S.kn1, err) ||
                    !upload_f32(dv, iqn1, S.iqn1, err) || !upload_f32(dv, ikn1, S.ikn1, err) ||
                    !upload_f32(dv, wq, S.wiq, err) || !upload_f32(dv, wik, S.wik, err))
                    return false;
                if (l.moe) {
                    if (j == 0 && !upload_f32(dv, l.gate, S.gate, err)) return false;
                    if (!upload_slice(dv, N + "block_sparse_moe.shared_experts.gate_proj", 'n', j * mfss, (j + 1) * mfss, S.sg, err) ||
                        !upload_slice(dv, N + "block_sparse_moe.shared_experts.up_proj", 'n', j * mfss, (j + 1) * mfss, S.su, err) ||
                        !upload_slice(dv, N + "block_sparse_moe.shared_experts.down_proj", 'k', j * mfss, (j + 1) * mfss, S.sd, err))
                        return false;
                } else {
                    if (!upload_slice(dv, N + "mlp.gate_proj", 'n', j * mfd, (j + 1) * mfd, S.mg, err) ||
                        !upload_slice(dv, N + "mlp.up_proj", 'n', j * mfd, (j + 1) * mfd, S.mu, err) ||
                        !upload_slice(dv, N + "mlp.down_proj", 'k', j * mfd, (j + 1) * mfd, S.md, err))
                        return false;
                }
                // (KV streaming: the VRAM caches a ring of kv_ring rows, tagged - the host's below)
                acat = kAKV;
                const size_t kvrows = (size_t) (kv_ring > 0 ? kv_ring : o.max_ctx) * mkv;
                uint8_t* kvb = nullptr;
                if (!dalloc(dv, kvb, k::kv_bytes(o.kv_k, o.kv_v, kvrows), err)) return false;
                k::kv_carve(kvb, kvrows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, S.kc, S.vc, S.kvq);
                if (!dalloc(dv, S.ic, (size_t) o.max_ctx * g.ix_dim, err)) return false;
                acct_ic[dv] += (size_t) o.max_ctx * g.ix_dim * 2;
                if (kv_ring > 0) {
                    S.kvh.ring = kv_ring;
                    if (!dalloc(dv, S.kvh.tag, (size_t) kv_ring, err)) return false;
                    TK(cudaMemset(S.kvh.tag, 0xff, (size_t) kv_ring * 4));
                }
                acat = kAW;
            }
        }
        acat = kAAct;
        // KV streaming: every slice's whole cache in host memory (one pinned, mapped allocation, carved), written as
        // the VRAM ring is and read where the ring no longer holds a key
        if (kv_ring > 0) {
            auto bytes_of = [&](int mkv) { return k::kv_bytes(o.kv_k, o.kv_v, (size_t) o.max_ctx * mkv); };
            size_t total = 0;
            for (const Layer& l : L) {
                const Group& gr = G[(size_t) l.group];
                total += gr.mem.size() * bytes_of(g.n_kv / (int) gr.mem.size());
            }
            // (the RAM this process can still get must hold it beside a tenth of the RAM, 4..16 GB: a clear refusal
            //  before the OS is asked to pin it)
            const double tot_ram = (double) os::ram_total(), avail = (double) os::ram_avail();
            const double head = tot_ram > 0 ? std::min(16.0, std::max(4.0, tot_ram / 10.0 / (1u << 30))) * (1u << 30)
                                            : 8.0 * (1u << 30);
            if (avail > 0 && (double) total + head > avail) {
                err = "tp: KV streaming needs " + std::to_string((int) ((double) total / (1u << 30) + 0.5)) +
                      " GB of pinned RAM for " + std::to_string(o.max_ctx) + " positions, " +
                      std::to_string((int) std::max(0.0, (avail - head) / (1u << 30))) +
                      " GB to spare: a smaller --max-context, or --tp 0 (the layer split sizes it itself)";
                return false;
            }
            uint8_t* kvp = nullptr;
            if (!halloc(kvp, total, cudaHostAllocMapped | cudaHostAllocPortable, err)) return false;
            size_t off = 0;
            for (Layer& l : L) {
                const Group& gr = G[(size_t) l.group];
                const int mkv = g.n_kv / (int) gr.mem.size();
                const size_t rows = (size_t) o.max_ctx * mkv, nb = bytes_of(mkv);
                for (size_t j = 0; j < gr.mem.size(); ++j) {
                    k::KvHost& h = l.sl[j].kvh;
                    k::kv_carve(kvp + off, rows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, h.kc, h.vc, h.q);
                    off += nb;
                    acct_kvh[M[(size_t) gr.mem[j]].id] += nb;
                }
            }
            st.kv_ring = kv_ring;
            st.kv_host_gb = (double) total / (1u << 30);
        }
        // the profile: routings [layer][expert]
        prof.assign((size_t) g.n_layers * g.n_exp, 0);
        bool have_prof = false;
        if (!o.profile.empty()) {
            if (std::FILE* f = std::fopen(o.profile.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long sz = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (sz == (long) (prof.size() * 4) && std::fread(prof.data(), 4, prof.size(), f) == prof.size())
                    have_prof = true;
                std::fclose(f);
            }
            if (!have_prof) std::fill(prof.begin(), prof.end(), 0u);
        }
        acat = kAPool;
        // the routed experts in VRAM, group by group: the most routed of its layers first while its tightest member
        // keeps reserve_gb free (each member holds its slice of every one).  A prompt stages a layer's experts outside
        // VRAM all at once in the group's cache, so the cache and the residents share the measured free memory: at
        // least --cache-gb of slots, more where the memory is tight - each layer then keeps n_exp - slots experts in
        // VRAM at least; an error (the layer split instead: Model::load) only when not even a whole layer's experts
        // fit as slots
        const int cap = o.resident_cap < 0 ? g.n_exp : std::min(o.resident_cap, g.n_exp);
        for (size_t gi = 0; gi < G.size(); ++gi) {
            Group& gr = G[gi];
            size_t budget = SIZE_MAX;
            for (int mi : gr.mem) {
                TK(cudaSetDevice(M[(size_t) mi].id));
                size_t fr = 0, tot = 0;
                TK(cudaMemGetInfo(&fr, &tot));
                const size_t reserve = (size_t) ((o.reserve_gb < 0 ? 0.3 : o.reserve_gb) * (1u << 30));
                budget = std::min(budget, fr > reserve ? fr - reserve : 0);
            }
            std::vector<int> ml;
            size_t blob_max = 0;
            for (int li = gr.l0; li < gr.l1; ++li)
                if (L[(size_t) li].moe) {
                    ml.push_back(li);
                    blob_max = std::max(blob_max, L[(size_t) li].blob);
                }
            gr.slot_bytes = blob_max;
            int floor_res = 0;   // (each MoE layer's experts in VRAM at least)
            if (blob_max > 0) {
                const int asked = std::max(1, (int) ((size_t) (std::max(0.0, o.cache_gb) * (1u << 30)) / blob_max));
                int S = std::min(g.n_exp, std::max(asked, g.n_exp - cap));
                auto cost = [&](int s) {
                    return (double) s * (double) blob_max + (double) ml.size() * (double) (g.n_exp - s) * (double) blob_max;
                };
                while (S < g.n_exp && cost(S) > (double) budget) ++S;
                if (cost(S) > (double) budget) {
                    err = "tp: layers " + std::to_string(gr.l0) + "-" + std::to_string(gr.l1 - 1) + " do not fit: a "
                          "layer's " + std::to_string(g.n_exp) + " expert slices (" +
                          std::to_string(((size_t) g.n_exp * blob_max) >> 20) + " MB) need more than the " +
                          std::to_string(budget >> 20) + " MB free on its tightest GPU";
                    return false;
                }
                if (S > asked)
                    info.push_back("NOTE: tp: layers " + std::to_string(gr.l0) + "-" + std::to_string(gr.l1 - 1) +
                                   ": an expert cache of " + std::to_string(S) + " slots (" +
                                   std::to_string(((size_t) S * blob_max) >> 20) + " MB a GPU, --cache-gb asked " +
                                   std::to_string(asked) + "): the GPU memory is tight, a prompt stages a layer's "
                                   "experts outside VRAM at once");
                gr.n_slots = S;
                floor_res = g.n_exp - S;
                budget -= (size_t) S * blob_max;
            }
            if (o.verbose) {
                std::fprintf(stderr, "tp: group %zu (layers %d-%d): expert budget %zu MB a member, free", gi, gr.l0,
                             gr.l1 - 1, budget >> 20);
                for (int mi : gr.mem) {
                    cudaSetDevice(M[(size_t) mi].id);
                    size_t fr = 0, tot = 0;
                    cudaMemGetInfo(&fr, &tot);
                    std::fprintf(stderr, " CUDA%d %zu MB", M[(size_t) mi].id, fr >> 20);
                }
                std::fprintf(stderr, "\n");
            }
            struct C {
                uint32_t n;
                int li, e;
            };
            std::vector<C> c;
            for (int li : ml)
                for (int e = 0; e < g.n_exp; ++e) c.push_back({have_prof ? prof[(size_t) li * g.n_exp + e] : 0u, li, e});
            if (have_prof)
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.n > b.n; });
            else   // (round robin: expert 0 of every layer, then 1, ...)
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.e < b.e; });
            std::map<int, std::vector<int>> chosen;
            std::vector<char> took(c.size(), 0);
            for (int pass = 0; pass < 2; ++pass)   // (each layer's floor first - its most routed - then by the profile)
                for (size_t i = 0; i < c.size(); ++i) {
                    const C& x = c[i];
                    const Layer& l = L[(size_t) x.li];
                    const int lim = pass == 0 ? floor_res : cap;
                    if (!took[i] && (int) chosen[x.li].size() < lim && budget >= l.blob) {
                        budget -= l.blob;
                        chosen[x.li].push_back(x.e);
                        took[i] = 1;
                    }
                }
            for (int li : ml) {
                Layer& l = L[(size_t) li];
                std::vector<int>& v = chosen[li];
                std::sort(v.begin(), v.end());
                l.res_slot.assign((size_t) g.n_exp, -1);
                l.res_ids = v;
                for (size_t s = 0; s < v.size(); ++s) l.res_slot[(size_t) v[s]] = (int) s;
                st.resident += (long) v.size();
                st.vram_gb += (double) (v.size() * l.blob * gr.mem.size()) / (1u << 30);
                for (size_t j = 0; j < gr.mem.size(); ++j) {
                    Member& m = M[(size_t) gr.mem[j]];
                    if (!v.empty() && !dalloc(m.id, l.sl[j].res_base, v.size() * l.blob, err)) return false;
                }
            }
        }
        for (const Layer& l : L)
            if (l.moe) st.experts += g.n_exp;
        // the resident slices from the files, a thread a GPU (two pinned buffers each, alternating)
        {
            std::vector<std::thread> th;
            std::atomic<int> bad{0};
            for (size_t gi = 0; gi < G.size(); ++gi)
                for (size_t j = 0; j < G[gi].mem.size(); ++j)
                    th.emplace_back([&, gi, j] {
                        Member& m = M[(size_t) G[gi].mem[j]];
                        cudaSetDevice(m.id);
                        size_t mx = 0;
                        for (int li = G[gi].l0; li < G[gi].l1; ++li) mx = std::max(mx, L[(size_t) li].blob);
                        if (mx == 0) return;
                        uint8_t* pin[2] = {nullptr, nullptr};
                        cudaEvent_t ev[2];
                        for (int b = 0; b < 2; ++b) {
                            cudaMallocHost(&pin[b], mx);
                            cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming);
                            cudaEventRecord(ev[b], m.s);
                        }
                        int k = 0;
                        for (int li = G[gi].l0; li < G[gi].l1; ++li) {
                            const Layer& l = L[(size_t) li];
                            for (size_t s = 0; s < l.res_ids.size(); ++s, ++k) {
                                const int b = k & 1;
                                cudaEventSynchronize(ev[b]);
                                assemble(l, l.res_ids[s], (int) j, (int) G[gi].mem.size(), pin[b]);
                                cudaMemcpyAsync(l.sl[j].res_base + s * l.blob, pin[b], l.blob, cudaMemcpyHostToDevice, m.s);
                                cudaEventRecord(ev[b], m.s);
                            }
                        }
                        if (cudaStreamSynchronize(m.s) != cudaSuccess) ++bad;
                        for (int b = 0; b < 2; ++b) {
                            cudaFreeHost(pin[b]);
                            cudaEventDestroy(ev[b]);
                        }
                    });
            for (auto& x : th) x.join();
            if (bad) {
                err = "tp: copying the resident experts failed";
                return false;
            }
        }
        // the expert cache's slots, group by group (n_slots: sized with the residents above)
        for (Group& gr : G) {
            const int t = (int) gr.mem.size();
            if (gr.slot_bytes == 0 || gr.n_slots == 0) continue;
            gr.slot_base.assign((size_t) t, nullptr);
            gr.cs.assign((size_t) t, nullptr);
            gr.slot_ev.assign((size_t) t, {});
            acat = kAPool;
            for (int j = 0; j < t; ++j) {
                Member& m = M[(size_t) gr.mem[(size_t) j]];
                if (!dalloc(m.id, gr.slot_base[(size_t) j], (size_t) gr.n_slots * gr.slot_bytes, err)) return false;
                TK(cudaStreamCreateWithFlags(&gr.cs[(size_t) j], cudaStreamNonBlocking));
                gr.slot_ev[(size_t) j].resize((size_t) gr.n_slots);
                for (auto& e : gr.slot_ev[(size_t) j]) TK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            }
            gr.p_ev.assign((size_t) t, nullptr);
            for (int j = 0; j < t; ++j) {
                TK(cudaSetDevice(M[(size_t) gr.mem[(size_t) j]].id));
                TK(cudaEventCreateWithFlags(&gr.p_ev[(size_t) j], cudaEventDisableTiming));
            }
            // (the promotions' cap: a layer's experts outside VRAM, the most of the group's, stay free for prompts)
            int reserve = 0;
            for (int li = gr.l0; li < gr.l1; ++li)
                if (L[(size_t) li].moe) reserve = std::max(reserve, g.n_exp - (int) L[(size_t) li].res_ids.size());
            gr.slot_prom.assign((size_t) gr.n_slots, 0);
            gr.prom_cap = std::max(0, gr.n_slots - reserve);
            gr.slot_owner.assign((size_t) gr.n_slots, -1);
            gr.slot_pend.assign((size_t) gr.n_slots, -1);
            gr.slot_iss.assign((size_t) gr.n_slots, 0);
            acat = kAAct;
            gr.slot_used.assign((size_t) gr.n_slots, -1);
        }
        for (Layer& l : L)
            if (l.moe) l.slot_of.assign((size_t) g.n_exp, -1);
        // the pinned pool: every expert not in VRAM, its tp slices one after another (the CPU lane's sub-experts)
        {
            size_t total = 0;
            for (const Layer& l : L)
                if (l.moe) total += (size_t) (g.n_exp - (int) l.res_ids.size()) * l.blob * G[(size_t) l.group].mem.size();
            if (total > 0) {
                TK(cudaSetDevice(M[0].id));
                if (cudaHostAlloc((void**) &pool, total, cudaHostAllocPortable) != cudaSuccess) {
                    pool = nullptr;
                    err = "tp: the pinned pool (" + std::to_string(total >> 30) + " GB) not allocated";
                    return false;
                }
                std::vector<std::pair<int, int>> todo;
                size_t off = 0;
                for (int li = 0; li < g.n_layers; ++li) {
                    Layer& l = L[(size_t) li];
                    if (!l.moe) continue;
                    l.host.assign((size_t) g.n_exp, nullptr);
                    for (int e = 0; e < g.n_exp; ++e)
                        if (l.res_slot[(size_t) e] < 0) {
                            l.host[(size_t) e] = pool + off;
                            off += l.blob * G[(size_t) l.group].mem.size();
                            todo.push_back({li, e});
                        }
                }
                std::atomic<size_t> next{0};
                std::vector<std::thread> th;
                const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
                for (unsigned t = 0; t < nt; ++t)
                    th.emplace_back([&] {
                        for (size_t i; (i = next.fetch_add(1)) < todo.size();) {
                            const Layer& l = L[(size_t) todo[i].first];
                            const int e = todo[i].second;
                            const int tm = (int) G[(size_t) l.group].mem.size();
                            for (int j = 0; j < tm; ++j)
                                assemble(l, e, j, tm, (uint8_t*) l.host[(size_t) e] + (size_t) j * l.blob);
                        }
                    });
                for (auto& x : th) x.join();
                st.pinned_gb = (double) total / (1u << 30);
                st.pooled = (long) todo.size();
            }
        }
        // the CPU lane: sub-experts of a group's width (a member's slice; whole experts in a group of one GPU)
        {
            int mw = 0, me = 0;
            for (const Group& gr : G) {
                mw = std::max(mw, g.ff / (int) gr.mem.size());
                me = std::max(me, kWin * g.k_exp * (int) gr.mem.size());
            }
            lane = std::make_unique<CpuLane>(std::max(0, o.cpu_threads), g.dim, mw, g.alpha, g.limit, me);
        }
        h.resize((size_t) kWin * g.dim);
        pipe_buf.assign(G.size() + 1, std::vector<float>((size_t) chunk * g.dim));
        for (Group& gr : G) gr.p_cpu.assign((size_t) chunk * g.dim, 0.0f);
        use_score.assign((size_t) g.n_layers * 1024, 0.0f);   // (the pipeline's threads score their layers' keys)
        use_last.assign(use_score.size(), 0);
        if (const char* pc = std::getenv("MM3_PROMPT_CPU")) {
            prompt_cpu = std::string(pc) != "off";
            if (prompt_cpu && std::string(pc) != "on") std::sscanf(pc, "%lf,%lf,%lf", &pc_a, &pc_b, &pc_bw);
        }
        if (const char* pp = std::getenv("MM3_TP_PIPE")) pipe_on = std::atoi(pp) != 0;
        cpu_out.resize((size_t) kWin * g.dim);
        rl.resize((size_t) kWin * g.n_exp);
        for (Member& m : M) m.w->start();
        // the infos
        for (const Group& gr : G) {
            std::string s = "layers " + std::to_string(gr.l0) + "-" + std::to_string(gr.l1 - 1) + " on CUDA";
            for (size_t j = 0; j < gr.mem.size(); ++j)
                s += (j ? "," : "") + std::to_string(M[(size_t) gr.mem[j]].id);
            long nr = 0;
            for (int li = gr.l0; li < gr.l1; ++li) nr += (long) L[(size_t) li].res_ids.size();
            s += " (tp " + std::to_string(gr.mem.size()) + "), " + std::to_string(nr) + " experts in VRAM";
            if (gr.n_slots) s += ", a cache of " + std::to_string(gr.n_slots);
            info.push_back(s);
        }
        for (const Member& m : M) {
            cudaSetDevice(m.id);
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            info.push_back("CUDA" + std::to_string(m.id) + ": " + std::to_string(fr >> 20) + " MB free");
        }
        info.push_back("CPU lane: " + std::to_string(lane->threads()) + " threads");
        st.cpu_threads = lane->threads();
        {
            std::string kv = std::string("KV cache: K ") + k::kv_name(o.kv_k) + ", V " + k::kv_name(o.kv_v) +
                             (o.kv_rot && (o.kv_k || o.kv_v) ? " (rows rotated)" : "");
            if (kv_ring > 0)
                kv += ", streamed: " + std::to_string(kv_ring) + " positions a layer in VRAM, all " +
                      std::to_string(o.max_ctx) + " in host memory (" + std::to_string((int) (st.kv_host_gb + 0.5)) +
                      " GB pinned)";
            info.push_back(kv);
        }
        // the Monitor's map of each GPU: what each part holds there - the dense weights, the KV caches (less tp's
        // copies), the scratch, the DSpark draft, the routed experts' slots (resident and the cache), the vision tower,
        // the prompt path's buffers; tp: what every member of a group holds whole (the index-key caches), less one
        for (const Group& gr : G)
            for (int mi : gr.mem) {
                const Member& m = M[(size_t) mi];
                cudaSetDevice(m.id);
                size_t fr = 0, tot = 0;
                cudaMemGetInfo(&fr, &tot);
                std::array<size_t, kAN> a{};
                if (auto it = acct.find(m.id); it != acct.end()) a = it->second;
                const size_t tn = gr.mem.size();
                const size_t ic = acct_ic.count(m.id) ? acct_ic.at(m.id) : 0;
                const size_t tpb = tn > 1 ? ic / tn * (tn - 1) : 0;
                const bool head = mi == head_m;
                const size_t drafter = a[kADraft] + (head && draftm ? draftm->bytes() : 0);
                const size_t vision = vis && vis->device() == m.id ? vis->bytes() : 0;
                long long slots = gr.n_slots;
                for (int li = gr.l0; li < gr.l1; ++li) slots += (long long) L[(size_t) li].res_ids.size();
                char pci[32] = "-";
                if (cudaDeviceGetPCIBusId(pci, sizeof pci, m.id) != cudaSuccess) std::snprintf(pci, sizeof pci, "-");
                char b[512];
                std::snprintf(b, sizeof b,
                              "VRAM dev=%d pci=%s total=%zu free=%zu weights=%zu kv=%zu kv_ram=%zu act=%zu drafter=%zu "
                              "window=0 pool=%zu tail=0 borrow=0 vision=0 vision_own=%zu prompt=%zu tp=%zu tp_n=%zu "
                              "slots=%lld ctx=%d layers=%d-%d",
                              m.id, pci, tot, fr, a[kAW], a[kAKV] - std::min(a[kAKV], tpb),
                              acct_kvh.count(m.id) ? acct_kvh.at(m.id) : 0, a[kAAct], drafter, a[kAPool], vision,
                              a[kAPrompt], tpb, tn, slots, o.max_ctx, gr.l0, gr.l1 - 1);
                vram.push_back(b);
            }
        st.load_s = now() - t0;
        return true;
    }

    // ---------------------------------------------------------------- the expert cache
    long long fwd = 0;                        // windows so far
    std::vector<float> use_score;             // a non-resident expert's uses (decayed kDecay a window) ...
    std::vector<long long> use_last;          // ... as of this window
    // (MM3_PROMOTE="threshold,pending,decay": the promotion's knobs while they are tuned)
    float kDecay = 0.95f, kPromote = 1.8f;
    int kMaxPending = 16;   // (the copies run after the window, during the draft: 16 a group was best of 4 / 16)

    float score(long long key, bool add) {
        if (use_score.empty()) {
            use_score.assign((size_t) g.n_layers * 1024, 0.0f);
            use_last.assign(use_score.size(), 0);
        }
        float& sc = use_score[(size_t) key];
        sc *= std::pow(kDecay, (float) (fwd - use_last[(size_t) key]));
        use_last[(size_t) key] = fwd;
        if (add) sc += 1.0f;
        return sc;
    }

    // a window's start: the slots whose copies began promote_lag windows ago are their experts' (copies waited for)
    // a picture's room (Vision::set_lender) when its GPU has none free: the expert cache of the group holding that GPU,
    // between forwards - the group's cached experts dropped, staged or promoted again as the windows after it ask
    int lent_g = -1;
    void* lend_slots(int dev, size_t need) {
        for (size_t gi = 0; gi < G.size(); ++gi) {
            Group& gr = G[gi];
            for (size_t j = 0; j < gr.mem.size(); ++j) {
                if (M[(size_t) gr.mem[j]].id != dev || gr.n_slots == 0 || j >= gr.slot_base.size() ||
                    (size_t) gr.n_slots * gr.slot_bytes < need)
                    continue;
                for (size_t q = 0; q < gr.mem.size(); ++q) {   // (every member's: a slot is the same expert on all)
                    const Member& m = M[(size_t) gr.mem[q]];
                    cudaSetDevice(m.id);
                    if (cudaStreamSynchronize(m.s) != cudaSuccess ||
                        (q < gr.cs.size() && gr.cs[q] && cudaStreamSynchronize(gr.cs[q]) != cudaSuccess))
                        return nullptr;
                }
                cudaSetDevice(dev);
                lent_g = (int) gi;
                return gr.slot_base[j];
            }
        }
        return nullptr;
    }
    void drop_slots() {
        if (lent_g < 0) return;
        Group& gr = G[(size_t) lent_g];
        for (int s = 0; s < gr.n_slots; ++s) {
            const long long key = gr.slot_owner[(size_t) s];
            if (key >= 0) L[(size_t) (key / 1024)].slot_of[(size_t) (key % 1024)] = -1;
            gr.slot_owner[(size_t) s] = -1;
            gr.slot_pend[(size_t) s] = -1;
            gr.slot_used[(size_t) s] = -1;
            gr.slot_iss[(size_t) s] = 0;
            gr.slot_prom[(size_t) s] = 0;
        }
        gr.n_pending = 0;
        gr.n_prom = 0;
        gr.promo_q.clear();
        lent_g = -1;
    }

    bool refresh(Group& gr, std::string& err) {
        if (gr.n_pending == 0) return true;
        for (int s = 0; s < gr.n_slots; ++s) {
            const long long key = gr.slot_pend[(size_t) s];
            if (key < 0 || fwd - gr.slot_iss[(size_t) s] < o.promote_lag) continue;
            for (size_t j = 0; j < gr.mem.size(); ++j) {
                TK(cudaSetDevice(M[(size_t) gr.mem[j]].id));
                TK(cudaEventSynchronize(gr.slot_ev[j][(size_t) s]));
            }
            gr.slot_owner[(size_t) s] = key;
            gr.slot_pend[(size_t) s] = -1;
            gr.slot_used[(size_t) s] = fwd;
            L[(size_t) (key / 1024)].slot_of[(size_t) (key % 1024)] = s;
            --gr.n_pending;
        }
        return true;
    }

    // expert e of layer li ran on the CPU lane: when it keeps coming back, a slot (the one whose expert scores least,
    // not used in this window) is its, its copy after the window
    void consider(Group& gr, int li, int e) {
        const long long key = (long long) li * 1024 + e;
        const float sc = score(key, true);
        if (!o.promote || gr.n_slots == 0 || sc < kPromote || gr.n_pending >= kMaxPending) return;
        int best = -1;
        float bs = sc;
        const bool full = gr.n_prom >= gr.prom_cap;   // (then a promotion takes a promoted slot)
        for (int s = 0; s < gr.n_slots; ++s) {
            if (gr.slot_pend[(size_t) s] >= 0 || gr.slot_used[(size_t) s] == fwd) continue;
            if (full && !gr.slot_prom[(size_t) s]) continue;
            const long long ok = gr.slot_owner[(size_t) s];
            const float os = ok < 0 ? -1.0f : score(ok, false);
            if (os < bs) {
                bs = os;
                best = s;
            }
        }
        if (best < 0) return;
        if (!gr.slot_prom[(size_t) best]) {
            gr.slot_prom[(size_t) best] = 1;
            ++gr.n_prom;
        }
        const long long old = gr.slot_owner[(size_t) best];
        if (old >= 0) L[(size_t) (old / 1024)].slot_of[(size_t) (old % 1024)] = -1;
        gr.slot_owner[(size_t) best] = -1;
        gr.slot_pend[(size_t) best] = key;
        ++gr.n_pending;
        gr.promo_q.push_back({best, key});
        ++st.promoted;
    }

    // after a window (its GPU work done): the promotions' copies, each member its slice
    bool issue(Group& gr, std::string& err) {
        for (const auto& [s, key] : gr.promo_q) {
            const Layer& l = L[(size_t) (key / 1024)];
            const uint8_t* src = l.host[(size_t) (key % 1024)];
            for (size_t j = 0; j < gr.mem.size(); ++j) {
                TK(cudaSetDevice(M[(size_t) gr.mem[j]].id));
                TK(cudaMemcpyAsync(gr.slot_base[j] + (size_t) s * gr.slot_bytes, src + j * l.blob, l.blob,
                                   cudaMemcpyHostToDevice, gr.cs[j]));
                TK(cudaEventRecord(gr.slot_ev[j][(size_t) s], gr.cs[j]));
            }
            gr.slot_iss[(size_t) s] = fwd;
        }
        gr.promo_q.clear();
        return true;
    }

    CpuLane& lane_for(const Group& gr) {
        lane->set_width(g.ff / (int) gr.mem.size());
        return *lane;
    }


    // ---------------------------------------------------------------- the prompt path (more than kWin tokens)
    // A prompt in chunks of `chunk` tokens.  A member's steps of a chunk are enqueued one at a time as the host
    // publishes each one's input (a worker thread a member spins on the host's counter: no device waits on the host
    // here).  The chunks go through the groups as a pipeline (MM3_TP_PIPE=0: one chunk at a time): a host thread a
    // group runs its layers' host steps for chunk c while the group after it runs them for chunk c - 1, each chunk's
    // residual rows in a buffer of its own (pipe_buf, in turn).  A MoE layer's plan for a chunk: its experts in VRAM by
    // resident slot, those in the cache's slots as they are, (prompt_cpu, opt-in) the cold ones - the fewest tokens,
    // while the CPU lane would finish them before the copies of the rest - on the CPU lane meanwhile, the others copied
    // into slots the decode's promotions do not hold, on each member's copy stream while it runs its experts in VRAM.

    // The FFN half of a chunk's layer runs in slices of ffn_rows() tokens (Options::tp_ffn_rows): the dense FFN's, the
    // shared expert's and the routed experts' scratch a slice's (p_y, p_sh, p_h16 - the gate / up rows in p_y), while
    // the staged experts are still copied once a chunk.  The same bits (a row's sums are its own whatever the rows
    // beside it); slices of 256 at the server's settings: 37 more experts in VRAM, a 53K prompt 13% slower (each slice
    // reads its experts' weights again) - the whole chunk by default.
    int ffn_f = 0;
    int ffn_rows() const { return std::max(1, std::min(ffn_f, chunk)); }
    int n_slices_max() const { return (chunk + ffn_rows() - 1) / ffn_rows(); }
    static constexpr int kMaxSlices = 128;

    // a chunk's MoE plan in mapped memory: a header (the copies, the slices), a header a slice (its passes' sizes,
    // where its sorted rows start), then the arrays - by entry t K + i its row in its slice's p_y and its weight, by
    // sorted row its token, each slice's set bounds, the copies
    struct PHdr {
        int n_copy, n_slices;
    };
    struct PSlice {
        int n_res, max_res, rows_res, n_stg, max_stg, base;
    };
    size_t plan_p_bytes() const {
        const size_t rows = (size_t) chunk * g.k_exp, ns = (size_t) n_slices_max();
        return 64 + kMaxSlices * sizeof(PSlice) + rows * 4 * 3 + ns * ((size_t) g.n_exp + 1) * 4 + ns * 4097 * 4 +
               (size_t) g.n_exp * 8;
    }
    struct PPtr {
        PHdr* hdr;
        PSlice* sl;
        int *rowidx, *invrow, *bres, *bstg, *cslot, *cexp;   // (bres / bstg: slice s's at s (n_exp + 1) / s 4097)
        float* w;
    };
    PPtr pplan(const Group& gr) const {
        const size_t rows = (size_t) chunk * g.k_exp, ns = (size_t) n_slices_max();
        PPtr p;
        uint8_t* b = gr.p_plan;
        p.hdr = (PHdr*) b;
        p.sl = (PSlice*) (b + 64);
        p.rowidx = (int*) (b + 64 + kMaxSlices * sizeof(PSlice));
        p.invrow = p.rowidx + rows;
        p.w = (float*) (p.invrow + rows);
        p.bres = (int*) (p.w + rows);
        p.bstg = p.bres + ns * ((size_t) g.n_exp + 1);
        p.cslot = p.bstg + ns * 4097;
        p.cexp = p.cslot + g.n_exp;
        return p;
    }

    // a prompt's chunks: [a, a + T) of the call, at positions pos0 + a
    struct PChunk {
        int a = 0, T = 0, nblk = 0;
    };
    std::atomic<int> pipe_lim{0};       // the chunks that start (group 0 lowers it at a stop)
    std::atomic<bool> pipe_stop{false};   // (the progress callback asked for a stop)

    // a worker's wait on a mapped counter (the prompt path's steps are enqueued once the host's input is there); false
    // with no error when chunk c is past pipe_lim (the prompt stopped before it)
    bool cpu_wait(const unsigned* f, unsigned v, std::string& err, int c = -1) {
        const double t0 = now();
        for (long spin = 0; (int) (*(const volatile unsigned*) f - v) < 0; ++spin) {
            host::pause();
            if ((spin & 1023) == 1023) {
                std::this_thread::yield();
                if (abort_.load()) {
                    err = "tp: stopped";
                    return false;
                }
                if (c >= 0 && c >= pipe_lim.load()) return false;
                if (now() - t0 > 120.0) {
                    err = "tp: the host gave no input in two minutes";
                    return false;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        return true;
    }

    // member j of group gi's part of a prompt's chunks, one after another: each step enqueued once the host has
    // published its input
    bool prompt_member(int gi, int j, const std::vector<PChunk>& cks, int pos0, unsigned in0, unsigned out0,
                       unsigned rout0, std::string& err) {
        Group& gr = G[(size_t) gi];
        Member& m = M[(size_t) gr.mem[(size_t) j]];
        const int t = (int) gr.mem.size();
        const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
        const int mfs = g.ff / t, K = g.k_exp;
        TK(cudaSetDevice(m.id));
        float* part = gr.p_part + (size_t) j * chunk * g.dim;
        unsigned* fo = gr.fout + (size_t) j * kLine;
        const float scale = 1.0f / std::sqrt((float) g.hd), one = 1.0f, zero = 0.0f;
        unsigned vin = in0, vout = out0, vrout = rout0;
        for (int c = 0; c < (int) cks.size(); ++c) {
            const int T = cks[(size_t) c].T, nblk = cks[(size_t) c].nblk;
            bool first = true;
            for (int li = gr.l0; li < gr.l1; ++li) {
                const Layer& l = L[(size_t) li];
                const Slice& S = l.sl[(size_t) j];
                // attention
                if (!cpu_wait(gr.fin, ++vin, err, c)) return err.empty();
                if (first) k::iota(m.p_pos, T, pos0 + cks[(size_t) c].a, m.s);
                first = false;
                TK(cudaMemcpyAsync(m.p_xn, gr.p_xa, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, m.s));
                k::to_f16(m.p_xn, T, g.dim, g.dim, m.p_xn16, g.dim, m.s);
                x3::gemm_rows(S.q, m.p_xn16, g.dim, T, m.p_q, mh * g.hd, 0.0f, m.s);
                x3::gemm_rows(S.k, m.p_xn16, g.dim, T, m.p_k, mkv * g.hd, 0.0f, m.s);
                x3::gemm_rows(S.v, m.p_xn16, g.dim, T, m.p_v, mkv * g.hd, 0.0f, m.s);
                k::qk_norm_rope(m.p_q, m.p_k, m.p_v, S.qn1, S.kn1, m.p_pos, T, mh, mkv, g.hd, g.n_rot, m.inv, g.eps,
                                S.kc, S.vc, m.s, 0, S.kvq, S.kvh);
                if (cublasSgemm(m.blas, CUBLAS_OP_T, CUBLAS_OP_N, mix * g.ix_dim, T, g.dim, &one, S.wiq, g.dim, m.p_xn,
                                g.dim, &zero, m.p_iq, mix * g.ix_dim) != CUBLAS_STATUS_SUCCESS ||
                    cublasSgemm(m.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.ix_dim, T, g.dim, &one, S.wik, g.dim, m.p_xn,
                                g.dim, &zero, m.p_ik, g.ix_dim) != CUBLAS_STATUS_SUCCESS) {
                    err = "tp: cuBLAS (the indexer)";
                    return false;
                }
                k::index_norm_rope(m.p_iq, m.p_ik, S.iqn1, S.ikn1, m.p_pos, T, mix, g.ix_dim, g.n_rot, m.inv, g.eps,
                                   S.ic, m.s);
                k::block_scores(m.p_iq, S.ic, m.p_pos, T, mix, nblk, g.blk, m.p_bs, m.s);
                k::select_blocks(m.p_bs, m.p_pos, T, mix, nblk, g.blk, g.topk, m.p_kept, m.s);
                k::sparse_attn(m.p_q, S.kc, S.vc, m.p_kept, m.p_pos, T, mh, mkv, g.topk, g.blk, scale, m.p_att, m.s,
                               S.kvq, S.kvh);
                k::to_f16(m.p_att, T, mh * g.hd, mh * g.hd, m.p_att16, mh * g.hd, m.s);
                x3::gemm_rows(S.o, m.p_att16, mh * g.hd, T, m.p_xn, g.dim, 0.0f, m.s);
                k::publish(m.p_xn, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
                // the FFN
                if (!cpu_wait(gr.fin, ++vin, err, c)) return err.empty();
                TK(cudaMemcpyAsync(m.p_xn, gr.p_xb, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, m.s));
                k::to_f16(m.p_xn, T, g.dim, g.dim, m.p_xn16, g.dim, m.s);
                const int F = ffn_rows(), ns = (T + F - 1) / F;
                if (!l.moe) {   // (slice by slice: gate / up in p_y, the output into p_xn's rows)
                    const int wd = g.ff_dense / t;
                    for (int q = 0; q < ns; ++q) {
                        const int t0 = q * F, Tq = std::min(F, T - t0);
                        x3::gemm_rows(S.mg, m.p_xn16 + (size_t) t0 * g.dim, g.dim, Tq, m.p_gg, wd, 0.0f, m.s);
                        x3::gemm_rows(S.mu, m.p_xn16 + (size_t) t0 * g.dim, g.dim, Tq, m.p_uu, wd, 0.0f, m.s);
                        k::swigluoai(m.p_gg, m.p_uu, (size_t) Tq * wd, g.alpha, g.limit, nullptr, m.p_h16, m.s);
                        x3::gemm_rows(S.md, m.p_h16, wd, Tq, m.p_xn + (size_t) t0 * g.dim, g.dim, 0.0f, m.s);
                    }
                    k::publish(m.p_xn, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
                    continue;
                }
                if (j == 0) {   // the router's logits for the host
                    k::rows_dot(S.gate, g.n_exp, g.dim, m.p_xn, T, m.p_rlog, m.s);
                    k::publish(m.p_rlog, gr.p_rlog, (size_t) T * g.n_exp, m.counter, gr.frout, ++vrout, m.s);
                }
                const int ws = g.ff_shared / t;
                // a slice's shared expert (gate / up in p_y, the output p_sh): slice 0's while the host plans
                const auto shared = [&](int q) {
                    const int t0 = q * F, Tq = std::min(F, T - t0);
                    x3::gemm_rows(S.sg, m.p_xn16 + (size_t) t0 * g.dim, g.dim, Tq, m.p_gg, ws, 0.0f, m.s);
                    x3::gemm_rows(S.su, m.p_xn16 + (size_t) t0 * g.dim, g.dim, Tq, m.p_uu, ws, 0.0f, m.s);
                    k::swigluoai(m.p_gg, m.p_uu, (size_t) Tq * ws, g.alpha, g.limit, nullptr, m.p_h16, m.s);
                    x3::gemm_rows(S.sd, m.p_h16, ws, Tq, m.p_sh, g.dim, 0.0f, m.s);
                };
                shared(0);
                // the routed experts: the host's plan; its staged experts' slices into their slots on the copy stream
                // (the slots' last readers, the layers before, are done: the host had their parts before this plan),
                // then slice by slice the experts in VRAM, the staged ones (their copies in by the first), the sum
                if (!cpu_wait(gr.fin, ++vin, err, c)) return err.empty();
                const PPtr pp = pplan(gr);
                const PHdr hd = *pp.hdr;
                const int rows = T * K;
                cudaStream_t cs = gr.cs[(size_t) j];
                for (int q = 0; q < hd.n_copy; ++q) {
                    const int sl = pp.cslot[q], e = pp.cexp[q];
                    TK(cudaMemcpyAsync(gr.slot_base[(size_t) j] + (size_t) sl * gr.slot_bytes,
                                       l.host[(size_t) e] + (size_t) j * l.blob, l.blob, cudaMemcpyHostToDevice, cs));
                }
                if (hd.n_copy > 0) TK(cudaEventRecord(gr.p_ev[(size_t) j], cs));
                TK(cudaMemcpyAsync(m.p_rowidx, pp.rowidx, (size_t) rows * 4, cudaMemcpyHostToDevice, m.s));
                TK(cudaMemcpyAsync(m.p_invrow, pp.invrow, (size_t) rows * 4, cudaMemcpyHostToDevice, m.s));
                TK(cudaMemcpyAsync(m.p_w, pp.w, (size_t) rows * 4, cudaMemcpyHostToDevice, m.s));
                TK(cudaMemcpyAsync(m.p_bres, pp.bres, (size_t) hd.n_slices * (g.n_exp + 1) * 4, cudaMemcpyHostToDevice, m.s));
                TK(cudaMemcpyAsync(m.p_bstg, pp.bstg, (size_t) hd.n_slices * 4097 * 4, cudaMemcpyHostToDevice, m.s));
                x3::SetArgs sa;
                sa.lay = l.lay;
                sa.n_embd = g.dim;
                sa.n_ff = mfs;
                sa.limit = g.limit;
                for (int q = 0; q < hd.n_slices; ++q) {
                    const int t0 = q * F, Tq = std::min(F, T - t0);
                    const PSlice sp = pp.sl[q];
                    if (q > 0) shared(q);
                    if (sp.rows_res > 0) {
                        sa.wbase = l.sl[(size_t) j].res_base;
                        sa.stride = l.blob;
                        sa.n_exp = sp.n_res;
                        sa.bounds = m.p_bres + (size_t) q * (g.n_exp + 1);
                        sa.row0 = 0;
                        sa.max_rows = sp.max_res;
                        x3::set_gate_up(sa, m.p_xn, g.dim, m.p_rowidx + sp.base, m.p_h16, mfs, m.s);
                        x3::set_down(sa, m.p_h16, mfs, m.p_y, g.dim, m.s);
                    }
                    if (q == 0 && hd.n_copy > 0) TK(cudaStreamWaitEvent(m.s, gr.p_ev[(size_t) j], 0));
                    if (sp.max_stg > 0) {
                        sa.wbase = gr.slot_base[(size_t) j];
                        sa.stride = gr.slot_bytes;
                        sa.n_exp = sp.n_stg;
                        sa.bounds = m.p_bstg + (size_t) q * 4097;
                        sa.row0 = sp.rows_res;
                        sa.max_rows = sp.max_stg;
                        x3::set_gate_up(sa, m.p_xn, g.dim, m.p_rowidx + sp.base + sp.rows_res, m.p_h16, mfs, m.s);
                        x3::set_down(sa, m.p_h16, mfs, m.p_y + (size_t) sp.rows_res * g.dim, g.dim, m.s);
                    }
                    // (the CPU lane's entries: weight 0 on p_y's zero row, its last; the host adds their sums)
                    k::moe_combine(nullptr, m.p_y, m.p_invrow + (size_t) t0 * K, m.p_w + (size_t) t0 * K, m.p_sh, Tq,
                                   K, g.dim, g.scale, m.p_xn + (size_t) t0 * g.dim, m.s);
                }
                k::publish(m.p_xn, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
            }
        }
        return true;
    }

    // the prompt's CPU lane for the cold experts (prompt_cpu, MM3_PROMPT_CPU=on or "a,b,bw": its costs - ms a whole
    // expert's pass, its weights decoded for up to 8 tokens; ms a token; a link's GB/s): off by default - with the
    // groups' pipeline the copies hide well enough that a 53K prompt read 4% faster without it
    bool prompt_cpu = false;
    double pc_a = 0.55, pc_b = 0.02, pc_bw = 6.0;
    bool pipe_on = true;            // (MM3_TP_PIPE=0: a chunk through every group before the next)
    std::mutex lane_mu;             // (the pipeline's host threads take the lane one at a time)

    // a chunk's MoE plan for group gr (T tokens, top / w [T][K]): the resident experts by resident slot; the cold ones
    // not in VRAM or the cache onto the CPU lane (cpu: those, the fewest tokens first while the lane - lane_share of it
    // - finishes them before the copies of the rest would); the others staged into the cache's slots (an expert
    // already in one keeps it; a new one takes the least recently used slot that is neither pending nor a promotion's,
    // its copies by the members) - its slots the cache's after.  The CPU lane's entries: weight 0 on the zero row.
    void plan_prompt(Group& gr, Layer& l, int li, int T, const std::vector<int>& top, const std::vector<float>& w,
                     double lane_share, std::vector<int>& cpu, long& staged) {
        const int K = g.k_exp, rows = T * K;
        const PPtr pp = pplan(gr);
        std::vector<std::vector<int>> rows_of((size_t) g.n_exp);
        for (int i = 0; i < rows; ++i) rows_of[(size_t) top[(size_t) i]].push_back(i);
        cpu.clear();
        std::vector<char> on_cpu((size_t) g.n_exp, 0);
        if (prompt_cpu && lane && lane->threads() > 1) {
            std::vector<int> cold;
            for (int e = 0; e < g.n_exp; ++e)
                if (!rows_of[(size_t) e].empty() && l.res_slot[(size_t) e] < 0 && l.slot_of[(size_t) e] < 0)
                    cold.push_back(e);
            std::stable_sort(cold.begin(), cold.end(),
                             [&](int a, int b) { return rows_of[(size_t) a].size() < rows_of[(size_t) b].size(); });
            const double link = (double) l.blob / (pc_bw * 1e9) * 1e3;   // (ms; the members copy theirs in parallel)
            double t_cpu = 0;
            int n_gpu = (int) cold.size();
            for (int e : cold) {
                const int nt = (int) rows_of[(size_t) e].size();
                const double c = (pc_a * ((nt + 7) / 8) + pc_b * nt) / std::max(0.05, lane_share);
                if (t_cpu + c > (n_gpu - 1) * link) break;
                t_cpu += c;
                --n_gpu;
                on_cpu[(size_t) e] = 1;
                cpu.push_back(e);
            }
        }
        PHdr hd{};
        // the staged ones' slots
        std::vector<int> slot_exp((size_t) gr.n_slots, -1);
        for (int e = 0; e < g.n_exp; ++e) {
            if (rows_of[(size_t) e].empty() || l.res_slot[(size_t) e] >= 0 || on_cpu[(size_t) e]) continue;
            const long long key = (long long) li * 1024 + e;
            int sl = l.slot_of[(size_t) e];
            if (sl < 0) {
                for (int pass = 0; pass < 2 && sl < 0; ++pass)   // (a promotion's slot only when no other is free)
                    for (int c = 0; c < gr.n_slots; ++c)
                        if (gr.slot_pend[(size_t) c] < 0 && slot_exp[(size_t) c] < 0 && (pass || !gr.slot_prom[(size_t) c]) &&
                            (sl < 0 || gr.slot_used[(size_t) c] < gr.slot_used[(size_t) sl]))
                            sl = c;
                if (sl < 0) {   // (every slot this layer's or pending: the CPU lane takes it)
                    on_cpu[(size_t) e] = 1;
                    cpu.push_back(e);
                    continue;
                }
                if (gr.slot_prom[(size_t) sl]) {
                    gr.slot_prom[(size_t) sl] = 0;
                    --gr.n_prom;
                }
                const long long old = gr.slot_owner[(size_t) sl];
                if (old >= 0) L[(size_t) (old / 1024)].slot_of[(size_t) (old % 1024)] = -1;
                gr.slot_owner[(size_t) sl] = key;
                l.slot_of[(size_t) e] = sl;
                pp.cslot[hd.n_copy] = sl;
                pp.cexp[hd.n_copy] = e;
                ++hd.n_copy;
                ++staged;
            }
            slot_exp[(size_t) sl] = e;
            gr.slot_used[(size_t) sl] = fwd;
            for (size_t u = 0; u < rows_of[(size_t) e].size(); ++u) score(key, true);
        }
        // the slices (ffn_rows() tokens each): the resident experts' rows by resident slot, then the staged ones' by
        // slot, a slice's rows numbered from 0 (its p_y's) and sorted after the slices before it (base)
        const int F = ffn_rows(), ns = (T + F - 1) / F;
        std::vector<size_t> at((size_t) g.n_exp, 0);   // (each expert's next entry: its entries ascend)
        int base = 0;
        for (int q = 0; q < ns; ++q) {
            const int e1 = std::min(rows, (q + 1) * F * K);
            PSlice sp{};
            sp.n_res = (int) l.res_ids.size();
            sp.n_stg = gr.n_slots;
            sp.base = base;
            int* bres = pp.bres + (size_t) q * (g.n_exp + 1);
            int* bstg = pp.bstg + (size_t) q * 4097;
            int pos = 0;
            const auto take = [&](int e, int& mx) {
                const std::vector<int>& r = rows_of[(size_t) e];
                size_t& i = at[(size_t) e];
                const int p0 = pos;
                for (; i < r.size() && r[i] < e1; ++i) {
                    pp.rowidx[base + pos] = r[i] / K;
                    pp.invrow[r[i]] = pos++;
                }
                mx = std::max(mx, pos - p0);
            };
            for (int s2 = 0; s2 < sp.n_res; ++s2) {
                bres[s2] = pos;
                take(l.res_ids[(size_t) s2], sp.max_res);
            }
            bres[sp.n_res] = pos;
            sp.rows_res = pos;
            for (int c = 0; c < gr.n_slots; ++c) {
                bstg[c] = pos;
                if (slot_exp[(size_t) c] >= 0) take(slot_exp[(size_t) c], sp.max_stg);
            }
            bstg[gr.n_slots] = pos;
            base += pos;
            pp.sl[q] = sp;
        }
        hd.n_slices = ns;
        for (int i = 0; i < rows; ++i) pp.w[i] = w[(size_t) i];
        for (int e : cpu)
            for (int ent : rows_of[(size_t) e]) {
                pp.invrow[ent] = F * K;   // (the slice's p_y's zero row)
                pp.w[ent] = 0.0f;
            }
        *pp.hdr = hd;
    }

    // the cold experts of a chunk's layer on the CPU lane: each as tm sub-experts (the members' slices) for its tokens,
    // in runs of the lane's entries (an expert of more tokens than one run takes in parts), into out [T][dim] (zeroed
    // first); x the normed rows
    void prompt_lane(const Group& gr, const Layer& l, int T, const std::vector<int>& cpu, const std::vector<int>& top,
                     const std::vector<float>& w, const float* x, float* out) {
        const int K = g.k_exp, tm = (int) gr.mem.size();
        std::fill(out, out + (size_t) T * g.dim, 0.0f);
        std::vector<CpuExpert> ex;
        std::vector<std::vector<std::pair<int, float>>> users;
        int ents = 0;
        bool any = false;
        auto run = [&]() {
            if (ex.empty()) return;
            lane->run_multi(ex, users, x, T, out, nullptr, any);
            any = true;
            ex.clear();
            users.clear();
            ents = 0;
        };
        std::vector<std::vector<std::pair<int, float>>> who((size_t) g.n_exp);
        for (int i = 0; i < T * K; ++i) {
            const int e = top[(size_t) i];
            who[(size_t) e].push_back({i / K, w[(size_t) i] * g.scale});
        }
        const int per = std::max(1, lane->max_entries() / tm);   // (an expert's tokens a run at most)
        for (int e : cpu) {
            const auto& all = who[(size_t) e];
            for (size_t u0 = 0; u0 < all.size(); u0 += (size_t) per) {
                const int n = (int) std::min(all.size() - u0, (size_t) per);
                if (ents + n * tm > lane->max_entries()) run();
                for (int j = 0; j < tm; ++j) {
                    const uint8_t* b = l.host[(size_t) e] + (size_t) j * l.blob;
                    CpuExpert c;
                    for (int r = 0; r < 3; ++r) {
                        c.p[r].trellis = (const uint32_t*) (b + l.lay.trellis[r]);
                        c.p[r].suh = (const uint16_t*) (b + l.lay.suh[r]);
                        c.p[r].svh = (const uint16_t*) (b + l.lay.svh[r]);
                        c.p[r].k = r < 2 ? g.dim : g.ff / tm;
                        c.p[r].n = r < 2 ? g.ff / tm : g.dim;
                        c.p[r].k2 = l.lay.k2[r];
                    }
                    c.w = 1.0f;
                    ex.push_back(c);
                    users.emplace_back(all.begin() + (long) u0, all.begin() + (long) (u0 + (size_t) n));
                }
                ents += n * tm;
            }
        }
        run();
    }

    // a stage's tallies (the pipeline's host threads add theirs at the end)
    struct PStat {
        long staged = 0, profiled = 0, cpu_experts = 0;
        double t_lane = 0, t_sync = 0;
    };

    // group gi's host steps for the prompt's chunks: its layers for chunk c (rows pipe_buf[c % NB]) once group gi - 1
    // is done with it (ready[c] == gi); then ready[c] = gi + 1 - the last group's, the chunk done
    bool prompt_stage(int gi, const int* ids, const std::vector<PChunk>& cks, int pos0, int n_call,
                      std::vector<std::atomic<int>>& ready, double lane_share, PStat& ps, std::string& err) {
        Group& gr = G[(size_t) gi];
        const int ng = (int) G.size(), NB = (int) pipe_buf.size();
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        std::vector<int> top, cpu;
        std::vector<float> w;
        std::vector<float>& cpu_out = gr.p_cpu;
        const int R = tap_rows;
        for (int c = 0; c < (int) cks.size(); ++c) {
            const int T = cks[(size_t) c].T, a = cks[(size_t) c].a;
            // the chunk's turn: group gi - 1 done with it; group 0 also waits for its buffer (the chunk NB before done)
            // or, without the pipeline, the chunk before it done
            const double tw = now();
            for (long spin = 0;; ++spin) {
                const bool mine = ready[(size_t) c].load(std::memory_order_acquire) == gi;
                const int dep = gi > 0 ? -1 : pipe_on ? c - NB : c - 1;
                const bool free = dep < 0 || ready[(size_t) dep].load(std::memory_order_acquire) == ng;
                if (mine && free) break;
                if (abort_.load()) {
                    err = "tp: stopped";
                    return false;
                }
                if (gi == 0 && mine && free && pipe_stop.load()) break;
                if (gi > 0 && c >= pipe_lim.load()) return true;   // (group 0 stopped before this chunk)
                host::pause();
                if ((spin & 255) == 255) std::this_thread::yield();
            }
            ps.t_sync += now() - tw;
            if (gi == 0 && pipe_stop.load()) {   // (the progress callback asked for a stop: no chunk starts after these)
                pipe_lim.store(c);
                return true;
            }
            float* H = pipe_buf[(size_t) (c % NB)].data();
            if (gi == 0) {   // the embeddings (and the pictures' rows)
                for (int t = 0; t < T; ++t) {
                    const uint16_t* r = e16 + (size_t) ids[a + t] * g.dim;
                    float* hr = H + (size_t) t * g.dim;
                    for (int i = 0; i < g.dim; ++i) {
                        const uint32_t u = (uint32_t) r[i] << 16;
                        std::memcpy(&hr[i], &u, 4);
                    }
                }
                for (const Model::Image& im : imgs) {
                    const int n = (int) (im.rows.size() / (size_t) g.dim);
                    for (int q = std::max(im.start, pos0 + a); q < std::min(im.start + n, pos0 + a + T); ++q)
                        std::memcpy(H + (size_t) (q - pos0 - a) * g.dim, im.rows.data() + (size_t) (q - im.start) * g.dim,
                                    (size_t) g.dim * 4);
                }
            }
            bool normed = false;
            for (int li = gr.l0; li < gr.l1; ++li) {
                Layer& l = L[(size_t) li];
                if (!normed) norm(H, l.in1.data(), T, gr.p_xa);
                publish_in(gr);
                if (!wait_parts(gr, err)) return false;
                sum_norm(gr, T, nullptr, l.post1.data(), gr.p_xb, gr.p_part, chunk, H);
                publish_in(gr);
                const float* extra = nullptr;
                if (l.moe) {
                    if (!wait_rout(gr, err)) return false;
                    route(l, gr.p_rlog, T, top, w);
                    for (int i = 0; i < T * g.k_exp; ++i) ++prof[(size_t) li * g.n_exp + top[(size_t) i]];
                    ps.profiled += (long) T * g.k_exp;
                    plan_prompt(gr, l, li, T, top, w, lane_share, cpu, ps.staged);
                    publish_in(gr);
                    if (!cpu.empty()) {   // (the cold experts while the GPUs run the rest)
                        const double tl = now();
                        {
                            std::lock_guard<std::mutex> lk(lane_mu);
                            lane_for(gr);
                            prompt_lane(gr, l, T, cpu, top, w, gr.p_xb, cpu_out.data());
                        }
                        ps.t_lane += now() - tl;
                        ps.cpu_experts += (long) cpu.size();
                        extra = cpu_out.data();
                    }
                }
                if (!wait_parts(gr, err)) return false;
                const bool nx = li + 1 < gr.l1;   // (the next layer's input norm with the sum, in this group)
                sum_norm(gr, T, extra, nx ? L[(size_t) li + 1].in1.data() : nullptr, nx ? gr.p_xa : nullptr, gr.p_part,
                         chunk, H);
                normed = nx;
                if (draftm && tap_of[(size_t) li] >= 0 && R > 0) {   // (the draft's taps: the last R positions kept)
                    const int j = tap_of[(size_t) li];
                    for (int t = std::max(0, T - R); t < T; ++t)
                        std::memcpy(tap_ring + ((size_t) j * R + (size_t) ((pos0 + a + t) % R)) * g.dim,
                                    H + (size_t) t * g.dim, (size_t) g.dim * 4);
                }
            }
            if (gi == ng - 1 && ppl_on && !ppl_chunk(ids, n_call, a, T, H, err)) return false;
            ready[(size_t) c].store(gi + 1, std::memory_order_release);
            if (gi == ng - 1 && progress && !progress(a + T)) pipe_stop.store(true);   // (the chunks begun finish)
        }
        return true;
    }

    // (set_ppl: the next-token NLL of every position of a prompt, its chunks one at a time - the head on each chunk's
    //  rows as the last group finishes it, the GPUs idle but the head's)
    bool ppl_on = false;
    std::vector<double> ppl_nll;   // [position]: -log p(ids[position + 1])

    bool ppl_chunk(const int* ids, int n_call, int a, int T, const float* H, std::string& err) {
        Member& hm = M[(size_t) head_m];
        TK(cudaSetDevice(hm.id));
        if ((int) ppl_nll.size() < n_call) ppl_nll.resize((size_t) n_call, 0.0);
        const int cols = std::max(1, x3::mv_cols(head));
        for (int r0 = 0; r0 < T; r0 += kWin) {
            const int rows = std::min(kWin, T - r0);
            rms_rows(H + (size_t) r0 * g.dim, norm1.data(), rows, g.dim, g.eps, xf);
            TK(cudaMemcpyAsync(hm.xn, xf, (size_t) rows * g.dim * 4, cudaMemcpyHostToDevice, hm.s));
            for (int c0 = 0; c0 < rows; c0 += cols) {
                x3::MvJob jb;
                jb.m = &head;
                jb.xf = hm.xn + (size_t) c0 * g.dim;
                jb.y = lg + (size_t) c0 * g.vocab;
                jb.ncol = std::min(cols, rows - c0);
                jb.xf_cs = (size_t) g.dim;
                jb.y_cs = (size_t) g.vocab;
                if (!x3::mv(&jb, 1, hm.s)) {
                    err = "tp: mv head";
                    return false;
                }
            }
            TK(cudaMemcpyAsync(lgh, lg, (size_t) rows * g.vocab * 4, cudaMemcpyDeviceToHost, hm.s));
            TK(cudaStreamSynchronize(hm.s));
            for (int r = 0; r < rows; ++r) {
                const int p = a + r0 + r;
                if (p + 1 >= n_call) continue;
                const float* row = lgh + (size_t) r * g.vocab;
                double mx = row[0], se = 0;
                for (int i = 1; i < g.vocab; ++i) mx = std::max(mx, (double) row[i]);
                for (int i = 0; i < g.vocab; ++i) se += std::exp((double) row[i] - mx);
                ppl_nll[(size_t) p] = mx + std::log(se) - row[ids[p + 1]];
            }
        }
        return true;
    }

    std::vector<std::vector<float>> pipe_buf;   // the chunks' residual rows, a buffer a chunk in flight
    float* tap_ring = nullptr;                  // (mapped) the prompt's taps for the draft: [taps][tap_rows][dim]
    int tap_rows = 0;                           // (the draft's window and its slack: a position at row pos % tap_rows)
    std::function<bool(int)> progress;          // (set_progress)
    int last_done = 0;                          // the last forward's tokens in the cache

    // a prompt (n > kWin tokens from pos0) in chunks through the groups' pipeline; logits (or the candidates) of rows
    // [r0, r1) of the call, which fall in its last chunk
    bool prompt(const int* ids, int n, int pos0, int r0, int r1, float* logits, std::string& err) {
        if (pos0 + n > o.max_ctx) {
            err = "the context is full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        for (const Group& gr : G)   // (a layer's experts not in VRAM all staged at once: slots no promotion holds)
            for (int li = gr.l0; li < gr.l1; ++li)
                if (L[(size_t) li].moe && g.n_exp - (int) L[(size_t) li].res_ids.size() > gr.n_slots) {
                    err = "tp: layer " + std::to_string(li) + " has more experts outside VRAM than its group's cache "
                          "has slots (" + std::to_string(gr.n_slots) + "): a larger --cache-gb";
                    return false;
                }
        for (int t = 0; t < n; ++t)
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
        // the chunks: of `chunk` tokens, the last holding every wanted row
        std::vector<PChunk> cks;
        for (int a = 0; a < n; a += chunk) cks.push_back({a, std::min(chunk, n - a), 0});
        if (r1 > r0) {
            if (r1 - r0 > kWin) {
                err = "tp: more than " + std::to_string(kWin) + " rows of logits from a prompt";
                return false;
            }
            PChunk& lc = cks.back();
            if (r0 < lc.a && cks.size() > 1) {   // (the rows before the last chunk: it takes them from the one before)
                const int mv = lc.a - r0;
                cks[cks.size() - 2].T -= mv;
                lc.a -= mv;
                lc.T += mv;
            }
        }
        for (PChunk& ck : cks) ck.nblk = (pos0 + ck.a + ck.T - 1) / g.blk + 1;
        ++fwd;
        for (Group& gr : G)
            if (!refresh(gr, err)) return false;
        const int nc = (int) cks.size(), ng = (int) G.size();
        pipe_lim.store(nc);
        pipe_stop.store(false);
        in_prompt = true;
        struct Off {
            bool& f;
            ~Off() { f = false; }
        } off{in_prompt};
        for (Member& m : M) m.err.clear();
        for (size_t gi = 0; gi < G.size(); ++gi)
            for (size_t j = 0; j < G[gi].mem.size(); ++j) {
                Member& m = M[(size_t) G[gi].mem[j]];
                const unsigned in0 = G[gi].seq_in, out0 = G[gi].seq_out[j], rout0 = G[gi].seq_rout;
                m.w->submit([this, gi, j, &cks, pos0, in0, out0, rout0, &m] {
                    std::string e;
                    if (!prompt_member((int) gi, (int) j, cks, pos0, in0, out0, rout0, e)) {
                        m.err = e;
                        abort_.store(true);
                    }
                });
            }
        // a host thread a group (this one the last group's)
        std::vector<std::atomic<int>> ready((size_t) nc);
        for (auto& x : ready) x.store(0);
        const bool pipe_was = pipe_on;
        if (ppl_on) pipe_on = false;   // (the head's GPU the last group's alone while it takes a chunk's rows)
        struct Back {
            bool& f;
            bool v;
            ~Back() { f = v; }
        } back{pipe_on, pipe_was};
        const double share = pipe_on ? 1.0 / std::max(1, std::min(nc, ng)) : 1.0;
        std::vector<PStat> ps((size_t) ng);
        std::vector<std::string> errs((size_t) ng);
        std::vector<std::thread> th;
        std::atomic<bool> bad{false};
        for (int gi = 0; gi + 1 < ng; ++gi)
            th.emplace_back([&, gi] {
                if (!prompt_stage(gi, ids, cks, pos0, n, ready, share, ps[(size_t) gi], errs[(size_t) gi])) {
                    bad.store(true);
                    abort_.store(true);
                }
            });
        bool ok = prompt_stage(ng - 1, ids, cks, pos0, n, ready, share, ps[(size_t) ng - 1], errs[(size_t) ng - 1]);
        if (!ok) abort_.store(true);
        for (auto& x : th) x.join();
        ok = ok && !bad.load();
        for (int gi = 0; gi < ng; ++gi) {
            const PStat& s = ps[(size_t) gi];
            st.staged += s.staged;
            st.profiled += s.profiled;
            st.cpu_experts += s.cpu_experts;
            st.t_lane += s.t_lane;
            if (err.empty() && !errs[(size_t) gi].empty() && errs[(size_t) gi] != "tp: stopped") err = errs[(size_t) gi];
        }
        if (!ok || abort_.load()) {
            abort_.store(true);
            release_all();
        }
        for (Member& m : M) m.w->wait();
        if (!ok || abort_.load()) {
            if (err.empty()) err = "tp: " + member_errors();
            return false;
        }
        for (Member& m : M) {
            cudaSetDevice(m.id);
            TK(cudaStreamSynchronize(m.s));
        }
        for (Group& gr : G)
            for (size_t j = 0; j < gr.cs.size(); ++j) {
                cudaSetDevice(M[(size_t) gr.mem[j]].id);
                TK(cudaStreamSynchronize(gr.cs[j]));
            }
        const int e = x3::launch_errors();
        if (e) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        const int lim = std::min(nc, pipe_lim.load());
        last_done = lim > 0 ? cks[(size_t) lim - 1].a + cks[(size_t) lim - 1].T : 0;
        // the head (the GPUs idle now: the host may use them) and the draft's context
        Member& hm = M[(size_t) head_m];
        TK(cudaSetDevice(hm.id));
        if (r1 > r0 && last_done == n) {
            const PChunk& lc = cks.back();
            const float* H = pipe_buf[(size_t) ((nc - 1) % (int) pipe_buf.size())].data();
            const int rows = r1 - r0;
            rms_rows(H + (size_t) (r0 - lc.a) * g.dim, norm1.data(), rows, g.dim, g.eps, xf);
            TK(cudaMemcpyAsync(hm.xn, xf, (size_t) rows * g.dim * 4, cudaMemcpyHostToDevice, hm.s));
            const int cols = std::max(1, x3::mv_cols(head));
            for (int c0 = 0; c0 < rows; c0 += cols) {
                x3::MvJob jb;
                jb.m = &head;
                jb.xf = hm.xn + (size_t) c0 * g.dim;
                jb.y = lg + (size_t) c0 * g.vocab;
                jb.ncol = std::min(cols, rows - c0);
                jb.xf_cs = (size_t) g.dim;
                jb.y_cs = (size_t) g.vocab;
                if (!x3::mv(&jb, 1, hm.s)) {
                    err = "tp: mv head";
                    return false;
                }
            }
            if (cand_on) {
                k::topk_rows(lg, rows, g.vocab, cand_kc, cand_ex, cand_ne, cand_h, hm.s);
                TK(cudaStreamSynchronize(hm.s));
            } else {
                TK(cudaMemcpyAsync(lgh, lg, (size_t) rows * g.vocab * 4, cudaMemcpyDeviceToHost, hm.s));
                TK(cudaStreamSynchronize(hm.s));
                std::memcpy(logits, lgh, (size_t) rows * g.vocab * 4);
            }
        }
        if (draftm && last_done > 0 && tap_rows > 0) {   // (the draft's context: the last tap_rows positions done)
            const int end = pos0 + last_done, s0 = std::max(pos0, end - tap_rows);
            for (int p = s0; p < end;) {   // (in runs that do not wrap the ring)
                const int r = p % tap_rows, nr = std::min(end - p, tap_rows - r);
                if (!draftm->context(tap_ring + (size_t) r * g.dim, nr, p, err, (size_t) tap_rows)) return false;
                p += nr;
            }
        }
        for (Group& gr : G)
            if (!issue(gr, err)) return false;
        return true;
    }

    // ---------------------------------------------------------------- a forward of one window
    // member j of group gi's work for T tokens: every layer of the group, each step waiting on the host's counter
    bool enqueue(int gi, int j, int T, int nblk, unsigned in0, unsigned out0, unsigned rout0, std::string& err) {
        Group& gr = G[(size_t) gi];
        Member& m = M[(size_t) gr.mem[(size_t) j]];
        const int t = (int) gr.mem.size();
        const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
        const int mfs = g.ff / t;
        TK(cudaSetDevice(m.id));
        // (the positions made on the device: a copy would wait behind the promotions' copies, on the copy engine)
        k::iota(m.pos, T, m.pos_h[0], m.s);
        float* part = gr.part + (size_t) j * kWin * g.dim;
        unsigned* fo = gr.fout + (size_t) j * kLine;
        const auto job = [&](const x3::Mat& w, const float* x, int ldx, float* y, int ldy) {
            x3::MvJob jb;
            jb.m = &w;
            jb.xf = x;
            jb.y = y;
            jb.ncol = T;
            jb.xf_cs = (size_t) ldx;
            jb.y_cs = (size_t) ldy;
            return jb;
        };
        const float scale = 1.0f / std::sqrt((float) g.hd);
        unsigned vin = in0, vout = out0, vrout = rout0;
        for (int li = gr.l0; li < gr.l1; ++li) {
            const Layer& l = L[(size_t) li];
            const Slice& S = l.sl[(size_t) j];
            // attention
            if (!pull_in(m, gr.fin, ++vin, m.xn, gr.xa, (size_t) T * g.dim, err)) return false;
            const x3::MvJob qkv[3] = {job(S.q, m.xn, g.dim, m.q, mh * g.hd), job(S.k, m.xn, g.dim, m.k, mkv * g.hd),
                                      job(S.v, m.xn, g.dim, m.v, mkv * g.hd)};
            if (!x3::mv(qkv, 3, m.s)) {
                err = "tp: mv q/k/v";
                return false;
            }
            k::qk_norm_rope(m.q, m.k, m.v, S.qn1, S.kn1, m.pos, T, mh, mkv, g.hd, g.n_rot, m.inv, g.eps, S.kc, S.vc, m.s, 0,
                            S.kvq, S.kvh);
            k::rows_dot(S.wiq, mix * g.ix_dim, g.dim, m.xn, T, m.iq, m.s);
            k::rows_dot(S.wik, g.ix_dim, g.dim, m.xn, T, m.ik, m.s);
            k::index_norm_rope(m.iq, m.ik, S.iqn1, S.ikn1, m.pos, T, mix, g.ix_dim, g.n_rot, m.inv, g.eps, S.ic, m.s);
            k::block_scores(m.iq, S.ic, m.pos, T, mix, nblk, g.blk, m.bs, m.s);
            k::select_blocks(m.bs, m.pos, T, mix, nblk, g.blk, g.topk, m.kept, m.s);
            k::sparse_attn(m.q, S.kc, S.vc, m.kept, m.pos, T, mh, mkv, g.topk, g.blk, scale, m.att, m.s, S.kvq, S.kvh,
                           attn_splits, m.attp);
            const x3::MvJob oj = job(S.o, m.att, mh * g.hd, m.out, g.dim);
            if (!x3::mv(&oj, 1, m.s)) {
                err = "tp: mv o";
                return false;
            }
            k::publish(m.out, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
            // the FFN: every member its slice of the shared expert while the host routes (from the normed rows it
            // published: host_router; else member 0 the router's logits first), then the plans
            if (!pull_in(m, gr.fin, ++vin, m.xn, gr.xb, (size_t) T * g.dim, err)) return false;
            if (l.moe) {
                if (j == 0 && !host_router) {
                    k::rows_dot(S.gate, g.n_exp, g.dim, m.xn, T, m.rlog_d, m.s);
                    k::publish(m.rlog_d, gr.rlog, (size_t) T * g.n_exp, m.counter, gr.frout, ++vrout, m.s);
                }
                const int ws = g.ff_shared / t;
                const x3::MvJob gu[2] = {job(S.sg, m.xn, g.dim, m.gg, ws), job(S.su, m.xn, g.dim, m.uu, ws)};
                if (!x3::mv(gu, 2, m.s)) {
                    err = "tp: mv shared gate / up";
                    return false;
                }
                k::swigluoai(m.gg, m.uu, (size_t) T * ws, g.alpha, g.limit, m.gg, nullptr, m.s);
                const x3::MvJob sd = job(S.sd, m.gg, ws, m.sh, g.dim);
                if (!x3::mv(&sd, 1, m.s)) {
                    err = "tp: mv shared down";
                    return false;
                }
                if (!pull_in(m, gr.fin, ++vin, (float*) m.plan_d, (const float*) (gr.plans + (size_t) j * plan_bytes),
                             plan_bytes / 4, err))
                    return false;
                const size_t off_w = (size_t) kWin * 8 * 8, off_gp = off_w + (size_t) kWin * 8 * 4,
                             off_gs = off_gp + (size_t) kWin * 8 * 8, off_ge = off_gs + ((size_t) kWin * 8 + 1) * 4,
                             off_ng = off_ge + (size_t) kWin * 8 * 4;
                x3::WinArgs a;
                a.grp_ptr = (const unsigned long long*) (m.plan_d + off_gp);
                a.grp_start = (const int*) (m.plan_d + off_gs);
                a.grp_ent = (const int*) (m.plan_d + off_ge);
                a.n_groups = (const int*) (m.plan_d + off_ng);
                a.plan_ptr = (const unsigned long long*) m.plan_d;
                a.plan_w = (const float*) (m.plan_d + off_w);
                a.x = m.xn;
                a.x_stride = (size_t) g.dim;
                a.h = m.win_h;
                a.part = m.win_part;
                a.sh_out = m.sh;
                a.out = m.out;
                a.out_stride = (size_t) g.dim;
                a.lay = l.lay;
                a.T = T;
                a.k = g.k_exp;
                a.n_embd = g.dim;
                a.n_ff = mfs;
                a.limit = g.limit;
                if (!x3::win_experts(a, m.s)) {
                    err = "tp: win_experts";
                    return false;
                }
            } else {
                const int wd = g.ff_dense / t;
                const x3::MvJob gu[2] = {job(S.mg, m.xn, g.dim, m.gg, wd), job(S.mu, m.xn, g.dim, m.uu, wd)};
                if (!x3::mv(gu, 2, m.s)) {
                    err = "tp: mv dense gate / up";
                    return false;
                }
                k::swigluoai(m.gg, m.uu, (size_t) T * wd, g.alpha, g.limit, m.gg, nullptr, m.s);
                const x3::MvJob dj = job(S.md, m.gg, wd, m.out, g.dim);
                if (!x3::mv(&dj, 1, m.s)) {
                    err = "tp: mv dense down";
                    return false;
                }
            }
            k::publish(m.out, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
        }
        return true;
    }

    // the draft's context steps for a window, on the head member's stream after its layers: tap j once the host has
    // written it (ftap)
    bool enqueue_taps(int T, unsigned v0, std::string& err) {
        Member& m = M[(size_t) head_m];
        TK(cudaSetDevice(m.id));
        const int nt = (int) draftm->geo().taps.size();
        for (int j = 0; j < nt; ++j) {
            float* ds = taps_d + (size_t) j * kWin * g.dim;
            k::wait_flag(ftap, v0 + (unsigned) j + 1, m.s);
            k::pull(ds, taps_h + (size_t) j * kWin * g.dim, (size_t) T * g.dim, m.s);   // (not a copy: see pull)
            if (!draftm->context_tap(j, ds, T, err)) return false;
        }
        return true;
    }

    // the head on its member: the host's normed rows -> logits (pinned host), head_ev after them
    bool enqueue_head(int rows, unsigned v, std::string& err) {
        Member& m = M[(size_t) head_m];
        TK(cudaSetDevice(m.id));
        float* x = m.xn;   // (the member's own steps are done by then: the host publishes the head's last)
        if (!pull_in(m, fhead, v, x, xf, (size_t) rows * g.dim, err)) return false;
        const int cols = std::max(1, x3::mv_cols(head));
        for (int c0 = 0; c0 < rows; c0 += cols) {
            x3::MvJob j;
            j.m = &head;
            j.xf = x + (size_t) c0 * g.dim;
            j.y = lg + (size_t) c0 * g.vocab;
            j.ncol = std::min(cols, rows - c0);
            j.xf_cs = (size_t) g.dim;
            j.y_cs = (size_t) g.vocab;
            if (!x3::mv(&j, 1, m.s)) {
                err = "tp: mv head";
                return false;
            }
        }
        if (cand_on) k::topk_rows(lg, rows, g.vocab, cand_kc, cand_ex, cand_ne, cand_h, m.s);
        else TK(cudaMemcpyAsync(lgh, lg, (size_t) rows * g.vocab * 4, cudaMemcpyDeviceToHost, m.s));
        TK(cudaEventRecord(head_ev, m.s));
        return true;
    }

    // (the knobs, MM3_TP_OLD=wait,split: the paths before, to time them against; MM3_TP_HOST_ROUTER=1: the router's
    //  logits on the host from the normed rows - one round trip less a layer, but its rows read from memory every
    //  window, ~6 ms of a window's host time)
    const std::string tp_old = std::getenv("MM3_TP_OLD") ? std::getenv("MM3_TP_OLD") : "";
    const bool old_wait = tp_old.find("wait") != std::string::npos;
    const bool host_router = std::getenv("MM3_TP_HOST_ROUTER") != nullptr;
    int attn_splits = tp_old.find("split") != std::string::npos ? 1 : 16;   // (load: at most topk)

    // a member's wait for the host's counter f to reach v, then n floats of the host's (mapped) into dst: one kernel
    // (wait_pull); old_wait: wait_flag, then the copy engine
    bool pull_in(Member& m, const unsigned* f, unsigned v, float* dst, const float* src, size_t n, std::string& err) {
        if (!old_wait) {
            k::wait_pull(f, v, dst, src, n, m.s);
            return true;
        }
        k::wait_flag(f, v, m.s);
        TK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyHostToDevice, m.s));
        return true;
    }

    // the host waits for the members' counters of group gr to pass their next value
    bool wait_parts(Group& gr, std::string& err) {
        const double t0 = now();
        for (size_t j = 0; j < gr.mem.size(); ++j) {
            const unsigned want = gr.seq_out[j] + 1;
            const volatile unsigned* f = gr.fout + j * kLine;
            for (long spin = 0; (int) (*f - want) < 0; ++spin) {
                host::pause();
                if ((spin & 0xfffff) == 0xfffff) {
                    if (abort_.load()) {
                        err = "tp: a member's work failed: " + member_errors();
                        return false;
                    }
                    if (now() - t0 > 60.0) {
                        err = "tp: CUDA" + std::to_string(M[(size_t) gr.mem[j]].id) + " gave no answer in a minute";
                        return false;
                    }
                }
            }
            gr.seq_out[j] = want;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!in_prompt) st.t_sync += now() - t0;
        return true;
    }

    bool wait_rout(Group& gr, std::string& err) {
        const double t0 = now();
        const unsigned want = gr.seq_rout + 1;
        const volatile unsigned* f = gr.frout;
        for (long spin = 0; (int) (*f - want) < 0; ++spin) {
            host::pause();
            if ((spin & 0xfffff) == 0xfffff) {
                if (abort_.load()) {
                    err = "tp: a member's work failed: " + member_errors();
                    return false;
                }
                if (now() - t0 > 60.0) {
                    err = "tp: the router's logits did not come in a minute";
                    return false;
                }
            }
        }
        gr.seq_rout = want;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!in_prompt) st.t_sync += now() - t0;
        return true;
    }

    std::string member_errors() const {
        std::string s;
        for (const Member& m : M)
            if (!m.err.empty()) s += "CUDA" + std::to_string(m.id) + ": " + m.err + "; ";
        return s;
    }

    void publish_in(Group& gr) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *(volatile unsigned*) gr.fin = ++gr.seq_in;
    }

    // y [rows][dim] = RMSNorm(x) . w, a row a job on the CPU lane's threads
    void norm(const float* x, const float* w, int rows, float* y) {
        const std::function<void(int)> f = [&](int r) {
            rms_rows(x + (size_t) r * g.dim, w, 1, g.dim, g.eps, y + (size_t) r * g.dim);
        };
        if (lane && rows > 1) {
            std::lock_guard<std::mutex> lk(lane_mu);
            lane->parallel_for(rows, f);
        } else {
            rms_rows(x, w, rows, g.dim, g.eps, y);
        }
    }

    // h [T][dim] += the members' partials (member order) + extra, then (w given) y = RMSNorm(h) . w: a row a job, the
    // same bits as add_parts then norm
    void sum_norm(const Group& gr, int T, const float* extra, const float* w, float* y, const float* base = nullptr,
                  int stride = kWin, float* hh = nullptr) {
        const int t = (int) gr.mem.size();
        if (!base) base = gr.part;
        if (!hh) hh = h.data();
        const std::function<void(int)> f = [&](int r) {
            const float* pp[8];
            for (int j = 0; j < t; ++j) pp[j] = base + ((size_t) j * stride + r) * g.dim;
            float* hr = hh + (size_t) r * g.dim;
            sum_into(hr, pp, t, extra ? extra + (size_t) r * g.dim : nullptr, (size_t) g.dim);
            if (w) rms_rows(hr, w, 1, g.dim, g.eps, y + (size_t) r * g.dim);
        };
        if (lane && T > 1) {
            std::lock_guard<std::mutex> lk(lane_mu);
            lane->parallel_for(T, f);
        } else {
            for (int r = 0; r < T; ++r) f(r);
        }
    }

    // rl [T][E] = x [T][dim] . the router's F32 rows (dot's fixed order: a token's logits the same bits whatever T),
    // eight experts a job on the lane
    void router_logits(const Layer& l, const float* x, int T, float* rl) {
        const int E = g.n_exp, per = 8;
        const std::function<void(int)> f = [&](int j) {
            for (int e = j * per; e < std::min(E, (j + 1) * per); ++e)
                for (int r = 0; r < T; ++r)
                    rl[(size_t) r * E + e] = dot(l.gate.data() + (size_t) e * g.dim, x + (size_t) r * g.dim, g.dim);
        };
        const int nj = (E + per - 1) / per;
        if (lane) {
            std::lock_guard<std::mutex> lk(lane_mu);
            lane->parallel_for(nj, f);
        } else {
            for (int j = 0; j < nj; ++j) f(j);
        }
    }

    // h [T][dim] += the members' partials (member order) + extra
    void add_parts(const Group& gr, int T, const float* extra, const float* base = nullptr, int stride = kWin) {
        const float* p[8];
        const int t = (int) gr.mem.size();
        if (!base) base = gr.part;
        for (int j = 0; j < t; ++j) p[j] = base + (size_t) j * stride * g.dim;
        const size_t n = (size_t) T * g.dim, piece = 3072;   // (pieces of the rows, on the CPU lane's threads)
        const int np = (int) ((n + piece - 1) / piece);
        const std::function<void(int)> f = [&](int q) {
            const size_t a = (size_t) q * piece, b = std::min(n, a + piece);
            const float* pp[8];
            for (int j = 0; j < t; ++j) pp[j] = p[j] + a;
            sum_into(h.data() + a, pp, t, extra ? extra + a : nullptr, b - a);
        };
        if (lane && np > 1) lane->parallel_for(np, f);
        else f(0), (void) 0;
        if (!lane || np <= 1) return;
    }

    // routing: member 0's router logits (rlog [T][E], published) -> top [T][K], w [T][K]
    void route(const Layer& l, const float* rlog, int T, std::vector<int>& top, std::vector<float>& w) {
        const int E = g.n_exp, K = g.k_exp;
        top.assign((size_t) T * K, 0);
        w.assign((size_t) T * K, 0.0f);
        std::vector<float> sg((size_t) E), ch((size_t) E);
        std::vector<int> idx((size_t) E);
        for (int t = 0; t < T; ++t) {
            for (int e = 0; e < E; ++e) {
                sg[(size_t) e] = 1.0f / (1.0f + std::exp(-rlog[(size_t) t * E + e]));
                ch[(size_t) e] = sg[(size_t) e] + l.bias[(size_t) e];
                idx[(size_t) e] = e;
            }
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
                return ch[(size_t) a] > ch[(size_t) b] || (ch[(size_t) a] == ch[(size_t) b] && a < b);
            });
            float sum = 0.0f;
            for (int r = 0; r < K; ++r) {
                top[(size_t) t * K + r] = idx[(size_t) r];
                sum += sg[(size_t) idx[(size_t) r]];
            }
            for (int r = 0; r < K; ++r) w[(size_t) t * K + r] = sg[(size_t) idx[(size_t) r]] / sum;
        }
    }

    const bool trace = ::mm3::env("MM3_TP_TRACE") != nullptr;
    void tr(const char* what, int a = -1) {
        if (trace) std::fprintf(stderr, "[tp %.3f] %s %d\n", now(), what, a);
    }

    bool window(const int* ids, int T, int pos0, int r0, int r1, float* logits, std::string& err) {
        const int nblk = (pos0 + T - 1) / g.blk + 1;
        const double tw0 = now();
        ++fwd;
        for (Group& gr : G)
            if (!refresh(gr, err)) return false;
        if (pos0 + T > o.max_ctx) {
            err = "the context is full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        // the embeddings (and the pictures' rows)
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
            const uint16_t* r = e16 + (size_t) ids[t] * g.dim;
            for (int i = 0; i < g.dim; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&h[(size_t) t * g.dim + i], &u, 4);
            }
        }
        for (const Model::Image& im : imgs) {
            const int n = (int) (im.rows.size() / (size_t) g.dim);
            for (int q = std::max(im.start, pos0); q < std::min(im.start + n, pos0 + T); ++q)
                std::memcpy(&h[(size_t) (q - pos0) * g.dim], im.rows.data() + (size_t) (q - im.start) * g.dim,
                            (size_t) g.dim * 4);
        }
        // the members' work, enqueued meanwhile (a thread a GPU)
        for (Member& m : M) {
            for (int t = 0; t < T; ++t) m.pos_h[t] = pos0 + t;
            m.err.clear();
        }
        const bool want_head = r1 > r0;
        for (size_t gi = 0; gi < G.size(); ++gi)
            for (size_t j = 0; j < G[gi].mem.size(); ++j) {
                Member& m = M[(size_t) G[gi].mem[j]];
                const unsigned in0 = G[gi].seq_in, out0 = G[gi].seq_out[j], rout0 = G[gi].seq_rout;
                const bool head = want_head && G[gi].mem[j] == head_m;
                const unsigned hv = seq_head + 1;
                const int rows = r1 - r0;
                const bool taps = draftm && G[gi].mem[j] == head_m;
                const unsigned tv = seq_tap;
                m.w->submit([this, gi, j, T, nblk, in0, out0, rout0, head, hv, rows, taps, tv, &m] {
                    std::string e;
                    tr("worker start", (int) G[gi].mem[j]);
                    if (!enqueue((int) gi, (int) j, T, nblk, in0, out0, rout0, e) || (taps && !enqueue_taps(T, tv, e)) ||
                        (head && !enqueue_head(rows, hv, e))) {
                        m.err = e;
                        abort_.store(true);
                    }
                    tr("worker done", (int) G[gi].mem[j]);
                });
            }
        bool head_alone = want_head;   // (the head's member in no group: enqueue it apart)
        for (const Group& gr : G)
            for (int mi : gr.mem)
                if (mi == head_m) head_alone = false;
        if (head_alone && !enqueue_head(r1 - r0, seq_head + 1, err)) return false;
        // the host's side: layer by layer
        double tq = now();
        tp_start += tq - tw0;
        ++tp_windows;
        auto lap = [&](double& acc) {
            const double t2 = now();
            acc += t2 - tq;
            tq = t2;
        };
        bool ok = true;
        std::vector<int> top;
        std::vector<float> w;
        bool normed = false;   // (the layer's input norm done already: the sum before it took it)
        for (int li = 0; li < g.n_layers && ok; ++li) {
            Layer& l = L[(size_t) li];
            Group& gr = G[(size_t) l.group];
            tr("layer", li);
            if (!normed) norm(h.data(), l.in1.data(), T, gr.xa);
            lap(tp_norm);
            publish_in(gr);
            if (!wait_parts(gr, err)) {
                ok = false;
                break;
            }
            lap(tp_wait);
            sum_norm(gr, T, nullptr, l.post1.data(), gr.xb);   // (h += the attention's parts, xb its norm)
            lap(tp_sum);
            const float* extra = nullptr;
            if (l.moe) {
                CpuLane& ln = lane_for(gr);
                publish_in(gr);   // (xb: the shared expert starts; the host routes meanwhile)
                if (host_router) {
                    router_logits(l, gr.xb, T, rl.data());
                    route(l, rl.data(), T, top, w);
                } else {
                    if (!wait_rout(gr, err)) {
                        ok = false;
                        break;
                    }
                    route(l, gr.rlog, T, top, w);
                }
                lap(tp_route);
                const int K = g.k_exp, tm = (int) gr.mem.size();
                for (int i = 0; i < T * K; ++i) ++prof[(size_t) li * g.n_exp + top[(size_t) i]];
                st.profiled += (long) T * K;
                // the plans: each member's slices of the experts in VRAM (0: the CPU lane's), grouped by blob
                for (int j = 0; j < tm; ++j) {
                    uint8_t* base = gr.plans + (size_t) j * plan_bytes;
                    unsigned long long* pp = (unsigned long long*) base;
                    float* pw = (float*) (pp + (size_t) kWin * 8);
                    unsigned long long* gp = (unsigned long long*) (pw + (size_t) kWin * 8);
                    int* gs = (int*) (gp + (size_t) kWin * 8);
                    int* ge = gs + kWin * 8 + 1;
                    int* ng = ge + kWin * 8;
                    std::vector<unsigned long long> by_ptr;
                    std::vector<std::vector<int>> ents;
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < 8; ++i) {
                            const int e = t * 8 + i;
                            pp[e] = 0;
                            pw[e] = 0.0f;
                            if (i >= K) continue;
                            const int x = top[(size_t) t * K + i];
                            pw[e] = w[(size_t) t * K + i] * g.scale;
                            const int s = l.res_slot[(size_t) x];
                            if (s >= 0) {
                                pp[e] = (unsigned long long) (l.sl[(size_t) j].res_base + (size_t) s * l.blob);
                            } else if (l.slot_of[(size_t) x] >= 0) {
                                const int c = l.slot_of[(size_t) x];
                                pp[e] = (unsigned long long) (gr.slot_base[(size_t) j] + (size_t) c * gr.slot_bytes);
                                if (j == 0) {
                                    gr.slot_used[(size_t) c] = fwd;
                                    score((long long) li * 1024 + x, true);   // (a cached expert's use keeps it)
                                }
                            } else {
                                continue;
                            }
                            size_t gi2 = 0;
                            while (gi2 < by_ptr.size() && by_ptr[gi2] != pp[e]) ++gi2;
                            if (gi2 == by_ptr.size()) {
                                by_ptr.push_back(pp[e]);
                                ents.emplace_back();
                            }
                            ents[gi2].push_back(e);
                        }
                    int ne = 0;
                    for (size_t q = 0; q < by_ptr.size(); ++q) {
                        gp[q] = by_ptr[q];
                        gs[q] = ne;
                        for (int e : ents[q]) ge[ne++] = e;
                    }
                    gs[by_ptr.size()] = ne;
                    *ng = (int) by_ptr.size();
                }
                // the CPU lane's experts: each as tm sub-experts (the members' slices), summed token by token in slot
                // order, sub-expert by sub-expert
                std::vector<CpuExpert> cpu;
                std::vector<std::vector<std::pair<int, float>>> users;
                std::vector<std::pair<int, int>> order;
                std::vector<int> cpu_of((size_t) g.n_exp, -1);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < K; ++i) {
                        const int x = top[(size_t) t * K + i];
                        if (l.res_slot[(size_t) x] >= 0 || l.slot_of[(size_t) x] >= 0) continue;
                        const float wt = w[(size_t) t * K + i] * g.scale;
                        if (cpu_of[(size_t) x] < 0) {
                            consider(gr, li, x);
                            cpu_of[(size_t) x] = (int) cpu.size();
                            for (int j = 0; j < tm; ++j) {
                                const uint8_t* b = l.host[(size_t) x] + (size_t) j * l.blob;
                                CpuExpert c;
                                for (int r = 0; r < 3; ++r) {
                                    c.p[r].trellis = (const uint32_t*) (b + l.lay.trellis[r]);
                                    c.p[r].suh = (const uint16_t*) (b + l.lay.suh[r]);
                                    c.p[r].svh = (const uint16_t*) (b + l.lay.svh[r]);
                                    c.p[r].k = r < 2 ? g.dim : g.ff / tm;
                                    c.p[r].n = r < 2 ? g.ff / tm : g.dim;
                                    c.p[r].k2 = l.lay.k2[r];
                                }
                                c.w = 1.0f;
                                cpu.push_back(c);
                                users.emplace_back();
                            }
                        }
                        for (int j = 0; j < tm; ++j) {
                            const int ci = cpu_of[(size_t) x] + j;
                            users[(size_t) ci].push_back({t, wt});
                            order.push_back({ci, (int) users[(size_t) ci].size() - 1});
                        }
                    }
                publish_in(gr);
                lap(tp_plan);
                if (!cpu.empty()) {
                    const double tl = now();
                    ln.run_multi(cpu, users, gr.xb, T, cpu_out.data(), &order);
                    st.t_lane += now() - tl;
                    st.cpu_experts += (long) cpu.size() / tm;
                    extra = cpu_out.data();
                }
                lap(tp_lane);
            } else {
                publish_in(gr);
            }
            if (!wait_parts(gr, err)) {
                ok = false;
                break;
            }
            lap(tp_wait);
            // (h += the FFN's parts and the CPU lane's, the next layer's input norm with it)
            const Layer* nx = li + 1 < g.n_layers ? &L[(size_t) li + 1] : nullptr;
            sum_norm(gr, T, extra, nx ? nx->in1.data() : nullptr, nx ? G[(size_t) nx->group].xa : nullptr);
            normed = nx != nullptr;
            lap(tp_sum);
            if (draftm && tap_of[(size_t) li] >= 0) {   // (the draft's context takes the tap now: its steps wait on ftap)
                const int j = tap_of[(size_t) li];
                std::memcpy(taps_h + (size_t) j * kWin * g.dim, h.data(), (size_t) T * g.dim * 4);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *(volatile unsigned*) ftap = ++seq_tap;
            }
        }
        if (ok && want_head) {
            rms_rows(h.data() + (size_t) r0 * g.dim, norm1.data(), r1 - r0, g.dim, g.eps, xf);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) fhead = ++seq_head;
        }
        if (!ok || abort_.load()) {   // (the devices released first: a worker may be waiting on them)
            abort_.store(true);
            release_all();
        }
        for (Member& m : M) m.w->wait();
        if (!ok || abort_.load()) {
            if (err.empty()) err = "tp: " + member_errors();
            abort_.store(true);
            release_all();
            return false;
        }
        lap(tp_end);
        if (want_head) {
            cudaSetDevice(M[(size_t) head_m].id);
            TK(cudaEventSynchronize(head_ev));
            if (!cand_on) std::memcpy(logits, lgh, (size_t) (r1 - r0) * g.vocab * 4);
        }
        lap(tp_head);
        for (Member& m : M) {   // (every member done: the next forward may use the buffers)
            cudaSetDevice(m.id);
            TK(cudaStreamSynchronize(m.s));
        }
        const int e = x3::launch_errors();
        if (e) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        for (Group& gr : G)
            if (!issue(gr, err)) return false;
        if (draftm && !draftm->context_finish(T, pos0, err)) return false;
        lap(tp_draft);
        return true;
    }

    bool forward(const int* ids, int n, int pos0, int want, float* logits, std::string& err) {
        if (abort_.load()) {
            err = "tp: the engine stopped after an error";
            return false;
        }
        const double tc = now();
        want = std::max(0, std::min(want, n));
        const bool any = (logits != nullptr || cand_on) && want > 0;
        last_done = 0;
        if (n > kWin) {   // (a prompt: its chunks through the groups' pipeline)
            if (!prompt(ids, n, pos0, any ? n - want : 0, any ? n : 0, logits, err)) return false;
        } else {
            if (!window(ids, n, pos0, any ? n - want : 0, any ? n : 0, logits, err)) return false;
            last_done = n;
        }
        if (n == 1) st.t_token += now() - tc;
        return true;
    }

    // the last `want` rows as candidates: picked on the head's GPU when they all come from the call's last piece (a
    // window, or a prompt's last chunk) - else, and for a row with more ties than its room, from the logits on the host
    bool forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err) {
        want = std::max(0, std::min(want, n));
        c.top.assign((size_t) want, {});
        c.ex.assign((size_t) want, {});
        if (want > kWin || (int) c.extra.size() > k::kCandExtra) {
            std::vector<float> lgv((size_t) want * g.vocab);
            if (!forward(ids, n, pos0, want, lgv.data(), err)) return false;
            for (int r = 0; r < want; ++r)
                host_cands(lgv.data() + (size_t) r * g.vocab, g.vocab, c.kc, c.extra, c.top[(size_t) r], c.ex[(size_t) r]);
            return true;
        }
        cand_kc = std::max(1, c.kc);
        cand_ne = (int) c.extra.size();
        for (int i = 0; i < cand_ne; ++i) cand_ex[i] = c.extra[(size_t) i];
        cand_on = true;
        const bool ok = forward(ids, n, pos0, want, nullptr, err);
        cand_on = false;
        if (!ok) return false;
        std::vector<float> row;
        for (int r = 0; r < want; ++r) {
            const k::CandRow& cr = cand_h[r];
            if (cr.n < 0) {   // (more ties than room: the row itself)
                row.resize((size_t) g.vocab);
                cudaSetDevice(M[(size_t) head_m].id);
                TK(cudaMemcpy(row.data(), lg + (size_t) r * g.vocab, (size_t) g.vocab * 4, cudaMemcpyDeviceToHost));
                host_cands(row.data(), g.vocab, c.kc, c.extra, c.top[(size_t) r], c.ex[(size_t) r]);
                continue;
            }
            auto& t = c.top[(size_t) r];
            t.resize((size_t) cr.n);
            for (int i = 0; i < cr.n; ++i) t[(size_t) i] = {cr.id[i], cr.v[i]};
            c.ex[(size_t) r].assign(cr.ex, cr.ex + cand_ne);
        }
        return true;
    }
};

TpEngine::TpEngine() : p(new Impl) {}
TpEngine::~TpEngine() = default;
bool TpEngine::load(const Options& o, std::string& err) { return p->load(o, err); }
bool TpEngine::forward(const int* ids, int n, int pos0, int want, float* logits, std::string& err) {
    return p->forward(ids, n, pos0, want, logits, err);
}
int TpEngine::vocab() const { return p->g.vocab; }
int TpEngine::max_ctx() const { return p->o.max_ctx; }
int TpEngine::bos() const { return p->g.bos; }
int TpEngine::eos() const { return p->g.eos; }
const Stats& TpEngine::stats() const { return p->st; }
std::vector<std::string> TpEngine::info_lines() const { return p->info; }
std::vector<std::string> TpEngine::vram_lines() const { return p->vram; }
bool TpEngine::save_profile(const std::string& path, std::string& err) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p->prof.data(), 4, p->prof.size(), f) != p->prof.size()) {
        if (f) std::fclose(f);
        err = path + ": cannot write";
        return false;
    }
    std::fclose(f);
    return true;
}
int TpEngine::draft_block() const { return p->draftm ? p->draftm->geo().block : 0; }
bool TpEngine::draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err) {
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    cudaSetDevice(p->M[(size_t) p->head_m].id);
    return p->draftm->propose(anchor, pos, n, drafts, conf, nullptr, err);
}
void TpEngine::set_progress(std::function<bool(int)> f) { p->progress = std::move(f); }
void TpEngine::set_ppl(bool on) {
    p->ppl_on = on;
    if (on) p->ppl_nll.clear();
}
const std::vector<double>& TpEngine::ppl_nll() const { return p->ppl_nll; }
int TpEngine::done() const { return p->last_done; }
bool TpEngine::forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err) {
    return p->forward_cands(ids, n, pos0, want, c, err);
}
bool TpEngine::draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                             std::vector<int>& drafts, std::vector<float>& conf,
                             std::vector<std::vector<std::pair<int, float>>>& q, std::string& err) {
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    k::SampleArgs sa;
    sa.temperature = ds.temperature;
    sa.top_p = ds.top_p;
    sa.min_p = ds.min_p;
    sa.top_k = ds.top_k;
    cudaSetDevice(p->M[(size_t) p->head_m].id);
    return p->draftm->propose_sampled(anchor, pos, n, sa, u, drafts, conf, q, err);
}
const VisionGeo* TpEngine::vision() const { return p->vis ? &p->vis->geo() : nullptr; }
bool TpEngine::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err) {
    if (!p->vis) {
        err = "no vision tower loaded (--vision on)";
        return false;
    }
    return p->vis->encode(rgb, width, height, rows, err);
}
void TpEngine::set_images(std::vector<Model::Image> images) { p->imgs = std::move(images); }

}  // namespace mm3
