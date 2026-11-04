// tests/ds4_multi_attn_test.cu - DS4_MULTI_ATTN (maya-ds4 --parallel): a multi step's attention for all its rows in one
// launch a step (attn_multi's launches) against attn_core's, a conversation's run of rows at a time, bit for bit on
// random rings, compressed caches and selections - q's rotation, the rings' new rows and their backups, the attention's
// output rotated back.  Decode rows (one a conversation, up to 16), verify windows (a few rows a conversation, and one
// too long for the shared launch), drafter blocks (their BF16 rows as the selection), positions near 0 (window rows
// before position 0), a layer without compressed rows.  --time: both ways' launches timed at a decode step's shapes.
// Needs a GPU, no model (built by name: ds4_multi_attn_test)
#include "../src/ds4/ds4_kernels.cuh"
#include "../src/ds4/ds4_multi_attn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds4;

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                    \
            std::exit(2);                                                                                       \
        }                                                                                                       \
    } while (0)

namespace {

constexpr int H = 64, HD = 512, kWinRows = 128, RD = 64, TS = 512, NCK = 2048;

template <class T>
T* dev(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
void put(float* p, std::mt19937& rng, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = u(rng);
    CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
}

struct Case {
    std::string name;
    std::vector<int> lens;          // each conversation's rows
    std::vector<long long> p0;      // ... its first position
    bool mtp = false;               // drafter blocks
    bool comp = true;               // compressed rows selected (else a window-only layer)
    bool spec = false;              // verify windows: the rings' backups
};

// the shared inputs and both ways' outputs
struct Bufs {
    float *q0, *kvr, *kvw, *freqs, *sink, *xs;
    int* sel0;
    float *qa, *qb, *oa, *ob;
    int *sela, *selb;
    uint16_t *ext, *mext;
    AttnScratch scr7, scr16;
    std::vector<uint8_t*> ring0, ringa, ringb, ckv;
    uint8_t *baka, *bakb;
};

size_t ring_bytes() { return (size_t) kWinRows * kv_row_bytes(kKvFp8, HD); }

// attn_core's launches: q rotated a row at a time (layer()'s per-row finish), then each run alone through one window
// buffer (ext), the few-queries scratch of kAttnScratchT queries
void per_run(const Case& c, const MaPlan& p, const int* nsel, Bufs& b, const int64_t* rpos, cudaStream_t s) {
    const int T = p.T;
    CK(cudaMemcpyAsync(b.qa, b.q0, (size_t) T * H * HD * 4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(b.sela, b.sel0, (size_t) T * TS * 4, cudaMemcpyDeviceToDevice, s));
    Finish fr;
    fr.rd = RD;
    fr.vpp = H;
    fr.freqs = b.freqs;
    for (int t = 0; t < T; ++t) {
        fr.pos = (int) rpos[t];
        finish(b.qa + (size_t) t * H * HD, H, HD, HD, fr, s);
    }
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        ring_to_ext(b.ringa[(size_t) k], b.ext, c.mtp ? (int64_t) kWinRows : u.p0, kWinRows, HD, s);
        Finish fk;
        fk.w = b.kvw;
        fk.eps = 1e-6f;
        fk.rd = RD;
        fk.pos = (int) u.p0;
        fk.freqs = b.freqs;
        fk.quant = 1;
        fk.dst = b.ext + (size_t) kWinRows * HD;
        fk.dst_stride = HD;
        finish(b.kvr + (size_t) u.r0 * HD, u.n, HD, HD, fk, s);
        if (!c.mtp)
            ext_to_ring(b.ext, b.ringa[(size_t) k], u.p0, kWinRows, u.n, HD, s,
                        c.spec ? b.baka + (size_t) u.r0 * kv_row_bytes(kKvFp8, HD) : nullptr);
        AttnRows ar;
        for (int t = 0; t < u.n; ++t) {
            if (c.mtp) {
                ar.win0[t] = 0;
                ar.nwin[t] = (int) std::min<int64_t>(kWinRows, u.p0);
                ar.nsel[t] = u.n;
                continue;
            }
            const int first = (int) std::max<int64_t>(kWinRows - u.p0, t + 1);
            ar.win0[t] = first;
            ar.nwin[t] = kWinRows + t - first + 1;
            ar.nsel[t] = nsel[u.r0 + t];
        }
        int* sel = b.sela + (size_t) u.r0 * TS;
        const void* ckv = c.comp ? b.ckv[(size_t) k] : nullptr;
        int fmt = kKvFp4;
        if (c.mtp) {
            ckv = b.ext + (size_t) kWinRows * HD;
            fmt = kKvBf16;
            iota_rows(sel, u.n, u.n, TS, s);
        }
        sparse_attn(b.qa + (size_t) u.r0 * H * HD, b.ext, ckv, sel, TS, ar, b.sink, H, HD, u.n, 1.0f / std::sqrt((float) HD),
                    b.oa + (size_t) u.r0 * H * HD, s, fmt, 0, &b.scr7);
        Finish fo;
        fo.rd = RD;
        fo.pos = (int) u.p0;
        fo.vpp = H;
        fo.freqs = b.freqs;
        fo.inverse = 1;
        fo.quant = 1;
        finish(b.oa + (size_t) u.r0 * H * HD, u.n * H, HD, HD, fo, s);
    }
}

// attn_multi's launches (layer()'s q rotation with it)
bool shared(const Case& c, const MaPlan& p, const int* nsel, Bufs& b, const int64_t* rpos, cudaStream_t s,
            int* spans_out = nullptr) {
    const int T = p.T;
    CK(cudaMemcpyAsync(b.qb, b.q0, (size_t) T * H * HD * 4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(b.selb, b.sel0, (size_t) T * TS * 4, cudaMemcpyDeviceToDevice, s));
    Finish fr;
    fr.rd = RD;
    fr.vpp = H;
    fr.freqs = b.freqs;
    FinishRows fq;
    for (int t = 0; t < T; ++t) fq.pos[t] = (int) rpos[t];
    if (!finish_rows(b.qb, T * H, HD, HD, fr, fq, s)) return false;
    RingRuns rr;
    rr.runs = p.nrun;
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        rr.ring[k] = b.ringb[(size_t) k];
        rr.bak[k] = c.spec ? b.bakb + (size_t) u.r0 * kv_row_bytes(kKvFp8, HD) : nullptr;
        rr.p0[k] = c.mtp ? (long long) kWinRows : (long long) u.p0;
        rr.e0[k] = u.e0;
        rr.T[k] = u.n;
    }
    ring_to_ext_runs(rr, b.mext, kWinRows, HD, s);
    Finish fk;
    fk.w = b.kvw;
    fk.eps = 1e-6f;
    fk.rd = RD;
    fk.freqs = b.freqs;
    fk.quant = 1;
    fk.dst = b.mext;
    fk.dst_stride = HD;
    FinishRows fr2;
    for (int t = 0; t < T; ++t) {
        fr2.pos[t] = p.pos[t];
        fr2.dst[t] = p.dst[t];
    }
    if (!finish_rows(b.kvr, T, HD, HD, fk, fr2, s)) return false;
    if (!c.mtp) ext_to_ring_runs(rr, b.mext, kWinRows, HD, s);
    AttnRows ar;
    AttnCkv ck;
    for (int t = 0; t < T; ++t) {
        const MaRun& u = p.run[p.row_run[t]];
        ar.win0[t] = p.win0[t];
        ar.nwin[t] = p.nwin[t];
        ar.nsel[t] = c.mtp ? u.n : nsel[t];
        ck.ckv[t] = c.comp ? b.ckv[(size_t) p.row_run[t]] : nullptr;
    }
    const float scale = 1.0f / std::sqrt((float) HD);
    int nspans = 1;
    if (c.mtp) {
        IotaRows io;
        for (int t = 0; t < T; ++t) {
            const MaRun& u = p.run[p.row_run[t]];
            io.base[t] = u.e0 + kWinRows;
            io.n[t] = u.n;
        }
        iota_rows_at(b.selb, T, io, TS, s);
        sparse_attn(b.qb, b.mext, b.mext, b.selb, TS, ar, b.sink, H, HD, T, scale, b.ob, s, kKvBf16, 0, &b.scr7);
    } else {
        int cls[kMaRows];
        for (int k = 0; k < p.nrun; ++k) cls[k] = attn_few(p.run[k].n, TS, &b.scr7) ? (p.run[k].n == 1 ? 1 : 2) : 0;
        MaSpan sp[kMaRows];
        nspans = ma_spans(p, cls, b.scr16.max_t, sp);
        for (int i = 0; i < nspans; ++i) {
            const MaSpan& m = sp[i];
            AttnRows a;
            AttnCkv cc;
            for (int t = 0; t < m.n; ++t) {
                a.win0[t] = ar.win0[m.r0 + t];
                a.nwin[t] = ar.nwin[m.r0 + t];
                a.nsel[t] = ar.nsel[m.r0 + t];
                cc.ckv[t] = ck.ckv[m.r0 + t];
            }
            float* qm = b.qb + (size_t) m.r0 * H * HD;
            float* om = b.ob + (size_t) m.r0 * H * HD;
            int* sel = b.selb + (size_t) m.r0 * TS;
            if (m.cls == 0) sparse_attn(qm, b.mext, cc.ckv[0], sel, TS, a, b.sink, H, HD, m.n, scale, om, s, kKvFp4, 0, &b.scr7);
            else if (!sparse_attn_rows(qm, b.mext, cc, sel, TS, a, b.sink, H, HD, m.n, scale, om, s, m.cls == 1, &b.scr16))
                return false;
        }
    }
    Finish fo;
    fo.rd = RD;
    fo.vpp = H;
    fo.freqs = b.freqs;
    fo.inverse = 1;
    fo.quant = 1;
    FinishRows fp;
    for (int t = 0; t < T; ++t) fp.pos[t] = p.pos[t];
    if (!finish_rows(b.ob, T * H, HD, HD, fo, fp, s)) return false;
    if (spans_out != nullptr) *spans_out = nspans;
    return true;
}

// a case's random state: each conversation's ring (FP8 rows, the window's own rounding) and compressed rows (FP4), the
// queries, the KV rows, each row's selection
void fill(const Case& c, Bufs& b, std::mt19937& rng, int T, std::vector<int>& nsel) {
    const size_t rb = kv_row_bytes(kKvFp8, HD), cb = kv_row_bytes(kKvFp4, HD);
    for (size_t k = 0; k < c.lens.size(); ++k) {
        put(b.xs, rng, (size_t) NCK * HD, -3.0f, 3.0f);
        Finish fp;
        fp.quant = 1;
        fp.pdst = b.ring0[k];
        fp.pdst_stride = (long long) rb;
        finish(b.xs, kWinRows, HD, HD, fp, nullptr);
        fp.quant = 2;
        fp.pdst = b.ckv[k];
        fp.pdst_stride = (long long) cb;
        finish(b.xs, NCK, HD, HD, fp, nullptr);
        CK(cudaMemcpy(b.ringa[k], b.ring0[k], ring_bytes(), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(b.ringb[k], b.ring0[k], ring_bytes(), cudaMemcpyDeviceToDevice));
    }
    put(b.q0, rng, (size_t) T * H * HD, -1.0f, 1.0f);
    put(b.kvr, rng, (size_t) T * HD, -2.0f, 2.0f);
    std::vector<int> sel((size_t) T * TS);
    nsel.assign((size_t) T, 0);
    for (int t = 0; t < T; ++t) {
        const int pick = (int) (rng() % 4);
        nsel[(size_t) t] = !c.comp ? 0 : pick == 0 ? 0 : pick == 1 ? TS : 1 + (int) (rng() % TS);
        for (int i = 0; i < TS; ++i) sel[(size_t) t * TS + i] = (int) (rng() % NCK);
    }
    CK(cudaMemcpy(b.sel0, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(b.oa, 0, (size_t) kWin * H * HD * 4));
    CK(cudaMemset(b.ob, 0xff, (size_t) kWin * H * HD * 4));   // (any row the shared way leaves unwritten shows)
    CK(cudaMemset(b.baka, 0, (size_t) kWin * rb));
    CK(cudaMemset(b.bakb, 0, (size_t) kWin * rb));
    CK(cudaDeviceSynchronize());
}

int same(const char* what, const void* a, const void* b, size_t n) {
    if (std::memcmp(a, b, n) == 0) return 0;
    const uint8_t *x = (const uint8_t*) a, *y = (const uint8_t*) b;
    size_t i = 0;
    while (x[i] == y[i]) ++i;
    std::printf("    %s differs from byte %zu\n", what, i);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool time = argc > 1 && std::strcmp(argv[1], "--time") == 0;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));   // (a blocking stream: after fill's work on the default one)
    const size_t rb = kv_row_bytes(kKvFp8, HD), cb = kv_row_bytes(kKvFp4, HD);
    Bufs b;
    b.q0 = dev<float>((size_t) kWin * H * HD);
    b.qa = dev<float>((size_t) kWin * H * HD);
    b.qb = dev<float>((size_t) kWin * H * HD);
    b.oa = dev<float>((size_t) kWin * H * HD);
    b.ob = dev<float>((size_t) kWin * H * HD);
    b.kvr = dev<float>((size_t) kWin * HD);
    b.kvw = dev<float>(HD);
    b.freqs = dev<float>(RD / 2);
    b.sink = dev<float>(H);
    b.xs = dev<float>((size_t) NCK * HD);
    b.sel0 = dev<int>((size_t) kWin * TS);
    b.sela = dev<int>((size_t) kWin * TS);
    b.selb = dev<int>((size_t) kWin * TS);
    b.ext = dev<uint16_t>((size_t) (kWinRows + kWin) * HD);
    b.mext = dev<uint16_t>((size_t) ma_rows_max(kWinRows) * HD);
    b.scr7.rows = dev<uint16_t>((size_t) kAttnScratchT * TS * HD);
    b.scr7.max_t = kAttnScratchT;
    b.scr7.cap = TS;
    b.scr16.rows = dev<uint16_t>((size_t) kWin * TS * HD);
    b.scr16.max_t = kWin;
    b.scr16.cap = TS;
    b.baka = dev<uint8_t>((size_t) kWin * rb);
    b.bakb = dev<uint8_t>((size_t) kWin * rb);
    for (int k = 0; k < kMaRows; ++k) {
        b.ring0.push_back(dev<uint8_t>(ring_bytes()));
        b.ringa.push_back(dev<uint8_t>(ring_bytes()));
        b.ringb.push_back(dev<uint8_t>(ring_bytes()));
        b.ckv.push_back(dev<uint8_t>((size_t) NCK * cb));
    }
    std::mt19937 rng(20261028);
    put(b.kvw, rng, HD, 0.5f, 1.5f);
    put(b.sink, rng, H, -1.0f, 1.0f);
    {
        std::vector<float> f(RD / 2);
        for (int i = 0; i < RD / 2; ++i) f[(size_t) i] = std::pow(10000.0f, -2.0f * (float) i / (float) RD);
        CK(cudaMemcpy(b.freqs, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    }

    std::vector<Case> cases = {
        {"decode 2", {1, 1}, {0, 300}},
        {"decode 4", {1, 1, 1, 1}, {5, 127, 128, 70000}},
        {"decode 8", std::vector<int>(8, 1), {1, 2, 64, 127, 129, 1000, 4096, 100000}},
        {"decode 16", std::vector<int>(16, 1), std::vector<long long>(16, 5000)},
        {"decode 4, no compressed rows", {1, 1, 1, 1}, {3, 130, 900, 20000}, false, false},
        {"verify 4 x 4", {4, 4, 4, 4}, {0, 125, 2000, 60000}, false, true, true},
        {"verify 2 + 5 + 3", {2, 5, 3}, {126, 9, 777}, false, true, true},
        {"verify 9 + 1 (a long window alone)", {9, 1}, {120, 40}, false, true, true},
        {"verify 8 + 8 (both alone)", {8, 8}, {3000, 4}, false, true, true},
        {"verify 1 + 3 + 1 + 3", {1, 3, 1, 3}, {50, 51, 52, 53}, false, true, true},
        {"drafter 4 x 4", {4, 4, 4, 4}, {0, 60, 128, 9000}, true},
        {"drafter 3 + 3 + 3", {3, 3, 3}, {127, 2, 500}, true},
        {"drafter 2 x 8", std::vector<int>(8, 2), {1, 10, 100, 126, 127, 128, 129, 50000}, true},
    };
    int bad = 0;
    for (int rep = 0; rep < 2; ++rep)
        for (const Case& c : cases) {
            int T = 0;
            int rseq[kMaRows];
            int64_t rpos[kMaRows];
            for (size_t k = 0; k < c.lens.size(); ++k)
                for (int i = 0; i < c.lens[k]; ++i, ++T) {
                    rseq[T] = 7 + (int) k;
                    rpos[T] = c.p0[k] + i;
                }
            MaPlan p;
            if (!ma_plan(rseq, rpos, T, kWinRows, c.mtp, p)) {
                std::printf("FAIL %s: no plan\n", c.name.c_str());
                ++bad;
                continue;
            }
            std::vector<int> nsel;
            fill(c, b, rng, T, nsel);
            per_run(c, p, nsel.data(), b, rpos, s);
            int spans = 0;
            if (!shared(c, p, nsel.data(), b, rpos, s, &spans)) {
                std::printf("FAIL %s: a launch refused\n", c.name.c_str());
                ++bad;
                continue;
            }
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            int e = 0;
            const size_t nq = (size_t) T * H * HD;
            e += same("q (rotated)", host(b.qa, nq).data(), host(b.qb, nq).data(), nq * 4);
            e += same("the attention's output", host(b.oa, nq).data(), host(b.ob, nq).data(), nq * 4);
            for (int k = 0; k < p.nrun; ++k)
                e += same(("ring " + std::to_string(k)).c_str(), host(b.ringa[(size_t) k], ring_bytes()).data(),
                          host(b.ringb[(size_t) k], ring_bytes()).data(), ring_bytes());
            if (c.spec) e += same("the rings' backups", host(b.baka, (size_t) T * rb).data(), host(b.bakb, (size_t) T * rb).data(), (size_t) T * rb);
            std::printf("%s %s: %d rows, %d runs, %d attention launch%s\n", e ? "FAIL" : "ok  ", c.name.c_str(), T, p.nrun,
                        c.mtp ? 1 : spans, (c.mtp || spans == 1) ? "" : "es");
            bad += e != 0;
        }
    if (time) {   // a decode step's attention launches (q's rotation to the output's finish), N conversations
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int N : {2, 4, 8, 16}) {
            Case c{"time", std::vector<int>((size_t) N, 1), std::vector<long long>((size_t) N, 20000)};
            int rseq[kMaRows];
            int64_t rpos[kMaRows];
            for (int t = 0; t < N; ++t) {
                rseq[t] = t;
                rpos[t] = 20000;
            }
            MaPlan p;
            ma_plan(rseq, rpos, N, kWinRows, false, p);
            std::vector<int> nsel;
            fill(c, b, rng, N, nsel);
            for (int t = 0; t < N; ++t) nsel[(size_t) t] = TS;
            float ms[2] = {0, 0};
            for (int way = 0; way < 2; ++way) {
                for (int w = 0; w < 20; ++w) way ? (void) shared(c, p, nsel.data(), b, rpos, s) : per_run(c, p, nsel.data(), b, rpos, s);
                CK(cudaEventRecord(e0, s));
                const int it = 200;
                for (int w = 0; w < it; ++w) way ? (void) shared(c, p, nsel.data(), b, rpos, s) : per_run(c, p, nsel.data(), b, rpos, s);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[way], e0, e1));
                ms[way] = ms[way] * 1000.0f / it;
            }
            std::printf("decode %2d conversations: a conversation at a time %.1f us, all in one launch a step %.1f us\n", N,
                        ms[0], ms[1]);
        }
    }
    std::printf(bad ? "ds4_multi_attn_test: %d cases FAILED\n" : "ds4_multi_attn_test: ok\n", bad);
    return bad ? 1 : 0;
}
