// src/ds4/ds4_part_impl.inc - inside Model::Impl: the pool.  Two PCs run one model: the front (the PC the server
// talks to) runs the trunk's layers [0, K), the part server the layers [K, nl), the head, the sampling and the DSpark
// drafter.  Every window's and chunk's rows leaving layer K - 1 (the hand-off a layer split passes between GPUs: the
// streams, pre, the index picks and the candidates, F32 as computed - bit for bit what the next layer would read on
// one PC) go to the part with the new rows of the KV sources its layers read; the part sends back what the front needs
// (a sampled token, a verify's verdict and the next round's drafts).  Each side keeps its own layers' state; every
// call that changes the state (windows, runs of chunks, a verify's undo, checkpoints, restore points, reset) is made on
// both, in order, so the two halves always hold the same positions.  A link that fails or times out: the front drops
// it, reads the conversation again through all its layers (it keeps the whole model) and finishes the request alone.
//
// One conversation (--parallel 1); no pictures, no slots and no log-probabilities through the link (each falls back).

// ---- state
int part_lo = 0, part_hi = 1 << 30;   // the trunk layers this process runs now (a front with a live link: [0, K))
bool part_srv = false;                // this process is a part server
std::unique_ptr<part::Conn> pc;       // a front's link to its part (null or closed: none)
std::string part_host;
int part_port = 0, part_k = 0;
bool part_failed = false;   // a front: the link failed under the last call
bool part_pending = false;  // a front: the link failed between calls - the conversation must be read again here
bool part_defer = false;    // a front: forward() keeps its window's frame in part_w (verify sends it)
int part_timeout_ms = 20000;
part::Writer part_w;
double part_net_ms = 0, part_wait_ms = 0, part_local_ms = 0;   // a front: time sending; waiting for replies; own layers
double part_remote_ms = 0;   // a front: the part's own time on the calls it answered (it says so in each answer)
double part_busy_ms = 0;     // a part: its time on the calls since its last answer
double part_busy_total = 0;
int64_t part_waits = 0, part_pre_hits = 0;
int64_t part_rounds = 0, part_tx_bytes = 0;
struct PartSrc {
    int sidx = -1;
};
std::vector<PartSrc> part_srcs;   // the KV sources before K that layers from K on read
// a front's run of chunks: two pinned slots, a sender thread
struct PartTx {
    uint8_t* slot[2] = {nullptr, nullptr};
    size_t cap = 0;
    cudaEvent_t ev[2] = {nullptr, nullptr};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::array<int64_t, 4>> q;   // k, slot, hdr index, raw bytes
    std::vector<std::vector<uint8_t>> hdr;  // a header a chunk
    std::atomic<int> busy[2];
    std::atomic<bool> bad{false};
    bool done = false;
    std::thread th;
    std::string err;
    int dev_id = 0;
    double send_ms = 0;
} ptx;
// a part server's input: the frame being read, its sources' rows and its hand-off rows
struct PartRows {
    std::vector<uint8_t> buf;
    struct S {
        int sidx;
        int64_t g_first;
        int ng;
        const uint8_t *ckv, *ik;
    };
    std::vector<S> src;
    const uint8_t* hand = nullptr;
    int T = 0, r0 = 0;   // the hand-off's rows: [r0, T) of the window / chunk
    int k = -1;
};
const PartRows* part_cur = nullptr;
const FeedPlan* part_plan_force = nullptr;
// a part server's chunks: frames the reader thread queued (k -1: the front's run ended; -2: the link closed)
std::mutex prx_mu;
std::condition_variable prx_cv;
std::deque<std::unique_ptr<PartRows>> prx_chunks;
std::deque<std::pair<uint32_t, std::vector<uint8_t>>> prx_ops;
std::unique_ptr<PartRows> prx_hold[2];
// pre-drafting: the part drafts the next round right after a verify, with the last draft call's arguments
struct DraftArgs {
    bool set = false;
    float t = 0, tp = 1, mp = 0, conf_min = 0, cum_min = 0;
    int tk = 0, max_n = 0, min_n = 0;
    uint64_t seed = 0, counter = 0;
    bool same(const DraftArgs& o, bool counter_too) const {
        return set && o.set && t == o.t && tp == o.tp && mp == o.mp && conf_min == o.conf_min && cum_min == o.cum_min &&
               tk == o.tk && max_n == o.max_n && min_n == o.min_n && seed == o.seed && (!counter_too || counter == o.counter);
    }
};
DraftArgs part_dargs;   // the part: the last draft()'s; the front: the args its pre-draft was made with
struct PreDraft {
    bool have = false, q = false;
    int y = -1;
    std::vector<int> d;
    std::vector<float> c;
} part_pre;

// ---- common
bool away(int il) const { return il < g.nl && (il < part_lo || il >= part_hi); }
bool part_link() const { return pc != nullptr && pc->ok() && part_hi < g.nl; }
size_t part_hand_row() const { return (size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4; }
void part_src_list(int K) {
    part_srcs.clear();
    for (size_t s = 0; s < srcs.size(); ++s) {
        if (srcs[s].layer >= K) continue;
        bool used = false;
        for (int il = K; il < g.nl; ++il) used = used || L[(size_t) il].src == (int) s || L[(size_t) il].ik_src == (int) s;
        if (used) part_srcs.push_back({(int) s});
    }
}
// a run's / window's rows of the sources: g_first, ng of source s for positions [p0, p0 + T)
void part_src_rows(int s, int64_t p0, int T, int64_t& g_first, int& ng) const {
    const Src& sr = srcs[(size_t) s];
    g_first = p0 / sr.ratio;
    ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
}
// the header of a rows block (sources, then the hand-off of T rows or none) and its raw bytes
size_t part_rows_hdr(part::Writer& w, int64_t p0, int T, bool hand, int dev, int r0 = 0) const {
    size_t raw = 0;
    w.put<uint32_t>((uint32_t) part_srcs.size());
    for (const PartSrc& ps : part_srcs) {
        const Src& sr = srcs[(size_t) ps.sidx];
        int64_t gf;
        int ng;
        part_src_rows(ps.sidx, p0, T, gf, ng);
        const bool hc = sr.ckv[(size_t) dev] != nullptr, hi = sr.ik[(size_t) dev] != nullptr;
        w.put<int32_t>(ps.sidx);
        w.put<int64_t>(gf);
        w.put<int32_t>(ng);
        w.put<uint8_t>(hc ? 1 : 0);
        w.put<uint8_t>(hi ? 1 : 0);
        raw += (size_t) ng * ((hc ? ckv_rb() : 0) + (hi ? ik_rb() : 0));
    }
    w.put<int32_t>(hand ? T : 0);
    w.put<int32_t>(hand ? r0 : 0);
    if (hand) raw += (size_t) (T - r0) * part_hand_row();
    return raw;
}
// the raw bytes after it, from device dev into dst on stream s: each source's KV rows, its index keys, then the four
// parts of the hand-off (x: the window's or the chunk's buffers)
bool part_rows_d2h(uint8_t* dst, int64_t p0, int T, const Xfer* hand, int dev, cudaStream_t s, std::string& err,
                   int r0 = 0) {
    for (const PartSrc& ps : part_srcs) {
        const Src& sr = srcs[(size_t) ps.sidx];
        int64_t gf;
        int ng;
        part_src_rows(ps.sidx, p0, T, gf, ng);
        if (ng <= 0) continue;
        if (sr.ckv[(size_t) dev] != nullptr) {
            CK(cudaMemcpyAsync(dst, sr.ckv[(size_t) dev] + (size_t) gf * ckv_rb(), (size_t) ng * ckv_rb(),
                               cudaMemcpyDeviceToHost, s));
            dst += (size_t) ng * ckv_rb();
        }
        if (sr.ik[(size_t) dev] != nullptr) {
            CK(cudaMemcpyAsync(dst, sr.ik[(size_t) dev] + (size_t) gf * ik_rb(), (size_t) ng * ik_rb(),
                               cudaMemcpyDeviceToHost, s));
            dst += (size_t) ng * ik_rb();
        }
    }
    if (hand != nullptr)
        for (int i = 0; i < hand->n; ++i) {   // (each part's rows [r0, T))
            const size_t row = hand->bytes[i] / (size_t) T, n = (size_t) (T - r0) * row;
            CK(cudaMemcpyAsync(dst, (const uint8_t*) hand->src[i] + (size_t) r0 * row, n, cudaMemcpyDeviceToHost, s));
            dst += n;
        }
    return true;
}
// a part server: a rows block read from r (pointers into its frame)
bool part_rows_parse(part::Reader& r, PartRows& pr) {
    const uint32_t ns = r.get<uint32_t>();
    struct M {
        int sidx;
        int64_t gf;
        int ng;
        bool hc, hi;
    };
    std::vector<M> ms;
    for (uint32_t i = 0; i < ns && !r.bad; ++i) {
        M m;
        m.sidx = r.get<int32_t>();
        m.gf = r.get<int64_t>();
        m.ng = r.get<int32_t>();
        m.hc = r.get<uint8_t>() != 0;
        m.hi = r.get<uint8_t>() != 0;
        ms.push_back(m);
    }
    pr.T = r.get<int32_t>();
    pr.r0 = r.get<int32_t>();
    if (r.bad || pr.r0 < 0 || (pr.T > 0 && pr.r0 >= pr.T)) return false;
    pr.src.clear();
    for (const M& m : ms) {
        if (m.sidx < 0 || m.sidx >= (int) srcs.size()) return false;
        PartRows::S s{m.sidx, m.gf, m.ng, nullptr, nullptr};
        if (m.hc) s.ckv = r.bytes((size_t) m.ng * ckv_rb());
        if (m.hi) s.ik = r.bytes((size_t) m.ng * ik_rb());
        pr.src.push_back(s);
    }
    pr.hand = pr.T > 0 ? r.bytes((size_t) (pr.T - pr.r0) * part_hand_row()) : nullptr;
    return !r.bad;
}
// a part server: the sources' rows of pr into this device's copies (before the first layer reads them)
bool part_src_h2d(const PartRows& pr, int di, cudaStream_t s, std::string& err) {
    for (const PartRows::S& x : pr.src) {
        Src& sr = srcs[(size_t) x.sidx];
        if (x.ng <= 0) continue;
        if (x.ckv != nullptr && sr.ckv[(size_t) di] != nullptr)
            CK(cudaMemcpyAsync(sr.ckv[(size_t) di] + (size_t) x.g_first * ckv_rb(), x.ckv, (size_t) x.ng * ckv_rb(),
                               cudaMemcpyHostToDevice, s));
        if (x.ik != nullptr && sr.ik[(size_t) di] != nullptr)
            CK(cudaMemcpyAsync(sr.ik[(size_t) di] + (size_t) x.g_first * ik_rb(), x.ik, (size_t) x.ng * ik_rb(),
                               cudaMemcpyHostToDevice, s));
    }
    return true;
}
bool part_hand_h2d(const PartRows& pr, const Xfer& x, int T, cudaStream_t s, std::string& err) {
    if (pr.hand == nullptr || pr.T != T) {
        err = "pool: a hand-off of " + std::to_string(pr.T) + " rows for a window of " + std::to_string(T);
        return false;
    }
    const uint8_t* q = pr.hand;
    for (int i = 0; i < x.n; ++i) {
        const size_t row = x.bytes[i] / (size_t) T, n = (size_t) (T - pr.r0) * row;
        CK(cudaMemcpyAsync((uint8_t*) x.dst[i] + (size_t) pr.r0 * row, q, n, cudaMemcpyHostToDevice, s));
        q += n;
    }
    return true;
}

// ---- the front
bool part_send(uint32_t op, const part::Writer& w, const void* raw = nullptr, size_t nraw = 0) {
    if (!part_link()) return false;
    const double t0 = now_ms();
    const bool ok = pc->send_frame(op, w.b.data(), w.b.size(), raw, nraw);
    part_net_ms += now_ms() - t0;
    part_tx_bytes += (int64_t) (w.b.size() + nraw);
    if (!ok) part_lost("sending: " + pc->last_error());
    return ok;
}
// a reply: its status (0 ok, else the part's error, given in err)
bool part_reply(std::vector<uint8_t>& out, std::string& err, int timeout_ms = -1) {
    if (!part_link()) {
        err = "pool: no link";
        part_failed = true;
        return false;
    }
    const double t0 = now_ms();
    pc->set_timeout(timeout_ms > 0 ? timeout_ms : part_timeout_ms);
    uint32_t op = 0;
    const bool ok = pc->recv_frame(op, out);
    part_wait_ms += now_ms() - t0;
    if (!ok || op != part::kReply) {
        err = "pool: " + (ok ? std::string("an unexpected frame") : pc->last_error());
        part_lost(err);
        return false;
    }
    part::Reader r(out);
    const int32_t st = r.get<int32_t>();
    if (st != 0) {
        err = "pool: the part failed: " + r.str();
        part_lost(err);
        return false;
    }
    if (out.size() >= 12) {   // (the part's busy time, after the status: taken out here)
        double b = 0;
        std::memcpy(&b, out.data() + 4, 8);
        part_remote_ms += b;
        out.erase(out.begin() + 4, out.begin() + 12);
    }
    ++part_waits;
    return true;
}
// the link is gone: the front takes every layer back; the conversation is read again before the next call uses it
void part_lost(const std::string& why) {
    if (pc != nullptr && part_hi < g.nl)
        std::fprintf(stderr, "ds4 pool: the link to the part failed (%s) - this PC runs every layer from here\n",
                     why.c_str());
    if (pc != nullptr) pc->close();
    part_hi = 1 << 30;
    part_failed = true;
    part_pending = true;
    part_pre.have = false;
}
bool part_connect(std::string& err) {
    pc = std::make_unique<part::Conn>();
    if (!pc->connect(part_host, part_port, 5000, err)) return false;
    pc->set_timeout(part_timeout_ms * 6);   // (the first answer: the part may still warm up)
    part::Writer w;
    w.put<uint32_t>(1);
    w.put<int32_t>(g.nl);
    w.put<int32_t>(g.dim);
    w.put<int32_t>(g.vocab);
    w.put<int64_t>(o.max_context);
    w.put<int32_t>(chunk);
    w.put<int32_t>(g.n_mtp);
    if (!pc->send_frame(part::kHello, w.b.data(), w.b.size())) {
        err = "pool: " + pc->last_error();
        return false;
    }
    uint32_t op = 0;
    std::vector<uint8_t> out;
    if (!pc->recv_frame(op, out) || op != part::kReply) {
        err = "pool: no answer to hello (" + pc->last_error() + ")";
        pc->close();
        return false;
    }
    part::Reader r(out);
    const int32_t st = r.get<int32_t>();
    const std::string msg = r.str();
    const int32_t K = r.get<int32_t>();
    if (st != 0 || r.bad || K <= 0 || K >= g.nl) {
        err = "pool: the part refused: " + msg;
        pc->close();
        return false;
    }
    part_k = K;
    part_hi = K;
    part_src_list(K);
    part_pending = false;
    part_failed = false;
    std::fprintf(stderr, "ds4 pool: layers 0-%d here, %d-%d, the head, sampling and the drafter on %s (%s); %zu KV "
                 "source(s) shipped, %zu bytes a row\n",
                 K - 1, K, g.nl - 1, pc->peer().c_str(), msg.c_str(), part_srcs.size(), part_hand_row());
    return true;
}
// forward()'s end on a front: the window's frame (its tokens, the sources' new rows, the hand-off) built - sent as
// kWin, or kept for verify (part_defer)
bool part_window_out(int64_t p0, int T, const int* tk, int logits_rows, int seed_rows, std::string& err) {
    const int dl = L[(size_t) part_hi - 1].dev;
    Dev& d = devs[(size_t) dl];
    cudaSetDevice(d.id);
    part::Writer w;
    w.put<int64_t>(p0);
    w.put<int32_t>(T);
    w.put<int32_t>(logits_rows);
    w.put<int32_t>(seed_rows);
    w.put_bytes(tk, (size_t) T * 4);
    const size_t raw = part_rows_hdr(w, p0, T, true, dl);
    const size_t at = w.b.size();
    w.b.resize(at + raw);
    const Xfer x = hand_xfer(nullptr, d, T, true);
    Xfer hx = x;   // (the device side of the hand-off: x.src)
    if (!part_rows_d2h(w.b.data() + at, p0, T, &hx, dl, d.st, err)) return false;
    CK(cudaStreamSynchronize(d.st));
    part_local_ms += 0;
    if (part_defer) {
        part_w = std::move(w);
        return true;
    }
    if (!part_send(part::kWin, w)) {
        err = "pool: the link failed";
        return false;
    }
    return true;
}
bool part_verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                 uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err) {
    const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
    const int T = n + 1;
    const int64_t p0 = pos;
    int tk[kWin];
    tk[0] = y;
    for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
    part_defer = true;
    spec_win = true;
    const double tv = now_ms();
    const bool ok = forward(tk, T, T, err, 0);
    spec_win = false;
    part_defer = false;
    if (!ok) return false;
    part_local_ms += now_ms() - tv;
    part::Writer w;
    w.put<float>(temperature);
    w.put<float>(top_p);
    w.put<int32_t>(top_k);
    w.put<float>(min_p);
    w.put<uint64_t>(seed);
    w.put<uint64_t>(counter);
    w.put_bytes(part_w.b.data(), part_w.b.size());
    if (!part_send(part::kVerify, w)) {
        err = "pool: the link failed";
        return false;
    }
    std::vector<uint8_t> out;
    if (!part_reply(out, err)) return false;
    part::Reader r(out);
    (void) r.get<int32_t>();
    const int m = r.get<int32_t>();
    next = r.get<int32_t>();
    part_take_pre(r, next);
    if (r.bad || m < 0 || m > n) {
        err = "pool: a bad verify answer";
        part_lost(err);
        return false;
    }
    st.verify_ms += now_ms() - tv;
    drafts_q = false;
    rollback(p0, T, m + 1);
    accepted = m;
    ++part_rounds;
    st.drafts_offered += n;
    st.drafts_accepted += m;
    for (int i = 0; i < n && i <= m && i < 8; ++i) {
        ++st.draft_reached[i];
        if (i < m) ++st.draft_kept[i];
    }
    return true;
}
bool part_draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature,
                float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter, int max_n, float conf_min,
                float cum_min, int min_n) {
    DraftArgs a;
    a.set = true;
    a.t = temperature;
    a.tp = top_p;
    a.tk = top_k;
    a.mp = min_p;
    a.seed = seed;
    a.counter = counter;
    a.max_n = max_n;
    a.conf_min = conf_min;
    a.cum_min = cum_min;
    a.min_n = min_n;
    const double t0 = now_ms();
    DraftArgs a_cut = a;   // (the pre-draft's: the whole block, cut here)
    a_cut.max_n = part_dargs.max_n;
    a_cut.conf_min = part_dargs.conf_min;
    a_cut.cum_min = part_dargs.cum_min;
    a_cut.min_n = part_dargs.min_n;
    if (part_pre.have && part_pre.y == y && a_cut.same(part_dargs, part_pre.q)) {   // (drafted with the last answer)
        out.clear();
        conf.clear();
        double cum = 1.0;
        for (size_t i = 0; i < part_pre.d.size(); ++i) {   // (round()'s cut: draft_'s, on the block's rows)
            if (max_n > 0 && (int) i >= max_n) break;
            cum *= part_pre.c[i];
            if ((int) i >= min_n && (part_pre.c[i] < conf_min || cum < cum_min)) break;
            out.push_back(part_pre.d[i]);
            conf.push_back(part_pre.c[i]);
        }
        drafts_q = part_pre.q;
        part_pre.have = false;
        ++part_pre_hits;
    } else {
        part_pre.have = false;
        part::Writer w;
        w.put<int32_t>(y);
        w.put<float>(a.t);
        w.put<float>(a.tp);
        w.put<int32_t>(a.tk);
        w.put<float>(a.mp);
        w.put<uint64_t>(a.seed);
        w.put<uint64_t>(a.counter);
        w.put<int32_t>(a.max_n);
        w.put<float>(a.conf_min);
        w.put<float>(a.cum_min);
        w.put<int32_t>(a.min_n);
        if (!part_send(part::kDraft, w)) {
            err = "pool: the link failed";
            return false;
        }
        std::vector<uint8_t> rep;
        if (!part_reply(rep, err)) return false;
        part::Reader r(rep);
        (void) r.get<int32_t>();
        const int nd = r.get<int32_t>();
        drafts_q = r.get<uint8_t>() != 0;
        out.assign((size_t) std::max(0, nd), 0);
        conf.assign((size_t) std::max(0, nd), 0.0f);
        for (int i = 0; i < nd; ++i) out[(size_t) i] = r.get<int32_t>();
        for (int i = 0; i < nd; ++i) conf[(size_t) i] = r.get<float>();
        if (r.bad) {
            err = "pool: a bad draft answer";
            part_lost(err);
            return false;
        }
        dq_temp = temperature;
        dq_top_p = top_p;
        dq_top_k = top_k;
        dq_min_p = min_p;
        dq_seed = seed;
        dq_counter = counter;
    }
    part_dargs = a;
    if (max_drafts > 0 && (int) out.size() > max_drafts) {
        out.resize((size_t) max_drafts);
        conf.resize((size_t) max_drafts);
    }
    st.draft_ms += now_ms() - t0;
    ++st.drafts_rounds;
    return true;
}
// sample() on a front: -2 when the link failed (Model::sample reads the conversation again and samples here)
int part_sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
    part::Writer w;
    w.put<float>(temperature);
    w.put<float>(top_p);
    w.put<int32_t>(top_k);
    w.put<float>(min_p);
    w.put<uint64_t>(seed);
    w.put<uint64_t>(counter);
    if (!part_send(part::kSample, w)) return -2;
    std::vector<uint8_t> rep;
    std::string err;
    if (!part_reply(rep, err)) return -2;
    part::Reader r(rep);
    (void) r.get<int32_t>();
    const int t = r.get<int32_t>();
    if (r.bad) return -2;
    part_take_pre(r, t);
    return t;
}
// a simple call made on the part too (no answer: a failure there shows at the next answer)
void part_mirror(uint32_t op, const part::Writer& w) {
    if (part_link()) (void) part_send(op, w);
}
bool part_restore_check(int64_t limit, bool here, int64_t here_pos) {
    part::Writer w;
    w.put<int64_t>(limit);
    if (!part_send(part::kRestore, w)) return false;
    std::vector<uint8_t> rep;
    std::string err;
    if (!part_reply(rep, err)) return false;
    part::Reader r(rep);
    (void) r.get<int32_t>();
    const bool ok = r.get<int32_t>() != 0;
    const int64_t p = r.get<int64_t>();
    if (ok != here || (ok && p != here_pos)) {
        part_lost("the part restored to " + std::to_string(ok ? p : -1) + ", this PC to " +
                  std::to_string(here ? here_pos : -1));
        return false;
    }
    return true;
}
// the conversation's first keep tokens read again through every layer here (after the link failed); logits: the last
// row's, for a sample
bool part_recover(int64_t keep, bool logits, std::string& err) {
    std::vector<int> t(toks.begin(), toks.begin() + std::min<int64_t>(keep, (int64_t) toks.size()));
    if (pc != nullptr) pc->close();
    part_hi = 1 << 30;
    part_pending = false;
    part_failed = false;
    reset();
    if (t.empty()) return true;
    const double t0 = now_ms();
    halted = false;
    fed_n = (int) t.size();
    const int n = (int) t.size();
    bool ok;
    if (chunk > 0 && n >= chunk_min) ok = forward_chunks(t.data(), n, logits ? 1 : 0, err);
    else ok = feed_windows(t.data(), n, logits ? 1 : 0, err);
    std::fprintf(stderr, "ds4 pool: %d tokens read again here in %.0f ms (%s)\n", n, now_ms() - t0,
                 ok ? "ok" : err.c_str());
    return ok;
}
// the front's run of chunks: the header (the plan, the call, the tokens) before the devices start
bool part_chunks_begin(const FeedPlan& fp, bool whole, int k0, int k1, const int* tk, int n, int logits_rows,
                       int snap_slot, int after, const std::vector<std::pair<int64_t, bool>>* inner, int per,
                       std::string& err) {
    part::Writer w;
    w.put<uint8_t>(whole ? 1 : 0);
    w.put<int64_t>(fp.at);
    w.put<int32_t>(fp.n);
    w.put<int32_t>(fp.logits_rows);
    w.put<int32_t>(fp.slot);
    w.put<int32_t>(fp.skip);
    w.put<int32_t>(fp.cut_k);
    w.put<int32_t>(fp.cut_r);
    w.put<uint32_t>((uint32_t) fp.ends.size());
    for (int e : fp.ends) w.put<int32_t>(e);
    w.put<int32_t>(k0);
    w.put<int32_t>(k1);
    w.put<int32_t>(n);
    w.put<int32_t>(logits_rows);
    w.put<int32_t>(snap_slot);
    w.put<int32_t>(after);
    w.put<int32_t>(inner != nullptr ? (int) inner->size() : -1);
    if (inner != nullptr)
        for (const auto& m : *inner) {
            w.put<int64_t>(m.first);
            w.put<uint8_t>(m.second ? 1 : 0);
        }
    w.put<int64_t>(pos);
    w.put_bytes(tk, (size_t) n * 4);
    if (!part_send(part::kChunks, w)) {
        err = "pool: the link failed";
        return false;
    }
    // the slots: a chunk's rows at most (its sources' rows and its hand-off)
    size_t need = (size_t) per * part_hand_row();
    for (const PartSrc& ps : part_srcs) {
        (void) ps;
        need += (size_t) (per + 2) * (ckv_rb() + ik_rb());
    }
    const int dl = L[(size_t) part_hi - 1].dev;
    cudaSetDevice(devs[(size_t) dl].id);
    if (ptx.cap < need) {
        for (auto& s : ptx.slot)
            if (s != nullptr) cudaFreeHost(s);
        ptx.slot[0] = ptx.slot[1] = nullptr;
        ptx.cap = 0;
        if (!halloc(ptx.slot[0], need, err) || !halloc(ptx.slot[1], need, err)) return false;
        ptx.cap = need;
    }
    for (auto& e : ptx.ev)
        if (e == nullptr) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    ptx.busy[0].store(0);
    ptx.busy[1].store(0);
    ptx.bad.store(false);
    ptx.done = false;
    ptx.err.clear();
    ptx.q.clear();
    ptx.hdr.clear();
    ptx.hdr.reserve(fp.ends.size() + 1);
    ptx.send_ms = 0;
    ptx.dev_id = devs[(size_t) dl].id;
    ptx.th = std::thread([this] { part_tx_loop(); });
    return true;
}
void part_tx_loop() {
    cudaSetDevice(ptx.dev_id);
    for (;;) {
        std::array<int64_t, 4> j;
        {
            std::unique_lock<std::mutex> lk(ptx.mu);
            ptx.cv.wait(lk, [this] { return !ptx.q.empty() || ptx.done; });
            if (ptx.q.empty()) return;
            j = ptx.q.front();
            ptx.q.pop_front();
        }
        const int sl = (int) j[1];
        if (!ptx.bad.load()) {
            if (cudaEventSynchronize(ptx.ev[sl]) != cudaSuccess) {
                ptx.err = "pool: a chunk's rows could not be read back";
                ptx.bad.store(true);
            } else {
                const std::vector<uint8_t>& h = ptx.hdr[(size_t) j[2]];
                const double t0 = now_ms();
                if (!pc->send_frame(part::kChunk, h.data(), h.size(), ptx.slot[sl], (size_t) j[3])) {
                    ptx.err = "pool: sending a chunk: " + pc->last_error();
                    ptx.bad.store(true);
                }
                ptx.send_ms += now_ms() - t0;
                part_tx_bytes += (int64_t) (h.size() + (size_t) j[3]);
            }
        }
        ptx.busy[sl].store(0, std::memory_order_release);
    }
}
// run_dev_chunks on a front, after chunk k's layers here: its rows into slot k % 2 and on to the sender
bool part_chunk_out(Dev& d, int di, Chunk& c, const Run& run, int k, cudaStream_t s, std::string& err) {
    const int sl = k % 2;
    while (ptx.busy[sl].load(std::memory_order_acquire) != 0) {
        if (ptx.bad.load()) break;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (ptx.bad.load()) {
        err = ptx.err;
        return false;
    }
    const int64_t p0 = run.p0[(size_t) k];
    const int T = run.T[(size_t) k];
    const bool hand = run.runs(k, part_hi);
    const int r0 = hand && run.yoco >= 0 && k == run.cut_k ? std::max(0, std::min(run.cut_r, T - 1)) : 0;
    std::vector<uint8_t> h;
    size_t raw = 0;
    {
        part::Writer w;
        w.put<int32_t>(k);
        raw = part_rows_hdr(w, p0, T, hand, di, r0);
        if (raw > ptx.cap) {
            err = "pool: a chunk larger than its slot";
            return false;
        }
        h = std::move(w.b);
    }
    const Xfer x = chunk_xfer(nullptr, c, T, true);
    if (!part_rows_d2h(ptx.slot[sl], p0, T, hand ? &x : nullptr, di, s, err, r0)) return false;
    CK(cudaEventRecord(ptx.ev[sl], s));
    ptx.busy[sl].store(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(ptx.mu);
        ptx.hdr.push_back(std::move(h));
        ptx.q.push_back({k, sl, (int64_t) ptx.hdr.size() - 1, (int64_t) raw});
    }
    ptx.cv.notify_one();
    (void) d;
    return true;
}
// the front's run over (any way): the sender drained, the end said (took: the tokens the run took), the part's answer
bool part_chunks_end(int took, bool ok_here, std::string& err) {
    {
        std::lock_guard<std::mutex> lk(ptx.mu);
        ptx.done = true;
    }
    ptx.cv.notify_one();
    if (ptx.th.joinable()) ptx.th.join();
    part_net_ms += ptx.send_ms;
    if (ptx.bad.load()) {
        err = ptx.err;
        part_lost(err);
        return false;
    }
    part::Writer w;
    w.put<int32_t>(-1);
    w.put<int32_t>(took);
    w.put<int32_t>(ok_here ? 1 : 0);
    if (!part_send(part::kChunk, w)) {
        err = "pool: the link failed";
        return false;
    }
    std::vector<uint8_t> rep;
    if (!part_reply(rep, err, part_timeout_ms * 6)) return false;
    part::Reader r(rep);
    (void) r.get<int32_t>();
    const int32_t their = r.get<int32_t>();   // 1: read, 0: stopped (as this side was)
    if (ok_here && their != 1) {
        err = "pool: the part did not finish the run";
        part_lost(err);
        return false;
    }
    return true;
}

// ---- the part server
// run_dev_chunks on a part: chunk k's frame (false: the front's run ended before it, or the link closed)
bool part_chunk_in(int k, std::string& err) {
    std::unique_lock<std::mutex> lk(prx_mu);
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(part_timeout_ms * 6);
    while (prx_chunks.empty()) {
        if (prx_cv.wait_until(lk, t_end) == std::cv_status::timeout) {
            err = "pool: no chunk from the front in time";
            return false;
        }
    }
    std::unique_ptr<PartRows> f = std::move(prx_chunks.front());
    if (f->k != k) {
        if (f->k == -1) {   // (the front's run ended: kept for the main thread)
            err = kHalted;
            return false;
        }
        prx_chunks.pop_front();
        err = f->k == -2 ? "pool: the link closed" : "pool: chunk " + std::to_string(f->k) + " where " +
                                                         std::to_string(k) + " was due";
        return false;
    }
    prx_chunks.pop_front();
    prx_hold[k % 2] = std::move(f);
    return true;
}
void part_reader_loop() {
    for (;;) {
        uint32_t op = 0;
        std::vector<uint8_t> b;
        pc->set_timeout(0);
        const bool ok = pc->recv_frame(op, b);
        std::lock_guard<std::mutex> lk(prx_mu);
        if (!ok) {
            auto f = std::make_unique<PartRows>();
            f->k = -2;
            prx_chunks.push_back(std::move(f));
            prx_ops.push_back({part::kBye, {}});
            prx_cv.notify_all();
            return;
        }
        if (op == part::kChunk) {
            auto f = std::make_unique<PartRows>();
            f->buf = std::move(b);
            part::Reader r(f->buf);
            f->k = r.get<int32_t>();
            if (f->k >= 0) {
                if (!part_rows_parse(r, *f)) f->k = -3;
            } else {   // (the run's end: the front's tokens taken, its own run done)
                f->T = r.get<int32_t>();
            }
            prx_chunks.push_back(std::move(f));
        } else
            prx_ops.push_back({op, std::move(b)});
        prx_cv.notify_all();
    }
}
bool part_reply_send(int32_t st, const std::string& msg, const part::Writer* w = nullptr) {
    part::Writer h;
    h.put<int32_t>(st);
    if (st != 0) h.put_str(msg);
    else {
        h.put<double>(part_busy_ms);
        part_busy_ms = 0;
    }
    return pc->send_frame(part::kReply, h.b.data(), h.b.size(), w ? w->b.data() : nullptr, w ? w->b.size() : 0);
}
// the part server's loop: one front at a time; each connection starts from an empty conversation
bool part_serve(int port, std::string& err) {
    pc = std::make_unique<part::Conn>();
    if (!pc->listen(port, err)) return false;
    std::fprintf(stderr, "ds4 pool: part server on port %d, layers %d-%d, the head%s\n", port, part_lo, g.nl - 1,
                 g.n_mtp > 0 ? " and the drafter" : "");
    std::fflush(stderr);
    for (;;) {
        if (!pc->accept(err)) return false;
        std::fprintf(stderr, "ds4 pool: front %s connected\n", pc->peer().c_str());
        reset();
        part_dargs = DraftArgs();
        prx_chunks.clear();
        prx_ops.clear();
        std::thread rd([this] { part_reader_loop(); });
        const Stats s0 = st;
        const double tb0 = part_busy_total;
        std::string why = part_session();
        std::fprintf(stderr, "ds4 pool: the session: busy %.0f ms; layers %.0f ms (head wait %.0f), %lld tokens; drafting "
                     "%.0f ms in %lld drafts; verify windows %.0f ms\n", part_busy_total - tb0, st.layer_ms - s0.layer_ms,
                     st.head_ms - s0.head_ms, (long long) (st.tokens - s0.tokens), st.draft_ms - s0.draft_ms,
                     (long long) (st.drafts_rounds - s0.drafts_rounds), st.verify_ms - s0.verify_ms);
        pc->close();
        if (rd.joinable()) rd.join();
        prx_chunks.clear();
        prx_ops.clear();
        prx_hold[0].reset();
        prx_hold[1].reset();
        std::fprintf(stderr, "ds4 pool: front gone (%s)\n", why.c_str());
        std::fflush(stderr);
    }
}
std::string part_session() {
    std::string fail;   // (the first failure: every answer after it says so)
    for (;;) {
        std::pair<uint32_t, std::vector<uint8_t>> m;
        {
            std::unique_lock<std::mutex> lk(prx_mu);
            prx_cv.wait(lk, [this] { return !prx_ops.empty(); });
            m = std::move(prx_ops.front());
            prx_ops.pop_front();
        }
        const uint32_t op = m.first;
        if (op == part::kBye) return "the link closed";
        const double t_op = now_ms();
        part::Reader r(m.second);
        std::string err;
        part::Writer w;
        bool reply = false, ok = true;
        switch (op) {
        case part::kHello: {
            reply = true;
            (void) r.get<uint32_t>();
            const int nl = r.get<int32_t>(), dim = r.get<int32_t>(), vocab = r.get<int32_t>();
            const int64_t ctx = r.get<int64_t>();
            const int ch = r.get<int32_t>();
            (void) r.get<int32_t>();
            if (nl != g.nl || dim != g.dim || vocab != g.vocab) ok = false, err = "another model";
            else if (ctx > o.max_context) ok = false, err = "the front's context " + std::to_string(ctx) + " > " +
                                                         std::to_string(o.max_context);
            else if (ch > chunk) ok = false, err = "the front's prompt chunk " + std::to_string(ch) + " > " +
                                                      std::to_string(chunk);
            if (ok) {
                char b[128];
                std::snprintf(b, sizeof b, "layers %d-%d, chunk %d", part_lo, g.nl - 1, chunk);
                part::Writer h;
                h.put<int32_t>(0);
                h.put_str(b);
                h.put<int32_t>(part_lo);
                pc->send_frame(part::kReply, h.b.data(), h.b.size());
                reply = false;
            }
            break;
        }
        case part::kWin: {
            const int64_t p0 = r.get<int64_t>();
            const int T = r.get<int32_t>(), lr = r.get<int32_t>(), sr = r.get<int32_t>();
            const uint8_t* tk = r.bytes((size_t) std::max(0, T) * 4);
            PartRows pr;
            if (r.bad || tk == nullptr || !part_rows_parse(r, pr)) ok = false, err = "a bad window";
            else if (p0 != pos) ok = false, err = "a window at " + std::to_string(p0) + ", the part at " + std::to_string(pos);
            if (ok && fail.empty()) {
                std::vector<int> t((size_t) T);
                std::memcpy(t.data(), tk, (size_t) T * 4);
                part_cur = &pr;
                ok = forward(t.data(), T, lr, err, sr);
                part_cur = nullptr;
            }
            break;
        }
        case part::kVerify: {
            reply = true;
            const float t = r.get<float>(), tp = r.get<float>();
            const int tk = r.get<int32_t>();
            const float mp = r.get<float>();
            const uint64_t seed = r.get<uint64_t>(), counter = r.get<uint64_t>();
            const int64_t p0 = r.get<int64_t>();
            const int T = r.get<int32_t>();
            (void) r.get<int32_t>();
            (void) r.get<int32_t>();
            const uint8_t* tq = r.bytes((size_t) std::max(0, T) * 4);
            PartRows pr;
            if (r.bad || tq == nullptr || T < 1 || !part_rows_parse(r, pr)) ok = false, err = "a bad verify";
            else if (p0 != pos) ok = false, err = "a verify at " + std::to_string(p0) + ", the part at " + std::to_string(pos);
            if (ok && fail.empty()) {
                std::vector<int> ids((size_t) T);
                std::memcpy(ids.data(), tq, (size_t) T * 4);
                std::vector<int> d(ids.begin() + 1, ids.end());
                int acc = 0, next = 0;
                part_cur = &pr;
                ok = verify(ids[0], d, t, tp, tk, mp, seed, counter, acc, next, err);
                part_cur = nullptr;
                if (ok) {
                    w.put<int32_t>(acc);
                    w.put<int32_t>(next);
                    part_predraft(next, w, err);
                }
            }
            break;
        }
        case part::kSample: {
            reply = true;
            const float t = r.get<float>(), tp = r.get<float>();
            const int tk = r.get<int32_t>();
            const float mp = r.get<float>();
            const uint64_t seed = r.get<uint64_t>(), counter = r.get<uint64_t>();
            if (fail.empty()) {
                const int x = sample(t, tp, tk, mp, seed, counter);
                if (x < 0) ok = false, err = "the sample failed";
                w.put<int32_t>(x);
                part_predraft(x, w, err);
            }
            break;
        }
        case part::kDraft: {
            reply = true;
            const int y = r.get<int32_t>();
            DraftArgs a;
            a.set = true;
            a.t = r.get<float>();
            a.tp = r.get<float>();
            a.tk = r.get<int32_t>();
            a.mp = r.get<float>();
            a.seed = r.get<uint64_t>();
            a.counter = r.get<uint64_t>();
            a.max_n = r.get<int32_t>();
            a.conf_min = r.get<float>();
            a.cum_min = r.get<float>();
            a.min_n = r.get<int32_t>();
            part_dargs = a;
            if (fail.empty()) {
                std::vector<int> d;
                std::vector<float> c;
                ok = draft(y, d, c, err, a.t, a.tp, a.tk, a.mp, a.seed, a.counter, a.max_n, a.conf_min, a.cum_min,
                           a.min_n);
                if (ok) {
                    w.put<int32_t>((int32_t) d.size());
                    w.put<uint8_t>(drafts_q ? 1 : 0);
                    for (int x : d) w.put<int32_t>(x);
                    for (float x : c) w.put<float>(x);
                }
            }
            break;
        }
        case part::kChunks: {
            FeedPlan fp;
            const bool whole = r.get<uint8_t>() != 0;
            fp.at = r.get<int64_t>();
            fp.n = r.get<int32_t>();
            fp.logits_rows = r.get<int32_t>();
            fp.slot = r.get<int32_t>();
            fp.skip = r.get<int32_t>();
            fp.cut_k = r.get<int32_t>();
            fp.cut_r = r.get<int32_t>();
            const uint32_t ne = r.get<uint32_t>();
            for (uint32_t i = 0; i < ne && !r.bad; ++i) fp.ends.push_back(r.get<int32_t>());
            const int k0 = r.get<int32_t>(), k1 = r.get<int32_t>(), n = r.get<int32_t>(), lr = r.get<int32_t>();
            const int snap = r.get<int32_t>(), after = r.get<int32_t>(), ni = r.get<int32_t>();
            std::vector<std::pair<int64_t, bool>> inner;
            for (int i = 0; i < ni && !r.bad; ++i) {
                const int64_t a = r.get<int64_t>();
                inner.push_back({a, r.get<uint8_t>() != 0});
            }
            const int64_t p0 = r.get<int64_t>();
            const uint8_t* tq = r.bytes((size_t) std::max(0, n) * 4);
            bool run_ok = false;
            if (r.bad || tq == nullptr) ok = false, err = "a bad run of chunks";
            else if (p0 != pos) ok = false, err = "a run at " + std::to_string(p0) + ", the part at " + std::to_string(pos);
            if (ok && fail.empty()) {
                std::vector<int> t((size_t) n);
                std::memcpy(t.data(), tq, (size_t) n * 4);
                halted = false;
                fed_n = n;
                if (whole) run_ok = forward_chunks(t.data(), n, lr, err, snap, &fp, k0, k1, after, nullptr);
                else {
                    part_plan_force = &fp;
                    run_ok = forward_chunks(t.data(), n, lr, err, snap, nullptr, 0, 0, after, ni >= 0 ? &inner : nullptr);
                    part_plan_force = nullptr;
                }
                if (!run_ok && !halted) ok = false;
            }
            // the front's end of the run: wait for it (its chunks this run did not take dropped)
            int took = -1, front_ok = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(prx_mu);
                const bool got = prx_cv.wait_for(lk, std::chrono::milliseconds(part_timeout_ms * 6),
                                                 [this] { return !prx_chunks.empty(); });
                if (!got) {
                    if (ok) ok = false, err = "pool: no end of the run from the front";
                    break;
                }
                std::unique_ptr<PartRows> f = std::move(prx_chunks.front());
                prx_chunks.pop_front();
                if (f->k == -1) {
                    took = f->T;
                    front_ok = 1;
                    break;
                }
                if (f->k == -2) return "the link closed";
            }
            (void) took;
            (void) front_ok;
            prx_hold[0].reset();
            prx_hold[1].reset();
            if (!pc->ok()) return "the link closed";
            if (ok && fail.empty()) {
                w.put<int32_t>(run_ok ? 1 : 0);
            }
            reply = true;
            break;
        }
        case part::kReset:
            reset();
            fail.clear();
            break;
        case part::kCheckpoint:
            checkpoint(r.get<int32_t>());
            break;
        case part::kRestore: {
            reply = true;
            const bool b = restore(r.get<int64_t>());
            w.put<int32_t>(b ? 1 : 0);
            w.put<int64_t>(pos);
            break;
        }
        case part::kPin:
            pin();
            break;
        case part::kMark:
            mark(r.get<uint8_t>() != 0);
            break;
        case part::kBegin:
            begin_request();
            part_dargs = DraftArgs();   // (a request's first draft call says whether it drafts)
            break;
        case part::kPenalties: {
            const int n = r.get<int32_t>();
            const float a = r.get<float>(), b = r.get<float>(), c = r.get<float>();
            set_pen(n, a, b, c);
            break;
        }
        case part::kBias: {
            const uint32_t n = r.get<uint32_t>();
            std::vector<std::pair<int, float>> e;
            for (uint32_t i = 0; i < n && !r.bad; ++i) {
                const int id = r.get<int32_t>();
                e.push_back({id, r.get<float>()});
            }
            set_logit_bias(e);
            break;
        }
        case part::kLogprobs:
            break;   // (the front keeps none through the link)
        case part::kSync:
            reply = true;
            break;
        default:
            ok = false;
            err = "an unknown frame " + std::to_string(op);
            reply = true;
        }
        part_busy_ms += now_ms() - t_op;
        part_busy_total += now_ms() - t_op;
        if (!ok && fail.empty()) {
            fail = err.empty() ? "a call failed" : err;
            std::fprintf(stderr, "ds4 pool: %s\n", fail.c_str());
            std::fflush(stderr);
        }
        if (reply && !pc->ok()) return "the link closed";
        if (reply && !part_reply_send(fail.empty() ? 0 : 1, fail, fail.empty() ? &w : nullptr))
            return "the link closed while answering";
    }
}
// the part: the next round's drafts after token y, with the last draft call's arguments, into an answer (n -1: none) -
// the front takes them when its draft() asks the same, a round trip fewer a round
void part_predraft(int y, part::Writer& w, std::string& err) {
    std::vector<int> nd;
    std::vector<float> nc;
    const DraftArgs& a = part_dargs;
    if (a.set && g.n_mtp > 0 && y >= 0 &&
        draft(y, nd, nc, err, a.t, a.tp, a.tk, a.mp, a.seed, a.counter, 0, 0.0f, 0.0f, 0)) {
        w.put<int32_t>((int32_t) nd.size());
        w.put<uint8_t>(drafts_q ? 1 : 0);
        for (int x : nd) w.put<int32_t>(x);
        for (float x : nc) w.put<float>(x);
    } else {
        err.clear();
        w.put<int32_t>(-1);
        w.put<uint8_t>(0);
    }
}
// the front: an answer's drafts for the round after token y
void part_take_pre(part::Reader& r, int y) {
    const int nd = r.get<int32_t>();
    part_pre.have = nd >= 0;
    part_pre.y = y;
    part_pre.q = r.get<uint8_t>() != 0;
    part_pre.d.assign((size_t) std::max(0, nd), 0);
    part_pre.c.assign((size_t) std::max(0, nd), 0.0f);
    for (int i = 0; i < nd; ++i) part_pre.d[(size_t) i] = r.get<int32_t>();
    for (int i = 0; i < nd; ++i) part_pre.c[(size_t) i] = r.get<float>();
    if (r.bad) part_pre.have = false;
}
void set_pen(int last_n, float repeat, float freq, float present) {
    const bool on = last_n > 0 && (repeat != 1.0f || freq != 0.0f || present != 0.0f);
    pen_n = on ? last_n : 0;
    pen_repeat = repeat;
    pen_freq = freq;
    pen_present = present;
}
std::string part_stats() const {
    char b[320];
    std::snprintf(b, sizeof b, "pool: %lld verify rounds, %lld answers, %lld drafts from answers; own layers (verify) %.0f ms, sending %.0f ms, "
                  "waiting for answers %.0f ms (the part busy %.0f ms of it, the link the rest), %.1f MB sent",
                  (long long) part_rounds, (long long) part_waits, (long long) part_pre_hits, part_local_ms, part_net_ms, part_wait_ms,
                  part_remote_ms, part_tx_bytes / 1e6);
    return b;
}
