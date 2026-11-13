    // ---- MAYA_MIXED_HELPER: a helper in another engine (two GPUs of two vendors on one PC, Linux) ----
    // The lead (DS4_RHELPER=<name>: a CUDA engine holding every layer, its VRAM the most-routed experts) hands each
    // decode window's picks of the experts its VRAM leaves over to a second engine (DS4_RHELPER_SERVE=<name>: a HIP
    // engine holding every expert), as DS4_HELPER's helper GPU takes them in one engine: the same requests (reqs), the
    // same take / sum kernels on the helper's side, the same flags and sums (helper_flags, helper_res) - all in one
    // shared memory block both engines map, and the window's jobs (layer, seq, T) through a ring in it in place of hq.
    // The lead's service thread waits for the helper's flag and adds its sums (helper_answer); no answer in
    // kHelperWaitMs: its CPU lane takes the picks from then on (helper_dead), as with DS4_HELPER.
    struct RhHdr {
        unsigned magic, nl, n_exp, dim;
        volatile unsigned lead_ready, helper_ready, quit, dead, bits_ready, pdone, pad_[2];   // (pdone: a prompt job's seq)
        volatile unsigned long long head, tail;   // jobs written by the lead / taken by the helper
    };
    struct RhJob {
        int il, T;
        unsigned seq, pad;
    };
    static constexpr unsigned kRhMagic = 0x4d524831u;   // "MRH1"
    static constexpr size_t kRhJobs = 4096;
    bool rh_lead = false, rh_serve = false;
    // DS4_RHELPER_HOT=1 (both engines): the helper's own VRAM experts are its share (the lead keeps every expert and
    // leaves those to it in decode) - a lead that holds the whole model, a helper with the fastest memory
    bool rh_hot = std::getenv("DS4_RHELPER_HOT") != nullptr && std::atoi(std::getenv("DS4_RHELPER_HOT")) != 0;
    // DS4_RHELPER_ONLY=1 (the hot helper): its GPU holds experts only - no layer's dense weights, no head - so its
    // VRAM takes about twice the hottest experts (it never runs a layer itself)
    bool rh_only() const {
        static const bool on = std::getenv("DS4_RHELPER_ONLY") != nullptr && std::atoi(std::getenv("DS4_RHELPER_ONLY")) != 0;
        return on && rh_serve && rh_hot;
    }
    float* rh_dout = nullptr;   // (the hot helper: its sums on its GPU, copied out by DMA to rh_hres)
    float* rh_hres = nullptr;
    uint8_t* rh_base = nullptr;
    size_t rh_bytes = 0;
    RhHdr* rh_hdr = nullptr;
    RhJob* rh_jobs = nullptr;
    uint8_t* rh_bits = nullptr;   // [nl][n_exp]: 1 the helper's
    size_t rh_off_flags = 0, rh_off_res = 0, rh_off_reqs = 0;
    std::string rh_name;

    // DS4_RHELPER_PROMPT=1 (the hot helper, both engines): a prompt chunk's experts of the helper's VRAM go to it too -
    // the chunk's rows and picks out, its sums of the experts the lead leaves to it back, added before the layer's end
    static constexpr int kRhPRows = 4096;   // (a chunk of more rows: the lead alone)
    size_t rh_off_px = 0, rh_off_pidx = 0, rh_off_pw = 0, rh_off_ptake = 0, rh_off_pout = 0, rh_off_end = 0;
    size_t rh_size() const {
        rh_layout();
        return rh_off_end;
    }
    void rh_layout() const {
        auto* self = const_cast<Impl*>(this);
        auto al = [](size_t b) { return (b + 4095) & ~(size_t) 4095; };
        size_t off = 4096 + al(sizeof(RhJob) * kRhJobs);
        off += al((size_t) g.nl * (size_t) g.n_exp);
        self->rh_off_flags = off;
        off += 4096;
        self->rh_off_res = off;
        off += al((size_t) kRing * kWin * (size_t) g.dim * 4);
        self->rh_off_reqs = off;
        off += al(sizeof(Req) * kRing);
        const size_t R = (size_t) kRhPRows;
        self->rh_off_px = off;
        off += al(R * (size_t) g.dim * 4);
        self->rh_off_pidx = off;
        off += al(R * kPicks * 4);
        self->rh_off_pw = off;
        off += al(R * kPicks * 4);
        self->rh_off_ptake = off;
        off += al((size_t) g.n_exp);
        self->rh_off_pout = off;
        off += al(R * (size_t) g.dim * 4);
        self->rh_off_end = off;
    }
    bool rh_map(bool make, std::string& err) {
#ifdef _WIN32
        (void) make;
        err = "a helper in another engine: Linux only";
        return false;
#else
        rh_bytes = rh_size();
        if (make) ::shm_unlink(rh_name.c_str());   // (a stale block of an engine that ended)
        int fd = -1;
        const double t0 = now_ms();
        while ((fd = ::shm_open(rh_name.c_str(), make ? O_CREAT | O_EXCL | O_RDWR : O_RDWR, 0600)) < 0) {
            if (make || now_ms() - t0 > 900000.0) {
                err = "a helper in another engine: shm_open " + rh_name + ": errno " + std::to_string(errno);
                return false;
            }
            ::usleep(200000);   // (the lead waits for the helper engine, which makes the block once it is loaded)
        }
        if (make && ::ftruncate(fd, (off_t) rh_bytes) != 0) {
            ::close(fd);
            err = "a helper in another engine: shared memory of " + std::to_string(rh_bytes >> 20) + " MiB";
            return false;
        }
        struct stat sb{};
        if (!make) {   // (the helper sizes it before its magic: wait for both)
            for (;;) {
                if (::fstat(fd, &sb) == 0 && (size_t) sb.st_size >= rh_bytes) break;
                if (now_ms() - t0 > 900000.0) {
                    ::close(fd);
                    err = "a helper in another engine: its block is smaller than this model's";
                    return false;
                }
                ::usleep(100000);
            }
        }
        void* v = ::mmap(nullptr, rh_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
        ::close(fd);
        if (v == MAP_FAILED) {
            err = "a helper in another engine: mmap errno " + std::to_string(errno);
            return false;
        }
        rh_base = (uint8_t*) v;
        rh_hdr = (RhHdr*) rh_base;
        rh_jobs = (RhJob*) (rh_base + 4096);
        rh_bits = rh_base + 4096 + ((sizeof(RhJob) * kRhJobs + 4095) & ~(size_t) 4095);
        if (cudaHostRegister(rh_base, rh_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "a helper in another engine: registering its shared memory with the GPU";
            return false;
        }
        if (make) {
            std::memset(rh_base, 0, rh_off_reqs);
            rh_hdr->nl = (unsigned) g.nl;
            rh_hdr->n_exp = (unsigned) g.n_exp;
            rh_hdr->dim = (unsigned) g.dim;
            std::atomic_thread_fence(std::memory_order_release);
            rh_hdr->magic = kRhMagic;
        } else {
            while (*(volatile unsigned*) &rh_hdr->magic != kRhMagic) {
                if (now_ms() - t0 > 900000.0) {
                    err = "a helper in another engine: no helper";
                    return false;
                }
                ::usleep(100000);
            }
            if (rh_hdr->nl != (unsigned) g.nl || rh_hdr->n_exp != (unsigned) g.n_exp || rh_hdr->dim != (unsigned) g.dim) {
                err = "a helper in another engine: it loaded another model";
                return false;
            }
        }
        return true;
#endif
    }
    template <class P>
    P* rh_dev(size_t off) {   // (the block's device address of offset off, on the current GPU)
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, rh_base + off, 0) != cudaSuccess) return nullptr;
        return (P*) d;
    }

    // (the lead, in place of helper_select) the helper takes every expert of a trunk layer this GPU's VRAM leaves over
    bool rh_lead_select(std::string& err) {
        if (!rh_map(false, err)) return false;
        helper_flags = (unsigned*) (rh_base + rh_off_flags);
        helper_res = (float*) (rh_base + rh_off_res);
        unsigned long long left = 0;
        if (rh_hot) return rh_lead_hot(err);
        for (Layer& l : L) {
            if (l.mtp || l.nm != 1) continue;
            uint8_t* bits = rh_bits + (size_t) l.il * (size_t) g.n_exp;
            for (int e = 0; e < l.n_exp; ++e) {
                if (l.slot[(size_t) e] >= 0) continue;
                if (l.helper.empty()) l.helper.assign((size_t) l.n_exp, nullptr);
                l.helper[(size_t) e] = (uint8_t*) rh_base;   // (a marker: the helper's, its blob in the other engine)
                bits[e] = 1;
                ++helper_n;
                ++left;
            }
            if (!l.helper.empty()) l.helper_tab = (unsigned long long*) rh_base;   // (a marker: the layer has its jobs)
        }
        st.helper_held = helper_n;
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->lead_ready = 1;
        const double t0 = now_ms();
        while (rh_hdr->helper_ready == 0) {
            if (rh_hdr->dead != 0 || now_ms() - t0 > 600000.0) {
                err = "a helper in another engine: it did not get ready";
                return false;
            }
            ::usleep(20000);
        }
        std::fprintf(stderr, "ds4: helper engine %s: it computes %llu experts this GPU's VRAM leaves over (every "
                             "trunk layer's), in %.0f ms ready\n", rh_name.c_str(), left, now_ms() - t0);
        return true;
    }
    Req* rh_dreq = nullptr;     // (the lead: this GPU's mirror of the request a layer publishes, copied out by DMA)
    Resp* rh_dresp = nullptr;   // (... and of its answer, copied in by DMA)
    bool rh_req_out(int slot, int T, cudaStream_t s, std::string& err) {   // the body, then the rows, then seq: seq lands last
        Req* h = reqs + slot;
        const size_t x0 = offsetof(Req, x);
        CK(cudaMemcpyAsync((uint8_t*) h + 4, (const uint8_t*) rh_dreq + 4, x0 - 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpy2DAsync(&h->x[0][0], kMaxDim * 4, &rh_dreq->x[0][0], kMaxDim * 4, (size_t) g.dim * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(&h->seq, &rh_dreq->seq, 4, cudaMemcpyDeviceToHost, s));
        return true;
    }
    bool rh_resp_in(int slot, int T, cudaStream_t s, std::string& err) {   // (after the wait for its seq) the flags, the rows
        const Resp* h = resps + slot;
        CK(cudaMemcpyAsync(rh_dresp->cpu, h->cpu, sizeof(h->cpu) + sizeof(h->pmask), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpy2DAsync(&rh_dresp->part[0][0], kMaxDim * 4, &h->part[0][0], kMaxDim * 4, (size_t) g.dim * 4,
                             (size_t) T, cudaMemcpyHostToDevice, s));
        return true;
    }
    // DS4_RHELPER_DIRECT (default on; 0 off): on a layer the helper takes every non-VRAM expert of, the lead's GPU
    // waits on the helper's own answered flag and copies its sums in, without the service thread's hop
    bool rh_direct = true;
    int* rh_ones = nullptr;   // (device: kWin ones - every token's part counted; a token with no helper pick has 0s)
    bool rh_direct_on(const Layer& l) const {
        return rh_direct && l.helper_tab != nullptr && !helper_dead.load(std::memory_order_relaxed);
    }
    // DS4_RHELPER_LATE=1 (a hot helper's lead, verify windows): the helper's sums read in place by the experts' combine,
    // which waits for its flag on this GPU after this GPU's own experts (not before them) - the same sums, in the
    // same order; the lead's memory must reach the block at full speed (an integrated GPU's host memory)
    // (decode's routing) the helper's experts left out of this GPU's table while it answers; gone: the whole table again
    unsigned long long* rh_vtab(const Layer& l) const {
        return l.rh_tab != nullptr && !helper_dead.load(std::memory_order_relaxed) ? l.rh_tab : l.vram_tab;
    }
    bool rh_late_on() const {
        static const bool on = std::getenv("DS4_RHELPER_LATE") != nullptr && std::atoi(std::getenv("DS4_RHELPER_LATE")) != 0;
        return on && rh_hot;
    }
    bool rh_help_in(int slot, unsigned sq, int T, cudaStream_t s, std::string& err) {
        wait_seq(helper_flags + kRing + slot, sq, s);
        const float* y = helper_res + (size_t) slot * kWin * (size_t) g.dim;
        CK(cudaMemcpy2DAsync(&rh_dresp->part[0][0], kMaxDim * 4, y, (size_t) g.dim * 4, (size_t) g.dim * 4, (size_t) T,
                             cudaMemcpyHostToDevice, s));
        return true;
    }
    // (the service thread: the helper gave no answer, the CPU lane computed its picks into rp) - its sums where the
    // helper's go, then its flag, so a GPU that waits on that flag goes on
    void rh_fallback_flag(const Resp& rp, int slot, unsigned seq, int T) {
        const size_t D = (size_t) g.dim;
        for (int t = 0; t < T; ++t) {
            float* y = helper_res + ((size_t) slot * kWin + (size_t) t) * D;
            if (rp.cpu[t]) std::memcpy(y, rp.part[t], D * 4);
            else std::memset(y, 0, D * 4);
        }
        std::atomic_thread_fence(std::memory_order_release);
        *(volatile unsigned*) (helper_flags + kRing + slot) = seq;
    }
    bool rh_lead_hot(std::string& err) {   // (DS4_RHELPER_HOT) the helper's VRAM experts, left to it in decode
        const double t0 = now_ms();
        while (rh_hdr->bits_ready == 0) {
            if (rh_hdr->dead != 0 || now_ms() - t0 > 600000.0) {
                err = "a helper in another engine: no list of its experts";
                return false;
            }
            ::usleep(20000);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        long long took = 0;
        for (Layer& l : L) {
            if (l.mtp || l.nm != 1 || l.vram_tab == nullptr) continue;
            const uint8_t* bits = rh_bits + (size_t) l.il * (size_t) g.n_exp;
            std::vector<unsigned long long> tb((size_t) l.n_exp, 0ull);
            if (cudaMemcpy(tb.data(), l.vram_tab, tb.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "a helper in another engine: layer " + std::to_string(l.il) + "'s VRAM table";
                return false;
            }
            int n = 0;
            for (int e = 0; e < l.n_exp; ++e) {
                if (!bits[e] || l.slot[(size_t) e] < 0) continue;
                if (l.helper.empty()) l.helper.assign((size_t) l.n_exp, nullptr);
                l.helper[(size_t) e] = (uint8_t*) rh_base;
                tb[(size_t) e] = 0ull;
                ++n;
            }
            if (n == 0) continue;
            void* p = nullptr;
            if (cudaMalloc(&p, tb.size() * 8) != cudaSuccess ||
                cudaMemcpy(p, tb.data(), tb.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "a helper in another engine: a decode table";
                return false;
            }
            devs[(size_t) l.dev].allocs.push_back(p);
            l.rh_tab = (unsigned long long*) p;
            l.helper_tab = (unsigned long long*) rh_base;
            helper_n += n;
            took += n;
        }
        st.helper_held = helper_n;
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->lead_ready = 1;
        while (rh_hdr->helper_ready == 0) {
            if (rh_hdr->dead != 0 || now_ms() - t0 > 600000.0) {
                err = "a helper in another engine: it did not get ready";
                return false;
            }
            ::usleep(20000);
        }
        std::fprintf(stderr, "ds4: helper engine %s: it computes %lld of this GPU's experts in decode (its own VRAM's; "
                             "DS4_RHELPER_HOT)\n", rh_name.c_str(), took);
        return true;
    }
    bool rh_reqs(std::string& err) {   // (the lead: the requests in the block, where the helper's takes read them)
        if (devs.size() != 1 || lane_par) {
            err = "a helper in another engine: a lead of one GPU (and no MAYA_DS4_LANE_PARITY)";
            return false;
        }
        cudaSetDevice(devs[0].id);
        if (cudaMalloc((void**) &rh_dreq, sizeof(Req)) != cudaSuccess ||
            cudaMalloc((void**) &rh_dresp, sizeof(Resp)) != cudaSuccess ||
            cudaMemset(rh_dreq, 0, sizeof(Req)) != cudaSuccess || cudaMemset(rh_dresp, 0, sizeof(Resp)) != cudaSuccess) {
            err = "a helper in another engine: its request mirrors";
            return false;
        }
        {
            const int ones[kWin] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            if (cudaMalloc((void**) &rh_ones, sizeof ones) != cudaSuccess ||
                cudaMemcpy(rh_ones, ones, sizeof ones, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "a helper in another engine: its flags";
                return false;
            }
            devs[0].allocs.push_back(rh_ones);
        }
        if (const char* v = std::getenv("DS4_RHELPER_DIRECT")) rh_direct = std::atoi(v) != 0;
        if (rh_cmp != 0) rh_direct = false;   // (DS4_RHELPER_CMP: the service thread reads every answer)
        if (rh_hot && !rh_p_alloc(err)) return false;
        devs[0].allocs.push_back(rh_dreq);
        devs[0].allocs.push_back(rh_dresp);
        reqs = (Req*) (rh_base + rh_off_reqs);
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, reqs, 0) != cudaSuccess || d != (void*) reqs) {
            err = "a helper in another engine: the requests' device address is not the host's";
            return false;
        }
        return true;
    }
    void rh_push(int T, const WinPlan& wp) {
        unsigned long long h = rh_hdr->head;
        for (int il = 0; il < g.nl; ++il) {
            if (L[(size_t) il].helper_tab == nullptr) continue;
            while (h - rh_hdr->tail >= kRhJobs) cpu_relax();
            RhJob& j = rh_jobs[h % kRhJobs];
            j.il = il;
            j.T = T;
            j.seq = wp.seq[il];
            j.pad = 0;   // (a decode job)
            ++h;
        }
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->head = h;
    }
    // ---- DS4_RHELPER_PROMPT: a prompt chunk's helper experts (the lead's side) ----
    bool rh_pon = std::getenv("DS4_RHELPER_PROMPT") != nullptr && std::atoi(std::getenv("DS4_RHELPER_PROMPT")) != 0;
    bool rh_pdead = false;
    unsigned rh_pseq = 0;
    int* rh_ppos = nullptr;     // (device: row t's one "pick" is row t - pout added to acc through moe_combine)
    float* rh_pone = nullptr;   // (device: weight 1)
    double rh_pwait_ms = 0, rh_phelp_ms = 0;
    long long rh_pjobs = 0, rh_prows = 0, rh_pall = 0, rh_plate = 0;
    // DS4_RHELPER_PSHARE (0.45): the share of a chunk's routed rows the helper takes at most (its experts, busiest
    // first); DS4_RHELPER_PMIN (512): fewer rows a chunk - the lead alone
    double rh_pshare = std::getenv("DS4_RHELPER_PSHARE") ? std::atof(std::getenv("DS4_RHELPER_PSHARE")) : 0.45;
    int rh_pmin = std::getenv("DS4_RHELPER_PMIN") ? std::atoi(std::getenv("DS4_RHELPER_PMIN")) : 512;
    bool rh_p_ok(const Layer& L0, const Chunk& c, int T) const {
        static std::atomic<bool> said{false};
        if (rh_pon && L0.il == 10 && T >= 512 && !said.exchange(true))
            std::fprintf(stderr, "ds4: helper engine: a prompt chunk (layer %d, %d rows): lead %d hot %d dead %d rows %d "
                                 "tab %d own %zu members %d acc64 %d host %zu helper_dead %d\n", L0.il, T, (int) rh_lead,
                         (int) rh_hot, (int) rh_pdead, (int) (rh_ppos != nullptr), (int) (L0.helper_tab != nullptr),
                         L0.helper.size(), L0.nm, (int) (c.acc64 != nullptr), L0.host_list.size(),
                         (int) helper_dead.load());
        return rh_lead && rh_hot && rh_pon && !rh_pdead && rh_ppos != nullptr && L0.helper_tab != nullptr &&
               !L0.helper.empty() && L0.nm == 1 && !(c.acc64 != nullptr && chunk_par()) && (c.flow || L0.host_list.empty()) &&
               T >= rh_pmin && T <= kRhPRows && !helper_dead.load(std::memory_order_relaxed);
    }
    bool rh_p_alloc(std::string& err) {   // (the lead, rh_reqs: its add's rows and weights)
        if (!rh_pon) return true;
        std::vector<int> p((size_t) kRhPRows * kPicks, -1);
        std::vector<float> w((size_t) kRhPRows * kPicks, 0.0f);
        for (int t = 0; t < kRhPRows; ++t) p[(size_t) t * kPicks] = t, w[(size_t) t * kPicks] = 1.0f;
        if (cudaMalloc((void**) &rh_ppos, p.size() * 4) != cudaSuccess ||
            cudaMalloc((void**) &rh_pone, w.size() * 4) != cudaSuccess ||
            cudaMemcpy(rh_ppos, p.data(), p.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(rh_pone, w.data(), w.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "a helper in another engine: its prompt rows";
            return false;
        }
        devs[0].allocs.push_back(rh_ppos);
        devs[0].allocs.push_back(rh_pone);
        std::fprintf(stderr, "ds4: helper engine: prompt chunks of %d-%d rows give it its experts (at most %.0f %% of "
                             "the rows; DS4_RHELPER_PROMPT)\n", rh_pmin, kRhPRows, rh_pshare * 100.0);
        return true;
    }
    // (before the routing's sync) the chunk's rows and picks into the block
    bool rh_p_out(const Chunk& c, int T, cudaStream_t s, std::string& err) {
        CK(cudaMemcpyAsync(rh_base + rh_off_px, c.x, (size_t) T * g.dim * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(rh_base + rh_off_pidx, c.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(rh_base + rh_off_pw, c.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
        return true;
    }
    // (after it, the bounds b known) the helper's experts, busiest first, up to its share; vt[v] = 1: its slot v
    unsigned rh_p_push(const Layer& L0, int T, const std::vector<int>& b, std::vector<char>& vt) {
        const int ne = L0.n_exp;
        vt.assign(b.size(), 0);
        std::vector<std::pair<int, int>> own;   // (rows, expert)
        for (int e = 0; e < ne; ++e)
            if (L0.helper[(size_t) e] != nullptr && L0.vs_pin[e] < L0.pv_n) {   // (a streamed one: the lead's flow)
                const int v = L0.vs_pin[e];
                own.push_back({b[(size_t) v + 1] - b[(size_t) v], e});
            }
        std::sort(own.begin(), own.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& c) {
            return a.first != c.first ? a.first > c.first : a.second < c.second;
        });
        uint8_t* take = rh_base + rh_off_ptake;
        std::memset(take, 0, (size_t) ne);
        const long long cap = (long long) (rh_pshare * (double) T * (double) L0.k_exp);
        long long rows = 0;
        for (const auto& o : own) {
            if (o.first == 0 || rows + o.first > cap) continue;
            rows += o.first;
            take[o.second] = 1;
            vt[(size_t) L0.vs_pin[o.second]] = 1;
        }
        rh_prows += rows;
        rh_pall += (long long) T * L0.k_exp;
        const unsigned sq = ++rh_pseq;
        unsigned long long h = rh_hdr->head;
        while (h - rh_hdr->tail >= kRhJobs) cpu_relax();
        RhJob& j = rh_jobs[h % kRhJobs];
        j.il = L0.il;
        j.T = T;
        j.seq = sq;
        j.pad = 1;   // (a prompt job)
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->head = h + 1;
        ++rh_pjobs;
        return sq;
    }
    // (the lead, right after the routing's sort) the rows, picks and bounds out; the job pushed once they landed - the
    // shared expert still running here
    cudaEvent_t rh_pev = nullptr;
    bool rh_p_early(const Chunk& c, const Layer& L0, int T, int n_vs, cudaStream_t s, std::vector<char>& vt,
                    unsigned& sq, std::string& err) {
        if (!rh_p_out(c, T, s, err)) return false;
        if (rh_pev == nullptr) CK(cudaEventCreateWithFlags(&rh_pev, cudaEventDisableTiming));
        CK(cudaEventRecord(rh_pev, s));
        CK(cudaEventSynchronize(rh_pev));
        const std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
        sq = rh_p_push(L0, T, b, vt);
        return true;
    }
    // the lead's own experts of the chunk: the runs of slots the helper does not take
    bool rh_p_lead(Dev& d, Chunk& c, const Layer& L0, const Layer& l, const std::vector<int>& b,
                   const std::vector<char>& vt, int v0, int n, int T, std::string& err) {
        int v = v0;
        while (v < n) {
            while (v < n && vt[(size_t) v]) ++v;
            int w = v;
            while (w < n && !vt[(size_t) w]) ++w;
            if (w > v && !run_vram(d, c, L0, l, l, b.data(), v, w, T, err)) return false;
            v = w;
        }
        return true;
    }
    // its sums waited for (the host: no answer in kHelperWaitMs - the lead computes those experts itself from then on)
    bool rh_p_join(Dev& d, Chunk& c, const Layer& L0, const Layer& l, const std::vector<int>& b,
                   const std::vector<char>& vt, unsigned sq, int T, std::string& err) {
        const double t0 = now_ms();
        bool late = false;
        while ((int) (*(volatile unsigned*) &rh_hdr->pdone - sq) < 0) {
            if (!late && cudaStreamQuery(d.st) == cudaSuccess) late = true;   // (this GPU done first: the helper late)
            if (rh_hdr->dead != 0 || now_ms() - t0 > (double) kHelperWaitMs) {
                std::fprintf(stderr, "ds4: helper engine: no answer to a prompt chunk in %.0f ms - prompts on this GPU "
                                     "alone from now on\n", now_ms() - t0);
                rh_pdead = true;
                std::vector<char> rest(vt.size(), 0);
                for (size_t i = 0; i < vt.size(); ++i) rest[i] = !vt[i];
                return rh_p_lead(d, c, L0, l, b, rest, 0, L0.pv_n, T, err);
            }
            cpu_relax();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        rh_pwait_ms += now_ms() - t0;
        rh_plate += late;
        float* y = rh_dev<float>(rh_off_pout);
        if (y == nullptr) {
            err = "a helper in another engine: its prompt sums' device address";
            return false;
        }
        moe_combine(c.acc, y, rh_ppos, rh_pone, T, 1, 0, T, g.dim, d.st);
        return true;
    }
    // ---- (the helper's side) a prompt job: its experts the lead left to it, over the chunk's rows ----
    std::vector<int> rh_pvs_h;
    int* rh_pvs = nullptr;
    bool rh_p_serve(const RhJob& j, std::string& err) {
        Layer& l = L[(size_t) j.il];
        Dev& d = devs[0];
        if (chunks.empty() || chunks[0] == nullptr || j.T > chunks[0]->C) {
            err = "DS4_RHELPER_SERVE: a prompt job of " + std::to_string(j.T) + " rows (its chunks: --prefill-chunk " +
                  std::to_string(chunks.empty() || chunks[0] == nullptr ? 0 : chunks[0]->C) + ")";
            return false;
        }
        Chunk& c = *chunks[0];
        cudaStream_t s = d.st;
        const int T = j.T, D = g.dim, ne = l.n_exp, nv = l.pv_n + 1;
        if (rh_pvs == nullptr && cudaMalloc((void**) &rh_pvs, (size_t) ne * 4) != cudaSuccess) {
            err = "DS4_RHELPER_SERVE: its prompt slots";
            return false;
        }
        const uint8_t* take = rh_base + rh_off_ptake;
        rh_pvs_h.assign((size_t) ne, l.pv_n);   // (not its: the last slot, never run)
        for (int e = 0; e < ne; ++e)
            if (take[e] && l.slot[(size_t) e] >= 0 && l.slot[(size_t) e] < l.pv_n) rh_pvs_h[(size_t) e] = l.slot[(size_t) e];
        const double t0 = now_ms();
        CK(cudaMemcpyAsync(rh_pvs, rh_pvs_h.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, s));
        // DS4_RHELPER_PPIECE=N: the rows in pieces of N, each piece's copy in (stream ci) and its sums out (co) beside
        // the compute of the others (the same sums: a row's experts are summed in one piece in slot order)
        static const int piece = std::getenv("DS4_RHELPER_PPIECE") ? std::max(0, std::atoi(std::getenv("DS4_RHELPER_PPIECE"))) : 0;
        const int P = piece > 0 ? piece : T;
        if (rh_pci == nullptr) {
            CK(cudaStreamCreateWithFlags(&rh_pci, cudaStreamNonBlocking));
            CK(cudaStreamCreateWithFlags(&rh_pco, cudaStreamNonBlocking));
            for (auto& e : rh_pev_in) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            for (auto& e : rh_pev_done) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        const int np = (T + P - 1) / P;
        if (np > (int) rh_pev_in.size()) {
            err = "DS4_RHELPER_PPIECE: more than " + std::to_string(rh_pev_in.size()) + " pieces a chunk";
            return false;
        }
        for (int p = 0; p < np; ++p) {   // (every piece's rows in at once, in order: the link streams them)
            const size_t r0 = (size_t) p * P, n = (size_t) std::min(P, T - (int) r0);
            CK(cudaMemcpyAsync(c.x + r0 * D, rh_base + rh_off_px + r0 * D * 4, n * D * 4, cudaMemcpyHostToDevice, rh_pci));
            CK(cudaMemcpyAsync(c.ridx + r0 * kPicks, rh_base + rh_off_pidx + r0 * kPicks * 4, n * kPicks * 4,
                               cudaMemcpyHostToDevice, rh_pci));
            CK(cudaMemcpyAsync(c.rw + r0 * kPicks, rh_base + rh_off_pw + r0 * kPicks * 4, n * kPicks * 4,
                               cudaMemcpyHostToDevice, rh_pci));
            CK(cudaEventRecord(rh_pev_in[(size_t) p], rh_pci));
        }
        CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * 4, s));
        float *x0 = c.x, *a0 = c.acc, *w0 = c.rw;
        int* i0 = c.ridx;
        bool ok = true;
        for (int p = 0; p < np && ok; ++p) {
            const size_t r0 = (size_t) p * P;
            const int n = std::min(P, T - (int) r0);
            CK(cudaStreamWaitEvent(s, rh_pev_in[(size_t) p], 0));
            c.x = x0 + r0 * D, c.acc = a0 + r0 * D, c.rw = w0 + r0 * kPicks, c.ridx = i0 + r0 * kPicks;
            x3::rows_f16(c.x, D, nullptr, n, D, c.x16, s);
            moe_sort(c.ridx, n, l.k_exp, rh_pvs, nv, c.bounds, c.row_tok, c.pos, s);
            ok = cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (nv + 1) * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
                 cudaStreamSynchronize(s) == cudaSuccess;
            std::vector<int> b(c.bounds_pin, c.bounds_pin + nv + 1);
            ok = ok && run_vram(d, c, l, l, l, b.data(), 0, l.pv_n, n, err) &&
                 cudaEventRecord(rh_pev_done[(size_t) p], s) == cudaSuccess &&
                 cudaStreamWaitEvent(rh_pco, rh_pev_done[(size_t) p], 0) == cudaSuccess &&
                 cudaMemcpyAsync(rh_base + rh_off_pout + r0 * D * 4, c.acc, (size_t) n * D * 4, cudaMemcpyDeviceToHost,
                                 rh_pco) == cudaSuccess;
            c.x = x0, c.acc = a0, c.rw = w0, c.ridx = i0;
        }
        if (!ok || cudaStreamSynchronize(rh_pco) != cudaSuccess) {
            if (err.empty()) err = "DS4_RHELPER_SERVE: a prompt job's copies or kernels";
            return false;
        }
        rh_pserve_ms += now_ms() - t0;
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->pdone = j.seq;
        return true;
    }
    cudaStream_t rh_pci = nullptr, rh_pco = nullptr;
    std::array<cudaEvent_t, 16> rh_pev_in{}, rh_pev_done{};
    double rh_pserve_ms = 0;
    void rh_lead_end() {
        if (rh_hdr == nullptr) return;
        helper_cmp_end();
        std::fprintf(stderr, "ds4: helper engine: the service thread waited %.1f ms for its answers, %lld experts "
                             "theirs, %lld of its own (CPU lane)\n", st.helper_ms, (long long) st.helper_experts,
                     (long long) st.cpu_experts);
        if (rh_pjobs > 0)
            std::fprintf(stderr, "ds4: helper engine: %lld prompt jobs, it took %.1f %% of their routed rows, the lead "
                                 "waited %.1f ms for it (%lld chunks its sums came after this GPU's own)\n", rh_pjobs,
                         rh_pall > 0 ? 100.0 * (double) rh_prows / (double) rh_pall : 0.0, rh_pwait_ms, rh_plate);
        rh_hdr->quit = 1;
        if (helper_flags != nullptr) *(volatile unsigned*) (helper_flags + 2 * kRing) = 1u;
    }

    // (the helper engine, DS4_RHELPER_TIME) its GPU's time a job: after the take (the request in), after the sums
    std::vector<cudaEvent_t> rh_ev;
    size_t rh_evn = 0;
    double rh_comp_ms = 0, rh_gap_ms = 0;
    long long rh_timed = 0;
    void rh_mark(int) {
        if (rh_ev.empty() || rh_evn + 1 >= rh_ev.size()) return;   // (full: the rest of this window untimed)
        cudaEventRecord(rh_ev[rh_evn++], helper_st);
    }
    void rh_ev_sum() {   // (idle: the window's events read)
        if (rh_evn >= 2) {
            cudaStreamSynchronize(helper_st);
            for (size_t i = 0; i + 1 < rh_evn; i += 2) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, rh_ev[i], rh_ev[i + 1]) == cudaSuccess) rh_comp_ms += ms, ++rh_timed;
                if (i >= 2 && cudaEventElapsedTime(&ms, rh_ev[i - 1], rh_ev[i]) == cudaSuccess) rh_gap_ms += ms;
            }
        }
        rh_evn = 0;
    }
    // (the helper engine, after its load: every expert in its VRAM) answers one lead until it ends
    bool rhelper_serve(std::string& err) {
        if (devs.size() != 1) {
            err = "DS4_RHELPER_SERVE: one GPU";
            return false;
        }
        cudaSetDevice(devs[0].id);
        if (!rh_map(true, err)) return false;
        helper_id = devs[0].id;
        bool ok = cudaStreamCreateWithFlags(&helper_st, cudaStreamNonBlocking) == cudaSuccess &&
                  (pk.gguf ? gg::prepare(helper_st) : x3::prepare(helper_st));
        auto dev = [&](size_t b) -> void* {
            void* p = nullptr;
            if (!ok || cudaMalloc(&p, align256(b)) != cudaSuccess) {
                ok = false;
                return nullptr;
            }
            helper_allocs.push_back(p);
            return p;
        };
        const size_t D = (size_t) g.dim, FF = (size_t) g.ff, WE = (size_t) kWin * kPicks;
        helper_plan = (unsigned long long*) dev(WE * 8);
        helper_w = (float*) dev(WE * 4);
        helper_x = (float*) dev((size_t) kWin * D * 4);
        helper_h = (float*) dev((size_t) kPicks * FF * 4);
        ok = ok && cudaMemset(helper_h, 0, (size_t) kPicks * FF * 4) == cudaSuccess;
        helper_hw = (float*) dev(WE * FF * 4);
        helper_part = (float*) dev(WE * D * 4);
        helper_grp_ptr = (unsigned long long*) dev(WE * 8);
        helper_grp_start = (int*) dev((WE + 1) * 4);
        helper_grp_ent = (int*) dev(WE * 4);
        helper_n_groups = (int*) dev(4);
        // (its kernels go by the block's device addresses)
        reqs = rh_dev<Req>(rh_off_reqs);
        helper_flags = rh_dev<unsigned>(rh_off_flags);
        helper_res = rh_dev<float>(rh_off_res);
        if (!ok || reqs == nullptr || helper_flags == nullptr || helper_res == nullptr) {
            err = "DS4_RHELPER_SERVE: its scratch or the block's device addresses";
            rh_hdr->dead = 1;
            return false;
        }
        if (rh_hot) {   // (its own VRAM experts are its share: listed for the lead)
            rh_hres = (float*) (rh_base + rh_off_res);
            rh_dout = (float*) dev((size_t) kWin * D * 4);
            for (Layer& l : L) {
                if (l.mtp || l.nm != 1 || l.vram_tab == nullptr) {
                    if (l.n_vram > 0)
                        std::fprintf(stderr, "ds4: helper engine: layer %d's %d VRAM experts not the lead's (drafter %d, "
                                             "members %d)\n", l.il, l.n_vram, (int) l.mtp, l.nm);
                    continue;
                }
                std::vector<unsigned long long> tb((size_t) l.n_exp, 0ull);
                if (cudaMemcpy(tb.data(), l.vram_tab, tb.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
                uint8_t* bits = rh_bits + (size_t) l.il * (size_t) g.n_exp;
                for (int e = 0; e < l.n_exp; ++e) bits[e] = tb[(size_t) e] != 0ull;
            }
            if (!ok) {
                err = "DS4_RHELPER_SERVE: its VRAM tables";
                rh_hdr->dead = 1;
                return false;
            }
            std::atomic_thread_fence(std::memory_order_release);
            rh_hdr->bits_ready = 1;
        }
        std::fprintf(stderr, "ds4: helper engine %s: %.1f MiB block, waiting for a lead\n", rh_name.c_str(),
                     rh_bytes / 1048576.0);
        while (rh_hdr->lead_ready == 0) {
            if (rh_hdr->quit) return true;
            ::usleep(20000);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        long long took = 0;
        for (Layer& l : L) {   // each layer's table: its own VRAM experts the lead left to it, 0 the rest
            const uint8_t* bits = rh_bits + (size_t) l.il * (size_t) g.n_exp;
            bool any = false;
            for (int e = 0; e < l.n_exp && !any; ++e) any = bits[e] != 0;
            if (!any || l.mtp) continue;
            std::vector<unsigned long long> t((size_t) l.n_exp, 0ull);
            if (l.vram_tab == nullptr ||
                cudaMemcpy(t.data(), l.vram_tab, t.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "DS4_RHELPER_SERVE: layer " + std::to_string(l.il) + "'s VRAM table";
                rh_hdr->dead = 1;
                return false;
            }
            for (int e = 0; e < l.n_exp; ++e) {
                if (!bits[e]) t[(size_t) e] = 0ull;
                else if (t[(size_t) e] == 0ull) {
                    err = "DS4_RHELPER_SERVE: layer " + std::to_string(l.il) + " expert " + std::to_string(e) +
                          " is not in this GPU's VRAM";
                    rh_hdr->dead = 1;
                    return false;
                }
                took += bits[e] != 0;
            }
            void* p = dev(t.size() * 8);
            if (!ok || cudaMemcpy(p, t.data(), t.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "DS4_RHELPER_SERVE: a layer's table";
                rh_hdr->dead = 1;
                return false;
            }
            l.helper_tab = (unsigned long long*) p;
        }
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->helper_ready = 1;
        std::fprintf(stderr, "ds4: helper engine: %lld experts for the lead\n", took);
        if (std::getenv("DS4_RHELPER_TIME") != nullptr) {
            rh_ev.resize(1024);
            for (auto& e : rh_ev) cudaEventCreate(&e);
        }
        unsigned long long t = rh_hdr->tail;
        double idle0 = now_ms();
        long long jobs = 0, pjobs = 0;
        while (!rh_hdr->quit) {
            const unsigned long long h = rh_hdr->head;
            if (t == h) {
                if (now_ms() - idle0 > 2.0) {
                    if (rh_evn > 0) rh_ev_sum();
                    ::usleep(20);
                } else cpu_relax();
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            for (; t < h; ++t) {
                const RhJob j = rh_jobs[t % kRhJobs];
                if (j.il < 0 || j.il >= g.nl || L[(size_t) j.il].helper_tab == nullptr) continue;
                if (j.pad == 1) {   // (a prompt chunk's experts: answered before the next job)
                    if (!rh_p_serve(j, err)) {
                        rh_hdr->dead = 1;
                        return false;
                    }
                    ++pjobs;
                    continue;
                }
                helper_enqueue(L[(size_t) j.il], (int) (j.seq % kRing), j.seq, j.T);
                ++jobs;
            }
            rh_hdr->tail = t;
            idle0 = now_ms();
            if (helper_dead.load()) {
                rh_hdr->dead = 1;
                err = "DS4_RHELPER_SERVE: a launch failed";
                return false;
            }
        }
        std::fprintf(stderr, "ds4: helper engine: the lead ended after %lld jobs (%lld prompt chunks', %.1f ms a chunk)\n", jobs, pjobs,
                     pjobs > 0 ? rh_pserve_ms / (double) pjobs : 0.0);
        if (rh_timed > 0)
            std::fprintf(stderr, "ds4: helper engine: a job %.3f ms on its GPU (the sums), %.3f ms between jobs (the next "
                                 "request's wait and take), %lld timed\n", rh_comp_ms / rh_timed, rh_gap_ms / rh_timed,
                         rh_timed);
        cudaStreamSynchronize(helper_st);
        ::shm_unlink(rh_name.c_str());
        return true;
    }
