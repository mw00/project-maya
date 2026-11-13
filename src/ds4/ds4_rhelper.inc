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
        volatile unsigned lead_ready, helper_ready, quit, dead, bits_ready, pad_[3];
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
    float* rh_dout = nullptr;   // (the hot helper: its sums on its GPU, copied out by DMA to rh_hres)
    float* rh_hres = nullptr;
    uint8_t* rh_base = nullptr;
    size_t rh_bytes = 0;
    RhHdr* rh_hdr = nullptr;
    RhJob* rh_jobs = nullptr;
    uint8_t* rh_bits = nullptr;   // [nl][n_exp]: 1 the helper's
    size_t rh_off_flags = 0, rh_off_res = 0, rh_off_reqs = 0;
    std::string rh_name;

    size_t rh_size() const {
        rh_layout();
        return rh_off_reqs + sizeof(Req) * kRing;
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
            ++h;
        }
        std::atomic_thread_fence(std::memory_order_release);
        rh_hdr->head = h;
    }
    void rh_lead_end() {
        if (rh_hdr == nullptr) return;
        std::fprintf(stderr, "ds4: helper engine: the service thread waited %.1f ms for its answers, %lld experts "
                             "theirs, %lld of its own (CPU lane)\n", st.helper_ms, (long long) st.helper_experts,
                     (long long) st.cpu_experts);
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
                if (l.mtp || l.nm != 1 || l.vram_tab == nullptr) continue;
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
        long long jobs = 0;
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
        std::fprintf(stderr, "ds4: helper engine: the lead ended after %lld jobs\n", jobs);
        if (rh_timed > 0)
            std::fprintf(stderr, "ds4: helper engine: a job %.3f ms on its GPU (the sums), %.3f ms between jobs (the next "
                                 "request's wait and take), %lld timed\n", rh_comp_ms / rh_timed, rh_gap_ms / rh_timed,
                         rh_timed);
        cudaStreamSynchronize(helper_st);
        ::shm_unlink(rh_name.c_str());
        return true;
    }
