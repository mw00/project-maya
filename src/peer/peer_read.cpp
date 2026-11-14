#include "peer_read.hpp"

#include <algorithm>
#include <cerrno>
#include <time.h>

#include <cstring>

namespace strata::peer {
namespace {

int64_t now_ms() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

}   // namespace

Reader::Reader(ibv_qp* qp, ibv_cq* cq, int window, size_t chunk)
    : qp_(qp), cq_(cq), window_(window), chunk_(chunk) {
    inflight_.resize((size_t) window * 2 + 8);
}

bool Reader::poll_once(int64_t deadline_ms, std::string& err) {
    ibv_wc wc[8];
    const int n = ibv_poll_cq(cq_, 8, wc);
    if (n < 0) {
        err = "ibv_poll_cq failed";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            err = std::string("read completion failed: ") + ibv_wc_status_str(wc[i].status) +
                  " (opcode " + std::to_string(wc[i].opcode) + ")";
            return false;
        }
        if (wc[i].wr_id >= inflight_.size() || !inflight_[wc[i].wr_id].posted) {
            err = "a completion for an unknown wr_id " + std::to_string(wc[i].wr_id);
            return false;
        }
        inflight_[wc[i].wr_id].posted = false;
        --outstanding_;
    }
    if (n == 0 && now_ms() > deadline_ms) {
        err = "read timed out past the deadline";
        return false;
    }
    return true;
}

bool Reader::read(uint64_t src_addr, uint32_t src_rkey, ibv_mr* dst_mr, uint64_t dst_off, size_t bytes,
                  int timeout_ms, std::string& err) {
    if (bytes == 0) {
        ++rows_;
        return true;
    }
    const int64_t t0 = now_ms();
    const int64_t deadline = t0 + timeout_ms;
    const size_t n_chunks = (bytes + chunk_ - 1) / chunk_;
    // the ring is indexed by chunk (wr_id = chunk): grow it to this row's chunk count - the window
    // bounds what is outstanding at once, the ring only needs an entry per posted chunk
    if (inflight_.size() < n_chunks) inflight_.resize(n_chunks);
    ibv_sge sge = {};                       // the driver copies the WR and its sge at post time, so
    for (size_t c = 0; c < n_chunks; ++c) { // one reusable struct serves every chunk of the row
        // keep the window: first drain completions, then post - the ENOSPC guard
        while (outstanding_ >= window_) {
            if (!poll_once(deadline, err)) return false;
        }
        const size_t at = c * chunk_;
        const size_t len = std::min(chunk_, bytes - at);
        ibv_send_wr wr = {};
        wr.wr_id = c;                       // the ring is sized past any row's chunk count
        wr.opcode = IBV_WR_RDMA_READ;
        sge.addr = (uint64_t) (uintptr_t) dst_mr->addr + dst_off + at;
        sge.length = (uint32_t) len;
        sge.lkey = dst_mr->lkey;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;   // one completion per chunk keeps the ring exact
        wr.wr.rdma.remote_addr = src_addr + at;
        wr.wr.rdma.rkey = src_rkey;
        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp_, &wr, &bad) != 0) {
            err = "ibv_post_send failed (errno " + std::to_string(errno) + ", chunk " +
                  std::to_string(c) + " of " + std::to_string(n_chunks) + ")";
            return false;
        }
        inflight_[c] = {true};
        ++outstanding_;
    }
    // drain the tail
    while (outstanding_ > 0) {
        if (!poll_once(deadline, err)) {
            return false;
        }
    }
    ++rows_;
    bytes_ += bytes;
    us_ += (uint64_t) (now_ms() - t0);
    return true;
}

}   // namespace strata::peer
