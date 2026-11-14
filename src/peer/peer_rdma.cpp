#include "peer_rdma.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <poll.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <ctime>

namespace strata::peer {

void say(const std::string& line) {
    timespec ts = {};
    clock_gettime(CLOCK_REALTIME, &ts);
    char stamp[32];
    tm tmv = {};
    localtime_r(&ts.tv_sec, &tmv);
    strftime(stamp, sizeof stamp, "%H:%M:%S", &tmv);
    std::fprintf(stderr, "[peer %s.%03d] %s\n", stamp, (int) (ts.tv_nsec / 1000000), line.c_str());
}

// -------------------------------------------------------------------------------------- Device

Device::~Device() {
    close();
}

/// The ibv device's netdev via sysfs (rocep7s0 -> .../device/net/enp7s0np0)
static std::string netdev_of(const std::string& ibv_name) {
    if (ibv_name.empty()) return "";
    const std::string dir = "/sys/class/infiniband/" + ibv_name + "/device/net";
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return "";
    std::string out;
    dirent* e = nullptr;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') continue;
        out = e->d_name;
        break;
    }
    closedir(d);
    return out;
}

bool Device::open(const std::string& device_name, std::string& err) {
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    if (list == nullptr || n == 0) {
        err = "no RDMA device found (ibv_get_device_list): is rdma-core loaded and a RoCE NIC up?";
        if (list != nullptr) ibv_free_device_list(list);
        return false;
    }
    ibv_device* dev = nullptr;
    for (int i = 0; i < n; ++i) {
        const std::string name = ibv_get_device_name(list[i]);
        if (device_name.empty() ? i == 0 : name == device_name) {
            dev = list[i];
            name_ = name;
            break;
        }
    }
    ibv_free_device_list(list);          // the contexts keep the devices alive past the list
    netdev_ = netdev_of(name_);          // best effort: an unmapped name just leaves this empty
    if (dev == nullptr) {
        err = "no RDMA device named " + (device_name.empty() ? std::string("<first>") : device_name);
        return false;
    }
    ctx_ = ibv_open_device(dev);
    if (ctx_ == nullptr) {
        err = "ibv_open_device " + name_ + " failed";
        return false;
    }
    // port 1: a ConnectX's port 0 of 2, the one the direct link lands on (np0 in the netdev name)
    if (ibv_query_port(ctx_, 1, &port_attr_) != 0 || port_attr_.state != IBV_PORT_ACTIVE) {
        err = "RDMA device " + name_ + " port 1 is not UP (link down? wrong port?)";
        return false;
    }
    port_num_ = 1;
    pd_ = ibv_alloc_pd(ctx_);
    if (pd_ == nullptr) {
        err = "ibv_alloc_pd failed (memlock limit? ulimit -l) on " + name_;
        return false;
    }
    // the GID table: RoCEv2 preferred (the direct link's both ends speak it), v1 as the fall-back.
    // ibv_query_gid_table hands out each entry's type, on rdma-core new enough for it; the manual
    // scan behind it covers an older table the same way by taking the first entry at all.
    bool found = false;
    const size_t want = port_attr_.gid_tbl_len < 256 ? port_attr_.gid_tbl_len : 256;
    std::vector<ibv_gid_entry> entries(want);
    const int got = ibv_query_gid_table(ctx_, entries.data(), want, port_num_);
    int pick_v1 = -1;
    for (int i = 0; i < got; ++i) {
        if (entries[i].gid_type == IBV_GID_TYPE_ROCE_V2) {
            gid_index_ = entries[i].gid_index;
            gid_ = gid_hex(entries[i].gid);
            roce_ = true;
            found = true;
            break;
        }
        if (entries[i].gid_type == IBV_GID_TYPE_ROCE_V1 && pick_v1 < 0) pick_v1 = entries[i].gid_index;
    }
    if (!found && pick_v1 >= 0) {
        ibv_gid g = {};
        ibv_query_gid(ctx_, port_num_, pick_v1, &g);
        gid_index_ = pick_v1;
        gid_ = gid_hex(g);
        roce_ = true;
        found = true;
    }
    if (!found && port_attr_.gid_tbl_len > 0) {          // plain IB (or a table with no RoCE entry)
        ibv_gid g = {};
        ibv_query_gid(ctx_, port_num_, 0, &g);
        gid_index_ = 0;
        gid_ = gid_hex(g);
        found = true;
    }
    if (!found) {
        err = "device " + name_ + " has an empty GID table (driver not fully up?)";
        return false;
    }
    const int cq = 256;                              // the reader sizes its window below this
    cq_ = ibv_create_cq(ctx_, cq, nullptr, nullptr, 0);
    if (cq_ == nullptr) {
        err = "ibv_create_cq failed on " + name_;
        return false;
    }
    say("device " + name_ + " port 1 up, " + std::string(roce_ ? "RoCE" : "IB") + " gid index " +
        std::to_string(gid_index_) + " " + gid_);
    return true;
}

bool Device::close() {
    if (cq_ != nullptr) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
    }
    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
        pd_ = nullptr;
    }
    if (ctx_ != nullptr) {
        ibv_close_device(ctx_);
        ctx_ = nullptr;
    }
    return true;
}

int Device::mtu() const {
    const int active = port_attr_.active_mtu;
    return active < IBV_MTU_4096 ? active : IBV_MTU_4096;
}

// ------------------------------------------------------------------------------------------ Mr

Mr::~Mr() {
    release();
}

bool Mr::reg(ibv_pd* pd, void* buf, size_t bytes, int access, std::string& err) {
    release();
    buf_ = buf;
    bytes_ = bytes;
    owned_ = false;
    mr_ = ibv_reg_mr(pd, buf, bytes, access);
    if (mr_ == nullptr) {
        err = "ibv_reg_mr of " + std::to_string(bytes) + " bytes failed (memlock limit? ulimit -l)";
        buf_ = nullptr;
        bytes_ = 0;
        return false;
    }
    return true;
}

bool Mr::alloc(ibv_pd* pd, size_t bytes, int access, std::string& err) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0) {
        err = "out of memory allocating " + std::to_string(bytes) + " bytes";
        return false;
    }
    if (!reg(pd, p, bytes, access, err)) {
        free(p);
        return false;
    }
    owned_ = true;
    say("mr up: " + std::to_string(bytes >> 20) + " MiB at rkey " + std::to_string(rkey()));
    return true;
}

void Mr::release() {
    if (mr_ != nullptr) ibv_dereg_mr(mr_);
    if (owned_ && buf_ != nullptr) free(buf_);
    mr_ = nullptr;
    buf_ = nullptr;
    bytes_ = 0;
    owned_ = false;
}

// ------------------------------------------------------------------------------------------ Qp

Qp::~Qp() {
    destroy();
}

bool Qp::create(Device& dev, int max_send_wr, int max_recv_wr, std::string& err) {
    destroy();
    ibv_qp_init_attr attr = {};
    attr.qp_type = IBV_QPT_RC;
    attr.send_cq = dev.cq();
    attr.recv_cq = dev.cq();
    attr.sq_sig_all = 1;                 // the posts carry no per-WR flag; every send completes
    attr.cap.max_send_wr = max_send_wr;
    attr.cap.max_recv_wr = max_recv_wr;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    qp_ = ibv_create_qp(dev.pd(), &attr);
    if (qp_ == nullptr) {
        err = std::string("ibv_create_qp failed: ") + strerror(errno);
        return false;
    }
    psn_ = qp_->qp_num & 0xffffff;       // llama.cpp's choice: a deterministic, unique starting PSN
    gid_ = dev.gid();
    return true;
}

bool Qp::arm(Device& dev, uint32_t remote_qpn, uint32_t remote_psn, const std::string& remote_gid,
             std::string& err) {
    if (qp_ == nullptr) {
        err = "the QP was never created";
        return false;
    }
    // INIT: pkey 0 is named explicitly (mlx5 quirk on a direct link) and the access flags carry
    // REMOTE_READ - a responder without it NAKs incoming RDMA READs (the cm's accept path sets
    // REMOTE_WRITE-only, which is why the whole rdma_cm bring-up was dropped)
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = dev.port();
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
            err = std::string("qp INIT failed: ") + strerror(errno);
            return false;
        }
    }
    ibv_gid dgid;
    if (!gid_parse(remote_gid, &dgid)) {
        err = "the peer's GID did not parse: " + remote_gid;
        return false;
    }
    // RTR: the peer's QPN/PSN and GID, the port's own active MTU (both ends of a direct DAC link
    // are raised together, so the port's value is the path's)
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = dev.port_attr().active_mtu;
        a.dest_qp_num = remote_qpn;
        a.rq_psn = remote_psn;
        a.max_dest_rd_atomic = 1;        // one inbound RDMA READ at a time (llama.cpp's value)
        a.min_rnr_timer = 1;
        a.ah_attr.is_global = 1;         // RoCE: the GRH carries the GID, the DLID stays 0
        a.ah_attr.grh.dgid = dgid;
        a.ah_attr.grh.hop_limit = 1;
        a.ah_attr.grh.sgid_index = (uint8_t) dev.gid_index();
        a.ah_attr.dlid = 0;
        a.ah_attr.port_num = dev.port();
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            err = std::string("qp RTR failed: ") + strerror(errno);
            return false;
        }
    }
    // RTS: our own starting PSN; a short transport timeout with hard retries suits a lossy
    // direct-attach link with no link-level flow control
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = psn_;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            err = std::string("qp RTS failed: ") + strerror(errno);
            return false;
        }
    }
    armed_ = true;
    return true;
}

void Qp::destroy() {
    armed_ = false;
    if (qp_ != nullptr) {
        ibv_destroy_qp(qp_);
        qp_ = nullptr;
    }
}

// ----------------------------------------------------------------------------------------- misc

std::string gid_hex(const ibv_gid& g) {
    static const char* hex = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[2 * i] = hex[g.raw[i] >> 4];
        out[2 * i + 1] = hex[g.raw[i] & 15];
    }
    return out;
}

bool gid_parse(const std::string& hex, ibv_gid* out) {
    if (hex.size() != 32) return false;
    auto nib = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    for (int i = 0; i < 16; ++i) {
        const int hi = nib(hex[2 * i]), lo = nib(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out->raw[i] = (uint8_t) ((hi << 4) | lo);
    }
    return true;
}

uint64_t fnv1a(const void* p, size_t bytes) {
    const uint8_t* b = (const uint8_t*) p;
    uint64_t h = 1469598103934665603ULL;             // the 64-bit FNV-1a offset basis
    for (size_t i = 0; i < bytes; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;                       // the FNV prime
    }
    return h;
}

}   // namespace strata::peer
