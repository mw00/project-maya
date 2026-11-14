// strata_peer's verbs primitives: a device/PD/CQ holder, a registered slab, and an RC QP brought up
// RESET -> INIT -> RTR -> RTS for RoCEv2.  Kept free of the GPU runtimes on purpose: the peer daemon
// runs beside (not inside) an engine and must not pull strata_core's HIP dependency.
//
// The transport is RDMA READ (the initiator pulls; the responder runs no software per row): the
// initiator's destination buffer needs LOCAL_WRITE, the responder's slab REMOTE_READ (both plus the
// local-access bits ibv_reg_mr implies).  Chunking, the outstanding window and the completion loop
// live in peer_read; this file only knows QPs, memory and GIDs.
//
// RoCE lessons baked in (from the pair's own RDMA spike, see the rust-rdma-roce-bindings skill):
// the QP endpoints travel as qpn + 16-byte GID over the TCP control channel (newline-delimited JSON,
// explicitly flushed - a buffered writer deadlocks a full-duplex handshake), and posted work is
// bounded by the completion loop (posting faster than completions end in ENOSPC on ibv_post_send).
#pragma once

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::peer {

/// "peer: ..." - the daemon's and the client's shared log prefix (the engine's lines are "glm ...").
void say(const std::string& line);

/// A verbs device + protection domain + completion queue.  The device is picked by name (sysfs
/// basename, e.g. rocep7s0 as ibv_devices prints it) or the first non-default one found.
class Device {
public:
    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool open(const std::string& device_name, std::string& err);   // "" = first device
    bool close();

    ibv_context* ctx() const { return ctx_; }
    ibv_pd* pd() const { return pd_; }
    ibv_cq* cq() const { return cq_; }
    uint8_t port() const { return port_num_; }
    const ibv_port_attr& port_attr() const { return port_attr_; }
    ibv_qp* qp_holder() const { return nullptr; }   // (unused; Qp holds the QP)
    /// The RoCEv2 GID this device speaks (v2 preferred, v1 as the fall-back) and its index, empty on
    /// a pure InfiniBand link where the handshake carries the DLID instead.
    const std::string& gid() const { return gid_; }
    int gid_index() const { return gid_index_; }
    bool is_roce() const { return roce_; }
    /// The ibv device's kernel netdev (rocep7s0 -> enp7s0np0, via sysfs), empty if unmapped - the
    /// test and the engine use it to find which host IP rides the RDMA link.
    const std::string& netdev() const { return netdev_; }
    /// IBV_MTU_*, the smaller of the port's active MTU and 4096 (a direct-attach link's safe ceiling
    /// unless both ends were raised together - the peer setup raises them in step).
    int mtu() const;

private:
    ibv_context* ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    ibv_cq* cq_ = nullptr;
    ibv_port_attr port_attr_ = {};
    uint8_t port_num_ = 1;
    std::string gid_;
    int gid_index_ = -1;
    bool roce_ = false;
    std::string name_;
    std::string netdev_;
};

/// Pinned, registered memory: page-aligned malloc + one MR.  A slab piece on the daemon, a
/// destination staging buffer on the client - both are just memory plus keys here.
class Mr {
public:
    Mr() = default;
    ~Mr();
    Mr(const Mr&) = delete;
    Mr& operator=(const Mr&) = delete;

    /// Registers [buf, buf + bytes) held by the caller (daemon arenas, the client's staging), or
    /// allocates the bytes itself when buf is null (page-aligned, mlock'd through the MR pin).
    bool reg(ibv_pd* pd, void* buf, size_t bytes, int access, std::string& err);
    /// Allocates `bytes` page-aligned and registers them; the memory lives as long as the Mr.
    bool alloc(ibv_pd* pd, size_t bytes, int access, std::string& err);
    void release();

    void* buf() const { return buf_; }
    size_t bytes() const { return bytes_; }
    ibv_mr* mr_ptr() const { return mr_; }
    uint32_t lkey() const { return mr_ ? mr_->lkey : 0; }
    uint32_t rkey() const { return mr_ ? mr_->rkey : 0; }

private:
    void* buf_ = nullptr;
    size_t bytes_ = 0;
    bool owned_ = false;
    ibv_mr* mr_ = nullptr;
};

/// One RC QP over the RoCE link, armed by hand - the llama.cpp ggml-rpc way (transport.cpp
/// rdma_activate): both sides create their QP on the device's PD, exchange {qpn, psn, gid} over
/// the TCP control channel that is already here, and each runs INIT -> RTR -> RTS from the
/// peer's numbers.  No rdma_cm: the kernel's cm forces qp_access_flags to REMOTE_WRITE-only on
/// accept, and an RC responder without REMOTE_READ NAKs every incoming RDMA READ ("remote access
/// error") - the exact failure this pair hit for a whole debugging session.
class Qp {
public:
    Qp() = default;
    ~Qp();
    Qp(const Qp&) = delete;
    Qp& operator=(const Qp&) = delete;

    /// Creates the QP on the device's PD, both CQs on the device's shared CQ (reset state).
    bool create(Device& dev, int max_send_wr, int max_recv_wr, std::string& err);
    /// Arms it against the peer: INIT (full access flags: REMOTE_READ is what the responder
    /// side serves reads with) -> RTR (the peer's qpn/psn/gid, the port's active MTU) -> RTS.
    bool arm(Device& dev, uint32_t remote_qpn, uint32_t remote_psn, const std::string& remote_gid,
             std::string& err);
    void destroy();

    uint32_t qpn() const { return qp_ ? qp_->qp_num : 0; }
    uint32_t psn() const { return psn_; }              // this QP's starting PSN (sent to the peer)
    const std::string& gid() const { return gid_; }    // this side's GID, 32 hex nibbles
    ibv_qp* qp_ptr() const { return qp_; }
    bool established() const { return armed_; }

private:
    ibv_qp* qp_ = nullptr;
    uint32_t psn_ = 0;
    std::string gid_;
    bool armed_ = false;
};

/// The device's first RoCEv2 GID (its v1 sibling as the fall-back), as 32 hex nibbles.
std::string gid_hex(const ibv_gid& g);

/// 32 hex nibbles -> bytes (false on a malformed string).
bool gid_parse(const std::string& hex, ibv_gid* out);

/// FNV-1a 64 of a buffer - the pack fingerprint's primitive (chosen so the daemon can verify a
/// shard without pulling in any hash library).
uint64_t fnv1a(const void* p, size_t bytes);

}   // namespace strata::peer
