// glm_link: one GLM layer split across TWO PROCESSES (two rigs) over TCP - Phase 1 of the two-rig split.
//
// Off unless MAYA_GLM_ROLE (or MAYA_GLM_ROLE, or `--role`) is front|back: then nothing here runs and the engine is the
// unmodified one.  With a role the process loads ONE part of the two-part split at the boundary given by --layer-split K
// (front: layers [0, K) + the embedding; back: [K, n_layers) + the NextN block + the output head + the sampler) and
// both processes run the same driver in lock step (SPMD): the front tees every control line it reads (GEN, QUIT, ...)
// to the back, which reads them instead of its stdin, so both make the same model calls in the same order.  Where the
// in-process split copies the residual through a pinned hop buffer, the front sends it over the link and the back
// receives it; the back returns every token it picks.
//
// Two TCP connections, both opened by the front (MAYA_GLM_LINK=host:port; the back listens on it):
//   data: framed messages (Hdr + payload), TCP_NODELAY.  front -> back: HELLO, HOP (residual rows of n positions from
//         pos, n x hc x n_embd fp32); back -> front: HELLO, TOK (a token picked at pos), ERR (text)
//   ctl:  newline-terminated control lines, front -> back (the back's stdin)
// Any link failure fails the request on both sides and takes the link down; the front connects again at its next
// command (MAYA_GLM_RELINK_WAIT seconds, default 5) and the back, which stays up, accepts it (Phase 2).
// No wait on the link is unbounded: TCP keepalive and the user timeout end a peer that is gone
// (MAYA_GLM_LINK_DEAD_S, 30), a wait of MAYA_GLM_LINK_STALL_S seconds (60) for a peer that is connected but sends
// or takes nothing ends the link too, and the handshakes run under deadlines (glm_link.cpp).
// Every request is bracketed: kBegin (both sides' reuse length and decode mode: a mismatch reads the prompt from 0)
// and kEnd/kAck (the front's end: the back stops wherever it waits - kEnd in place of any message is a cancel - and
// acknowledges with its position; anything stale before the marker is dropped).
// MAYA_GLM_LINK_BF16=1 (default): the boundary rows cross as bf16 (rounded to nearest even, upcast on the back).
// The back listens on MAYA_GLM_LINK_BIND (else the --link host; an empty host is loopback, never every interface) and
// accepts a front only from the addresses in MAYA_GLM_LINK_ALLOW / MAYA_GLM_LINK_ALLOW (comma-separated IPv4;
// unset: loopback only).  Malformed lengths (an ERR over 4 KiB, a payload on a message without one, a packed block
// that does not fit its own length) are protocol errors: the link drops.
// Several local GPUs per side: --layer-split K:b1,b2,.. (glm_model.cu load_pack_place).
// MAYA_GLM_LINK_KEY_FILE / MAYA_GLM_LINK_KEY_FILE: a shared key (>= 32 bytes) - every connection runs an HMAC-SHA256
// challenge-response both ways before HELLO (glm_link.cpp); a connection that fails it is closed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace maya::core::glmlink {

enum Role : int { kNone = 0, kFront = 1, kBack = 2 };
enum Type : uint32_t { kHello = 1, kHop = 2, kTok = 3, kSample = 4, kErr = 5, kSpec = 6, kCtl = 7, kBegin = 8, kEnd = 9, kAck = 10,
                    kChunk = 13 };
// kEnd's val (front -> back): bit 0 PAUSE, bit 1 its keep, bit 2 the request was cancelled (a STOP, a client gone)
// kChunk (both ways, at the start of a prompt-path call): pos = the call's first position, n = its tokens, val = the
// chunk this side laid out for it, in tokens (0: this side has no prompt path now - both read the call token by
// token; -1: the layout does not fit - the request fails on both, unless the other side says 0).  Both read the
// call in the smaller one's chunks
// kSpec (back -> front, the pipelined spec decode): pos = the tail's position, val = its token y, n = the draft d for
// the position after (-1: none).  kCtl (front -> back): pos = the position the step is for, val = 1 go on / 0 the end

struct Hdr {
    uint32_t magic;   // kMagic
    uint32_t type;    // Type
    int64_t pos;      // HOP: first position; TOK/SAMPLE: the position the token was picked at; HELLO: protocol version
    int32_t n;        // HOP: positions; HELLO: the split's boundary layer
    int32_t val;      // TOK/SAMPLE: the token id (-1: failed, an ERR follows); HELLO: the prompt chunk (0: none)
    uint64_t bytes;   // payload bytes that follow; HELLO: max context
};
constexpr uint32_t kMagic = 0x4b4e4c4d;   // "MLNK"
constexpr int64_t kVersion = 3;

Role role();                          // from MAYA_GLM_ROLE / MAYA_GLM_ROLE (cached)
const char* role_name();
bool up();                            // the link is connected and healthy
// the link key (MAYA_GLM_LINK_KEY_FILE): false + err when it is set and the file is missing or short (checked at load)
bool key_ready(std::string& err);

// open the link (front: connect, retrying up to MAYA_GLM_LINK_WAIT seconds, default 1800; back: listen, accept both
// connections) and exchange HELLO.  my_chunk: this side's prompt chunk; peer_chunk: the other side's.
bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err);
// the boundary layer K (set at load, before open: the parts size their RAM tiers over this side's layers only)
void set_split(int split_at);
/// What both sides must have loaded (glm_model.cu: the pack's geometry and GGUF shards); the HELLO compares it.
void set_model(const std::string& id);
int split_at();

// data channel
bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err);
// receive a HOP into dst (cap bytes), checking that it carries exactly (pos, n) and `bytes` bytes
bool recv_hop(int64_t pos, int n, void* dst, size_t bytes, std::string& err);
bool send_tok(Type t, int64_t pos, int32_t tok);
// the back failed a request: tell the front why (an ERR in place of what it waits for) and take the link down
void abort_link(const std::string& why);
bool recv_tok(Type t, int64_t pos, int32_t& tok, std::string& err);
bool send_pair(Type t, int64_t pos, int32_t val, int32_t n);                          // kSpec / kCtl
bool recv_pair(Type t, int64_t pos, int32_t& val, int32_t& n, std::string& err);

// control channel: the front sends each line it reads; the back reads them as its stdin (ctl_fd: -1 off)
void ctl_send_line(const std::string& line);
void ctl_close();                     // front: no more lines (EOF on the back)
int ctl_fd();                         // back: the fd to read the control lines from

// Phase 2: requests, cancel, reconnect
void request_begin();                 // both sides, a request's start: no cancel seen yet
bool cancelled();                     // the back: the front's kEnd came in place of what it waited for (err "cancelled")
// both: the request's start - the front sends (reuse, mode), the back answers with its own
bool begin_exchange(int64_t reuse, int32_t mode, int64_t& peer_reuse, int32_t& peer_mode, std::string& err);
/// A prompt-path call's chunk (Glm5Model::prefill): each side lays its chunk out on its own, so both say what they
/// came to (mine: tokens, 0 or -1 as kChunk's val) for the call's n tokens at pos, and read the other's.  Sides
/// that are not at the same call (pos, n) fail here, before any row crosses.  err "cancelled" on the back: the front
/// ended the request.
bool chunk_exchange(int64_t pos, int32_t n, int32_t mine, int32_t& peer, std::string& err);
// the front: kEnd{pos, flags}, then everything up to the back's kAck is dropped; agree: the back stands at pos too
bool end_front(int64_t pos, int32_t flags, bool& agree, std::string& err);
// the back: (unless the kEnd came already) everything up to the front's kEnd is dropped, then kAck{pos, agree}
bool end_back(int64_t pos, int32_t& flags, bool& agree, std::string& err);
bool bf16();                          // the hop's wire format
bool idle_check();                    // the front between commands: false if the back has gone (the link is down)
// the front: connect again - MAYA_GLM_RELINK_WAIT seconds (5; 1 at least) for all of it: both connections, their
// proofs and HELLO
bool reopen(std::string& err);
// the front's reader thread, on a STOP / PAUSE / QUIT: from the first one the link is waited on for 30 s more in
// all (then it drops; the next command connects again).  stop_clear: the front takes up a GEN
void stop_hint();
void stop_clear();
// the back's SIGTERM / SIGINT handler (async-signal-safe: a flag): it stops in order - every wait on the link looks
// at the flag once a second, nothing more is accepted, reaccept fails - or at once on a second signal or before it
// listens
void term();
// the back serves one front after another (--serve, MAYA_GLM_BACK_STAY not 0): a front whose HELLO does not fit is
// turned away and the next one awaited.  Otherwise the back takes one front and stops listening
void set_stay(bool stay);
// the back's reader thread: the control connection's next bytes (0: its end, or the back was told to stop)
long ctl_read(char* p, size_t n);
void wake();                          // the back: the old link's data socket shut (a main thread blocked on it returns)
bool reaccept(std::string& err);      // the back: accept the front's next connection and HELLO

// stats (stderr at exit): hops, bytes, time blocked in send/recv
void report();

}  // namespace maya::core::glmlink
