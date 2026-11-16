// src/ds4/ds4_part_net.hpp - the pool's link: one TCP connection between the front (the PC that runs the first layers
// and talks to the server) and a part server (the PC that runs the last layers, the head, the sampling and the DSpark
// drafter).  Frames: op (u32), flags (u32), length (u64), the payload; both ends little-endian.  TCP_NODELAY, large
// buffers, a timeout on every wait (a dead link is an error, never a hang).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ds4::part {

enum Op : uint32_t {
    kHello = 1,
    kWin = 2,       // a window through the part's layers (prompt windows, a decode step)
    kVerify = 3,    // a verify window: the part judges the drafts, keeps / undoes its rows, drafts the next round
    kSample = 4,
    kDraft = 5,
    kChunks = 6,    // a run of chunks: its plan and tokens; kChunk frames follow, one a chunk
    kChunk = 7,
    kReset = 8,
    kCheckpoint = 9,
    kRestore = 10,
    kPin = 11,
    kMark = 12,
    kBegin = 13,
    kPenalties = 14,
    kBias = 15,
    kLogprobs = 16,
    kSync = 17,
    kBye = 18,
    kShm = 19,       // a front on the part's own PC: the name and size of the shared memory its chunks' rows go through
    kChunkShm = 20,  // a chunk whose rows wait in a slot of that memory: the slot, the bytes, then kChunk's header
    kReply = 100,
};

class Conn {
public:
    Conn() = default;
    ~Conn();
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;
    // the front: connect to host:port within timeout_ms
    bool connect(const std::string& host, int port, int timeout_ms, std::string& err);
    // the part server: listen on port, then accept() one front at a time
    bool listen(int port, std::string& err);
    bool accept(std::string& err);
    void close();
    bool ok() const { return fd_ >= 0; }
    // every wait after this gives up after ms (0: none)
    void set_timeout(int ms);
    bool send_frame(uint32_t op, const void* a, size_t na, const void* b = nullptr, size_t nb = 0,
                    const void* c = nullptr, size_t nc = 0);
    // a frame's header, then its payload into `out` (resized)
    bool recv_frame(uint32_t& op, std::vector<uint8_t>& out);
    std::string last_error() const { return err_; }
    std::string peer() const { return peer_; }

private:
    bool send_all(const void* p, size_t n);
    bool recv_all(void* p, size_t n);
    long long fd_ = -1, lfd_ = -1;
    std::string err_, peer_;
};

// Two programs on one PC (a front and a part, each on its own GPU): a block of host memory both map.  The front's GPU
// writes a chunk's rows straight into one of its two slots (the front pins the block), the part copies them out and
// frees the slot - no socket carries the rows.  A header page holds a state word a slot (0 free, 1 full).
class Shm {
public:
    Shm() = default;
    ~Shm() { close(); }
    Shm(const Shm&) = delete;
    Shm& operator=(const Shm&) = delete;
    // the front: a new block of two slots of `cap` bytes each (the name: for the part's open())
    bool create(size_t cap, std::string& err);
    // the part: the front's block
    bool open(const std::string& name, size_t cap, std::string& err);
    // the name off the file system (the mappings stay): once both sides hold it
    void unlink();
    void close();
    bool ok() const { return p_ != nullptr; }
    const std::string& name() const { return name_; }
    size_t cap() const { return cap_; }
    uint8_t* base() const { return p_; }
    size_t bytes() const { return bytes_; }
    uint8_t* slot(int i) const { return p_ + kHdr + (size_t) i * cap_; }
    bool full(int i) const;
    void set_full(int i, bool f);
    static constexpr size_t kHdr = 4096;

private:
    uint8_t* p_ = nullptr;
    size_t bytes_ = 0, cap_ = 0;
    std::string name_;
    bool owner_ = false;
};

// a payload written and read field by field
struct Writer {
    std::vector<uint8_t> b;
    template <class T> void put(const T& v) {
        const size_t o = b.size();
        b.resize(o + sizeof(T));
        std::memcpy(b.data() + o, &v, sizeof(T));
    }
    void put_bytes(const void* p, size_t n) {
        const size_t o = b.size();
        b.resize(o + n);
        if (n) std::memcpy(b.data() + o, p, n);
    }
    void put_str(const std::string& s) {
        put<uint32_t>((uint32_t) s.size());
        put_bytes(s.data(), s.size());
    }
};
struct Reader {
    const uint8_t* p = nullptr;
    size_t n = 0, at = 0;
    bool bad = false;
    Reader(const std::vector<uint8_t>& v) : p(v.data()), n(v.size()) {}
    template <class T> T get() {
        T v{};
        if (at + sizeof(T) > n) {
            bad = true;
            return v;
        }
        std::memcpy(&v, p + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    const uint8_t* bytes(size_t k) {
        if (at + k > n) {
            bad = true;
            return nullptr;
        }
        const uint8_t* q = p + at;
        at += k;
        return q;
    }
    std::string str() {
        const uint32_t k = get<uint32_t>();
        const uint8_t* q = bytes(k);
        return q ? std::string((const char*) q, k) : std::string();
    }
};

}  // namespace strata::ds4::part
