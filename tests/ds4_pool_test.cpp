// ds4_pool_test: IoQueue's device-copy jobs (DS4_FLOW_HELPER's copies out of the helper's pool), CPU only - a fake
// copy function stands in for the GPU runtime; DS4_FLOW_DOWN_PART's compact_rows - in place against a naive gather,
// alone and on IoQueue's three kinds of job; DS4_TP_BORROW's tail arithmetic (ds4_borrow.hpp); and Pool::start waking
// a worker that is going to sleep (a batch no caller helps).
// Build: g++ -std=c++17 -O2 -pthread -I src/ds4 tests/ds4_pool_test.cpp
#include "ds4_borrow.hpp"
#include "ds4_pool.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using strata::ds4::IoQueue;
using strata::ds4::Pool;

namespace {

int failures = 0;
#define EXPECT(c)                                                            \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

std::atomic<int> copies{0}, last_dev{-1};
bool fake_copy(int dev, void* dst, const void* src, size_t n) {
    last_dev.store(dev);
    std::memcpy(dst, src, n);
    copies.fetch_add(1);
    return true;
}
bool fake_fail(int, void*, const void*, size_t) { return false; }

void wait_left(IoQueue& q, const std::atomic<int>& left) {
    for (int i = 0; i < 2000 && left.load() != 0; ++i) q.wait_any(1000);
}

void device_copy_ok() {
    IoQueue q(2);
    std::vector<uint8_t> src(1 << 20), dst(1 << 20, 0), keep(16, 7);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) (i * 31 + 5);
    std::atomic<int> left{1};
    std::atomic<bool> bad{false};
    IoQueue::Job j;
    j.len = src.size();
    j.off = 12345;   // (ignored by a device job: its bytes land at dst exactly)
    j.dst = dst.data();
    j.left = &left;
    j.bad = &bad;
    j.dmem = src.data();
    j.ddev = 2;
    j.dcopy = &fake_copy;
    j.keep = keep.data();   // (a read's: a device job leaves it alone)
    q.push(&j, 1);
    wait_left(q, left);
    EXPECT(left.load() == 0);
    EXPECT(!bad.load());
    EXPECT(dst == src);
    EXPECT(last_dev.load() == 2);
    EXPECT(q.device_bytes() == src.size());
    EXPECT(q.direct_bytes() == 0 && q.buffered_bytes() == 0);
    EXPECT(keep[0] == 7 && keep[15] == 7);
}

void device_copy_fails() {
    IoQueue q(1);
    std::vector<uint8_t> src(4096, 1), dst(4096, 0);
    std::atomic<int> left{2};
    std::atomic<bool> bad{false};
    IoQueue::Job a;
    a.len = src.size();
    a.dst = dst.data();
    a.left = &left;
    a.bad = &bad;
    a.dmem = src.data();
    a.dcopy = &fake_fail;
    IoQueue::Job b = a;
    b.dcopy = nullptr;   // (no copy function: a failure, never a read of fd -1)
    IoQueue::Job jobs[2] = {a, b};
    q.push(jobs, 2);
    wait_left(q, left);
    EXPECT(left.load() == 0);   // (counted down even when they fail, as a read's)
    EXPECT(bad.load());
    EXPECT(q.device_bytes() == 0);
}

void memory_copy_unchanged() {
    IoQueue q(1);
    std::vector<uint8_t> src(8192, 9), dst(8192, 0);
    std::atomic<int> left{1};
    std::atomic<bool> bad{false};
    IoQueue::Job j;
    j.len = src.size();
    j.dst = dst.data();
    j.left = &left;
    j.bad = &bad;
    j.mem = src.data();
    q.push(&j, 1);
    wait_left(q, left);
    EXPECT(left.load() == 0 && !bad.load() && dst == src);
    EXPECT(q.device_bytes() == 0);
}

void many_device_copies() {   // (an item's three parts, many items, several threads)
    IoQueue q(8);
    const int items = 300;
    std::vector<std::vector<uint8_t>> src(items * 3), dst(items * 3);
    std::vector<std::atomic<int>> left(items);
    std::vector<std::atomic<bool>> bad(items);
    std::vector<IoQueue::Job> jobs;
    uint64_t total = 0;
    copies.store(0);
    for (int i = 0; i < items; ++i) {
        left[(size_t) i].store(3);
        bad[(size_t) i].store(false);
        for (int r = 0; r < 3; ++r) {
            const size_t k = (size_t) i * 3 + (size_t) r, n = 1000 + (size_t) (i * 7 + r * 13) % 5000;
            src[k].assign(n, (uint8_t) (i + r));
            dst[k].assign(n, 0);
            IoQueue::Job j;
            j.len = n;
            j.dst = dst[k].data();
            j.left = &left[(size_t) i];
            j.bad = &bad[(size_t) i];
            j.dmem = src[k].data();
            j.ddev = 2;
            j.dcopy = &fake_copy;
            jobs.push_back(j);
            total += n;
        }
    }
    q.push(jobs.data(), (int) jobs.size());
    for (int i = 0; i < items; ++i) wait_left(q, left[(size_t) i]);
    bool same = true, any_bad = false;
    for (size_t k = 0; k < src.size(); ++k) same = same && src[k] == dst[k];
    for (auto& b : bad) any_bad = any_bad || b.load();
    EXPECT(same && !any_bad);
    EXPECT(copies.load() == items * 3);
    EXPECT(q.device_bytes() == total);
}

// DS4_TP_BORROW: a group's k is the most any member needs (its buffers + 256 over its parts of a slot), capped by the
// group's fewest slots a layer less one; every member's buffers then fit its own tail; a layer keeps the slots wholly
// before a member's buffers, and the group the fewest of its members'
void tp_borrow_arithmetic() {
    using strata::ds4::tail_keep;
    using strata::ds4::tp_tail_fits;
    using strata::ds4::tp_tail_k;
    // (two members of an uneven group: 1280- and 1024-wide parts, so the wider one's step is larger)
    const std::vector<size_t> need = {100, 90}, step = {50, 40};
    EXPECT(tp_tail_k(need, step, 10) == 9);   // (356 / 50 -> 8, 346 / 40 -> 9)
    EXPECT(tp_tail_fits(need, step, 9));
    EXPECT(!tp_tail_fits(need, step, 8));    // (the second member: 8 x 40 < 346)
    EXPECT(tp_tail_k(need, step, 5) == 5);   // (capped: the group's fewest slots a layer less one)
    EXPECT(!tp_tail_fits(need, step, 5));    // (then the chunk shrinks, or the group keeps its own room)
    EXPECT(tp_tail_k(need, step, 0) == 0);
    EXPECT(tp_tail_k({100}, {0}, 10) == -1);   // (a member without layers)
    EXPECT(tp_tail_k({100, 90}, {50}, 10) == -1);
    EXPECT(tp_tail_k({}, {}, 10) == -1);
    // the 092i shape: 32K-row buffers of ~5.9 GB a member over parts of ~0.24 GB a slot (40 layers x ~6 MB)
    const std::vector<size_t> n32 = {5900000000ull, 5700000000ull}, s32 = {241000000ull, 236000000ull};
    const int k32 = tp_tail_k(n32, s32, 2000);
    // (5,900,000,256 / 241,000,000 -> 25; 5,700,000,256 / 236e6 -> 25)
    EXPECT(k32 == 25 && tp_tail_fits(n32, s32, k32));
    // the slots wholly before the buffers at b0: parts of 40 B, 100 apart from offset 0
    EXPECT(tail_keep(0, 40, 100, 250, 5) == 3);   // (0-40, 100-140, 200-240 before 250; 300-340 not)
    EXPECT(tail_keep(0, 40, 100, 250, 2) == 2);   // (at most the layer's tail slots)
    EXPECT(tail_keep(60, 40, 100, 250, 5) == 2);  // (60-100, 160-200; 260-300 not)
    EXPECT(tail_keep(0, 40, 100, 30, 5) == 0);
    // the group keeps the fewest of its members' (pv_n = n_main + min): here the second member's 2
    const int keep = std::min(tail_keep(0, 40, 100, 250, 5), tail_keep(60, 40, 100, 250, 5));
    EXPECT(keep == 2);
}

// DS4_FLOW_DOWN_PART: compact_rows(base, rows, pitch, off, len) leaves row r's bytes [off, off + len) at r len, rows
// side by side - in place, the bytes a naive gather into a buffer of its own gives; nothing past rows x pitch is
// touched. A pattern that differs at every byte of a row and between rows, so a misplaced byte shows
uint8_t pat(size_t i) { return (uint8_t) (i * 131 + (i >> 8) * 7 + 1); }

// Pool::start stores the generation, then loads the sleeper count; a worker going to sleep adds itself to that count,
// then loads the generation.  Without a fence between the two (store -> load) both can read the old value: the worker
// sleeps through the batch and the starter skips the notify.  A batch nobody help()s (run_nodes' second pool) then
// never runs, so the test is a batch no caller helps, many times over, on a worker that sleeps as soon as it is idle.
// Its first batch also checks the worker's start: one that read gen_ only once its thread ran took a batch start()ed
// before that as already seen (on a 16-vCPU Xeon VM, 9 runs in 10 hung at batch 0 that way).
void pool_start_wakes_sleeper() {
    Pool p(1, {}, 0);
    const int batches = 200000;
    std::atomic<int> ran{0}, done{0};
    std::atomic<bool> over{false};
    const std::function<void(int)> fn = [&](int) { ran.fetch_add(1, std::memory_order_relaxed); };
    std::thread dog([&] {   // (a missed wakeup hangs wait() for good: report it and exit)
        int last = -1;
        for (int still = 0; !over.load(); std::this_thread::sleep_for(std::chrono::milliseconds(100))) {
            const int d = done.load();
            still = d == last ? still + 1 : 0;
            last = d;
            if (still == 50) {
                std::fprintf(stderr, "FAILED %s:%d: batch %d of a pool nobody helps never ran (a missed wakeup)\n",
                             __FILE__, __LINE__, d);
                std::_Exit(1);
            }
        }
    });
    for (int i = 0; i < batches; ++i) {
        p.start(1, fn);
        p.wait();
        done.store(i + 1);
    }
    over.store(true);
    dog.join();
    EXPECT(ran.load() == batches);
}

bool compacts(size_t rows, size_t pitch, size_t off, size_t len) {
    const size_t n = rows * pitch, guard = 64;
    std::vector<uint8_t> buf(n + guard), want(rows * len);
    for (size_t i = 0; i < n; ++i) buf[i] = pat(i);
    for (size_t i = n; i < n + guard; ++i) buf[i] = 0xA5;
    for (size_t r = 0; r < rows; ++r)   // (the naive gather: row r's part, from the untouched input)
        for (size_t i = 0; i < len; ++i) want[r * len + i] = pat(r * pitch + off + i);
    strata::ds4::compact_rows(buf.data(), rows, pitch, off, len);
    bool ok = std::memcmp(buf.data(), want.data(), want.size()) == 0;
    for (size_t i = n; i < n + guard; ++i) ok = ok && buf[i] == 0xA5;
    return ok;
}

void compact_rows_shapes() {
    // every shape of up to 5 rows of up to 12 bytes: every offset, every length that fits - off 0, len == pitch (the
    // whole row: nothing moves) and the parts at a row's end among them
    int shapes = 0, bad = 0;
    for (size_t rows = 0; rows <= 5; ++rows)
        for (size_t pitch = 1; pitch <= 12; ++pitch)
            for (size_t off = 0; off < pitch; ++off)
                for (size_t len = 1; off + len <= pitch; ++len, ++shapes)
                    if (!compacts(rows, pitch, off, len)) {
                        if (bad++ < 5)
                            std::fprintf(stderr, "compact_rows(%zu, %zu, %zu, %zu) differs\n", rows, pitch, off, len);
                    }
    EXPECT(bad == 0 && shapes == 2184);   // (6 row counts x 364 (off, len) pairs)
    // an uneven split's two members of one down, rows of a quantized row's bytes: member 0 its first blocks (off 0),
    // member 1 the rest (off = member 0's len, to the row's end); a row of one 4-byte block; many rows
    const size_t pitch = 594, m0 = 330;   // (e.g. 2304 columns in 18 blocks of 33 bytes: 10 blocks and 8)
    EXPECT(compacts(4096, pitch, 0, m0));
    EXPECT(compacts(4096, pitch, m0, pitch - m0));
    EXPECT(compacts(4096, pitch, 0, pitch));
    EXPECT(compacts(7, 4, 0, 4) && compacts(1, 594, 264, 330) && compacts(0, 594, 0, 330));
    for (size_t m1 : {(size_t) 1, (size_t) 33, (size_t) 297, (size_t) 593})   // (member 1's offset, every width)
        EXPECT(compacts(257, pitch, m1, pitch - m1) && compacts(257, pitch, 0, m1));
}

// the same through IoQueue: a copy (memory or device) compacted where it landed - dst itself, whatever its off -, a
// read at dst + off % kDirectAlign; the bytes past the part as they landed (compact_rows leaves them), `left` counted
// after
void compact_rows_jobs() {
    const size_t rows = 64, pitch = 594, off = 330, len = 264, n = rows * pitch;
    std::vector<uint8_t> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = pat(i);
    std::vector<uint8_t> want(rows * len);
    for (size_t r = 0; r < rows; ++r) std::memcpy(want.data() + r * len, src.data() + r * pitch + off, len);
    auto job = [&](IoQueue::Job& j, std::vector<uint8_t>& dst, std::atomic<int>& left, std::atomic<bool>& bad) {
        j.len = n;
        j.dst = dst.data();
        j.left = &left;
        j.bad = &bad;
        j.crows = rows;
        j.cpitch = pitch;
        j.coff = off;
        j.clen = len;
    };
    {   // (a memory copy: off ignored, compacted at dst)
        IoQueue q(1);
        std::vector<uint8_t> dst(n, 0);
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        IoQueue::Job j;
        job(j, dst, left, bad);
        j.off = 12345;
        j.mem = src.data();
        q.push(&j, 1);
        wait_left(q, left);
        EXPECT(left.load() == 0 && !bad.load());
        EXPECT(std::memcmp(dst.data(), want.data(), want.size()) == 0);
        EXPECT(std::memcmp(dst.data() + want.size(), src.data() + want.size(), n - want.size()) == 0);
    }
    {   // (a device copy - the helper pool's: compacted at dst, the device bytes the whole copy's)
        IoQueue q(2);
        std::vector<uint8_t> dst(n, 0);
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        IoQueue::Job j;
        job(j, dst, left, bad);
        j.off = 777;
        j.dmem = src.data();
        j.ddev = 2;
        j.dcopy = &fake_copy;
        q.push(&j, 1);
        wait_left(q, left);
        EXPECT(left.load() == 0 && !bad.load() && q.device_bytes() == n);
        EXPECT(std::memcmp(dst.data(), want.data(), want.size()) == 0);
    }
    {   // (a failed device copy: nothing compacted, counted, bad)
        IoQueue q(1);
        std::vector<uint8_t> dst(n, 0);
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        IoQueue::Job j;
        job(j, dst, left, bad);
        j.dmem = src.data();
        j.dcopy = &fake_fail;
        q.push(&j, 1);
        wait_left(q, left);
        EXPECT(left.load() == 0 && bad.load() && dst == std::vector<uint8_t>(n, 0));
    }
#if !defined(_WIN32)
    {   // (a buffered read of a file at an offset: its bytes at dst + off % kDirectAlign, compacted there; the lead
        //  before them untouched; a keep copy the read's bytes as read, before the compaction)
        char path[] = "/tmp/ds4_pool_test_XXXXXX";
        const int fd = mkstemp(path);
        EXPECT(fd >= 0);
        if (fd < 0) return;
        const uint64_t at = 3 * 4096 + 100, lead = at % strata::ds4::os::kDirectAlign;
        std::vector<uint8_t> file(at + n, 0x11);
        std::memcpy(file.data() + at, src.data(), n);
        EXPECT(::write(fd, file.data(), file.size()) == (ssize_t) file.size());
        ::close(fd);
        const strata::ds4::os::fd_t f = strata::ds4::os::open_par(path);
        EXPECT(f != strata::ds4::os::kBadFd);
        IoQueue q(1);
        std::vector<uint8_t> dst(lead + n, 0x5A), keep(n, 0);
        std::atomic<int> left{1};
        std::atomic<bool> bad{false};
        IoQueue::Job j;
        job(j, dst, left, bad);
        j.fd = f;
        j.off = at;
        j.keep = keep.data();
        q.push(&j, 1);
        wait_left(q, left);
        EXPECT(left.load() == 0 && !bad.load() && q.buffered_bytes() == n);
        EXPECT(std::memcmp(dst.data() + lead, want.data(), want.size()) == 0);
        bool lead_ok = true;
        for (uint64_t i = 0; i < lead; ++i) lead_ok = lead_ok && dst[i] == 0x5A;
        EXPECT(lead_ok && keep == src);
        strata::ds4::os::close_fd(f);
        std::remove(path);
    }
#endif
}

}  // namespace

int main() {
    device_copy_ok();
    device_copy_fails();
    memory_copy_unchanged();
    many_device_copies();
    tp_borrow_arithmetic();
    compact_rows_shapes();
    compact_rows_jobs();
    pool_start_wakes_sleeper();
    if (failures) {
        std::fprintf(stderr, "ds4_pool_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("ds4_pool_test: all passed\n");
    return 0;
}
