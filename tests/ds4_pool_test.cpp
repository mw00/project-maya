// ds4_pool_test: IoQueue's device-copy jobs (DS4_FLOW_HELPER's copies out of the helper's pool), CPU only - a fake
// copy function stands in for the GPU runtime.  Build: g++ -std=c++17 -O2 -pthread -I src/ds4 tests/ds4_pool_test.cpp
#include "ds4_pool.hpp"

#include <cstdio>
#include <vector>

using strata::ds4::IoQueue;

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

}  // namespace

int main() {
    device_copy_ok();
    device_copy_fails();
    memory_copy_unchanged();
    many_device_copies();
    if (failures) {
        std::fprintf(stderr, "ds4_pool_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("ds4_pool_test: all passed\n");
    return 0;
}
