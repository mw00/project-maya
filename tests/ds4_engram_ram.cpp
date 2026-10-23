// CPU-only: raw/Q6_K rows, split FP8 and MXFP8 use the locked bytes, with no file fallback.
#include "../src/ds4/ds4_engram_io.hpp"
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::ds4;
static void check(bool ok, const char* what) { if (!ok) { std::cerr << what << '\n'; std::exit(1); } }
int main() {
    char name[] = "/tmp/maya-engram-ram-XXXXXX";
    int fd = mkstemp(name); check(fd >= 0, "mkstemp"); close(fd);
    std::vector<uint8_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (i % 251);
    { std::ofstream f(name, std::ios::binary); f.write((char*) data.data(), data.size()); }
    std::string err;
    { EgReader bad;
      check(!bad.pin_range(name, 0, 4000, 256, err), "truncated range accepted");
      check(!bad.pin_range(name, 0, 0, 0, err), "empty range accepted"); }
    // A locked-memory limit failure must be reported, never turned into ordinary file reads.
    pid_t child = fork(); check(child >= 0, "fork");
    if (child == 0) {
        struct rlimit lim = {0, 0};
        if (setrlimit(RLIMIT_MEMLOCK, &lim) != 0) _exit(2);
        EgReader reader;
        _exit(reader.pin_range(name, 0, 0, 32, err) ? 3 : 0);
    }
    int status = 0; check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                          "mlock failure not reported (run this test without CAP_IPC_LOCK)");
    EgReader disk, ram;
    check(ram.pin_range(name, 0, 192, 256, err), err.c_str());
    check(ram.pin_range(name, 0, 1024, 8, err), err.c_str());
    ram.require_ram(); disk.start({name}, 1); ram.start({name}, 2);
    uint8_t a[64] = {}, z[64] = {};
    EgRead q; q.file = 0; q.off = 200; q.n = 32; q.dst = a;
    EgBatch one; one.reads.push_back(q); disk.submit(one); disk.wait(one);
    check(one.ok[0] && !std::memcmp(a, data.data() + 200, 32), "default file read changed");
    disk.stop(); check(unlink(name) == 0, "unlink");
    // A fresh RAM worker cannot open this name anymore.
    EgBatch two; q.dst = z; q.file2 = 0; q.off2 = 1025; q.n2 = 1; uint8_t scale = 0; q.scales = &scale;
    two.reads.push_back(q); ram.submit(two); ram.wait(two);
    check(two.ok[0] && !std::memcmp(a, z, 32) && scale == data[1025], "split FP8 RAM read differs");
    EgBatch mx; q.off = 256; q.n = 33; q.n2 = 0; q.mx8 = true; mx.reads.push_back(q); ram.submit(mx); ram.wait(mx);
    check(mx.ok[0] && scale == data[256] && !std::memcmp(z, data.data() + 257, 32), "MXFP8 decode differs");
    // Recreate the path: a disk fallback would succeed here and must still be rejected.
    { std::ofstream f(name, std::ios::binary); f.write((char*) data.data(), data.size()); }
    EgBatch missing; q.mx8 = false; q.off = 2048; q.n = 32; missing.reads.push_back(q);
    ram.submit(missing); ram.wait(missing); check(!missing.ok[0], "missing RAM range silently fell back");
    ram.stop(); check(unlink(name) == 0, "cleanup"); std::cout << "locked RAM, default reads, FP8/MXFP8, bounds, and failure checks passed\n";
}
