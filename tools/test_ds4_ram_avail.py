"""DeepSeek's Linux RAM budget, including reclaimable ZFS ARC and cgroup v2 limits.

Compile the engine's actual helpers with fixture-backed procfs reads, following the
other host-only DeepSeek tests. No ZFS installation, model or GPU is required.

    python -m unittest tools.test_ds4_ram_avail
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


FIXTURE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>

std::unordered_map<std::string, std::string> files;
std::string read_file(const std::string& path) {
    const auto it = files.find(path);
    return it == files.end() ? std::string() : it->second;
}

HELPERS

constexpr uint64_t GiB = uint64_t{1} << 30;
int checks = 0;
void base(uint64_t available = 8) {
    files.clear();
    files["/proc/meminfo"] = "MemTotal: 67108864 kB\nMemFree: 1024 kB\nMemAvailable: " +
        std::to_string(available << 20) + " kB\n";
}
void arc(uint64_t size = 16, uint64_t floor = 2) {
    files["/proc/spl/kstat/zfs/arcstats"] = "13 1 0x01 0 0\nname type data\nhits 4 123\nsize 4 " +
        std::to_string(size * GiB) + "\nc_min 4 " + std::to_string(floor * GiB) + "\n";
}
void cgroup(const std::string& path, uint64_t limit, uint64_t current, uint64_t file = 0) {
    const std::string dir = "/sys/fs/cgroup" + path;
    files[dir + "/memory.max"] = std::to_string(limit * GiB);
    files[dir + "/memory.current"] = std::to_string(current * GiB);
    files[dir + "/memory.stat"] = "anon 0\nfile " + std::to_string(file * GiB) + "\n";
}
void check(const char* name, uint64_t expected) {
    const uint64_t got = ram_avail();
    if (got != expected) {
        std::fprintf(stderr, "%s: got %llu, expected %llu\n", name,
                     (unsigned long long)got, (unsigned long long)expected);
        std::exit(1);
    }
    ++checks;
}
int main() {
    base(); check("no ZFS", 8 * GiB);
    base(); files["/proc/spl/kstat/zfs/arcstats"] = "size bad malformed\nc_min 4 10\n";
    check("malformed ARC row", 8 * GiB);
    base(); files["/proc/spl/kstat/zfs/arcstats"] = "size 4 17179869184\n";
    check("missing ARC floor", 8 * GiB);
    base(); files["/proc/spl/kstat/zfs/arcstats"] = "c_min 4 2147483648\n";
    check("missing ARC size", 8 * GiB);
    base(); arc(1, 2); check("ARC below floor", 8 * GiB);
    base(); arc(2, 2); check("ARC at floor", 8 * GiB);
    base(); arc(); check("only ARC above floor is reclaimable", 22 * GiB);
    base(); arc(16, 0); check("zero ARC floor", 24 * GiB);
    base(60); arc(); check("physical RAM ceiling", 64 * GiB);
    base(); files["/proc/spl/kstat/zfs/arcstats"] = "size 4 18446744073709551615\nc_min 4 0\n";
    check("ARC addition cannot overflow", 64 * GiB);
    base(); arc(); files["/proc/meminfo"] = "MemAvailable: 8388608 kB\n";
    check("missing MemTotal", 22 * GiB);
    base(); arc(); files["/proc/meminfo"] = "MemTotal: 67108864 kB\n";
    check("missing MemAvailable stays unknown", ~uint64_t{0});
    files.erase("/proc/meminfo");
    check("missing meminfo stays unknown", ~uint64_t{0});

    base(); arc(); files["/proc/self/cgroup"] = "0::/user.slice/maya.service\n";
    files["/sys/fs/cgroup/user.slice/maya.service/memory.max"] = "max\n";
    check("unlimited cgroup", 22 * GiB);
    cgroup("/user.slice/maya.service", 20, 10, 4);
    check("cgroup cap applies after ARC and reclaims file cache", 14 * GiB);
    cgroup("/user.slice", 10, 8);
    check("tighter ancestor cgroup", 2 * GiB);
    cgroup("/user.slice", 10, 12);
    check("exhausted ancestor cgroup", 0);
    cgroup("/user.slice", 10, 8, 12);
    check("file cache above current cannot underflow", 10 * GiB);
    files["/proc/meminfo"] = "MemTotal: 67108864 kB\n";
    check("cgroup caps unknown host availability", 10 * GiB);
    std::printf("RAM availability: %d fixture checks passed\n", checks);
}
'''


@unittest.skipUnless(sys.platform != "win32" and shutil.which("c++"), "host C++ compiler required")
class RamAvailableTests(unittest.TestCase):
    def test_arc_and_cgroup_limits(self):
        source = (Path(__file__).resolve().parents[1] / "src/ds4/ds4_model.cu").read_text()
        helpers = re.findall(r"^uint64_t (?:zfs_arc_reclaimable|ram_avail)\(\) \{.*?^\}", source, re.M | re.S)
        self.assertEqual(len(helpers), 2, "expected the engine's ARC and RAM helpers")
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d) / "ram_avail.cpp", Path(d) / "ram_avail"
            cpp.write_text(FIXTURE.replace("HELPERS", "\n".join(helpers)))
            build = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)],
                                   capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("19 fixture checks passed", run.stdout)


if __name__ == "__main__":
    unittest.main()
