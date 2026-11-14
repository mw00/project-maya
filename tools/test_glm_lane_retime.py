"""Run the production lane planner and retimer without CUDA (requires a C++17 compiler)."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def function(source, signature):
    start = source.index(signature)
    end = source.index("\n}", start) + 2
    return source[start:end]


class LaneRetimeTest(unittest.TestCase):
    def test_decode_replans_to_pcie_when_cpu_stays_slow(self):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("cl")
        if compiler is None:
            self.skipTest("a C++17 compiler is required")
        source = (Path(__file__).resolve().parents[1] / "src/core/glm_fast_path.cu").read_text()
        # Compile the actual methods. Only the surrounding GPU model/state is replaced.
        code = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
struct FastState {
    void* cpu_pool = this;
    bool cpu_retime = true;
    int rt_windows = 0, rt_changes = 0, rt_dir = 0;
    uint64_t rt_e0 = 0, rt_us0 = 0, rt_r0 = 0;
    std::atomic<uint64_t> cpu_experts{0}, cpu_us{0}, cpu_routes{0};
    unsigned long long cpu_plan = 0, cpu_plan_start = 0;
    double rt_c_ms = 0.23, cpu_p_ms = 0.33, rt_prev_ms = 0;
};
struct Glm5Model {
    FastState* fast_;
    std::unique_ptr<Glm5Model> split_next_;
    int dev_ = 0;
    void lane_retime();
};
constexpr double kLaneOverMs = 0.04;
'''
        code += function(source, "static unsigned long long lane_plan(") + "\n"
        code += function(source, "static double plan_share(") + "\n"
        code += function(source, "void Glm5Model::lane_retime()") + "\n"
        code += r'''
void window(Glm5Model& m, double ms) {
    auto& f = *m.fast_;
    f.cpu_experts += 2000;
    f.cpu_routes += 1000;
    f.cpu_us += (uint64_t) std::llround((2000 * ms + 1000 * kLaneOverMs) * 1000);
    m.lane_retime();
}
void init(FastState& f) {
    f.cpu_plan = f.cpu_plan_start = lane_plan(f.rt_c_ms, f.cpu_p_ms);
    assert(f.cpu_plan != 0);
}
int main() {
    FastState f;
    init(f);
    Glm5Model m{&f};
    const auto initial = f.cpu_plan;
    window(m, 10.0); // warmup is ignored
    window(m, 10.0); // one slow window is insufficient
    assert(f.cpu_plan == initial);
    window(m, 10.0);
    assert(f.cpu_plan == 0); // all misses now use PCIe
    assert(f.cpu_plan_start == 0); // request overrides restore this new default
    assert(plan_share(f.cpu_plan) == 1.0);
    assert(f.rt_changes == 1);
    assert(f.cpu_pool != nullptr); // keep allocations and explicit tuning available
    m.lane_retime(); // no new CPU samples: no further change
    assert(f.rt_changes == 1);

    FastState explicit_plan;
    init(explicit_plan);
    explicit_plan.cpu_retime = false;
    Glm5Model manual{&explicit_plan};
    for (int i = 0; i < 4; ++i) window(manual, 10.0);
    assert(explicit_plan.cpu_plan == initial);

    FastState transient;
    init(transient);
    Glm5Model noisy{&transient};
    window(noisy, 0.23);
    window(noisy, 10.0);
    window(noisy, 0.23);
    assert(transient.cpu_plan == initial);

    FastState faster;
    init(faster);
    Glm5Model improving{&faster};
    for (int i = 0; i < 3; ++i) window(improving, 0.10);
    assert(faster.cpu_plan == lane_plan(0.10, faster.cpu_p_ms));
    assert(faster.cpu_plan != initial);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "retime.cpp"
            exe = Path(tmp) / "retime.exe"
            src.write_text(code)
            if Path(compiler).stem.lower() == "cl":
                cmd = [compiler, "/nologo", "/std:c++17", "/EHsc", str(src), f"/Fe:{exe}"]
            else:
                cmd = [compiler, "-std=c++17", str(src), "-o", str(exe)]
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
