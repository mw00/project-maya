"""Host contracts for the mixed helper's direct response, using the production code.

No GPU/model required. These checks cover path selection and the service-reader
acknowledgement; native repeated-request tests still need both GPUs.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get("CXX", "c++")


@unittest.skipUnless(shutil.which(CXX), "C++ compiler required")
class HelperResponseTests(unittest.TestCase):
    def compile_run(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            source, exe = Path(tmp) / "check.cpp", Path(tmp) / "check"
            source.write_text("""
#include <atomic>
#include <cstdio>
#include <vector>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "failed line %d: %s\\n", __LINE__, #c); return 1; } } while (0)
""" + body)
            build = subprocess.run([CXX, "-std=c++17", "-O2", "-DNDEBUG", str(source), "-o", str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_direct_response_requires_complete_expert_coverage(self):
        text = (ROOT / "src/ds4/ds4_rhelper.inc").read_text()
        start = text.index("    bool rh_direct_on(")
        method = text[start:text.index("\n    }", start) + 6]
        self.compile_run("""
struct Layer { void* helper_tab = (void*)1; int n_vram = 384, n_exp = 384; };
struct Model {
    bool rh_direct = true, rh_hot = true;
    std::atomic<bool> helper_dead{false};
""" + method + """
};
int main() {
    Model m; Layer l;
    CHECK(m.rh_direct_on(l)); // helper + lead cover the fully resident layer
    for (int resident : {0, 1, 383}) {
        l.n_vram = resident;
        CHECK(!m.rh_direct_on(l)); // even one CPU/SSD expert needs the combined response
    }
    m.rh_hot = false;
    CHECK(m.rh_direct_on(l)); // non-hot helper owns all nonresident experts
    m.helper_dead = true; CHECK(!m.rh_direct_on(l));
    m.helper_dead = false; m.rh_direct = false; CHECK(!m.rh_direct_on(l));
    m.rh_direct = true; l.helper_tab = nullptr; CHECK(!m.rh_direct_on(l));
    return 0;
}
""")

    def test_direct_response_retires_service_reader(self):
        text = (ROOT / "src/ds4/ds4_model.cu").read_text()
        start = text.index('err = "exl3 win_experts failed";')
        start = text.index("\n            }", start) + len("\n            }")
        end = text.index("            if (mi == 0) st.experts", start)
        retirement = text[start:end]
        self.compile_run("""
struct Wait { const unsigned* flag; unsigned value; };
unsigned service = 0, response = 7, helper = 7;
unsigned* lane_done = &service;
unsigned sq = 7;
int s = 0;
std::vector<Wait> waits;
void wait_seq(const unsigned* flag, unsigned value, int) { waits.push_back({flag, value}); }
bool reusable() {
    for (const auto& w : waits) if (*w.flag != w.value) return false;
    return true;
}
void finish(bool rhd) {
""" + retirement + """
}
int main() {
    // GPU/helper completed and the router self-answered response.seq, but the
    // service thread still reads this request/result. Neither flag retires it.
    finish(true);
    CHECK(waits.size() == 1);
    CHECK(waits[0].flag == lane_done && waits[0].value == sq);
    CHECK(response == sq && helper == sq);
    CHECK(!reusable());
    service = sq; CHECK(reusable());
    waits.clear();
    finish(false); CHECK(waits.empty()); // ordinary combined path keeps its existing wait
    return 0;
}
""")


if __name__ == "__main__":
    unittest.main()
