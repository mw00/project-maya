"""The engine's prompt-skip eligibility with layer splits; no GPU or model needed.

Compile the actual yoco_layer method against a small host-only model, so these checks exercise the
engine's predicate without making the Python CI depend on CUDA or ROCm.

    python -m unittest tools.test_ds4_prompt_skip
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PromptSkipTests(unittest.TestCase):
    def test_layer_split_eligibility(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a host C++ compiler is needed")
        source = (Path(__file__).resolve().parents[1] / "src/ds4/ds4_model.cu").read_text()
        start = source.index("    bool yoco_one_gpu() const {")
        layer = source.index("    int yoco_layer() const {", start)
        end = source.index("\n    }", layer) + len("\n    }")
        method = source[start:end]
        fixture = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Layer { int nm = 1, dev = 0, eg = -1; bool kv_src = false; };
struct Chunk { bool flow = false; };
struct Model {
    struct { int nl = 40; std::vector<int> ratios; } g;
    int chunk = 4096, mode = 1;
    std::vector<Layer> L;
    Chunk streamed{true};
    std::vector<Chunk*> chunks;
    Model() : L(40) {
        g.ratios.assign(40, 4);
        for (int i = 20; i < 40; ++i) g.ratios[(size_t)i] = 1;
        L[20].kv_src = true;
    }
    int yoco_mode() const { return mode; }
METHOD
};
void check(const Model& m, int expected, const char* name) {
    const int got = m.yoco_layer();
    if (got != expected) {
        std::fprintf(stderr, "%s: got %d, expected %d\n", name, got, expected);
        std::exit(1);
    }
}
int main() {
    Model one;
    check(one, 20, "single GPU");
    // A split at or before the source: every row still reaches that source's KV-only work.
    for (int boundary : {1, 10, 19, 20}) {
        Model split;
        for (int i = boundary; i < 40; ++i) split.L[(size_t)i].dev = 1;
        check(split, 20, "prefix split");
    }
    Model many;
    for (int i = 0; i < 40; ++i) many.L[(size_t)i].dev = i < 10 ? 0 : i < 20 ? 1 : 2;
    check(many, 20, "several prefix groups");
    // The flow streams a split's layers: only a suffix on one GPU keeps its readers' acknowledgements in step.
    for (int boundary : {1, 10, 19, 20}) {
        Model split;
        for (int i = boundary; i < 40; ++i) split.L[(size_t)i].dev = 1;
        split.chunks.push_back(&split.streamed);
        check(split, 20, "prefix split, flow");
    }
    for (int boundary : {21, 30, 39}) {
        Model split;
        for (int i = boundary; i < 40; ++i) split.L[(size_t)i].dev = 1;
        check(split, 20, "suffix split, no flow");
        split.chunks.push_back(&split.streamed);
        check(split, -1, "suffix split, flow");
    }
    for (int layer : {0, 19, 20, 39}) {
        Model tp;
        tp.L[(size_t)layer].nm = 2;
        check(tp, 20, "tensor parallel, no flow");
        tp.chunks.push_back(&tp.streamed);
        check(tp, -1, "tensor parallel, flow");
    }
    Model engram_one; engram_one.L[25].eg = 0; check(engram_one, 20, "Engram layer, one GPU");
    Model engram_split; engram_split.L[25].eg = 0;
    for (int i = 10; i < 40; ++i) engram_split.L[(size_t)i].dev = 1;
    check(engram_split, -1, "Engram layer in a split's suffix");
    Model off; off.mode = 0; check(off, -1, "disabled");
    Model fast; fast.mode = 2; check(fast, 20, "bounded replay");
    Model windows; windows.chunk = 0; check(windows, -1, "window path");
    Model missing; missing.L[20].kv_src = false; check(missing, -1, "missing source");
    Model own; own.L[30].kv_src = true; check(own, -1, "suffix owns KV");
    Model ratio; ratio.g.ratios[30] = 4; check(ratio, -1, "suffix compresses");
    Model short_meta; short_meta.g.ratios.resize(30); check(short_meta, -1, "missing ratios");
    Model no_source; no_source.g.ratios.assign(40, 4); check(no_source, -1, "no ratio-one source");
    Model first; first.g.ratios[0] = 1; first.L[0].kv_src = true; check(first, -1, "source at first layer");
}
'''.replace("METHOD", method)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            cpp = path / "prompt_skip.cpp"
            exe = path / "prompt_skip"
            cpp.write_text(fixture)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


    def test_flow_starts_at_first_chunk_with_experts(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a host C++ compiler is needed")
        source = (Path(__file__).resolve().parents[1] / "src/ds4/ds4_model.cu").read_text()
        rows_start = source.index("    static int fl_rows(")
        rows_end = source.index("\n    }", rows_start) + len("\n    }")
        start = source.index("            if (!c.fl_layers.empty()) {", source.index("    bool run_dev_chunks("))
        end = source.index("            if (c.lm)", start)
        fixture = r'''
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
struct Chunk {
    std::vector<int> fl_layers, fl_T;
    int fl_yoco = -1, fl_cut_k = 0, fl_cut_r = 0;
};
struct Run { std::vector<int> T; int yoco = -1, cut_k = 0; };
struct Model {
    int spec_k = -1, spec_il = -1, spec_T = -1;
    void flow_spec(Chunk&, int k, int il, int T) { spec_k = k; spec_il = il; spec_T = T; }
    bool flow_pump(Chunk&, std::string&) { return true; }
ROWS
    bool seed(Chunk& c, const Run& run, std::string& err) {
SEED
        return true;
    }
};
void check(int first, int yoco, int cut_k, int cut_r) {
    Run run; run.T.assign(8, 8192); run.yoco = yoco; run.cut_k = cut_k;
    Chunk c; c.fl_layers.push_back(first); c.fl_T = run.T;
    c.fl_yoco = yoco; c.fl_cut_k = cut_k; c.fl_cut_r = cut_r;
    Model m; std::string err;
    const bool suffix = yoco >= 0 && first >= yoco;
    const int expected_k = suffix ? cut_k : 0;
    const int expected_T = suffix ? 8192 - cut_r : 8192;
    if (!m.seed(c, run, err) || m.spec_k != expected_k || m.spec_il != first || m.spec_T != expected_T) {
        std::fprintf(stderr, "flow seed first=%d yoco=%d cut=%d:%d: got (%d,%d,%d), expected (%d,%d,%d)\n",
                     first, yoco, cut_k, cut_r, m.spec_k, m.spec_il, m.spec_T,
                     expected_k, first, expected_T);
        std::exit(1);
    }
}
int main() {
    check(0, -1, 0, 0);                   // no skip, one GPU
    check(20, -1, 0, 0);                  // no skip, split at source
    for (int k : {0, 1, 3, 7})
        for (int r : {0, 5524}) {
            check(0, 20, k, r);           // single GPU's first layer still has full work
            check(19, 20, k, r);          // prefix layer advances the default split's stream
            check(20, 20, k, r);          // source-first group skips leading KV-only chunks
        }
    Chunk empty; Run run; std::string err; Model m;
    if (!m.seed(empty, run, err) || m.spec_k != -1) return 1;
}
'''.replace("ROWS", source[rows_start:rows_end]).replace("SEED", source[start:end])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            cpp = path / "flow_seed.cpp"
            exe = path / "flow_seed"
            cpp.write_text(fixture)
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
