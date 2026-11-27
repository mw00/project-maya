"""tools/test_changed.py - tools/maya_changed.py: changed paths -> engines, models and the test plan (no git, no GPU).

    python -m unittest tools.test_changed -v
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import maya_changed as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILT = None


def built():
    global BUILT
    BUILT = BUILT or C.engine_files(ROOT)
    return BUILT


class CMakeReading(unittest.TestCase):
    TEXT = """
    # add_executable(ghost src/ghost/a.cpp)   a comment is not a command
    set(_lane src/b/lane.cpp src/a/rows.cpp)
    add_library(kern STATIC src/k/k1.cu
                src/k/k2.cu)
    add_executable(eng-b src/b/main.cpp ${_lane})
    if(X)
      target_sources(eng-b PRIVATE src/a/extra.cu)
    endif()
    target_link_libraries(eng-b PRIVATE kern Threads::Threads $<$<BOOL:1>:m>)
    """

    def test_sources_variables_and_links(self):
        src, link = C.cmake_targets(self.TEXT)
        self.assertNotIn("ghost", src)
        self.assertEqual(src["eng-b"], ["src/b/main.cpp", "src/b/lane.cpp", "src/a/rows.cpp", "src/a/extra.cu"])
        self.assertIn("kern", link["eng-b"])
        self.assertEqual(C.target_sources("eng-b", src, link),
                         {"src/b/main.cpp", "src/b/lane.cpp", "src/a/rows.cpp", "src/a/extra.cu", "src/k/k1.cu",
                          "src/k/k2.cu"})

    def test_include_closure_follows_quoted_includes(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            for rel, text in {"src/b/main.cpp": '#include "b.hpp"\n#include "maya/x.hpp"\n#include <vector>\n',
                              "src/b/b.hpp": '#include "../a/os.hpp"\n', "src/a/os.hpp": '#include "deep.hpp"\n',
                              "src/a/deep.hpp": "", "src/a/unused.hpp": "", "include/maya/x.hpp": ""}.items():
                (r / rel).parent.mkdir(parents=True, exist_ok=True)
                (r / rel).write_text(text)
            got = C.include_closure(["src/b/main.cpp"], r)
            self.assertEqual(got, {"src/b/main.cpp", "src/b/b.hpp", "src/a/os.hpp", "src/a/deep.hpp",
                                   "include/maya/x.hpp"})


class RealTree(unittest.TestCase):
    def test_every_engine_is_found_with_its_main(self):
        b = built()
        for e, main in {"maya": "src/program/generate.cpp", "maya-ds4": "src/ds4/ds4_main.cpp",
                        "maya-mm3": "src/mm3/mm3_main.cpp", "maya-hy4": "src/hy4/hy4_main.cpp",
                        "maya-ik2": "src/ik2/ik2_main.cu"}.items():
            self.assertIn(main, b[e], e)
            self.assertGreater(len(b[e]), 20, e)

    def test_own_folder_means_own_model(self):
        b = built()
        self.assertEqual(C.classify("src/ik2/ik2_main.cu", b)[0], ["Inkling"])
        self.assertEqual(C.classify("src/ds4/ds4_model.cu", b)[0], ["DeepSeek"])
        self.assertEqual(C.classify("src/mm3/mm3_main.cpp", b)[0], ["MiniMax"])
        self.assertEqual(C.classify("src/hy4/hy4_main.cpp", b)[0], ["Hy4"])
        self.assertEqual(C.classify("src/core/glm_model.cu", b)[0], ["GLM", "Qwen"])
        for e, (dirs, models) in C.ENGINES.items():       # (a new file in an engine's folder: its engine at least)
            got = C.classify(dirs[0] + "/brand_new_file.cpp", b)
            self.assertTrue(set(models) <= set(got[0]) and e in got[2], e)

    def test_a_file_another_engine_is_built_from_touches_it_too(self):
        b = built()
        self.assertEqual(set(C.classify("src/hy4/hy4_lane.cpp", b)[0]), {"Hy4", "Inkling"})    # maya-ik2's CPU lane
        self.assertLessEqual({"DeepSeek", "MiniMax", "Hy4"}, set(C.classify("src/ds4/ds4_os.hpp", b)[0]))
        self.assertLessEqual({"DeepSeek", "Hy4", "Inkling"}, set(C.classify("src/ds4/ds4_iq_cpu.cpp", b)[0]))

    def test_shared_python_and_build_files_touch_all(self):
        b = built()
        for p in ("serve/server.py", "maya.py", "tools/maya_bench.py", "CMakeLists.txt", "cmake/build_id.cmake",
                  "third_party/ggml/x.h", "some/new/place.bin"):
            self.assertEqual(C.classify(p, b)[0], list(C.MODELS), p)
        self.assertEqual(C.classify("CMakeLists.txt", b)[2], list(C.ENGINES))
        self.assertEqual(C.classify("serve/server.py", b)[2], [])

    def test_named_python_and_data_touch_their_model(self):
        b = built()
        self.assertEqual(C.classify("tools/hy4/pack.py", b)[0], ["Hy4"])
        self.assertEqual(C.classify("serve/ds4_vision.py", b)[0], ["DeepSeek"])
        self.assertEqual(C.classify("data/expert-profile-ik2mini.bin", b)[0], ["Inkling"])
        self.assertEqual(C.classify("tools/calibrate_mm3_prefill.py", b)[0], ["MiniMax"])

    def test_tests_docs_and_ci_touch_no_model(self):
        b = built()
        for p, kind in (("tests/ds4_pcie_test.cpp", "test"), ("tools/test_bench.py", "test"),
                        ("serve/test_fair.py", "test"), ("src/spec/controller_test.cpp", "test"),
                        ("tools/maya_nightly.py", "test"), ("docs/SETUP.md", "none"), ("README.md", "none"),
                        (".github/workflows/tests.yml", "none"), ("src/metal/serve.cpp", "none")):
            self.assertEqual(C.classify(p, b)[:2], ([], kind), p)


class Plan(unittest.TestCase):
    def test_one_engine_change(self):
        pl = C.plan(["src/ik2/ik2_main.cu", "docs/INKLING2.md", "tests/ds4_pcie_test.cpp"], built())
        self.assertEqual(list(pl["full"]), ["Inkling"])
        self.assertEqual(pl["smoke"], ["GLM", "Qwen", "DeepSeek", "MiniMax", "Hy4"])
        self.assertEqual(pl["builds"], ["maya-ik2"])
        self.assertEqual(pl["cpp_tests"], ["tests/ds4_pcie_test.cpp"])
        self.assertTrue(any("--bench-text-ref" in n for n in pl["notes"]))
        text = "\n".join(C.plan_text(pl, "v1.1.76"))
        self.assertIn("FULL   Inkling   (maya-ik2): src/ik2/ik2_main.cu", text)
        self.assertIn("SMOKE  Hy4", text)
        self.assertIn("build: maya-ik2", text)
        self.assertIn("unittest discover -s serve", text)

    def test_docs_only_builds_nothing(self):
        pl = C.plan(["docs/SETUP.md", "CHANGELOG.md"], built())
        self.assertEqual((pl["full"], pl["builds"]), ({}, []))
        self.assertEqual(pl["smoke"], list(C.MODELS))
        self.assertIn("build: no engine", "\n".join(C.plan_text(pl, "v1")))

    def test_server_change_is_full_for_all_and_builds_nothing(self):
        pl = C.plan(["serve/server.py"], built())
        self.assertEqual(list(pl["full"]), list(C.MODELS))
        self.assertEqual(pl["builds"], [])


if __name__ == "__main__":
    unittest.main()
