"""Guard: GitHub's macOS CI builds maya-metal only, and that build never reaches glm-dsa (GLM-5.3) code.

.github/workflows/macos.yml configures this tree with -DMAYA_ENABLE_METAL=ON and builds maya-metal and its tests:
CMakeLists.txt runs cmake/metal_backend.cmake and returns, so that build compiles src/metal, the headers it includes and
ggml from the pinned llama.cpp.  glm-dsa is built and tested on the CUDA and AMD builds only (cmake/glm_dsa.cmake,
included at the very end of CMakeLists.txt), never on a Mac, so the Metal build must not compile any of it: a slip
would first show as a red macOS CI, which no Linux or Windows build catches.  No Mac or compiler needed here:

  1. The macOS workflow configures the Metal build only (-DMAYA_ENABLE_METAL=ON): the premise of the checks below.
  2. CMakeLists.txt: include(cmake/metal_backend.cmake) and the return() right after it come before every line that
     names glm-dsa code (include(cmake/glm_dsa.cmake) among them), so a Metal configure never reaches one.
  3. cmake/metal_backend.cmake (comments aside) names no glm-dsa file or symbol.
  4. Every file the Metal build compiles - the sources cmake/metal_backend.cmake names, every source under src/metal
     and what they #include / #import "..." (transitively, resolved as the compiler does: next to the including file,
     then maya_metal's include directory src/metal) - is no glm-dsa file, and its code (comments and strings aside)
     names no glm-dsa symbol.

glm-dsa files: the port's (PORT_GLOBS / PORT_FILES of tools/test_port_portability.py), the sources the glm-dsa CMake
files name (cmake/glm_dsa.cmake, cmake/glm_dsa.d/*.cmake, tests/glm_dsa/*.cmake) and every glmd_* / glm_dsa* file.
glm-dsa symbols: glm_dsa / GlmDsa / GLM-DSA in any case or spelling, glmd_*, and the MAYA_GLM53 switch.
The rules for the builds that do compile glm-dsa (MSVC, ARM64 / POWER, POSIX-only calls) are
tools/test_port_portability.py's.  If glm-dsa comes to Metal on purpose one day, this guard goes (or narrows) with
that work.

`python tools/test_metal_build_inputs.py --list` prints the files the Metal build compiles and every finding.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_port_portability import PORT_FILES, PORT_GLOBS, strip_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/macos.yml"
METAL_CMAKE = "cmake/metal_backend.cmake"
METAL_DIR = "src/metal"
METAL_INCLUDE_DIRS = ("src/metal",)   # target_include_directories(maya_metal PUBLIC src/metal)
GLM_DSA_CMAKE = ("cmake/glm_dsa.cmake", "cmake/glm_dsa.d/*.cmake", "tests/glm_dsa/*.cmake")
SRC_EXT = {".c", ".cc", ".cpp", ".cxx", ".m", ".mm", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".cu", ".cuh"}
SRC_TOP = ("src", "include", "tests", "tools")

SYMBOL = re.compile(r"\w*glm[_-]?dsa\w*|\bglmd_\w+|\bMAYA_GLM53\b", re.I)
DSA_NAME = re.compile(r"^(glmd_|glm[_-]?dsa)", re.I)
INCLUDE = re.compile(r'^[ \t]*#[ \t]*(?:include|import)[ \t]*"([^"\n]+)"', re.M)
CMAKE_TOKEN = re.compile(r'[^\s()"]+')
SOURCE_DIR_VAR = re.compile(r"^\$\{(?:CMAKE_CURRENT_SOURCE_DIR|CMAKE_SOURCE_DIR|PROJECT_SOURCE_DIR)\}/")
METAL_INCLUDE = re.compile(r"\binclude\s*\(\s*cmake/metal_backend\.cmake\s*\)")
RETURN = re.compile(r"^\s*return\s*\(\s*\)\s*$")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def cmake_code(text: str) -> str:
    """CMake text with its comments blanked (# outside a quoted argument to the end of the line, #[[...]]); line
    numbers hold."""
    out, i, n, quoted = [], 0, len(text), False
    while i < n:
        c = text[i]
        if quoted:
            if c == "\\" and i + 1 < n:
                out.append(text[i:i + 2])
                i += 2
                continue
            quoted = c != '"'
            out.append(c)
        elif c == '"':
            quoted = True
            out.append(c)
        elif c == "#":
            m = re.match(r"#\[(=*)\[.*?\]\1\]", text[i:], re.S)
            end = i + m.end() if m else (text.find("\n", i) if "\n" in text[i:] else n)
            out.append(re.sub(r"[^\n]", " ", text[i:end]))
            i = end
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def cmake_sources(code: str) -> set:
    """The repository sources (src/, include/, tests/, tools/) a CMake text names, alone or after
    ${CMAKE_CURRENT_SOURCE_DIR}/ and the like; paths under any other variable (ggml's copy, the build tree) are not
    this repository's."""
    out = set()
    for tok in CMAKE_TOKEN.findall(code):
        tok = SOURCE_DIR_VAR.sub("", tok)
        if "$" in tok or "*" in tok or posixpath.splitext(tok)[1] not in SRC_EXT:
            continue
        tok = posixpath.normpath(tok)
        if tok.split("/")[0] in SRC_TOP:
            out.add(tok)
    return out


def registered(root: Path = ROOT) -> set:
    """The sources the glm-dsa CMake files name."""
    out = set()
    for g in GLM_DSA_CMAKE:
        for p in sorted(root.glob(g)):
            out |= cmake_sources(cmake_code(read(p)))
    return out


def is_glm_dsa(r: str, reg: set = frozenset()) -> bool:
    return (r in PORT_FILES or any(fnmatch.fnmatch(r, g) for g in PORT_GLOBS) or r in reg
            or bool(DSA_NAME.match(posixpath.basename(r))))


def metal_dir_sources(root: Path = ROOT) -> set:
    d = root / METAL_DIR
    return {p.relative_to(root).as_posix() for p in d.rglob("*") if p.suffix in SRC_EXT and p.is_file()} \
        if d.is_dir() else set()


def compiled_files(root: Path, starts, include_dirs=METAL_INCLUDE_DIRS) -> list:
    """Every repository file the given sources compile: themselves and what they #include / #import "..."
    transitively (next to the including file first, then the include directories).  Headers outside the repository
    (ggml, the SDK) and third_party/ are not followed."""
    seen, todo = set(), sorted(starts)
    while todo:
        r = todo.pop()
        if r in seen:
            continue
        seen.add(r)
        p = root / r
        if not p.is_file():
            continue
        here = posixpath.dirname(r)
        for inc in INCLUDE.findall(read(p)):
            for base in (here, *include_dirs):
                cand = posixpath.normpath(posixpath.join(base, inc))
                if cand.startswith("../") or cand == ".." or not (root / cand).is_file():
                    continue
                if not cand.startswith("third_party/"):
                    todo.append(cand)
                break
    return sorted(seen)


def cmakelists_findings(text: str) -> list:
    """[(line, what)] of check 2."""
    lines = cmake_code(text).split("\n")
    inc = next((i for i, line in enumerate(lines) if METAL_INCLUDE.search(line)), None)
    if inc is None:
        return [(0, "no include(cmake/metal_backend.cmake): where does the Metal configure start?")]
    ret = next((i for i in range(inc + 1, len(lines)) if lines[i].strip()), None)
    if ret is None or not RETURN.match(lines[ret]):
        return [(inc + 1, "include(cmake/metal_backend.cmake) is not followed by return(): the Metal configure runs "
                          "on into the rest of CMakeLists.txt")]
    return [(i + 1, "names glm-dsa code before the Metal build's return(): " + lines[i].strip())
            for i in range(ret) if SYMBOL.search(lines[i])]


def metal_build(root: Path = ROOT):
    """(the sources cmake/metal_backend.cmake names, every file the Metal build compiles)."""
    listed = cmake_sources(cmake_code(read(root / METAL_CMAKE)))
    return listed, compiled_files(root, listed | metal_dir_sources(root))


def findings(root: Path = ROOT) -> list:
    """Every finding of checks 2-4 as 'file:line: what'."""
    out = ["CMakeLists.txt:%d: %s" % f for f in cmakelists_findings(read(root / "CMakeLists.txt"))]
    reg = registered(root)
    code = cmake_code(read(root / METAL_CMAKE))
    for no, line in enumerate(code.split("\n"), 1):
        if SYMBOL.search(line):
            out.append("%s:%d: names glm-dsa code: %s" % (METAL_CMAKE, no, line.strip()))
    listed, files = metal_build(root)
    for r in files:
        if is_glm_dsa(r, reg):
            out.append("%s: a glm-dsa file the Metal build compiles%s" % (
                r, " (named in %s)" % METAL_CMAKE if r in listed else " (#included)"))
            continue
        p = root / r
        if p.is_file():
            for no, line in enumerate(strip_code(read(p)).split("\n"), 1):
                m = SYMBOL.search(line)
                if m:
                    out.append("%s:%d: names glm-dsa code in the Metal build (%s)" % (r, no, m.group(0)))
    return out


def _tree(files: dict) -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory()
    for r, text in files.items():
        p = Path(d.name) / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


GOOD_CMAKELISTS = ("project(x)\nif(MAYA_ENABLE_METAL)\n  include(cmake/metal_backend.cmake)\n"
                   "  # never glm_dsa here\n  return()\nendif()\nadd_library(k src/kernels/cuda/glmd_x.cu)\n"
                   "include(cmake/glm_dsa.cmake OPTIONAL)\n")


class MetalBuildNeverReachesGlmDsa(unittest.TestCase):
    def test_the_workflow_builds_metal_only(self):
        wf = read(ROOT / WORKFLOW)
        configures = re.findall(r"\bcmake\s+-S\b", wf)
        self.assertTrue(configures, "%s configures no CMake build" % WORKFLOW)
        self.assertEqual(len(configures), len(re.findall(r"-DMAYA_ENABLE_METAL=ON\b", wf)),
                         "%s: a CMake configure without -DMAYA_ENABLE_METAL=ON (this guard covers the Metal "
                         "build only)" % WORKFLOW)

    def test_tree(self):
        self.assertEqual([], findings(), "the Metal build reaches glm-dsa code (see this file's docstring)")

    def test_the_parse_finds_the_build(self):
        listed, files = metal_build()
        self.assertIn("src/metal/main.cpp", listed)
        self.assertEqual([], [r for r in sorted(listed) if not (ROOT / r).is_file()],
                         "%s names sources this parse does not find" % METAL_CMAKE)
        self.assertTrue(set(files) >= listed | metal_dir_sources())
        self.assertTrue(registered(), "no source found in the glm-dsa CMake files: the parse is off")


class TheGuardCatchesBreaks(unittest.TestCase):
    def test_cmakelists_order(self):
        self.assertEqual([], cmakelists_findings(GOOD_CMAKELISTS))
        self.assertEqual([1], [n for n, _ in cmakelists_findings("include(cmake/glm_dsa.cmake OPTIONAL)\n"
                                                                 + GOOD_CMAKELISTS)])
        self.assertEqual([3], [n for n, _ in cmakelists_findings(GOOD_CMAKELISTS.replace("  return()\n", ""))])
        self.assertEqual([0], [n for n, _ in cmakelists_findings("project(x)\n")])

    def test_cmake_sources_and_comments(self):
        code = cmake_code('add_library(m STATIC src/metal/a.cpp\n  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/glm_dsa_x.cpp '
                          '${_copy}/src/ggml.c)  # src/core/glmd_y.cu in a comment\n'
                          'message(STATUS "a # in a string")\n#[[ glm_dsa\n in a bracket comment ]]\n')
        self.assertEqual({"src/metal/a.cpp", "src/core/glm_dsa_x.cpp"}, cmake_sources(code))
        self.assertEqual([2], [no for no, line in enumerate(code.split("\n"), 1) if SYMBOL.search(line)])
        self.assertIn("a # in a string", code)
        self.assertTrue(is_glm_dsa("src/core/glm_dsa_x.cpp"))
        self.assertTrue(is_glm_dsa("include/strata/core/glm_arch.hpp"))     # a port file (PORT_FILES)
        self.assertTrue(is_glm_dsa("src/core/new_part.cpp", {"src/core/new_part.cpp"}))   # registered
        self.assertFalse(is_glm_dsa("src/metal/glm_metal.cpp"))

    def test_symbols_in_code_only(self):
        code = strip_code('// glm_dsa in a comment\nconst char *m = "MAYA_GLM53 in a string";\nint glmd_rows = 0;\n'
                          'bool GlmDsa = false;\nint glm_metal = 0;\n')
        self.assertEqual([3, 4], [no for no, line in enumerate(code.split("\n"), 1) if SYMBOL.search(line)])

    def test_an_include_chain_into_glm_dsa(self):
        files = {
            "CMakeLists.txt": GOOD_CMAKELISTS,
            METAL_CMAKE: "add_library(maya_metal STATIC src/metal/a.cpp)\n",
            "src/metal/a.cpp": '#include "b.hpp"\n#include "ggml.h"\n',
            "src/metal/b.hpp": '#include "../../include/strata/platform/c.hpp"   // header only\n',
            "include/strata/platform/c.hpp": '#include "../core/glm_arch.hpp"\n',
            "include/strata/core/glm_arch.hpp": "enum class Arch { Flash, GlmDsa };\n",
        }
        with _tree(files) as d:
            root = Path(d)
            self.assertEqual(sorted(set(files) - {"CMakeLists.txt", METAL_CMAKE}), metal_build(root)[1])
            self.assertEqual(["include/strata/core/glm_arch.hpp: a glm-dsa file the Metal build compiles (#included)"],
                             findings(root))
        files["include/strata/platform/c.hpp"] = "inline bool glmd_on() { return false; }\n"
        with _tree(files) as d:
            self.assertEqual(["include/strata/platform/c.hpp:1: names glm-dsa code in the Metal build (glmd_on)"],
                             findings(Path(d)))
        files[METAL_CMAKE] = "add_library(maya_metal STATIC src/metal/a.cpp src/core/glm_dsa_geometry.cpp)\n"
        files["include/strata/platform/c.hpp"] = "\n"
        files["src/core/glm_dsa_geometry.cpp"] = "\n"
        with _tree(files) as d:
            self.assertEqual(["cmake/metal_backend.cmake:1: names glm-dsa code: add_library(maya_metal STATIC "
                              "src/metal/a.cpp src/core/glm_dsa_geometry.cpp)",
                              "src/core/glm_dsa_geometry.cpp: a glm-dsa file the Metal build compiles (named in "
                              "cmake/metal_backend.cmake)"], findings(Path(d)))


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(metal_build()[1]))
        print("\n".join(findings()) or "no findings")
    else:
        unittest.main()
