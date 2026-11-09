"""Portability lint: GNU-only C++ where MSVC compiles (every source); x86-only code outside an x86 guard and POSIX-only
calls (glm-dsa files).

Windows NVIDIA builds compile every source's host side with MSVC (maya.py runs vcvars64 + nvcc, cl.exe is the host
compiler), and nvcc's own front end then runs in MSVC mode for the device side too.  Linux builds use GCC and the AMD
build's compiler is ROCm's clang, both of which take GNU-only constructs, so such a construct builds everywhere we test
and stops every Windows NVIDIA build (glmd_mv_dots.cuh's h2f() used __builtin_memcpy: nvcc 12.8 + MSVC 19.44 say
'identifier "__builtin_memcpy" is undefined').  No compiler needed here; the rules follow what nvcc 12.8 + cl.exe
19.44 (VS 2022) accept, measured on a Windows PC:

  1. Every source under src/, include/ and tests/: no GNU-only construct in code nvcc / cl.exe compile on Windows -
     __builtin_* (except the ones MSVC and nvcc both take: ALLOWED_BUILTINS, and CUDA's hint builtins in device code:
     DEVICE_BUILTINS), __attribute__ (refused in device code too), the __atomic_* / __sync_* builtins, typeof,
     __int128, __PRETTY_FUNCTION__, and inline asm in host code (PTX asm in device code is fine).
     Not checked: preprocessor branches Windows NVIDIA builds never take (defined(__GNUC__) / __clang__ / __linux__ /
     __APPLE__, the AMD build's STRATA_USE_HIP / __HIPCC__ / __HIP_DEVICE_COMPILE__, GCC's arch macros such as
     __x86_64__ that MSVC never defines, !defined(_MSC_VER) / !defined(_WIN32), #if 0, and macros only ever defined
     inside such a branch), the AMD build's own folders (MSVC_SKIP_DIRS) and the sources CMake never builds on
     Windows (MSVC_SKIP_FILES).  A CUDA device branch (__CUDA_ARCH__) is NOT exempt: nvcc's MSVC-mode front end
     refuses __builtin_memcpy / __attribute__ there as well.
  2. The glm-dsa port's files (PORT_GLOBS / PORT_FILES): x86-only code inside an x86 guard, for aarch64 Linux and
     Apple silicon - <immintrin.h> and the other <*intrin.h> under #if defined(__x86_64__) || defined(_M_X64) ...
     (or a macro only defined under one, such as STRATA_X86); in cmake/glm_dsa.d, -mavx* / -mf16c / -mfma /
     -march= / /arch:AVX* under an x86 condition (STRATA_CPU_X86, CMAKE_SYSTEM_PROCESSOR MATCHES x86/AMD64, MSVC:
     Maya's Windows builds are x64).  X86_PENDING: files with known findings not fixed yet - the test fails once a
     listed file is clean, so the list only shrinks.
  3. The port's files: no POSIX-only C library call or header where Windows compiles (setenv, unsetenv, usleep,
     <unistd.h>, pthread_*, mmap ...: MSVC's CRT has none of them; the AMD build compiles with the same CRT, but not
     the CUDA-only files).  setenv / unsetenv / getpid are fine in files that include tests/glm_dsa/glm_test_env.hpp,
     the Windows mapping.

`python tools/test_port_portability.py --list` prints every finding (pending ones included).
"""
from __future__ import annotations

import fnmatch
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIRS = ("src", "include", "tests")
SRC_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".inl", ".ipp"}

# --- check 1 --------------------------------------------------------------------------------------------------------
MSVC_SKIP_DIRS = ("include/strata/hip_compat/", "tests/hip/")   # the AMD build's only (ROCm clang compiles them)
MSVC_SKIP_FILES = {
    "tests/mm3_lane_test.cpp": "CMakeLists.txt builds it only if(NOT WIN32)",
}
# taken by nvcc 12.8 (host and device code) and cl.exe 19.44 alike
ALLOWED_BUILTINS = {"__builtin_bit_cast", "__builtin_addressof", "__builtin_offsetof", "__builtin_huge_val",
                    "__builtin_huge_valf", "__builtin_nan", "__builtin_nanf"}
# CUDA's documented hint builtins: fine in device code on Windows too, refused by cl.exe in host code
DEVICE_BUILTINS = {"__builtin_expect", "__builtin_assume", "__builtin_assume_aligned", "__builtin_unreachable"}
GNU_ONLY = [
    (re.compile(r"\b__attribute__\b"), "__attribute__"),
    (re.compile(r"\b__atomic_\w+\s*\("), "GNU __atomic builtin (std::atomic works everywhere)"),
    (re.compile(r"\b__sync_\w+\s*\("), "GNU __sync builtin (std::atomic works everywhere)"),
    (re.compile(r"\b__typeof__\b|\btypeof\s*\("), "typeof (decltype works everywhere)"),
    (re.compile(r"\b__u?int128(_t)?\b"), "__int128"),
    (re.compile(r"\b__PRETTY_FUNCTION__\b"), "__PRETTY_FUNCTION__"),
    # GCC / clang intrinsic names MSVC lacks (MSVC: _mm_cvtph_ps / _mm_cvtps_ph, __cpuidex)
    (re.compile(r"\b(_cvtsh_ss|_cvtss_sh|__get_cpuid\w*|__cpuid_count)\s*\("), "GCC-only intrinsic"),
]
HOST_ASM = re.compile(r"\b__asm__\b|\basm\s+(volatile|__volatile__)\b|\basm\s*\(")
BUILTIN = re.compile(r"\b__builtin_\w+")
ANY_GNU = re.compile(r"__builtin_|__attribute__|__atomic_|__sync_|typeof|__u?int128|__PRETTY_FUNCTION__|"
                     r"\basm\b|__asm__|_cvtsh_ss|_cvtss_sh|__get_cpuid|__cpuid_count")
# defined: Windows NVIDIA builds (nvcc + cl.exe) never compile this branch
NON_MSVC = {
    "__GNUC__", "__clang__", "__linux__", "__linux", "__gnu_linux__", "__unix__", "__unix", "__APPLE__", "__MACH__",
    "__FreeBSD__", "__HIP_DEVICE_COMPILE__", "__HIP__", "__HIPCC__", "__HIP_PLATFORM_AMD__", "__HIP_PLATFORM_HCC__",
    "STRATA_USE_HIP", "__x86_64__", "__x86_64", "__amd64__", "__amd64", "__i386__", "__i386", "__aarch64__",
    "__arm__", "__powerpc64__", "__ppc64__", "__PPC64__",
}
AMD_GPU = re.compile(r"__(gfx|GFX)\w*__$|__AMDGCN\w*__$|__AMDGPU\w*__$")   # __gfx906__, __GFX11__: the AMD build's
MSVC_ONLY = {"_MSC_VER", "_WIN32", "_WIN64"}   # defined: only Windows compiles this branch (negated: never)
DEVICE = {"__CUDA_ARCH__", "__HIP_DEVICE_COMPILE__"}   # defined: a device pass

# --- check 2 --------------------------------------------------------------------------------------------------------
X86_HEADER = re.compile(r"^\s*#\s*include\s*<\s*(immintrin|x86intrin|xmmintrin|emmintrin|pmmintrin|tmmintrin|"
                        r"smmintrin|nmmintrin|wmmintrin|ammintrin|avxintrin|avx2intrin|avx512fintrin|f16cintrin|"
                        r"fmaintrin|cpuid)\.h\s*>")
X86 = {"__x86_64__", "__x86_64", "__amd64__", "__amd64", "_M_X64", "_M_AMD64", "__i386__", "__i386", "_M_IX86",
       "__AVX__", "__AVX2__", "__AVX512F__", "__F16C__", "__FMA__", "__SSE2__", "__SSE3__", "__SSSE3__", "__SSE4_1__",
       "__SSE4_2__", "_MSC_VER", "_WIN64", "STRATA_X86"}
CMAKE_X86_FLAG = re.compile(r"(?<![\w-])(-mavx\w*|-mf16c|-mfma\b|-mbmi\w*|-march=[^\s;\")]+|/arch:AVX\w*)", re.I)

# --- check 3 --------------------------------------------------------------------------------------------------------
# POSIX-only C library calls and headers (MSVC's CRT has none of these) in the port's files, outside the branches
# Windows never takes.  setenv / unsetenv / getpid are fine where tests/glm_dsa/glm_test_env.hpp (the Windows mapping)
# is included; CMake builds glm_cpu_bench.cpp only if(NOT WIN32).
POSIX_CALL = re.compile(r"(?<![\w.>:])(?:::)?(setenv|unsetenv|getpid|strcasecmp|strncasecmp|clock_gettime|usleep|"
                        r"nanosleep|posix_memalign|aligned_alloc|strtok_r|localtime_r|gmtime_r|realpath|sysconf|mmap|"
                        r"munmap|madvise|mlock|munlock|getpagesize|sched_getaffinity|sched_setaffinity|sched_getcpu|"
                        r"pthread_\w+|dlopen|dlsym|gettimeofday|fork|pread|pwrite|ftruncate|fsync)\s*\(")
POSIX_HEADER = re.compile(r"^\s*#\s*include\s*<(unistd|pthread|dlfcn|sys/(mman|time|resource|sysinfo|syscall|ioctl|"
                          r"wait|socket|uio|prctl|statvfs|vfs|file|param)|sched|numa|numaif|dirent|strings|poll|"
                          r"execinfo|libgen|fnmatch|glob|getopt|termios|linux/\w+|endian|byteswap|alloca)\.h>")
TEST_ENV_SHIM = "tests/glm_dsa/glm_test_env.hpp"
SHIM_INCLUDE = re.compile(r'^\s*#\s*include\s*"(?:[\w./]*/)?glm_test_env\.hpp"', re.M)   # (on the raw text)
SHIMMED = {"setenv", "unsetenv", "getpid"}

# The glm-dsa port's files: every file the port added matches one of these.
PORT_GLOBS = (
    "src/kernels/cuda/glmd_*", "include/strata/kernels/glmd_*", "src/core/glm_dsa_*", "tests/glm_dsa/*",
    "cmake/glm_dsa.d/*.cmake",
)
PORT_FILES = (
    "include/strata/core/glm_arch.hpp", "include/strata/core/glm_cpu_affinity.hpp",
    "include/strata/core/glm_draft_adaptive.hpp", "include/strata/core/glm_env.hpp",
    "include/strata/core/glm_heat_evict.hpp", "include/strata/core/glm_maya_sync.hpp",
    "include/strata/core/glm_mtp_state.hpp", "include/strata/core/glm_ngram_draft.hpp",
    "include/strata/core/glm_pool_classes.hpp", "include/strata/core/glm_prompt_cache.hpp",
    "include/strata/core/glm_req_knobs.hpp", "include/strata/core/glm_state_layout.hpp",
    "include/strata/core/glm_warm_seq.hpp", "include/strata/kernels/cpu/glm_lane_row4.hpp",
    "include/strata/prefill/moe_mmq_strata.hpp", "src/core/glm_cb_dump.hpp", "src/core/glm_geometry_check.cpp",
    "src/core/glm_guards_host.cpp", "src/core/glm_heat_evict.cpp", "src/core/glm_maya_sync.cpp",
    "src/core/glm_mtp_state.cpp", "src/core/glm_mtp_state_model.cu", "src/core/glm_ngram_draft.cpp",
    "src/core/glm_pool_classes.cpp", "src/core/glm_prefill_dsa.cu", "src/core/glm_prefill_dsa.hpp",
    "src/core/glm_prefill_types.hpp", "src/core/glm_prompt_cache.cpp", "src/core/glm_prompt_cache_engine.cpp",
    "src/core/glm_prompt_cache_model.cu", "src/core/glm_req_knobs.cpp", "src/core/glm_residual.cu",
    "src/core/glm_residual.hpp", "src/core/glm_state_bind.cpp", "src/core/glm_state_layout.cpp",
    "src/core/glm_tier_lane.cpp", "src/core/glm_tier_lane.hpp", "src/core/glm_tier_numa.cu",
    "src/core/glm_tier_numa.hpp", "src/core/glm_tier_plan.cpp", "src/core/glm_tier_plan.hpp",
    "src/core/glm_warm_seq.cpp", "src/core/glm_warm_seq_model.cu", "src/kernels/cpu/glm_lane_row4.cpp",
    "src/kernels/cpu/glm_lane_row4_avx2.cpp", "src/kernels/cpu/glm_lane_row4_x86.hpp",
    "src/kernels/cuda/iq_kquant.cuh", "src/prefill/moe_mmq_strata.cu",
)
# port files with x86 findings known and not fixed yet (the test fails once one of them is clean: take it out then).
# Empty since lane C item 2: glm_lane_row4.cpp / glm_lane_row4_avx2.cpp keep <immintrin.h> / <cpuid.h> under an x86-64
# guard and 70-glm-lane-row4.cmake gives the AVX2 flags only under STRATA_CPU_X86.
X86_PENDING: dict = {}


# ---------------------------------------------------------------------------------------------------------------------
_LITERALS = re.compile(
    r"//[^\n]*"                                       # line comment
    r"|/\*.*?(?:\*/|\Z)"                              # block comment
    r"|\bR\"([^()\\ \n]{0,16})\(.*?(?:\)\1\"|\Z)"     # raw string R"x(...)x"
    r"|\"(?:\\.|[^\"\\\n])*\"?"                       # string literal
    r"|(?<![0-9A-Fa-f])'(?:\\.|[^'\\\n])*'",          # char literal (not a digit separator: 1'000'000)
    re.S)


def strip_code(text: str) -> str:
    """Comments and string / char literals blanked (newlines kept, so line numbers hold)."""
    return _LITERALS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def joined_lines(text: str):
    """(line number, logical line) with backslash continuations joined; continued lines yield ''."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        start, buf = i, lines[i]
        while buf.endswith("\\") and i + 1 < len(lines):
            i += 1
            buf = buf[:-1] + " " + lines[i]
        yield start + 1, buf
        for k in range(start + 1, i + 1):
            yield k + 1, ""
        i += 1


def _directive(line: str):
    m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif|define)\b\s*(.*)", line)
    return (m.group(1), m.group(2).strip()) if m else (None, None)


class Cond:
    """A preprocessor condition parsed and reduced to E: 'this branch is only compiled when one of POS is defined
    (or one of NEG is not)'.  Unknown pieces count as 'could be anything'."""

    def __init__(self, pos: set, neg: set = frozenset(), pos_re=None):
        self.pos, self.neg, self.pos_re = pos, neg, pos_re

    def parse(self, s: str):
        self.toks = re.findall(r"defined\s*\(\s*\w+\s*\)|defined\s+\w+|\w+|&&|\|\||[<>=!]=|[!()<>]|\S", s)
        self.k = 0
        try:
            node = self._or()
        except IndexError:
            return ("?",)
        return node if self.k == len(self.toks) else ("?",)

    def _peek(self):
        return self.toks[self.k] if self.k < len(self.toks) else None

    def _or(self):
        a = [self._and()]
        while self._peek() == "||":
            self.k += 1
            a.append(self._and())
        return a[0] if len(a) == 1 else ("or", a)

    def _and(self):
        a = [self._unary()]
        while self._peek() == "&&":
            self.k += 1
            a.append(self._unary())
        return a[0] if len(a) == 1 else ("and", a)

    def _unary(self):
        if self._peek() == "!":
            self.k += 1
            return ("not", self._unary())
        a = self._prim()
        op = self._peek()
        if op in ("==", ">=", ">", "<=", "<", "!="):
            self.k += 1
            b = self._prim()
            # NAME >= 700, NAME > 0, NAME == 3: false while NAME is undefined (0); NAME < 610 is not
            if a[0] == "def" and b[0] == "num" and (op == ">" or (op == ">=" and b[1] != "0") or
                                                     (op == "==" and b[1] != "0")):
                return ("def", a[1])
            return ("?",)
        return a

    def _prim(self):
        t = self.toks[self.k]
        self.k += 1
        if t == "(":
            node = self._or()
            if self._peek() == ")":
                self.k += 1
            return node
        m = re.match(r"defined\s*\(?\s*(\w+)", t)
        if m:
            return ("def", m.group(1))
        if re.fullmatch(r"\d+[uUlL]*", t):
            return ("num", t.rstrip("uUlL"))
        if re.fullmatch(r"[A-Za-z_]\w*", t):
            if self._peek() == "(":   # a function-like macro: unknown
                depth = 0
                while self.k < len(self.toks):
                    depth += {"(": 1, ")": -1}.get(self.toks[self.k], 0)
                    self.k += 1
                    if depth == 0:
                        break
                return ("?",)
            return ("def", t)   # a bare macro: true only when defined and non-zero
        return ("?",)

    def e(self, node, negated=False) -> bool:
        kind = node[0]
        if kind == "not":
            return self.e(node[1], not negated)
        if kind in ("and", "or"):
            parts = [self.e(x, negated) for x in node[1]]
            return any(parts) if (kind == "and") != negated else all(parts)   # De Morgan
        if kind == "num":
            return (node[1] == "0") != negated   # #if 0 (or the #else of #if 1)
        if kind == "def":
            if negated:
                return node[1] in self.neg
            return node[1] in self.pos or bool(self.pos_re and self.pos_re.match(node[1]))
        return False


def branch_flags(text: str, pos: set, neg: set = frozenset(), pos_re=None) -> list:
    """Per line (1-based): inside a preprocessor branch only compiled when one of pos (or a name pos_re matches) is
    defined, or one of neg is not."""
    cond = Cond(pos, neg, pos_re)
    out = [False] * (text.count("\n") + 3)
    stack = []   # per open #if: [conditions of the branches before, node of this branch, its E]
    for no, line in joined_lines(text):
        kind, arg = _directive(line)
        if kind in ("if", "ifdef", "ifndef"):
            name = (arg.split() or ["?"])[0]
            node = cond.parse(arg) if kind == "if" else (("def", name) if kind == "ifdef" else ("not", ("def", name)))
            stack.append([[], node, cond.e(node)])
        elif kind in ("elif", "else") and stack:
            prev = stack[-1][0] + [stack[-1][1]]
            node = ("and", [("not", p) for p in prev] + ([cond.parse(arg)] if kind == "elif" else []))
            stack[-1] = [prev, node, cond.e(node)]
        elif kind == "endif" and stack:
            stack.pop()
        out[no] = any(f[2] for f in stack)
    return out


def derived(texts, pos: set, neg: set = frozenset(), pos_re=None) -> set:
    """Object-like macros #defined (to non-zero) only inside branches that give E - e.g. GLMD_MVQ_DEVICE (1 under
    defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__), 0 otherwise) or STRATA_X86."""
    inside, outside = set(), set()
    for text in texts:
        if "define" not in text:
            continue
        flags = branch_flags(text, pos, neg, pos_re)
        for no, line in joined_lines(text):
            kind, arg = _directive(line)
            m = re.match(r"(\w+)(?:\s+(.*))?$", arg or "") if kind == "define" else None
            if not m or (m.group(2) or "").strip() in ("0", "false"):
                continue
            (inside if flags[no] else outside).add(m.group(1))
    return inside - outside


DEVICE_QUAL = re.compile(r"\b(__global__|__device__)\b")
HOST_QUAL = re.compile(r"\b__host__\b")


def device_only_macros(texts) -> set:
    """Qualifier macros that expand to __device__ / __global__ without __host__ (a function body follows them)."""
    out = set()
    for text in texts:
        for _, line in joined_lines(text):
            kind, arg = _directive(line)
            m = re.match(r"(\w+)\s+(.*)$", arg or "") if kind == "define" else None
            if m and DEVICE_QUAL.search(m.group(2)) and not HOST_QUAL.search(m.group(2)) and "{" not in m.group(2):
                out.add(m.group(1))
    return out


def device_bodies(code: str, qual_macros: set) -> set:
    """Lines inside the bodies of __global__ / __device__ (not __host__) functions and lambdas (code: stripped)."""
    macro = re.compile(r"\b(" + "|".join(sorted(qual_macros)) + r")\b") if qual_macros else None
    if not DEVICE_QUAL.search(code) and not (macro and macro.search(code)):
        return set()
    src = code.split("\n")
    pp = set()   # preprocessor lines (continuations included) stay out of the brace scan
    for no, line in joined_lines(code):
        if line.lstrip().startswith("#"):
            k = no
            pp.add(k)
            while src[k - 1].endswith("\\") and k < len(src):
                k += 1
                pp.add(k)
    text = "\n".join("" if i + 1 in pp else s for i, s in enumerate(src))
    lines, depth, start, line, dev_depth = set(), 0, 0, 1, None
    for m in re.finditer(r"[{};\n]", text):
        c, i = m.group(0), m.start()
        if c == "\n":
            line += 1
            if dev_depth is not None:
                lines.add(line)
        elif c == "{":
            if dev_depth is None:
                head = text[start:i]
                if not HOST_QUAL.search(head) and (DEVICE_QUAL.search(head) or (macro and macro.search(head))):
                    dev_depth = depth
                    lines.add(line)
            depth += 1
            start = i + 1
        elif c == "}":
            depth -= 1
            if dev_depth is not None and depth <= dev_depth:
                dev_depth = None
            start = i + 1
        else:
            start = i + 1
    return lines


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def is_port(r: str) -> bool:
    return r in PORT_FILES or any(fnmatch.fnmatch(r, g) for g in PORT_GLOBS)


_TREE = {}


def tree():
    """The tree's sources (stripped) and the macros derived from them, read once."""
    if not _TREE:
        texts, shimmed = {}, set()
        for d in SRC_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.suffix in SRC_EXT and p.is_file():
                    raw = p.read_text(encoding="utf-8", errors="replace")
                    texts[rel(p)] = strip_code(raw)
                    if SHIM_INCLUDE.search(raw):
                        shimmed.add(rel(p))
        _TREE.update(texts=texts, shimmed=shimmed,
                     non_msvc=NON_MSVC | derived(texts.values(), NON_MSVC, MSVC_ONLY, AMD_GPU),
                     device=DEVICE | derived(texts.values(), DEVICE), x86=X86 | derived(texts.values(), X86),
                     dev_quals=device_only_macros(texts.values()))
    return _TREE


def gnu_findings(r: str, code: str, t=None):
    """[(line, what)] of check 1 in one stripped source r (repo-relative path)."""
    t = t or tree()
    if r in MSVC_SKIP_FILES or any(r.startswith(d) for d in MSVC_SKIP_DIRS) or not ANY_GNU.search(code):
        return []
    skip = branch_flags(code, t["non_msvc"], MSVC_ONLY, AMD_GPU)
    dev = branch_flags(code, t["device"])
    bodies = device_bodies(code, t["dev_quals"])
    found = []
    for no, line in enumerate(code.split("\n"), 1):
        if skip[no] or not line.strip():
            continue
        device = dev[no] or no in bodies
        for m in BUILTIN.finditer(line):
            name = m.group(0)
            if name not in ALLOWED_BUILTINS and not (device and name in DEVICE_BUILTINS):
                found.append((no, name))
        for rx, what in GNU_ONLY:
            if rx.search(line):
                found.append((no, what))
        if not device and HOST_ASM.search(line):
            found.append((no, "inline asm in host code"))
    return found


def posix_findings(r: str, code: str, t=None, shimmed=None):
    """[(line, what)] of check 3 in one stripped port source r (shimmed: it includes glm_test_env.hpp)."""
    if r == TEST_ENV_SHIM or r in MSVC_SKIP_FILES or any(r.startswith(d) for d in MSVC_SKIP_DIRS):
        return []
    if not POSIX_CALL.search(code) and not re.search(r"#\s*include\s*<", code):
        return []
    t = t or tree()
    skip = branch_flags(code, t["non_msvc"], MSVC_ONLY, AMD_GPU)
    if shimmed is None:
        shimmed = r in t["shimmed"]
    found = []
    for no, line in enumerate(code.split("\n"), 1):
        if skip[no]:
            continue
        m = POSIX_HEADER.match(line)
        if m and not (shimmed and m.group(1) == "unistd"):
            found.append((no, "<%s.h>" % m.group(1)))
        for c in POSIX_CALL.finditer(line):
            if not (shimmed and c.group(1) in SHIMMED):
                found.append((no, c.group(1) + "()"))
    return found


def x86_findings(code: str, t=None):
    if "intrin.h" not in code and "cpuid.h" not in code:
        return []
    t = t or tree()
    ok = branch_flags(code, t["x86"])
    return [(no, "<%s.h>" % m.group(1)) for no, line in enumerate(code.split("\n"), 1)
            for m in [X86_HEADER.match(line)] if m and not ok[no]]


def cmake_x86_findings(text: str):
    """-mavx* etc. outside an x86 condition in one CMake file."""
    stack, found = [], []
    for no, raw in enumerate(text.split("\n"), 1):
        line = re.sub(r"#.*", "", raw) if '"' not in raw else raw
        m = re.match(r"\s*(if|elseif|else|endif)\s*\((.*)\)\s*$", line, re.I)
        if m:
            kind, arg = m.group(1).lower(), m.group(2)
            x86 = bool(re.search(r"(?<!NOT )\bSTRATA_CPU_X86\b", arg) or
                       re.search(r"(?<!NOT )CMAKE_SYSTEM_PROCESSOR\s+MATCHES\s+\S*(x86|AMD64|i686)", arg, re.I) or
                       re.fullmatch(r"\s*MSVC\s*", arg))
            if kind == "if":
                stack.append(x86)
            elif kind == "elseif" and stack:
                stack[-1] = x86
            elif kind == "else" and stack:
                stack[-1] = False
            elif kind == "endif" and stack:
                stack.pop()
            continue
        found += [(no, f.group(1)) for f in CMAKE_X86_FLAG.finditer(line) if not any(stack)]
    return found


def all_findings():
    t = tree()
    if "findings" not in t:
        gnu = [(r,) + f for r, code in t["texts"].items() for f in gnu_findings(r, code, t)]
        x86 = [(r,) + f for r, code in t["texts"].items() if is_port(r) for f in x86_findings(code, t)]
        for p in sorted((ROOT / "cmake" / "glm_dsa.d").glob("*.cmake")):
            x86 += [(rel(p),) + f for f in cmake_x86_findings(p.read_text(encoding="utf-8", errors="replace"))]
        posix = [(r,) + f for r, code in t["texts"].items() if is_port(r) for f in posix_findings(r, code, t)]
        t["findings"] = (gnu, x86, posix)
    return t["findings"]


def scan_snippet(text: str, path="src/x.cu"):
    """Check 1 on a snippet with the tree's derived macros (for the tests below)."""
    t = dict(tree())
    code = strip_code(text)
    t["non_msvc"] = t["non_msvc"] | derived([code], NON_MSVC, MSVC_ONLY, AMD_GPU)
    t["device"] = t["device"] | derived([code], DEVICE)
    return gnu_findings(path, code, t)


class GnuOnlyWhereMsvcCompiles(unittest.TestCase):
    def test_tree(self):
        gnu = all_findings()[0]
        self.assertEqual([], ["%s:%d: %s" % f for f in gnu],
                         "GNU-only constructs in code Windows NVIDIA builds compile (see this file's docstring)")

    def test_skip_lists_current(self):
        for r in MSVC_SKIP_FILES:
            self.assertTrue((ROOT / r).is_file(), r)
        for d in MSVC_SKIP_DIRS:
            self.assertTrue((ROOT / d).is_dir(), d)

    def test_the_h2f_break_is_caught(self):
        src = ("#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n#define T_DEV 1\n#else\n"
               "#define T_DEV 0\n#endif\n"
               "__host__ __device__ inline float h2f(unsigned x) {\n#if T_DEV\n    return __uint_as_float(x);\n"
               "#else\n    float f;\n    __builtin_memcpy(&f, &x, 4);\n    return f;\n#endif\n}\n")
        self.assertEqual([(11, "__builtin_memcpy")], scan_snippet(src))
        self.assertEqual([], scan_snippet(src.replace("__builtin_memcpy", "std::memcpy")))

    def test_device_code_not_exempt_but_hints_are(self):
        src = ("__device__ float f(const float* p, unsigned u) {\n    float x;\n    __builtin_memcpy(&x, p, 4);\n"
               "    if (__builtin_expect(u > 3, 0)) x += 1.0f;\n    asm volatile(\"\" ::: \"memory\");\n"
               "    return x;\n}\n")
        self.assertEqual([(3, "__builtin_memcpy")], scan_snippet(src))
        host = src.replace("__device__", "__host__ __device__")
        self.assertEqual([(3, "__builtin_memcpy"), (4, "__builtin_expect"), (5, "inline asm in host code")],
                         scan_snippet(host))

    def test_guards(self):
        body = "int f(int* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }\n"
        flagged = [(1, "GNU __atomic builtin (std::atomic works everywhere)")]
        self.assertEqual(flagged, scan_snippet(body, "src/x.cpp"))
        for guard in ("defined(__GNUC__) || defined(__clang__)", "!defined(_MSC_VER)", "defined(__linux__)",
                      "defined(STRATA_USE_HIP)", "defined(__x86_64__) && !defined(_WIN32)", "0"):
            self.assertEqual([], scan_snippet("#if %s\n%s#endif\n" % (guard, body), "src/x.cpp"), guard)
        for guard in ("defined(__CUDA_ARCH__)", "!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700",
                      "defined(__GNUC__) || defined(_MSC_VER)", "__CUDA_ARCH__ < 610", "1"):
            self.assertEqual(flagged, [(n - 1, w) for n, w in scan_snippet("#if %s\n%s#endif\n" % (guard, body),
                                                                             "src/x.cpp")], guard)
        self.assertEqual([], scan_snippet("#ifdef _MSC_VER\nint g;\n#else\n%s#endif\n" % body, "src/x.cpp"))
        self.assertEqual([(4, flagged[0][1])],
                         scan_snippet("#ifndef _MSC_VER\nint g;\n#else\n%s#endif\n" % body, "src/x.cpp"))
        self.assertEqual([], scan_snippet("// __builtin_memcpy in a comment\nconst char* s = \"__attribute__\";\n"))


class X86Guards(unittest.TestCase):
    def test_port_files(self):
        x86 = all_findings()[1]
        new = ["%s:%d: %s outside an x86 guard" % f for f in x86 if f[0] not in X86_PENDING]
        self.assertEqual([], new)

    def test_pending_entries_still_apply(self):
        x86 = all_findings()[1]
        self.assertEqual([], sorted(set(X86_PENDING) - {f[0] for f in x86}),
                         "fixed: take these files out of X86_PENDING")

    def test_rules(self):
        self.assertEqual([(1, "<immintrin.h>")], x86_findings("#include <immintrin.h>\n"))
        self.assertEqual([], x86_findings("#if defined(__x86_64__) || defined(_M_X64)\n#include <immintrin.h>\n"
                                          "#endif\n"))
        cm = ("if(MSVC)\n  set(F /arch:AVX2)\nelse()\n  set(F -mavx2)\nendif()\n"
              "if(STRATA_CPU_X86)\n  set(G -mavx2;-mfma)\nendif()\n")
        self.assertEqual([(4, "-mavx2")], cmake_x86_findings(cm))


class PosixOnlyInPortFiles(unittest.TestCase):
    def test_port_files(self):
        posix = all_findings()[2]
        self.assertEqual([], ["%s:%d: %s (not in MSVC's CRT)" % f for f in posix])

    def test_shim_exists(self):
        self.assertTrue((ROOT / TEST_ENV_SHIM).is_file())

    def test_rules(self):
        body = "#include <cstdlib>\nvoid f() { setenv(\"A\", \"1\", 1); unsetenv(\"A\"); }\n"
        self.assertEqual([(2, "setenv()"), (2, "unsetenv()")], posix_findings("tests/glm_dsa/x.cpp", strip_code(body)))
        shim = '#include "glm_test_env.hpp"\n' + body
        self.assertIsNotNone(SHIM_INCLUDE.search(shim))
        self.assertEqual([], posix_findings("tests/glm_dsa/x.cpp", strip_code(shim), shimmed=True))
        self.assertEqual([(2, "<unistd.h>"), (3, "usleep()")], posix_findings(
            "src/core/x.cpp", strip_code("#include <cstdio>\n#include <unistd.h>\nvoid g() { usleep(1); }\n")))
        linux_only = "#if defined(__linux__)\n#include <unistd.h>\nvoid g() { usleep(1); }\n#endif\n"
        self.assertEqual([], posix_findings("src/core/x.cpp", strip_code(linux_only + "void h(S& s) { s.fork(); }\n")))


if __name__ == "__main__":
    if "--list" in sys.argv:
        g, x, p = all_findings()
        for f in g:
            print("msvc  %s:%d: %s" % f)
        for f in x:
            print("x86   %s:%d: %s%s" % (f + (" (pending)" if f[0] in X86_PENDING else "",)))
        for f in p:
            print("posix %s:%d: %s" % f)
        sys.exit(0)
    unittest.main()
