#!/bin/bash
# tools/k31/selftest_baseline.sh - the other engines' baseline for the Kimi K3.1 work: record every existing self-test,
# parity test and ctest result once, then show that a K3.1 change moved none of them (docs/KIMI-K31.md; K3.1 code
# lives in src/k31/, tests/k31_*, tools/k31/ and one CMake block, and must leave GLM, DeepSeek, MiniMax and EXL3 as
# they were).
#
#   tools/k31/selftest_baseline.sh record  [options]   run everything and write a baseline directory
#   tools/k31/selftest_baseline.sh compare [options]   run everything again and diff it against the baseline
#   tools/k31/selftest_baseline.sh diff    [options]   diff a finished compare run (--out) again, running nothing
#
# Options:
#   --build DIR       the CMake build tree (default: $K31_BUILD_DIR, else <repo>/build-k31); every section tests
#                     the sources it was configured from
#   --baseline DIR    the baseline: record writes it, compare reads it (default: $K31_BASELINE_DIR, else
#                     <build>/selftest-baseline)
#   --out DIR         where compare writes its own run and report (default: <build>/selftest-compare)
#   --only LIST       sections, comma-separated (default: all for record, the baseline's for compare):
#                       ctest        every test the build registered (ctest -j1), each one's status and full output
#                       ci-host      CI's host-only C++ steps (.github/workflows/tests.yml), -Wall -Wextra -Werror
#                       hip-compat   tools/check_hip_compat.py
#                       guards       no src/k31 file reaches ds4_kernels.cuh or ds4_model.hpp through its includes
#                       k31-py       pytest tools/test_kimi_k31*.py tools/test_k31_*.py, each test's outcome
#                       py-unittest  CI's server and tools unittest runs, in a clean environment, outcomes
#   --ctest-regex RE  limit the ctest section (ctest -R RE); compare takes the baseline's unless given, and a
#                     narrower one also narrows the baseline tests it is compared with
#   --python PY       the Python for the Python sections (default: $PYTHON, else python3)
#   --gpu-lock FILE   hold flock(1) on FILE while the sections that use the GPU (ctest, k31-py) run
#   --jobs N          parallel jobs for the build before ctest (default 8)
#   --no-build        do not build before ctest (the run is marked so, and compare says so)
#   --timeout S       limit for a test run outside ctest after it hit its own ctest TIMEOUT (default 1800)
#   --force           record over an existing baseline; with --only, replace just those sections in it
#
# Before ctest the script builds `all` and every executable a selected test runs (EXCLUDE_FROM_ALL test targets
# included, so no stale binary is tested), and deletes the GLM big fixture when glm_model_big_gen is selected, so its
# generator's self-test runs every time. A test that hits its own ctest TIMEOUT is run once more outside ctest with
# --timeout; its status then says so ("PASS(outside ctest's 300 s limit)").
#
# What compare checks, per section and test (exit 1 on any DIFF, 0 otherwise):
#   - a test's status must be the same, and its output byte-identical once the volatile spans below are masked
#     (ctest, ci-host, hip-compat, guards; for the Python sections the outcome alone is compared). This holds for a
#     test that fails in the baseline too: a failing test whose failure moved is a DIFF. Such tests are listed as
#     KNOWN in the report: only their output is gated, not what a pass would show;
#   - a passing test that is gone, a status that changed for the worse or between two failures, and a new test that
#     does not pass are DIFFs; a section with no results or a broken run (no junit.xml, a failed build: status
#     BROKEN), in the baseline or in the run, is a DIFF, and record refuses to write such a baseline;
#   - a test that now passes, a failing test that is gone, or a pass/fail flip of a FLAKY test is a NOTE.
#   Masked: the build tree and repo paths (each run's own, so two build trees compare), a number followed by a time or
#   rate unit (ms, s, GB/s, tok/s, ...), and the spans in VOLATILE below, each measured to change between two runs of
#   the same binary or with the GPU's other users. FLAKY_LINES name single checks measured to pass and fail on one
#   binary: their verdict and numbers are not compared, and a test whose only failures are those checks counts as a
#   pass; every other line of that test is compared exactly.
#
# GPU: the tests use what CUDA_VISIBLE_DEVICES shows them. Use one GPU and the same one for record and compare: some
# outputs name the device. Every test runs one at a time. K3.1's Python tests read KIMI_K31_MOONSHOT_DIR and
# KIMI_K31_FIXTURE when set (the two live tests also need torch with CUDA and fla). Variables that steer the engines
# (STRATA_*, MAYA_*, DS4_*, GLM_*, MM3_*, EXL3_*) are unset for the run, all but STRATA_GGUF_PY (where the tools find
# gguf-py). py-unittest runs under env -i with PATH, HOME, LANG, TMPDIR, USER, LOGNAME, CUDA_HOME, CUDA_PATH and
# STRATA_GGUF_PY only, and CUDA_VISIBLE_DEVICES set empty: CI has no GPU, and no unittest may reach one (it runs
# outside the GPU lock). test_maya_metal's check that the Metal server's environment has no CUDA_VISIBLE_DEVICES
# fails for that reason, in both runs. Unset, STRATA_GGUF_PY becomes the build's llama.cpp checkout's gguf-py
# (STRATA_GGML_DIR), as CI checks out gguf-py, when third_party/llama.cpp has none.
set -u

usage() { sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

repo=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:-}
case "$mode" in
  record|compare|diff) shift ;;
  -h|--help) usage 0 ;;
  *) usage ;;
esac

build=${K31_BUILD_DIR:-}
baseline=${K31_BASELINE_DIR:-}
out=""
only=""
ctest_re=""
ctest_re_set=0
py=${PYTHON:-python3}
gpu_lock=""
force=0
jobs=8
no_build=0
rerun_timeout=1800
while [ $# -gt 0 ]; do
  case "$1" in
    --build) build=$2; shift 2 ;;
    --baseline) baseline=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --only) only=$2; shift 2 ;;
    --ctest-regex) ctest_re=$2; ctest_re_set=1; shift 2 ;;
    --python) py=$2; shift 2 ;;
    --gpu-lock) gpu_lock=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --no-build) no_build=1; shift ;;
    --timeout) rerun_timeout=$2; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "selftest_baseline: unknown option $1" >&2; usage ;;
  esac
done
build=$(cd "${build:-$repo/build-k31}" 2>/dev/null && pwd) || { echo "selftest_baseline: no build tree at ${build:-$repo/build-k31}" >&2; exit 2; }
[ -f "$build/CTestTestfile.cmake" ] || { echo "selftest_baseline: $build is not a configured CMake build tree" >&2; exit 2; }
baseline=${baseline:-$build/selftest-baseline}
# --baseline and --out are read from where the script was started: the sections run in other directories (ctest
# writes its junit.xml relative to the build tree)
case "$baseline" in /*) ;; *) baseline=$PWD/$baseline ;; esac
case "$out" in ""|/*) ;; *) out=$PWD/$out ;; esac
# the sources that build tree was configured from: every section tests those (this script's checkout otherwise)
src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build/CMakeCache.txt")
[ -n "$src" ] && [ -d "$src" ] && repo=$(cd "$src" && pwd)
all_sections="ci-host hip-compat guards ctest k31-py py-unittest"

# ctest and cmake from the build's own CMake (the one that configured the tree)
cmake_bin=$(sed -n 's/^CMAKE_COMMAND:INTERNAL=//p' "$build/CMakeCache.txt")
ctest_bin=$(dirname "$cmake_bin")/ctest
[ -x "$ctest_bin" ] || ctest_bin=$(command -v ctest) || { echo "selftest_baseline: no ctest" >&2; exit 2; }
# the tools tests call cmake by name: give them the same one
[ -x "$cmake_bin" ] && export PATH="$(dirname "$cmake_bin"):$PATH"
command -v "$py" >/dev/null 2>&1 || { echo "selftest_baseline: no Python at $py" >&2; exit 2; }

if [ "$mode" = record ]; then
  dest=$baseline
  if [ -e "$dest/manifest.txt" ] && [ $force -eq 0 ]; then
    echo "selftest_baseline: $dest already holds a baseline (--force records over it)" >&2
    exit 2
  fi
  sections=${only:-$all_sections}
  if [ -n "$only" ] && [ -f "$dest/manifest.txt" ]; then
    for k in build repo; do
      v=$(sed -n "s/^$k: //p" "$dest/manifest.txt" | head -1)
      want=$build
      [ $k = repo ] && want=$repo
      [ "$v" = "$want" ] || { echo "selftest_baseline: $dest was recorded with $k $v, not $want: record a whole baseline there" >&2; exit 2; }
    done
  fi
else
  [ -f "$baseline/manifest.txt" ] || { echo "selftest_baseline: no baseline at $baseline (record one first, or --baseline DIR)" >&2; exit 2; }
  dest=${out:-$build/selftest-compare}
  if [ "$mode" = diff ]; then
    [ -f "$dest/manifest.txt" ] || { echo "selftest_baseline: no finished compare run at $dest" >&2; exit 2; }
    sections=${only:-$(sed -n 's/^sections: //p' "$dest/manifest.txt")}
  elif [ -z "$only" ]; then
    sections=$(sed -n 's/^sections: //p' "$baseline/manifest.txt")
  else
    sections=$only
  fi
  if [ $ctest_re_set -eq 0 ]; then
    ctest_re=$(sed -n 's/^ctest-regex: //p' "$baseline/manifest.txt")
  fi
fi
sections=$(echo "$sections" | tr ',' ' ')
for s in $sections; do
  case " $all_sections " in *" $s "*) ;; *) echo "selftest_baseline: unknown section $s" >&2; exit 2 ;; esac
done
[ "$mode" = diff ] || {
rm -rf "$dest.partial"
mkdir -p "$dest.partial" || exit 2
run=$dest.partial

# a clean environment for the engines: nothing set by hand steers a test
scrubbed=$(env | sed -n 's/^\(\(STRATA\|MAYA\|DS4\|GLM\|MM3\|EXL3\)_[A-Za-z0-9_]*\)=.*/\1/p' | grep -vx STRATA_GGUF_PY | sort | tr '\n' ' ')
for v in $scrubbed; do unset "$v"; done

{
  echo "mode: $mode"
  echo "sections: $sections"
  echo "ctest-regex: $ctest_re"
  echo "build: $build"
  echo "repo: $repo"
  echo "built before ctest: $([ $no_build -eq 1 ] && echo "no (--no-build)" || echo yes)"
  echo "commit: $(git -C "$repo" describe --tags --always --dirty 2>/dev/null) $(git -C "$repo" rev-parse HEAD 2>/dev/null)"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$run/manifest.txt"
{
  echo "uname: $(uname -srm)"
  echo "cxx: $(c++ --version 2>/dev/null | head -1)"
  echo "nvcc: $(nvcc --version 2>/dev/null | tail -1)"
  echo "python: $("$py" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null) ($py)"
  echo "ctest: $("$ctest_bin" --version | head -1)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>} CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER-<unset>}"
  echo "KIMI_K31_MOONSHOT_DIR=${KIMI_K31_MOONSHOT_DIR-<unset>} KIMI_K31_FIXTURE=${KIMI_K31_FIXTURE-<unset>}"
  echo "unset for the run: ${scrubbed:-none}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|STRATA_[A-Z_]+|MAYA_[A-Z_]+|DS4_[A-Z_]+):' "$build/CMakeCache.txt" | sed 's/^/cmake: /'
} > "$run/env.txt"

say() { echo "selftest_baseline: $*"; }

# ---- ctest: every registered test, serially, each one's whole output
sec_ctest() {
  local d=$run/ctest
  mkdir -p "$d/out"
  local sel=()
  [ -n "$ctest_re" ] && sel=(-R "$ctest_re")
  "$ctest_bin" --test-dir "$build" "${sel[@]}" --show-only=json-v1 > "$d/tests.json" 2> "$d/tests.err"
  if [ $no_build -eq 0 ]; then
    # the executables the selected tests run that are targets of this build (EXCLUDE_FROM_ALL ones included); ctest
    # shows no command for an executable that does not exist yet, so the paths come from the CTestTestfile.cmake files
    : > "$d/targets.known"
    if grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja' "$build/CMakeCache.txt"; then
      "$(sed -n 's/^CMAKE_MAKE_PROGRAM:FILEPATH=//p' "$build/CMakeCache.txt")" -C "$build" -t targets all 2>/dev/null |
        sed -n 's/^\([^:/ ]*\): .*/\1/p' > "$d/targets.known"
    fi
    local targets
    targets=$("$py" - "$d/tests.json" "$build" "$d/targets.known" <<'PY'
import json, os, pathlib, re, sys
tests, build = json.load(open(sys.argv[1])).get("tests", []), pathlib.Path(sys.argv[2])
known = set(open(sys.argv[3]).read().split())
add = re.compile(r'^add_test\((?:"([^"]*)"|\[(=*)\[(.*?)\]\2\])\s+"([^"]*)"', re.M)
sub = re.compile(r'^subdirs\("([^"]*)"\)', re.M)
cmd, todo = {}, [build]
while todo:
    f = todo.pop() / "CTestTestfile.cmake"
    if f.is_file():
        t = f.read_text(errors="replace")
        for m in add.finditer(t):
            cmd.setdefault(m.group(1) if m.group(1) is not None else m.group(3), m.group(4))
        todo += [f.parent / x for x in sub.findall(t)]
names = set()
for t in tests:
    c = (t.get("command") or [cmd.get(t["name"], "")])[0]
    if c and pathlib.Path(c).parent == build and (not known or pathlib.Path(c).name in known):
        names.add(pathlib.Path(c).name)
print(" ".join(sorted(names)))
PY
)
    echo "targets: all $targets" > "$d/build.log"
    if ! "$cmake_bin" --build "$build" --parallel "$jobs" --target all $targets >> "$d/build.log" 2>&1; then
      say "error: the build failed (see $d/build.log)"
      printf 'build\tBROKEN(the build failed, see build.log)\n' > "$d/results.tsv"
      return
    fi
    # now with every command ctest can resolve
    "$ctest_bin" --test-dir "$build" "${sel[@]}" --show-only=json-v1 > "$d/tests.json" 2> "$d/tests.err"
  fi
  # a test whose executable does not exist shows as NOTRUN: name it
  "$py" - "$d/tests.json" > "$d/unbuilt.txt" <<'PY'
import json, os, sys
for t in json.load(open(sys.argv[1])).get("tests", []):
    c = t.get("command") or []
    if not c or not os.path.exists(c[0]):
        print(t["name"])
PY
  if [ -s "$d/unbuilt.txt" ]; then
    say "warning: $(wc -l < "$d/unbuilt.txt") registered tests have no executable and will not run: $(tr '\n' ' ' < "$d/unbuilt.txt")"
  fi
  # the GLM big fixture's generator keeps a fixture it finds, and then never runs its self-test: start without one
  "$py" - "$d/tests.json" <<'PY'
import json, os, sys
for t in json.load(open(sys.argv[1])).get("tests", []):
    c = t.get("command") or []
    if t["name"] == "glm_model_big_gen" and "--out" in c[:-1]:
        p = c[c.index("--out") + 1]
        if os.path.isfile(p):
            os.remove(p)
PY
  local args=(--test-dir "$build" -j1 --timeout 1800 --no-tests=error "${sel[@]}"
              --output-junit "$d/junit.xml" --test-output-size-passed 16777216 --test-output-size-failed 16777216)
  "$ctest_bin" "${args[@]}" > "$d/ctest.log" 2>&1
  echo "$?" > "$d/rc"
  "$py" - "$d" "$rerun_timeout" <<'PY'
import json, os, pathlib, subprocess, sys, xml.etree.ElementTree as ET
d, limit = pathlib.Path(sys.argv[1]), int(sys.argv[2])
if not (d / "junit.xml").exists():
    rc = (d / "rc").read_text().strip()
    (d / "results.tsv").write_text(f"ctest\tBROKEN(no junit.xml, ctest rc {rc}, see ctest.log)\n")
    sys.exit(0)
tests = {t["name"]: t for t in json.load(open(d / "tests.json")).get("tests", [])}
rows, rerun = [], []
for tc in ET.parse(d / "junit.xml").getroot().iter("testcase"):
    name, st = tc.get("name"), tc.get("status", "")
    out = tc.findtext("system-out") or ""
    if st == "run":
        status = "PASS"
    else:
        msg = ""
        for child in tc:
            if child.tag in ("failure", "skipped", "error") and child.get("message"):
                msg = child.get("message")
        status = {"fail": "FAIL", "notrun": "NOTRUN", "disabled": "DISABLED"}.get(st, st.upper() or "UNKNOWN")
        status += f"({msg})" if msg else ""
    t = tests.get(name, {})
    props = {p["name"]: p["value"] for p in t.get("properties", [])}
    own = props.get("TIMEOUT")
    if status == "FAIL(Timeout)" and own is not None and float(own) < limit and t.get("command"):
        # the test's own ctest limit, not its code: run it once more without that limit
        env = dict(os.environ)
        for kv in props.get("ENVIRONMENT", []):
            k, _, v = kv.partition("=")
            env[k] = v
        try:
            p = subprocess.run(t["command"], cwd=props.get("WORKING_DIRECTORY") or None, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=limit)
            out = p.stdout.decode(errors="replace")
            if p.returncode == 0:
                status = f"PASS(outside ctest's {float(own):g} s limit)"
            elif p.returncode == props.get("SKIP_RETURN_CODE"):
                status = f"SKIP(return code {p.returncode})"
            else:
                status = f"FAIL(exit {p.returncode}, outside ctest's {float(own):g} s limit)"
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace")
            status = f"FAIL(Timeout, also after {limit} s outside ctest)"
        rerun.append(f"{name}\t{status}")
    (d / "out" / f"{name}.txt").write_text(out)
    rows.append(f"{name}\t{status}")
(d / "results.tsv").write_text("".join(r + "\n" for r in rows))
(d / "rerun.tsv").write_text("".join(r + "\n" for r in rerun))
PY
}

# ---- ci-host: the CPU-only C++ steps of .github/workflows/tests.yml, with CI's flags
sec_ci_host() {
  local d=$run/ci-host
  mkdir -p "$d/out" "$d/bin"
  : > "$d/results.tsv"
  local flags=(-std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror)
  ci_one() {   # <name> <compile args ...>
    local name=$1
    shift
    local o=$d/out/$name.txt
    if (cd "$repo" && c++ "${flags[@]}" "$@" -o "$d/bin/$name") > "$o" 2>&1; then
      if (cd "$repo" && "$d/bin/$name") >> "$o" 2>&1; then
        printf '%s\tPASS\n' "$name" >> "$d/results.tsv"
      else
        printf '%s\tFAIL(exit %s)\n' "$name" "$?" >> "$d/results.tsv"
      fi
    else
      printf '%s\tFAIL(compile)\n' "$name" >> "$d/results.tsv"
    fi
  }
  ci_one glm_memory_test tests/glm_memory_test.cpp
  ci_one ds4_cpu_prepare_rows_test tests/ds4_cpu_prepare_rows_test.cpp
  ci_one ds4_cpu_prepare_stub_test -DDS4_CPU_PREPARE_STUB_TEST=1 tests/ds4_cpu_prepare_rows_test.cpp src/ds4/ds4_iq_cpu.cpp
}

# ---- hip-compat: no CUDA name without its AMD mapping
sec_hip_compat() {
  local d=$run/hip-compat
  mkdir -p "$d/out"
  if (cd "$repo" && "$py" tools/check_hip_compat.py) > "$d/out/check_hip_compat.txt" 2>&1; then
    printf 'check_hip_compat\tPASS\n' > "$d/results.tsv"
  else
    printf 'check_hip_compat\tFAIL(exit %s)\n' "$?" > "$d/results.tsv"
  fi
}

# ---- guards: K3.1 code stays off DeepSeek's pick-stride kernels and model (KIMI-K31 plan, k <= 8 inventory): no file
# under src/k31 includes ds4_kernels.cuh or ds4_model.hpp, directly or through the repo's headers
sec_guards() {
  local d=$run/guards
  mkdir -p "$d/out"
  if (cd "$repo" && "$py" - "$d/checked.txt" <<'PY') > "$d/out/k31_no_ds4_kernels.txt" 2>&1; then
import pathlib, re, sys
root = pathlib.Path(".").resolve()
banned = {"ds4_kernels.cuh", "ds4_model.hpp"}
inc = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)
exts = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".inl", ".inc"}
starts = sorted(p for p in (root / "src" / "k31").rglob("*") if p.is_file() and p.suffix in exts) \
    if (root / "src" / "k31").is_dir() else []
bad, seen = [], set()
def walk(f, chain):
    if f in seen:
        return
    seen.add(f)
    for name in inc.findall(f.read_text(errors="replace")):
        if pathlib.PurePath(name).name in banned:
            bad.append(" -> ".join(chain + [name]))
            continue
        for base in (f.parent, root / "src", root / "include", root):
            g = (base / name).resolve()
            if g.is_file() and root in g.parents:
                walk(g, chain + [str(g.relative_to(root))])
                break
for f in starts:
    walk(f.resolve(), [str(f.relative_to(root))])
open(sys.argv[1], "w").write(f"{len(starts)} files under src/k31, {len(seen)} files read\n")
for b in bad:
    print(f"guard: {b}")
if bad:
    sys.exit(1)
print("guard: no src/k31 file includes ds4_kernels.cuh or ds4_model.hpp (directly or through the repo's headers)")
PY
    printf 'k31_no_ds4_kernels\tPASS\n' > "$d/results.tsv"
  else
    printf 'k31_no_ds4_kernels\tFAIL(exit %s)\n' "$?" > "$d/results.tsv"
  fi
}

# ---- k31-py: the K3.1 oracle, fixture, gate and tool tests, each test's outcome
sec_k31_py() {
  local d=$run/k31-py
  mkdir -p "$d"
  local files=() f
  for f in "$repo"/tools/test_kimi_k31*.py "$repo"/tools/test_k31_*.py; do
    [ -f "$f" ] && files+=("tools/${f##*/}")
  done
  if [ ${#files[@]} -eq 0 ]; then
    printf 'pytest\tBROKEN(no tools/test_kimi_k31*.py or tools/test_k31_*.py)\n' > "$d/results.tsv"
    return
  fi
  (cd "$repo" && "$py" -m pytest -q -p no:cacheprovider --junitxml="$d/junit.xml" "${files[@]}") > "$d/pytest.log" 2>&1
  echo "$?" > "$d/rc"
  [ -f "$d/junit.xml" ] && sed -i 's/ hostname="[^"]*"/ hostname=""/' "$d/junit.xml"
  "$py" - "$d" <<'PY'
import pathlib, sys, xml.etree.ElementTree as ET
d = pathlib.Path(sys.argv[1])
rows = []
if (d / "junit.xml").exists():
    for tc in ET.parse(d / "junit.xml").getroot().iter("testcase"):
        name = f"{tc.get('classname')}::{tc.get('name')}"
        status = "PASS"
        for child in tc:
            if child.tag in ("failure", "error"):
                status = "FAIL"
            elif child.tag == "skipped":
                status = f"SKIP({(child.get('message') or '').strip()})"
        rows.append(f"{name}\t{status}")
else:
    rows.append("pytest\tBROKEN(no junit.xml, see pytest.log)")
(d / "results.tsv").write_text("".join(r + "\n" for r in sorted(rows)))
PY
  tail -1 "$d/pytest.log" > "$d/summary.txt"
}

# ---- py-unittest: CI's two unittest runs, in a clean environment (as on CI's runner: no GPU), each test's outcome
sec_py_unittest() {
  local d=$run/py-unittest
  mkdir -p "$d"
  cat > "$d/runner.py" <<'PY'
import sys, unittest
start, top, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
class Result(unittest.TextTestResult):
    def addSuccess(self, t): super().addSuccess(t); rows.append((t.id(), "PASS"))
    def addFailure(self, t, e): super().addFailure(t, e); rows.append((t.id(), "FAIL"))
    def addError(self, t, e): super().addError(t, e); rows.append((t.id(), "FAIL(error)"))
    def addSkip(self, t, r): super().addSkip(t, r); rows.append((t.id(), f"SKIP({r})"))
    def addExpectedFailure(self, t, e): super().addExpectedFailure(t, e); rows.append((t.id(), "XFAIL"))
    def addUnexpectedSuccess(self, t): super().addUnexpectedSuccess(t); rows.append((t.id(), "FAIL(unexpected success)"))
    def addSubTest(self, t, s, e):
        super().addSubTest(t, s, e)
        if e is not None:
            rows.append((s.id(), "FAIL(subtest)"))
suite = unittest.TestLoader().discover(start, pattern="test_*.py", top_level_dir=top)
ok = unittest.TextTestRunner(resultclass=Result, verbosity=1).run(suite).wasSuccessful()
with open(tsv, "w") as f:
    for name, status in sorted(rows):
        f.write(f"{name}\t{status}\n")
sys.exit(0 if ok else 1)
PY
  : > "$d/results.tsv"
  local gg ggpy=${STRATA_GGUF_PY:-}
  gg=$(sed -n 's/^STRATA_GGML_DIR:PATH=//p' "$build/CMakeCache.txt")
  if [ -z "$ggpy" ] && [ ! -d "$repo/third_party/llama.cpp/gguf-py/gguf" ] && [ -d "$gg/gguf-py/gguf" ]; then
    ggpy=$gg/gguf-py
  fi
  # nothing of the session reaches the tests (a failing assertion may print the whole environment)
  local keep=(PATH="$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" CUDA_VISIBLE_DEVICES=) v
  for v in TMPDIR USER LOGNAME CUDA_HOME CUDA_PATH; do
    [ -n "${!v:-}" ] && keep+=("$v=${!v}")
  done
  [ -n "$ggpy" ] && keep+=(STRATA_GGUF_PY="$ggpy")
  printf '%s\n' "${keep[@]%%=*}" | sed 's/^/kept: /' >> "$d/env.txt"
  echo "STRATA_GGUF_PY=${ggpy:-<unset>} CUDA_VISIBLE_DEVICES=<empty>" >> "$d/env.txt"
  local part
  for part in serve tools; do
    if [ "$part" = serve ]; then
      (cd "$repo" && env -i "${keep[@]}" "$py" "$d/runner.py" serve . "$d/$part.tsv") > "$d/$part.log" 2>&1
    else
      (cd "$repo" && env -i "${keep[@]}" "$py" "$d/runner.py" tools tools "$d/$part.tsv") > "$d/$part.log" 2>&1
    fi
    echo "$?" > "$d/$part.rc"
    if [ -s "$d/$part.tsv" ]; then
      sed "s/^/$part:/" "$d/$part.tsv" >> "$d/results.tsv"
    else
      printf '%s\tBROKEN(no results, see %s.log)\n' "$part" "$part" >> "$d/results.tsv"
    fi
  done
}

locked=0
lock_gpu() {
  [ -n "$gpu_lock" ] && [ $locked -eq 0 ] || return 0
  exec 9>>"$gpu_lock" || exit 2
  if ! flock -n 9; then
    say "waiting for $gpu_lock ..."
    flock 9 || exit 2
  fi
  locked=1
}
unlock_gpu() {
  [ $locked -eq 1 ] || return 0
  flock -u 9
  exec 9>&-
  locked=0
}

t0=$(date +%s)
missing=""
for s in $sections; do
  t1=$(date +%s)
  mkdir -p "$run/$s"
  cp "$run/env.txt" "$run/$s/env.txt"
  say "$mode: $s ..."
  case "$s" in ctest|k31-py) lock_gpu ;; *) unlock_gpu ;; esac
  case "$s" in
    ctest) sec_ctest ;;
    ci-host) sec_ci_host ;;
    hip-compat) sec_hip_compat ;;
    guards) sec_guards ;;
    k31-py) sec_k31_py ;;
    py-unittest) sec_py_unittest ;;
  esac
  if [ ! -s "$run/$s/results.tsv" ] || grep -q $'\tBROKEN(' "$run/$s/results.tsv"; then
    missing="$missing $s"
    say "$mode: $s: NO RESULTS ($(( $(date +%s) - t1 )) s)"
    continue
  fi
  say "$mode: $s: $(cut -f2 "$run/$s/results.tsv" | sed 's/(.*//' | sort | uniq -c | awk '{printf "%s %s  ", $2, $1}')($(( $(date +%s) - t1 )) s)"
done
unlock_gpu
echo "seconds: $(( $(date +%s) - t0 ))" >> "$run/manifest.txt"

if [ "$mode" = record ]; then
  if [ -n "$missing" ]; then
    say "error: no results, or a broken run, in:$missing; nothing recorded (the run is in $run)"
    exit 1
  fi
  if [ -n "$only" ] && [ -f "$dest/manifest.txt" ]; then
    # replace these sections and keep the rest; the manifest lists every section the baseline now holds
    for s in $sections; do
      rm -rf "${dest:?}/$s"
      mv "$run/$s" "$dest/$s"
    done
    keep=$(sed -n 's/^sections: //p' "$dest/manifest.txt")
    now=""
    for s in $all_sections; do
      case " $keep $sections " in *" $s "*) now="$now $s" ;; esac
    done
    sed -i "s/^sections: .*/sections:$now/" "$dest/manifest.txt"
    echo "rerecorded: $sections ($(sed -n 's/^date: //p' "$run/manifest.txt"), $(sed -n 's/^commit: //p' "$run/manifest.txt"))" >> "$dest/manifest.txt"
    rm -rf "$run"
    say "sections $sections re-recorded in $dest"
    exit 0
  fi
  rm -rf "$dest"
  mv "$run" "$dest"
  say "baseline recorded in $dest"
  exit 0
fi
rm -rf "$dest"
mv "$run" "$dest"
}

# ---- compare (and diff)
"$py" - "$baseline" "$dest" $sections <<'PY'
import difflib, pathlib, re, sys
base, now, sections = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]

def manifest(d):
    out = {}
    for line in (d / "manifest.txt").read_text().splitlines():
        k, sep, v = line.partition(": ")
        if sep and k not in out:
            out[k] = v
    return out

bm, nm = manifest(base), manifest(now)

UNIT = r"(?:ns|us|µs|ms|s|sec|secs|seconds|min|GB/s|GiB/s|MB/s|MiB/s|tok/s|t/s|it/s|GFLOP/s|TFLOP/s|x)"
TIMED = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?=\s*" + UNIT + r"(?![\w/]))")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# (test, span): the numbers inside the span change between two runs of the same binary, or with what else holds the
# GPU (measured), the rest of the line does not
VOLATILE = [
    (r"glm_kv_stream_parity", r"\d+ of \d+ block reads hit VRAM"),          # which KV blocks are still in VRAM
    (r"kv_stream_parity", r"\d+ misses \([\d.]+% hit\)"),                    # the same, the streamed KV cache
    (r"pool_stress", r"\d+ batches \(\d+ after the workers slept\)"),          # how many batches fit the time
    (r"cuda_device_selftest", r"/ [\d.]+ GiB \(\d+ B\)$"),                   # the card's free VRAM (other users)
    (r"cuda_device_selftest", r"card free\s+[\d.]+ GiB \(\d+ B\)"),
]
VOLATILE = [(re.compile(t), re.compile(l)) for t, l in VOLATILE]

# (section/test): measured to pass and fail on the same binary, so a flip between the two is a note
FLAKY = [re.compile(t) for t in (
    r"py-unittest/serve:serve\.test_mcp\.Client\.test_crash_mid_call_then_restart",                     # timing
    r"py-unittest/serve:serve\.test_response_jobs\.BackgroundResponses\.test_cursor_retention_and_event_byte_limits",
)]
# (section/test): single checks measured to pass and fail on one binary. Their verdict and numbers are not
# compared; the test passes when its summary counts only them; every other line is compared exactly.
FLAKY_LINES = {
    # IQ2_XXS 400-row set gate/up: rows of 0 in 5 of 12 full runs, in none of 6 runs of --experts alone
    "ctest/mm3_gguf_test": dict(
        lines=[r"IQ2_XXS/IQ2_XS/IQ2_XS set gate/up(?: \(F16 h\))?, 400 rows\b"],
        summary=r"(?:FAILED|ALL OK) \((\d+) failed\)"),
}
FLAKY_LINES = {k: dict(lines=[re.compile(x) for x in v["lines"]], summary=re.compile(v["summary"]))
               for k, v in FLAKY_LINES.items()}
VERDICT = re.compile(r"^(.*?)\s+(ok|FAIL)(?=\s|$)")

def paths(m):
    p = [(m.get("build", ""), "<build>"), (m.get("repo", ""), "<repo>")]
    return sorted([x for x in p if x[0]], key=lambda x: -len(x[0]))

def flaky_lines(tag, text):
    """(masked lines, effective-pass?) for a test with FLAKY_LINES, else (None, None)."""
    spec = FLAKY_LINES.get(tag)
    if spec is None:
        return None, None, 0
    lines, flaky_fail, counted = [], 0, None
    for line in text.splitlines():
        m = VERDICT.match(line)
        if m and any(r.match(line) for r in spec["lines"]):
            flaky_fail += m.group(2) == "FAIL"
            lines.append(f"{m.group(1)} <known flaky: not compared>")
            continue
        s = spec["summary"].fullmatch(line.strip())
        if s:
            counted = int(s.group(1))
            lines.append("<summary>")
            continue
        lines.append(line)
    return lines, (counted is not None and counted == flaky_fail), flaky_fail

def mask(tag, name, text, side):
    fl = flaky_lines(tag, text)
    src = fl[0] if fl[0] is not None else text.splitlines()
    rules = [l for t, l in VOLATILE if t.fullmatch(name)]
    lines = []
    for line in src:
        for p, r in paths(side):
            line = line.replace(p, r)
        line = TIMED.sub("#", line)
        for l in rules:
            line = l.sub(lambda m: NUM.sub("#", m.group(0)), line)
        lines.append(line)
    return lines

def results(d, s):
    p = d / s / "results.tsv"
    if not p.exists() or not p.read_text().strip():
        return None
    out = {}
    for row in p.read_text().splitlines():
        if row:
            name, status = row.split("\t", 1)
            out[name] = status
    return out

def passed(status):
    return status == "PASS" or status.startswith("PASS(")

def ok(status):
    return passed(status) or status.startswith("SKIP") or status == "XFAIL"

diffs, notes, known, same = [], [], [], 0
for s in sections:
    for old in (now / s / "out").glob("*.diff"):
        old.unlink()
    b, n = results(base, s), results(now, s)
    if b is None:
        listed = s in bm.get("sections", "").split()
        diffs.append(f"{s}: " + ("the baseline lists this section but holds no results for it (a broken baseline: "
                                 "record it again)" if listed else "not in the baseline (record --only "
                                 f"{s} --force adds it)"))
        continue
    if any(v.startswith("BROKEN") for v in b.values()):
        diffs.append(f"{s}: the baseline's run of this section was broken ("
                     + "; ".join(f"{k} {v}" for k, v in b.items() if v.startswith("BROKEN")) + "): record it again")
        continue
    if n is None:
        diffs.append(f"{s}: section did not run or wrote no results")
        continue
    if s == "ctest":
        bre, nre = bm.get("ctest-regex", ""), nm.get("ctest-regex", "")
        if nre and nre != bre:
            kept = {k: v for k, v in b.items() if re.search(nre, k)}
            notes.append(f"ctest: this run took -R '{nre}': compared with the {len(kept)} of {len(b)} baseline tests "
                         "that match it")
            b = kept
        if not nm.get("built before ctest", "yes").startswith("yes"):
            notes.append("ctest: this run did not build first (--no-build): its binaries may be stale")
    for name in sorted(set(b) | set(n)):
        bs, ns = b.get(name), n.get(name)
        tag = f"{s}/{name}"
        bo, no = base / s / "out" / f"{name}.txt", now / s / "out" / f"{name}.txt"
        btext = bo.read_text(errors="replace") if bo.exists() else None
        ntext = no.read_text(errors="replace") if no.exists() else None
        # a test whose only failures are its known-flaky checks counts as a pass
        for side in ("b", "n"):
            text, st = (btext, bs) if side == "b" else (ntext, ns)
            if st is not None and text is not None and tag in FLAKY_LINES and st in ("PASS", "FAIL(Failed)"):
                _, eff, nfail = flaky_lines(tag, text)
                if eff:
                    if side == "n" and nfail:
                        notes.append(f"{tag}: {nfail} known-flaky check(s) failed in this run, not compared")
                    st = "PASS"
                elif st == "PASS":
                    st = "FAIL(its summary line is missing or counts other failures)"
                if side == "b":
                    bs = st
                else:
                    ns = st
        if ns is None:
            # a failing placeholder that went away (unittest's _FailedTest for a module that now imports) is a note
            (diffs if ok(bs) else notes).append(f"{tag}: gone (baseline {bs})")
            continue
        if bs is None:
            (notes if ok(ns) else diffs).append(f"{tag}: new test, {ns}")
            continue
        if not ok(bs):
            known.append(f"{tag}: baseline {bs}")
        flaky = any(t.fullmatch(tag) for t in FLAKY)
        if bs != ns:
            if passed(bs) and passed(ns):
                notes.append(f"{tag}: {bs} -> {ns}")        # the same pass, one of them outside ctest's limit
            elif ok(ns) and not passed(bs):
                notes.append(f"{tag}: {bs} -> {ns}")        # a fix, or a skip's reason
                continue
            elif flaky and (passed(bs) or passed(ns)):
                notes.append(f"{tag}: {bs} -> {ns} (known flaky)")
                continue
            else:
                diffs.append(f"{tag}: {bs} -> {ns}")
                continue
        if btext is not None and ntext is not None:
            bl, nl = mask(tag, name, btext, bm), mask(tag, name, ntext, nm)
            if bl != nl:
                d = list(difflib.unified_diff(bl, nl, f"baseline/{tag}", f"now/{tag}", lineterm="", n=1))
                (now / s / "out" / f"{name}.diff").write_text("\n".join(d) + "\n")
                moved = sum(1 for x in d[2:] if x[:1] in "+-")
                msg = f"{tag}: output moved ({moved} lines, {s}/out/{name}.diff)"
                (notes if flaky else diffs).append(msg + (" (known flaky)" if flaky else ""))
                continue
        elif (btext is None) != (ntext is None):
            diffs.append(f"{tag}: output " + ("gone" if ntext is None else "new"))
            continue
        same += 1

report = [f"compare: {now} against {base}"]
for line in (base / "env.txt").read_text().splitlines():
    if line.startswith(("gpu:", "CUDA_VISIBLE_DEVICES")):
        other = [x for x in (now / "env.txt").read_text().splitlines() if x.split(":")[0] == line.split(":")[0]]
        if line not in other:
            notes.append(f"env: baseline '{line}' is not in this run's env.txt (outputs may differ by device)")
report += [f"DIFF {x}" for x in diffs] + [f"NOTE {x}" for x in notes]
report += [f"KNOWN {x} (fails in the baseline: only its output and status are gated)" for x in known]
report.append(f"compare: {same} identical, {len(notes)} notes, {len(diffs)} differences, {len(known)} failing in the "
              "baseline -> " + ("FAIL" if diffs else "PASS"))
(now / "compare.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
sys.exit(1 if diffs else 0)
PY
