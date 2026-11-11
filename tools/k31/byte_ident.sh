#!/bin/bash
# tools/k31/byte_ident.sh - STRATA_BUILD_K31 changes maya-k31's own targets and nothing else (plan G-ID): with the option
# ON, every other target's build commands, its outputs (objects, libraries, programs: sha256) and the tests it
# registers are those of the option OFF, byte for byte.
#
#   tools/k31/byte_ident.sh [--build DIR] [--jobs N] [--leave on|off] [-- CMAKE_ARGS...]
#
#   --build DIR   a Ninja build tree, configured (default: $K31_BUILD_DIR, else <repo>/build-k31); CMAKE_ARGS (after --)
#                 go to each configure as well
#   --jobs N      ninja -j (default 12); the builds run under nice -n 15
#   --leave X     configure the tree ON (default) or OFF at the end, built
#
# In the one tree, so that every path in every command is the same:
#   1. OFF: configure STRATA_BUILD_K31=OFF, build everything, then build again with nothing changed: the edges that
#      run anyway (maya_build_id's commit check) are the "always" set.
#      "Everything" is the default targets plus the programs of registered tests that are EXCLUDE_FROM_ALL.
#   2. Record every output file's sha256 (ninja -t targets all), every edge's command in the whole graph (ninja -t
#      commands over every output, built or not) and the registered tests (ctest --show-only=json-v1: name, command,
#      properties).
#   3. ON: configure STRATA_BUILD_K31=ON, build everything.  The edges ninja ran (.ninja_log entries that are new or
#      changed) must each be a k31 edge (an output of maya-k31, k31_*_test, k31_geo_test) or in the "always" set (k31
#      outputs already up to date from an earlier ON build are not rebuilt, which is fine); every
#      step-2 output keeps its sha256; every step-2 command is still in the graph and the new ones are all k31's; every
#      step-2 test is registered as before and the new tests are all k31's.
#   4. OFF again: configure OFF and build: nothing runs outside the "always" set, and the commands, tests and sha256s
#      are step 2's.
#   5. Leave the tree configured as --leave says, built.
# Exit 0: identical; 1: a difference (each printed); 2: usage, or a configure or build that failed.
set -u
set -o pipefail
export LC_ALL=C

usage() { sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

repo=$(cd "$(dirname "$0")/../.." && pwd)
build=${K31_BUILD_DIR:-$repo/build-k31}
jobs=12
leave=on
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --build) build=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --leave) leave=$2; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) usage 0 ;;
    *) echo "byte_ident: unknown option $1" >&2; usage ;;
  esac
done
case "$leave" in on|off) ;; *) echo "byte_ident: --leave takes on or off" >&2; exit 2 ;; esac
[ -f "$build/CMakeCache.txt" ] || { echo "byte_ident: $build is not a configured CMake build tree" >&2; exit 2; }
build=$(cd "$build" && pwd)
grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja' "$build/CMakeCache.txt" ||
  { echo "byte_ident: $build is not a Ninja build (the check reads ninja's log and graph)" >&2; exit 2; }
cmake_bin=$(sed -n 's/^CMAKE_COMMAND:INTERNAL=//p' "$build/CMakeCache.txt")
ctest_bin=$(dirname "$cmake_bin")/ctest
ninja_bin=$(sed -n 's/^CMAKE_MAKE_PROGRAM:FILEPATH=//p' "$build/CMakeCache.txt")
[ -x "$ctest_bin" ] || ctest_bin=$(command -v ctest)
py=$(command -v python3) || { echo "byte_ident: no python3" >&2; exit 2; }
work=$build/byte-ident
rm -rf "$work" && mkdir -p "$work"

say() { echo "byte_ident: $*"; }
configure() {   # configure on|off
  local v=OFF
  [ "$1" = on ] && v=ON
  "$cmake_bin" -S "$repo" -B "$build" -DSTRATA_BUILD_K31=$v "${extra[@]}" > "$work/configure-$1.log" 2>&1 ||
    { say "configure $v failed (see $work/configure-$1.log):"; tail -20 "$work/configure-$1.log"; exit 2; }
}
build_all() {   # build_all <tag>: the default targets, plus the tests' EXCLUDE_FROM_ALL programs ($test_targets)
  nice -n 15 "$ninja_bin" -C "$build" -j "$jobs" all $test_targets > "$work/build-$1.log" 2>&1 ||
    { say "build ($1) failed (see $work/build-$1.log):"; grep -m 20 -iE 'error|failed' "$work/build-$1.log"; exit 2; }
}
# the .ninja_log entries as "output<TAB>start end mtime hash" (the last entry an output has)
log_entries() {
  "$py" - "$build/.ninja_log" <<'PY'
import sys
last = {}
with open(sys.argv[1], errors="replace") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            last[p[3]] = " ".join(p[:3] + p[4:5])
for k in sorted(last):
    print(f"{k}\t{last[k]}")
PY
}
# the edges that ran between two log_entries snapshots: outputs that are new or whose entry changed
ran_between() {
  "$py" - "$1" "$2" <<'PY'
import sys
def load(p):
    d = {}
    for line in open(p):
        k, v = line.rstrip("\n").split("\t", 1)
        d[k] = v
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
for k in sorted(b):
    if a.get(k) != b[k]:
        print(k)
PY
}
k31_re='^(CMakeFiles/(maya-k31|k31_[a-z0-9_]+_test)\.dir/|(maya-k31|k31_[a-z0-9_]+_test)(\.exe)?$)'
# every build output: ninja -t targets all, without phony aliases and the configure step's products (the RERUN_CMAKE
# edge: build.ninja, CMakeCache.txt, CTestTestfile.cmake - the registered tests are compared on their own)
outputs() { "$ninja_bin" -C "$build" -t targets all | awk -F': ' '$NF != "phony" && $NF != "RERUN_CMAKE" { print $1 }' | sort -u; }
hash_outputs() {   # hash_outputs <list> <out>
  (cd "$build" && while IFS= read -r f; do [ -f "$f" ] && [ ! -L "$f" ] && echo "$f"; done < "$1" |
     grep -vx 'build.ninja' | tr '\n' '\0' | xargs -0 -r sha256sum) | sort -k2 > "$2"
}
tests_json() {   # tests_json <out>: name -> command + properties, one JSON line a test
  "$ctest_bin" --test-dir "$build" --show-only=json-v1 2>/dev/null | "$py" -c '
import json, sys
for t in json.load(sys.stdin).get("tests", []):
    props = {p["name"]: p["value"] for p in t.get("properties", []) if p["name"] not in ("_BACKTRACE_TRIPLES",)}
    print(json.dumps({"name": t["name"], "command": t.get("command"), "properties": props}, sort_keys=True))' |
    sort > "$1"
}
# every edge's command in the whole graph (EXCLUDE_FROM_ALL targets too: ninja -t commands over every output)
graph_commands() { outputs | xargs "$ninja_bin" -C "$build" -t commands | sort -u > "$1"; }
# the registered tests whose program the default build does not make (EXCLUDE_FROM_ALL): their targets
test_target_list() {
  "$ninja_bin" -C "$build" -t targets all > "$work/targets.txt"
  "$ctest_bin" --test-dir "$build" --show-only=json-v1 2>/dev/null | "$py" -c '
import json, os, sys
for t in json.load(sys.stdin).get("tests", []):
    c = t.get("command") or []
    if not c or not os.path.exists(c[0]):
        print(t["name"])' | while IFS= read -r n; do
    grep -q "^$n: " "$work/targets.txt" && echo "$n"
  done | sort -u | tr '\n' ' '
}
test_targets=""
fail=0
bad() { say "DIFF: $*"; fail=1; }

t0=$(date +%s)
# ---- 1. OFF, built, and the edges that run on a build with nothing to do
configure off
test_targets=$(test_target_list)
[ -n "$test_targets" ] && say "the tests' programs outside the default build, built too: $test_targets"
build_all off
log_entries > "$work/log-off-0.txt"
build_all off-again
log_entries > "$work/log-off-1.txt"
ran_between "$work/log-off-0.txt" "$work/log-off-1.txt" > "$work/always.txt"
say "OFF built; $(wc -l < "$work/always.txt") edge(s) run on every build: $(tr '\n' ' ' < "$work/always.txt")"

# ---- 2. record
outputs > "$work/outputs-off.txt"
hash_outputs "$work/outputs-off.txt" "$work/sha-off.txt"
graph_commands "$work/commands-off.txt"
tests_json "$work/tests-off.txt"
say "recorded $(wc -l < "$work/sha-off.txt") output files, $(wc -l < "$work/commands-off.txt") commands, $(wc -l < "$work/tests-off.txt") tests"

# ---- 3. ON
configure on
build_all on
log_entries > "$work/log-on.txt"
ran_between "$work/log-off-1.txt" "$work/log-on.txt" > "$work/ran-on.txt"
grep -vxF -f "$work/always.txt" "$work/ran-on.txt" | grep -vE "$k31_re" | grep -vx 'build.ninja' > "$work/ran-on-other.txt"
if [ -s "$work/ran-on-other.txt" ]; then
  bad "ON rebuilt $(wc -l < "$work/ran-on-other.txt") edge(s) that are not k31's:"; sed 's/^/    /' "$work/ran-on-other.txt" | head -40
fi
n_k31=$(grep -cE "$k31_re" "$work/ran-on.txt")
outputs > "$work/outputs-on.txt"
n_k31_out=$(grep -cE "$k31_re" "$work/outputs-on.txt")
[ "$n_k31_out" -gt 0 ] || bad "the ON graph has no k31 output (is the k31 block in CMakeLists.txt?)"
hash_outputs "$work/outputs-off.txt" "$work/sha-on.txt"
if ! diff -q "$work/sha-off.txt" "$work/sha-on.txt" > /dev/null; then
  bad "output files changed OFF -> ON:"; diff "$work/sha-off.txt" "$work/sha-on.txt" | head -40 | sed 's/^/    /'
fi
graph_commands "$work/commands-on.txt"
comm -23 "$work/commands-off.txt" "$work/commands-on.txt" > "$work/commands-gone.txt"
comm -13 "$work/commands-off.txt" "$work/commands-on.txt" > "$work/commands-new.txt"
[ -s "$work/commands-gone.txt" ] && { bad "$(wc -l < "$work/commands-gone.txt") OFF command(s) are not in the ON graph:"; cut -c1-300 "$work/commands-gone.txt" | head -10 | sed 's/^/    /'; }
grep -vE 'CMakeFiles/(maya-k31|k31_[a-z0-9_]+_test)\.dir/' "$work/commands-new.txt" > "$work/commands-new-other.txt"
[ -s "$work/commands-new-other.txt" ] && { bad "$(wc -l < "$work/commands-new-other.txt") new ON command(s) are not k31's:"; cut -c1-300 "$work/commands-new-other.txt" | head -10 | sed 's/^/    /'; }
tests_json "$work/tests-on.txt"
comm -23 "$work/tests-off.txt" "$work/tests-on.txt" > "$work/tests-gone.txt"
comm -13 "$work/tests-off.txt" "$work/tests-on.txt" > "$work/tests-new.txt"
[ -s "$work/tests-gone.txt" ] && { bad "OFF test(s) missing or changed with ON:"; cut -c1-300 "$work/tests-gone.txt" | sed 's/^/    /'; }
"$py" -c '
import json, re, sys
bad = [json.loads(l)["name"] for l in open(sys.argv[1]) if not re.match(r"^(k31_|maya_k31_)", json.loads(l)["name"])]
sys.stdout.write("".join(b + "\n" for b in bad))' "$work/tests-new.txt" > "$work/tests-new-other.txt"
[ -s "$work/tests-new-other.txt" ] && { bad "new ON test(s) that are not k31's: $(tr '\n' ' ' < "$work/tests-new-other.txt")"; }
say "ON: $(wc -l < "$work/ran-on.txt") edge(s) ran, $n_k31 of them k31's ($n_k31_out k31 outputs in the graph); $(wc -l < "$work/commands-new.txt") new commands, $(wc -l < "$work/tests-new.txt") new tests ($(sed -n 's/.*"name": "\([^"]*\)".*/\1/p' "$work/tests-new.txt" | tr '\n' ' '))"

# ---- 4. OFF again: a build has nothing to do but the always-run edges
configure off
log_entries > "$work/log-off-2a.txt"
build_all off-final
log_entries > "$work/log-off-2.txt"
ran_between "$work/log-off-2a.txt" "$work/log-off-2.txt" | grep -vxF -f "$work/always.txt" | grep -vx 'build.ninja' > "$work/ran-off-final.txt"
[ -s "$work/ran-off-final.txt" ] && { bad "OFF again rebuilt:"; sed 's/^/    /' "$work/ran-off-final.txt" | head -40; }
hash_outputs "$work/outputs-off.txt" "$work/sha-off-final.txt"
diff -q "$work/sha-off.txt" "$work/sha-off-final.txt" > /dev/null || { bad "output files changed ON -> OFF:"; diff "$work/sha-off.txt" "$work/sha-off-final.txt" | head -20 | sed 's/^/    /'; }
graph_commands "$work/commands-off-final.txt"
diff -q "$work/commands-off.txt" "$work/commands-off-final.txt" > /dev/null || bad "the OFF commands differ after a round trip"
tests_json "$work/tests-off-final.txt"
diff -q "$work/tests-off.txt" "$work/tests-off-final.txt" > /dev/null || bad "the OFF tests differ after a round trip"

# ---- 5. leave it as asked
if [ "$leave" = on ]; then
  configure on
  build_all leave-on
fi
if [ $fail -eq 0 ]; then
  say "IDENTICAL: $(wc -l < "$work/sha-off.txt") output files, $(wc -l < "$work/commands-off.txt") commands and $(wc -l < "$work/tests-off.txt") tests the same OFF -> ON -> OFF; ON adds only k31 edges and tests ($(( $(date +%s) - t0 )) s; left $leave; details in $work)"
  exit 0
fi
say "NOT IDENTICAL (details in $work)"
exit 1
