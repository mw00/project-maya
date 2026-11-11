#!/usr/bin/env bash
# tools/strata/glm53_gpu_checks.sh - the GPU checks of the glm-dsa (full GLM-5.3) port, one command each.  They need
# the GPUs (the owner frees them first: nothing may run while the 8080 dashboard serves) and, for glm-dsa, the weights.
#
#   glm53_gpu_checks.sh flash-ab  --flash-pack DIR [--ab-tokens F] [--base B] [--port P]   Flash bit identity, base vs port
#   glm53_gpu_checks.sh gpu-ctest [--base B] [--port P]                    every GPU ctest on both trees, Maya's compared
#   glm53_gpu_checks.sh parity    --model GGUF --pack DIR [--gate M2]      glm-dsa vs llama.cpp (tools/strata/parity_run.sh)
#   glm53_gpu_checks.sh kld       --pack DIR --ref FILE [--ctx N]          glm-dsa KL against a dump or a top-k reference
#   glm53_gpu_checks.sh loop      --pack DIR --tok JSON [--max-new N]      the release loop test on glm-dsa
#   glm53_gpu_checks.sh long      --pack DIR --long-tokens F[,F...] [--head N] [--tol T]
#                                 long context: each file's last 400 tokens after the whole file read (DSA selection)
#                                 vs after only 1,200 tokens of it (no selection) - tools/strata/glm53_long_gate.py
#   glm53_gpu_checks.sh all       --flash-pack DIR --model GGUF --pack DIR --ref FILE
#
# B (default ~/glm53-base/build) is a v1.1.55 build with tests; P (default this tree's build) the port's.  Results go
# to --out (default ~/logs/glm53-gpu-<date>); every check prints PASS / FAIL and the last line is the summary.
# flash-ab: the same Flash pack, the same token file (data/glm-dsa/parity's first case, cut to 2,048 tokens), on both
# builds: (1) MAYA_GLM_SCORE log-softmax dumps compared byte for byte, (2) 256 greedy tokens compared with the rest of
# the run's output (its INFO / VRAM memory layout and its DONE counts; gen_answer), each on one GPU (MAYA_GLM_SPLIT=0)
# and on the default split; any difference is a FAIL - the port must not move a Flash bit.  A run that exits non-zero,
# prints ERR or ends without its DONE line is a FAIL too, whatever the other build did.
# Both runs go with the CPU lane off (MAYA_GLM_CPU_LANE=0 MAYA_GLM_PREFILL_CPU=0): the lane's plan comes from
# timings measured at load, and its results are not bit-equal to the GPU's, so with it on even base vs base differs
# (GPU day 1: base run 1 vs 2 differed; with the lane off both builds repeat and match byte for byte).
# A machine with a card the build has no kernels for (e.g. a Quadro P400) needs CUDA_VISIBLE_DEVICES set by the caller.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=$HOME/glm53-base/build
PORT=$ROOT/build
OUT=$HOME/logs/glm53-gpu-$(date +%Y%m%d-%H%M)
FLASH= MODEL= PACK= REF= TOK= ABTOK= GATE=M2 CTX=2048 MAXNEW=6000 RUNS=1 FORCE=0 LONGTOK= LHEAD=20 LTOL=0.25
cmd=${1:-}; shift || true
while [ $# -gt 0 ]; do
    case $1 in
        --flash-pack) FLASH=$2; shift 2 ;; --model) MODEL=$2; shift 2 ;; --pack) PACK=$2; shift 2 ;;
        --ref) REF=$2; shift 2 ;; --gate) GATE=$2; shift 2 ;; --ctx) CTX=$2; shift 2 ;; --base) BASE=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --max-new) MAXNEW=$2; shift 2 ;;
        --runs) RUNS=$2; shift 2 ;; --force) FORCE=1; shift ;; --tok) TOK=$2; shift 2 ;;
        --ab-tokens) ABTOK=$2; shift 2 ;;
        --long-tokens) LONGTOK=$2; shift 2 ;; --head) LHEAD=$2; shift 2 ;; --tol) LTOL=$2; shift 2 ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done
mkdir -p "$OUT"
FAILS=0
pass() { echo "PASS $*" | tee -a "$OUT/summary.txt"; }
fail() { echo "FAIL $*" | tee -a "$OUT/summary.txt"; FAILS=$((FAILS + 1)); }
need() { [ -n "$2" ] || { echo "$1 is required for $cmd"; exit 2; }; }

# the owner's dashboard holds the GPUs: never run under it
if [ $FORCE = 0 ] && (ss -ltn 2>/dev/null | grep -q ':8080 '); then
    echo "port 8080 is serving (the dashboard holds the GPUs): ask the owner to free them, then rerun (--force skips this)"
    exit 3
fi

tokens_file() {   # the first parity case's prompt, at most CTX + 256 tokens, as one comma list
    local f=$ABTOK; [ -n "$f" ] || f=$(ls "$ROOT"/data/glm-dsa/parity/*.tokens 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "no data/glm-dsa/parity/*.tokens" >&2; return 1; }
    tr -s ' \n,' '\n' < "$f" | grep -E '^[0-9]+$' | head -n $((CTX + 256)) | paste -sd, - > "$OUT/ab.tokens"
    echo "$OUT/ab.tokens"
}

# a greedy run's output as the check compares it: every line - the INFO / VRAM lines' expert tiers and memory layout
# (weights=, kv=, act=, drafter=, pool=, slots=, ...), the T lines, the DONE line's counts, finish and drafting counters
# (DONE produced prompt prompt-ms decode-ms finish accepted offered reuse hits lookups) - with only what follows the
# clock or the memory free at load masked: the DONE line's two wall-clock fields, and the VRAM lines' free= and window=
# (both read from the device's free memory, window= as its drop across the verify window's allocations: they move when
# anything else holds GPU memory - the desktop on a Windows APU - and with the size of the build's own GPU code on HIP:
# v1.1.56 against the port on the 8065S, free= 16 MiB and window= 140 KiB apart, every other field equal).  The greedy
# runs fix the MTP draft length (MAYA_GLM_MTP_ADAPT=0: the cap, every round) so that the drafting counters repeat:
# the adaptive length (glm_mtp.cu mtp_choose; the one-GPU MTP decode, on by default on the Halo APUs, and the
# MAYA_GLM_SPEC_WIN pipe) follows measured round times, and with it the drafts and the decode's expert lookups
gen_answer() {
    sed -E -e 's/^(DONE [0-9]+ [0-9]+) [0-9.]+ [0-9.]+( |$)/\1 - -\2/' -e '/^VRAM /s/ (free|window)=[0-9]+/ \1=-/g' "$1"
}
# a greedy run that finished: exit code 0, no ERR line, and a last line DONE whose token count is its T lines' (> 0)
gen_done() {   # $1 = the run's stdout, $2 = its exit code
    [ "$2" = 0 ] && [ -f "$1" ] && ! grep -q '^ERR' "$1" || return 1
    local last n; last=$(tail -n 1 "$1"); n=$(grep -c '^T ' "$1")
    [[ $last =~ ^DONE\ ([0-9]+)\  ]] && [ "${BASH_REMATCH[1]}" = "$n" ] && [ "$n" -gt 0 ]
}

flash_ab() {
    need --flash-pack "$FLASH"
    local tok; tok=$(tokens_file) || return 1
    local -A rc=()
    for split in 0 auto; do
        for side in base port; do
            local bin=$BASE/maya; [ $side = port ] && bin=$PORT/maya
            local sp=(env MAYA_GLM_CPU_LANE=0 MAYA_GLM_PREFILL_CPU=0); [ $split = 0 ] && sp+=(MAYA_GLM_SPLIT=0)
            rm -f "$OUT/flash.$split.$side.score"   # (an earlier run's, in a reused --out)
            "${sp[@]}" env MAYA_GLM_SCORE=$CTX MAYA_GLM_SCORE_DUMP="$OUT/flash.$split.$side.score" \
                "$bin" --glm-pack "$FLASH" --tokens-file "$tok" > "$OUT/flash.$split.$side.score.log" 2>&1
            rc[score.$side]=$?
            head -c 200000 "$tok" | cut -d, -f1-$CTX > "$OUT/ab.prompt"
            "${sp[@]}" MAYA_GLM_MTP_ADAPT=0 "$bin" --glm-pack "$FLASH" --tokens-file "$OUT/ab.prompt" --greedy \
                --max-new 256 > "$OUT/flash.$split.$side.gen" 2> "$OUT/flash.$split.$side.gen.log"
            rc[gen.$side]=$?
        done
        if [ "${rc[score.base]}" != 0 ] || [ "${rc[score.port]}" != 0 ]; then
            fail "flash-ab split=$split: a score run failed (exit codes base ${rc[score.base]}, port ${rc[score.port]}; see $OUT/flash.$split.*.score.log)"
        elif [ -s "$OUT/flash.$split.base.score" ] && cmp -s "$OUT/flash.$split.base.score" "$OUT/flash.$split.port.score"; then
            pass "flash-ab split=$split: score dumps byte-identical ($(stat -c %s "$OUT/flash.$split.base.score") B)"
        else
            fail "flash-ab split=$split: score dumps differ or are missing (see $OUT/flash.$split.*.score.log)"
        fi
        local bad=
        gen_done "$OUT/flash.$split.base.gen" "${rc[gen.base]}" || bad+=" base (exit code ${rc[gen.base]})"
        gen_done "$OUT/flash.$split.port.gen" "${rc[gen.port]}" || bad+=" port (exit code ${rc[gen.port]})"
        if [ -n "$bad" ]; then
            fail "flash-ab split=$split: a greedy run did not finish (no DONE line counting its tokens, or ERR):$bad - see $OUT/flash.$split.{base,port}.gen*"
        elif diff <(gen_answer "$OUT/flash.$split.base.gen") <(gen_answer "$OUT/flash.$split.port.gen") \
                > "$OUT/flash.$split.gen.diff"; then
            pass "flash-ab split=$split: 256 greedy tokens identical, with the memory layout and the DONE counts"
        else
            fail "flash-ab split=$split: the greedy runs differ (masked: $OUT/flash.$split.gen.diff; raw: $OUT/flash.$split.{base,port}.gen)"
        fi
    done
}

gpu_ctest() {
    export LC_ALL=C
    (cd "$BASE" && ctest -L gpu -j 1 --timeout 1200 > "$OUT/ctest.base.log" 2>&1)
    (cd "$BASE" && ctest -j 1 --timeout 1200 > "$OUT/ctest.base.all.log" 2>&1)
    (cd "$PORT" && ctest -j 1 --timeout 1200 > "$OUT/ctest.port.log" 2>&1)
    st() { grep -E "Test +#[0-9]+: " "$1" | sed -E 's/.*Test +#[0-9]+: ([^ ]+) [ .*]*(.*)/\1 \2/;s/ +[0-9.]+ sec$//' | sort; }
    st "$OUT/ctest.base.all.log" > "$OUT/st.base"; st "$OUT/ctest.port.log" > "$OUT/st.port"
    local lost; lost=$(join "$OUT/st.base" "$OUT/st.port" | awk '$2=="Passed" && $3!="Passed"')
    if [ -z "$lost" ]; then pass "gpu-ctest: every Maya test that passes on v1.1.55 passes on the port"
    else fail "gpu-ctest: Maya tests lost: $(echo "$lost" | tr '\n' ';')"; fi
    local dsa_bad; dsa_bad=$(join -v2 "$OUT/st.base" "$OUT/st.port" | grep -vE " (Passed|Skipped)$")
    if [ -z "$dsa_bad" ]; then pass "gpu-ctest: every glm-dsa test passes"
    else fail "gpu-ctest: glm-dsa tests failing: $(echo "$dsa_bad" | tr '\n' ';')"; fi
}

parity() {
    need --model "$MODEL"; need --pack "$PACK"
    if (cd "$ROOT" && tools/strata/parity_run.sh --gate "$GATE" --model "$MODEL" --pack "$PACK" --out "$OUT/parity" \
            > "$OUT/parity.log" 2>&1); then pass "parity $GATE: $(head -1 "$OUT/parity/gate.md" 2>/dev/null)"
    else fail "parity $GATE (see $OUT/parity.log, $OUT/parity/report.md)"; fi
}

kld() {
    need --pack "$PACK"; need --ref "$REF"
    local tok; tok=$(tokens_file) || return 1
    local var=MAYA_GLM_SCORE_REF; head -c 4 "$REF" | od -An -tu4 | grep -qE '^ *(16|32|64|128|256) *$' && var=MAYA_GLM_SCORE_TOPK
    env MAYA_GLM_SCORE=$CTX $var="$REF" "$PORT/maya" --glm-pack "$PACK" --tokens-file "$tok" \
        > "$OUT/kld.log" 2>&1 && pass "kld: $(grep -iE 'kl|ppl|top-1' "$OUT/kld.log" | tail -2 | tr '\n' ' ')" \
        || fail "kld (see $OUT/kld.log)"
}

loop() {
    need --pack "$PACK"
    local py=${STRATA_PYTHON3:-$HOME/project-maya/.venv/bin/python3}
    (cd "$ROOT" && "$py" tools/maya_quant/loop_test.py --exe "$PORT/maya" --pack "$PACK" ${TOK:+--tok "$TOK"} --max-new "$MAXNEW" \
        --runs "$RUNS" > "$OUT/loop.log" 2>&1) && pass "loop: $(tail -1 "$OUT/loop.log")" || fail "loop (see $OUT/loop.log)"
}

long_gate() {
    need --pack "$PACK"; need --long-tokens "$LONGTOK"
    local py=${STRATA_PYTHON3:-$HOME/project-maya/.venv/bin/python3} f i=0
    for f in ${LONGTOK//,/ }; do
        i=$((i + 1))
        if "$py" "$ROOT/tools/strata/glm53_long_gate.py" --exe "$PORT/maya" --pack "$PACK" --tokens "$f" \
                --head "$LHEAD" --tol "$LTOL" --out "$OUT/long$i" > "$OUT/long$i.log" 2>&1; then
            pass "long $(basename "$f"): $(tail -1 "$OUT/long$i.log" | sed 's/^PASS long gate: //')"
        else
            fail "long $(basename "$f"): $(tail -1 "$OUT/long$i.log" | sed 's/^FAIL long gate: //')"
        fi
    done
}

case $cmd in
    flash-ab) flash_ab ;; gpu-ctest) gpu_ctest ;; parity) parity ;; kld) kld ;; loop) loop ;; long) long_gate ;;
    all) flash_ab; gpu_ctest; parity; kld; loop ;;
    *) sed -n 2,19p "$0"; exit 2 ;;
esac
echo "glm53 gpu checks: $FAILS failure(s); results in $OUT"
exit $((FAILS > 0))
