#!/usr/bin/env bash
# tools/strata/parity_run.sh - teacher-forced parity of the engine against llama.cpp aa94f208 (BUILD-PLAN B7.4a, B7.4b /
# M2, M2-alt, B7.6).  Procedure, gates and the results table: docs/strata/volta3/parity.md.
#
#   tools/strata/parity_run.sh --gate B7.4a --model $M --pack $P             # slow path vs llama.cpp, 4 short prompts
#   tools/strata/parity_run.sh --gate M2 --model $M --pack $P                # fast path, full corpus, criteria (a)-(i)
#   tools/strata/parity_run.sh --gate M2-alt --model $M --pack $P --llama-pin-dump ~/src/llama.cpp-pin/llm_logits_dump
#   tools/strata/parity_run.sh --gate M2 ... --stages compare,gate           # recompute from the files already there
#   tools/strata/parity_run.sh ... --dry-run                                 # print every command, run nothing
#
# Corpus (--corpus, default data/glm-dsa/parity): <case>.tokens (the prompt and the reference greedy continuation,
# token ids), <case>.ctx (the STRATA_GLM_SCORE context: the prompt length; positions ctx .. n-2 are scored) and
# cases.txt with one "<case> short|long" line per case (short: <= 2,048 + 256 tokens; long: 8,192 + 256).  B7.4a runs
# the short cases only.  The same <case>.topk the refs stage writes is what tools/strata/volta3_gate.sh reads.
#
# Stages (--stages, comma list; default all of: refs,engine,compare,gate[,bisect][,longctx]):
#   refs     llama.cpp logits of every case: R_cpu (--ngl 0) and R_cuda (--ngl 999 --ot exps=CPU, --ubatch 512); M2-alt
#            also R_cuda of the pin build (--llama-pin-dump).  <out>/ref/<case>.{cpu,cuda,pin}.bin, and <case>.topk
#            (top-64 + rest, tools/strata/llama_to_score_ref.py) from the primary reference.
#   engine   the engine's log-softmax rows (STRATA_GLM_SCORE_DUMP) under strict numerics: STRATA_GLM_IDX_LN_EPS=0 (the
#            indexer LayerNorm eps llama.cpp uses), STRATA_GLM_IDX_FP16=0 (F32 index keys).  B7.4a: the slow path
#            (STRATA_GLM_SLOW=1 STRATA_GLM_SPLIT=26,54 STRATA_GLM_NO_PREFILL=1).  M2: the fast path strict, the fast
#            path with the production eps (criterion g), and slow + fast on the first 512 positions of each short
#            case (criterion d).
#   compare  tools/strata/parity_compare.py compare for every pair: floor F (R_cpu vs R_cuda; M2-alt: R_cuda vs the
#            pin's R_cuda), R vs A, A_slow vs A_fast, strict vs production.  <out>/cmp/*.json
#   gate     <out>/manifest.json -> tools/strata/parity_compare.py gate -> <out>/gate.md, gate.json, report.md
#   bisect   (after a failed gate, or --stages bisect) the worst position of the worst case: engine GLM_CB_DIR vs
#            llama.cpp LLM_CB_DIR on the prefix that ends there -> tools/strata/seam_bisect.py bisect
#   longctx  B7.6 / M2 (f) on the long cases: the top-2048 selections appended per token on layers --overlap-layers
#            (engine GLM_CB_APPEND=1 GLM_CB_ONLY, llama LLM_CB_APPEND=1 LLM_CB_ONLY) -> seam_bisect.py overlap; and
#            the engine's F32 vs FP16 index keys (STRATA_GLM_IDX_FP16=1) the same way (docs/strata/volta3/long-context.md)
#
# Options: --model GGUF (first shard), --pack DIR, --strata build/strata, --llama-dump build/llm_logits_dump,
# --llama-pin-dump PATH (M2-alt), --corpus DIR, --out DIR (default bench/results/strata/parity/<gate>-<date>),
# --n-vocab 154880, --max-context 12288, --split 26,54, --sf-positions 512, --overlap-layers 0,26,54,74,
# --top-k 2048 (the indexer's: selection row width, overlap from position top_k + 1),
# --engine-args "...", --cases a,b (subset), --prefill 0|1 (fast path: the unscored context through the prompt path,
# default 1), --dry-run.  Exit: 0 gate PASS, 1 FAIL or INCOMPLETE, 2 usage error.
set -u
export LC_ALL=C

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python3}
GATE=M2
MODEL=""
PACK=""
STRATA=build/strata
LLAMA=build/llm_logits_dump
LLAMA_PIN=""
CORPUS=data/glm-dsa/parity
OUT=""
NV=154880
MAXCTX=12288
SPLIT=26,54
SF=512
OVL=0,26,54,74
TOPK=2048
ENGINE_ARGS=""
ONLY=""
PREFILL=1
DRY=0
STAGES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gate) GATE=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --pack) PACK=$2; shift 2 ;;
    --strata) STRATA=$2; shift 2 ;;
    --llama-dump) LLAMA=$2; shift 2 ;;
    --llama-pin-dump) LLAMA_PIN=$2; shift 2 ;;
    --corpus) CORPUS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --n-vocab) NV=$2; shift 2 ;;
    --max-context) MAXCTX=$2; shift 2 ;;
    --split) SPLIT=$2; shift 2 ;;
    --sf-positions) SF=$2; shift 2 ;;
    --overlap-layers) OVL=$2; shift 2 ;;
    --top-k) TOPK=$2; shift 2 ;;
    --engine-args) ENGINE_ARGS=$2; shift 2 ;;
    --cases) ONLY=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "parity_run: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$GATE" in B7.4a|M2|M2-alt) ;; *) echo "parity_run: --gate B7.4a | M2 | M2-alt" >&2; exit 2 ;; esac
[ -z "$STAGES" ] && STAGES="refs,engine,compare,gate,bisect$([ "$GATE" != B7.4a ] && echo ,longctx)"
[ -z "$OUT" ] && OUT="bench/results/strata/parity/${GATE}-$(date -u +%Y-%m-%d)"
if [ "$GATE" = M2-alt ] && [ -z "$LLAMA_PIN" ]; then
  echo "parity_run: M2-alt needs --llama-pin-dump (llm_logits_dump built against the pin 3cf03257, CUDA)" >&2; exit 2
fi
case ",$STAGES," in *,refs,*|*,engine,*|*,bisect,*|*,longctx,*)
  if [ -z "$MODEL" ] && [ "$DRY" = 0 ]; then echo "parity_run: --model is required for these stages" >&2; exit 2; fi
  if [ -z "$PACK" ] && [ "$DRY" = 0 ]; then echo "parity_run: --pack is required for these stages" >&2; exit 2; fi ;;
esac
cd "$ROOT" || exit 2
mkdir -p "$OUT/ref" "$OUT/eng" "$OUT/cmp" "$OUT/log" || exit 2
log() { echo "[parity_run $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
stage() { case ",$STAGES," in *,"$1",*) return 0 ;; esac; return 1; }
run() {   # run <log> <command...>: logged, or printed with --dry-run
  local lg=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ $*"; return 0; fi
  echo "+ $*" >> "$OUT/run.log"
  "$@" > "$lg" 2>&1
}
renv() {  # renv <log> VAR=val ... -- command...: the same with environment assignments
  local lg=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  if [ "$DRY" = 1 ]; then echo "+ ${envs[*]} $*"; return 0; fi
  echo "+ ${envs[*]} $*" >> "$OUT/run.log"
  env "${envs[@]}" "$@" > "$lg" 2>&1
}

# ---------------------------------------------------------------- cases
CASES=()
KIND=()
if [ ! -f "$CORPUS/cases.txt" ]; then echo "parity_run: $CORPUS/cases.txt is missing (lines: <case> short|long)" >&2; exit 2; fi
while read -r c k _; do
  case "$c" in ''|'#'*) continue ;; esac
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$c,"; then continue; fi
  if [ "$GATE" = B7.4a ] && [ "$k" != short ]; then continue; fi
  if [ ! -f "$CORPUS/$c.tokens" ] || [ ! -f "$CORPUS/$c.ctx" ]; then echo "parity_run: $c: missing $c.tokens or $c.ctx" >&2; exit 2; fi
  CASES+=("$c"); KIND+=("$k")
done < "$CORPUS/cases.txt"
[ ${#CASES[@]} -gt 0 ] || { echo "parity_run: no cases selected" >&2; exit 2; }
ntok() { tr ',[]' '   ' < "$1" | wc -w; }
prefix() {   # prefix <tokens file> <n> <out>: the first n ids
  tr ',[]' '   ' < "$1" | tr -s ' \n' '\n\n' | grep -v '^$' | head -n "$2" | tr '\n' ' ' > "$3"
}
log "gate $GATE, ${#CASES[@]} cases (${CASES[*]}), out $OUT, revision $(git rev-parse --short HEAD 2>/dev/null)"
PRIMARY=cpu
[ "$GATE" = M2-alt ] && PRIMARY=cuda

# ---------------------------------------------------------------- refs
if stage refs; then
  for c in "${CASES[@]}"; do
    tf="$CORPUS/$c.tokens"
    ctx=$(cat "$CORPUS/$c.ctx")
    n=$(ntok "$tf")
    log "refs $c: $n tokens, context $ctx"
    run "$OUT/log/ref-$c-cpu.txt" "$LLAMA" --ngl 0 --ctx $((n + 8)) "$MODEL" "$OUT/ref/$c.cpu.bin" "@$tf" || log "refs $c: R_cpu FAILED"
    run "$OUT/log/ref-$c-cuda.txt" "$LLAMA" --ngl 999 --ot exps=CPU --ubatch 512 --ctx $((n + 8)) "$MODEL" \
      "$OUT/ref/$c.cuda.bin" "@$tf" || log "refs $c: R_cuda FAILED"
    if [ "$GATE" = M2-alt ]; then
      run "$OUT/log/ref-$c-pin.txt" "$LLAMA_PIN" --ngl 999 --ot exps=CPU --ubatch 512 --ctx $((n + 8)) "$MODEL" \
        "$OUT/ref/$c.pin.bin" "@$tf" || log "refs $c: pin R_cuda FAILED"
    fi
    run "$OUT/log/ref-$c-topk.txt" $PY tools/strata/llama_to_score_ref.py --dump "$OUT/ref/$c.$PRIMARY.bin" \
      --n-vocab "$NV" --tokens "$tf" --context "$ctx" --out "$OUT/ref/$c.topk" || log "refs $c: topk FAILED"
  done
fi

# ---------------------------------------------------------------- engine
STRICT=(STRATA_GLM_IDX_LN_EPS=0 STRATA_GLM_IDX_FP16=0 STRATA_GLM_USAGE=0)
SLOW=(STRATA_GLM_SLOW=1 STRATA_GLM_SPLIT=$SPLIT STRATA_GLM_NO_PREFILL=1)
FASTP=()
[ "$PREFILL" = 1 ] && FASTP=(STRATA_GLM_SCORE_PREFILL=1)
engine() {   # engine <tag> <case> <tokens file> <context> <env...>
  local tag=$1 c=$2 tf=$3 ctx=$4; shift 4
  log "engine $tag $c (context $ctx)"
  # shellcheck disable=SC2086
  renv "$OUT/log/eng-$tag-$c.txt" "$@" STRATA_GLM_SCORE="$ctx" STRATA_GLM_SCORE_DUMP="$OUT/eng/$c.$tag.lsm" -- \
    "$STRATA" --glm-pack "$PACK" --tokens-file "$tf" --max-context "$MAXCTX" $ENGINE_ARGS \
    || log "engine $tag $c FAILED (see $OUT/log/eng-$tag-$c.txt)"
  [ "$DRY" = 1 ] || grep -m1 '^SCORE' "$OUT/log/eng-$tag-$c.txt" | sed "s/^/[$tag $c] /" | tee -a "$OUT/run.log"
}
if stage engine; then
  for i in "${!CASES[@]}"; do
    c=${CASES[$i]}
    tf="$CORPUS/$c.tokens"
    ctx=$(cat "$CORPUS/$c.ctx")
    if [ "$GATE" = B7.4a ]; then
      engine slow "$c" "$tf" "$ctx" "${STRICT[@]}" "${SLOW[@]}"
      continue
    fi
    engine fast "$c" "$tf" "$ctx" "${STRICT[@]}" "${FASTP[@]}"
    engine prod "$c" "$tf" "$ctx" STRATA_GLM_IDX_FP16=0 STRATA_GLM_USAGE=0 "${FASTP[@]}"
    if [ "${KIND[$i]}" = short ]; then   # criterion (d): positions 0 .. SF-1, both paths token by token
      prefix "$tf" $((SF + 1)) "$OUT/eng/$c.sf.tokens"
      engine sfslow "$c" "$OUT/eng/$c.sf.tokens" 0 "${STRICT[@]}" "${SLOW[@]}"
      engine sffast "$c" "$OUT/eng/$c.sf.tokens" 0 "${STRICT[@]}" STRATA_GLM_NO_PREFILL=1
    fi
  done
fi

# ---------------------------------------------------------------- compare
cmp_() {   # cmp_ <name> <ref spec> <test spec> [--positions a:b]
  local name=$1 ref=$2 test=$3; shift 3
  run "$OUT/log/cmp-$name.txt" $PY tools/strata/parity_compare.py compare --ref "$ref" --test "$test" --n-vocab "$NV" \
    --name "$name" --out "$OUT/cmp/$name.json" "$@" || { log "compare $name FAILED: $(tail -1 "$OUT/log/cmp-$name.txt" 2>/dev/null)"; return 1; }
  [ "$DRY" = 1 ] || log "compare $(cat "$OUT/log/cmp-$name.txt")"
}
FLOOR=()
AVR=()
SVF=()
PVS=()
if stage compare || stage gate; then
  for i in "${!CASES[@]}"; do
    c=${CASES[$i]}
    ctx=$(cat "$CORPUS/$c.ctx")
    n=$(ntok "$CORPUS/$c.tokens")
    if stage compare; then
      if [ "$GATE" = M2-alt ]; then cmp_ "$c.floor" "$OUT/ref/$c.cuda.bin@0" "$OUT/ref/$c.pin.bin@0" --positions "$ctx:$((n - 1))"
      else cmp_ "$c.floor" "$OUT/ref/$c.cpu.bin@0" "$OUT/ref/$c.cuda.bin@0" --positions "$ctx:$((n - 1))"; fi
    fi
    FLOOR+=("cmp/$c.floor.json")
    if [ "$GATE" = B7.4a ]; then
      stage compare && cmp_ "$c.a_vs_r" "$OUT/ref/$c.topk@$ctx" "$OUT/eng/$c.slow.lsm@$ctx"
      AVR+=("cmp/$c.a_vs_r.json")
      continue
    fi
    if stage compare; then
      cmp_ "$c.a_vs_r" "$OUT/ref/$c.topk@$ctx" "$OUT/eng/$c.fast.lsm@$ctx"
      cmp_ "$c.prod_vs_strict" "$OUT/eng/$c.fast.lsm@$ctx" "$OUT/eng/$c.prod.lsm@$ctx"
    fi
    AVR+=("cmp/$c.a_vs_r.json")
    PVS+=("cmp/$c.prod_vs_strict.json")
    if [ "${KIND[$i]}" = short ]; then
      stage compare && cmp_ "$c.slow_vs_fast" "$OUT/eng/$c.sfslow.lsm@0" "$OUT/eng/$c.sffast.lsm@0" --positions "0:$SF"
      SVF+=("cmp/$c.slow_vs_fast.json")
    fi
  done
fi

# ---------------------------------------------------------------- long context (B7.6, M2 f) before the gate reads it
EXTRA_F=""
if stage longctx && [ "$GATE" != B7.4a ]; then
  only=""
  for l in ${OVL//,/ }; do only="$only${only:+,}top_k-$l"; done
  for i in "${!CASES[@]}"; do
    [ "${KIND[$i]}" = long ] || continue
    c=${CASES[$i]}
    tf="$CORPUS/$c.tokens"
    n=$(ntok "$tf")
    for side in eng eng16 llama; do rm -rf "$OUT/sel/$c.$side"; mkdir -p "$OUT/sel/$c.$side"; done
    log "longctx $c: selections on layers $OVL, token path"
    # shellcheck disable=SC2086
    renv "$OUT/log/sel-$c-eng.txt" "${STRICT[@]}" STRATA_GLM_NO_PREFILL=1 STRATA_GLM_DSA_GRAPHS=0 GLM_CB_DIR="$OUT/sel/$c.eng" \
      GLM_CB_APPEND=1 GLM_CB_ONLY="$only" STRATA_GLM_SCORE=$((n - 1)) -- \
      "$STRATA" --glm-pack "$PACK" --tokens-file "$tf" --max-context "$MAXCTX" $ENGINE_ARGS || log "longctx $c: engine FAILED"
    # shellcheck disable=SC2086
    renv "$OUT/log/sel-$c-eng16.txt" STRATA_GLM_IDX_LN_EPS=0 STRATA_GLM_IDX_FP16=1 STRATA_GLM_USAGE=0 STRATA_GLM_NO_PREFILL=1 \
      STRATA_GLM_DSA_GRAPHS=0 GLM_CB_DIR="$OUT/sel/$c.eng16" GLM_CB_APPEND=1 GLM_CB_ONLY="$only" STRATA_GLM_SCORE=$((n - 1)) -- \
      "$STRATA" --glm-pack "$PACK" --tokens-file "$tf" --max-context "$MAXCTX" $ENGINE_ARGS || log "longctx $c: engine FP16 FAILED"
    renv "$OUT/log/sel-$c-llama.txt" LLM_CB_DIR="$OUT/sel/$c.llama" LLM_CB_APPEND=1 LLM_CB_PAD="$TOPK" LLM_CB_ONLY="$only" -- \
      "$LLAMA" --ngl 999 --ot exps=CPU --ubatch 512 --ctx $((n + 8)) "$MODEL" "$OUT/sel/$c.llama.bin" "@$tf" \
      || log "longctx $c: llama FAILED"
    run "$OUT/log/ovl-$c.txt" $PY tools/strata/seam_bisect.py overlap --a "$OUT/sel/$c.eng" --b "$OUT/sel/$c.llama" \
      --layers "$OVL" --positions "$((TOPK + 1)):$((n - 1))" --top-k "$TOPK" --json "$OUT/cmp/$c.overlap.json"
    log "longctx $c vs llama.cpp: $(tail -1 "$OUT/log/ovl-$c.txt" 2>/dev/null)"
    run "$OUT/log/ovl16-$c.txt" $PY tools/strata/seam_bisect.py overlap --a "$OUT/sel/$c.eng16" --b "$OUT/sel/$c.eng" \
      --layers "$OVL" --positions "$((TOPK + 1)):$((n - 1))" --top-k "$TOPK" --json "$OUT/cmp/$c.overlap_fp16.json"
    log "longctx $c FP16 vs F32 keys: $(tail -1 "$OUT/log/ovl16-$c.txt" 2>/dev/null)"
    EXTRA_F="$EXTRA_F $c"
  done
  if [ -n "$EXTRA_F" ] && [ "$DRY" = 0 ]; then   # (f) passes when every long case passes
    $PY - "$OUT" $EXTRA_F <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); cases = sys.argv[2:]
docs = {c: json.loads((out / "cmp" / f"{c}.overlap.json").read_text()) for c in cases
        if (out / "cmp" / f"{c}.overlap.json").exists()}
ok = len(docs) == len(cases) and all(d.get("pass") for d in docs.values())
(out / "cmp" / "overlap.json").write_text(json.dumps({"pass": ok, "cases": {c: d.get("pass") for c, d in docs.items()}}) + "\n")
EOF
  fi
fi

# ---------------------------------------------------------------- gate
STATUS=0
if stage gate; then
  $PY - "$OUT/manifest.json" "$GATE" "${FLOOR[*]:-}" "${AVR[*]:-}" "${SVF[*]:-}" "${PVS[*]:-}" "$OUT" <<'EOF'
import json, pathlib, sys
path, gate, floor, avr, svf, pvs, out = sys.argv[1:8]
out = pathlib.Path(out)
keep = lambda s: [p for p in s.split() if (out / p).exists()]
m = {"gate": gate, "floor": keep(floor), "a_vs_r": keep(avr)}
if gate != "B7.4a":
    m["slow_vs_fast"] = keep(svf)
    m["prod_vs_strict"] = keep(pvs)
    extra = {}
    for cid, f in (("e", "cmp/moe.json"), ("f", "cmp/overlap.json"), ("h", "cmp/slice.json"), ("i", "cmp/tokenizer.json")):
        if (out / f).exists():
            extra[cid] = f
    m["extra"] = extra
pathlib.Path(path).write_text(json.dumps(m, indent=1) + "\n")
EOF
  if [ "$DRY" = 1 ]; then echo "+ $PY tools/strata/parity_compare.py gate $OUT/manifest.json"; else
    $PY tools/strata/parity_compare.py gate "$OUT/manifest.json" --json "$OUT/gate.json" --md "$OUT/gate.md"
    STATUS=$?
    {
      echo "## Parity run $(date -u +%Y-%m-%d) - $GATE"
      echo
      echo "Revision $(git rev-parse --short HEAD 2>/dev/null), model \`$MODEL\`, pack \`$PACK\`, cases: ${CASES[*]}."
      echo
      echo "| Case / pair | Positions | n | mean KL | p99 KL | max KL | top-1 | near-ties | misses outside ties |"
      echo "|---|---|---|---|---|---|---|---|---|"
      for f in "$OUT"/cmp/*.json; do
        $PY -c 'import json,sys; r=json.load(open(sys.argv[1]))
if "kl" in r: print("| %s | %d..%d | %d | %.5f | %.4f | %.4f | %.2f%% | %d | %d |" % (r["name"], r["first"], r["last"], r["n"], r["kl_mean"], r["kl_p99"], r["kl_max"], r["top1_pct"], len(r["near_ties"]), len(r["top1_miss_outside_ties"])))' "$f"
      done
      echo
      cat "$OUT/gate.md"
    } > "$OUT/report.md"
    log "gate $GATE: $(head -1 "$OUT/gate.md")  (report: $OUT/report.md; paste it into docs/strata/volta3/parity.md)"
  fi
fi

# ---------------------------------------------------------------- bisect (a failed gate, or asked for)
if stage bisect && { [ "$STATUS" != 0 ] || [ "$STAGES" = bisect ]; } && [ "$DRY" = 0 ]; then
  worst=$($PY - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); best = None
for f in sorted((out / "cmp").glob("*.a_vs_r.json")):
    r = json.loads(f.read_text())
    i = max(range(len(r["kl"])), key=lambda j: r["kl"][j])
    if best is None or r["kl"][i] > best[0]:
        best = (r["kl"][i], r["name"].rsplit(".", 1)[0], r["first"] + i)
print("" if best is None else "%s %d %.5f" % (best[1], best[2], best[0]))
EOF
)
  if [ -n "$worst" ]; then
    set -- $worst
    c=$1; p=$2
    log "bisect: the worst position is $p of $c (KL $3); seams of the prefix 0..$p"
    rm -rf "$OUT/bisect" && mkdir -p "$OUT/bisect/eng" "$OUT/bisect/llama"
    prefix "$CORPUS/$c.tokens" $((p + 2)) "$OUT/bisect/tokens"
    prefix "$CORPUS/$c.tokens" $((p + 1)) "$OUT/bisect/tokens_llama"
    benv=("${STRICT[@]}" STRATA_GLM_NO_PREFILL=1 STRATA_GLM_DSA_GRAPHS=0)
    [ "$GATE" = B7.4a ] && benv+=("${SLOW[@]}")
    # the engine forwards tokens 0 .. p (scores none): its seam files hold position p
    # shellcheck disable=SC2086
    renv "$OUT/log/bisect-eng.txt" "${benv[@]}" GLM_CB_DIR="$OUT/bisect/eng" STRATA_GLM_SCORE=$((p + 1)) -- \
      "$STRATA" --glm-pack "$PACK" --tokens-file "$OUT/bisect/tokens" --max-context "$MAXCTX" $ENGINE_ARGS
    # llama.cpp: one decode call over the whole prefix, so its seam files hold positions 0 .. p
    ng=999; [ "$PRIMARY" = cpu ] && ng=0
    renv "$OUT/log/bisect-llama.txt" LLM_CB_DIR="$OUT/bisect/llama" -- "$LLAMA" --ngl $ng --ot exps=CPU \
      --ubatch $((p + 1)) --ctx $((p + 8)) "$MODEL" "$OUT/bisect/llama.bin" "@$OUT/bisect/tokens_llama"
    run "$OUT/log/bisect.txt" $PY tools/strata/seam_bisect.py bisect --a "$OUT/bisect/eng" --b "$OUT/bisect/llama" \
      --a-tokens 1 --b-tokens $((p + 1)) --json "$OUT/bisect/bisect.json"
    log "bisect: $(tail -1 "$OUT/log/bisect.txt")"
    log "next: ref/glm_dsa.py --layers A:B --hidden-in $OUT/bisect/llama/l_out-<A-1>.f32 (B7.5 slice oracle) on the diverging layer"
  fi
fi
exit "$STATUS"
