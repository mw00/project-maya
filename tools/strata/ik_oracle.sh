#!/usr/bin/env bash
# tools/strata/ik_oracle.sh - glm-dsa B14.10: ik_llama.cpp as the third oracle for glm-dsa (GLM-5.x DSA) triage.
#
# Mainline llama.cpp stays the primary oracle (B7.3 / B7.4).  When glm-dsa and mainline disagree, ik shows which side
# is wrong; it also gives free per-layer top-k and routing traces.  See docs/strata/engine/oracle.md, "ik_llama.cpp as
# a third oracle", for the procedure and research/ik-llama.md section 5 for the evidence.
#
# Subcommands (run with --help for the flags):
#   build       configure and build ik (CPU, or --cuda for sm_70) out of tree plus tools/strata/ik_dsa_logits.cpp
#   logits      teacher-forced logits (T x n_vocab f32) and per-position top-k from ik, deviations neutralized
#   trace       ik's llama-eval-callback filtered to the DSA selection / routing tensors
#   remap-kv    ik's MLA cache rows [k_pe 64 | c_kv 512] -> glm-dsa / mainline [c_kv 512 | k_pe 64] (f32 dumps)
#   check-pattern  does a GGUF's indexer schedule equal ik's hard-coded one (il <= 1 or il % 4 == 2)?
#   deviations  the known ik deviations and how each is neutralized
#   selftest    the script parses, --help works, remap-kv round-trips, the dumper compiles against ik's headers
#
# ik's source tree is never written: the build is out of tree (a generated wrapper project that add_subdirectory()s
# it), or from a copy with --copy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
envd() { local v; v="$(printenv "$1" || true)"; echo "${v:-$2}"; }   # an env switch with its default
IK_SRC_DEFAULT="$(envd "STRATA_IK_SRC" "$HOME/ik_llama.cpp")"
IK_BUILD_DEFAULT="$(envd "STRATA_IK_BUILD" "$HOME/ik-build")"
JOBS_DEFAULT="$(envd "STRATA_JOBS" 2)"
PY="${STRATA_PYTHON:-python3}"

usage() {
    cat <<'EOF'
usage: tools/strata/ik_oracle.sh <command> [options]

  build [--src DIR] [--build DIR] [--cuda] [--copy] [--jobs N]
        Configure and build ik_llama.cpp out of tree (default src $STRATA_IK_SRC or ~/ik_llama.cpp, build
        $STRATA_IK_BUILD or ~/ik-build) with the flags for GLM-5.x DSA on the test PC:
          CPU   -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON
                (Skylake-SP: AVX-512 F/CD/BW/DQ/VL, no VNNI, no VBMI: GGML_AVX512_VNNI/VBMI stay OFF)
                -DGGML_IQK_MUL_MAT=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=ON
          --cuda  adds -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=OFF
        Targets: ik_dsa_logits (tools/strata/ik_dsa_logits.cpp), llama-eval-callback, llama-perplexity, llama-cli.
        --copy rsyncs the source to <build>/src first (when the clone must not even be read by cmake's configure).

  logits --model M.gguf --out OUT.f32 (--tokens 1,2,3 | --tokens-file F) [--build DIR]
         [--dense] [--sink] [--hadamard-off] [--ngl N] [--mla N] [--ctk f32|f16|q8_0] [--threads N] [--topk K]
        Teacher-forced logits (n_tokens x n_vocab float32, row-major) and OUT.f32.topk.txt.  Defaults neutralize the
        deviations that are switches: DSA on (--dense: ik's dense MLA, the identity selection), DSA_SINK=0 (--sink:
        keep ik's +1e20 sink boost), CPU only (--ngl 0: f32 indexer scores), F32 MLA cache, f16 indexer cache,
        flash attention off, -sm layer.  Prints the glm-dsa / oracle settings that match.
        --tokens-file reads ids separated by commas, spaces or newlines.

  trace --model M.gguf --prompt TEXT [--build DIR] [--ngl N] [--out FILE] [-- extra eval-callback args]
        llama-eval-callback with --dsa, DSA_SINK=0, -fa 0, keeping the tensors dsa_sorted / dsa_indexer_score
        (per-layer top-k for B7.6) and ffn_moe_topk (routing for B8.5).

  remap-kv IN.f32 OUT.f32 [--to-ik]
        Reorders each 576-float MLA cache row: ik [k_pe 64 | c_kv 512] -> [c_kv 512 | k_pe 64] (--to-ik: back).

  check-pattern M.gguf
        Compares glm-dsa.attention.indexer.types with ik's hard-coded full-indexer layers (il <= 1 or il % 4 == 2).
        Exit 0 when they agree (GLM-5.3: 21 full layers of 78), 1 when ik's numbers are not comparable.

  deviations      prints the deviation table.
  selftest        the CPU checks ctest runs (strata_ik_oracle_help); 77 only if nothing could be checked.
EOF
}

die() { echo "ik_oracle: $*" >&2; exit 1; }

deviations() {
    cat <<'EOF'
Known ik_llama.cpp deviations for glm-dsa and how to neutralize them (research/ik-llama.md section 5):

| deviation                  | ik                                   | HF / mainline / glm-dsa        | neutralize                                   |
|----------------------------|--------------------------------------|-------------------------------|----------------------------------------------|
| attention sink boost       | +1e20 on each sequence's first       | none                          | DSA_SINK=0 (this script's default; --sink    |
|                            | present token, on by default         |                               | keeps it)                                    |
| indexer LayerNorm eps      | 1e-5 (mirrors the RMS eps;           | HF 1e-6, mainline 0           | on the glm-dsa side: STRATA_GLM_IDX_LN_EPS=1e-5|
|                            | llama-hparams.cpp:2106-2113)         |                               | (engine), ref/glm_dsa.py --idx-ln-eps 1e-5   |
| full-indexer layer pattern | hard-coded il <= 1 or il % 4 == 2;   | GGUF attention.indexer.types  | compare only where they agree: check-pattern |
|                            | GGUF indexer.types ignored           |                               | (GLM-5.3 real file: 21 full layers, agrees)  |
| MLA K cache row            | [k_pe 64 | c_kv 512]                 | [c_kv 512 | k_pe 64]          | remap-kv on cache dumps / seam bisects       |
| CUDA indexer scores        | cuBLAS COMPUTE_16F, f16 output       | f32                           | --ngl 0 (CPU), or expect near-tie set flips  |
| indexer q / k Hadamard     | on (DSA_HADAMARD_DISABLE=1 off)      | glm-dsa off by default         | score-preserving; glm-dsa's B14.11            |
|                            |                                      |                               | STRATA_GLM_IDX_HADAMARD=1 stores the same    |
|                            |                                      |                               | rotated FP16 keys; --hadamard-off for ik     |
| Q8_K block layout          | extra float sum field                | none                          | never mix ik and mainline Q8_K buffers       |
| published DSA vs dense PPL | measured before the RoPE fix 96472464| n/a                           | do not use as targets                        |

Flags never to use in oracle runs: -sm graph (ignores --dsa), -fa 1 on V100 (sm_70 FA refuses 576/512: attention on
the CPU), -rtr, -ser, -khad, a q8_0 indexer cache.
EOF
}

find_bin() {   # <build> <name>
    local b="$1" n="$2" p
    for p in "$b/$n" "$b/bin/$n" "$b/ik/bin/$n"; do [[ -x "$p" ]] && { echo "$p"; return 0; }; done
    p="$(find "$b" -maxdepth 4 -type f -name "$n" -perm -u+x 2>/dev/null | head -1)"
    [[ -n "$p" ]] && { echo "$p"; return 0; }
    return 1
}

cmd_build() {
    local src="$IK_SRC_DEFAULT" build="$IK_BUILD_DEFAULT" cuda=0 copy=0 jobs="$JOBS_DEFAULT"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --src) src="$2"; shift 2 ;;
            --build) build="$2"; shift 2 ;;
            --cuda) cuda=1; shift ;;
            --copy) copy=1; shift ;;
            --jobs) jobs="$2"; shift 2 ;;
            *) die "build: unknown option $1" ;;
        esac
    done
    [[ -f "$src/include/llama.h" ]] || die "no ik_llama.cpp source at $src (set --src or STRATA_IK_SRC)"
    mkdir -p "$build"
    if [[ $copy == 1 ]]; then
        rsync -a --delete --exclude .git "$src/" "$build/src/"
        src="$build/src"
    fi
    local wrap="$build/wrapper"
    mkdir -p "$wrap"
    cat > "$wrap/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.14)
project(strata_ik_oracle C CXX)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("$src" ik)
add_executable(ik_dsa_logits "$ROOT/tools/strata/ik_dsa_logits.cpp")
target_link_libraries(ik_dsa_logits PRIVATE llama)
EOF
    local flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON
                 -DGGML_AVX512=ON -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_BF16=OFF
                 -DGGML_IQK_MUL_MAT=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF)
    if [[ $cuda == 1 ]]; then
        flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=OFF)
    else
        flags+=(-DGGML_CUDA=OFF)
    fi
    local gen=()
    command -v ninja >/dev/null 2>&1 && gen=(-G Ninja)
    cmake -S "$wrap" -B "$build/out" "${gen[@]}" "${flags[@]}"
    cmake --build "$build/out" -j "$jobs" --target ik_dsa_logits llama-eval-callback llama-perplexity llama-cli
    echo "ik_oracle: built in $build/out (ik_dsa_logits: $(find_bin "$build/out" ik_dsa_logits || echo '?'))"
}

read_tokens() {   # --tokens / --tokens-file -> space separated ids
    local list="$1" file="$2"
    if [[ -n "$file" ]]; then
        tr ',\n\t' '   ' < "$file"
    else
        echo "${list//,/ }"
    fi
}

cmd_logits() {
    local model="" out="" tokens="" tfile="" build="$IK_BUILD_DEFAULT" dense=0 sink=0 hoff=0 ngl=0 mla=1 ctk=f32
    local threads="" topk=20
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            --tokens) tokens="$2"; shift 2 ;;
            --tokens-file) tfile="$2"; shift 2 ;;
            --build) build="$2"; shift 2 ;;
            --dense) dense=1; shift ;;
            --sink) sink=1; shift ;;
            --hadamard-off) hoff=1; shift ;;
            --ngl) ngl="$2"; shift 2 ;;
            --mla) mla="$2"; shift 2 ;;
            --ctk) ctk="$2"; shift 2 ;;
            --threads) threads="$2"; shift 2 ;;
            --topk) topk="$2"; shift 2 ;;
            *) die "logits: unknown option $1" ;;
        esac
    done
    [[ -n "$model" && -n "$out" ]] || die "logits: --model and --out are required"
    [[ -n "$tokens" || -n "$tfile" ]] || die "logits: --tokens or --tokens-file is required"
    local bin
    bin="$(find_bin "$build/out" ik_dsa_logits)" || die "logits: no ik_dsa_logits under $build/out (run build first)"
    local ids
    ids="$(read_tokens "$tokens" "$tfile")"
    local envs=(IK_DSA=$((1 - dense)) IK_NGL="$ngl" IK_MLA="$mla" IK_CTK="$ctk" IK_IDX_CTK=f16 IK_TOPK="$topk"
                IK_THREADS="$threads" DSA_SINK="$sink")
    [[ $hoff == 1 ]] && envs+=(DSA_HADAMARD_DISABLE=1)
    # shellcheck disable=SC2086
    env "${envs[@]}" "$bin" "$model" "$out" $ids
    cat <<EOF
ik_oracle: compare with glm-dsa / the oracle under matching settings:
  engine   STRATA_GLM_IDX_LN_EPS=1e-5$( [[ $dense == 1 ]] && echo ' STRATA_GLM_DSA_DENSE=1' )$( [[ $hoff == 0 ]] && echo ' (optional: STRATA_GLM_IDX_HADAMARD=1 stores the keys as ik does)' )
  oracle   python3 ref/glm_dsa.py $model --mode llama --idx-ln-eps 1e-5 --tokens <same> --logits-out ref.f32
  pattern  tools/strata/ik_oracle.sh check-pattern $model   (ik's numbers compare only when it agrees)
EOF
}

cmd_trace() {
    local model="" prompt="" build="$IK_BUILD_DEFAULT" ngl=0 out=""
    local extra=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --prompt) prompt="$2"; shift 2 ;;
            --build) build="$2"; shift 2 ;;
            --ngl) ngl="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            --) shift; extra=("$@"); break ;;
            *) die "trace: unknown option $1" ;;
        esac
    done
    [[ -n "$model" && -n "$prompt" ]] || die "trace: --model and --prompt are required"
    local bin
    bin="$(find_bin "$build/out" llama-eval-callback)" || die "trace: no llama-eval-callback under $build/out"
    local filt='dsa_sorted|dsa_indexer_score|ffn_moe_topk'
    if [[ -n "$out" ]]; then
        DSA_SINK=0 "$bin" -m "$model" -p "$prompt" --dsa -fa 0 -sm layer -ngl "$ngl" "${extra[@]}" 2>&1 \
            | grep -E -A12 "$filt" > "$out"
        echo "ik_oracle: trace in $out"
    else
        DSA_SINK=0 "$bin" -m "$model" -p "$prompt" --dsa -fa 0 -sm layer -ngl "$ngl" "${extra[@]}" 2>&1 \
            | grep -E -A12 "$filt"
    fi
}

cmd_remap() {
    [[ $# -ge 2 ]] || die "remap-kv: IN.f32 OUT.f32 [--to-ik]"
    local to_ik=0
    [[ "${3:-}" == "--to-ik" ]] && to_ik=1
    "$PY" - "$1" "$2" "$to_ik" <<'EOF'
import sys
import numpy as np
src, dst, to_ik = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
a = np.fromfile(src, dtype=np.float32)
if a.size % 576:
    sys.exit(f"remap-kv: {a.size} floats is not a multiple of the 576-float MLA row")
a = a.reshape(-1, 576)
b = np.concatenate([a[:, 512:], a[:, :512]], axis=1) if to_ik else np.concatenate([a[:, 64:], a[:, :64]], axis=1)
b.astype(np.float32).tofile(dst)
print(f"remap-kv: {a.shape[0]} rows {'to' if to_ik else 'from'} ik's [k_pe | c_kv]")
EOF
}

cmd_check_pattern() {
    [[ $# -ge 1 ]] || die "check-pattern: M.gguf"
    "$PY" - "$1" <<'EOF'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields.get("glm-dsa.attention.indexer.types")
nl = r.fields.get("glm-dsa.block_count")
n = int(nl.parts[nl.data[0]][0]) if nl else None
if f is None:
    print("check-pattern: no glm-dsa.attention.indexer.types (llama.cpp's back-compat schedule applies); ik ignores it")
    sys.exit(1)
vals = [int(f.parts[i][0]) for i in f.data]
trunk = vals[:n] if n else vals
ik = [1 if (il <= 1 or il % 4 == 2) else 0 for il in range(len(trunk))]
full = [i for i, v in enumerate(trunk) if v]
if [1 if v else 0 for v in trunk] == ik:
    print(f"check-pattern: agrees with ik's hard-coded schedule ({len(full)} full layers of {len(trunk)})")
    sys.exit(0)
diff = [i for i in range(len(trunk)) if (1 if trunk[i] else 0) != ik[i]]
print(f"check-pattern: DIFFERS from ik's schedule at layers {diff[:20]}{' ...' if len(diff) > 20 else ''}: "
      "ik's numbers are not comparable on this file")
sys.exit(1)
EOF
}

cmd_selftest() {
    local checked=0 fail=0
    bash -n "${BASH_SOURCE[0]}" && { echo "ok   the script parses"; checked=$((checked + 1)); } || { echo "FAIL parse"; fail=1; }
    local h
    h="$("${BASH_SOURCE[0]}" --help)"
    if grep -q "remap-kv" <<<"$h" && grep -q "GGML_AVX512" <<<"$h" && grep -q "DSA_SINK" <<<"$h"; then
        echo "ok   --help names the commands, the build flags and the sink switch"; checked=$((checked + 1))
    else
        echo "FAIL --help"; fail=1
    fi
    local d
    d="$(deviations)"
    if grep -q "1e-5" <<<"$d" && grep -q "il % 4 == 2" <<<"$d" && grep -q "k_pe 64 | c_kv 512" <<<"$d"; then
        echo "ok   deviations: sink boost, eps 1e-5, the hard-coded pattern, the [k_pe | c_kv] row"; checked=$((checked + 1))
    else
        echo "FAIL deviations"; fail=1
    fi
    if "$PY" -c "import numpy" 2>/dev/null; then
        local tmp
        tmp="$(mktemp -d)"
        "$PY" -c "import numpy as np; np.arange(576*3, dtype=np.float32).tofile('$tmp/a.f32')"
        cmd_remap "$tmp/a.f32" "$tmp/b.f32" >/dev/null && cmd_remap "$tmp/b.f32" "$tmp/c.f32" --to-ik >/dev/null
        if "$PY" -c "
import numpy as np, sys
a = np.fromfile('$tmp/a.f32', np.float32).reshape(-1, 576); b = np.fromfile('$tmp/b.f32', np.float32).reshape(-1, 576)
c = np.fromfile('$tmp/c.f32', np.float32)
sys.exit(0 if (b[:, :512] == a[:, 64:]).all() and (b[:, 512:] == a[:, :64]).all() and (c == a.ravel()).all() else 1)"; then
            echo "ok   remap-kv: [k_pe | c_kv] -> [c_kv | k_pe] and back"; checked=$((checked + 1))
        else
            echo "FAIL remap-kv"; fail=1
        fi
        rm -rf "$tmp"
    else
        echo "skip remap-kv (no numpy for $PY)"
    fi
    local src="$IK_SRC_DEFAULT" cxx="${CXX:-g++}"
    if [[ -f "$src/include/llama.h" && -f "$src/ggml/include/ggml.h" ]] && command -v "$cxx" >/dev/null 2>&1; then
        if "$cxx" -fsyntax-only -std=c++17 -I "$src/include" -I "$src/ggml/include" "$ROOT/tools/strata/ik_dsa_logits.cpp"; then
            echo "ok   ik_dsa_logits.cpp compiles against $src's headers"; checked=$((checked + 1))
        else
            echo "FAIL ik_dsa_logits.cpp against $src"; fail=1
        fi
    else
        echo "skip the dumper's compile check (no ik_llama.cpp at $src; set STRATA_IK_SRC)"
    fi
    [[ $fail == 0 ]] || exit 1
    [[ $checked -gt 0 ]] || exit 77
    echo "selftest: $checked checks passed"
}

case "${1:-}" in
    build) shift; cmd_build "$@" ;;
    logits) shift; cmd_logits "$@" ;;
    trace) shift; cmd_trace "$@" ;;
    remap-kv) shift; cmd_remap "$@" ;;
    check-pattern) shift; cmd_check_pattern "$@" ;;
    deviations) deviations ;;
    selftest) cmd_selftest ;;
    -h|--help|help) usage ;;
    "") usage; exit 2 ;;
    *) usage >&2; exit 2 ;;
esac
