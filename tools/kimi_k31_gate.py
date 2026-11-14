"""tools/kimi_k31_gate.py - the Kimi K3.1 Phase 1 gate: ref/kimi_k31.py against an independent model.

The independent model is Moonshot's own K3 modeling code (HF moonshotai/Kimi-K3 @ f831ab6, imported
unchanged) with the real fla 0.6.0 KDA kernels on CUDA and a lightning-indexer patch that follows SGLang
kimi-k3.1-support @ accf045c and vLLM PR #65685 @ 9d6ac73f (tools/kimi_k31_moonshot_oracle.py). It shares
no code with the reference. Its K3 part is Moonshot's code; its indexer half is not: Moonshot ships no K3.1
code, so the patch is a transcription of the engines' formulas, written from the same notes as the
reference, and a misreading shared by both would pass this gate. tools/test_kimi_k31_oracle.py
(KimiK31IndexerEngines) closes that gap: it runs the golden's recorded indexer inputs through vLLM's own
reference score function and SGLang's torch fallback and checks they give the golden's scores.
Its fp32 outputs on the synthetic fixture (tools/kimi_k31_synth_gguf.py, seed 3101) are committed as a
small subset under data/kimi-k31-synth-golden/, so the gate itself needs numpy only:

    golden.npz    per sequence (seq0: 32 tokens, seq1: 5, seq2: 80, crossing fla's 64-token chunk) and
                  mode (prefill; decode = prefill of decode_from tokens, then one token per step through
                  Moonshot's cache), tokens first like the checkpoint:
                    G/<seq>.<mode>/logits             [T, vocab] float32, the full vocabulary
                    G/<seq>.<mode>/layers.0.attn_out  [T, n_embd] the first KDA layer's self_attn output
                    G/<seq>.<mode>/layers.3.attn_out  [T, n_embd] the first MLA layer's self_attn output
                    G/<seq>.<mode>/moe.L.expert_ids   [T, k] int16 ascending, L = 1..8
                    G/<seq>.<mode>/moe.L.cut_margin   [T] k-th minus (k+1)-th biased router score
                    G/<seq>.<mode>/dsa.L.selected     [T, top_k] int16 ascending key positions, -1 pads
                    G/<seq>.<mode>/dsa.L.cut_margin   [T] k-th minus (k+1)-th visible score (inf: dense row)
                    G/<seq>.prefill/dsa.L.scores      [T, T] float32 indexer scores, -inf on future keys
                    G/seq0.prefill/layers.L.attn_in, dsa.L.qr   each MLA layer's indexer inputs (the
                                                      attention input and the q_a_layernorm output)
                    E/<seq>.<mode>/layers.3.attn_out  the same seam from a run with q_a/kv_a eps 1e-5
                  G = "mla_eps_1e-6", the golden: Moonshot's q_a_layernorm / kv_a_layernorm eps (the
                  KimiRMSNorm class default 1e-6). E = "mla_eps_1e-5": the serving engines' rms_norm_eps
                  1e-5, everything else identical; it carries only the seam that resolves the eps.
    golden.json   provenance (sha256 of the oracle, the indexer patch and Moonshot's files; fla / triton /
                  torch versions, GPU, kernel precision), the sha256 of every fixture file the golden was
                  made from, the sequences, and the oracle's noise measurements.

Ties. Neither engine specifies how a top-k breaks ties, at the indexer cut or the router cut (their
selections are sets), and the reference and the oracle each pick a rule of their own (the earlier key
or expert). A golden whose logits depended on that rule would fail a correct implementation, so the gate
is built tie-free: the fixture's test sequences are grown token by token so that every top-k cut has room
(tools/kimi_k31_synth_gguf.py TEST_TOKENS, picked by `pick` below), and gate() itself fails if any row of
the golden has an indexer cut margin under TIE_FREE (10) x IDX_SCORE_ATOL or a router cut margin under
10 x MOE_TIE. Here the smallest are 4.18e-3 (14x) and 6.26e-4 (12.5x). Breaking every tie toward the
later key instead passes with every selection identical. Exact ties are exercised separately, on indexer
inputs built to tie (tools/test_kimi_k31_oracle.py KimiK31Ties), where selections are compared as sets.

The checks, per sequence and mode (Report.failures collects every one that does not hold):
  * logits, full vocabulary: max |ref - golden| <= logits_atol; pooled over all six runs,
    rms <= logits_rms;
  * the layer-3 MLA seam, the first place the q_a/kv_a eps acts, before later layers mix fla's noise in:
    max <= seam_atol, pooled rms <= seam_rms;
  * the layer-0 KDA seam, the first place the KDA q/k l2 eps acts: max <= kda_seam_atol, pooled rms <=
    kda_seam_rms;
  * the golden is tie-free on the row (above);
  * the indexer scores of every MLA layer, visible keys: max <= IDX_SCORE_ATOL, and -inf exactly on
    the golden's future keys;
  * selected experts EXACTLY equal as sets on every (layer, token) row. A row whose GOLDEN router cut
    margin is below MOE_TIE would be a near-tie, reported (Report.moe_near_ties) instead of judged - by
    the golden's margin alone, so a reference cannot excuse a disagreement by squeezing its own scores
    (none here: the golden is tie-free); a set-size difference is always a failure;
  * indexer selections equal as sets on every row, up to ties: same-size sets that differ only in keys
    scoring within tie_tol of the cut (golden scores; tie_tol = 2 x the layer's observed score
    difference, capped at 2 x IDX_SCORE_ATOL) agree; a selected future key never does.

Tolerances. The independent model's own noise is what every reference is compared through: real fla is
not bit-reproducible (two identical runs differ by 4.7e-5 max / 3.3e-6 rms on the logits, whose peak is
|4.0|; by 2.4e-7 / 2.6e-8 on the layer-0 output), so the float64 reference - exact to ~1e-13 on this
fixture - measures fla's error. Observed (tools/test_kimi_k31_oracle.py re-measures and prints these;
"repeat" = a second fresh oracle run):

                         f64 ref vs golden / repeat     f32 ref vs golden / repeat     tolerance f64 / f32
    logits max           8.0e-5 / 7.1e-5               1.9e-4 / 1.8e-4                3e-4 / 5e-4
    logits rms (pooled)  5.6e-6 / 5.5e-6               9.5e-6 / 9.2e-6                1e-5 / 2e-5
    layer-3 seam max     2.1e-6 / 1.9e-6               2.6e-6 / 2.5e-6                5e-6
    layer-3 seam rms     2.9e-7 / 2.9e-7               4.4e-7 / 4.4e-7                7e-7
    layer-0 seam max     2.3e-6 / 2.3e-6               2.2e-6 / 2.2e-6                3.7e-6
    layer-0 seam rms     2.5e-7 / 2.5e-7               3.3e-7 / 3.3e-7                4.0e-7 / 4.7e-7
    indexer scores max   6.6e-5 / 7.4e-5               7.2e-5 / 6.8e-5                3e-4
    router scores max    9.2e-6                        2.2e-5                         (near-tie bound 5e-5)

  * logits_atol is 3.7x (f64) / 2.6x (f32) the largest observed difference. It cannot hide a structural
    bug: every structural mutation in tools/test_kimi_k31_oracle.py moves the logits by more than 1, four
    orders of magnitude above it, and the discrete checks (experts, indexer sets) do not depend on any
    float tolerance.
  * logits_rms is a backstop for a small systematic error spread over the vocabulary: about 2x fla's
    own error (f64) and 2x the combined fla + float32 error (f32).
  * The q_a/kv_a eps (1e-6 vs 1e-5) is a precision-level choice. On the logits the wrong eps gives
    2.1e-4 / 2.3e-4 max (f64 / f32, under logits_atol) and 1.2e-5 / 1.3e-5 rms (the f64 bound trips).
    The layer-3 seam resolves it: wrong eps 1.9e-6 rms against 2.9e-7 / 4.4e-7 matched, so seam_rms
    sits 2.4x / 1.6x above the matched and 2.7x below the wrong one, in both directions (runs G and E).
  * The KDA q/k l2 eps (fla's kernels 1e-6, llama.cpp's graph the rms eps 1e-5) is resolved by the
    layer-0 seam, narrowly: wrong eps 6.5e-7 / 6.8e-7 rms and 6.1e-6 / 6.0e-6 max against 2.5e-7 /
    3.3e-7 rms and 2.3e-6 / 2.2e-6 max matched; the bounds sit about 1.6x (f64) / 1.4x (f32) from each
    side, with fla's run-to-run spread there 10x smaller than either gap. Only that direction can be
    tested (fla's eps is fixed). The wrong eps also trips the layer-3 seam rms (8.5e-7) and, in f64,
    the logits rms (1.5e-5); its logits max (1.5e-4 / 2.3e-4) stays under logits_atol.
  * MOE_TIE 5e-5 is 2.3x the float32 router-score difference; IDX_SCORE_ATOL 3e-4 is 4x the indexer-score
    difference (scores reach |4|). Both only matter at a near-tie, and the golden has none.

An engine's seam dumps (check-dumps). An engine run on the golden's sequences (each prefill, and each decode:
decode_from tokens, then one token per call) dumps its seams under ref/kimi_k31.py's cb() names (`name-<layer>`,
`result_output` for the logits), one run per entry of the dump directory:
    <seq>.<mode>/manifest.txt + <name>.f32   ref/glm.py's raw layout: manifest lines `name d0 d1 ...` in ggml
                                             order, raw little-endian data, ffn_moe_topk / top_k as int32
    <seq>.<mode>.npz                         the same arrays, numpy, tokens first (the ggml dims reversed)
Each seam covers the whole sequence (the calls' tokens concatenated): T token rows. indexer_score and
kq_mask_top_k are [T, T] with -inf on future keys; top_k is [T, top_k] key positions, -1 pads; selections are
compared as sets. `dump-ref` writes the reference's own dumps in this layout. The seam ladder (plan M1a), each
rung a gate, judged in order on the fixture's geometry (here layers 0 KDA + dense, 1.. MoE, 3 MLA, pushes 0/3/6):
    1  kda_out-0                     vs the golden's layers.0.attn_out   max / pooled rms: kda_seam_atol / _rms
    2  l_out-0 (dense SiTU FFN)      vs the f64 reference                REF64_TOL
    3  ffn_moe_out-1                 vs the f64 reference                REF64_TOL
       ffn_moe_topk-1                vs the golden's moe.1.expert_ids    exact sets, every row
    4  mla_out-3                     vs the golden's layers.3.attn_out   seam_atol / seam_rms
       top_k-3, indexer_score-3      vs the golden's dsa.3.*             sets up to ties, scores IDX_SCORE_ATOL
    5  res_probs_ffn-3, res_mix_ffn-3  (the first mix after the push at layer 3) vs the f64 reference, REF64_TOL
    6  result_output                 vs the golden's logits              logits_atol / logits_rms
       ffn_moe_topk-L, top_k-L, indexer_score-L for every MoE / MLA layer: all 1,872 expert and 702 indexer rows
The golden is used only when the engine's eps are its own (Moonshot's: run G; q_a/kv_a 1e-5: run E, which
carries only the layer-3 seam); everything the golden does not carry, and every seam with --reference ref64,
is judged against ref/kimi_k31.py in float64 with the engine's eps, with the same bounds. The golden's rows
must be tie-free here too. Other dumped seams are validated, not judged. Refused (exit 2) before anything is
compared: a run the golden does not have or a missing one, a name ref/kimi_k31.py does not dump, a seam on a
layer that has no such seam (kda_out-3: layer 3 is MLA) or a non-canonical layer (l_out-03), a shape other
than the reference's (for ffn_moe_topk / ffn_moe_weights* / top_k / res_probs* only the last axis, a count the
model decides, may differ: that is judged), a seam a judged rung needs that is missing, a manifest that does
not match its files, a file that cannot be read, a --gguf that is not a GGUF the reference reads (its header,
its tensor types, tensor data cut short) or is of another geometry (from its header and size, before any tensor
is read or the file hashed), a GGUF that is not the golden's fixture (the bounds are measured on it), and
--golden on a geometry without a golden. dump-ref refuses the same unreadable or other-geometry --gguf and the
same --golden, but not another GGUF of the fixture's geometry (it writes that GGUF's reference dumps); an error
once the checks have passed is a real one and is raised, not refused.
On another fixture geometry (--geometry big) there is no golden: the runs are that fixture's own gate sequences
(kimi_k31_synth_gguf.py gate_sequences, tie-free), every seam is judged against the float64 reference (plan M1a
gate 2) with that geometry's tie bands (TIE_BANDS) and bounds (DUMP_TOL / DUMP_REF64_TOL: 2x the float32
reference's measured gap), on the same ladder (big: KDA + dense layer 0, MLA 3/8/9, pushes 0/3/6/9), and the
GGUF must be the one its sequences were picked on (BIG_PICK).
tools/test_kimi_k31_check_dumps.py is the self-test: the reference's own float32 dumps pass, a 1e-3 change of
one value fails the rung that judges it, malformed dumps are refused, and every M1a mutation switch (the plan's
eleven, the converter's A_log slice included) fails first on its rung.

CLI:
  python tools/kimi_k31_gate.py pick [--geometry tiny|big] [--gguf tiny.gguf] [--seed N] [--verify]
      grow tie-free test sequences for the fixture (prints JSON; numpy only) at pick_bounds(geometry),
      under each eps preset in PICK_EPS; --verify re-checks the sequences the synth tool records, each
      as one whole prefill under every eps preset. A --gguf of another geometry is refused. On a busy
      machine run it with OPENBLAS_NUM_THREADS=1: the big fixture's one-token steps are small
      matrix-vector products, and a full BLAS thread pool competing for the cores made them ~20x slower.
  python tools/kimi_k31_gate.py gap [--geometry tiny|big] [--gguf big.gguf] [--seed N]
      the float32 reference against the float64 one on the fixture's sequences, per eps preset (prints
      JSON; numpy only): what TIE_BANDS is sized from, and the big manifest's reference_gap.
  python tools/kimi_k31_gate.py export --run-1e-6 RUN --run-1e-5 RUN [--golden-json GOLDEN.json] --out DIR
      cut the committed subset out of two oracle run directories (needs safetensors)
  python tools/kimi_k31_gate.py check [--gguf tiny.gguf] [--golden DIR] [--dtype float64] [--eps-mla 1e-6]
      run the reference through the gate and print the report (numpy only)
  python tools/kimi_k31_gate.py check-dumps --dumps DIR [--dtype float32] [--geometry tiny|big] [--gguf tiny.gguf]
                                [--golden DIR (tiny only)] [--eps moonshot|llama.cpp] [--eps-mla X] [--eps-kda X]
                                [--reference golden|ref64] [--rungs N]
      judge an engine's seam dumps on the seam ladder, rungs 1..N (exit 0 pass, 1 fail, 2 refused: a
      `check-dumps: refused:` line; argparse's usage errors exit 2 too). `check-engine` is the same command.
      The last line is `check-dumps PASS` only when all six rungs were judged and passed.
  python tools/kimi_k31_gate.py dump-ref --out DIR [--dtype float32] [--format raw|npz] [--geometry tiny|big] [--eps ...]
      write the reference's own dumps of every run check-dumps judges on that geometry, in its layout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass, field

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

GOLDEN_DIR = HERE.parent / "data" / "kimi-k31-synth-golden"
SEAM = "layers.3.attn_out"            # golden key; the reference's cb() name is mla_out-3
SEAM_LAYER = 3
KDA_SEAM = "layers.0.attn_out"        # golden key; the reference's cb() name is kda_out-0
KDA_SEAM_LAYER = 0
INDEXER_INPUT_SEQ = "seq0"            # the prefill whose indexer inputs (attn_in, qr) the golden carries
MODES = ("prefill", "decode")
GOLD = "mla_eps_1e-6"                 # Moonshot's q_a/kv_a eps: the golden
ENGINE_EPS = "mla_eps_1e-5"           # the serving engines' eps: the layer-3 seam only

# per reference dtype (see the module docstring for the measurements behind each number)
TOL = {
    "float64": {"logits_atol": 3e-4, "logits_rms": 1e-5, "seam_atol": 5e-6, "seam_rms": 7e-7,
                "kda_seam_atol": 3.7e-6, "kda_seam_rms": 4.0e-7},
    "float32": {"logits_atol": 5e-4, "logits_rms": 2e-5, "seam_atol": 5e-6, "seam_rms": 7e-7,
                "kda_seam_atol": 3.7e-6, "kda_seam_rms": 4.7e-7},
}
IDX_SCORE_ATOL = 3e-4
MOE_TIE = 5e-5
# The golden is built tie-free: on every row judged on logits, the golden's indexer cut margin is at least
# TIE_FREE x IDX_SCORE_ATOL and its router cut margin at least TIE_FREE x MOE_TIE (gate() checks this).
# The test sequences are picked with a little more room, for fla's noise on the golden's own margins.
TIE_FREE = 10.0
PICK_ROOM = 12.0
# The near-tie band of the router and the indexer cut margins, per fixture geometry (kimi_k31_synth_gguf.py
# GEOMETRIES): a golden must clear TIE_FREE x the band on every row, and the test sequences are picked
# at PICK_ROOM x it. "Shift" below is the largest change of any row's cut margin that the float32
# reference, or the other eps preset, makes against the float64 reference (`gap` measures it).
#   tiny: MOE_TIE / IDX_SCORE_ATOL, the gate's own numbers (its golden is built on them): 1.9x / 3.8x
#         tiny's shift (router 2.6e-5, indexer 7.9e-5).
#   big:  sized from big's own shift, measured on its sequences under both eps presets (router 9.1e-5,
#         indexer 3.4e-4; `gap --geometry big`, the big manifest's reference_gap): the bands are 1.4x / 1.2x
#         it, so the golden check sits at 14x / 12x the shift and the pick leaves 2.8x / 2.3x of it as room
#         above the check. (Tiny's 10x / 12x of 1.9x / 3.8x bands cannot be picked at big's density: with
#         those bands about 1 random candidate in 150 clears every cut where the indexer selects.) Big's
#         fla noise is not measured yet: a big golden whose margin shift against the float64 reference
#         exceeds the band needs a larger band and a new pick.
TIE_BANDS = {"tiny": {"router": MOE_TIE, "indexer": IDX_SCORE_ATOL},
             "big": {"router": 1.25e-4, "indexer": 4e-4}}
# The eps presets (ref/kimi_k31.py EPS_PRESETS) a picked sequence must clear the pick bounds under, each
# as a float64 run. tiny's TEST_TOKENS were picked under moonshot only (and its golden is built on them);
# they clear the bounds under llama.cpp's too, and `pick --verify` checks every preset for both.
PICK_EPS = {"tiny": ("moonshot",), "big": ("moonshot", "llama.cpp")}


def pick_bounds(geometry: str) -> dict[str, float]:
    """The smallest router / indexer cut margin a picked sequence may have, for a fixture geometry."""
    return {k: PICK_ROOM * v for k, v in TIE_BANDS[geometry].items()}


def tie_free_bounds(geometry: str) -> dict[str, float]:
    """The smallest router / indexer cut margin a golden made on the geometry's sequences may have."""
    return {k: TIE_FREE * v for k, v in TIE_BANDS[geometry].items()}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================ the golden ================================

def load_golden(d=GOLDEN_DIR) -> tuple[dict, dict]:
    d = pathlib.Path(d)
    meta = json.loads((d / "golden.json").read_text())
    with np.load(d / "golden.npz") as z:
        arrs = {k: z[k] for k in z.files}
    return meta, arrs


def part(arrs: dict, run: str, seq: str, mode: str) -> dict:
    pre = f"{run}/{seq}.{mode}/"
    return {k[len(pre):]: v for k, v in arrs.items() if k.startswith(pre)}


# ================================ running the reference ================================

def run_reference(model, tokens, decode_from: int | None = None) -> dict:
    """One sequence through ref/kimi_k31.py: a fresh prefill (decode_from None), or a prefill of
    decode_from tokens and then one token per call through the state, as the golden's decode runs.
    Returns tokens-first float64 arrays in the golden's layout, plus the reference's own router cut
    margins and indexer scores ([T, T], -inf where the reference masked)."""
    st = model.new_state()
    calls = [tokens] if decode_from is None else [tokens[:decode_from]] + [[t] for t in tokens[decode_from:]]
    T = len(tokens)
    c = model.cfg
    moe_layers = [il for il in range(c.n_layers) if il >= c.n_layer_dense_lead]
    mla = c.mla_layers()
    logits, seam, kseam = [], [], []
    ids = {il: [] for il in moe_layers}
    biased = {il: [] for il in moe_layers}
    sel = {il: [] for il in mla}
    scores = {il: np.full((T, T), -np.inf) for il in mla}
    t0 = 0
    for chunk in calls:
        lg = model.forward(chunk, st)
        n = len(chunk)
        logits.append(lg.T.astype(np.float64))
        seam.append(model.dumps[f"mla_out-{SEAM_LAYER}"].T.astype(np.float64))
        kseam.append(model.dumps[f"kda_out-{KDA_SEAM_LAYER}"].T.astype(np.float64))
        for il in moe_layers:
            ids[il].append(np.sort(model.dumps[f"ffn_moe_topk-{il}"], axis=0).T)
            biased[il].append(model.dumps[f"ffn_moe_probs_biased-{il}"].T.astype(np.float64))
        for il in mla:
            if f"top_k-{il}" in model.dumps:
                top = model.dumps[f"top_k-{il}"]
                sel[il] += [sorted(int(i) for i in top[:, j] if i >= 0) for j in range(n)]
                sc = model.dumps[f"indexer_score-{il}"]                   # (n_kv, n)
                scores[il][t0:t0 + n, :sc.shape[0]] = sc.T
            else:                                                       # indexer off: dense causal
                sel[il] += [list(range(t0 + j + 1)) for j in range(n)]
        t0 += n
    out = {"logits": np.concatenate(logits), SEAM: np.concatenate(seam), KDA_SEAM: np.concatenate(kseam)}
    for il in moe_layers:
        out[f"moe.{il}.expert_ids"] = np.concatenate(ids[il])
        b = np.sort(np.concatenate(biased[il]), axis=1)[:, ::-1]
        k = out[f"moe.{il}.expert_ids"].shape[1]
        out[f"moe.{il}.cut_margin"] = b[:, k - 1] - b[:, k] if b.shape[1] > k else np.full(T, np.inf)
    for il in mla:
        out[f"dsa.{il}.selected"] = sel[il]
        out[f"dsa.{il}.scores"] = scores[il]
    return out


# ================================ comparing ================================

@dataclass
class Report:
    failures: list = field(default_factory=list)       # hard disagreements: the gate fails on any
    moe_near_ties: list = field(default_factory=list)  # rows whose GOLDEN router cut margin < MOE_TIE (not judged)
    idx_ties: list = field(default_factory=list)       # rows whose sets differ only within tie_tol of the cut
    idx_exact_tie_rows: int = 0                        # golden rows with an exact tie at the cut (margin 0)
    stats: dict = field(default_factory=dict)          # observed maxima / rms, rows judged

    @property
    def ok(self) -> bool:
        return not self.failures

    def checks_failed(self) -> list[str]:
        """Which kinds of check failed (for the mutation table)."""
        kinds = []
        for f in self.failures:
            k = ("logits" if "logits" in f else "seam" if SEAM in f else "kda-seam" if KDA_SEAM in f
                 else "tie-free" if "tie-free" in f else "experts" if " moe." in f
                 else "indexer-scores" if "score" in f else "indexer-sets" if " dsa." in f else "other")
            if k not in kinds:
                kinds.append(k)
        return kinds

    def summary(self) -> str:
        s = self.stats
        lines = [f"gate {'PASS' if self.ok else 'FAIL'} ({s.get('dtype')}, {s.get('runs', 0)} runs): "
                 f"logits max {s.get('logits_max', 0):.3e} rms {s.get('logits_rms', 0):.3e}; "
                 f"{SEAM} max {s.get('seam_max', 0):.3e} rms {s.get('seam_rms', 0):.3e}; "
                 f"{KDA_SEAM} max {s.get('kda_seam_max', 0):.3e} rms {s.get('kda_seam_rms', 0):.3e}; "
                 f"indexer scores max {s.get('idx_score_max', 0):.3e}; "
                 f"expert rows {s.get('moe_rows', 0)}, {s.get('moe_rows_differ', 0)} differ, "
                 f"{len(self.moe_near_ties)} near-ties reported (golden min cut margin {s.get('moe_min_margin', 0):.2e}); "
                 f"indexer rows {s.get('idx_rows', 0)}, {s.get('idx_rows_differ', 0)} differ "
                 f"(golden min cut margin {s.get('idx_min_margin', 0):.2e}) "
                 f"({len(self.idx_ties)} only at a tie), golden rows with an exact tie at the cut "
                 f"{self.idx_exact_tie_rows}"]
        for f in self.failures[:12]:
            lines.append("  FAIL " + f)
        if len(self.failures) > 12:
            lines.append(f"  ... {len(self.failures) - 12} more")
        for t in self.moe_near_ties[:6]:
            lines.append("  NEAR-TIE (reported, not judged) " + t)
        for t in self.idx_ties[:6]:
            lines.append("  TIE (sets differ only at the cut) " + t)
        return "\n".join(lines)


def _bump(stats: dict, key: str, v: float) -> None:
    stats[key] = max(stats.get(key, 0.0), v)


def compare(ref: dict, gold: dict, gold_scores: dict, tag: str, rep: Report, acc: dict, tol: dict) -> None:
    """Compare one (sequence, mode) of the reference with the golden; adds to rep and the pooled sums."""
    if not _logits(ref, gold, tag, rep, acc, tol):
        return
    _seam(ref, gold, tag, rep, acc, tol)
    _tie_free(gold, tag, rep)
    _experts(ref, gold, tag, rep)
    _indexer(ref, gold, gold_scores, tag, rep)


def _logits(ref: dict, gold: dict, tag: str, rep: Report, acc: dict, tol: dict, vs: str = "golden") -> bool:
    """Full-vocabulary logits: max |diff| <= logits_atol (the pooled rms is the caller's). False when the
    logits cannot be compared at all (shape, non-finite). vs names what `gold` is, in the messages."""
    st = rep.stats
    lg, g = ref["logits"].astype(np.float64), gold["logits"].astype(np.float64)
    if lg.shape != g.shape:
        rep.failures.append(f"{tag}: logits shape {lg.shape} vs {vs} {g.shape}")
        return False
    if not np.isfinite(lg).all():
        rep.failures.append(f"{tag}: non-finite logits")
        return False
    d = np.abs(lg - g)
    acc["lg_ss"] += float((d ** 2).sum())
    acc["lg_n"] += d.size
    _bump(st, "logits_max", float(d.max()))
    if d.max() > tol["logits_atol"]:
        t, v = np.unravel_index(int(d.argmax()), d.shape)
        rep.failures.append(f"{tag}: logits max |diff| {d.max():.3e} > {tol['logits_atol']:.0e} "
                            f"(token {t}, vocab id {v}); argmax differs on "
                            f"{int((lg.argmax(1) != g.argmax(1)).sum())}/{lg.shape[0]} tokens")
    return True


def _experts(ref: dict, gold: dict, tag: str, rep: Report, vs: str = "golden", who: str = "the reference",
             tie: float = MOE_TIE) -> None:
    """Experts: exact sets on every row of every MoE layer the golden carries; a near-tie (by the GOLDEN's
    margin alone, so a reference cannot excuse its own disagreement by squeezing its scores) is reported,
    never silently judged. vs / who name `gold` and `ref` in the messages; tie is the near-tie band
    (TIE_BANDS[geometry]["router"])."""
    st = rep.stats
    for key in sorted(k for k in gold if k.startswith("moe.") and k.endswith(".expert_ids")):
        L = key.split(".")[1]
        gi, ri = gold[key], ref.get(key)
        if ri is None:
            rep.failures.append(f"{tag}: {who} has no MoE layer {L}")
            continue
        if ri.ndim != 2 or ri.shape[0] != gi.shape[0]:
            rep.failures.append(f"{tag} moe.{L}: expert ids shape {ri.shape} vs {vs} {gi.shape}")
            continue
        gm = gold[f"moe.{L}.cut_margin"].astype(np.float64)
        st["moe_min_margin"] = min(st.get("moe_min_margin", np.inf), float(gm.min()))
        for t in range(gi.shape[0]):
            st["moe_rows"] = st.get("moe_rows", 0) + 1
            margin = float(gm[t])
            same = ri.shape[1] == gi.shape[1] and np.array_equal(ri[t], gi[t])
            if margin < tie:
                rep.moe_near_ties.append(f"{tag} moe.{L} token {t}: {vs} cut margin {margin:.2e} < {tie:.1e}, "
                                         f"{'same' if same else 'DIFFERENT'} experts")
            if same:
                continue
            st["moe_rows_differ"] = st.get("moe_rows_differ", 0) + 1
            if ri.shape[1] != gi.shape[1] or margin >= tie:
                rep.failures.append(f"{tag} moe.{L} token {t}: experts {ri[t].tolist()} vs {vs} "
                                    f"{gi[t].tolist()} ({vs} cut margin {margin:.2e}, {who} "
                                    f"{float(ref[f'moe.{L}.cut_margin'][t]):.2e})")


def _indexer(ref: dict, gold: dict, gold_scores: dict, tag: str, rep: Report, vs: str = "golden",
             who: str = "the reference", score_atol: float = IDX_SCORE_ATOL) -> None:
    """Indexer, every MLA layer the golden carries: scores (-inf exactly where the golden's are -inf, no NaN
    or +inf anywhere), then the selected sets with tie tolerance against the golden scores. A reference's
    selection rows may carry -1 pads (they are not keys). vs / who name `gold` and `ref` in the messages;
    score_atol bounds the scores (IDX_SCORE_ATOL; check-dumps' own per geometry)."""
    import kimi_k31  # ref/kimi_k31.py: topk_sets_agree
    st = rep.stats
    for key in sorted(k for k in gold if k.startswith("dsa.") and k.endswith(".selected")):
        L = key.split(".")[1]
        gsel, rsel = gold[key], ref.get(key)
        if rsel is None:
            rep.failures.append(f"{tag}: {who} has no indexer on layer {L}")
            continue
        gsc = gold_scores[f"dsa.{L}.scores"].astype(np.float64)
        rsc = np.asarray(ref[f"dsa.{L}.scores"], dtype=np.float64)
        if rsc.shape != gsc.shape or len(rsel) != gsel.shape[0]:
            rep.failures.append(f"{tag} dsa.{L}: indexer scores shape {rsc.shape} vs {vs} {gsc.shape}, "
                                f"{len(rsel)} selection rows vs {gsel.shape[0]}")
            continue
        rep.idx_exact_tie_rows += int((gold[f"dsa.{L}.cut_margin"] == 0).sum())
        gneg, rneg = gsc == -np.inf, rsc == -np.inf       # future keys (padding included): -inf, nothing else
        odd = ~np.isfinite(rsc) & ~rneg
        if odd.any():
            rep.failures.append(f"{tag} dsa.{L}: indexer scores NaN on {int(np.isnan(rsc).sum())} and +inf on "
                                f"{int((rsc == np.inf).sum())} (query, key) cells")
        if not np.array_equal(gneg, rneg):
            rep.failures.append(f"{tag} dsa.{L}: indexer scores not -inf on {int((gneg & ~rneg).sum())} future "
                                f"and -inf on {int((rneg & ~gneg).sum())} visible (query, key) cells")
        both = np.isfinite(gsc) & np.isfinite(rsc)
        eps_obs = float(np.abs(rsc[both] - gsc[both]).max()) if both.any() else 0.0
        _bump(st, "idx_score_max", eps_obs)
        if eps_obs > score_atol:
            rep.failures.append(f"{tag} dsa.{L}: indexer score max |diff| {eps_obs:.3e} > {score_atol:.1e}")
        tie_tol = max(1e-6, 2.0 * min(eps_obs, score_atol))
        for t in range(gsel.shape[0]):
            st["idx_rows"] = st.get("idx_rows", 0) + 1
            gs = [int(i) for i in gsel[t] if i >= 0]
            rs = [int(i) for i in rsel[t] if i >= 0]
            if sorted(gs) == sorted(rs):
                continue
            st["idx_rows_differ"] = st.get("idx_rows_differ", 0) + 1
            if len(gs) == len(rs) and kimi_k31.topk_sets_agree(gsc[t], gs, rs, tie_tol):
                rep.idx_ties.append(f"{tag} dsa.{L} query {t}: keys {sorted(set(rs) ^ set(gs))} within "
                                    f"{tie_tol:.1e} of the cut")
            else:
                fut = [i for i in rs if i > t]
                rep.failures.append(f"{tag} dsa.{L} query {t}: keys {sorted(rs)} vs {vs} {sorted(gs)}"
                                    + (f" (future keys {fut})" if fut else ""))


def _seam(ref: dict, gold: dict, tag: str, rep: Report, acc: dict, tol: dict) -> None:
    for key, pre in ((SEAM, "seam"), (KDA_SEAM, "kda_seam")):
        if key not in gold:
            continue
        ds = np.abs(ref[key] - gold[key].astype(np.float64))
        acc[pre + "_ss"] += float((ds ** 2).sum())
        acc[pre + "_n"] += ds.size
        _bump(rep.stats, pre + "_max", float(ds.max()))
        if ds.max() > tol[pre + "_atol"]:
            rep.failures.append(f"{tag}: {key} max |diff| {ds.max():.3e} > {tol[pre + '_atol']:.0e}")


def _tie_free(gold: dict, tag: str, rep: Report, vs: str = "golden", geometry: str = "tiny") -> None:
    """The golden itself must be tie-free on every row judged on logits: its own indexer and router cut
    margins at least TIE_FREE x the geometry's tie bands (tie_free_bounds; the engines leave tie-breaking
    unspecified)."""
    tf = tie_free_bounds(geometry)
    for key in sorted(gold):
        if not key.endswith(".cut_margin"):
            continue
        m = gold[key].astype(np.float64)
        m = m[np.isfinite(m)]                             # inf: a dense indexer row, no cut
        if not m.size:
            continue
        dsa = key.startswith("dsa.")
        stat, bound = ("idx_min_margin", tf["indexer"]) if dsa else ("moe_min_margin", tf["router"])
        rep.stats[stat] = min(rep.stats.get(stat, np.inf), float(m.min()))
        if float(m.min()) < bound:
            rep.failures.append(f"{tag} {key}: the {vs} is not tie-free here: {int((m < bound).sum())} rows with a "
                                f"cut margin under {bound:.1e} (min {float(m.min()):.2e})")


def gate(model, golden: tuple[dict, dict] | None = None, run: str = GOLD, seqs=None, modes=MODES) -> Report:
    """The whole gate for one reference model against golden run `run` (GOLD: Moonshot's q_a/kv_a eps;
    ENGINE_EPS: the engines' eps, which carries only the layer-3 seam, so only that is checked)."""
    meta, arrs = golden if golden is not None else load_golden()
    dtype = "float64" if model.dt == np.float64 else "float32"
    tol = TOL[dtype]
    rep = Report(stats={"dtype": dtype, "runs": 0})
    acc = {"lg_ss": 0.0, "lg_n": 0, "seam_ss": 0.0, "seam_n": 0, "kda_seam_ss": 0.0, "kda_seam_n": 0}
    for s in meta["sequences"]:
        if seqs is not None and s["name"] not in seqs:
            continue
        scores = part(arrs, GOLD, s["name"], "prefill")      # both modes are judged against the prefill scores
        for mode in modes:
            gold = part(arrs, run, s["name"], mode)
            if not gold:
                continue
            ref = run_reference(model, s["tokens"], None if mode == "prefill" else s["decode_from"])
            tag = f"{s['name']}.{mode}"
            rep.stats["runs"] += 1
            if "logits" in gold:
                compare(ref, gold, scores, tag, rep, acc, tol)
            else:
                _seam(ref, gold, tag, rep, acc, tol)
    if acc["lg_n"]:
        rep.stats["logits_rms"] = math.sqrt(acc["lg_ss"] / acc["lg_n"])
        if rep.stats["logits_rms"] > tol["logits_rms"]:
            rep.failures.append(f"logits rms {rep.stats['logits_rms']:.3e} > {tol['logits_rms']:.0e} "
                                f"(pooled over {rep.stats['runs']} runs)")
    for key, pre in ((SEAM, "seam"), (KDA_SEAM, "kda_seam")):
        if acc[pre + "_n"]:
            rep.stats[pre + "_rms"] = math.sqrt(acc[pre + "_ss"] / acc[pre + "_n"])
            if rep.stats[pre + "_rms"] > tol[pre + "_rms"]:
                rep.failures.append(f"{key} rms {rep.stats[pre + '_rms']:.3e} > {tol[pre + '_rms']:.0e} "
                                    f"(pooled over {rep.stats['runs']} runs)")
    if rep.stats["runs"] == 0:
        rep.failures.append(f"nothing compared: golden run {run!r} has no matching sequence/mode")
    return rep


# ================================ picking the test sequences ================================

def _position_margins(model) -> tuple[float, float]:
    """After a forward call: the smallest router cut margin over the MoE layers and the smallest indexer
    cut margin over the MLA layers, over every position of the call (inf where at most top_k keys are
    visible)."""
    c = model.cfg
    moe = idx = math.inf
    for il in range(c.n_layer_dense_lead, c.n_layers):
        bt = model.dumps[f"ffn_moe_probs_biased-{il}"].astype(np.float64)
        k = c.n_expert_used
        for t in range(bt.shape[1]):
            b = np.sort(bt[:, t])[::-1]
            if b.size > k:
                moe = min(moe, float(b[k - 1] - b[k]))
    for il in c.mla_layers():
        st = model.dumps[f"indexer_score-{il}"].astype(np.float64)
        k = c.idx_top_k
        for t in range(st.shape[1]):
            sc = st[:, t]
            vis = np.sort(sc[np.isfinite(sc)])[::-1]
            if vis.size > k:
                idx = min(idx, float(vis[k - 1] - vis[k]))
    return moe, idx


def sequence_margins(model, tokens: list[int]) -> tuple[float, float]:
    """The smallest router and indexer cut margins over every position of `tokens`, run as one prefill:
    an independent check of a picked sequence (the pick grows it one token at a time)."""
    model.forward(list(tokens), model.new_state())
    return _position_margins(model)


def pick_tokens(model, n: int, rng, moe_min: float = PICK_ROOM * MOE_TIE,
                idx_min: float = PICK_ROOM * IDX_SCORE_ATOL, tries: int = 256) -> tuple[list[int], dict]:
    """Grow a test sequence one position at a time, so no cut sits at or near a tie. A candidate token
    (drawn from rng) is kept only if, at its own position, every MoE layer's router cut margin is
    >= moe_min and every MLA layer's indexer cut margin (when more than top_k keys are visible) is
    >= idx_min. Everything is causal, so a kept position never changes later. Use float64 models;
    `model` may be a list (one per eps preset), and then the candidate must clear the bounds in each.
    worst["moe"] / ["idx"] are the smallest margins over all of them, worst["per_model"] each one's."""
    import copy
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    states = [m.new_state() for m in models]
    toks: list[int] = []
    worst = {"moe": math.inf, "idx": math.inf, "draws": 0,
             "per_model": [{"moe": math.inf, "idx": math.inf} for _ in models]}
    for t in range(n):
        for _ in range(tries):
            cand = int(rng.integers(0, models[0].cfg.n_vocab))
            worst["draws"] += 1
            trials, got = [], []
            for m, st in zip(models, states):
                trial = copy.deepcopy(st)
                m.forward([cand], trial)
                moe, idx = _position_margins(m)
                trials.append(trial)
                got.append((moe, idx))
                if moe < moe_min or idx < idx_min:
                    break
            if len(got) == len(models) and all(a >= moe_min and b >= idx_min for a, b in got):
                break
        else:
            raise RuntimeError(f"position {t}: no token in {tries} draws keeps every cut margin above the bounds")
        toks.append(cand)
        states = trials
        for w, (moe, idx) in zip(worst["per_model"], got):
            w["moe"], w["idx"] = min(w["moe"], moe), min(w["idx"], idx)
        worst["moe"] = min(worst["moe"], min(a for a, _ in got))
        worst["idx"] = min(worst["idx"], min(b for _, b in got))
    return toks, worst


def check_geometry(cfg, geometry: str) -> None:
    """Refuse a GGUF whose geometry is not the named fixture geometry (pick / verify / gap take the
    sequence lengths and the bounds from the name, the weights from the file)."""
    import kimi_k31_synth_gguf as synth
    g = synth.GEOMETRIES[geometry]
    want = {"block_count": g["n_layers"], "embedding_length": g["n_embd"], "expert_count": g["n_expert"],
            "expert_used_count": g["n_expert_used"], "attention.indexer.top_k": g["idx_topk"],
            "vocab_size": g["vocab"]}
    have = {"block_count": cfg.n_layers, "embedding_length": cfg.n_embd, "expert_count": cfg.n_expert,
            "expert_used_count": cfg.n_expert_used, "attention.indexer.top_k": cfg.idx_top_k,
            "vocab_size": cfg.n_vocab}
    bad = {k: (have[k], want[k]) for k in want if have[k] != want[k]}
    if bad:
        raise ValueError(f"the GGUF is not the {geometry} fixture geometry (--geometry {geometry}): "
                         + ", ".join(f"{k} {h} != {w}" for k, (h, w) in bad.items()))


# ================================ export ================================

def export(run_a: pathlib.Path, run_b: pathlib.Path, golden_json: pathlib.Path | None, out: pathlib.Path) -> dict:
    """Cut the committed subset out of two tools/kimi_k31_moonshot_oracle.py run directories
    (run_a: Moonshot's q_a/kv_a eps -> GOLD, run_b: --mla-norm-eps 1e-5 -> ENGINE_EPS, same fixture)."""
    from safetensors.numpy import load_file
    ra = json.loads((run_a / "run.json").read_text())
    rb = json.loads((run_b / "run.json").read_text())
    if ra["fixture"] != rb["fixture"] or ra["sequences"] != rb["sequences"]:
        raise ValueError("runs a and b were made from different fixtures or sequences")
    for name, r in ((GOLD, ra), (ENGINE_EPS, rb)):
        fla = r.get("fla") or {}
        if r.get("dtype") != "float32" or not str(fla.get("fla", "")).startswith("fla-core") \
                or fla.get("fp32_dots") != "ieee":
            raise ValueError(f"run {name} is not an fp32 run of the real fla kernels with ieee dots: "
                             f"dtype {r.get('dtype')}, fla {fla}")
    arrs = {}
    for run, d in ((GOLD, run_a), (ENGINE_EPS, run_b)):
        for s in ra["sequences"]:
            for mode in MODES:
                f = load_file(str(d / f"{s['name']}.{mode}.safetensors"))
                if not np.array_equal(f["input_ids"], np.asarray(s["tokens"], dtype=np.int32)):
                    raise ValueError(f"{d}: {s['name']}.{mode} input ids do not match run.json")
                pre = f"{run}/{s['name']}.{mode}/"
                arrs[pre + SEAM] = f[SEAM].astype(np.float32)
                if run == ENGINE_EPS:
                    continue
                arrs[pre + KDA_SEAM] = f[KDA_SEAM].astype(np.float32)
                arrs[pre + "logits"] = f["logits"].astype(np.float32)
                if s["name"] == INDEXER_INPUT_SEQ and mode == "prefill":   # the indexer's inputs, for
                    for k, v in f.items():                                # tools/test_kimi_k31_oracle.py's
                        if k.startswith("dsa.") and k.endswith(".qr"):    # engine-formula check
                            L = k.split(".")[1]
                            arrs[pre + k] = v.astype(np.float32)
                            arrs[pre + f"layers.{L}.attn_in"] = f[f"layers.{L}.attn_in"].astype(np.float32)
                for k, v in f.items():
                    if k.startswith("moe.") and k.endswith((".expert_ids", ".cut_margin")):
                        arrs[pre + k] = v.astype(np.int16 if k.endswith("ids") else np.float32)
                    elif k.startswith("dsa.") and k.endswith((".selected", ".cut_margin")):
                        arrs[pre + k] = v.astype(np.int16 if k.endswith("selected") else np.float32)
                    elif k.startswith("dsa.") and k.endswith(".scores") and mode == "prefill":
                        arrs[pre + k] = v.astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "golden.npz", **arrs)
    keep = ("tool", "tool_sha256", "indexer_patch_sha256", "moonshot", "indexer_sources", "fla",
            "versions", "device", "dtype", "settings", "load", "seconds")   # file sha256s, not commits
    def scrub(r: dict) -> dict:                       # no local install paths in the committed metadata
        r = {k: r.get(k) for k in keep}
        r["fla"] = {k: v for k, v in (r.get("fla") or {}).items() if not k.endswith("_path")}
        return r

    meta = {
        "what": "Golden outputs of tools/kimi_k31_moonshot_oracle.py (Moonshot's Kimi K3 modeling code, real fla "
                "kernels in fp32 with ieee dots, lightning-indexer patch) on the tools/kimi_k31_synth_gguf.py "
                "fixture: the subset tools/kimi_k31_gate.py compares ref/kimi_k31.py against (see its docstring).",
        "runs": {GOLD: scrub(ra), ENGINE_EPS: scrub(rb)},
        "source_files_sha256": {GOLD: ra.get("files"), ENGINE_EPS: rb.get("files")},
        # the fixture files' sha256 identify the weights; manifest.json's own hash is left out, since the
        # manifest records the generator's sha256 and so changes with any edit of the tool
        "fixture": {k: v for k, v in ra["fixture"].items() if k != "manifest_sha256"},
        "sequences": ra["sequences"],
        "keys": sorted(arrs),
    }
    meta["notes"] = [
        f"{GOLD} is the golden: the real-fla fp32 run with Moonshot's q_a/kv_a eps 1e-6; {ENGINE_EPS} is the same "
        f"run with the engines' 1e-5 and carries only the {SEAM} seam.",
        "real fla is not bit-reproducible run to run (the noise section: '*repeat*'); the stand-in (pure-torch fla "
        "ops) is reproducible but is only a measurement here, never the golden.",
        "the sequences are picked tie-free (tools/kimi_k31_synth_gguf.py TEST_TOKENS): no indexer or router top-k "
        "cut lies near a tie, so the golden does not depend on how an implementation breaks ties.",
        f"{INDEXER_INPUT_SEQ}.prefill also carries each MLA layer's indexer inputs (layers.L.attn_in, dsa.L.qr), "
        "for the check against the engines' own score functions.",
    ]
    if golden_json is not None and golden_json.exists():
        g = json.loads(golden_json.read_text())
        meta["noise"] = {"note": "worst-case differences between independent-model runs (the oracle's compare: "
                                 "logits / hidden max abs and rel, selection rows that differ); run names: "
                                 "fla-* = real fla on CUDA, standin-* = the pure-torch stand-in",
                         **{k: v.get("worst") for k, v in g.get("measurements", {}).items()}}
    (out / "golden.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ================================ an engine's seam dumps (check-dumps) ================================

DUMP_INT = ("ffn_moe_topk", "top_k")      # selection seams: int32 in the raw layout, -1 pads
DUMP_PAD_INF = ("indexer_score", "kq_mask_top_k")   # (n_kv, n) per call: padded to T keys with -inf
# Seams whose last axis is a count the model itself decides (the experts or keys a token picks, the
# attention-residual blocks a mix reads): that width is part of the answer, judged where the seam is compared
# (experts truncated to 8 or a block pushed a layer late FAIL their rung, they are not refused), so only the
# other axes are validated. Every other seam must have the reference's shape exactly.
DUMP_COUNT_AXIS = ("ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "top_k",
                   "res_probs", "res_probs_attn", "res_probs_ffn")
# The seams the golden does not carry are judged against the float64 reference: (max |diff|, pooled rms).
# Each bound is about 3x (max) / 2x (rms) the float32 reference's own gap to the float64 one on the tiny
# fixture, all six runs (measured: l_out-0 3.3e-6 / 3.3e-7, ffn_moe_out-1 3.5e-6 / 3.4e-7, res_probs_ffn-3
# 2.4e-6 / 5.4e-7, res_mix_ffn-3 1.35e-5 / 1.37e-6, on values up to |2.1|, |1.1|, |0.98|, |4.8|), the same
# margins as the golden's bounds. tools/test_kimi_k31_check_dumps.py re-measures them, and checks that every
# ladder mutation moves the seam it is caught on by at least 10x its bound.
REF64_TOL = {"l_out": (1e-5, 7e-7), "ffn_moe_out": (1.1e-5, 7e-7),
             "res_probs_ffn": (7.5e-6, 1.1e-6), "res_mix_ffn": (4.1e-5, 2.8e-6)}
REF64 = "ref f64"
# The committed golden is the tiny fixture's. On any other fixture geometry check-dumps has no golden: every seam
# is judged against the float64 reference (plan M1a gate 2), on that fixture's own gate sequences
# (kimi_k31_synth_gguf.py gate_sequences, picked tie-free), with that geometry's tie bands (TIE_BANDS) and
# bounds (DUMP_TOL).
GOLDEN_GEOMETRY = "tiny"
# The big fixture (kimi_k31_synth_gguf.py --geometry big): each bound is 2x the float32 reference's own gap to
# the float64 one, max and pooled rms over all six runs (each prefill and each decode of seq0/1/2), the larger
# of the two eps presets, rounded up to two digits. Measured (max moonshot / llama.cpp; pooled rms, the larger):
#   kda_out-0 1.60e-6 / 1.71e-6; 2.29e-7     l_out-0 9.94e-6 / 8.12e-6; 4.35e-7
#   ffn_moe_out-1 5.12e-6 / 5.98e-6; 3.99e-7 mla_out-3 4.11e-6 / 5.30e-6; 3.47e-7
#   res_probs_ffn-3 1.04e-5 / 1.47e-5; 1.45e-6  res_mix_ffn-3 7.17e-5 / 9.25e-5; 2.62e-6
#   result_output 1.45e-3 / 1.50e-3; 4.23e-5  indexer scores (layers 3, 8, 9) 3.73e-4 / 3.67e-4
#   every expert and indexer set equal
# (scores reach |3.8|, logits |4.9|). The prefill-only gap (synth BIG_GAP, `gap --geometry big`) agrees.
# tools/test_kimi_k31_check_dumps.py (KIMI_K31_BIG_FIXTURE) re-measures them: each bound must be 1.9-2.5x the gap
# (measured 2.00-2.07x; the 1.9 leaves room for the BLAS thread count moving the gap).
DUMP_TOL = {
    "big": {"logits_atol": 3.1e-3, "logits_rms": 8.5e-5, "seam_atol": 1.1e-5, "seam_rms": 7.0e-7,
            "kda_seam_atol": 3.5e-6, "kda_seam_rms": 4.6e-7, "idx_score_atol": 7.5e-4},
}
DUMP_REF64_TOL = {
    "big": {"l_out": (2.0e-5, 8.7e-7), "ffn_moe_out": (1.2e-5, 8.0e-7),
            "res_probs_ffn": (3.0e-5, 2.9e-6), "res_mix_ffn": (1.9e-4, 5.3e-6)},
}


def dump_bounds(geometry: str, dtype: str) -> tuple[dict, dict]:
    """check-dumps' bounds on a fixture geometry: (the gate-layout seams' TOL-style bounds, idx_score_atol
    included; the REF64_TOL-style bounds of the seams only the float64 reference carries). On the golden's
    geometry these are the golden's own (TOL[dtype], IDX_SCORE_ATOL) and REF64_TOL; elsewhere DUMP_TOL /
    DUMP_REF64_TOL, the same for either engine dtype."""
    if geometry == GOLDEN_GEOMETRY:
        return {**TOL[dtype], "idx_score_atol": IDX_SCORE_ATOL}, REF64_TOL
    if geometry not in DUMP_TOL:
        raise DumpsRefused(f"geometry {geometry!r}: check-dumps has bounds for {GOLDEN_GEOMETRY}, "
                           + ", ".join(DUMP_TOL))
    return DUMP_TOL[geometry], DUMP_REF64_TOL[geometry]


class DumpsRefused(ValueError):
    """The dumps cannot be judged: unknown run, unknown seam, a seam on a layer that has no such seam, a
    malformed file, a missing seam a rung needs. Raised before anything is compared."""


def _split(name: str) -> tuple[str, int]:
    """`base-<layer>` -> (base, layer); anything else (a whole-model seam, `l_out-03`) -> (name, -1)."""
    base, _, il = name.rpartition("-")
    ok = base and il.isascii() and il.isdigit() and str(int(il)) == il
    return (base, int(il)) if ok else (name, -1)


def _is_int(name: str) -> bool:
    return _split(name)[0] in DUMP_INT


def reference_dumps(model, tokens, decode_from: int | None = None) -> dict:
    """Every cb() seam of one run of ref/kimi_k31.py (a fresh prefill, or decode_from tokens and then one
    token per call) in the dump layout check-dumps reads: tokens first (each array is the ggml-order tensor,
    axes fully reversed), the calls concatenated, indexer_score / kq_mask_top_k padded to T keys with -inf,
    top_k padded to top_k columns with -1."""
    st = model.new_state()
    calls = [tokens] if decode_from is None else [tokens[:decode_from]] + [[t] for t in tokens[decode_from:]]
    T = len(tokens)
    parts: dict[str, list] = {}
    for chunk in calls:
        model.forward(chunk, st)
        for name, a in model.dumps.items():
            parts.setdefault(name, []).append(a)
    out = {}
    for name, ps in parts.items():
        base = _split(name)[0]
        if base in DUMP_PAD_INF:
            ps = [np.pad(p, ((0, T - p.shape[0]), (0, 0)), constant_values=-np.inf) for p in ps]
        elif base == "top_k":
            w = max(model.cfg.idx_top_k, max(p.shape[0] for p in ps))
            ps = [np.pad(p, ((0, w - p.shape[0]), (0, 0)), constant_values=-1) for p in ps]
        a = np.concatenate(ps, axis=-1)
        out[name] = np.ascontiguousarray(a.transpose(tuple(range(a.ndim - 1, -1, -1))))
    return out


def golden_runs(meta: dict) -> list[tuple[str, list, int | None]]:
    """(run tag "<seq>.<mode>", tokens, decode_from or None) for every sequence and mode of the golden."""
    return [(f"{s['name']}.{mode}", s["tokens"], None if mode == "prefill" else s["decode_from"])
            for s in meta["sequences"] for mode in MODES]


def dump_runs(geometry: str = GOLDEN_GEOMETRY, meta: dict | None = None) -> list[tuple[str, list, int | None]]:
    """The runs check-dumps judges and dump-ref writes on a fixture geometry: the golden's (meta, default the
    committed one) on its geometry, otherwise the fixture's gate sequences at the fixture seed."""
    if geometry == GOLDEN_GEOMETRY:
        return golden_runs(meta if meta is not None else load_golden()[0])
    import kimi_k31_synth_gguf as synth
    return golden_runs({"sequences": synth.gate_sequences(synth.GEOMETRIES[geometry], synth.SEED)})


def write_dumps(runs: dict, out, fmt: str = "raw") -> None:
    """Write {run tag: {seam: tokens-first array}} as check-dumps reads it. raw: one directory per run
    with ref/glm.py's layout (manifest.txt lines `name d0 d1 ...` in ggml order, `name.f32` raw data,
    the selection seams int32); npz: one `<run>.npz` per run, the same arrays tokens first."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, dumps in runs.items():
        if fmt == "npz":
            np.savez(out / f"{tag}.npz", **dumps)
            continue
        d = out / tag
        d.mkdir(exist_ok=True)
        lines = []
        for name, a in dumps.items():
            np.ascontiguousarray(a, dtype=np.int32 if _is_int(name) else np.float32).tofile(d / f"{name}.f32")
            lines.append(f"{name} {' '.join(str(x) for x in reversed(a.shape))}")
        (d / "manifest.txt").write_text("\n".join(lines) + "\n")


def _load_raw(d: pathlib.Path) -> dict:
    mf = d / "manifest.txt"
    if not mf.is_file():
        raise DumpsRefused(f"{d.name}/: no manifest.txt")
    arrs: dict[str, np.ndarray] = {}
    for n, line in enumerate(mf.read_text().splitlines(), 1):
        f = line.split()
        if not f:
            continue
        name, dims = f[0], f[1:]
        if not dims or not all(x.isdigit() and int(x) > 0 for x in dims):
            raise DumpsRefused(f"{d.name}/manifest.txt line {n}: {line!r} is not `name d0 d1 ...`")
        if name in arrs:
            raise DumpsRefused(f"{d.name}/manifest.txt: {name} listed twice")
        shape = tuple(int(x) for x in reversed(dims))                  # ggml order -> tokens first
        p = d / f"{name}.f32"
        want = int(np.prod(shape)) * 4
        if not p.is_file() or p.stat().st_size != want:
            raise DumpsRefused(f"{d.name}/{name}.f32: {p.stat().st_size if p.is_file() else 'no'} bytes, "
                               f"the manifest says {want}")
        arrs[name] = np.fromfile(p, dtype=np.int32 if _is_int(name) else np.float32).reshape(shape)
    stray = sorted(p.stem for p in d.glob("*.f32") if p.stem not in arrs)
    if stray:
        raise DumpsRefused(f"{d.name}/: {', '.join(stray)} not in manifest.txt")
    return arrs


def _load_npz(p: pathlib.Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    for name, a in arrs.items():
        want = np.integer if _is_int(name) else np.floating
        if not np.issubdtype(a.dtype, want):
            raise DumpsRefused(f"{p.name}: {name} is {a.dtype}; {'integer' if _is_int(name) else 'float'} expected")
    return arrs


def load_dumps(d) -> dict:
    """{run tag: {seam: tokens-first array}} from an engine's dump directory: per run either a directory
    `<seq>.<mode>/` in the raw layout or a `<seq>.<mode>.npz` (other files, such as a log, are ignored)."""
    d = pathlib.Path(d)
    if not d.is_dir():
        raise DumpsRefused(f"{d}: not a directory")
    runs: dict[str, dict] = {}
    for p in sorted(d.iterdir()):
        if not (p.is_dir() or p.suffix == ".npz"):
            continue
        try:
            tag, arrs = (p.name, _load_raw(p)) if p.is_dir() else (p.stem, _load_npz(p))
        except DumpsRefused:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as err:   # unreadable: not a FAIL
            raise DumpsRefused(f"{p.name}: cannot be read ({type(err).__name__}: {err})") from err
        if tag in runs:
            raise DumpsRefused(f"{tag}: dumped both as a directory and as an .npz")
        runs[tag] = arrs
    if not runs:
        raise DumpsRefused(f"{d}: no <seq>.<mode>/ or <seq>.<mode>.npz dumps")
    return runs


@dataclass(frozen=True)
class Rung:
    n: int
    what: str
    golden: tuple     # gate-layout keys: judged against the golden where it carries them, else the f64 ref
    ref64: tuple      # cb() seams only the float64 reference carries


def ladder(cfg) -> list[Rung]:
    """The M1a seam ladder on this geometry, in order (each rung is a gate)."""
    kda = [il for il in range(cfg.n_layers) if cfg.is_recr(il)]
    mla, moe = cfg.mla_layers(), list(range(cfg.n_layer_dense_lead, cfg.n_layers))
    push = cfg.attn_res_block_size
    if not kda or not mla or not moe or cfg.n_layer_dense_lead < 1 or push >= cfg.n_layers:
        raise DumpsRefused(f"geometry not covered by the seam ladder: KDA layers {kda}, MLA layers {mla}, "
                           f"{cfg.n_layer_dense_lead} dense, block size {push} of {cfg.n_layers} layers")
    k0, m0, a0 = kda[0], moe[0], mla[0]
    return [
        Rung(1, f"layer-{k0} KDA attn_out", (f"layers.{k0}.attn_out",), ()),
        Rung(2, "layer-0 l_out (the dense SiTU FFN)", (), ("l_out-0",)),
        Rung(3, f"layer-{m0} ffn_moe_out, expert ids equal", (f"moe.{m0}.expert_ids",), (f"ffn_moe_out-{m0}",)),
        Rung(4, f"layer-{a0} MLA attn_out, indexer sets equal", (f"layers.{a0}.attn_out", f"dsa.{a0}.selected"), ()),
        Rung(5, f"the block push at layer {push}", (), (f"res_probs_ffn-{push}", f"res_mix_ffn-{push}")),
        Rung(6, "the full logits, every expert set, every indexer set",
             ("logits",) + tuple(f"moe.{L}.expert_ids" for L in moe) + tuple(f"dsa.{L}.selected" for L in mla), ()),
    ]


def _cb_names(key: str, cfg) -> tuple:
    """The dumped seams a gate-layout key is built from (ffn_moe_probs_biased, for the router margins in
    failure messages, is optional)."""
    if key == "logits":
        return ("result_output",)
    kind, L, what = key.split(".")
    if kind == "layers":
        return (f"{'kda' if cfg.is_recr(int(L)) else 'mla'}_out-{L}",)
    return (f"ffn_moe_topk-{L}",) if kind == "moe" else (f"top_k-{L}", f"indexer_score-{L}")


def _idx_margins(sc: np.ndarray, k: int) -> np.ndarray:
    out = np.full(sc.shape[0], np.inf)
    for t in range(sc.shape[0]):
        v = np.sort(sc[t][np.isfinite(sc[t])])[::-1]
        if v.size > k:
            out[t] = v[k - 1] - v[k]
    return out


def to_layout(d: dict, cfg) -> dict:
    """Dumped seams (cb() names, tokens first) -> the golden's layout, for the keys the seams allow."""
    out = {}
    if "result_output" in d:
        out["logits"] = d["result_output"].astype(np.float64)
    for il in range(cfg.n_layers):
        a = d.get(f"{'kda' if cfg.is_recr(il) else 'mla'}_out-{il}")
        if a is not None:
            out[f"layers.{il}.attn_out"] = a.astype(np.float64)
        ids = d.get(f"ffn_moe_topk-{il}")
        if ids is not None and ids.ndim == 2:
            out[f"moe.{il}.expert_ids"] = np.sort(ids.astype(np.int64), axis=1)
            b, k = d.get(f"ffn_moe_probs_biased-{il}"), cfg.n_expert_used
            m = np.full(ids.shape[0], np.nan)
            if b is not None and b.ndim == 2 and b.shape[0] == ids.shape[0] and b.shape[1] > k:
                bs = np.sort(b.astype(np.float64), axis=1)[:, ::-1]
                m = bs[:, k - 1] - bs[:, k]
            out[f"moe.{il}.cut_margin"] = m
        top, sc = d.get(f"top_k-{il}"), d.get(f"indexer_score-{il}")
        if top is not None and sc is not None:
            out[f"dsa.{il}.selected"] = top.astype(np.int64)
            out[f"dsa.{il}.scores"] = sc.astype(np.float64)
            out[f"dsa.{il}.cut_margin"] = _idx_margins(sc.astype(np.float64), cfg.idx_top_k) \
                if sc.ndim == 2 else np.full(len(top), np.inf)
    return out


def _layer_kind(cfg, il: int) -> str:
    if not 0 <= il < cfg.n_layers:
        return f"there is no layer {il} (the model has {cfg.n_layers})"
    return (f"layer {il} is a{' KDA' if cfg.is_recr(il) else 'n MLA'} layer with a "
            f"{'dense FFN' if il < cfg.n_layer_dense_lead else 'MoE'}")


def validate_dumps(runs: dict, expected: dict, cfg, required: set) -> int:
    """Refuse (DumpsRefused) dumps that cannot be judged; returns the number of seams validated. A seam is
    known if the reference dumps it under that name on that run; it must have the reference's shape (T token
    rows first), except the last axis of the DUMP_COUNT_AXIS seams, a count judged where the seam is
    compared."""
    unknown = sorted(set(runs) - set(expected))
    if unknown:
        raise DumpsRefused(f"{', '.join(unknown)}: not a run of the golden (runs: {', '.join(expected)})")
    missing = [t for t in expected if t not in runs]
    if missing:
        raise DumpsRefused(f"runs missing: {', '.join(missing)}")
    layers: dict[str, set] = {}
    for exp in expected.values():
        for name in exp:
            b, il = _split(name)
            layers.setdefault(b, set()).add(il)
    n = 0
    for tag, arrs in runs.items():
        exp = expected[tag]
        for name, a in arrs.items():
            if name not in exp:
                b, il = _split(name)
                if b not in layers:
                    raise DumpsRefused(f"{tag}/{name}: not a seam name of ref/kimi_k31.py (cb() names, "
                                       f"`name-<layer>` per layer)")
                where = sorted(layers[b])
                if where == [-1]:
                    raise DumpsRefused(f"{tag}/{name}: {b} is a whole-model seam, it has no layer")
                if il < 0:
                    raise DumpsRefused(f"{tag}/{name}: {b} is a per-layer seam, `{b}-<layer>`")
                raise DumpsRefused(f"{tag}/{name}: wrong layer: {_layer_kind(cfg, il)}; {b} exists on layers "
                                   f"{','.join(str(x) for x in where)}")
            e = exp[name]
            count = _split(name)[0] in DUMP_COUNT_AXIS
            if a.ndim != e.ndim or (a.shape[:-1] != e.shape[:-1] if count else a.shape != e.shape):
                raise DumpsRefused(f"{tag}/{name}: shape {a.shape}, the reference's is {e.shape} "
                                   f"(tokens first, {e.shape[0]} token rows"
                                   + ("; only the last axis, a count, may differ)" if count else ")"))
            n += 1
        lack = sorted(required - set(arrs))
        if lack:
            raise DumpsRefused(f"{tag}: missing seams the rungs need: {', '.join(lack)}")
    return n


def golden_run_for(model) -> str | None:
    """The golden run whose eps the model's match: GOLD (Moonshot's), ENGINE_EPS (q_a/kv_a 1e-5, the rest
    Moonshot's; it carries only the layer-3 seam), or None (judged against the f64 reference alone)."""
    import kimi_k31
    want = kimi_k31.KimiK31Config(model.g.g.metadata).eps()
    got = model.cfg.eps()
    if got == want:
        return GOLD
    if got == {**want, "mla_latent": 1e-5}:
        return ENGINE_EPS
    return None


def _bounded(got: np.ndarray, want: np.ndarray, label: str, bounds: tuple, vs: str, tag: str,
             rep: Report) -> None:
    """max |got - want| <= bounds[0] on this run; the pooled rms (bounds[1]) is finished by _finish()."""
    s = rep.stats.setdefault("seams", {}).setdefault(
        label, {"max": 0.0, "ss": 0.0, "n": 0, "atol": bounds[0], "rms_bound": bounds[1], "vs": set()})
    s["vs"].add(vs)
    if got.shape != want.shape:
        rep.failures.append(f"{tag}: {label} shape {got.shape} vs {vs} {want.shape}")
        return
    if not np.isfinite(got).all():
        rep.failures.append(f"{tag}: {label} non-finite on {int((~np.isfinite(got)).sum())} values")
        return
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    s["ss"] += float((d ** 2).sum())
    s["n"] += d.size
    s["max"] = max(s["max"], float(d.max()))
    if d.max() > bounds[0]:
        where = np.unravel_index(int(d.argmax()), d.shape)
        rep.failures.append(f"{tag}: {label} max |diff| {d.max():.3e} > {bounds[0]:.1e} vs {vs} "
                            f"(token {where[0]}, index {tuple(int(i) for i in where[1:])})")


def _finish(rep: Report, acc: dict, tol: dict) -> None:
    if acc["lg_n"]:
        rms = math.sqrt(acc["lg_ss"] / acc["lg_n"])
        rep.stats.setdefault("seams", {})["logits"] = {
            "max": rep.stats.get("logits_max", 0.0), "rms": rms, "atol": tol["logits_atol"],
            "rms_bound": tol["logits_rms"], "vs": acc["lg_vs"]}
    for label, s in rep.stats.get("seams", {}).items():
        if "ss" in s:
            s["rms"] = math.sqrt(s["ss"] / s["n"]) if s["n"] else 0.0
        if s["rms"] > s["rms_bound"]:
            rep.failures.append(f"{label} rms {s['rms']:.3e} > {s['rms_bound']:.1e} (pooled over "
                                f"{rep.stats['runs']} runs)")


def check_dumps(runs: dict, model64, golden: tuple[dict, dict] | None = None, dtype: str = "float32",
                rungs: int = 6, reference: str = "golden", expected: dict | None = None,
                geometry: str = GOLDEN_GEOMETRY) -> list[tuple[Rung, Report]]:
    """Judge an engine's dumps (load_dumps) rung by rung on the seam ladder. model64 is ref/kimi_k31.py in
    float64 with the eps the engine ran with; it gives the seams the golden does not carry, and every seam
    when reference is "ref64" or the eps match no golden run (expected: its reference_dumps per run, when
    the caller has them already). geometry names the fixture (kimi_k31_synth_gguf.py GEOMETRIES); on any but
    the golden's there is no golden (golden is ignored): the runs are dump_runs(geometry), every seam is
    judged against model64, with dump_bounds(geometry) and TIE_BANDS[geometry]. Raises DumpsRefused before
    comparing anything when the dumps cannot be judged."""
    if reference not in ("golden", "ref64"):
        raise ValueError(f"reference {reference!r}: golden or ref64")
    if model64.dt != np.float64:
        raise ValueError("check_dumps needs the float64 reference")
    c = model64.cfg
    try:
        check_geometry(c, geometry)
    except (KeyError, ValueError) as err:
        raise DumpsRefused(str(err)) from None
    tol, rtol = dump_bounds(geometry, dtype)
    tie = TIE_BANDS[geometry]["router"]
    if geometry == GOLDEN_GEOMETRY:
        meta, arrs = golden if golden is not None else load_golden()
        run = golden_run_for(model64) if reference == "golden" else None
    else:
        meta, arrs, run = None, {}, None
    tagged = dump_runs(geometry, meta)
    steps = [r for r in ladder(c) if r.n <= rungs]
    if expected is None:
        expected = {tag: reference_dumps(model64, toks, df) for tag, toks, df in tagged}
    required = {n for r in steps for k in r.golden for n in _cb_names(k, c)} | {n for r in steps for n in r.ref64}
    n_valid = validate_dumps(runs, expected, c, required)
    out = []
    for rung in steps:
        rep = Report(stats={"dtype": dtype, "runs": 0, "golden_run": run, "validated": n_valid})
        acc = {"lg_ss": 0.0, "lg_n": 0, "lg_vs": set()}
        for tag, toks, df in tagged:
            seq, mode = tag.split(".")
            eng, r64 = to_layout(runs[tag], c), to_layout(expected[tag], c)
            gpart = part(arrs, run, seq, mode) if run else {}
            rep.stats["runs"] += 1
            for key in rung.golden:
                g = key in gpart
                src, vs = (gpart, "golden " + run) if g else (r64, REF64)
                kind = key.split(".")[0]
                if key == "logits":
                    acc["lg_vs"].add(vs)
                    _logits(eng, src, tag, rep, acc, tol, vs)
                elif kind == "layers":
                    pre = "kda_seam" if c.is_recr(int(key.split(".")[1])) else "seam"
                    _bounded(eng[key], src[key], key, (tol[pre + "_atol"], tol[pre + "_rms"]), vs, tag, rep)
                else:
                    L = key.split(".")[1]
                    sub = {key: src[key], f"{kind}.{L}.cut_margin": src[f"{kind}.{L}.cut_margin"]}
                    _tie_free(sub, tag, rep, vs, geometry)
                    if kind == "moe":
                        _experts(eng, sub, tag, rep, vs, "the engine", tie)
                    else:
                        scores = part(arrs, GOLD, seq, "prefill") if g else r64
                        _indexer(eng, sub, scores, tag, rep, vs, "the engine", tol["idx_score_atol"])
            for name in rung.ref64:
                _bounded(runs[tag][name], expected[tag][name], name, rtol[_split(name)[0]], REF64, tag, rep)
        _finish(rep, acc, tol)
        out.append((rung, rep))
    return out


def rung_summary(rung: Rung, rep: Report) -> str:
    s = rep.stats
    parts = [f"{label} vs {'/'.join(sorted(v['vs']))} max {v['max']:.2e} (<= {v['atol']:.1e}) "
             f"rms {v['rms']:.2e} (<= {v['rms_bound']:.1e})" for label, v in s.get("seams", {}).items()]
    if "moe_rows" in s:
        parts.append(f"expert rows {s['moe_rows']}, {s.get('moe_rows_differ', 0)} differ")
    if "idx_rows" in s:
        parts.append(f"indexer rows {s['idx_rows']}, {s.get('idx_rows_differ', 0)} differ "
                     f"({len(rep.idx_ties)} only at a tie), scores max {s.get('idx_score_max', 0):.2e}")
    lines = [f"rung {rung.n} {'PASS' if rep.ok else 'FAIL'}: {rung.what}: " + "; ".join(parts)]
    lines += ["  FAIL " + f for f in rep.failures[:8]]
    if len(rep.failures) > 8:
        lines.append(f"  ... {len(rep.failures) - 8} more")
    lines += ["  NEAR-TIE (reported, not judged) " + t for t in rep.moe_near_ties[:4]]
    lines += ["  TIE (sets differ only at the cut) " + t for t in rep.idx_ties[:4]]
    return "\n".join(lines)


def fixture_gguf(meta: dict) -> tuple[str, int | None] | None:
    """(sha256, bytes) of the GGUF the golden was made from (its fixture's one .gguf), if recorded."""
    files = (meta.get("fixture") or {}).get("manifest_files") or {}
    ggufs = [(v["sha256"], v.get("bytes")) for k, v in files.items() if k.endswith(".gguf")]
    return ggufs[0] if len(ggufs) == 1 else None


def is_fixture_gguf(path: pathlib.Path, meta: dict) -> bool:
    """path is the golden's fixture GGUF: its size first (a real K3.1 GGUF is refused without reading it),
    then its sha256."""
    want = fixture_gguf(meta)
    if want is None:
        return False
    sha, size = want
    return (size is None or path.stat().st_size == size) and sha256(path) == sha


def verify_picks(path, geometry: str, seed: int) -> dict:
    """`pick --verify`: the recorded sequences (tools/kimi_k31_synth_gguf.py gate_sequences) as whole
    float64 prefills, under every eps preset of the reference; every router / indexer cut margin must
    clear the geometry's pick bounds. Returns the report (res["ok"])."""
    import kimi_k31
    import kimi_k31_synth_gguf as synth
    seqs = synth.gate_sequences(synth.GEOMETRIES[geometry], seed)
    bounds = pick_bounds(geometry)
    res, ok = {"bounds": bounds, "presets": {}}, True
    for eps in sorted(kimi_k31.EPS_PRESETS, key=lambda p: p != "moonshot"):
        m = kimi_k31.KimiK31(path, dtype="float64", eps=eps, cache_weights=True)
        r = res["presets"][eps] = {}
        try:
            check_geometry(m.cfg, geometry)
            for s in seqs:
                moe, idx = sequence_margins(m, s["tokens"])
                good = s["tie_free"] and moe >= bounds["router"] and idx >= bounds["indexer"]
                ok = ok and good
                r[s["name"]] = {"tokens": len(s["tokens"]), "tie_free_recorded": s["tie_free"],
                                "min_router_margin": moe, "min_indexer_margin": idx, "ok": good}
        finally:
            m.close()
    res["ok"] = ok
    return res


def pick(path, geometry: str, seed: int) -> dict:
    """`pick`: grow seq0 and seq2 for the geometry (lengths from kimi_k31_synth_gguf.seq_plan, candidates
    from default_rng(seed + 1) / (seed + 2)) under every preset in PICK_EPS[geometry]. Returns the report."""
    import kimi_k31
    import kimi_k31_synth_gguf as synth
    plan = synth.seq_plan(synth.GEOMETRIES[geometry])
    bounds = pick_bounds(geometry)
    presets = PICK_EPS[geometry]
    models = []
    try:
        for eps in presets:
            models.append(kimi_k31.KimiK31(path, dtype="float64", eps=eps, cache_weights=True))
            check_geometry(models[-1].cfg, geometry)
        res = {"bounds": bounds, "presets": list(presets)}
        for name, off in (("seq0", 1), ("seq2", 2)):
            toks, worst = pick_tokens(models, plan[name][0], np.random.default_rng(seed + off),
                                      moe_min=bounds["router"], idx_min=bounds["indexer"])
            res[name] = {"tokens": toks, "min_router_margin": worst["moe"], "min_indexer_margin": worst["idx"],
                         "draws": worst["draws"]}
            if len(presets) > 1:
                res[name]["per_preset"] = {p: {"min_router_margin": w["moe"], "min_indexer_margin": w["idx"]}
                                           for p, w in zip(presets, worst["per_model"])}
    finally:
        for m in models:
            m.close()
    return res


# ================================ the float32-vs-float64 gap ================================

def _gap_run(path, geometry: str, dtype: str, eps: str, seqs: list[dict]) -> dict:
    """One reference run of every sequence (one prefill each): the arrays `gap` compares."""
    import kimi_k31
    m = kimi_k31.KimiK31(path, dtype=dtype, eps=eps, cache_weights=True)
    out = {}
    try:
        check_geometry(m.cfg, geometry)
        c = m.cfg
        k, ki = c.n_expert_used, c.idx_top_k
        for s in seqs:
            lg = m.forward(list(s["tokens"]), m.new_state())
            d = {"logits": np.asarray(lg, np.float64),
                 "mla": m.dumps[f"mla_out-{SEAM_LAYER}"].astype(np.float64),
                 "kda": m.dumps[f"kda_out-{KDA_SEAM_LAYER}"].astype(np.float64),
                 "router": {}, "experts": {}, "indexer": {}, "selected": {}}
            for il in range(c.n_layer_dense_lead, c.n_layers):
                b = m.dumps[f"ffn_moe_probs_biased-{il}"].astype(np.float64)
                sb = -np.sort(-b, axis=0)
                d["router"][il] = (b, sb[k - 1] - sb[k])
                d["experts"][il] = np.sort(m.dumps[f"ffn_moe_topk-{il}"], axis=0)
            for il in c.mla_layers():
                sc = m.dumps[f"indexer_score-{il}"].astype(np.float64)
                cut = np.full(sc.shape[1], np.inf)
                for t in range(sc.shape[1]):
                    v = np.sort(sc[np.isfinite(sc[:, t]), t])[::-1]
                    if v.size > ki:
                        cut[t] = v[ki - 1] - v[ki]
                d["indexer"][il] = (sc, cut)
                d["selected"][il] = np.sort(m.dumps[f"top_k-{il}"], axis=0)
            out[s["name"]] = d
    finally:
        m.close()
    return out


def _gap_stats(x: dict, y: dict) -> dict:
    """y against x (x the float64 run), one sequence."""
    def mx(a, b):
        return float(np.abs(a - b).max())

    def rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    r = {"logits_max": mx(x["logits"], y["logits"]), "logits_rms": rms(x["logits"], y["logits"]),
         "logits_peak": float(np.abs(x["logits"]).max()),
         "mla_seam_max": mx(x["mla"], y["mla"]), "mla_seam_rms": rms(x["mla"], y["mla"]),
         "kda_seam_max": mx(x["kda"], y["kda"]), "kda_seam_rms": rms(x["kda"], y["kda"]),
         "router_score_max": max(mx(x["router"][il][0], y["router"][il][0]) for il in x["router"]),
         "router_margin_shift_max": max(mx(x["router"][il][1], y["router"][il][1]) for il in x["router"]),
         "expert_rows_differ": int(sum(int((x["experts"][il] != y["experts"][il]).any(axis=0).sum())
                                       for il in x["experts"])),
         "indexer_rows_differ": int(sum(int((x["selected"][il] != y["selected"][il]).any(axis=0).sum())
                                        for il in x["selected"]))}
    smax = shift = peak = 0.0
    for il in x["indexer"]:
        (sa, ca), (sb, cb) = x["indexer"][il], y["indexer"][il]
        vis = np.isfinite(sa)
        if not np.array_equal(vis, np.isfinite(sb)):
            raise AssertionError(f"layer {il}: the two runs mask different keys")
        smax, peak = max(smax, mx(sa[vis], sb[vis])), max(peak, float(np.abs(sa[vis]).max()))
        cut = np.isfinite(ca)
        if cut.any():
            shift = max(shift, mx(ca[cut], cb[cut]))
    r.update({"indexer_score_max": smax, "indexer_score_peak": peak, "indexer_margin_shift_max": shift})
    return r


def gap(path, geometry: str, seed: int) -> dict:
    """`gap`: the float32 reference against the float64 reference on the geometry's gate sequences (each
    one prefill), under each eps preset - the logits, the layer-3 MLA and layer-0 KDA seams, the router
    and indexer scores and cut margins, and the selections - plus the largest cut-margin shift any run
    makes against float64 moonshot (what TIE_BANDS is sized from). Plan M1a gate 2 bounds the engine on
    the big fixture at 2x these gaps."""
    import kimi_k31
    import kimi_k31_synth_gguf as synth
    seqs = synth.gate_sequences(synth.GEOMETRIES[geometry], seed)
    presets = sorted(kimi_k31.EPS_PRESETS, key=lambda p: p != "moonshot")
    res = {"tool": f"python tools/kimi_k31_gate.py gap --geometry {geometry}", "presets": {}}
    base = None
    shift = {"router": 0.0, "indexer": 0.0}
    pooled_keys = (("logits", "logits_rms"), ("mla", "mla_seam_rms"), ("kda", "kda_seam_rms"))
    for eps in presets:
        f64 = _gap_run(path, geometry, "float64", eps, seqs)
        f32 = _gap_run(path, geometry, "float32", eps, seqs)
        if base is None:
            base = f64
        per = {s: _gap_stats(f64[s], f32[s]) for s in f64}
        worst = {k: max(v[k] for v in per.values()) for k in next(iter(per.values()))}
        for key, name in pooled_keys:
            num = sum(float(((f64[s][key] - f32[s][key]) ** 2).sum()) for s in f64)
            worst[name] = (num / sum(f64[s][key].size for s in f64)) ** 0.5      # pooled over the sequences
        res["presets"][eps] = {"f32_vs_f64": per, "all": worst}
        for run in ((f32,) if eps == presets[0] else (f64, f32)):
            for s in base:
                st = _gap_stats(base[s], run[s])
                shift["router"] = max(shift["router"], st["router_margin_shift_max"])
                shift["indexer"] = max(shift["indexer"], st["indexer_margin_shift_max"])
        del f64, f32
    res["margin_shift_vs_f64_moonshot"] = shift
    return res


# ================================ CLI ================================

def pinned_gguf(geometry: str) -> str | None:
    """The sha256 of the fixture GGUF a geometry's check-dumps bounds were measured on (the one its sequences
    were picked on, kimi_k31_synth_gguf.py pick_record), off the golden's geometry."""
    import kimi_k31_synth_gguf as synth
    g = synth.GEOMETRIES[geometry]
    rec = synth.pick_record(g, synth.SEED) or {}
    return rec.get("picked_on", {}).get("sha256", {}).get(synth.gguf_name(g))


def fixture_config(path: pathlib.Path, geometry: str, eps: str = "moonshot", eps_overrides: dict | None = None):
    """The reference's config of `path`, from its GGUF header alone, checked against the fixture geometry
    (check_geometry) before any tensor is opened or the file is hashed: a real K3.1 GGUF costs a header read.
    Whatever keeps the file from being judged - no such file, not a GGUF, a header the reference does not
    read, another geometry, then (from the tensor directory and the file's size alone) a tensor of a type
    ref/load.py does not decode or whose data runs past the end of the file - is a DumpsRefused."""
    import kimi_k31
    from gguf_reader import GGUFFile
    from load import GEOM
    path = pathlib.Path(path)
    if not path.is_file():
        raise DumpsRefused(f"{path}: no such file")
    try:
        gf = GGUFFile(path)
    except (OSError, ValueError, KeyError, EOFError, struct.error) as err:
        raise DumpsRefused(f"{path}: cannot be read as a GGUF ({type(err).__name__}: {err})") from None
    meta = gf.metadata
    try:
        cfg = kimi_k31.KimiK31Config(meta, eps=eps, eps_overrides=eps_overrides)
    except ValueError as err:
        raise DumpsRefused(f"{path}: {err}") from None
    try:
        check_geometry(cfg, geometry)
    except ValueError as err:
        raise DumpsRefused(str(err)) from None
    size = path.stat().st_size
    for t in gf.tensors:
        if t.type_name not in GEOM:
            raise DumpsRefused(f"{path}: tensor {t.name} is {t.type_name}, a type ref/load.py does not decode")
        el, by = GEOM[t.type_name]
        if t.elements % el:
            raise DumpsRefused(f"{path}: tensor {t.name} ({t.type_name}, {t.elements} elements) is not whole "
                               f"{el}-element blocks")
        end = gf.data_start + t.offset + t.elements // el * by
        if end > size:
            raise DumpsRefused(f"{path}: tensor {t.name} ends at byte {end}, past the end of the file ({size} "
                               f"bytes): its data is cut")
    return cfg


def _dumps_cli(a, path: pathlib.Path) -> int:
    """check-dumps / dump-ref. Refusals (exit 2, a `<cmd>: refused:` line) all come before the weights are
    read: --golden on a geometry without one, the GGUF's header and geometry, then (check-dumps) the fixture's
    identity, and the dumps themselves (check_dumps). An error after that is a real one and is raised."""
    import kimi_k31
    eps = {k: v for k, v in (("mla_latent", a.eps_mla), ("kda_l2", a.eps_kda)) if v is not None}
    geo = a.geometry
    try:
        if geo != GOLDEN_GEOMETRY and a.golden is not None:
            raise DumpsRefused(f"--golden: the {geo} fixture has no golden (every seam is judged against the "
                               f"float64 reference); --golden is for --geometry {GOLDEN_GEOMETRY} only")
        golden = load_golden(a.golden or GOLDEN_DIR) if geo == GOLDEN_GEOMETRY else None
        fixture_config(path, geo, a.eps, eps or None)
        if a.cmd == "check-dumps":
            if golden is not None:          # tiny: its size, then its sha256
                if not is_fixture_gguf(path, golden[0]):
                    want = fixture_gguf(golden[0])
                    raise DumpsRefused(f"{path} is not the GGUF the golden in {a.golden or GOLDEN_DIR} was made "
                                       f"from (sha256 {want[0] if want else None}, {want[1] if want else None} "
                                       f"bytes); the bounds were measured on that fixture")
            else:
                want = pinned_gguf(geo)
                if want is None or sha256(path) != want:
                    raise DumpsRefused(f"{path} is not the {geo} fixture GGUF (sha256 {want}); the bounds were "
                                       f"measured on that fixture")
    except DumpsRefused as err:
        print(f"{a.cmd}: refused: {err}")
        return 2
    if a.cmd == "dump-ref":
        m = kimi_k31.KimiK31(path, dtype=a.dtype, eps=a.eps, eps_overrides=eps or None)
        try:
            runs = {tag: reference_dumps(m, toks, df) for tag, toks, df in dump_runs(geo, golden and golden[0])}
        finally:
            m.close()
        write_dumps(runs, a.out, a.format)
        print(f"{a.out}: {len(runs)} runs, {sum(len(r) for r in runs.values())} seams ({a.dtype}, {a.format}, "
              f"{geo} fixture)")
        return 0
    m = kimi_k31.KimiK31(path, dtype="float64", eps=a.eps, eps_overrides=eps or None)
    try:
        runs = load_dumps(a.dumps)
        res = check_dumps(runs, m, golden, a.dtype, a.rungs, a.reference, geometry=geo)
        run, e = res[0][1].stats["golden_run"], m.cfg.eps()
        n_all = len(ladder(m.cfg))
    except DumpsRefused as err:
        print(f"check-dumps: refused: {err}")
        return 2
    finally:
        m.close()
    print(f"check-dumps {a.dumps}: {a.dtype} dumps, {geo} fixture, eps "
          + ", ".join(f"{k} {v:g}" for k, v in e.items())
          + f"; golden run {run or 'none (the float64 reference alone)'}; "
          f"{res[0][1].stats['validated']} seams validated, rungs 1-{a.rungs} of {n_all} judged")
    for rung, rep in res:
        print(rung_summary(rung, rep))
    bad = [rung.n for rung, rep in res if not rep.ok]
    if bad:
        print(f"check-dumps FAIL (failing rungs {bad}, first {bad[0]})")
    elif a.rungs < n_all:                     # never the bare PASS line a script greps for
        print(f"check-dumps rungs 1-{a.rungs} of {n_all} pass (rungs {a.rungs + 1}-{n_all} not judged)")
    else:
        print("check-dumps PASS")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="cut the golden subset out of two oracle run directories")
    e.add_argument("--run-1e-6", dest="a", required=True, help="oracle run directory with Moonshot's q_a/kv_a eps")
    e.add_argument("--run-1e-5", dest="b", required=True, help="oracle run directory with --mla-norm-eps 1e-5")
    e.add_argument("--golden-json", default=None, help="the oracle's golden.json (noise measurements)")
    e.add_argument("--out", default=str(GOLDEN_DIR))
    pk = sub.add_parser("pick", help="pick tie-free test sequences for the fixture (prints JSON)")
    pk.add_argument("--gguf", default=None, help="default: build the fixture into a temp dir")
    pk.add_argument("--seed", type=int, default=None, help="default: the fixture seed")
    pk.add_argument("--geometry", default="tiny", choices=("tiny", "big"),
                    help="the fixture geometry (tools/kimi_k31_synth_gguf.py --geometry); default tiny")
    pk.add_argument("--verify", action="store_true",
                    help="instead of picking: run the tool's recorded sequences for this geometry and seed as "
                         "whole prefills under each eps preset and check every cut margin against the pick "
                         "bounds (exit 1 if not)")
    gp = sub.add_parser("gap", help="float32 vs float64 reference on the fixture's sequences (prints JSON)")
    gp.add_argument("--gguf", default=None, help="default: build the fixture into a temp dir")
    gp.add_argument("--seed", type=int, default=None, help="default: the fixture seed")
    gp.add_argument("--geometry", default="tiny", choices=("tiny", "big"),
                    help="the fixture geometry (tools/kimi_k31_synth_gguf.py --geometry); default tiny")
    c = sub.add_parser("check", help="run ref/kimi_k31.py through the gate")
    c.add_argument("--gguf", default=None, help="default: build the fixture into a temp dir")
    c.add_argument("--golden", default=str(GOLDEN_DIR))
    c.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    c.add_argument("--eps-mla", type=float, default=None, help="override the q_a/kv_a eps (default 1e-6)")
    c.add_argument("--eps-kda", type=float, default=None, help="override the KDA q/k l2 eps (default 1e-6)")
    cd = sub.add_parser("check-dumps", aliases=["check-engine"],
                        help="judge an engine's seam dumps on the M1a seam ladder (exit 0 pass, 1 fail, 2 refused; "
                             "a refusal prints a `check-dumps: refused:` line, a usage error exits 2 too)")
    dr = sub.add_parser("dump-ref", help="write ref/kimi_k31.py's own dumps of every run check-dumps judges "
                                         "(the golden's on tiny; check-dumps' layout)")
    for p in (cd, dr):
        p.add_argument("--gguf", default=None, help="the fixture GGUF (default: build it into a temp dir)")
        p.add_argument("--geometry", default=GOLDEN_GEOMETRY, choices=("tiny", "big"),
                       help="the fixture geometry (tools/kimi_k31_synth_gguf.py --geometry); default tiny, the "
                            "golden's; on big every seam is judged against the float64 reference, on the big "
                            "fixture's own sequences and bounds (DUMP_TOL)")
        p.add_argument("--golden", default=None,
                       help=f"the golden directory, --geometry {GOLDEN_GEOMETRY} only (default "
                            f"{GOLDEN_DIR.parent.name}/{GOLDEN_DIR.name}); refused on a geometry without a golden")
        p.add_argument("--eps", default="moonshot", choices=("moonshot", "llama.cpp"),
                       help="the eps preset the engine ran with (ref/kimi_k31.py's presets)")
        p.add_argument("--eps-mla", type=float, default=None, help="override the q_a/kv_a eps")
        p.add_argument("--eps-kda", type=float, default=None, help="override the KDA q/k l2 eps")
    cd.add_argument("--dumps", required=True, help="the engine's dump directory (<seq>.<mode>/ or <seq>.<mode>.npz)")
    cd.add_argument("--dtype", default="float32", choices=("float32", "float64"),
                    help="the engine's activation dtype: picks the golden's bounds")
    cd.add_argument("--reference", default="golden", choices=("golden", "ref64"),
                    help="golden: the golden where it carries a seam, the f64 reference elsewhere; ref64: the f64 "
                         "reference for every seam")
    cd.add_argument("--rungs", type=int, default=6, choices=range(1, 7), help="judge rungs 1..N (default all 6)")
    dr.add_argument("--out", required=True)
    dr.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    dr.add_argument("--format", default="raw", choices=("raw", "npz"))
    a = ap.parse_args(argv)
    if a.cmd == "check-engine":                   # the plan's name for it
        a.cmd = "check-dumps"
    if a.cmd == "export":
        meta = export(pathlib.Path(a.a), pathlib.Path(a.b), pathlib.Path(a.golden_json) if a.golden_json else None,
                      pathlib.Path(a.out))
        print(f"{a.out}: {len(meta['keys'])} arrays, {(pathlib.Path(a.out) / 'golden.npz').stat().st_size} bytes")
        return 0
    import tempfile
    import kimi_k31
    import kimi_k31_synth_gguf as synth
    geo = synth.GEOMETRIES[getattr(a, "geometry", "tiny")]
    seed = synth.SEED if getattr(a, "seed", None) is None else a.seed
    with tempfile.TemporaryDirectory(prefix="kimi-k31-gate-") as tmp:
        path = a.gguf
        if path is None:
            path = pathlib.Path(tmp) / synth.gguf_name(geo)
            vals, _, _ = synth.make_weights(geo, seed)
            md, ts = synth.to_gguf(vals, geo)
            del vals
            synth.write_gguf_stream(path, md, ts)
            del md, ts
        if a.cmd in ("check-dumps", "dump-ref"):
            return _dumps_cli(a, pathlib.Path(path))
        if a.cmd in ("pick", "gap"):
            try:
                if a.cmd == "gap":
                    print(json.dumps(gap(path, a.geometry, seed)))
                    return 0
                if a.verify:
                    res = verify_picks(path, a.geometry, seed)
                    print(json.dumps(res))
                    return 0 if res["ok"] else 1
                res = pick(path, a.geometry, seed)
            except ValueError as err:                    # check_geometry: --gguf is another geometry
                print(f"kimi_k31_gate.py {a.cmd}: {err}", file=sys.stderr)
                return 2
            print(json.dumps(res))
            return 0
        eps = {k: v for k, v in (("mla_latent", a.eps_mla), ("kda_l2", a.eps_kda)) if v is not None}
        m = kimi_k31.KimiK31(path, dtype=a.dtype, eps_overrides=eps or None)
        try:
            golden = load_golden(a.golden)
            rep = gate(m, golden)
            print(rep.summary())
            rb = gate(m, golden, run=ENGINE_EPS)
            print(f"vs {ENGINE_EPS}, {SEAM} only: rms {rb.stats.get('seam_rms', 0):.3e} "
                  f"max {rb.stats.get('seam_max', 0):.3e} -> {'PASS' if rb.ok else 'FAIL'}")
        finally:
            m.close()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
