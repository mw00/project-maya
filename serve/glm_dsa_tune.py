"""serve/glm_dsa_tune.py - glm-dsa's per-request engine knobs (a request tunes the decode, no restart needed).

A request's `strata_tune` object may name, besides Maya's keys (pcie_frac, cpu_threads, read_chunks, spec, spec_min_p,
spec_n_min, spec_cum), the engine settings below; StrataEngine.sampling_keys (serve/server.py, one hook) appends what
engine_keys() returns to the GEN line.  The engine applies them from the request's first token and puts every knob the
request leaves out back to the value it started with (include/strata/core/glm_req_knobs.hpp), so a request without
them runs exactly as the engine started.  A value outside a key's range, or of the wrong type, is dropped here.

    promote_min      0..65535        STRATA_GLM_PROMOTE_MIN: keep a fetched expert only when its aged count clears it
    prefetch_n       0..2            STRATA_GLM_PREFETCH_N: predicted next-layer experts copied ahead (> 0: no graphs)
    prefetch_at      route|fetch|cpu STRATA_GLM_PREFETCH_AT: when that copy starts
    prefetch_rank    0..64           STRATA_GLM_PREFETCH_RANK: only the prediction's first ranks (0: any)
    prefetch_blocks  0..1024         STRATA_GLM_PREFETCH_BLOCKS: the copy's thread blocks (0: half the SMs)
    mtp_keep         0..8            STRATA_GLM_MTP_KEEP: the NextN draft fetches its missing experts among its n best
    cpu_split        0..64           STRATA_GLM_CPU_SPLIT: CPU-lane pieces a thread (0: about 48 in all)
    vram_evict       lfu|lru         STRATA_GLM_VRAM_EVICT
    ngram            on/off          STRATA_GLM_NGRAM_DRAFT (true / false, or 1 / 0)
    lane_row4        on/off          STRATA_GLM_LANE_ROW4
    graphs           on/off          off: this request decodes without the glm-dsa decode graphs (on: as started)
    pcie_frac_head   0..1            the speculative head's parts' CPU-lane share (the tail keeps pcie_frac)

Speculation off for one request (the token-by-token decode) is Maya's existing key: "spec": 1.
"""
from __future__ import annotations

# key -> (smallest, largest): integers (a JSON true / false is not a number here)
INT_KEYS = {"promote_min": (0, 65535), "prefetch_n": (0, 2), "prefetch_rank": (0, 64), "prefetch_blocks": (0, 1024),
            "mtp_keep": (0, 8), "cpu_split": (0, 64)}
# key -> the words it takes
WORD_KEYS = {"prefetch_at": ("route", "fetch", "cpu"), "vram_evict": ("lfu", "lru")}
# on/off keys: true / false or 1 / 0
SWITCH_KEYS = ("ngram", "lane_row4", "graphs")
# shares in [0, 1]
SHARE_KEYS = ("pcie_frac_head",)
KEYS = (*INT_KEYS, *WORD_KEYS, *SWITCH_KEYS, *SHARE_KEYS)


def engine_keys(tune) -> str:
    """The GEN keys (" k=v" each, in KEYS order) for the knobs of an `strata_tune` object; "" for anything else."""
    if not isinstance(tune, dict):
        return ""
    out = ""
    for k, (lo, hi) in INT_KEYS.items():
        v = tune.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
            out += f" {k}={v}"
    for k, words in WORD_KEYS.items():
        v = tune.get(k)
        if isinstance(v, str) and v in words:
            out += f" {k}={v}"
    for k in SWITCH_KEYS:
        v = tune.get(k)
        if isinstance(v, bool) or (isinstance(v, int) and v in (0, 1)):
            out += f" {k}={int(v)}"
    for k in SHARE_KEYS:
        v = tune.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
            out += f" {k}={float(v)!r}"
    return out
