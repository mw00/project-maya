"""tools/ling_quant/prep_calib.py - the calibration and evaluation token sequences of the Ling-3.2-air quants.

tools/maya_quant/prep_calib.py's sources and mix (chat, multilingual chat, reasoning traces, web code, other code,
tool calls; wikitext-2 in the eval set), rendered with Ling-3.2-air's own chat template with thinking on (reasoning in
<think>, tool calls as <tool_call>, tool results as <tool_response>), tokenised and packed into --len-token sequences.
Calibration and evaluation take disjoint samples of every source.

    python prep_calib.py --tok <model dir> --data DIR --out DIR [--n-calib 256] [--n-eval 32] [--len 2048]
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "maya_quant"))
import prep_calib as M  # noqa: E402  (the sources)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
    d = pathlib.Path(a.data)
    rng = random.Random(a.seed)
    T = a.len
    stats = {"ok": 0, "refused": 0}

    def render(s):
        try:
            text = tok.apply_chat_template(s["messages"], tools=s.get("tools"), tokenize=False, enable_thinking=True)
        except Exception:  # noqa: BLE001 - a sample the template refuses is skipped
            stats["refused"] += 1
            return None
        stats["ok"] += 1
        return tok(text, add_special_tokens=False)["input_ids"]

    sources = {
        "chat": list(M.ultrachat(d)),
        "multilingual": list(M.oasst(d)),
        "reasoning": list(M.openthoughts(d)),
        "tools": list(M.glaive(d)),
    }
    code = list(M.magicoder(d)) + list(M.stack(d)) + list(M.commitpack(d))
    sources["web"] = [s for s in code if s.get("web")] + list(M.extra(a.extra))
    sources["code"] = [s for s in code if not s.get("web")]
    sources["multilingual"].sort(key=lambda s: (s.get("lang") == "en", rng.random()))
    share = {"chat": 0.20, "multilingual": 0.12, "reasoning": 0.20, "web": 0.25, "code": 0.10, "tools": 0.13}
    for k, v in sources.items():
        if k != "multilingual":
            rng.shuffle(v)
        print(f"{k:12s} {len(v):7d} samples")

    def pack(n_seq, offsets):
        seqs, used = [], {}
        for k, frac in share.items():
            want = max(1, round(n_seq * frac))
            buf, i = [], offsets.get(k, 0)
            pool = sources[k]
            while want > 0 and i < len(pool):
                ids = render(pool[i])
                i += 1
                if not ids:
                    continue
                buf.extend(ids)
                while len(buf) >= T and want > 0:
                    seqs.append(buf[:T])
                    buf = buf[T:]
                    want -= 1
            used[k] = i
        return seqs[:n_seq], used

    ev, used = pack(a.n_eval - 4, {})
    wt = pq.read_table(d / "Salesforce/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    wids = tok("".join(wt), add_special_tokens=False)["input_ids"]
    ev += [wids[i * T:(i + 1) * T] for i in range(4)]
    cal, _ = pack(a.n_calib, {k: v + 1 for k, v in used.items()})   # disjoint from the eval samples, per source
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "calib.npy", np.array(cal, dtype=np.int32))
    np.save(out / "eval.npy", np.array(ev, dtype=np.int32))
    print(f"calib {len(cal)} x {T}, eval {len(ev)} x {T} -> {out}  (rendered {stats['ok']}, refused {stats['refused']})")


if __name__ == "__main__":
    main()
