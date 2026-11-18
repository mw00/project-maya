"""tools/state_decode_bench.py - does keeping conversation states slow the decode? A few minutes per build.

    python tools/state_decode_bench.py --config maya-glm.json --exe A [--exe B ...] [--rounds 2] --out results.json

Each `--exe` (optionally `PATH:KEY=VALUE,...` for its own environment, e.g. `build/maya:MAYA_GLM_STATE_CACHE=0`) starts
once per round and answers `--requests` fresh prompts of `--prompt` tokens, `--output` tokens each (greedy, the
end-of-text tokens banned), one after another (`--pause` seconds apart, as an agent's tools run between its turns). The first `--warm` requests are left out.
Rounds alternate the builds' order. Per request: the first-token time, the decode rate from the first token to the
last (the engine's own T line times), the median gap between tokens, and the engine's `glm stat` line (RAM fetches and
promotions per token: whether background expert moves lagged).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "ds4")]
import strata_tokenizer  # noqa: E402
from parallel_test import Multi  # noqa: E402
from slot_keep_test import text_tokens  # noqa: E402

STAT = re.compile(r"glm stat: decode ([\d.]+) ms/tok .*?ram fetches ([\d.]+)/tok .*?promotions ([\d.]+)/tok")


def ask(engine: Multi, prompt: list[int], args: argparse.Namespace) -> dict:
    with engine.cond:
        engine.rids += 1
        rid = engine.rids
        engine.got[rid] = []
    spec = f" spec={args.spec}" if args.spec > 0 else ""   # (none: the engine's own drafting, as the server sends)
    command = f"GEN {args.output} rid={rid} temperature=0 seed=17{spec} logit_bias={args.ban_eos} "
    started = time.monotonic()
    engine.proc.stdin.write((command + ",".join(map(str, prompt)) + "\n").encode())
    engine.proc.stdin.flush()
    result = engine.stream(rid, timeout=args.timeout)
    if result["error"] or len(result["times"]) < 2:
        raise RuntimeError(f"request {rid} failed: {result['error']}")
    t = result["times"]
    gaps = [b - a for a, b in zip(t, t[1:])]
    return {"ttft_ms": (t[0] - started) * 1000, "decode_tok_s": (len(t) - 1) / (t[-1] - t[0]),
            "gap_p50_ms": st.median(gaps) * 1000, "tokens": len(result["tokens"])}


def run(spec: str, round_: int, args: argparse.Namespace, config: dict, tok) -> dict:
    exe, _, extra = spec.partition(":")
    with tempfile.TemporaryDirectory(prefix="decode-bench-") as scratch:
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.update({k: str(v) for k, v in config.get("env", {}).items()})
        cache = {"CACHE": "1", "DIR": str(Path(scratch) / "state"), "RAM_GB": "16", "DISK_GB": "0", "MIGRATE": "0"}
        env.update({f"MAYA_GLM_STATE_{k}": v for k, v in cache.items()})
        env["MAYA_GLM_SLOT_DIR"] = str(Path(scratch) / "slots")
        env.update(dict(item.split("=", 1) for item in extra.split(",") if item))
        command = ["--serve"] + config["args"] + (["--layer-split", "auto"] if "--layer-split" not in config["args"]
                                                  else [])
        engine = Multi(exe, command, env)
        engine.start(timeout=600)
        rows = []
        try:
            for i in range(args.requests):
                prompt = text_tokens(tok, f"Round {round_} request {i}", args.prompt)
                if i and args.pause:
                    time.sleep(args.pause)   # (an agent's tools run between its turns)
                rows.append(ask(engine, prompt, args))
        finally:
            engine.quit(timeout=300)
            log = Path(engine.err_path).read_text(errors="replace") if engine.err_path else ""
            if engine.err_path:
                os.unlink(engine.err_path)
        stats = [tuple(map(float, m)) for m in STAT.findall(log)]
        for row, s in zip(rows, stats[-len(rows):] if len(stats) >= len(rows) else []):
            row.update(engine_ms_tok=s[0], ram_fetch=s[1], promo=s[2])
        return {"spec": spec, "round": round_, "rows": rows[args.warm:]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--exe", action="append", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--requests", type=int, default=12)
    p.add_argument("--warm", type=int, default=2)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--output", type=int, default=256)
    p.add_argument("--spec", type=int, default=0, help="GEN spec=N (1: no drafting); 0: the engine's default")
    p.add_argument("--ban-eos", default="154820:-100,154827:-100,154829:-100")
    p.add_argument("--pause", type=float, default=0, help="seconds between requests (0: back to back)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    tok = strata_tokenizer.Tokenizer.from_dir(Path(config["tokenizer"]))
    runs = []
    for r in range(args.rounds):
        for spec in (args.exe if r % 2 == 0 else list(reversed(args.exe))):
            runs.append(run(spec, r, args, config, tok))
            Path(args.out).write_text(json.dumps(runs, indent=1))
            rows = runs[-1]["rows"]
            print(f"round {r} {spec}: ttft {st.mean(x['ttft_ms'] for x in rows):.0f} ms, decode "
                  f"{st.mean(x['decode_tok_s'] for x in rows):.2f} tok/s, gap p50 "
                  f"{st.mean(x['gap_p50_ms'] for x in rows):.2f} ms, ram fetch "
                  f"{st.mean(x.get('ram_fetch', 0) for x in rows):.2f}/tok", flush=True)
    print("\nmean over rounds")
    for spec in args.exe:
        rows = [x for run_ in runs if run_["spec"] == spec for x in run_["rows"]]
        print(f"{spec}: ttft {st.mean(x['ttft_ms'] for x in rows):.0f} ms, decode "
              f"{st.mean(x['decode_tok_s'] for x in rows):.2f} ± {st.stdev(x['decode_tok_s'] for x in rows):.2f} "
              f"tok/s (n={len(rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
