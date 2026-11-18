"""tools/state_capture_bench.py - what saving and restoring conversation states costs, per prompt length.

    python tools/state_capture_bench.py --config maya-glm.json --exe build/maya [--exe other/maya]
                                        [--sizes 8192,65536,262144] [--repeats 1] --out results.json

For each prompt length N, one engine process (MAYA_GLM_STATE_CACHE=1) answers three requests in order:

  A1   conversation A's first turn: N tokens of real text, `--output` tokens out
  B1   another conversation (`--other` tokens), which moves A out of the engine's live state
  A2   A's next turn: A1's prompt, A1's answer and `--append` new tokens

After every request the engine's STATE line gives the cache's counters. A request's row is their change while it
ran: publications, publish milliseconds, bytes read from the GPUs, bytes hashed, restores and restore milliseconds,
next to its first-token time, its wall time and the prompt tokens it reused. A2's row shows what a returning turn
costs; its `reused` says whether it resumed A's state.

With two `--exe`, every repeat runs both, in alternating order (an A/B run). Greedy, seed 17, the end-of-text
tokens banned (`--ban-eos`), so each answer has exactly `--output` tokens. The engine gets the config's `env`, then
this run's cache settings (a private state directory, `--ram-gb`, no SSD tier).

`--deterministic` adds the settings docs/SETTINGS.md gives for answers that repeat bit for bit (every expert on the
GPU's kernels, prompt products independent of the expert cache, no background tier moves), and then every answer must
equal the same request's answer in every other run, token for token. Use it to check answers, not to time the
default recipe: it is slower.

Exit 0: every request answered (and, with `--deterministic`, every answer matched). 1: an answer differed.
2: the engine did not start, or a request failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "ds4")]
import strata_tokenizer  # noqa: E402
from parallel_test import Multi  # noqa: E402
from slot_keep_test import text_tokens  # noqa: E402

# docs/SETTINGS.md: the settings that make GLM's greedy answers repeat ("the same tokens after a restart"): every
# expert on the GPU's kernels, prompt products independent of the expert cache, no tier moves or replanning.
DETERMINISTIC = {"MAYA_GLM_CPU_LANE": "0", "MAYA_GLM_PREFILL_STABLE": "1", "MAYA_GLM_PROMOTE": "0",
                 "MAYA_GLM_CPU_RETIME": "0"}
COUNTERS = ("publishes", "publish_ms", "read_bytes", "reused_bytes", "hashed_bytes", "restores", "restore_ms", "admitted",
            "rejected", "evicted", "hits", "restored_tokens")


def state_counters(engine: Multi) -> dict:
    lines = [line for line in engine.other if line.startswith("STATE ")]
    if not lines:
        return {}
    fields = dict(item.split("=", 1) for item in lines[-1].split()[1:] if "=" in item)
    return {k: float(fields[k]) for k in COUNTERS if k in fields} | {
        k: float(fields[k]) for k in ("ram_bytes", "units") if k in fields}


def ask(engine: Multi, prompt: list[int], output: int, args: argparse.Namespace) -> dict:
    with engine.cond:
        engine.rids += 1
        rid = engine.rids
        engine.got[rid] = []
    bias = f" logit_bias={args.ban_eos}" if args.ban_eos else ""
    command = f"GEN {output} rid={rid} temperature=0 seed=17 spec={args.spec}{bias} "
    before = state_counters(engine)
    started = time.monotonic()
    engine.proc.stdin.write((command + ",".join(map(str, prompt)) + "\n").encode())
    engine.proc.stdin.flush()
    result = engine.stream(rid, timeout=args.timeout)
    ended = time.monotonic()
    if result["error"] or not result["times"]:
        raise RuntimeError(f"request {rid} failed: {result['error']}")
    time.sleep(0.2)   # (the STATE line follows DONE)
    after = state_counters(engine)
    return {
        "prompt_tokens": len(prompt),
        "reused": result["reused"],
        "ttft_ms": (result["times"][0] - started) * 1000,
        "wall_ms": (ended - started) * 1000,
        "tokens": result["tokens"],
        "cache": {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS},
        "ram_bytes": after.get("ram_bytes", 0),
    }


def run(exe: str, size: int, args: argparse.Namespace, config: dict, tok) -> dict:
    with tempfile.TemporaryDirectory(prefix="state-bench-", dir=args.scratch) as scratch:
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.update({k: str(v) for k, v in config.get("env", {}).items()})
        cache = {"CACHE": "1", "DIR": str(Path(scratch) / "state"), "RAM_GB": str(args.ram_gb), "DISK_GB": "0",
                 "MIGRATE": "0"}
        env.update({f"MAYA_GLM_STATE_{k}": v for k, v in cache.items()})
        env["MAYA_GLM_SLOT_DIR"] = str(Path(scratch) / "slots")
        if args.deterministic:
            env.update(DETERMINISTIC)
        env.update(dict(item.split("=", 1) for item in args.env))
        command = ["--serve"] + config["args"] + (["--layer-split", "auto"] if "--layer-split" not in config["args"]
                                                  else [])
        engine = Multi(exe, command, env)
        loaded = time.monotonic()
        engine.start(timeout=600)
        ready_s = time.monotonic() - loaded
        try:
            a1 = text_tokens(tok, f"Conversation A {size}", size)
            b1 = text_tokens(tok, f"Conversation B {size}", args.other)
            rows = {"A1": ask(engine, a1, args.output, args)}
            rows["B1"] = ask(engine, b1, args.output, args)
            a2 = a1 + rows["A1"]["tokens"] + text_tokens(tok, f"Tool result {size}", args.append)
            rows["A2"] = ask(engine, a2, args.output, args)
        finally:
            engine.quit(timeout=300)
            log = Path(engine.err_path).read_text(errors="replace").splitlines() if engine.err_path else []
            if engine.err_path:
                os.unlink(engine.err_path)
        return {"exe": exe, "size": size, "ready_s": ready_s, "rows": rows,
                "engine_log": [line for line in log if "state" in line.lower() or "publish" in line.lower()][-200:]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="a Maya config (maya-*.json): its engine args and env")
    p.add_argument("--exe", action="append", required=True, help="the native engine; twice for an A/B run")
    p.add_argument("--sizes", default="8192,65536,262144")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", type=int, default=64)
    p.add_argument("--append", type=int, default=64)
    p.add_argument("--other", type=int, default=1024)
    p.add_argument("--spec", type=int, default=1)
    p.add_argument("--ram-gb", type=float, default=96)
    p.add_argument("--ban-eos", default="154820:-100,154827:-100,154829:-100")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine, after the config's")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--deterministic", action="store_true", help="repeatable answers; every answer must match")
    p.add_argument("--scratch", default=None, help="where each run's private cache directory goes")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    tok = strata_tokenizer.Tokenizer.from_dir(Path(config["tokenizer"]))
    results = []
    for repeat in range(args.repeats):
        exes = args.exe if repeat % 2 == 0 else list(reversed(args.exe))
        for size in (int(s) for s in args.sizes.split(",")):
            for exe in exes:
                try:
                    r = run(exe, size, args, config, tok)
                except Exception as e:   # noqa: BLE001 - (a failed run is reported, never dropped)
                    print(f"FAILED {exe} N={size}: {e}", file=sys.stderr)
                    Path(args.out).write_text(json.dumps(results, indent=1))
                    return 2
                r["repeat"] = repeat
                results.append(r)
                Path(args.out).write_text(json.dumps(results, indent=1))
                for name, row in r["rows"].items():
                    c = row["cache"]
                    print(f"{Path(exe).parent.name:>12} N={size:<7} {name} reused={row['reused']:<7} "
                          f"ttft={row['ttft_ms']:9.1f} ms  publish={c['publish_ms']:8.1f} ms "
                          f"read={c['read_bytes'] / 2**20:9.1f} MiB  reused={c.get('reused_bytes', 0) / 2**20:9.1f} MiB  hashed={c['hashed_bytes'] / 2**20:9.1f} MiB "
                          f"restore={c['restore_ms']:7.1f} ms", flush=True)
    if args.deterministic:
        answers: dict = {}
        for r in results:
            for name, row in r["rows"].items():
                answers.setdefault((r["size"], name), set()).add(tuple(row["tokens"]))
        differing = sorted(key for key, seen in answers.items() if len(seen) > 1)
        for size, name in differing:
            print(f"DIFFERENT N={size} {name}: {len(answers[(size, name)])} distinct answers", flush=True)
        print(f"answers: {len(answers) - len(differing)} of {len(answers)} requests identical in every run", flush=True)
        return 1 if differing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
