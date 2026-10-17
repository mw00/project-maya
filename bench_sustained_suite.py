#!/usr/bin/env python3
"""
Sustained Decode Benchmark Suite (4K, 8K, 12K)
Usage:
  python3 bench_sustained_suite.py --target 4k
  python3 bench_sustained_suite.py --target 8k
  python3 bench_sustained_suite.py --target 12k
  python3 bench_sustained_suite.py --target all
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/svniak/project-maya")
sys.path.insert(0, str(ROOT))

import serve.server as SV
import strata_tokenizer as ST

BENCHMARKS = {
    "4k": {
        "prompt_len": 124000,
        "decode_len": 4096,
        "max_context": 131072,
        "name": "4K Decode @ 128K Context",
    },
    "8k": {
        "prompt_len": 240000,
        "decode_len": 8192,
        "max_context": 262144,
        "name": "8K Decode @ 256K Context",
    },
    "12k": {
        "prompt_len": 240000,
        "decode_len": 12288,
        "max_context": 262144,
        "name": "12K Decode @ 256K Context",
    },
}

def parse_telemetry(log_file):
    stats = {}
    if not log_file.exists():
        return stats
    content = log_file.read_text(encoding="utf-8", errors="replace")
    
    # Extract last decode stat line
    stat_matches = re.findall(r"glm stat: decode ([\d\.]+) ms/tok \(([\d\.]+) tok/s\) \| vram hit ([\d\.]+)% \| ram fetches ([\d\.]+)/tok.*?cpu lane ([\d\.]+)/tok in ([\d\.]+) ms/tok", content)
    if stat_matches:
        last = stat_matches[-1]
        stats["decode_ms_per_tok"] = float(last[0])
        stats["decode_tps"] = float(last[1])
        stats["vram_hit_pct"] = float(last[2])
        stats["ram_fetches_per_tok"] = float(last[3])
        stats["cpu_lane_experts_per_tok"] = float(last[4])
        stats["cpu_lane_ms_per_tok"] = float(last[5])

    # Extract speculative decoding
    spec_matches = re.findall(r"glm spec: (\d+) of (\d+) drafts accepted \(([\d\.]+)%\)", content)
    if spec_matches:
        last = spec_matches[-1]
        stats["spec_accepted"] = int(last[0])
        stats["spec_total"] = int(last[1])
        stats["spec_acceptance_pct"] = float(last[2])

    # Extract KV streaming
    kv_matches = re.findall(r"glm kv: KV streaming: ([\d\.]+)% of (\d+) block reads hit VRAM, ([\d\.]+) MiB read from RAM", content)
    if kv_matches:
        last = kv_matches[-1]
        stats["kv_vram_hit_pct"] = float(last[0])
        stats["kv_total_block_reads"] = int(last[1])
        stats["kv_ram_read_mib"] = float(last[2])

    # Extract KDA graphs
    graph_replays = re.findall(r"glm KDA graphs: \d+ captures, (\d+) replays, (\d+) invalidations", content)
    if graph_replays:
        total_replays = sum(int(m[0]) for m in graph_replays)
        total_invals = sum(int(m[1]) for m in graph_replays)
        stats["kda_graph_replays"] = total_replays
        stats["kda_graph_invalidations"] = total_invals

    return stats

def run_single_benchmark(target_key, config_template, doc_tokens, tok):
    spec = BENCHMARKS[target_key]
    prompt_len = spec["prompt_len"]
    decode_len = spec["decode_len"]
    max_context = spec["max_context"]
    name = spec["name"]

    print(f"\n==========================================================================")
    print(f"BENCHMARK START: {name}")
    print(f"Target: {prompt_len:,} prompt tokens + {decode_len:,} decode tokens")
    print(f"Max Context Limit: {max_context:,} (Envelope utilization: {(prompt_len + decode_len) / max_context * 100:.1f}%)")
    print(f"==========================================================================")
    sys.stdout.flush()

    # Prepare config with appropriate max-context
    cfg = json.loads(json.dumps(config_template))
    args = cfg.get("args") or []
    if "--max-context" in args[:-1]:
        args[args.index("--max-context") + 1] = str(max_context)
    else:
        args.extend(["--max-context", str(max_context)])
    cfg["args"] = args

    log_path = ROOT / f"scratch-bench-{target_key}.log"
    cmd = [cfg["exe"], "--serve"] + SV.engine_args(cfg)
    child_env = SV.child_env(cfg)

    print(f"Launching engine with max-context={max_context} ...")
    t_start = time.time()
    result = {
        "key": target_key,
        "name": name,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "max_context": max_context,
    }

    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=log, text=True, env=child_env, cwd=str(ROOT), bufsize=1)

        print("Waiting for engine READY ...")
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"Engine failed to start. See log: {log_path}")
            if line.startswith("READY"):
                print("Engine READY:", line.strip())
                break

        # Warmup
        chat = lambda q: tok.encode("[gMASK]<sop><|user|>\n" + q + "<|assistant|>\n</think>", parse_special=True)
        print("Warmup 64 tokens ...")
        p.stdin.write("GEN 64 temperature=0 " + ",".join(map(str, chat("Hello, please confirm ready."))) + "\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line or line.startswith("DONE") or line.startswith("ERR"):
                break
        print("Warmup complete.")

        # Construct prompt
        instruct = tok.encode("\n\nWrite a comprehensive, exhaustive, multi-chapter textbook covering computer architecture, compiler optimization, CUDA GPU programming, and operating systems in full technical detail.\n<|assistant|>\n</think>")
        tokens = doc_tokens[:prompt_len]
        tokens = tokens[:prompt_len - len(instruct)] + instruct
        actual_prompt_len = len(tokens)

        # Run benchmark request
        # Using temperature=0.7 top_p=0.95 penalty_repeat=1.05 logit_bias to avoid premature EOS
        print(f"Submitting GEN {decode_len} request ({actual_prompt_len:,} prompt tokens) ...")
        sys.stdout.flush()
        p.stdin.write(f"GEN {decode_len} temperature=0.7 top_p=0.95 penalty_repeat=1.05 logit_bias=154820:-100,154827:-100,154829:-100 " + ",".join(map(str, tokens)) + "\n")
        p.stdin.flush()

        tokens_generated = 0
        t_decode_start = None
        last_report = time.time()
        chunk_report = 0

        while True:
            line = p.stdout.readline()
            if not line:
                print(f"Engine stream closed. Check log: {log_path}")
                break
            line = line.strip()

            if line.startswith("PP "):
                parts = line.split()
                chunk_report += 1
                pos = int(parts[1]) if len(parts) > 1 else 0
                tot = int(parts[2]) if len(parts) > 2 else actual_prompt_len
                cur_tps = float(parts[4]) if len(parts) > 4 else 0.0
                print(f"  [Prefill chunk {chunk_report:2d}] pos {pos:7,d} / {tot:7,d} ({cur_tps:6.1f} tok/s)")
                sys.stdout.flush()

            elif line.startswith("T "):
                if t_decode_start is None:
                    t_decode_start = time.time()
                tokens_generated += 1
                now = time.time()
                if tokens_generated % 256 == 0 or (now - last_report > 15.0) or tokens_generated == decode_len:
                    elapsed_dec = now - t_decode_start
                    rate = tokens_generated / elapsed_dec if elapsed_dec > 0 else 0
                    print(f"  [Decode progress] {tokens_generated:5,d} / {decode_len:5,d} tokens ({rate:5.2f} tok/s, elapsed {elapsed_dec:.1f}s)")
                    sys.stdout.flush()
                    last_report = now

            elif line.startswith("DONE"):
                parts = line.split()
                n_gen = int(parts[1]) if len(parts) > 1 else tokens_generated
                p_toks = int(parts[2]) if len(parts) > 2 else actual_prompt_len
                p_ms = float(parts[3]) if len(parts) > 3 else 0.0
                d_ms = float(parts[4]) if len(parts) > 4 else 0.0
                p_tps = (p_toks / p_ms * 1000.0) if p_ms > 0 else 0.0
                d_tps = (n_gen / d_ms * 1000.0) if d_ms > 0 else 0.0
                print(f"\n---> DONE: Generated {n_gen:,} tokens from {p_toks:,} prompt tokens")
                print(f"     Prefill: {p_ms:,.1f} ms ({p_tps:,.1f} tok/s)")
                print(f"     Decode:  {d_ms:,.1f} ms ({d_tps:.2f} tok/s)")
                print(f"     Wall time: {time.time() - t_start:.1f} s")
                result.update({
                    "tokens_generated": n_gen,
                    "prompt_tokens": p_toks,
                    "prefill_ms": p_ms,
                    "prefill_tps": p_tps,
                    "decode_ms": d_ms,
                    "decode_tps": d_tps,
                    "wall_time_s": time.time() - t_start,
                })
                break

            elif line.startswith("ERR"):
                print(f"  ERROR from engine: {line}")
                break

        print("Stopping engine cleanly ...")
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=30)
        except Exception:
            p.kill()

    # Parse runtime telemetry from log
    telemetry = parse_telemetry(log_path)
    result["telemetry"] = telemetry
    print(f"\nTelemetry Extracted for {name}:")
    for k, v in telemetry.items():
        print(f"  {k}: {v}")
    print(f"==========================================================================\n")
    sys.stdout.flush()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["4k", "8k", "12k", "all"], default="4k")
    args = parser.parse_args()

    cfg_path = ROOT / "maya-maya-s-v2-iq2_xxs.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    cmd_args = cfg.get("args") or []
    pack = Path(cmd_args[cmd_args.index("--glm-pack") + 1]) if "--glm-pack" in cmd_args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    tok = ST.Tokenizer.from_dir(tp)

    # Prepare document corpus
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) +
                       sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
    doc_base = tok.encode(text)
    doc_tokens = (doc_base * 5)
    print(f"Token corpus loaded: {len(doc_tokens):,} tokens ready.")

    targets = ["4k", "8k", "12k"] if args.target == "all" else [args.target]
    all_results = {}
    results_path = ROOT / "sustained_bench_results.json"

    if results_path.exists():
        try:
            all_results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            all_results = {}

    for t in targets:
        res = run_single_benchmark(t, cfg, doc_tokens, tok)
        all_results[t] = res
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        time.sleep(3)  # brief pause for GPU memory reclamation

    print("\nALL REQUESTED BENCHMARKS FINISHED.")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
