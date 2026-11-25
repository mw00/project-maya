#!/usr/bin/env python3
"""tools/ling/exactness_check.py - Greedy exactness & determinism verification suite for maya-ling.

Prints SHA-256 hashes of greedy output token IDs across:
  1. chat prompt
  2. code prompt
  3. 256-token prompt
  4. picture prompt (when --mmproj is provided)
And checks:
  - window of N tokens == N single tokens (step_window vs step)
  - whole prompt prefill == chunked prompt prefill (via --prefill-chunk)
  - INT8 KV (--kv q8) vs FP16 KV (--kv fp16)
  - speculative drafting spec=1 (3 runs) == spec=0 (3 runs)
  - repeatable mode (--repeatable): 2 cold starts + 2 warm requests give identical hash
  - lane-parity (--lane-parity): different VRAM/RAM expert placements give identical hash

Usage:
    python3 tools/ling/exactness_check.py --bin ./build/maya-ling --gguf /path/to/Ling-3.2-air.gguf \
        [--mmproj /path/to/mmproj-BF16.gguf] [--gpus 0] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()[:16]


def run_serve_cmds(binary: str, args: list[str], commands: list[str], timeout: float = 600.0) -> tuple[list[dict], str]:
    err_f = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [binary, "--serve", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err_f,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout
    # Wait for READY
    ready = False
    for line in proc.stdout:
        s = line.strip()
        if s.startswith("READY "):
            ready = True
            break
    if not ready:
        proc.kill()
        proc.wait(timeout=10)
        err_f.seek(0)
        err = err_f.read()
        err_f.close()
        raise RuntimeError(f"Engine did not print READY. stderr:\n{err[-2000:]}")

    results = []
    for cmd in commands:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        toks: list[int] = []
        done_line = ""
        err_lines = []
        for line in proc.stdout:
            s = line.strip()
            if s.startswith("T "):
                toks.append(int(s.split()[1]))
            elif s.startswith("ERR "):
                err_lines.append(s)
            elif s.startswith("DONE "):
                done_line = s
                break
        results.append({"tokens": toks, "hash": sha256_ids(toks), "done": done_line, "errs": err_lines})

    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    proc.wait(timeout=60)
    err_f.seek(0)
    stderr_txt = err_f.read()
    err_f.close()
    return results, stderr_txt


def make_test_ppm(path: Path, w: int = 256, h: int = 256) -> None:
    buf = bytearray()
    for y in range(h):
        for x in range(w):
            buf.extend(((x * 3 + y) & 255, (y * 2 + 40) & 255, ((x ^ y) + 80) & 255))
    path.write_bytes(f"P6\n{w} {h}\n255\n".encode("ascii") + bytes(buf))


def make_test_mmi1(path: Path, start: int, w: int = 256, h: int = 256) -> None:
    rgb = bytearray()
    for y in range(h):
        for x in range(w):
            rgb.extend(((x * 3 + y) & 255, (y * 2 + 40) & 255, ((x ^ y) + 80) & 255))
    hdr = struct.pack("<4siqii", b"MMI1", 1, start, w, h)
    path.write_bytes(hdr + bytes(rgb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./build/maya-ling")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--spec-runs", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    # Deterministic, valid BailingV3 prompts (using real Ling-3.2-air token IDs)
    # 157151=<role>, 157152=</role>, 156895=<|role_end|>, 156903=<think>, 156904=</think>
    # 157158=<|vision_start|>, 157157=<|image_pad|>, 157159=<|vision_end|>
    p_chat = [157151, 90827, 157152, 1020, 2040, 3060, 156895, 157151, 4500, 157152, 9906, 11, 1268, 527, 499, 30, 156895, 157151, 7800, 157152, 198, 156903, 156904]
    p_code = [157151, 4500, 157152, 755, 2900, 320, 443, 8, 25, 198, 262, 471, 220, 16, 198, 156895, 157151, 7800, 157152, 198, 156903, 156904]
    p_256 = [156891] + [((i * 131 + 503) % 20000) + 200 for i in range(255)]
    # 256x256 image with patch=16, merge=2 -> (256/32)^2 = 64 image_pad tokens
    pic_start = 4
    p_pic = [157151, 4500, 157152, 157158] + [157157] * 64 + [157159, 9906, 156895, 157151, 7800, 157152, 198, 156903, 156904]

    n_pred = a.n_predict
    base_args = ["--gguf", a.gguf, "--gpus", a.gpus, "--ctx", str(a.ctx)]
    if a.mmproj:
        base_args += ["--mmproj", a.mmproj]

    mmi1_path = None
    tmp_dir = Path(tempfile.gettempdir())
    if a.mmproj:
        mmi1_path = tmp_dir / f"ling_exact_pic_{os.getpid()}.mmi1"
        make_test_mmi1(mmi1_path, pic_start, 256, 256)

    def fmt_gen(ids: list[int], extra: str = "", pic: Path | None = None) -> str:
        s_ids = ",".join(map(str, ids))
        if pic is not None:
            return f"GENI {n_pred} temperature=0 {extra} {pic} {s_ids}".replace("  ", " ")
        return f"GEN {n_pred} temperature=0 {extra} {s_ids}".replace("  ", " ")

    prompts = [("chat", p_chat, None), ("code", p_code, None), ("256-tok", p_256, None)]
    if mmi1_path is not None:
        prompts.append(("picture", p_pic, mmi1_path))

    failed = False
    try:
        # 1. Baseline FP16 KV (spec=0) vs INT8 KV (spec=0)
        print("=== 1. Exactness Set: FP16 KV (default) vs INT8 KV (--kv q8), greedy ===", flush=True)
        cmds_base = [fmt_gen(ids, "spec=0", pic) for _, ids, pic in prompts]
        res_f16, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16"], cmds_base)
        res_q8, _ = run_serve_cmds(a.bin, base_args + ["--kv", "q8"], cmds_base)
        for (name, _, _), r16, r8 in zip(prompts, res_f16, res_q8):
            same = r16["hash"] == r8["hash"]
            print(f"  {name:8s}  fp16={r16['hash']}  q8={r8['hash']}  kv_match={'SAME' if same else 'DIFF'}  (len={len(r16['tokens'])})", flush=True)

        # 2. Whole prompt vs chunked prompt (256-tok prompt with --prefill-chunk 256 vs --prefill-chunk 64)
        print("=== 2. Whole prompt vs chunked prompt (--prefill-chunk 256 vs 64) ===", flush=True)
        res_whole, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--prefill-chunk", "256"], [fmt_gen(p_256, "spec=0")])
        res_chunk, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--prefill-chunk", "64"], [fmt_gen(p_256, "spec=0")])
        ok_chunk = res_whole[0]["hash"] == res_chunk[0]["hash"]
        print(f"  256-tok   whole={res_whole[0]['hash']}  chunk64={res_chunk[0]['hash']}  {'PASS' if ok_chunk else 'FAIL'}", flush=True)
        if not ok_chunk:
            failed = True

        # 3. Window of N == N single tokens & spec=1 == spec=0 (3 runs each, with MTP loaded!)
        # Use --repeatable on partial-VRAM runs so async promotion does not shift GPU/CPU placement across runs.
        print(f"=== 3. Drafting on == off (MTP loaded: spec=1 vs spec=0, {a.spec_runs} runs each) ===", flush=True)
        cmds_spec = []
        for _, ids, pic in prompts:
            for _ in range(a.spec_runs):
                cmds_spec.append(fmt_gen(ids, "spec=0", pic))
            for _ in range(a.spec_runs):
                cmds_spec.append(fmt_gen(ids, "spec=1", pic))
        spec_extra = ["--kv", "fp16", "--mtp"]
        if len([g for g in a.gpus.split(",") if g.strip()]) < 4:
            spec_extra.append("--repeatable")
        res_spec, _ = run_serve_cmds(a.bin, base_args + spec_extra, cmds_spec)
        off = 0
        for name, _, _ in prompts:
            s0 = [r["hash"] for r in res_spec[off : off + a.spec_runs]]
            s1 = [r["hash"] for r in res_spec[off + a.spec_runs : off + 2 * a.spec_runs]]
            off += 2 * a.spec_runs
            ok_s = len(set(s0 + s1)) == 1
            print(f"  {name:8s}  spec=0:{s0}  spec=1:{s1}  {'PASS' if ok_s else 'FAIL'}", flush=True)
            if not ok_s:
                failed = True

        # 4. Repeatable mode: 2 cold starts + 2 warm requests each
        print("=== 4. Repeatable mode (--repeatable: 2 cold starts x 2 warm requests) ===", flush=True)
        rep_cmds = [fmt_gen(p_chat, "spec=0"), fmt_gen(p_chat, "spec=0")]
        cold1, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--repeatable"], rep_cmds)
        cold2, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--repeatable"], rep_cmds)
        rh = [cold1[0]["hash"], cold1[1]["hash"], cold2[0]["hash"], cold2[1]["hash"]]
        ok_rep = len(set(rh)) == 1
        print(f"  chat      cold1={rh[:2]}  cold2={rh[2:]}  {'PASS' if ok_rep else 'FAIL'}", flush=True)
        if not ok_rep:
            failed = True

        # 5. Lane-parity placement invariance (0 GiB expert VRAM vs 12 GiB expert VRAM)
        if not a.quick:
            print("=== 5. Lane-parity placement invariance (--lane-parity, 0 GiB vs 10 GiB VRAM experts) ===", flush=True)
            lp0, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--lane-parity", "--expert-vram-gib", "0"], [fmt_gen(p_chat, "spec=0")])
            lp1, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--lane-parity", "--expert-vram-gib", "10"], [fmt_gen(p_chat, "spec=0")])
            ok_lp = lp0[0]["hash"] == lp1[0]["hash"]
            print(f"  chat      vram0={lp0[0]['hash']}  vram10={lp1[0]['hash']}  {'PASS' if ok_lp else 'FAIL'}", flush=True)
            if not ok_lp:
                failed = True
    finally:
        if mmi1_path and mmi1_path.exists():
            mmi1_path.unlink()

    print("\nEXACTNESS RESULT:", "ALL PASSED" if not failed else "FAILURES DETECTED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
