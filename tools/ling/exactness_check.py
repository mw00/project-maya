#!/usr/bin/env python3
"""tools/ling/exactness_check.py - Exactness, determinism, sampled-drafting chi-square, and --no-promo
verification suite for maya-ling.

Prints SHA-256 hashes of greedy output token IDs across:
  1. chat prompt
  2. code prompt
  3. 256-token prompt
  4. picture prompt (when --mmproj is provided)
And checks:
  - whole prompt prefill == chunked prompt prefill (via --prefill-chunk)
  - INT8 KV (--kv q8) vs FP16 KV (--kv fp16)
  - speculative drafting spec=1 (3 runs) == spec=0 (3 runs)
  - repeatable mode (--repeatable): 2 cold starts + 2 warm requests give identical hash
  - repeatable stress (--repeat-n N, default 16): N identical chat requests in ONE --repeatable --mtp process,
    drafting off and on alternated, all give one hash (catches stream races that 2 requests miss)
  - lane-parity (--lane-parity): identical greedy hash across any VRAM vs CPU expert split on the same GPU set
    (same answers for 1 vs N GPUs hold with --repeatable only: a normal run can differ in the last digit)
  - --sampled-check: Pearson chi-square test of sampled spec=0 vs spec=1 draws
  - --no-promo-check: verifies --no-promo disables promotions (promo=0.00) and isolates cold profile hit lift

Usage:
    python3 tools/ling/exactness_check.py --bin ./build/maya-ling --gguf /path/to/Ling-3.2-air.gguf \
        [--mmproj /path/to/mmproj-BF16.gguf] [--gpus 0] [--quick] [--sampled-check] [--no-promo-check]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()[:16]


def _chi2_two_sample(c0: dict[int, int], c1: dict[int, int]) -> tuple[float, int, float]:
    """Two-sample Pearson chi-square test over discrete token counts -> (chi2, df, approx_p)."""
    keys = sorted(set(c0) | set(c1))
    n0 = sum(c0.values())
    n1 = sum(c1.values())
    ntot = n0 + n1
    if len(keys) <= 1 or n0 == 0 or n1 == 0:
        return 0.0, 0, 1.0
    chi2 = 0.0
    for k in keys:
        o0 = c0.get(k, 0)
        o1 = c1.get(k, 0)
        tot = o0 + o1
        e0 = tot * (n0 / ntot)
        e1 = tot * (n1 / ntot)
        if e0 > 0:
            chi2 += (o0 - e0) ** 2 / e0
        if e1 > 0:
            chi2 += (o1 - e1) ** 2 / e1
    df = len(keys) - 1
    if df <= 0:
        return chi2, 0, 1.0
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return chi2, df, p


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
        stats: list[str] = []
        done_line = ""
        err_lines = []
        for line in proc.stdout:
            s = line.strip()
            if s.startswith("T "):
                toks.append(int(s.split()[1]))
            elif s.startswith("STAT "):
                stats.append(s)
            elif s.startswith("ERR "):
                err_lines.append(s)
            elif s.startswith("DONE "):
                done_line = s
                break
        results.append({"tokens": toks, "hash": sha256_ids(toks), "done": done_line, "stats": stats, "errs": err_lines})

    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    proc.wait(timeout=60)
    err_f.seek(0)
    stderr_txt = err_f.read()
    err_f.close()
    return results, stderr_txt


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
    ap.add_argument("--repeat-n", type=int, default=16,
                    help="identical requests in one --repeatable process (drafting off/on alternated); 0 skips")
    ap.add_argument("--sampled-check", action="store_true", help="Run sampled drafting chi-square check (spec=0 vs spec=1)")
    ap.add_argument("--sampled-draws", type=int, default=200)
    ap.add_argument("--no-promo-check", action="store_true", help="Verify --no-promo disables promotions and isolates profile lift")
    a = ap.parse_args()

    p_chat = [157151, 90827, 157152, 1020, 2040, 3060, 156895, 157151, 4500, 157152, 9906, 11, 1268, 527, 499, 30, 156895, 157151, 7800, 157152, 198, 156903, 156904]
    p_code = [157151, 4500, 157152, 755, 2900, 320, 443, 8, 25, 198, 262, 471, 220, 16, 198, 156895, 157151, 7800, 157152, 198, 156903, 156904]
    p_256 = [156891] + [((i * 131 + 503) % 20000) + 200 for i in range(255)]
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

    def fmt_gen(ids: list[int], extra: str = "", pic: Path | None = None, n_tok: int | None = None) -> str:
        s_ids = ",".join(map(str, ids))
        k = n_pred if n_tok is None else n_tok
        if pic is not None:
            return f"GENI {k} temperature=0 {extra} {pic} {s_ids}".replace("  ", " ")
        return f"GEN {k} temperature=0 {extra} {s_ids}".replace("  ", " ")

    prompts = [("chat", p_chat, None), ("code", p_code, None), ("256-tok", p_256, None)]
    if mmi1_path is not None:
        prompts.append(("picture", p_pic, mmi1_path))

    failed = False
    try:
        # 1. Baseline FP16 KV (default) vs INT8 KV (spec=0)
        print("=== 1. Exactness Set: FP16 KV (default) vs INT8 KV (--kv q8), greedy ===", flush=True)
        cmds_base = [fmt_gen(ids, "spec=0", pic) for _, ids, pic in prompts]
        res_f16, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16"], cmds_base)
        res_q8, _ = run_serve_cmds(a.bin, base_args + ["--kv", "q8"], cmds_base)
        for (name, _, _), r16, r8 in zip(prompts, res_f16, res_q8):
            same = r16["hash"] == r8["hash"]
            print(f"  {name:8s}  fp16={r16['hash']}  q8={r8['hash']}  kv_match={'SAME' if same else 'DIFF'}  (len={len(r16['tokens'])})", flush=True)

        # 2. Whole prompt vs chunked prompt
        print("=== 2. Whole prompt vs chunked prompt (--prefill-chunk 256 vs 64) ===", flush=True)
        res_whole, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--prefill-chunk", "256"], [fmt_gen(p_256, "spec=0")])
        res_chunk, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--prefill-chunk", "64"], [fmt_gen(p_256, "spec=0")])
        ok_chunk = res_whole[0]["hash"] == res_chunk[0]["hash"]
        print(f"  256-tok   whole={res_whole[0]['hash']}  chunk64={res_chunk[0]['hash']}  {'PASS' if ok_chunk else 'FAIL'}", flush=True)
        if not ok_chunk:
            failed = True

        # 3. Drafting on == off (MTP loaded: spec=1 vs spec=0)
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

        # 4. Repeatable mode: 2 cold starts x 2 warm requests
        print("=== 4. Repeatable mode (--repeatable: 2 cold starts x 2 warm requests) ===", flush=True)
        rep_cmds = [fmt_gen(p_chat, "spec=0"), fmt_gen(p_chat, "spec=0")]
        cold1, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--repeatable"], rep_cmds)
        cold2, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--repeatable"], rep_cmds)
        rh = [cold1[0]["hash"], cold1[1]["hash"], cold2[0]["hash"], cold2[1]["hash"]]
        ok_rep = len(set(rh)) == 1
        print(f"  chat      cold1={rh[:2]}  cold2={rh[2:]}  {'PASS' if ok_rep else 'FAIL'}", flush=True)
        if not ok_rep:
            failed = True

        # 4b. Repeatable stress: many identical requests in one process, drafting off/on alternated
        if a.repeat_n > 0:
            print(f"=== 4b. Repeatable stress ({a.repeat_n} identical chat requests, one --repeatable --mtp process) ===", flush=True)
            st_cmds = [fmt_gen(p_chat, "spec=1" if i % 2 else "spec=0") for i in range(a.repeat_n)]
            st, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--mtp", "--repeatable"], st_cmds, 900.0)
            groups: dict[str, list[int]] = {}
            for i, r in enumerate(st):
                groups.setdefault(r["hash"], []).append(i)
            ok_st = len(st) == a.repeat_n and len(groups) == 1
            desc = "; ".join(f"{h} x{len(v)} at {v[:6]}" for h, v in groups.items())
            print(f"  chat      {len(groups)} distinct: {desc}  {'PASS' if ok_st else 'FAIL'}", flush=True)
            if not ok_st:
                failed = True

        # 5. Lane-parity placement invariance (0 GiB vs 10 GiB VRAM experts, same GPU set)
        if not a.quick:
            print("=== 5. Lane-parity VRAM/CPU split invariance (--lane-parity, 0 GiB vs 10 GiB VRAM experts) ===", flush=True)
            lp0, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--lane-parity", "--expert-vram-gib", "0"], [fmt_gen(p_chat, "spec=0")])
            lp1, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--lane-parity", "--expert-vram-gib", "10"], [fmt_gen(p_chat, "spec=0")])
            ok_lp = lp0[0]["hash"] == lp1[0]["hash"]
            print(f"  chat      vram0={lp0[0]['hash']}  vram10={lp1[0]['hash']}  {'PASS' if ok_lp else 'FAIL'}", flush=True)
            if not ok_lp:
                failed = True

        # 6. Optional sampled drafting chi-square check (spec=0 vs spec=1 at temperature=1.0)
        if a.sampled_check:
            nd = a.sampled_draws
            print(f"=== 6. Sampled drafting chi-square check (spec=0 vs spec=1, {nd} draws each, T=1.0 top_p=0.95 top_k=20) ===", flush=True)
            s_ids = ",".join(map(str, p_chat))
            cmds_samp = [f"GEN 4 temperature=1.0 top_p=0.95 top_k=20 seed={1000 + i} spec=0 {s_ids}" for i in range(nd)]
            cmds_samp += [f"GEN 4 temperature=1.0 top_p=0.95 top_k=20 seed={2000 + i} spec=1 {s_ids}" for i in range(nd)]
            res_samp, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--mtp", "--repeatable"], cmds_samp)
            c0 = collections.Counter(r["tokens"][1] for r in res_samp[:nd] if len(r["tokens"]) >= 2)
            c1 = collections.Counter(r["tokens"][1] for r in res_samp[nd:] if len(r["tokens"]) >= 2)
            chi2, df, pval = _chi2_two_sample(c0, c1)
            ok_chi = pval > 0.01
            print(f"  Sampled drafting chi-square ({nd} draws x 2): n0={sum(c0.values())} n1={sum(c1.values())} "
                  f"bins={len(set(c0)|set(c1))} df={df} chi2={chi2:.3f} (chi2/df={chi2/max(1,df):.2f}, p≈{pval:.3f}) "
                  f"{'PASS' if ok_chi else 'FAIL'}", flush=True)
            if not ok_chi:
                failed = True

        # 7. Optional --no-promo check (verifies promo=0.00 and isolates cold profile hit lift)
        if a.no_promo_check:
            print("=== 7. --no-promo verification (promo=0.00 & isolated cold-start profile hit lift) ===", flush=True)
            cmd_np = [fmt_gen(p_chat, "spec=0", n_tok=16)]
            r_prof, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--no-promo"], cmd_np)
            r_noprof, _ = run_serve_cmds(a.bin, base_args + ["--kv", "fp16", "--no-promo", "--no-profile"], cmd_np)
            def _parse_stat(res):
                st = {}
                if res[0]["stats"]:
                    for tok in res[0]["stats"][-1].split()[1:]:
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            st[k] = v
                return st
            sp_prof = _parse_stat(r_prof)
            sp_noprof = _parse_stat(r_noprof)
            promo_zero = (sp_prof.get("promo") == "0.00" and sp_noprof.get("promo") == "0.00")
            print(f"  with_profile --no-promo : vram_hit={sp_prof.get('vram_hit')} promo={sp_prof.get('promo')} tok_s={sp_prof.get('tok_s')}", flush=True)
            print(f"  no_profile   --no-promo : vram_hit={sp_noprof.get('vram_hit')} promo={sp_noprof.get('promo')} tok_s={sp_noprof.get('tok_s')}", flush=True)
            print(f"  --no-promo promo==0.00  : {'PASS' if promo_zero else 'FAIL'}", flush=True)
            if not promo_zero:
                failed = True
    finally:
        if mmi1_path and mmi1_path.exists():
            mmi1_path.unlink()

    print("\nEXACTNESS RESULT:", "ALL PASSED" if not failed else "FAILURES DETECTED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
