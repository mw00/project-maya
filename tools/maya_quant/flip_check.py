"""tools/maya_quant/flip_check.py - does a GLM engine give a prompt the same answer whatever came before?  The
regression check for two v1.1.58 bugs: the batched prompt path's numbers followed the expert cache (the same prompt
read again after other requests got another greedy answer), and so a request with refusal ablation switched off
after one with it on did not get the fresh "off" answer.

Three `--serve` engines with the same arguments plus `--ablate DIR --ablate-default off`:

    fresh off   GEN P, LOGP L                     -> the references for "off"
    fresh on    GEN ablate=1 P, GEN ablate=1 Q    -> the references for "on"
    sequence    on P, off P, on P, off P, on Q, off Q, LOGP L (the engine's own reuse on, as a server runs)

Every answer of the sequence must equal its fresh reference token for token (and LOGP to the last digit).  Greedy;
exactness flags (lane off etc.) go in the environment.  Exit 0 = all equal, 1 = a difference, 2 = an engine failed.

    python flip_check.py --exe build/maya --direction DIR.gguf --prompt P.ids --short Q.ids [--max-new 64]
                         -- --glm-pack PACK [engine args]
(an ids file: comma-separated token ids, as --tokens-file takes; L = the first --logp-ctx + 256 tokens of P)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys


def run_engine(cmd: list[str], lines: list[str]) -> list[str]:
    """One engine, one line at a time; each answer as 'md5 of its tokens | its DONE / LP line'."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         bufsize=1)
    out = []
    for line in lines:
        p.stdin.write(line + "\n")
        p.stdin.flush()
        toks = []
        while True:
            row = p.stdout.readline()
            if not row:
                p.kill()
                raise RuntimeError("the engine exited")
            f = row.split()
            if f and f[0] == "T":
                toks.append(f[-1])
            elif f and f[0] in ("DONE", "LP", "ERR"):
                if f[0] == "ERR":
                    raise RuntimeError(row.strip())
                key = row.split()[1] if f[0] == "LP" else hashlib.md5(" ".join(toks).encode()).hexdigest()[:12]
                out.append(f"{key} | {row.strip()[:80]}")
                break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--direction", required=True, help="a refusal direction (any unit vector works for exactness)")
    ap.add_argument("--scale", default="4")
    ap.add_argument("--prompt", required=True, help="ids file of a prompt long enough for the batched prompt path")
    ap.add_argument("--short", required=True, help="ids file of a short prompt")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--logp-ctx", type=int, default=512)
    ap.add_argument("engine", nargs=argparse.REMAINDER, help="-- then the engine's model arguments")
    a = ap.parse_args()
    eng = [x for x in a.engine if x != "--"]
    P = open(a.prompt).read().strip()
    Q = open(a.short).read().strip()
    L = ",".join(P.split(",")[: a.logp_ctx + 256])
    base = [a.exe, *eng, "--serve", "--ablate", a.direction, "--ablate-scale", a.scale, "--ablate-default", "off"]
    g = f"GEN {a.max_new} temperature=0"
    try:
        off = run_engine(base, [f"{g} {P}", f"LOGP {a.logp_ctx} {L}", f"{g} {Q}"])
        on = run_engine(base, [f"{g} ablate=1 {P}", f"{g} ablate=1 {Q}"])
        seq = run_engine(base, [f"{g} ablate=1 {P}", f"{g} {P}", f"{g} ablate=1 {P}", f"{g} {P}",
                                f"{g} ablate=1 {Q}", f"{g} {Q}", f"LOGP {a.logp_ctx} {L}"])
    except RuntimeError as e:
        print(f"flip check: {e}")
        return 2
    want = [on[0], off[0], on[0], off[0], on[1], off[2], off[1]]
    names = ["on P", "off P", "on P", "off P", "on Q", "off Q", "LOGP"]
    bad = 0
    for n, got, ref in zip(names, seq, want):
        same = got.split(" |")[0] == ref.split(" |")[0]
        bad += not same
        print(f"{n:6s} {'same' if same else 'DIFFERS'}: {got}" + ("" if same else f"  (fresh: {ref})"))
    if on[0].split(" |")[0] == off[0].split(" |")[0]:
        print("note: on and off gave the same answer to P - the direction changed nothing (a stronger --scale?)")
    print(f"flip check: {'PASS' if bad == 0 else f'FAIL ({bad} of {len(want)} differ)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
