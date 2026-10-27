"""The prefill calibration for MiniMax-M3.1 (maya-mm3): how this PC reads prompts fastest, measured on it.

maya-mm3 reads a prompt in chunks of tokens through its layers; two of its settings depend on the PC and are measured
rather than guessed (one engine start, a few minutes):
  - the chunk: more tokens a forward give each routed expert more rows to multiply at once (faster on a GPU that holds
    every expert), but its buffers take GPU memory - a chunk above the default only where every expert still fits;
  - the dense GEMMs' backend: the BLAS (cuBLAS / hipBLAS) or, on AMD, hipBLASLt with each matrix shape's fastest kernel
    timed on this GPU (its first product of a shape; the picks kept beside the model, lt-picks.txt).
The engine measures them itself (maya-mm3 --calibrate-prefill) over a real text's tokens (this repository's
CHANGELOG.md as one chat message, ~4K tokens, the config's tokenizer and chat template), each setting twice, and keeps
the fastest for this PC in the pack's prefill-tune.txt - a line keyed by the GPUs, their architecture, the GPU runtime
and the driver, which every later start reads (a new GPU or driver: the defaults until it is measured again).  The
config's own settings (context, KV cache, pictures, drafter) stay as they are, so the memory measured is the one the
model runs with.

    python tools/calibrate_mm3_prefill.py maya-mm3.json     # measure and print; ./maya.sh --calibrate does it too
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

TOKENS = 4096        # the prompt the settings are measured on
TIMEOUT_S = 3600     # the whole calibration, the model's load included
KEYS = ("--chunk", "--prefill-gemm", "--prefill-tune", "--no-prefill-tune", "--calibrate-prefill")   # (the measure's own)


def without(args: list, keys=KEYS) -> list:
    """args without the flags the calibration sets (each with its value; --no-prefill-tune alone)."""
    out, i = [], 0
    while i < len(args):
        if args[i] in keys:
            i += 1 if args[i] == "--no-prefill-tune" else 2
            continue
        out.append(args[i])
        i += 1
    return out


def prompt_ids(cfg: dict, n: int = TOKENS) -> list[int]:
    """CHANGELOG.md as one user message through the config's tokenizer and chat template (calibrate_ds4_tp's long
    prompt), about n tokens; [] when the tokenizer is not there (the engine then measures on its own token mix)."""
    try:
        import calibrate_ds4_tp as C4
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        tpath = Path(cfg["tokenizer"])
        tok = ST.Tokenizer.from_dir(tpath)
        tpl = ChatTemplate(tpath / "chat_template.jinja")
        return C4.long_prompt_ids(tok, tpl, n)
    except Exception:  # noqa: BLE001 - a config without its tokenizer: the engine's own prompt
        return []


def server():
    """serve/server.py (the engine's arguments and environment as the server starts it)."""
    import serve.server as srv
    return srv


def command(cfg: dict, ids_file: Path | None, out_file: Path) -> list[str]:
    args = without(server().engine_args(cfg))
    cmd = [cfg["exe"]] + args + ["--calibrate-prefill", str(out_file)]
    if ids_file is not None:
        cmd += ["--ids", "@" + str(ids_file)]
    return cmd


def run(cfg: dict, say=print, runner=None) -> dict:
    """Measure on the engine `cfg` describes: the engine's JSON ({"chunk", "gemm", "tok_s", "default_tok_s", "trials",
    ...}).  runner(cmd, env, cwd, timeout) -> (returncode, stderr lines) runs the engine (tests: a stand-in).  Raises
    when the engine did not finish."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="maya-prefill-") as tmp:
        tmp = Path(tmp)
        ids = prompt_ids(cfg)
        ids_file = None
        if ids:
            ids_file = tmp / "prompt.txt"
            ids_file.write_text(",".join(str(i) for i in ids), encoding="ascii")
        out = tmp / "prefill.json"
        cmd = command(cfg, ids_file, out)
        if runner is None:
            runner = run_engine
        rc, lines = runner(cmd, server().child_env(cfg), cfg.get("cwd"), TIMEOUT_S, say)
        if rc != 0 or not out.is_file():
            why = next((x for x in reversed(lines) if x.strip()), "no output")
            raise RuntimeError(f"the engine stopped (exit {rc}): {why.strip()[:300]}")
        res = json.loads(out.read_text(encoding="utf-8"))
    res["seconds_all"] = round(time.time() - t0)
    return res


def run_engine(cmd: list, env: dict, cwd, timeout: float, say=print) -> tuple[int, list[str]]:
    """The engine's calibration run, its "calibration:" lines shown as they come."""
    lines: list[str] = []
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                         errors="replace")
    t0 = time.time()
    try:
        for line in p.stderr:
            lines.append(line)
            if line.startswith("calibration:"):
                say("    " + line.rstrip()[len("calibration:"):].strip())
            if time.time() - t0 > timeout:
                p.kill()
                break
        rc = p.wait(timeout=max(1.0, timeout - (time.time() - t0)))
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -9
    return rc, lines[-40:]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_mm3_prefill.py <maya-mm3.json>")
    import maya_names
    c = maya_names.env_in(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    print(json.dumps(run(c), indent=1))
