"""The prefill calibration for MiniMax-M3.1 (maya-mm3): how this PC reads prompts fastest, measured on it.

maya-mm3 reads a prompt in chunks of tokens through its layers; two of its settings depend on the PC and are measured
rather than guessed (one engine start, a few minutes):
  - the chunk: more tokens a forward give each routed expert more rows to multiply at once (faster on a GPU that holds
    every expert), but its buffers take GPU memory - a chunk above the default where every expert still fits, or
    where not (each chunk then copies the experts VRAM does not hold: a larger one reads prompts much faster, #552)
    when it reads prompts at least WIDE_GAIN faster and decodes at most DECODE_LOSS slower than a second start with
    the default chunk (whose buffers leave the experts more room) measures;
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
KEYS = ("--chunk", "--prefill-gemm", "--prefill-tune", "--no-prefill-tune", "--calibrate-prefill",
        "--calibrate-decode")   # (the measure's own)
WIDE_GAIN = 0.25     # a larger chunk's prompt speed over the kept one's before its decode is checked (a second start)
DECODE_LOSS = 0.03   # the decode a larger chunk may cost (its buffers take room from the experts)
GEMMS = {"blas": 0, "hipblaslt": 1}


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
        import maya_tokenizer as ST
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


def command(cfg: dict, ids_file: Path | None, out_file: Path, decode_chunk: int = 0) -> list[str]:
    """The engine's calibration (decode_chunk: its decode check instead - a start with that chunk's buffers)."""
    args = without(server().engine_args(cfg))
    cmd = [cfg["exe"]] + args + (["--chunk", str(decode_chunk), "--calibrate-decode", str(out_file)] if decode_chunk
                                 else ["--calibrate-prefill", str(out_file)])
    if ids_file is not None:
        cmd += ["--ids", "@" + str(ids_file)]
    return cmd


def write_tune(file: str, key: str, settings: str) -> None:
    """The engine's prefill-tune.txt: this PC's line (`key<TAB>settings`) replaced, the others kept (Model::write_tune)."""
    p = Path(file)
    keep = [x for x in (p.read_text(encoding="utf-8").splitlines() if p.is_file() else [])
            if x and x.split("\t", 1)[0] != key]
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(x + "\n" for x in keep + [f"{key}\t{settings}"]), encoding="utf-8", newline="\n")
    tmp.replace(p)


def wide_check(cfg: dict, res: dict, ids_file: Path | None, tmp: Path, runner, say=print) -> dict | None:
    """Where not every expert is in VRAM: the larger chunk the engine measured (its buffers took room from the experts,
    and this load decoded at res["decode_tok_s"]) against a start with the kept chunk's buffers - kept for this PC
    when it reads prompts WIDE_GAIN faster and decodes at most DECODE_LOSS slower.  The check's record, or None (no
    larger chunk to check)."""
    wide, wt, dec = res.get("wide_chunk") or 0, res.get("wide_tok_s") or 0.0, res.get("decode_tok_s") or 0.0
    kept, kt = res.get("chunk") or 0, res.get("tok_s") or 0.0
    if not (wide > kept > 0 and kt > 0 and wt >= kt * (1 + WIDE_GAIN) and dec > 0 and res.get("key")
            and res.get("file")):
        return None
    say(f"    chunks of {wide} tokens read prompts {wt / kt:.1f}x as fast here, but leave fewer experts in VRAM: the "
        f"decode with chunks of {kept} measured too (a second start) ...")
    out = tmp / "decode.json"
    rc, lines = runner(command(cfg, ids_file, out, decode_chunk=kept), server().child_env(cfg), cfg.get("cwd"),
                       TIMEOUT_S, say)
    if rc != 0 or not out.is_file():
        why = next((x for x in reversed(lines) if x.strip()), "no output").strip()[:300]
        say(f"    the decode check did not finish ({why}): chunks of {kept} stay")
        return {"chunk": wide, "failed": why}
    d0 = json.loads(out.read_text(encoding="utf-8")).get("decode_tok_s") or 0.0
    rec = {"chunk": wide, "tok_s": wt, "decode_tok_s": dec, "kept_chunk": kept, "kept_decode_tok_s": d0,
           "taken": d0 > 0 and dec >= d0 * (1 - DECODE_LOSS)}
    if rec["taken"]:
        write_tune(res["file"], res["key"], f"chunk={wide} gemm={GEMMS.get(res.get('gemm'), 0)} tok_s={wt:.1f} "
                                            f"default_tok_s={res.get('default_tok_s') or 0:.1f} tokens={res.get('tokens')}")
        say(f"    decode {dec:.1f} tok/s with chunks of {wide}, {d0:.1f} with {kept}: chunks of {wide} kept for this PC")
    else:
        say(f"    decode {dec:.1f} tok/s with chunks of {wide}, {d0:.1f} with {kept}: chunks of {kept} stay (decode "
            "first)")
    return rec


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
        chk = wide_check(cfg, res, ids_file, tmp, runner, say)
        if chk is not None:
            res["wide_check"] = chk
            if chk.get("taken"):
                res["chunk"], res["tok_s"] = chk["chunk"], chk["tok_s"]
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
