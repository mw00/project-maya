#!/usr/bin/env python3
"""tools/ds4/slot_image_test.py - pictures in the kept conversations (the slots' DS4SLOT2 / DS4META3 format): a chat
with a picture in its history is set aside and taken back - for a prompt whose picture sits at the same place with the
same pixels, and for no other.

    python tools/ds4/slot_image_test.py --pack /mnt/data/models/.../maya-ds4 [--exe build/maya-ds4] [--gpus 0,1]
                                        [--config maya-ds4.json]

Two runs of the engine's `--serve` protocol (greedy, spec=0: the state's position is the answer's, exactly):
  no-keep   A (a picture), B, A's next turn         the reference: every turn starts over (reused 0), and its answers
                                                    are what the keep run's must be
  keep      A (a picture), B, A's next turn         the third request takes A's slot back whole: the picture's span is
            B again, A's turn plus a new picture     in the KV, nothing of the span is read again; the same at a
                                                    checkpoint, a picture added after the slot's end
            B again, A with another picture           another picture at the same place: no slot is taken (the tokens
                                                    of a span are all the image token, the pixels are not)

The picture: a synthetic RGB field of --pic-size x --pic-size pixels (the span is 2 + n_h * (n_w + 1) rows, n_h, n_w the
patch grid merged by 3), the prompt: real text (this repo's tokenizer corpus) around it, sent as GENI with a DSI1 file.

Exit 0: every check passed.  1: a check failed.  2: the engine or the tokenizer is not there.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slot_keep_test as SK  # noqa: E402  (the engine wrapper, the prompts, the .meta reader)

PATCH = 14   # vision.patch_size
MERGE = 3    # vision.downsample_ratio


def span_len(w: int, h: int) -> int:
    """The rows an image of w x h pixels makes: image_start, n_h rows of (n_w aligner rows, image_newline), image_end."""
    gh, gw = h // PATCH, w // PATCH
    n_h, n_w = -(-gh // MERGE), -(-gw // MERGE)
    return 2 + n_h * (n_w + 1)


def rgb_field(w: int, h: int, salt: str) -> bytes:
    """A w x h RGB field: waves in every channel (a picture with structure), the salt shifting them, so two fields of
    the same size differ everywhere."""
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:4], "little")
    out = bytearray(w * h * 3)
    i = 0
    for y in range(h):
        for x in range(w):
            out[i] = (x * 3 + y * 2 + seed) & 0xFF
            out[i + 1] = ((x ^ y) + (seed >> 8)) & 0xFF
            out[i + 2] = (x * 5 - y * 3 + (seed >> 16)) & 0xFF
            i += 3
    return bytes(out)


def dsi1(path: str, pics: list[tuple[int, int, int, bytes]]) -> str:
    """A pictures file (int32 'DSI1', int32 count, then per picture int64 start, int32 w, int32 h, its RGB)."""
    with open(path, "wb") as f:
        f.write(b"DSI1")
        f.write(struct.pack("<i", len(pics)))
        for start, w, h, rgb in pics:
            f.write(struct.pack("<qii", start, w, h))
            f.write(rgb)
    return path


def geni(eng, prompt: list[int], pics_path: str, max_new: int, spec: int = 0, timeout: float = 3600.0) -> dict:
    """A picture request: `GENI <max_new> [keys] <file> <ids>`."""
    eng.proc.stdin.write(f"GENI {max_new} spec={spec} {pics_path} {','.join(str(t) for t in prompt)}\n".encode())
    eng.proc.stdin.flush()
    out = {"tokens": [], "reused": None, "prompt": None, "finish": None, "s": 0.0}
    t0 = time.monotonic()
    end = t0 + timeout
    while True:
        line = eng._line(end)
        if line.startswith("T "):
            out["tokens"].append(int(line[2:]))
        elif line.startswith("ERR "):
            out["error"] = line[4:]
            return out
        elif line.startswith("DONE "):
            f = line.split()
            out["generated"], out["prompt"], out["finish"] = int(f[1]), int(f[2]), f[5]
            out["reused"] = int(f[8])
            out["s"] = time.monotonic() - t0
            return out
        elif line.startswith("NOTE "):
            eng.notes.append(line[5:].strip())


def meta_pics(path: str) -> list[tuple[int, int]] | None:
    """A slot's .meta (DS4META3): its pictures (start, hash) - None: not one of the engine's, or damaged."""
    try:
        data = open(path, "rb").read()
        m0, il = struct.unpack_from("<QQ", data, 0)
        if m0 != SK.META_MAGIC:
            return None
        size, nt = struct.unpack_from("<QQ", data, 16 + il)
        (nc,) = struct.unpack_from("<Q", data, 32 + il + 4 * nt)
        off = 40 + il + 4 * nt + 8 * nc
        (np,) = struct.unpack_from("<Q", data, off)
        off += 8
        pics = []
        for _ in range(np):
            start, h = struct.unpack_from("<qQ", data, off)
            pics.append((start, h))
            off += 16
        (m1,) = struct.unpack_from("<Q", data, off)
    except (OSError, struct.error):
        return None
    return pics if m1 == SK.META_MAGIC else None


def run(exe: str, gpus: list[int], pack: str, ctx: int, slots: str, keep: bool,
        steps: list[tuple[str, "callable"]], max_new: int, settings: dict) -> tuple[dict, list[str]]:
    """One engine process with the vision tower on; a step gives (prompt, pictures); text-only steps go as GEN."""
    env = SK.engine_env(settings, keep, slots, None)
    args = [*(("--gpus", ",".join(str(g) for g in gpus)) if gpus else ()), "--pack", pack, "--max-context", str(ctx),
            "--dspark", "on", "--threads", "on", "--vision", "on", "--expert-profile", "none", "--serve"]
    eng = SK.Engine(exe, args, env)
    answers: dict[str, dict] = {}
    d = tempfile.mkdtemp(prefix="ds4-pics-")
    try:
        eng.start(timeout=1800.0)
        for name, make in steps:
            prompt, pics = make(answers)
            if pics:
                answers[name] = geni(eng, prompt, dsi1(os.path.join(d, name + ".dsi1"), pics), max_new)
            else:
                answers[name] = eng.request(prompt, max_new)
            if "error" in answers[name]:
                raise RuntimeError(f"{name}: {answers[name]['error']}")
            answers[name]["len"] = len(prompt)
    finally:
        if eng.proc is not None:
            eng.quit()
        err = eng.stderr_lines()
    for line in err:
        if line.startswith("ds4 slots:") and "failed" in line:
            raise RuntimeError("the engine: " + line)
    return answers, eng.notes


def main() -> int:
    ap = argparse.ArgumentParser(description="pictures in the kept conversations")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--exe", default=str(Path(__file__).resolve().parents[2] / "build" / "maya-ds4"))
    ap.add_argument("--gpus", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--len-a", type=int, default=2000, help="A's text (tokens, before and after the picture)")
    ap.add_argument("--len-b", type=int, default=2500)
    ap.add_argument("--len-turn", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--pic-size", type=int, default=448, help="the picture's side (pixels, a multiple of 14)")
    ap.add_argument("--img-token", type=int, default=129264, help="the pack's image_token_id")
    a = ap.parse_args()
    settings = SK.config_env(a.config) if a.config else {}
    if not os.path.exists(a.exe):
        print(f"{a.exe}: no engine", flush=True)
        return 2
    try:
        from strata_tokenizer import Tokenizer
    except ImportError as e:
        print(f"the tokenizer cannot be imported: {e}", flush=True)
        return 2
    tok = Tokenizer.from_dir(os.path.join(a.pack, "tokenizer"))
    img_tok = a.img_token
    W = H = a.pic_size
    span = span_len(W, H)
    pre = SK.text_tokens(tok, "Before the picture.", a.len_a)
    post = SK.text_tokens(tok, "After the picture.", a.len_a)
    b = SK.text_tokens(tok, "Conversation B.", a.len_b)
    turn = SK.text_tokens(tok, "The next turn:", a.len_turn)
    base = pre + [img_tok] * span + post
    start = len(pre)
    rgb1, rgb2 = rgb_field(W, H, "one"), rgb_field(W, H, "two")
    one, two = [(start, W, H, rgb1)], [(start, W, H, rgb2)]
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    max_new = a.max_new
    ok = True

    def check(what: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + what, flush=True)
        ok = ok and cond

    def turn_prompt(answers) -> tuple[list[int], list]:
        return base + answers["a"]["tokens"] + turn, one

    # the reference: no slots
    ref, _ = run(a.exe, gpus, a.pack, a.ctx, tempfile.mkdtemp(), False, [
        ("a", lambda _: (base, one)),
        ("b", lambda _: (b, [])),
        ("turn", turn_prompt)], max_new, settings)
    check(f"no-keep: the next turn starts over (reused {ref['turn']['reused']})", ref["turn"]["reused"] == 0)

    # the same, with slots
    slots = tempfile.mkdtemp()

    def turned(answers) -> tuple[list[int], list]:
        ans = answers["a"]
        return base + ans["tokens"] + turn + [img_tok] * span + post, one + [(len(base) + len(ans["tokens"]) + len(turn), W, H, rgb2)]

    keep, notes = run(a.exe, gpus, a.pack, a.ctx, slots, True, [
        ("a", lambda _: (base, one)),
        ("b", lambda _: (b, [])),
        ("turn", turn_prompt),
        ("b2", lambda _: (b, [])),
        ("turned", turned),
        ("b3", lambda _: (b, [])),
        ("other", lambda _: (base, two))], max_new, settings)
    check("the picture chat's answer is the reference's", keep["a"]["tokens"] == ref["a"]["tokens"])
    want = keep["a"]["len"] + len(keep["a"]["tokens"])
    check(f"the next turn takes the slot back whole (reused {keep['turn']['reused']} of {want}), nothing of the span read",
          keep["turn"]["reused"] == want)
    check("the next turn's answer is the reference's", keep["turn"]["tokens"] == ref["turn"]["tokens"])
    check(f"a picture added after the slot: taken back to the turn's end (reused {keep['turned']['reused']})",
          keep["turned"]["reused"] >= keep["turn"]["len"] - 1)
    check("a picture added after the slot: the answer is the reference's", keep["turned"]["tokens"] == ref["turn"]["tokens"])
    check(f"another picture at the same place ({start}): no slot taken (reused {keep['other']['reused']})",
          keep["other"]["reused"] == 0)
    metas = [p for p in os.listdir(slots) if p.endswith(".meta")]
    named = [meta_pics(os.path.join(slots, p)) for p in metas]
    check(f"every .meta names its picture at {start}: {named}",
          bool(named) and all(len(m) == 1 and m[0][0] == start for m in named if m))
    print("\n".join("   note: " + n for n in notes), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
