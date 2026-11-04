#!/usr/bin/env python3
"""tools/ds4/slot_api_image_test.py - pictures in the kept conversations, through a RUNNING server's API (the real
path: server -> resize (serve/ds4_vision.py) -> GENI -> engine slot).  Meant for the running Maya (port 8090); it reads
maya-ds4.log for the engine's own lines.

    python tools/ds4/slot_api_image_test.py             (URL/LOG at the top of this file)

A chat with a picture is sent, another chat takes the model, then the chat's next turn (the picture again in its
history) - the engine must take the slot back and read only the new tokens.  Then the same with another picture at the
same place: the slot must NOT be taken.

Exit 0: every check passed.  1: a check failed.
"""
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
LOG = "/home/mark/LLM/maya/maya-ds4.log"

from PIL import Image  # noqa: E402


def png(salt: str, w: int = 900, h: int = 700) -> bytes:
    im = Image.new("RGB", (w, h))
    px = im.load()
    seed = sum(salt.encode()) & 0xFFF
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 3 + y * 2 + seed) & 255, ((x ^ y) + (seed >> 2)) & 255, (x * 5 - y * 3 + seed) & 255)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def text(salt: str, n: int) -> str:
    """A long, distinct passage (n characters) - the prompt's bulk."""
    base = ("Thema %s. Der Motor zieht gleichmäßig an, die Straße steigt, und der Fahrer zählt die Kurven. "
            "Werkstatt, Reifen, Getriebe, Bremsen: alles wird geprüft, bevor die Fahrt beginnt. ") % salt
    return (base * (n // len(base) + 1))[:n]


def post(messages, max_tokens=12, timeout=900):
    body = json.dumps({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.monotonic() - t0


def log_lines(since: int) -> list[str]:
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    return lines[since:]


def log_mark() -> int:
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            return len(f.read().splitlines())
    except OSError:
        return 0


def show(name: str, lines: list[str]) -> list[str]:
    keep = [l for l in lines if "ds4 slots:" in l or "maya ds4 serve: prompt" in l or l.startswith("PP ")]
    print(f"\n[{name}] engine lines:")
    for l in keep[-8:]:
        print("   " + l)
    return keep


def main() -> int:
    pic1, pic2 = data_url(png("picture one")), data_url(png("picture two"))
    block1 = {"type": "image_url", "image_url": {"url": pic1}}
    block2 = {"type": "image_url", "image_url": {"url": pic2}}
    sysmsg = {"role": "system", "content": "Du bist ein Test."}
    ask = "Beschreibe das Bild in fünf Worten."
    a1 = [sysmsg, {"role": "user", "content": [{"type": "text", "text": text("A", 16000)}, block1,
                                               {"type": "text", "text": ask}]}]
    b1 = [sysmsg, {"role": "user", "content": text("B", 12000) + " Antworte mit einem Wort."}]
    a3 = [sysmsg, {"role": "user", "content": [{"type": "text", "text": text("A", 16000)}, block2,
                                               {"type": "text", "text": ask}]}]
    ok = True

    def check(what, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + what)
        ok = ok and cond

    m = log_mark()
    ra1, s = post(a1)
    l1 = show("A1 (picture, cold)", log_lines(m))
    check(f"A1 is read cold ({s:.1f} s)", any("prompt" in l and " = 0 reused" in l for l in l1))

    m = log_mark()
    r, s = post(b1)
    l2 = show("B1 (another chat)", log_lines(m))
    check("A1's conversation was set aside to a slot", any("saved a" in l and "conversation" in l for l in l2))

    a2 = a1 + [{"role": "assistant", "content": ra1["choices"][0]["message"]["content"]},
               {"role": "user", "content": "Und jetzt noch ein Satz dazu."}]
    m = log_mark()
    r2, s2 = post(a2)
    l3 = show("A2 (the picture chat's next turn)", log_lines(m))
    took = [l for l in l3 if "took back a" in l]
    check(f"A2 took the picture conversation back from a slot ({s2:.1f} s): {took[-1] if took else 'none'}", bool(took))
    new = [int(re.search(r"(\d+) new tokens to read", l).group(1)) for l in took if re.search(r"(\d+) new tokens to read", l)]
    check(f"A2 read only a few new tokens ({new[-1] if new else '?'})", bool(new) and new[-1] < 500)

    m = log_mark()
    post(b1)
    l4 = show("B2 (another chat again)", log_lines(m))
    check("A2's conversation was set aside", any("saved a" in l for l in l4))

    m = log_mark()
    r3, s3 = post(a3)
    l5 = show("A3 (another picture, same place)", log_lines(m))
    check("A3 took NO slot (another picture)", not any("took back a" in l for l in l5))
    check(f"A3 is read cold ({s3:.1f} s)", any("prompt" in l and " = 0 reused" in l for l in l5))

    print(f"\nanswers: A1={ra1['choices'][0]['message']['content'][:40]!r}, A2={r2['choices'][0]['message']['content'][:40]!r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
