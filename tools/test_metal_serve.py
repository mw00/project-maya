"""tools/test_metal_serve.py - maya-metal --serve as serve/server.py drives it, on the synthetic fixture: a paused request's
follow-up (the prompt + what was written) reuses the state and writes what one uninterrupted request writes; a STOP
cancels; a seeded sampled request writes the same tokens twice; drafting (spec=1 or not) does not change the tokens.

    python tools/test_metal_serve.py MAYA_METAL GGUF
"""
import subprocess
import sys

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


class Engine:
    def __init__(self, exe: str, gguf: str):
        self.p = subprocess.Popen([exe, "--serve", "--gguf", gguf, "--max-context", "256"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self.p.stdout:
            if line.startswith("READY"):
                self.flags = line.split()[2:]
                return
        raise SystemExit("maya-metal exited before READY")

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def gen(self, max_new: int, ids: list, keys: str = "", pause_after: int = 0):
        """(tokens, DONE fields)"""
        self.send(f"GEN {max_new} {keys} {','.join(map(str, ids))}".replace("  ", " "))
        toks = []
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line[2:]))
                if pause_after and len(toks) == pause_after:
                    self.send("PAUSE")
            elif line.startswith("DONE"):
                return toks, line.split()
            elif line.startswith("ERR"):
                raise SystemExit(line.strip())
        raise SystemExit("maya-metal exited mid-request")

    def close(self):
        self.send("QUIT")
        self.p.wait(timeout=60)


def main() -> int:
    exe, gguf = sys.argv[1], sys.argv[2]
    e = Engine(exe, gguf)
    assert "stop" in e.flags and "pause" in e.flags, e.flags
    ref, done = e.gen(30, PROMPT)
    assert len(ref) == 30 and done[5] == "length", done

    # PAUSE after 6 tokens: cancel, then the follow-up continues from the kept state and writes the same tokens
    part, done = e.gen(30, PROMPT, pause_after=6)
    assert done[5] == "cancel" and ref[:len(part)] == part, (done, part)
    follow = PROMPT + part
    rest, done = e.gen(30 - len(part), follow)
    reuse = int(done[8])
    assert reuse == len(follow) - 1, f"the follow-up reused {reuse} of {len(follow) - 1} tokens"
    assert part + rest == ref, "a paused request's follow-up wrote other tokens"

    # seeded sampling: the same tokens twice; spec=1 (no drafting) the same tokens as drafting
    a, _ = e.gen(20, PROMPT, "temperature=0.8 top_k=40 seed=7")
    b, _ = e.gen(20, PROMPT, "temperature=0.8 top_k=40 seed=7 spec=1")
    assert a == b, "seeded sampling differs between two requests (or with spec=1)"
    g, _ = e.gen(20, PROMPT, "spec=1")
    assert g == ref[:20], "spec=1 changed the greedy tokens"
    e.close()
    print(f"test_metal_serve: PASS (pause reuse {reuse} tokens, {len(ref)} greedy and {len(a)} seeded tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
