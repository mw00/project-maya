#!/usr/bin/env python3
"""Native GLM protocol proof on a synthetic CUDA-compatible pack.

Checks answer continuation, SSD restart, context displacement with --parallel,
and read-only migration of compatible legacy files. No downloaded model needed.
"""

import argparse
import hashlib
import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path


class Engine:
    def __init__(self, exe, pack, directory, cache, parallel=1, legacy=None, spec=1):
        self.parallel = parallel
        self.offered = 0
        self.spec = spec
        env = dict(
            os.environ,
            MAYA_GLM_STATE_CACHE=str(int(cache)),
            MAYA_GLM_STATE_DIR=str(directory),
            MAYA_GLM_STATE_RAM_GB=os.environ.get("MAYA_TEST_STATE_RAM_GB", "1"),
            MAYA_GLM_STATE_DISK_GB="1",
        )
        if legacy is not None:
            env.update(
                MAYA_GLM_SLOT_KEEP="1",
                MAYA_GLM_SLOT_MIN="1",
                MAYA_GLM_SLOT_DIR=str(legacy),
            )
        self.log = open(str(directory) + ".stderr", "w")
        self.process = subprocess.Popen(
            [
                exe,
                "--serve",
                "--glm-pack",
                pack,
                "--max-context",
                "256",
                "--prefill",
                "16",
                "--parallel",
                str(parallel),
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            bufsize=0,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.input = self.process.stdin
        self.output = self.process.stdout
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.output, selectors.EVENT_READ)
        self.buffer = b""
        while not self.line().startswith("READY"):
            pass

    def line(self):
        end = time.monotonic() + 120
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0, end - time.monotonic())):
                raise TimeoutError("native engine timed out; " + self.log.name)
            data = os.read(self.output.fileno(), 65536)
            if not data:
                raise RuntimeError("native engine exited; " + self.log.name)
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode()

    def generate(self, prompt, n=8, rid=1):
        request_id = f"rid={rid} " if self.parallel > 1 else ""
        self.input.write(
            f"GEN {n} {request_id}spec={self.spec} spec_min_p=0 {','.join(map(str, prompt))}\n".encode()
        )
        self.input.flush()
        tokens = []
        while True:
            line = self.line()
            if line.startswith("T "):
                tokens.append(int(line.split()[-1]))
            elif line.startswith("ERR"):
                raise RuntimeError(line)
            elif line.startswith("DONE "):
                fields = line.split()
                self.offered += int(fields[8 if self.parallel > 1 else 7])
                return tokens, int(fields[9 if self.parallel > 1 else 8])

    def generate_together(self, requests):
        """Several GEN lines at once (--parallel): {rid: (tokens, reuse)} once every one is DONE."""
        for rid, (prompt, n) in requests.items():
            self.input.write(
                f"GEN {n} rid={rid} spec={self.spec} spec_min_p=0 {','.join(map(str, prompt))}\n".encode()
            )
        self.input.flush()
        tokens = {rid: [] for rid in requests}
        done = {}
        while len(done) < len(requests):
            line = self.line()
            if line.startswith("T "):
                _, rid, token = line.split()
                tokens[int(rid)].append(int(token))
            elif line.startswith("ERR"):
                raise RuntimeError(line)
            elif line.startswith("DONE "):
                fields = line.split()
                done[int(fields[1])] = (tokens[int(fields[1])], int(fields[9]))
        return done

    def close(self):
        if self.process.poll() is None:
            self.input.write(b"QUIT\n")
            self.input.flush()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument(
        "--ram-gb",
        default="1",
        help="retained RAM cap; use a tiny value to force bounded SSD spills",
    )
    parser.add_argument(
        "--skip-migration",
        action="store_true",
        help="legacy readers may require more RAM than a spill test permits",
    )
    parser.add_argument(
        "--require-drafts",
        action="store_true",
        help="fail if the native run never actually offered a draft",
    )
    args = parser.parse_args()
    engines = []
    all_engines = []
    os.environ["MAYA_TEST_STATE_RAM_GB"] = args.ram_gb
    with tempfile.TemporaryDirectory(prefix="maya-native-state-") as root:
        root = Path(root)
        prompt = list(range(1, 38))

        def engine(name, cache=True, parallel=1, legacy=None):
            result = Engine(
                args.engine,
                args.pack,
                root / name,
                cache,
                parallel,
                legacy,
                spec=0 if args.require_drafts and cache else 1,
            )
            engines.append(result)
            all_engines.append(result)
            return result

        try:
            baseline = engine("baseline", False)
            answer, _ = baseline.generate(prompt)
            following = prompt + answer
            baseline.generate(list(range(80, 101)))
            expected, _ = baseline.generate(following)
            baseline.close()
            engines.remove(baseline)
            producer = engine("persistent")
            actual, _ = producer.generate(prompt)
            assert actual == answer
            producer.close()
            engines.remove(producer)
            restarted = engine("persistent")
            restarted.generate(list(range(100, 122)))
            retry, reuse = restarted.generate(prompt)
            assert retry == answer and reuse == len(prompt) - 1, (
                reuse,
                len(prompt) - 1,
            )
            resumed, reuse = restarted.generate(following)
            assert resumed == expected and reuse == len(following) - 1, (
                reuse,
                len(following) - 1,
            )
            restarted.close()
            engines.remove(restarted)
            multi = engine("parallel", parallel=2)
            for rid, start in enumerate([1, 80, 120, 160], 1):
                tokens = prompt if rid == 1 else list(range(start, start + 37))
                result, _ = multi.generate(tokens, rid=rid)
                if rid == 1:
                    assert result == answer
            result, reuse = multi.generate(prompt, rid=5)
            assert result == answer and reuse == len(prompt) - 1, (
                reuse,
                len(prompt) - 1,
            )
            # prompts read while another request decodes, one of them ending at its first token: each prompt's
            # state is published after that token (or at the end), and a later request restores it from the cache
            first = list(range(200, 237))
            second = list(range(240, 277))
            done = multi.generate_together({6: (first, 8), 7: (second, 1)})
            assert len(done[7][0]) == 1, done
            for rid, start in enumerate([300, 340, 380], 8):   # (the two contexts hold other conversations now)
                multi.generate(list(range(start, start + 37)), rid=rid)
            for rid, tokens in ((11, first), (12, second)):
                result, reuse = multi.generate(tokens, rid=rid)
                assert reuse == len(tokens) - 1, (rid, reuse, len(tokens) - 1)
            multi.close()
            engines.remove(multi)
            if not args.skip_migration:
                source = root / "legacy-slots"
                legacy = engine("old-format", False, legacy=source)
                result, _ = legacy.generate(prompt)
                assert result == answer
                legacy.close()
                engines.remove(legacy)
                originals = {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.iterdir()
                }
                assert any(name.endswith(".meta") for name in originals), (
                    "legacy test produced no kept slot"
                )
                migrated = engine("migrated", True, legacy=source)
                result, reuse = migrated.generate(prompt)
                assert result == answer and reuse == len(prompt) - 1, reuse
                migrated.close()
                engines.remove(migrated)
                assert originals == {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.iterdir()
                }, "migration changed source files"
            offered = sum(e.offered for e in all_engines)
            assert not args.require_drafts or offered > 0, "native drafting never ran"
            print(f"native drafts offered: {offered}")
            print(
                "native GLM protocol: completed answer, SSD retry, displaced parallel contexts"
                + (" and legacy migration" if not args.skip_migration else "")
                + " passed"
            )
        except BaseException:
            for file in root.glob("*.stderr"):
                print(
                    f"--- {file.name} ---\n"
                    + "\n".join(file.read_text(errors="replace").splitlines()[-20:])
                )
            raise
        finally:
            for running in engines:
                running.close()


if __name__ == "__main__":
    main()
