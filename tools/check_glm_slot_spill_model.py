#!/usr/bin/env python3
"""One-shot native GLM slot-spill validation for the isolated PR #994 branch.

This is deliberately a driver, not a benchmark or a service controller.  It starts
exactly two ``--serve`` children and sends exactly four GEN requests when invoked:
phase 1 A/B/A then QUIT, phase 2 A then QUIT.  Its output directory is evidence,
including the complete native stdout/stderr streams and an atomic receipt written
before every request.  A failed assertion stops the sequence; it never retries or
widens a budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RAM_CAP = int(0.001 * 1024 ** 3)
DISK_CAP = 2 * 1024 ** 3
META_MAGIC = 0x3153534559414D47  # GMAYESS1, src/program/generate.cpp


class CheckFailure(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in ids).encode("ascii")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def readable(stream, timeout: float) -> bool:
    ready, _, _ = select.select([stream], [], [], timeout)
    return bool(ready)


def parse_meta(path: Path) -> dict[str, Any] | None:
    """Read only enough of Maya's GLM slot metadata to identify A and its bytes."""
    try:
        data = path.read_bytes()
        magic, identity_len = struct.unpack_from("<QQ", data, 0)
        if magic != META_MAGIC:
            return None
        offset = 16 + identity_len
        size, count = struct.unpack_from("<QQ", data, offset)
        offset += 16
        tokens = list(struct.unpack_from(f"<{count}i", data, offset))
        offset += 4 * count
        image_hash, tail = struct.unpack_from("<QQ", data, offset)
        if tail != META_MAGIC:
            return None
        return {"bytes": size, "tokens": tokens, "image_hash": image_hash}
    except (OSError, struct.error, ValueError):
        return None


def snapshot_slots(slots: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not slots.exists():
        return out
    for meta in sorted(slots.glob("*.meta")):
        parsed = parse_meta(meta)
        binary = meta.with_suffix(".bin")
        out.append({
            "meta": meta.name,
            "bin": binary.name,
            "meta_bytes": meta.stat().st_size,
            "bin_bytes": binary.stat().st_size if binary.exists() else None,
            "parsed": parsed,
        })
    return out


def matching_slot(slots: Path, prompt: list[int]) -> dict[str, Any] | None:
    # The saved state is immediately before the generation token, hence A[:-1].
    wanted = prompt[:-1]
    for entry in snapshot_slots(slots):
        parsed = entry["parsed"]
        if parsed and parsed["tokens"] == wanted and entry["bin_bytes"] is not None:
            return entry
    return None


def selected_gpu_preflight(gpus: list[int], strict: bool) -> dict[str, Any]:
    """Run only when the driver runs: reject selected GPUs with compute users."""
    cmd = ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"]
    try:
        info = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
                              timeout=20)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise CheckFailure(f"GPU preflight cannot run: {e}") from e
    rows: dict[int, dict[str, int]] = {}
    for line in info.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            continue
        rows[int(parts[0])] = {"free_mib": int(parts[1]), "total_mib": int(parts[2])}
    if any(g not in rows for g in gpus):
        raise CheckFailure(f"selected GPU absent from nvidia-smi: {gpus}")
    # nvidia-smi's CSV has UUID rather than index for compute apps.  Query each selected index separately,
    # avoiding assumptions about CUDA_VISIBLE_DEVICES remapping.
    busy: dict[int, str] = {}
    for gpu in gpus:
        p = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
                           timeout=20)
        if p.stdout.strip() and "No running processes found" not in p.stdout:
            busy[gpu] = p.stdout.strip()
    # The isolated four-card GLM load is not safe to start on a merely idle display-sized remainder.
    # 20 GiB/card is a conservative preflight floor; the engine still makes the definitive allocation check.
    low = {g: rows[g] for g in gpus if rows[g]["free_mib"] < 20 * 1024}
    receipt = {"selected": gpus, "gpus": rows, "compute_apps": apps.stdout.splitlines(), "busy": busy,
               "low_free": low}
    if strict and (busy or low):
        raise CheckFailure("GPU preflight failed: selected GPUs must be compute-idle with >=20480 MiB free: " +
                           json.dumps(receipt, ensure_ascii=False))
    return receipt


class Engine:
    def __init__(self, exe: Path, args: list[str], env: dict[str, str], evidence: Path, phase: str):
        self.exe, self.args, self.env, self.phase = exe, args, env, phase
        self.evidence = evidence
        self.proc: subprocess.Popen[bytes] | None = None
        self.stdout = None
        self.stderr = None
        self.buffer = b""
        self.notes: list[str] = []
        self.lines: list[str] = []

    def start(self, timeout: float = 600.0) -> None:
        self.stdout = (self.evidence / f"{self.phase}.stdout.log").open("wb", buffering=0)
        # A child writes directly to this inherited descriptor, so unbuffered ownership preserves stderr
        # even if protocol handling fails before normal close.
        self.stderr = (self.evidence / f"{self.phase}.stderr.log").open("wb", buffering=0)
        self.proc = subprocess.Popen([str(self.exe), *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, env=self.env, bufsize=0)
        deadline = time.monotonic() + timeout
        while True:
            line = self._line(deadline)
            if line.startswith("READY "):
                return
            if line.startswith("ERR "):
                raise CheckFailure(f"{self.phase}: engine refused startup: {line}")

    def _record(self, raw: bytes) -> str:
        assert self.stdout is not None
        self.stdout.write(raw + b"\n")
        self.stdout.flush()
        os.fsync(self.stdout.fileno())
        line = raw.decode("utf-8", "replace").strip()
        self.lines.append(line)
        if line.startswith("NOTE "):
            self.notes.append(line[5:])
        return line

    def _line(self, deadline: float) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        while b"\n" not in self.buffer:
            if self.proc.poll() is not None:
                raise CheckFailure(f"{self.phase}: engine exited {self.proc.returncode} before protocol completion")
            left = deadline - time.monotonic()
            if left <= 0:
                raise CheckFailure(f"{self.phase}: protocol deadline expired")
            if not readable(self.proc.stdout, min(left, 1.0)):
                continue
            chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
            if not chunk:
                raise CheckFailure(f"{self.phase}: engine stdout closed")
            self.buffer += chunk
        raw, self.buffer = self.buffer.split(b"\n", 1)
        return self._record(raw)

    def request(self, name: str, prompt: list[int], outdir: Path) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdin is not None
        request_receipt = {"phase": self.phase, "request": name, "before_send_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "prompt_token_count": len(prompt), "prompt_token_ids": prompt,
                           "prompt_tokens_sha256": sha_ids(prompt), "max_new": 16, "deadline_seconds": 120}
        atomic_json(outdir / f"request-{self.phase}-{name}-before.json", request_receipt)
        self.proc.stdin.write(("GEN 16 spec=0 " + ",".join(str(x) for x in prompt) + "\n").encode("ascii"))
        self.proc.stdin.flush()
        deadline = time.monotonic() + 120.0
        out: dict[str, Any] = {"tokens": [], "raw": [], "reused": None, "finish": None}
        while True:
            line = self._line(deadline)
            out["raw"].append(line)
            if line.startswith("T "):
                out["tokens"].append(int(line[2:]))
            elif line.startswith("ERR "):
                raise CheckFailure(f"{self.phase}/{name}: {line[4:]}")
            elif line.startswith("DONE "):
                fields = line.split()
                if len(fields) < 9:
                    raise CheckFailure(f"{self.phase}/{name}: malformed DONE: {line}")
                out.update({"generated": int(fields[1]), "prompt": int(fields[2]), "finish": fields[5],
                            "reused": int(fields[8]), "tokens_sha256": sha_ids(out["tokens"])})
                atomic_json(outdir / f"request-{self.phase}-{name}-result.json", out)
                return out

    def quit(self, timeout: float = 60.0) -> None:
        assert self.proc is not None
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except OSError:
            pass
        deadline = time.monotonic() + timeout
        while self.proc.poll() is None and time.monotonic() < deadline:
            assert self.proc.stdout is not None
            if readable(self.proc.stdout, 0.2):
                chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
                if chunk:
                    self.buffer += chunk
                    while b"\n" in self.buffer:
                        raw, self.buffer = self.buffer.split(b"\n", 1)
                        self._record(raw)
        if self.proc.poll() is None:
            self._stop_owned()
            raise CheckFailure(f"{self.phase}: QUIT exceeded {timeout:.0f}s")
        if self.proc.returncode != 0:
            raise CheckFailure(f"{self.phase}: QUIT exited {self.proc.returncode}, expected 0")

    def _stop_owned(self) -> None:
        # This Popen object is the only PID this driver may signal; no process discovery or service action occurs.
        assert self.proc is not None
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is not None and self.proc.stdout is not None:
            self.buffer += self.proc.stdout.read()
            while b"\n" in self.buffer:
                raw, self.buffer = self.buffer.split(b"\n", 1)
                self._record(raw)
            if self.buffer:
                self._record(self.buffer)
                self.buffer = b""
        for f in (self.stdout, self.stderr):
            if f is not None:
                f.flush()
                os.fsync(f.fileno())
                f.close()


def synthetic_prompt(template, tokenizer, label: str) -> list[int]:
    # The label begins the content, so A/B cannot share a prompt prefix.  Repetition produces a stable,
    # human-readable 1200-1800-token request through the actual pack tokenizer.
    text = label + ": retain this identifier exactly. "
    unit = ("This isolated slot recovery check uses synthetic prose only. It must not call tools, browse, "
            "or explain its reasoning. Summarize the requested identifier in a short final answer. ")
    for n in range(1, 200):
        rendered = template.render([{"role": "user", "content": text + unit * n +
                                      f"Return only {label}-OK."}], enable_thinking=False)
        ids = tokenizer.encode(rendered, parse_special=True)
        if 1200 <= len(ids) <= 1800:
            return ids
        if len(ids) > 1800:
            break
    raise CheckFailure(f"cannot freeze {label} into 1200-1800 tokens (got {len(ids)})")


def engine_environment(slots: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not (k.startswith("MAYA_") or k.startswith("STRATA_"))}
    env.update({
        "MAYA_GLM_SLOTS": "1", "MAYA_GLM_SLOT_SPILL": "1", "MAYA_GLM_SLOT_RAM_GB": "0.001",
        "MAYA_GLM_SLOT_GB": "2", "MAYA_GLM_SLOT_MIN": "1024", "MAYA_GLM_SLOT_DIR": str(slots),
        "MAYA_GLM_SLOT_KEEP": "1", "MAYA_GLM_CKPTS": "0", "MAYA_GLM_CKPT_EVERY": "0",
        "MAYA_GLM_PREFIX_CUT": "0", "MAYA_GLM_MULTI_CKPTS": "0",
        "MAYA_GLM_NO_MTP": "1", "MAYA_GLM_RAM_GB": "160",
        "MAYA_GLM_RAM_HEADROOM_GB": "32",
    })
    return env


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--gpus", default="0,1,2,3", help="physical CUDA indices, e.g. 0,1,2,3")
    ap.add_argument("--preflight-only", action="store_true", help="freeze prompts/imports and check GPUs; do not load a model")
    a = ap.parse_args()
    repo, exe, pack, tokenizer_dir, out = (p.resolve() for p in (a.repo, a.exe, a.pack, a.tokenizer, a.output))
    if not repo.is_dir() or not exe.is_file() or not pack.is_dir() or not tokenizer_dir.is_dir():
        raise SystemExit("--repo/--exe/--pack/--tokenizer must name existing paths")
    if out.exists():
        raise SystemExit(f"--output must be a fresh path: {out}")
    gpus = [int(x) for x in a.gpus.split(",") if x.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise SystemExit("this fixed four-part protocol requires four distinct --gpus indices")
    out.mkdir(parents=True)
    slots = out / "private-slots"
    slots.mkdir()
    try:
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "tools"))
        from serve.frontend import ChatTemplate  # pylint: disable=import-error
        from strata_tokenizer import Tokenizer  # pylint: disable=import-error
        template_path = tokenizer_dir / "chat_template.jinja"
        if not template_path.is_file():
            raise CheckFailure(f"model tokenizer lacks chat_template.jinja: {template_path}")
        tok = Tokenizer.from_dir(tokenizer_dir)
        template = ChatTemplate(template_path)
        prompt_a, prompt_b = synthetic_prompt(template, tok, "A-ISOLATED-ALPHA"), synthetic_prompt(template, tok, "B-ISOLATED-BRAVO")
        # Chat templates naturally share opening control tokens.  The user text begins with distinct labels;
        # require that neither resulting token list is a prefix of the other rather than pretending headers differ.
        require(prompt_a != prompt_b and not (prompt_a[:len(prompt_b)] == prompt_b or prompt_b[:len(prompt_a)] == prompt_a),
                "A and B must not be token-prefixes of one another")
        # Preparation is intentionally non-disruptive: it records busy production GPUs but does not
        # reject them.  The actual model run applies the strict idle/headroom gate immediately before launch.
        preflight = selected_gpu_preflight(gpus, strict=not a.preflight_only)
        args = ["--glm-pack", str(pack), "--max-context", "4096", "--parallel", "1", "--prefill", "1024",
                "--layer-split", "12,24,35", "--glm-ram-resident", "--seed", "1729", "--greedy",
                "--temperature", "0", "--serve"]
        env = engine_environment(slots)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Parse the exact command and resolve shared libraries before entering any model window.
        # --help exits in option parsing, before the GLM pack load or CUDA context initialization.
        cli = subprocess.run([str(exe), *args, "--help"], env=env, capture_output=True, timeout=20)
        (out / "native-help.stdout.log").write_bytes(cli.stdout)
        (out / "native-help.stderr.log").write_bytes(cli.stderr)
        require(cli.returncode == 0, f"native shared-library/CLI preflight failed, rc={cli.returncode}; see native-help logs")
        manifest = {"protocol": "P0-TTFT isolated disk spill", "exe": str(exe), "exe_sha256": sha_file(exe),
                    "pack": str(pack), "tokenizer": str(tokenizer_dir), "gpus": gpus, "engine_args": args,
                    "preflight": preflight, "prompt_a": {"tokens": prompt_a, "sha256": sha_ids(prompt_a)},
                    "prompt_b": {"tokens": prompt_b, "sha256": sha_ids(prompt_b)},
                    "tokenizer_json_sha256": sha_file(tokenizer_dir / "tokenizer.json"),
                    "chat_template_sha256": sha_file(template_path),
                    "source_sha256": sha_file(repo / "src/program/generate.cpp"),
                    "native_cli_check": {"returncode": cli.returncode, "model_loaded": False},
                    "environment": {k:v for k,v in env.items() if k.startswith(("MAYA_", "CUDA_")) or k == "LD_LIBRARY_PATH"}}
        atomic_json(out / "manifest.json", manifest)
        if a.preflight_only:
            atomic_json(out / "summary.json", {"status": "PREFLIGHT_PASS", "model_loaded": False,
                                                 "requests_sent": 0, "manifest": "manifest.json"})
            return 0
        answers: dict[str, dict[str, Any]] = {}
        e1 = Engine(exe, args, env, out, "phase1")
        try:
            e1.start()
            answers["A_cold"] = e1.request("A-cold", prompt_a, out)
            require(answers["A_cold"]["reused"] == 0 and bool(answers["A_cold"]["tokens"]), "A cold must be non-empty with reused=0")
            answers["B_evict"] = e1.request("B-evict", prompt_b, out)
            slot = matching_slot(slots, prompt_a)
            require(slot is not None, "B must spill matching A .bin/.meta before A restore")
            require(slot["bin_bytes"] > RAM_CAP, "A spilled .bin must exceed 0.001 GiB RAM cap")
            require(sum(x["bin_bytes"] or 0 for x in snapshot_slots(slots)) <= DISK_CAP, "slot files exceed 2 GiB disk cap")
            atomic_json(out / "after-B-slots.json", snapshot_slots(slots))
            answers["A_fromdisk"] = e1.request("A-fromdisk", prompt_a, out)
            require(any("took back" in n and "from disk" in n for n in e1.notes), "phase1 A must log disk restore")
            require(answers["A_fromdisk"]["reused"] == len(prompt_a) - 1, "phase1 A reuse must be len(A)-1")
            require(answers["A_fromdisk"]["tokens"] == answers["A_cold"]["tokens"] and
                    answers["A_fromdisk"]["finish"] == answers["A_cold"]["finish"], "phase1 A output/finish differs from cold")
            e1.quit()
        finally:
            if e1.proc is not None and e1.proc.poll() is None:
                e1._stop_owned()
            e1.close()
        atomic_json(out / "after-phase1-slots.json", snapshot_slots(slots))
        e2 = Engine(exe, args, env, out, "phase2")
        try:
            e2.start()
            answers["A_afterrestart"] = e2.request("A-afterrestart", prompt_a, out)
            require(any("kept from earlier runs" in n for n in e2.notes), "phase2 startup must index earlier slot files")
            require(any("took back" in n and "from disk" in n for n in e2.notes), "phase2 A must log disk restore")
            require(answers["A_afterrestart"]["reused"] == len(prompt_a) - 1, "restart A reuse must be len(A)-1")
            require(answers["A_afterrestart"]["tokens"] == answers["A_cold"]["tokens"] and
                    answers["A_afterrestart"]["finish"] == answers["A_cold"]["finish"], "restart A output/finish differs from cold")
            e2.quit()
        finally:
            if e2.proc is not None and e2.proc.poll() is None:
                e2._stop_owned()
            e2.close()
        atomic_json(out / "final-slots.json", snapshot_slots(slots))
        for answer in answers.values():
            answer["text"] = tok.decode(answer["tokens"])
        final = {"status": "PASS", "answers": answers, "model_loads": 2, "requests_sent": 4,
                 "slot_dir": str(slots)}
        atomic_json(out / "final-result.json", final)
        atomic_json(out / "summary.json", final)
        return 0
    except (CheckFailure, OSError, ValueError, ImportError) as e:
        final = {"status": "FAIL", "error": str(e)}
        atomic_json(out / "final-result.json", final)
        atomic_json(out / "summary.json", final)
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
