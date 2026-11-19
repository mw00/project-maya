#!/usr/bin/env python3
"""Compile actual learned-prefix/memory headers and run tiny Windows/POSIX fixtures.

No model/GPU/server/WSL. Linux cgroup parsing uses only isolated fixture files.
The receipt is exclusive-created: each attempt requires a different output path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("include/strata/artifact/gguf_reader.hpp", "include/strata/core/glm_pack_digest.hpp", "src/core/glm_model.cu",
          "include/strata/core/glm_prefix_admission.hpp", "include/strata/core/glm_cache_memory.hpp",
          "include/strata/core/glm_prefix_metadata.hpp", "include/strata/core/glm_slot_directory_scan.hpp",
          "include/strata/core/regular_file_stream.hpp", "include/strata/core/glm_state_layout.hpp",
          "include/strata/core/glm_model.hpp", "src/core/glm_pack_identity.cu",
          "tests/core/glm_prefix_cache_host_test.cpp", "tools/test_glm_prefix_cache_host.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    args = ap.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before compilation, without destroying previous evidence.
    with args.receipt.open("x", encoding="utf-8") as f:
        f.write('{"status":"started"}\n')
    receipt = {"scope": "actual helper headers; mocked cgroup files; no model/GPU/WSL", "inputs": {}, "status": "preflight"}
    rc = 2
    try:
        receipt["inputs"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}
        cxx = shutil.which(args.cxx)
        if not cxx:
            raise RuntimeError(f"Compiler not found: {args.cxx}; run from a developer shell or set --cxx")
        receipt["compiler"] = cxx
        with tempfile.TemporaryDirectory(prefix="maya-prefix-host-") as temp:
            td = Path(temp).resolve()
            source = ROOT / "tests/core/glm_prefix_cache_host_test.cpp"
            exe = td / ("fixture.exe" if os.name == "nt" else "fixture")
            if Path(cxx).stem.lower() in ("cl", "clang-cl"):
                cmd = [cxx, "/nologo", "/std:c++20", "/EHsc", "/W4", "/D_CRT_SECURE_NO_WARNINGS", f"/I{ROOT / 'include'}",
                       f"/Fo{td / 'fixture.obj'}", str(source), f"/Fe:{exe}"]
            else:
                cmd = [cxx, "-std=c++20", "-O0", "-Wall", "-Wextra", "-I", str(ROOT / "include"),
                       str(source), "-o", str(exe)]
            built = subprocess.run(cmd, cwd=td, text=True, capture_output=True, timeout=60)
            receipt["compile"] = {"exit_code": built.returncode, "stdout": built.stdout, "stderr": built.stderr}
            print(built.stdout, end=""); print(built.stderr, end="")
            rc = built.returncode
            if rc == 0:
                fixture_root = (td / "data").resolve()
                if not fixture_root.is_relative_to(td):
                    raise RuntimeError("Fixture escaped temporary directory")
                ran = subprocess.run([str(exe), str(fixture_root)], cwd=td, text=True, capture_output=True, timeout=30)
                receipt["run"] = {"exit_code": ran.returncode, "stdout": ran.stdout, "stderr": ran.stderr}
                print(ran.stdout, end=""); print(ran.stderr, end="")
                rc = ran.returncode
            receipt["status"] = "passed" if rc == 0 else "failed"
        after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}
        receipt["inputs_unchanged"] = after == receipt["inputs"]
        if not receipt["inputs_unchanged"]:
            receipt["status"] = "invalid_source_changed"
            receipt["inputs_after"] = after
            rc = 2
    except Exception as exc:
        rc = 2
        receipt["exception"] = f"{type(exc).__name__}: {exc}"
        receipt["status"] = "error"
        print(receipt["exception"])
    finally:
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
