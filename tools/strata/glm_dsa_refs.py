"""tools/strata/glm_dsa_refs.py - regenerate data/glm-dsa/ (reference hashes and logits).

What it writes (see data/glm-dsa/README.md for the formats):
  sha256.txt                       sha256 of every standard synthetic preset at the default seed
  oracle-hf-small-48.f32           ref/glm_dsa.py --mode hf logits, SMALL, 48 tokens
  oracle-hf-small-mtp-47.f32       its NextN draft logits for rows 0..46
  llama-small-48.f32               llama.cpp CPU logits (tools/strata/glm_dsa_llama_logits.cpp)
  llama-long-2080-tail32.f32       llama.cpp CPU logits, LONG, 2080 tokens, rows 2048..2079
  llama-pack-12.f32, llama-tiers-12.f32   llama.cpp CPU logits on the quantized presets

The llama.cpp part needs a CPU-only llama.cpp build (out of tree; the source is never written):
    python tools/strata/glm_dsa_refs.py --llama-src ~/ggml-org/llama.cpp --llama-build /tmp/llama-cpu
Without --llama-src only the hashes and the oracle logits are refreshed.

Token convention for every reference: token i = (i * 37 + 11) % n_vocab.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))

import glm_dsa_synth_gguf as synth  # noqa: E402


def tokens(n: int, n_vocab: int) -> list[int]:
    return [(i * 37 + 11) % n_vocab for i in range(n)]


def build_llama(src: pathlib.Path, build: pathlib.Path, jobs: int) -> pathlib.Path:
    if not (build / "src" / "libllama.a").exists():
        subprocess.run(["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF",
                        "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_COMMON=OFF"],
                       check=True)
        subprocess.run(["ninja", "-C", str(build), f"-j{jobs}", "llama"], check=True)
    exe = build / "glm_dsa_llama_logits"
    subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(src / "include"), "-I", str(src / "ggml" / "include"),
                    str(ROOT / "tools" / "strata" / "glm_dsa_llama_logits.cpp"), "-o", str(exe),
                    str(build / "src" / "libllama.a"), str(build / "ggml" / "src" / "libggml.a"),
                    str(build / "ggml" / "src" / "libggml-cpu.a"), str(build / "ggml" / "src" / "libggml-base.a"),
                    "-fopenmp", "-lm", "-pthread"], check=True)
    return exe


def llama_logits(exe: pathlib.Path, gguf: pathlib.Path, toks: list[int], n_vocab: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "l.f32"
        subprocess.run([str(exe), str(gguf), str(out)] + [str(t) for t in toks], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.fromfile(out, dtype=np.float32).reshape(len(toks), n_vocab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "glm-dsa"))
    ap.add_argument("--llama-src", default=None)
    ap.add_argument("--llama-build", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from glm_dsa import GlmDsa, Options
    with tempfile.TemporaryDirectory() as d:
        files = {}
        lines = []
        for stem, preset, kb in synth.default_set(include_large=False):
            p = pathlib.Path(d) / f"{stem}.gguf"
            synth.write(p, preset, synth.DEFAULT_SEED, kb)
            files[stem] = p
            lines.append(f"{synth.sha256(p)}  {stem}.gguf")
        (out / "sha256.txt").write_text("\n".join(lines) + "\n")
        print(f"sha256.txt: {len(lines)} presets")

        m = GlmDsa(files["glm-dsa-small"], Options(mode="hf"))
        t48 = tokens(48, m.cfg.n_vocab)
        lg = m.forward(t48)
        lg.T.astype(np.float32).tofile(out / "oracle-hf-small-48.f32")
        m.mtp(m.h_nextn[:, :-1], t48[1:]).T.astype(np.float32).tofile(out / "oracle-hf-small-mtp-47.f32")
        m.close()
        print("oracle-hf-small-48.f32, oracle-hf-small-mtp-47.f32")

        if a.llama_src:
            build = pathlib.Path(a.llama_build or (pathlib.Path(d) / "llama-cpu"))
            exe = build_llama(pathlib.Path(a.llama_src), build, a.jobs)
            llama_logits(exe, files["glm-dsa-small"], t48, 512).tofile(out / "llama-small-48.f32")
            llama_logits(exe, files["glm-dsa-long"], tokens(2080, 512), 512)[2048:].tofile(
                out / "llama-long-2080-tail32.f32")
            llama_logits(exe, files["glm-dsa-pack"], tokens(12, 1024), 1024).tofile(out / "llama-pack-12.f32")
            llama_logits(exe, files["glm-dsa-tiers"], tokens(12, 1024), 1024).tofile(out / "llama-tiers-12.f32")
            print("llama-*.f32 written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
