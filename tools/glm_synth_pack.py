"""tools/glm_synth_pack.py - generate the tiny pack fixture for glm_seq_dynamic_test.

The 6-layer synthetic GGUF (data/glm5-synth.gguf, the same one glm_model_test loads) packed
through tools/iq_pack.py -- the same packer that built the Maya-S24 pack, so Task 2.1's
lifecycle test runs the REAL pack loader instead of a plain-GGUF stand-in.  The pack lands beside
the GGUF it serves from (pack_shard_names resolves the shards one directory up, glm_model.cu:67-95).

The GGUF carries no tokenizer, so the two tokenizer files iq_pack.py's generator step checks are
pre-created empty (the engine never reads them).

    python tools/glm_synth_pack.py --out <build-tree>/glm5-synth-pack
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=str(HERE.parent / "data" / "glm5-synth.gguf"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    src = pathlib.Path(a.gguf)
    if out.exists():
        shutil.rmtree(out)
    (out / "tokenizer").mkdir(parents=True)
    (out / "tokenizer" / "vocab.json").touch()
    (out / "tokenizer" / "chat_template.jinja").touch()
    r = subprocess.run([sys.executable, str(HERE / "iq_pack.py"), "--gguf", str(src),
                        "--out", str(out), "--compat-bf16"])
    if r.returncode != 0:
        return r.returncode
    # the shards sit NEXT TO the pack (the loader opens pack_dir/../shard): the GGUF must be there
    if src.parent != out.parent:
        shutil.copy2(src, out.parent / src.name)
    print("glm5-synth-pack: ready at", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
