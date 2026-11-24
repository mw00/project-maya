"""tools/maya_golden.py - saved expected answers: --bench's fixed requests, greedy, kept per model / layout / release.

    python tools/maya_golden.py --config maya-<model>.json --capture --text-ref v1.1.76   # save this build's answers
    python tools/maya_golden.py --config maya-<model>.json --check                         # a build against them
    python tools/maya_golden.py --list

A capture runs ./maya.sh --bench once (its quiet-machine check, GPU lock and warm-up included) with --bench-tokens,
and stores every request's answer - the token ids, their hash and the first words as text - in
data/golden/<machine>/<model>-<layout>.json: one small file per model and layout, written on the machine that ran it
(nothing is shipped for a machine that is not ours; the folder is not in git).  --check runs the same requests on the
build in this folder and prints exactly which request differs, and from which token.  --from-tokens <file> does
either from a --bench-tokens file some other run wrote (the nightly's --compare run): no second engine start.

What makes two runs comparable is checked, not assumed:
  - the prompts: the 2K / 8K prompts are cut from the docs, so a set remembers the git ref they came from
    (--text-ref, passed on as --bench-text-ref) and each request's input hash; a request whose input changed is
    listed as "not compared", never as a difference;
  - the saved per-PC choices: some models' greedy answers legitimately depend on a measurement the engine saved on
    this PC (an expert profile, a route choice).  A set is keyed by the layout, the config's own settings and the hash
    of those files; a check takes the set with the same key, else the newest one - and then says which file changed,
    so a difference is read as "maybe the new measurement", not as an engine bug.

Exit code: 0 equal, 1 something differs, 2 nothing to compare with (no saved set) or the run failed.
The pure parts (the file's shape, picking a set, the comparison) need no GPU: tools/test_golden.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maya_bench as MB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEEP_SETS = 12           # sets kept a file (the oldest dropped): a few releases x a few saved-choice states


# ------------------------------------------------------------------------------------------------ pure parts
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def machine_name() -> str:
    return slug(socket.gethostname().split(".")[0]) or "this-pc"


def auto_layout(tokens: dict) -> str:
    """A layout's default name: its GPUs and context ('2x-tesla-v100-pcie-32gb-32k')."""
    return slug(f"{tokens.get('gpus', 'gpu')}-{int(tokens.get('context', 0)) // 1024}k")


def golden_path(folder: Path, tokens: dict, layout: str = "") -> Path:
    return folder / f"{slug(tokens.get('model', 'model'))}-{slug(layout) or auto_layout(tokens)}.json"


def config_hash(cfg: dict) -> str:
    """A hash of what a config asks the engine for (its arguments and environment - not the port, host or keys)."""
    core = {"args": cfg.get("args") or [], "env": cfg.get("env") or {}, "backend": cfg.get("backend", "cuda")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def choices_hash(choices: dict) -> str:
    return hashlib.sha256(json.dumps(choices or {}, sort_keys=True).encode()).hexdigest()[:16]


def ids_hash(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def make_set(tokens: dict, cfg: dict) -> dict:
    """A --bench-tokens file -> one saved set.  Rounds of this build that differ from each other (an engine that
    varies between its own starts) are all kept: 'also' holds the other answers seen."""
    rounds = tokens["runs"]["this"]
    answers = {}
    for name, ids in rounds[0].items():
        also = []
        for r in rounds[1:]:
            if name in r and r[name] != ids and r[name] not in also:
                also.append(r[name])
        answers[name] = {"n": len(ids), "hash": ids_hash(ids), "head": (tokens.get("heads") or {}).get(name, ""),
                         "ids": ids, **({"also": also} if also else {})}
    return {"release": tokens.get("version", "?"), "commit": tokens.get("commit", "?"),
            "date": tokens.get("date", time.strftime("%Y-%m-%d")), "text_ref": tokens.get("text_ref", ""),
            "gpus": tokens.get("gpus", ""), "driver": tokens.get("driver", ""), "context": tokens.get("context", 0),
            "config_hash": config_hash(cfg), "choices": tokens.get("choices") or {},
            "choices_hash": choices_hash(tokens.get("choices")), "inputs": tokens.get("inputs") or {},
            "answers": answers}


def set_key(s: dict) -> tuple:
    return s.get("release"), s.get("config_hash"), s.get("choices_hash")


def add_set(doc: dict, new: dict) -> dict:
    """The file with `new` in it: a set of the same release, config and saved choices is replaced."""
    sets = [s for s in doc.get("sets", []) if set_key(s) != set_key(new)] + [new]
    return dict(doc, sets=sets[-KEEP_SETS:])


def pick_set(doc: dict, cfg_hash: str, ch_hash: str, release: str = ""):
    """The set to check against (None: none saved).  Newest first: the same config and saved choices; else the
    same config; else the newest (compare() then says what changed).  `release` narrows it to that release's sets."""
    sets = [s for s in doc.get("sets", []) if not release or s.get("release") == release.lstrip("v")]
    for cfg_h, ch_h in ((cfg_hash, ch_hash), (cfg_hash, None), (None, None)):
        for s in reversed(sets):
            if cfg_h in (None, s.get("config_hash")) and ch_h in (None, s.get("choices_hash")):
                return s
    return None


def changed_choices(saved: dict, now: dict) -> list:
    """The saved-choice files that differ between a set and this run."""
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def compare(s: dict, tokens: dict, cfg_hash: str) -> dict:
    """One saved set against a run's tokens -> {'status': 'PASS' / 'DIFF' / 'NOTHING', 'diffs': [...],
    'skipped': [...], 'notes': [...], 'compared': n}.  Every round of this build has to match (the saved answer or
    one of its 'also' answers)."""
    diffs, skipped, compared = [], [], 0
    rounds = tokens["runs"]["this"]
    now_in = tokens.get("inputs") or {}
    for name, a in s["answers"].items():
        if s.get("inputs", {}).get(name) != now_in.get(name):
            skipped.append(f"{name}: not compared - its prompt changed (docs edited? check with --text-ref "
                           f"{s.get('text_ref') or '<the ref it was saved with>'})")
            continue
        compared += 1
        for k, r in enumerate(rounds):
            tag = f" (round {k + 1})" if len(rounds) > 1 else ""
            if name not in r:
                diffs.append(f"{name}{tag}: missing in this run")
                continue
            hits = [MB.first_difference(v, r[name]) for v in [a["ids"]] + a.get("also", [])]
            if None in hits:
                continue
            i = max(hits)                        # (the saved answer it follows longest)
            saved = ([a["ids"]] + a.get("also", []))[hits.index(i)]
            diffs.append(f"{name}{tag}: differs from token {i} (saved {saved[i] if i < len(saved) else 'end'}, now "
                         f"{r[name][i] if i < len(r[name]) else 'end'}; {len(saved)} vs {len(r[name])} tokens)")
            break
    notes = []
    ch = changed_choices(s.get("choices") or {}, tokens.get("choices") or {})
    if ch:
        notes.append("saved per-PC choices changed since the capture: " + ", ".join(ch[:4]) +
                     (" ..." if len(ch) > 4 else "") +
                     (" - a new measurement can explain a difference; capture again if it is meant" if diffs
                      else " - the answers are still equal"))
    if s.get("config_hash") != cfg_hash:
        notes.append("the config's settings changed since the capture")
    status = "DIFF" if diffs else "PASS" if compared else "NOTHING"
    return {"status": status, "diffs": diffs, "skipped": skipped, "notes": notes, "compared": compared,
            "against": f"v{s.get('release')} {s.get('commit')} {s.get('date')}"}


def verdict_line(res: dict) -> str:
    """One line for a results table: 'PASS 10/10', 'DIFF decode 2: differs from token 117 ... (+1 more)'."""
    total = res["compared"] + len(res["skipped"])
    if res["status"] == "DIFF":
        more = f" (+{len(res['diffs']) - 1} more)" if len(res["diffs"]) > 1 else ""
        return f"DIFF {res['diffs'][0]}{more}"
    if res["status"] == "NOTHING":
        return f"NOTHING compared ({len(res['skipped'])} prompts changed)"
    return f"PASS {res['compared']}/{total}" + (" (saved choices changed)" if res["notes"] else "")


def report(res: dict, path: Path) -> list:
    out = [f"saved answers: {path.name} ({res['against']})"]
    out += ["  DIFF  " + d for d in res["diffs"]] + ["  skip  " + x for x in res["skipped"]]
    out += ["  note  " + n for n in res["notes"]]
    out.append(f"  {res['status']}: {res['compared']} request(s) compared, {len(res['diffs'])} differ, "
               f"{len(res['skipped'])} not compared")
    return out


# ------------------------------------------------------------------------------------------------ files and runs
def read_doc(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_doc(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def golden_dir(root: Path = ROOT) -> Path:
    return root / "data" / MB.GOLDEN_DIR / machine_name()


def bench_command(root: Path, config: Path, tokens_file: Path, text_ref: str = "", compare: str = "",
                  rounds: int = 2, wait_min: float = 10.0, smoke: bool = False) -> list:
    """The --bench run that writes a tokens file (this Python, this folder's maya.py).  smoke: the short run
    (--bench-smoke: two short answers, no 2K / 8K prompts, no other build)."""
    cmd = [sys.executable, str(root / "maya.py"), "--config", str(config), "--bench", "--bench-tokens",
           str(tokens_file), "--bench-wait", f"{wait_min:g}"]
    if smoke:
        return cmd + ["--bench-smoke"]
    if text_ref:
        cmd += ["--bench-text-ref", text_ref]
    if compare:
        cmd += ["--compare", compare, "--bench-rounds", str(rounds)]
    return cmd


def run_bench(root: Path, config: Path, text_ref: str, wait_min: float, smoke: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "tokens.json"
        r = subprocess.run(bench_command(root, config, f, text_ref, wait_min=wait_min, smoke=smoke), cwd=str(root))
        if r.returncode or not f.exists():
            raise RuntimeError(f"the --bench run failed (exit {r.returncode})")
        return json.loads(f.read_text(encoding="utf-8"))


def saved_text_ref(folder: Path, config: Path, layout: str = "") -> str:
    """The git ref the saved set of this config (and layout) cut its prompts from: a check asks the same."""
    ref = ""
    for p in sorted(folder.glob("*.json")):
        doc = read_doc(p)
        if doc.get("config") == str(Path(config).resolve()) and (not layout or doc.get("layout") == slug(layout)):
            ref = next((s.get("text_ref", "") for s in reversed(doc.get("sets", []))), ref)
    return ref


def capture(tokens: dict, cfg: dict, folder: Path, layout: str = "") -> tuple:
    path = golden_path(folder, tokens, layout)
    new = make_set(tokens, cfg)
    doc = read_doc(path) or {"model": tokens.get("model", ""), "machine": machine_name(),
                             "layout": slug(layout) or auto_layout(tokens)}
    doc["config"] = str(Path(tokens["config"]).resolve()) if tokens.get("config") else ""
    write_doc(path, add_set(doc, new))
    return path, new


def check(tokens: dict, cfg: dict, folder: Path, layout: str = "", release: str = ""):
    """(the comparison or None when there is no saved set, the golden file's path)."""
    path = golden_path(folder, tokens, layout)
    s = pick_set(read_doc(path), config_hash(cfg), choices_hash(tokens.get("choices")), release)
    return (compare(s, tokens, config_hash(cfg)) if s else None), path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saved expected answers of --bench's fixed requests.")
    ap.add_argument("--config", type=Path, help="the installed config (the one the setup wrote)")
    ap.add_argument("--capture", action="store_true", help="save this build's answers")
    ap.add_argument("--check", action="store_true", help="compare this build's answers with the saved ones")
    ap.add_argument("--list", action="store_true", help="the saved sets on this machine")
    ap.add_argument("--from-tokens", type=Path, metavar="FILE", help="use a --bench-tokens file instead of a run")
    ap.add_argument("--text-ref", default="", help="the git ref the 2K / 8K prompts are cut from (a release tag); "
                                                   "--check without it uses the saved set's")
    ap.add_argument("--layout", default="", help="a name for this layout (default: its GPUs and context)")
    ap.add_argument("--release", default="", help="--check: against this release's set (default: the newest)")
    ap.add_argument("--dir", type=Path, default=None, help="the folder of this machine's files "
                                                           "(default data/golden/<machine>)")
    ap.add_argument("--wait", type=float, default=10.0, help="minutes to wait for a quiet machine (default 10)")
    ap.add_argument("--smoke", action="store_true", help="the short run (--bench-smoke: two short answers, no 2K / 8K "
                                                         "prompts) - for a model too large for its box to run the "
                                                         "standard; give it its own --layout name")
    a = ap.parse_args(argv)
    folder = a.dir or golden_dir()
    if a.list:
        for p in sorted(folder.glob("*.json")):
            for s in read_doc(p).get("sets", []):
                print(f"{p.name}: v{s.get('release')} {s.get('commit')} {s.get('date')} "
                      f"{len(s.get('answers', {}))} answers, prompts from {s.get('text_ref') or 'the docs then'}, "
                      f"choices {s.get('choices_hash')}")
        return 0
    if not (a.capture or a.check) or (not a.config and not a.from_tokens):
        ap.error("give --capture or --check, with --config (or --from-tokens)")
    try:
        if a.from_tokens:
            tokens = json.loads(a.from_tokens.read_text(encoding="utf-8"))
        else:
            ref = a.text_ref or (saved_text_ref(folder, a.config, a.layout) if a.check else "")
            tokens = run_bench(ROOT, a.config, ref, a.wait, a.smoke)
        cfg = json.loads(Path(a.config or tokens["config"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"maya_golden: {e}", file=sys.stderr)
        return 2
    if a.capture:
        path, s = capture(tokens, cfg, folder, a.layout)
        print(f"saved: {path} - v{s['release']} {s['commit']}, {len(s['answers'])} answers, prompts from "
              f"{s['text_ref'] or 'the docs of this folder (give --text-ref <release tag> to pin them)'}")
        return 0
    res, path = check(tokens, cfg, folder, a.layout, a.release)
    if res is None:
        print(f"no saved answers for this model and layout yet ({path}): run --capture on the release first")
        return 2
    print("\n".join(report(res, path)))
    return {"PASS": 0, "DIFF": 1}.get(res["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
