"""tools/maya_changed.py <base-ref> - what changed since a release, and so what a release has to test.

    python tools/maya_changed.py v1.1.76            # the plan for HEAD (and uncommitted changes) against v1.1.76
    python tools/maya_changed.py v1.1.76 --json     # the same for a script

Every changed path is mapped to the engines (and their models) it can reach:
  - a file an engine is built from - the CMake target's sources, the libraries it links and every "#include" from
    there on (read from CMakeLists.txt and the sources, nothing listed by hand) - touches that engine;
  - a file in an engine's own folder (src/ik2 -> Inkling, src/hy4 -> Hy4, src/ds4 -> DeepSeek, src/mm3 -> MiniMax,
    the GLM / Qwen engine's folders) touches it too, whatever the scan says;
  - the build files, third_party, the server (serve/), maya.py and the setup, and tools the start imports touch all;
  - a Python or data file named after one model (tools/hy4/..., serve/ds4_vision.py) touches that model;
  - tests, docs and CI files touch no model (the tests themselves are in the plan).
The plan: FULL for the touched models (docs/RELEASE-CHECKS.md: --bench --compare and the saved answers on every
machine and layout), SMOKE for the others (last night's line, or one saved-answer check), the engines to build, the
Python suites (always: they are cheap) and the C++ tests that changed.  Anything not understood counts as "all".

The pure parts (the CMake reading, the include scan, the mapping, the plan's text) need no git: tools/test_changed.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# engine binary -> (its own source folders, the models it runs)
ENGINES = {
    "maya": (("src/core", "src/program", "src/kernels", "src/prefill", "src/spec", "src/ngram", "src/plan",
              "src/platform", "src/artifact"), ("GLM", "Qwen")),
    "maya-ds4": (("src/ds4",), ("DeepSeek",)),
    "maya-mm3": (("src/mm3",), ("MiniMax",)),
    "maya-hy4": (("src/hy4",), ("Hy4",)),
    "maya-ik2": (("src/ik2",), ("Inkling",)),
}
MODELS = tuple(m for _, ms in ENGINES.values() for m in ms)
# how a file or folder name names a model: a word starting with one of these (tools/hy4/pack.py,
# serve/ds4_vision.py, data/expert-profile-ik2mini.bin)
NAME_TOKENS = (("DeepSeek", ("ds4", "deepseek")), ("MiniMax", ("mm3", "minimax")), ("Hy4", ("hy4",)),
               ("Inkling", ("ik2", "inkling")), ("Qwen", ("qwen",)), ("GLM", ("glm",)))
# what every model's start runs through
ALL_FILES = ("maya.py", "setup.py", "chat.py", "maya.sh", "setup.sh", "START-MAYA.bat", "CMakeLists.txt", "VERSION")
ALL_DIRS = ("cmake/", "third_party/", "serve/", "tools/", "include/", "src/", "data/")
NO_MODEL_DIRS = ("docs/", "assets/", "ref/", "bench/", "docker/", ".github/")
# this pipeline's own tools: no engine start reads them
PIPELINE_TOOLS = ("tools/maya_changed.py", "tools/maya_golden.py", "tools/maya_nightly.py")
PY_SUITES = {"serve": 'python -m unittest discover -s serve -p "test_*.py" -t .',
             "tools": 'python -m unittest discover -s tools -p "test_*.py"'}


# ------------------------------------------------------------------------------------------------ CMakeLists.txt
def cmake_commands(text: str):
    """(name, [arguments]) of every command in a CMakeLists.txt (comments dropped, nested brackets kept whole)."""
    text = re.sub(r"#[^\n]*", "", text)
    i = 0
    for m in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*\(", text):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        i = j
        yield m.group(1).lower(), text[m.end():j - 1].split()


def cmake_targets(text: str) -> tuple:
    """({target: its source paths}, {target: the targets it links}) - set() variables holding sources followed."""
    src, link, var = {}, {}, {}
    is_path = re.compile(r"^(?:src|tests|bench|third_party|include)/[\w./+-]+\.\w+$")

    def paths(args):
        out = []
        for a in args:
            a = a.strip('"')
            m = re.fullmatch(r"\$\{(\w+)\}", a)
            if m:
                out += var.get(m.group(1), [])
            elif is_path.match(a):
                out.append(a)
        return out

    for name, args in cmake_commands(text):
        if not args:
            continue
        if name == "set":
            got = paths(args[1:])
            if got:
                var[args[0]] = var.get(args[0], []) + got if "${" + args[0] + "}" in args else got
        elif name == "list" and len(args) > 2 and args[0].upper() == "APPEND":
            var[args[1]] = var.get(args[1], []) + paths(args[2:])
        elif name in ("add_executable", "add_library", "target_sources"):
            src.setdefault(args[0], []).extend(paths(args[1:]))
        elif name == "target_link_libraries":
            link.setdefault(args[0], []).extend(a for a in args[1:] if re.fullmatch(r"[\w-]+", a))
    return src, link


def target_sources(target: str, src: dict, link: dict) -> set:
    """A target's sources with those of every library target it links (followed all the way)."""
    seen, todo, out = set(), [target], set()
    while todo:
        t = todo.pop()
        if t in seen:
            continue
        seen.add(t)
        out.update(src.get(t, []))
        todo += [x for x in link.get(t, []) if x in src]
    return out


# ------------------------------------------------------------------------------------------------ the include scan
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def include_closure(start, root: Path) -> set:
    """Every repository file reachable from `start` through #include "..." (looked up beside the including file,
    under include/, under src/ and in each src/<folder>: every place a target's include path can name)."""
    src_dirs = sorted(p for p in (root / "src").glob("*") if p.is_dir()) if (root / "src").is_dir() else []
    seen, todo = set(), [s for s in start]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        f = root / rel
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for inc in INCLUDE.findall(text):
            for base in [f.parent, root / "include", root / "src", *src_dirs]:
                c = (base / inc)
                if c.is_file():
                    try:
                        todo.append(c.resolve().relative_to(root.resolve()).as_posix())
                    except ValueError:
                        pass
                    break
    return seen


def engine_files(root: Path = ROOT) -> dict:
    """{engine: every file it is built from} - read from CMakeLists.txt and the sources."""
    cm = root / "CMakeLists.txt"
    src, link = cmake_targets(cm.read_text(encoding="utf-8", errors="replace")) if cm.exists() else ({}, {})
    return {e: include_closure(target_sources(e, src, link), root) for e in ENGINES}


# ------------------------------------------------------------------------------------------------ the mapping
def named_models(path: str) -> list:
    """The models a file or folder name names (tools/hy4/pack.py -> Hy4; ds4_os.hpp -> DeepSeek)."""
    words = [w for w in re.split(r"[^a-z0-9]+", path.lower()) if w]
    return [m for m, toks in NAME_TOKENS if any(w.startswith(toks) for w in words)]


def classify(path: str, built: dict) -> tuple:
    """A changed path -> (the models it touches, a kind: 'engine' / 'python' / 'data' / 'test' / 'none' / 'all',
    the engines to rebuild)."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if p in PIPELINE_TOOLS or name.startswith("test_") or p.startswith("tests/") or \
            re.search(r"_test\.(cpp|cu|mm)$", name):
        return [], "test", []
    if p.startswith(NO_MODEL_DIRS) or (("/" not in p) and (name.endswith(".md") or name.startswith("LICENSE")
                                                             or name.startswith("."))):
        return [], "none", []
    if p.startswith(("src/", "include/")):
        engines = [e for e, files in built.items() if p in files]
        engines += [e for e, (dirs, _) in ENGINES.items() if p.startswith(tuple(d + "/" for d in dirs))
                    and e not in engines]
        if p.startswith("src/metal/"):
            return [], "none", []                 # (the Mac engines: macos.yml builds and tests them)
        if not engines:                           # (not built into an engine as far as the scan sees: by its name)
            ms = named_models(p)
            engines = [e for e, (_, models) in ENGINES.items() if set(models) & set(ms)] or list(ENGINES)
        return [m for e in engines for m in ENGINES[e][1]], "engine", engines
    if p in ALL_FILES or p.startswith(("cmake/", "third_party/")):
        build = list(ENGINES) if p == "CMakeLists.txt" or p.startswith(("cmake/", "third_party/")) else []
        return list(MODELS), "all", build
    if p.startswith(("serve/", "tools/", "data/")):
        ms = named_models(p)
        return (ms or list(MODELS)), ("data" if p.startswith("data/") else "python"), []
    return list(MODELS), "all", list(ENGINES)     # (not understood: everything)


def plan(paths, built: dict) -> dict:
    """The test plan for a list of changed paths."""
    why = {m: [] for m in MODELS}
    builds, tests, quiet, docs = [], [], [], False
    for p in sorted(set(x.replace("\\", "/") for x in paths if x.strip())):
        models, kind, engines = classify(p, built)
        for m in models:
            why[m].append(p)
        builds += [e for e in engines if e not in builds]
        if kind == "test":
            tests.append(p)
        elif kind == "none":
            quiet.append(p)
            docs = docs or p == "README.md" or (p.startswith(("docs/", "bench/results/")) and p.endswith(".md"))
    notes = []
    if docs:
        notes.append("README / docs changed: --bench cuts its 2K / 8K prompts from them - compare prompt speeds and "
                     "saved answers with --bench-text-ref <the previous release> (the nightly does)")
    if any(p.startswith(".github/") for p in quiet):
        notes.append("CI files changed: look at the workflow run itself")
    if any(p.startswith("src/metal/") for p in quiet):
        notes.append("Mac engine files changed: macos.yml builds them; no Mac of ours tests them")
    return {"full": {m: v for m, v in why.items() if v}, "smoke": [m for m, v in why.items() if not v],
            "builds": [e for e in ENGINES if e in builds], "python": dict(PY_SUITES),
            "cpp_tests": [p for p in tests if not p.endswith(".py") and not p.endswith(".mjs")],
            "py_tests": [p for p in tests if p.endswith((".py", ".mjs"))], "notes": notes, "files": len(set(paths))}


def plan_text(pl: dict, base: str) -> list:
    """The plan as lines."""
    engine_of = {m: e for e, (_, ms) in ENGINES.items() for m in ms}
    out = [f"changed against {base}: {pl['files']} file(s)"]
    for m, files in pl["full"].items():
        shown = ", ".join(files[:3]) + (f" (+{len(files) - 3} more)" if len(files) > 3 else "")
        out.append(f"FULL   {m:<9} ({engine_of[m]}): {shown}")
    for m in pl["smoke"]:
        out.append(f"SMOKE  {m:<9} ({engine_of[m]}): nothing it runs changed")
    out.append("build: " + (", ".join(pl["builds"]) or "no engine (none of their code changed)"))
    out.append("Python suites (always): " + "; ".join(pl["python"].values()))
    if pl["cpp_tests"]:
        out.append("C++ tests changed (build and run them): " + ", ".join(pl["cpp_tests"]))
    out += ["note: " + n for n in pl["notes"]]
    out.append("FULL = --bench --compare " + base + " and the saved answers (maya_golden --check) on every machine "
               "and layout of that model; SMOKE = last night's PASS line, or one maya_golden --check.")
    return out


# ------------------------------------------------------------------------------------------------ git
def changed_paths(base: str, root: Path = ROOT) -> list:
    """Paths that differ between `base` and the working tree (committed or not), plus new untracked files."""
    def git(*a):
        r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()[-200:]}")
        return [x for x in r.stdout.splitlines() if x.strip()]
    return sorted(set(git("diff", "--name-only", base, "--") + git("ls-files", "--others", "--exclude-standard")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What changed since a release -> what to build and test.")
    ap.add_argument("base", help="the git ref to compare with (the previous release's tag)")
    ap.add_argument("--json", action="store_true", help="the plan as JSON")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        paths = changed_paths(a.base, a.root)
    except RuntimeError as e:
        print(f"maya_changed: {e}", file=sys.stderr)
        return 2
    pl = plan(paths, engine_files(a.root))
    print(json.dumps(pl, indent=1) if a.json else "\n".join(plan_text(pl, a.base)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
