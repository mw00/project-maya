"""tools/maya_bench.py - the speed-test standard behind ./maya.sh --bench (Windows: START-MAYA.bat --bench).

One engine start per round and every measurement inside that one session: a warm-up answer (a 2K prompt and a short
answer: the prompt path and the decode path both warm), 3 answers of 256 new tokens (greedy, thinking off) and the 2K
and 8K prompts 3 times each (different openings: nothing reused).  Medians with min-max, what drafting chose, the
wall time of each phase, and one result line (machine, GPU, driver, model, config, commit, release, date) appended to
maya-bench-results.txt.  --compare: the same on this build and another one (a folder or a git ref), alternated
A, B, A, B on the same model files, config and learned files (copied, never learned again): exactness first, then
the speed medians and the difference.

The pure parts (parsing, medians, the result line, the quiet-machine verdict, the comparison) have no GPU, network or
engine dependency: tools/test_bench.py tests them.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PROMPT_SIZES = (2048, 8192)
DECODE_TOKENS = 256
REPEATS = 3
WARM_TOKENS = 32
MIN_ANSWER = 64          # an answer shorter than this (the model stopped early) is listed, not in the decode median
RESULTS_FILE = "maya-bench-results.txt"
LEARNED_MAX = 64 << 20   # a learned file named in the config (an expert profile ...): at most this; a model is larger
# the GPU-lock folders a test box keeps (a file a GPU: gpu0, gpu1 ... or radeon / v100) and the "too hot" flags;
# MAYA_GPU_LOCKS / MAYA_HOT_FLAG add one each
LOCK_DIRS = ("~/.maya/locks", "~/hy4/locks")
HOT_FLAGS = ("~/.maya/hot", "~/.jupiter-hot")


# ------------------------------------------------------------------------------------------------ pure parts
def median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def stats(xs) -> tuple:
    """(median, min, max) of a list of speeds; zeros for none."""
    return (median(xs), min(xs), max(xs)) if xs else (0.0, 0.0, 0.0)


def fmt_stats(s, digits=1) -> str:
    m, lo, hi = s
    return f"{m:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def gen_line(n_new: int, ids, multi: bool) -> str:
    """The GEN line: an engine that runs several conversations (READY ... multi=<n>) takes a request with its id."""
    return f"GEN {n_new} temperature=0{' rid=0' if multi else ''} " + ",".join(map(str, ids)) + "\n"


def reply_fields(line: str, multi: bool):
    """A DONE / ERR line's fields as a one-conversation engine writes them (several conversations: DONE <rid> ...,
    the id dropped); None for another request's line or one that is neither."""
    f = line.split()
    if not f or f[0] not in ("DONE", "ERR"):
        return None
    if multi and len(f) > 1 and f[1].isdigit():
        if f[1] != "0":
            return None
        f = f[:1] + f[2:]
    return f


def token_of(line: str, multi: bool):
    """The token id of a 'T <id>' line ('T <rid> <id>' from a several-conversation engine, ours rid 0); else None."""
    if not line.startswith("T "):
        return None
    f = line.split()
    try:
        if multi and len(f) >= 3:
            return int(f[2]) if f[1] == "0" else None
        return int(f[-1])
    except ValueError:
        return None


def parse_done(f) -> dict:
    """DONE <produced> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafted> <reused> ... (every
    engine's serve contract); missing trailing fields read as 0."""
    def num(i, kind=float):
        try:
            return kind(f[i])
        except (IndexError, ValueError):
            return kind(0)
    return {"produced": num(1, int), "prompt": num(2, int), "prompt_ms": num(3), "decode_ms": num(4),
            "finish": f[5] if len(f) > 5 else "", "d_acc": num(6, int), "d_off": num(7, int), "reused": num(8, int)}


def decode_speed(d: dict) -> float:
    return d["produced"] / d["decode_ms"] * 1000.0 if d["decode_ms"] > 0 else 0.0


def prompt_speed(d: dict) -> float:
    read = d["prompt"] - d["reused"]
    return read / d["prompt_ms"] * 1000.0 if d["prompt_ms"] > 0 and read > 0 else 0.0


def decode_samples(answers) -> list:
    """The decode speeds the median covers: answers of MIN_ANSWER tokens or more (a short one - the model ended
    early - says little about the speed); all of them when none is."""
    full = [a for a in answers if a["produced"] >= MIN_ANSWER] or list(answers)
    return [decode_speed(a) for a in full if a["decode_ms"] > 0]


DRAFT_LINE = re.compile(r"^(?:INFO )?drafting: (.+?)(?: - |, \d|$)")


def drafting_choice(args, env: dict, engine_lines, answers) -> str:
    """What drafting the engine used: its own 'drafting: X - why' line (--drafting auto's choice), else the config's
    setting, else what its log shows; with the share of drafted tokens accepted when the engine counts them."""
    said = ""
    for x in engine_lines:
        m = DRAFT_LINE.match(x.strip())
        if m:
            said = m.group(1).strip()
    a = [str(x) for x in args or []]
    flag = lambda k: a[a.index(k) + 1] if k in a[:-1] else None   # noqa: E731
    if not said:
        for k, name in (("--drafting", "drafting"), ("--draft", "draft"), ("--dspark", "DSpark"),
                        ("--speculative", "speculative")):
            if flag(k) is not None:
                said = f"{name} {flag(k)}"
                break
    if not said:
        text = "\n".join(engine_lines).lower()
        said = ("MTP" if "glm mtp:" in text or "mtp decode" in text else
                "DFlash" if "dflash" in text else "DSpark" if "dspark" in text else "none seen")
    acc = sum(x["d_acc"] for x in answers)
    off = sum(x["d_off"] for x in answers)
    return said + (f", {100.0 * acc / off:.0f}% of drafts accepted" if off > 0 and acc <= off else "")


def first_difference(a, b):
    """Where two token lists first differ (None: equal)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def exactness(ref: dict, other: dict) -> list:
    """{request name: tokens} vs another run's: [] when every greedy answer is equal, else one line a request that
    differs ('decode 2: token 117 of 256')."""
    out = []
    for k in ref:
        if k not in other:
            out.append(f"{k}: missing in the other run")
            continue
        i = first_difference(ref[k], other[k])
        if i is not None:
            out.append(f"{k}: differs from token {i} ({len(ref[k])} vs {len(other[k])} tokens)")
    return out


def pct(new: float, base: float) -> str:
    return f"{(new - base) / base * 100.0:+.1f}%" if base > 0 else "n/a"


def record_line(r: dict) -> str:
    """The one-line result record (the standard's 'What a result line records')."""
    pr = " / ".join(f"{n // 1024}K {r['prompt'][n][0]:.0f}" for n in sorted(r["prompt"]))
    return " | ".join([r["machine"], r["gpus"], r["driver"], r["model"], r["config"], r["commit"], f"v{r['version']}",
                       f"decode {fmt_stats(r['decode'])}", f"prompt {pr or 'n/a'}", r["date"]] +
                      ([r["note"]] if r.get("note") else []))


RECORD_DECODE = re.compile(r"^decode ([\d.]+) \(([\d.]+)-([\d.]+)\)$")


def parse_record(line: str) -> dict:
    """A result line back into its fields (for tools reading maya-bench-results.txt)."""
    p = [x.strip() for x in line.strip().split(" | ")]
    if len(p) < 10:
        raise ValueError("not a --bench result line")
    m = RECORD_DECODE.match(p[7])
    if not m or not p[8].startswith("prompt "):
        raise ValueError("not a --bench result line")
    prompt = {}
    for part in p[8][len("prompt "):].split(" / "):
        if part != "n/a":
            k, v = part.split()
            prompt[int(k[:-1]) * 1024] = float(v)
    return {"machine": p[0], "gpus": p[1], "driver": p[2], "model": p[3], "config": p[4], "commit": p[5],
            "version": p[6].lstrip("v"), "decode": tuple(float(x) for x in m.groups()), "prompt": prompt,
            "date": p[9], "note": " | ".join(p[10:])}


def quiet_reasons(busy_gpus, locks, load, hot) -> list:
    """Why this machine is not quiet enough for a speed number (the standard's 'Before every run'): another program
    on the GPU(s), a GPU lock someone holds, CPU load 2 or more, a 'too hot' flag.  [] = quiet."""
    out = [f"GPU {g} is in use ({why})" for g, why in busy_gpus]
    out += [f"GPU lock {name} is held: {who or '(no note)'}" for name, who in locks]
    if load is not None and load >= 2.0:
        out.append(f"CPU load {load:.1f} (needs < 2)")
    out += [f"{p} exists (the machine is too hot)" for p in hot]
    return out


def gpus_busy_nvidia(apps_csv: str, util_csv: str, ours) -> list:
    """nvidia-smi's compute apps (gpu index,pid,name,MiB) and readings (index,utilization %) -> [(index, why)] of
    our GPUs (ours: index list, or None for all) another program uses."""
    out = []
    for row in apps_csv.splitlines():
        f = [x.strip() for x in row.split(",")]
        if len(f) >= 3 and f[0].isdigit() and (ours is None or int(f[0]) in ours):
            out.append((int(f[0]), f"pid {f[1]} {Path(f[2]).name}"))
    seen = {g for g, _ in out}
    for row in util_csv.splitlines():
        f = [x.strip() for x in row.split(",")]
        if len(f) >= 2 and f[0].isdigit() and f[1].isdigit() and int(f[1]) >= 20 and int(f[0]) not in seen and \
                (ours is None or int(f[0]) in ours):
            out.append((int(f[0]), f"{f[1]}% busy"))
    return out


def lock_names(backend: str, gpus) -> list:
    """The lock files our run takes: gpu<N> for each NVIDIA GPU it uses, radeon on AMD."""
    if backend == "hip":
        return ["radeon"]
    return [f"gpu{i}" for i in (gpus or [0])]


# ------------------------------------------------------------------------------------------------ the machine
def _out(cmd, timeout=15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lock_dirs() -> list:
    ds = list(LOCK_DIRS) + ([os.environ["MAYA_GPU_LOCKS"]] if os.environ.get("MAYA_GPU_LOCKS") else [])
    return [Path(d).expanduser() for d in ds if Path(d).expanduser().is_dir()]


def held_locks(names, mine="") -> list:
    out = []
    for d in lock_dirs():
        for n in names:
            p = d / n
            if p.exists():
                who = p.read_text(errors="replace").strip().splitlines()[:1] if p.is_file() else []
                if not (mine and who and who[0] == mine):
                    out.append((str(p), who[0] if who else ""))
    return out


def busy_threads(a, b) -> float:
    """Two /proc/stat 'cpu' lines a second apart -> how many CPU threads were busy on average (the load now)."""
    x, y = [int(v) for v in a.split()[1:]], [int(v) for v in b.split()[1:]]
    d = [q - p for p, q in zip(x, y)]
    total = sum(d[:8])
    idle = d[3] + (d[4] if len(d) > 4 else 0)            # (idle + iowait)
    return (total - idle) / total * (os.cpu_count() or 1) if total > 0 else 0.0


def cpu_load_now():
    """The CPU load over the last second (busy threads), not the 1-minute average - that one still counts a model
    load that just ended for a minute or two; the average where /proc/stat is missing, None on Windows."""
    try:
        a = Path("/proc/stat").read_text().splitlines()[0]
        time.sleep(1.0)
        return busy_threads(a, Path("/proc/stat").read_text().splitlines()[0])
    except OSError:
        pass
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None                          # (Windows: no load figure - the GPU check stands)


def machine_quiet(backend: str, gpus, lock_note="") -> list:
    if backend == "hip":
        busy = []
        txt = _out(["rocm-smi", "--showuse", "--json"])
        try:
            for card, v in json.loads(txt or "{}").items():
                u = next((int(float(x)) for k, x in v.items() if "GPU use" in k), 0)
                if u >= 20:
                    busy.append((card, f"{u}% busy"))
        except (ValueError, AttributeError):
            pass
    elif backend == "metal":
        busy = []
    else:
        # (compute apps name their GPU by bus id: mapped to nvidia-smi's index first)
        idx, util = {}, []
        for row in _out(["nvidia-smi", "--query-gpu=index,pci.bus_id,utilization.gpu",
                         "--format=csv,noheader,nounits"]).splitlines():
            f = [x.strip() for x in row.split(",")]
            if len(f) >= 3:
                idx[f[1].lower()] = f[0]
                util.append(f"{f[0]},{f[2]}")
        apps = []
        for row in _out(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name",
                         "--format=csv,noheader"]).splitlines():
            f = [x.strip() for x in row.split(",")]
            if len(f) >= 3 and f[0].lower() in idx:
                apps.append(",".join([idx[f[0].lower()], f[1], f[2]]))
        busy = gpus_busy_nvidia("\n".join(apps), "\n".join(util), gpus)
    load = cpu_load_now()
    hot = [p for p in list(HOT_FLAGS) + ([os.environ["MAYA_HOT_FLAG"]] if os.environ.get("MAYA_HOT_FLAG") else [])
           if Path(p).expanduser().exists()]
    return quiet_reasons(busy, held_locks(lock_names(backend, gpus), lock_note), load, hot)


def wait_quiet(backend, gpus, wait_min: float, say, lock_note="") -> list:
    """Wait (up to wait_min minutes) for a quiet machine; [] when quiet, else the reasons it still is not."""
    t_end = time.time() + wait_min * 60
    told = None
    while True:
        why = machine_quiet(backend, gpus, lock_note)
        if not why or time.time() >= t_end:
            return why
        kind = [re.sub(r"[\d.]+", "#", x) for x in why]     # (said again when the reasons change, not their numbers)
        if kind != told:
            say("  waiting for a quiet machine: " + "; ".join(why))
            told = kind
        time.sleep(15)


def take_locks(backend, gpus, note) -> list:
    taken = []
    for d in lock_dirs():
        for n in lock_names(backend, gpus):
            p = d / n
            if not p.exists():
                p.write_text(note + "\n")
                taken.append(p)
    return taken


def release_locks(taken, note):
    for p in taken:
        try:
            if p.read_text(errors="replace").strip().splitlines()[:1] == [note]:
                p.unlink()
        except OSError:
            pass


def gpu_summary(backend: str, found) -> tuple:
    """('3x Tesla V100-PCIE-32GB', driver) of the GPUs a run uses."""
    names = [g.get("name", "?") for g in found]
    if not names:
        return ("Apple GPU" if backend == "metal" else "?"), "?"
    groups = []
    for n in dict.fromkeys(names):
        c = names.count(n)
        groups.append(f"{c}x {n}" if c > 1 else n)
    drv = next((str(g["driver"]) for g in found if g.get("driver")), "")
    if not drv and backend == "hip":
        for p in ("/opt/rocm/.info/version",):
            if Path(p).exists():
                drv = "ROCm " + Path(p).read_text().strip()
    return ", ".join(groups), drv or "?"


def commit_of(root: Path) -> str:
    c = _out(["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"]).strip()
    if c and _out(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]).strip():
        c += "+changes"
    return c or "?"


# ------------------------------------------------------------------------------------------------ one session
class Session:
    """One engine process: start, READY, GEN requests (tokens + DONE), QUIT - every phase timed."""

    def __init__(self, cmd, env, cwd, log_path: Path):
        self.log_path = log_path
        self.log = open(log_path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                  env=env, cwd=cwd, bufsize=1)
        self.multi = False
        self.lines = []                      # (stdout lines before READY: an engine's own drafting choice ...)
        while True:
            line = self.p.stdout.readline()
            if not line:
                self.close()
                raise RuntimeError(f"the engine stopped while loading (its log: {log_path})")
            if line.startswith("READY"):
                self.multi = any(w.startswith("multi=") for w in line.split())
                break
            self.lines.append(line.rstrip())
        self.load_s = time.time() - self.t0

    def gen(self, ids, n_new):
        """(DONE fields as a dict, the answer's tokens); raises on ERR."""
        self.p.stdin.write(gen_line(n_new, ids, self.multi))
        self.p.stdin.flush()
        toks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"the engine stopped during the benchmark (its log: {self.log_path})")
            t = token_of(line, self.multi)
            if t is not None:
                toks.append(t)
                continue
            f = reply_fields(line, self.multi)
            if f is None:
                continue
            if f[0] == "ERR":
                raise RuntimeError("the engine refused a benchmark request: " + " ".join(f[1:]))
            return parse_done(f), toks

    def close(self) -> float:
        t = time.time()
        try:
            if self.p.poll() is None:
                self.p.stdin.write("QUIT\n")
                self.p.stdin.flush()
                self.p.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        if self.p.poll() is None:            # (a slow teardown costs the bench nothing it measures)
            self.p.kill()
            self.p.wait()
        self.log.close()
        return time.time() - t


def plan_prompts(doc, ctx: int, sizes=PROMPT_SIZES, repeats=REPEATS):
    """The prompts: for each size, `repeats` slices of the docs' tokens with different openings (nothing reused
    between them, nor with the warm-up's, which opens at the end) - sizes the config's context or the docs cannot
    hold are left out."""
    out = {}
    step = 997
    for n in sizes:
        if n + 64 > ctx or n + step * (len(sizes) * repeats) > len(doc):
            continue
        out[n] = [doc[(len(out) * repeats + k) * step:(len(out) * repeats + k) * step + n] for k in range(repeats)]
    warm = doc[-min(sizes[0], len(doc)):]
    return out, warm


def run_session(start, chats, prompts, warm, label, say) -> dict:
    """Everything one round measures, in one engine session: the warm-up answer, the decode answers and the prompts.
    Returns the samples, the greedy tokens of every request (exactness) and the wall time of each phase."""
    ph = {}
    s = Session(*start)
    ph["load"] = s.load_s
    say(f"  {label}loaded in {s.load_s:.0f} s")
    try:
        t = time.time()
        _, wtok = s.gen(warm, WARM_TOKENS)               # (warm-up: a 2K prompt and a short answer)
        ph["warm-up"] = time.time() - t
        answers, toks = [], {"warm-up": wtok}
        t = time.time()
        for i, ids in enumerate(chats):
            d, tk = s.gen(ids, DECODE_TOKENS)
            answers.append(d)
            toks[f"decode {i + 1}"] = tk
            say(f"  {label}decode  {decode_speed(d):6.1f} tokens/s  ({d['produced']} tokens)")
        ph["decode"] = time.time() - t
        t = time.time()
        reads, reused = {n: [] for n in prompts}, 0
        for k in range(max((len(v) for v in prompts.values()), default=0)):
            for n, slices in prompts.items():            # (2K, 8K, 2K, 8K ...: a slow drift touches both alike)
                d, tk = s.gen(slices[k], 1)
                reads[n].append(prompt_speed(d))
                reused += d["reused"]
                toks[f"prompt {n // 1024}K #{k + 1}"] = tk
                say(f"  {label}prompt  {reads[n][-1]:6.0f} tokens/s  ({n} tokens)")
        ph["prompts"] = time.time() - t
        lines = s.lines
    finally:
        ph["stop"] = s.close()
    lines = lines + s.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"answers": answers, "reads": reads, "tokens": toks, "phases": ph, "engine_lines": lines,
            "reused": reused}


# ------------------------------------------------------------------------------------------------ learned files
def learned_files(root: Path, cfg: dict) -> list:
    """The files an engine start reads and may rewrite (expert profiles, drafting and route choices, hot-expert
    scores): this build's data folder, and the files the config names that are small enough not to be a model."""
    out = [p for p in (root / "data").rglob("*") if p.is_file()] if (root / "data").is_dir() else []
    vals = [str(x) for x in cfg.get("args") or []] + [str(v) for v in (cfg.get("env") or {}).values()]
    for v in vals:
        try:
            p = Path(v)
            if p.is_absolute() and p.is_file() and p.stat().st_size <= LEARNED_MAX and p not in out:
                out.append(p)
        except (OSError, ValueError):
            pass
    return out


class LearnedState:
    """A snapshot of the learned files at the start: restored before each round (both builds start from the same
    state, nothing learned again) and at the end; a second build's data folder gets the same copies (its own files
    put back at the end)."""

    def __init__(self, root: Path, cfg: dict, snap_dir: Path):
        self.root, self.snap = root, snap_dir
        shutil.rmtree(snap_dir, ignore_errors=True)
        self.saved = {}
        for i, p in enumerate(learned_files(root, cfg)):
            dst = snap_dir / f"{i}-{p.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            self.saved[p] = dst
        self.others = {}                     # {other root: ({its data files at the start}, {file: bytes or None})}

    @staticmethod
    def _put(src: Path, dst: Path):
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    @staticmethod
    def _files(d: Path) -> set:
        return {p for p in d.rglob("*") if p.is_file()} if d.is_dir() else set()

    def _drop_new(self, d: Path, keep: set):
        """A file a run wrote that was not there at the start (a first start's measurement) goes too."""
        for p in self._files(d) - keep:
            try:
                p.unlink()
            except OSError:
                pass

    def restore(self, other: Path | None = None):
        """Put the snapshot back (this build's data folder and the config's files) - and, with `other`, the same
        copies into that build's data folder."""
        for p, s in self.saved.items():
            self._put(s, p)
        self._drop_new(self.root / "data", set(self.saved))
        if other is None:
            return
        start, backup = self.others.setdefault(other, (self._files(other / "data"), {}))
        mine = set()
        for p, s in self.saved.items():
            if self._under(p, self.root / "data"):
                q = other / p.relative_to(self.root)
                if q not in backup:
                    backup[q] = q.read_bytes() if q.exists() else None
                self._put(s, q)
                mine.add(q)
        self._drop_new(other / "data", start | mine)

    def adopt(self, cfg: dict):
        """Files the first round learned that the snapshot did not have (a first start's measurements) join it: no
        later round learns them again.  Files it rewrote keep their snapshot (each start begins the same)."""
        for p in learned_files(self.root, cfg):
            if p not in self.saved:
                dst = self.snap / f"{len(self.saved)}-{p.name}"
                shutil.copy2(p, dst)
                self.saved[p] = dst
        return self

    def finish(self):
        """The learned files as they were before the bench, in both builds."""
        self.restore()
        for other, (start, backup) in self.others.items():
            for q, data in backup.items():
                if data is None:
                    q.unlink(missing_ok=True)
                else:
                    q.write_bytes(data)
            self._drop_new(other / "data", start)
        shutil.rmtree(self.snap, ignore_errors=True)

    @staticmethod
    def _under(p: Path, d: Path) -> bool:
        try:
            p.relative_to(d)
            return True
        except ValueError:
            return False


# ------------------------------------------------------------------------------------------------ the other build
def other_root(spec: str, here: Path, exe: Path, say) -> Path:
    """--compare's other build: a Maya folder (its engines built), or a git ref of this repository - checked out in
    .bench-builds/<commit> and its engine built there with this build's CMake settings (kept for the next time)."""
    p = Path(spec).expanduser()
    if p.is_dir() and (p / "maya.py").exists():
        return p.resolve()
    sha = _out(["git", "-C", str(here), "rev-parse", "--verify", "--quiet", spec + "^{commit}"]).strip()
    if not sha:
        raise RuntimeError(f"--compare {spec}: neither a Maya folder nor a git ref of {here}")
    wt = here / ".bench-builds" / sha[:12]
    if not (wt / "maya.py").exists():
        say(f"  checking out {spec} ({sha[:8]}) in {wt}")
        r = subprocess.run(["git", "-C", str(here), "worktree", "add", "--detach", str(wt), sha],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"git worktree add failed: {r.stderr.strip()[-300:]}")
    rel_build = exe.parent.relative_to(here) if LearnedState._under(exe, here) else Path("build")
    bdir = wt / rel_build
    if not (bdir / exe.name).exists():
        cache = exe.parent / "CMakeCache.txt"
        opts = cmake_options(cache.read_text(errors="replace") if cache.exists() else "")
        t = time.time()
        say(f"  building {exe.name} of {sha[:8]} (once; kept in {bdir}) ...")
        gen = ["-G", "Ninja"] if shutil.which("ninja") else []
        for cmd in (["cmake", *gen, "-S", str(wt), "-B", str(bdir), *opts],
                    ["cmake", "--build", str(bdir), "--target", exe.stem, "-j", str(os.cpu_count() or 8)]):
            with open(wt / "bench-build.log", "a") as log:
                if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode:
                    raise RuntimeError(f"building {spec} failed: {wt / 'bench-build.log'}")
        say(f"  built in {time.time() - t:.0f} s")
    return wt


CMAKE_KEEP = re.compile(r"^((?:CMAKE_BUILD_TYPE|CMAKE_(?:CUDA|HIP)_ARCHITECTURES|CMAKE_(?:CUDA|HIP|CXX|C)_COMPILER|"
                        r"CMAKE_CUDA_HOST_COMPILER|(?:MAYA|STRATA|GGML)_[A-Z0-9_]+)):[A-Z]+=(.*)$")


def cmake_options(cache_text: str) -> list:
    """This build's CMake settings as -D options for another checkout (the older STRATA_ names given too, for a
    ref from before the rename)."""
    out = []
    for line in cache_text.splitlines():
        m = CMAKE_KEEP.match(line.strip())
        if not m or m.group(2) == "" or m.group(1).endswith(("_AR", "_RANLIB")):
            continue
        k, v = m.groups()
        out.append(f"-D{k}={v}")
        if k.startswith("MAYA_"):
            out.append(f"-DSTRATA_{k[5:]}={v}")
    return out


def map_cfg(cfg: dict, here: Path, there: Path) -> dict:
    """The same config for the other build: its engine binary (same place in its folder) and folder."""
    c = json.loads(json.dumps(cfg))
    exe = Path(c["exe"])
    c["exe"] = str(there / exe.relative_to(here)) if LearnedState._under(exe, here) else str(there / "build" / exe.name)
    if not c.get("cwd") or Path(c["cwd"]).resolve() == here.resolve():
        c["cwd"] = str(there)
    return c


def start_command(root: Path, cfg: dict, cfg_file: Path):
    """[cmd, env] the build in `root` starts this config's engine with - its own serve/server.py (engine_args,
    child_env), run as a separate Python so its modules never mix with this build's."""
    code = ("import json,sys; sys.path[:0]=[sys.argv[1], sys.argv[1]+'/tools']; import serve.server as SV; "
            "c=json.load(open(sys.argv[2])); print(json.dumps([[c['exe'],'--serve']+SV.engine_args(c),"
            "SV.child_env(c)]))")
    r = subprocess.run([sys.executable, "-c", code, str(root), str(cfg_file)], capture_output=True, text=True,
                       cwd=str(root))
    if r.returncode:
        raise RuntimeError(f"{root}: its start command could not be read: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------------ the result
def summarize(runs, prompts) -> dict:
    """Rounds of one build -> its decode / prompt samples and medians."""
    answers = [a for r in runs for a in r["answers"]]
    reads = {n: [x for r in runs for x in r["reads"].get(n, []) if x > 0] for n in prompts}
    return {"decode": stats(decode_samples(answers)), "prompt": {n: stats(v) for n, v in reads.items() if v},
            "answers": answers, "short": sum(a["produced"] < MIN_ANSWER for a in answers)}


def phase_line(runs) -> str:
    keys = ("load", "warm-up", "decode", "prompts", "stop")
    tot = {k: sum(r["phases"].get(k, 0.0) for r in runs) for k in keys}
    return ", ".join(f"{k} {tot[k]:.0f} s" for k in keys) + f" (total {sum(tot.values()):.0f} s)"


def compare_line(a: dict, b: dict, a_name: str, b_name: str, exact: list) -> str:
    """The one-line A/B summary: exactness, then this build's speed against the other's."""
    parts = [f"compare {b_name} vs {a_name}", "exact: yes" if not exact else f"exact: NO ({len(exact)} differ)",
             f"decode {b['decode'][0]:.1f} vs {a['decode'][0]:.1f} {pct(b['decode'][0], a['decode'][0])}"]
    for n in sorted(set(a["prompt"]) & set(b["prompt"])):
        parts.append(f"{n // 1024}K {b['prompt'][n][0]:.0f} vs {a['prompt'][n][0]:.0f} "
                     f"{pct(b['prompt'][n][0], a['prompt'][n][0])}")
    return " | ".join(parts)
