"""Tune the GLM engine's CPU lane for this PC (./maya.sh --calibrate) - Strata's tools/calibrate.py, for GLM-5.3-Flash.

On one or two GPUs most of a token's experts are not in VRAM; the CPU LANE computes the ones in the RAM tier while the
GPU runs the rest.  Two of its settings depend on the PC more than on the model:
  STRATA_GLM_PCIE_SHARE  the share of a route's RAM-tier experts copied over PCIe and run on the GPU instead of on the
                         CPU (0: the CPU takes every one).  A fast PCIe link and a slow CPU want more; an x8 link beside
                         a many-core CPU wants none.  The engine starts with a guess from timing one expert each way.
  STRATA_GLM_CPU_LANE    the CPU threads that compute them.  Every core is not always fastest: the experts are read
                         from RAM, and past the memory's bandwidth more threads only wait (or, on a hybrid CPU, the
                         efficiency cores make the rest wait for them).
  STRATA_GLM_READ_CHUNKS the pieces (x3) each expert the decode reads from the SSD comes in, when part of the model
                         lives there.  The best size depends on the drive, its driver and the CPU: 8 (the engine's) on
                         a Linux NVMe, 4 on a Windows laptop's Intel RST RAID (#67).
All are measured through ONE running engine - per-request `strata_tune` keys (pcie_frac, cpu_threads, read_chunks) -
because an engine start reads the whole expert set from disk (minutes).  Decode speed only: a prompt balances its CPU
and PCIe shares itself, layer by layer.

Every measurement answers prompts it has not answered before (PROMPTS), the way a chat's text is new to the expert
tiers, and the engine works on a copy of this PC's expert usage file (usage_copy) - its own answers must not lead the
next start's warm-up.

A setting is kept only when it beats the engine's own by more than MIN_GAIN in an interleaved re-measurement - the
expert tiers follow the text and the OS adds noise, so single measurements differ by a few percent.

    python tools/calibrate_glm.py maya-<model>.json      # measure and print; ./maya.sh --calibrate also saves it
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                          # a setting must beat the engine's own by this much to be kept
PCIE_SHARES = (0.0, 0.1, 0.25, 0.5, 0.75)
# a sweep stops once a setting falls this far behind its best so far: the next ones go further the same way, and a
# share far past the best runs at a crawl (one 3090 on PCIe 3.0 x8: 20.9 tok/s at none, 8.1 at 0.10, 2.1 at 0.75 -
# its last three shares took ~8 of the tuning's 19 minutes)
STOP_BELOW = 0.7
MAX_NEW = 128
# A fresh group of three - code, an explanation, a list - for every measurement.  The expert tiers follow the text they
# serve: three prompts answered over and over left them holding just those answers' experts (Maya-L on 9 GPUs: 99.2%
# VRAM hits and 2.6 experts a token on the CPU, 31.8 tok/s, where a chat on that engine had 95-96%, 13-16 on the CPU
# and 24-25 tok/s), so the CPU lane's settings were measured with next to nothing on it.  A run takes ~20 groups.
CODE = ("merges two sorted lists into one sorted list", "checks whether a string is a palindrome",
        "counts the words in a text file", "returns the n-th Fibonacci number with memoization",
        "parses a date like 2024-03-15 into day, month and year", "removes duplicates from a list and keeps the order",
        "finds the longest common prefix of a list of strings", "converts a Roman numeral to an integer",
        "rotates a matrix by 90 degrees", "validates an email address with a regular expression",
        "groups a list of words by their first letter", "computes the median of a list of numbers",
        "flattens a nested list of any depth", "checks whether two strings are anagrams",
        "returns the prime numbers below n", "reverses the words in a sentence",
        "reads a CSV file and sums one of its columns", "converts Celsius to Fahrenheit for a list of values",
        "finds the second largest number in a list", "splits a list into chunks of a given size")
EXPLAIN = ("a refrigerator moves heat from inside to outside", "vaccines train the immune system",
           "a bill becomes a law in a parliament", "GPS finds a position on Earth", "bread dough rises",
           "a bank decides on a loan", "tides follow the Moon", "a compiler turns source code into a program",
           "plants turn sunlight into sugar", "an electric motor turns current into motion",
           "inflation changes what money buys", "a search engine ranks web pages", "volcanoes form",
           "noise-cancelling headphones work", "the heart moves blood through the body",
           "a hash table finds a key quickly", "rainbows form", "a jet engine makes thrust",
           "public key encryption keeps a message secret", "coffee is decaffeinated")
LISTS = ("European capitals", "chemical elements", "famous painters", "programming languages", "African countries",
         "musical instruments", "planets and moons of the solar system", "Olympic sports", "kitchen herbs",
         "inventions of the 20th century", "dog breeds", "Shakespeare plays", "rivers of Asia", "board games",
         "computer scientists", "types of clouds", "Greek gods", "cheeses of France", "bridges around the world",
         "mathematicians")
PROMPTS = tuple(p for c, e, l in zip(CODE, EXPLAIN, LISTS) for p in (
    f"Write a Python function that {c}, with a docstring and two tests.",
    f"Explain in two paragraphs how {e}.",
    f"List twelve {l} with one sentence about each."))
GROUP = 3                                # prompts a measurement (the median of their rates)
SHARE_ENV, THREADS_ENV, CHUNKS_ENV = "STRATA_GLM_PCIE_SHARE", "STRATA_GLM_CPU_LANE", "STRATA_GLM_READ_CHUNKS"
KEYS = (SHARE_ENV, THREADS_ENV, CHUNKS_ENV)   # the environment settings a calibration owns (apply() sets or clears them)
READ_CHUNKS = (4, 2)                     # tried against the engine's own (8) when part of the model is read from the SSD


def prompt_ids(cfg: dict) -> list[list[int]]:
    """PROMPTS as the server sends them: the model's chat template, thinking off (decode speed, not reasoning)."""
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    tpl_path = tpath / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path if tpl_path.exists() else ROOT / "serve" / "chat_template.jinja")
    return [tok.encode(tpl.render([{"role": "user", "content": p}], **effort_kwargs("none")), parse_special=True)
            for p in PROMPTS]


def thread_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: three quarters, two thirds, a half and a quarter (at least 2), plus `extra`
    (the performance cores less one on a hybrid CPU, one socket's cores less one on a 2-socket PC), no repeats."""
    c = [default]
    for w in (round(default * 3 / 4), round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def cpu_list(text: str) -> list[int]:
    """A sysfs CPU list ("0-3,8,10-11") -> its CPU numbers."""
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def host_thread_extras(sys_cpu: Path = Path("/sys/devices/system/cpu"),
                       hybrid_cores: Path = Path("/sys/devices/cpu_core/cpus")) -> list[int]:
    """From the CPU topology: one socket's CPUs less one (several sockets), the performance cores' CPUs less one (a
    hybrid CPU).  [] when it cannot be read."""
    try:
        online = cpu_list((sys_cpu / "online").read_text())
        pkg = {c: (sys_cpu / f"cpu{c}" / "topology" / "physical_package_id").read_text().strip() for c in online}
    except (OSError, ValueError):
        return []
    out = []
    sockets = sorted(set(pkg.values()))
    if len(sockets) >= 2:
        out.append(min(sum(1 for p in pkg.values() if p == s) for s in sockets) - 1)
    try:
        p_cpus = [c for c in cpu_list(hybrid_cores.read_text()) if c in pkg]
        if 0 < len(p_cpus) < len(online):
            out.append(len(p_cpus) - 1)
    except (OSError, ValueError):
        pass
    return [w for w in out if w >= 2]


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


def tune_keys(share, threads, chunks=None) -> dict:
    """The request's `strata_tune`: None = what the engine started with."""
    t = {}
    if share is not None:
        t["pcie_frac"] = float(share)
    if threads is not None:
        t["cpu_threads"] = int(threads)
    if chunks is not None:
        t["read_chunks"] = int(chunks)
    return t


def reads_disk(info: dict) -> bool:
    """Whether part of the model lives on the SSD: more experts than the VRAM and RAM tiers have slots."""
    try:
        experts = int(info.get("experts") or 0)
        held = int(info.get("vram_slots") or 0) + int(info.get("ram_slots") or 0)
    except (TypeError, ValueError):
        return False
    return experts > held > 0


class Session:
    """One running engine: decode tok/s for a setting (the median of the rates of the next GROUP prompts - each
    measurement its own, in turn through `ids_list`)."""

    def __init__(self, engine, ids_list, group: int = GROUP):
        self.engine = engine
        self.ids_list = ids_list
        self.group = max(1, min(group, len(ids_list)))
        self.at = 0

    def next_group(self) -> list:
        n = len(self.ids_list)
        out = [self.ids_list[(self.at + i) % n] for i in range(self.group)]
        self.at = (self.at + self.group) % n
        return out

    def rate(self, tune: dict | None = None) -> float:
        rates = []
        for ids in self.next_group():
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, MAX_NEW, sampling, threading.Event()) if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def usage_copy(cfg: dict, tmp: Path) -> dict:
    """The env that has the calibration's engine read and write a copy of this PC's expert usage file in `tmp`: the
    engine saves the usage after every answer and the next start's warm-up loads the experts in that order, so ~60
    answers to the tuning's prompts would lead it with their experts.  The file is the config's STRATA_GLM_USAGE or
    the pack's expert_usage.txt (the engine's own default); "0" (no usage file) stays."""
    env = cfg.get("env") or {}
    own = str(env.get("STRATA_GLM_USAGE", ""))
    if own == "0":
        return {"STRATA_GLM_USAGE": "0"}
    args = list(cfg.get("args") or [])
    src = Path(own) if own else (Path(args[args.index("--glm-pack") + 1]) / "expert_usage.txt"
                                 if "--glm-pack" in args[:-1] else None)
    dst = tmp / "expert_usage.txt"
    if src is not None and src.is_file():
        shutil.copyfile(src, dst)
    return {"STRATA_GLM_USAGE": str(dst)}


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {env: value}, "report": {...}}.  The engine runs
    with the config's environment less the settings a calibration owns, so the start's own choice is the baseline,
    and on a copy of the expert usage file (usage_copy).
    `start_engine(cfg)` returns a started engine (serve.server.StrataEngine, or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    with tempfile.TemporaryDirectory(prefix="maya-calibrate-") as tmp:
        base = {**cfg, "env": {**apply(cfg.get("env") or {}, {}), **usage_copy(cfg, Path(tmp))}}
        return measure(base, prompt_ids(cfg), start_engine, say, host_thread_extras())


def measure(cfg: dict, ids_list, start_engine, say=print, extra_threads=()) -> dict:
    t0 = time.time()
    report: dict = {}
    settings: dict = {}
    say("  Loading the model for the measurements (it reads the experts into RAM and VRAM first) ...")
    eng = start_engine(cfg)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_threads = int(info.get("cpu_threads") or 0)
        d_share = float(info.get("pcie_share", 1.0))
        report["default"] = {"pcie_share": round(d_share, 2), "cpu_threads": d_threads}
        if d_threads <= 0:
            say("    no CPU lane on this PC (every expert the GPU lacks is copied over PCIe): nothing to tune")
            report["seconds"] = round(time.time() - t0)
            return {"settings": {}, "report": report}
        s = Session(eng, ids_list)
        s.warm_up()
        # 1. the PCIe share (None: the engine's own split), every thread
        by_share = {None: [s.rate()]}
        say(f"    the engine's own split (PCIe share {d_share:.2f}): {by_share[None][0]:.1f} tok/s")
        for f in PCIE_SHARES:
            by_share[f] = [s.rate(tune_keys(f, None))]
            say(f"    PCIe share {f:.2f}: {by_share[f][0]:.1f} tok/s")
            if by_share[f][0] < STOP_BELOW * max(v[0] for v in by_share.values()):
                break
        best_share = max(by_share, key=lambda k: by_share[k][0])
        # 2. the CPU threads, at that share
        by_threads = {}
        for w in sorted(thread_candidates(d_threads, extra_threads), reverse=True):
            by_threads[w] = [s.rate(tune_keys(best_share, None if w == d_threads else w))]
            say(f"    {w} CPU threads: {by_threads[w][0]:.1f} tok/s")
            if by_threads[w][0] < STOP_BELOW * max(v[0] for v in by_threads.values()):
                break
        best_threads = max(by_threads, key=lambda k: by_threads[k][0])
        # 3. the winner against the engine's own, interleaved, three times each
        dflt, cand = (None, d_threads), (best_share, best_threads)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            say("    confirming against the engine's own settings ...")
            for _ in range(3):
                for k in (dflt, cand):
                    confirm[k].append(s.rate(tune_keys(k[0], None if k[1] == d_threads else k[1])))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(share_sweep={("own" if k is None else f"{k:.2f}"): v for k, v in by_share.items()},
                      thread_sweep={str(k): v for k, v in by_threads.items()},
                      confirm={f"{'own' if k[0] is None else f'{k[0]:.2f}'}/{k[1]}": v for k, v in confirm.items()})
        rates = confirm.get(chosen) or (by_share.get(None) if chosen == dflt else None)
        # 4. the disk reads' size, at those settings - only when part of the model is read from the SSD while it
        #    answers; a size is kept as the others are, by beating the engine's own in an interleaved re-measurement
        chunks = None
        if reads_disk(info):
            at = (chosen[0], None if chosen[1] == d_threads else chosen[1])
            by_chunks = {None: [s.rate(tune_keys(*at))]}
            say(f"    the engine's own disk reads (8 pieces an expert part): {by_chunks[None][0]:.1f} tok/s")
            for c in READ_CHUNKS:
                by_chunks[c] = [s.rate(tune_keys(*at, c))]
                say(f"    {c} pieces: {by_chunks[c][0]:.1f} tok/s")
                if by_chunks[c][0] <= by_chunks[None][0]:   # fewer only get slower from there (Mercury: 8 > 4 > 2)
                    break
            best_c = max(by_chunks, key=lambda k: by_chunks[k][0])
            confirm_c = {None: [], best_c: []}
            if best_c is not None:
                say("    confirming the disk reads' size ...")
                for _ in range(3):
                    for k in (None, best_c):
                        confirm_c[k].append(s.rate(tune_keys(*at, k)))
                chunks = pick(confirm_c, None)
            report.update(chunk_sweep={("own" if k is None else str(k)): v for k, v in by_chunks.items()},
                          chunk_confirm={("own" if k is None else str(k)): v for k, v in confirm_c.items()})
            if chunks is not None:
                rates = confirm_c[chunks]
    finally:
        close(eng)
    if chosen[0] is not None:
        settings[SHARE_ENV] = f"{chosen[0]:.2f}"
    if chosen[1] != d_threads:
        settings[THREADS_ENV] = str(chosen[1])
    if chunks is not None:
        settings[CHUNKS_ENV] = str(chunks)
    report["tok_s"] = round(statistics.median(rates), 1) if rates else None
    report["seconds"] = round(time.time() - t0)
    return {"settings": settings, "report": report}


def engine_error(log: str | None, since: int = 0) -> str | None:
    """The engine's own reason for a failed start: the last line of its log written after byte `since` that is its own
    ("glm ...", "maya ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("glm", "maya", "ERR"))), lines[-1] if lines else None)


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(120)
    except Exception:
        proc.kill()


def apply(env: dict, settings: dict) -> dict:
    """`env` (a config's "env" block) with the calibrated settings; one the calibration did not set is removed, so the
    engine chooses it again and an older calibration's value never lingers."""
    out = {k: v for k, v in (env or {}).items() if k not in KEYS}
    for k in KEYS:
        if settings.get(k) is not None:
            out[k] = str(settings[k])
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_glm.py <maya-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    print(json.dumps(res, indent=1))
