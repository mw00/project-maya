#!/usr/bin/env python3
"""maya-keepwarm - keep Project Maya's model files resident in the OS page cache.

Why this exists: the DS4 engine reads the CPU-lane experts and the Engram
tables straight from the model's files - the page cache is their RAM tier
(SETTINGS.md: "the OS page cache decides which experts hold"). The engine's
own expert tier (the most-routed N GB read ahead at load) already restarts
warm from `expert-profile.bin`, which the engine saves with your routing
history after every run. What does NOT survive a reboot, a wipe or heavy
memory pressure is the page cache itself: afterwards the first prompts
re-read the tail of the files from SSD, and the cache rebuilds over hours of
traffic.

This tool bridges that:
  status     - the guards' lead (MemAvailable, io PSI), the state file's
               summary, three files' current residency.
  snapshot   - record which file ranges are resident right now (mincore), and
               merge them with the last saved set: what is still resident
               keeps generation 0, the uncovered remainder grows a generation
               older and dies after `--gen-life` (default 8) snapshots, with
               `--cap-gb` a backstop. A region the traffic briefly evicted
               stays in the restore set, without drifting growth.
  restore    - posix_fadvise(WILLNEED) the saved ranges so the kernel re-reads
               only them (never whole files, only what fit in cache before).
  daemon     - runs at boot: restore the saved set, then re-snapshot on a
               timer so the state tracks your real usage.

It never holds locks and never mlocks; pages land in the ordinary, kernel-
reclaimable page cache - it cannot out-promise the engine's own planning
MAYA_DS4_WARM_CAP_GB / DS4_RAM_MARGIN_GB, and never writes to model files.

The model's files come from the config the setup runs (the maya-*.json beside
the repo root that carries --pack), or pass the config / --pack explicitly.
Nothing runs off Linux (it needs /proc, mincore, posix_fadvise).

Usage:  maya_keepwarm.py status|snapshot|restore|daemon [config.json]
        [--pack DIR] [--cap-gb G] [--gen-life N] [--floor-gb G] [--psi-max P]
        [--force] [--interval S] [--no-boot-restore]
        (keepwarm is read-only: it never writes a model file)
"""
import argparse
import ctypes
import json
import os
import sys
import time

# the root the tool sits in (tools/maya_keepwarm.py): the config ./maya.sh
# runs and the default state file live beside it
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "maya-keepwarm.json")

# guards: never squeeze the box while the engine or anything real is working
FLOOR_GB_DEFAULT = 24.0    # hold off I/O when MemAvailable sinks below this
PSI_MAX_DEFAULT = 12.0     # /proc/pressure/io "some avg10" percent ceiling
COARSE = 2 * 1024 * 1024   # snap saved ranges to 2 MiB steps to keep state light
MERGE_CAP_GB_DEFAULT = 120.0   # the saved set's ceiling (the merge's cap)
MERGE_GEN_LIFE_DEFAULT = 8     # generations an unseen range survives (a gen = 1 snapshot)

PROT_READ = 0x1            # Linux
MAP_SHARED = 0x01          # Linux
POSIX_FADV_WILLNEED = 3
CHUNK = 256 * 1024 * 1024        # map/inspect 256 MiB at a time
_PAGE = None
_LIBC = None
RES_TABLE = bytes((1 if b & 1 else 0) for b in range(256))  # mincore bit0 -> 0/1


def _coarse(x):
    "A byte offset snapped down to a COARSE boundary."
    return (x // COARSE) * COARSE


def _merge_runs(runs):
    "Sort [start, len) spans and coalesce overlap/adjacency."
    out = []
    for s, n in sorted(runs):
        if n <= 0:
            continue
        if out and s <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], s + n - out[-1][0])
        else:
            out.append([s, n])
    return out


def page_size():
    global _PAGE
    if _PAGE is None:
        try:
            _PAGE = os.sysconf("SC_PAGESIZE")
        except (AttributeError, OSError, ValueError):
            _PAGE = 4096
    return _PAGE


def libc():
    """Loaded on first use so the module imports cleanly off Linux (CI, Windows)."""
    global _LIBC
    if _LIBC is None:
        c = ctypes.CDLL("libc.so.6", use_errno=True)
        c.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        c.mmap.restype = ctypes.c_void_p          # (void*)-1 arrives as None
        c.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        c.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.POINTER(ctypes.c_ubyte)]
        c.mincore.restype = ctypes.c_int
        c.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_longlong,
                                    ctypes.c_longlong, ctypes.c_int]
        c.posix_fadvise.restype = ctypes.c_int
        _LIBC = c
    return _LIBC


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S ") + msg, flush=True)


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)  # kB -> GB
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def psi_io_pct():
    try:
        with open("/proc/pressure/io") as f:
            first = f.readline()
        return float(first.split("avg10=")[1].split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def resolve_pack(config, pack):
    """The pack directory: --pack wins; else the given config's --pack; else
    the maya-*.json files beside the repo root - the one that carries a
    --pack. Returns None (after a clear message) when it is none or several."""
    if pack:
        return pack
    configs = [config] if config else []
    if not config:
        try:
            for name in sorted(os.listdir(ROOT)):
                if name.startswith("maya-") and name.endswith(".json"):
                    configs.append(os.path.join(ROOT, name))
        except OSError:
            pass
    packs = {}
    for c in configs:
        try:
            with open(c, encoding="utf-8", errors="replace") as f:
                args = json.load(f).get("args") or []
        except OSError as e:
            log("! cannot read config %s: %s" % (c, e.strerror or e))
            continue
        except (ValueError, AttributeError):
            log("! not a json object with an args list: %s" % c)
            continue
        if "--pack" in args[:-1]:
            packs[args[args.index("--pack") + 1]] = c
    if len(packs) == 1:
        return next(iter(packs))
    if not packs:
        log("! no config found carrying a --pack%s - pass the config .json "
            "or --pack DIR" % ("" if config else " (looked in %s)" % ROOT))
    else:
        log("! several configs carry a --pack: %s - pass the config .json "
            "or --pack DIR" % ", ".join("%s (%s)" % (k, os.path.basename(v))
                                        for k, v in sorted(packs.items())))
    return None


def pack_files(pack):
    """The pack's weight files exactly as the engine opens them (tensors.txt
    maps `file N name` relative to the pack meta dir), plus the drafter dir."""
    files, seen = [], set()

    def add(path):
        path = os.path.realpath(path)
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            files.append(path)

    try:
        parent = os.path.dirname(pack)
        with open(os.path.join(pack, "tensors.txt"), encoding="utf-8",
                  errors="replace") as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[0] == "file":
                    rel = p[2]
                    # names resolve against the model dir (the pack's parent)
                    # like ds4_packinfo does; relative ones work from either
                    for base in (parent, pack):
                        cand = os.path.normpath(os.path.join(base, rel))
                        if os.path.isfile(cand):
                            add(cand)
                            break
                    else:
                        log("  ! not found (skipping): file %s" % rel)
    except OSError:
        pass
    dspark = os.path.join(os.path.dirname(pack), "dspark")
    if os.path.isdir(dspark):
        for name in sorted(os.listdir(dspark)):
            if not name.endswith(".json"):
                add(os.path.join(dspark, name))
    return pack, files


def resident_ranges(path):
    "Byte ranges of `path` presently in the page cache, snapped to COARSE."
    runs = []
    size = os.path.getsize(path)
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        log("  ! cannot open %s: %s" % (path, e.strerror))
        return runs
    try:
        l = libc()
        page = page_size()
        win_pages = COARSE // page                    # 2 MiB a window
        for base in range(0, size, CHUNK):
            n = min(CHUNK, size - base)
            addr = l.mmap(None, n, PROT_READ, MAP_SHARED, fd, base)
            if not addr:                              # None = MAP_FAILED
                continue
            pages = (n + page - 1) // page
            vec = (ctypes.c_ubyte * pages)()
            ok = l.mincore(addr, n, vec) == 0
            l.munmap(addr, n)
            if not ok:
                continue
            vb = bytes(vec)                           # page_size steps
            one = b"\x01" * win_pages
            for wstart in range(0, pages, win_pages):
                s0 = base + wstart * page             # window's first byte
                # a window's resident picture in one translate: table maps
                # mincore's byte (bit0 = present) to 1/0
                w = vb[wstart:wstart + win_pages].translate(RES_TABLE)
                if w == one:                          # whole window resident
                    runs.append([s0, win_pages * page])
                elif w.count(0) == len(w):
                    continue                          # nothing here
                else:                                 # partial: exact pages
                    v = int.from_bytes(w, "little")
                    off = 0
                    while v:
                        b = (v & -v).bit_length() - 1
                        t = v >> b
                        run = (t ^ (t + 1)).bit_length() - 1
                        runs.append([_coarse(s0 + (off + b) * page),
                                     _coarse(s0 + (off + b + run) * page)
                                     - _coarse(s0 + (off + b) * page)])
                        v >>= (b + run)
                        off += b + run
    finally:
        os.close(fd)
    return _merge_runs(runs)


def _coalesce_gens(spans):
    """Sort [start, len, gen) spans; fuse overlap/adjacency within the SAME
    generation (a later generation can overlap an earlier one: the bytes are
    both resident and learned, restore order is indifferent, fadvise twice
    costs nothing)."""
    out = []
    last_end = {}
    for s, n, g in sorted(spans, key=lambda sp: (sp[0], sp[2])):
        end = s + n
        if n <= 0:
            continue
        if last_end.get(g) is not None and s <= last_end[g]:
            for sp in out:
                if sp[2] == g and sp[0] <= s <= sp[0] + sp[1]:
                    sp[1] = max(sp[1], end - sp[0])
                    break
            last_end[g] = max(last_end[g], end)
        else:
            out.append([s, n, g])
            last_end[g] = end
    return sorted(out, key=lambda sp: (sp[0], sp[2]))


def _merge_generations(now, prev, gen_life, cap):
    """One snapshot's merge: what is resident now is gen 0; the saved set's
    uncovered remainder grows a generation older and dies after gen_life.
    Returns the kept set as [start, len, gen] spans sorted by start.
    `cap` (bytes) is a backstop: once exceeded, the oldest generations go
    first; what the traffic is using right now is never dropped."""
    base = _merge_runs(now)
    kept = [[s, n, 0] for s, n in base]
    for p in prev:
        s, n = p[0], p[1]
        g = (p[2] + 1) if len(p) > 2 else 1
        if g > gen_life:
            continue
        e = s + n
        cur = s
        for cs, cn in base:                 # base sorted, disjoint
            if cs + cn <= cur:
                continue
            if cs >= e:
                break
            if cs > cur:
                kept.append([cur, cs - cur, g])
            if cur < cs + cn:
                cur = cs + cn
            if cur >= e:
                break
        if cur < e:
            kept.append([cur, e - cur, g])
    kept = _coalesce_gens(kept)
    total = sum(sp[1] for sp in kept)
    if cap is not None and total > cap:
        for sp in sorted((sp for sp in kept if sp[2] > 0), key=lambda sp: -sp[2]):
            if total <= cap:
                break
            kept.remove(sp)
            total -= sp[1]
        kept = _coalesce_gens(kept)         # removals may open merges
    return kept


def take_snapshot(pack, merge=True, cap_gb=MERGE_CAP_GB_DEFAULT,
                  gen_life=MERGE_GEN_LIFE_DEFAULT):
    _, files = pack_files(pack)
    if not files:
        log("! no warm-set files found%s - nothing to snapshot"
            % (" (pack %s)" % pack if pack else ""))
        return
    prev = {}
    if merge and os.path.isfile(STATE):
        try:
            with open(STATE) as f:
                old = json.load(f)
            prev = {e["path"]: e.get("ranges", []) for e in old.get("files", ())}
        except (OSError, ValueError):
            prev = {}
    cap = cap_gb * 2**30
    # pass 1: what every file holds resident (that branch is never dropped,
    # so it must not eat the learned budget)
    nows = []
    resident_total = 0
    for path in files:
        now = resident_ranges(path)
        nows.append((path, now))
        resident_total += sum(r[1] for r in now)
    learned_budget = max(0.0, cap - resident_total)
    learned_used = 0.0
    out = {"version": 3, "pack": pack, "saved": time.strftime("%F %T"),
           "files": []}
    total_kept = total_size = 0
    for path, now in nows:
        size = os.path.getsize(path)
        now_bytes = sum(r[1] for r in now)
        old = [r for r in prev.get(path, []) if len(r) >= 2]
        kept = [[s, n, 0] for s, n in now]
        if old:
            kept = _merge_generations(now, old, gen_life,
                                      max(0.0, learned_budget - learned_used))
        learned_used += max(0.0, sum(sp[1] for sp in kept) - now_bytes)
        kept_bytes = sum(sp[1] for sp in kept)
        total_kept += kept_bytes
        total_size += size
        out["files"].append({"path": path, "size": size, "ranges": kept})
        log("  %.60s  now %6.1f -> kept %6.1f GiB (of %.1f)" %
            (os.path.basename(path), now_bytes / 2**30, kept_bytes / 2**30,
             size / 2**30))
    out["resident_gb"] = resident_total / 2**30
    out["kept_gb"] = total_kept / 2**30
    log("snapshot saved: %.1f GiB resident now, %.1f GiB kept (cap %.0f, "
        "gen-life %d), over %d files" %
        (resident_total / 2**30, total_kept / 2**30, cap_gb, gen_life,
         len(files)))
    tmp = "%s.tmp.%d" % (STATE, os.getpid())   # unique: the daemon and a
    with open(tmp, "w") as f:                  # manual snapshot do not race
        json.dump(out, f)
    os.replace(tmp, STATE)

def restore_saved(force, floor_gb=FLOOR_GB_DEFAULT, psi_max=PSI_MAX_DEFAULT):
    if not os.path.isfile(STATE):
        log("no state yet - run 'snapshot' after a session of real use")
        return 1
    try:
        with open(STATE) as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        log("! cannot read %s: %s" % (STATE, e))
        return 1
    log("restoring saved ranges of %d files (saved %s)" %
        (len(state.get("files", ())), state.get("saved", "?")))
    for entry in state.get("files", ()):
        path = entry["path"]
        if not os.path.isfile(path):
            log("  ! missing now: %s" % path)
            continue
        if not force and (mem_available_gb() < floor_gb
                          or psi_io_pct() > psi_max):
            log("  ! guard hit (MemAvailable %.1f GB, io %.1f%%) - stopping"
                % (mem_available_gb(), psi_io_pct()))
            return 2
        total = 0
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            log("  ! cannot open %s: %s" % (path, e.strerror))
            continue
        try:
            adv = libc().posix_fadvise
            for rng in entry.get("ranges", ()):
                adv(fd, int(rng[0]), int(rng[1]), POSIX_FADV_WILLNEED)
                total += int(rng[1])
        finally:
            os.close(fd)
        log("  %.60s  %6.1f GiB asked back" %
            (os.path.basename(path), total / 2**30))
    return 0


def status(pack=None):
    log("MemAvailable %.1f GB, io PSI %.1f%%" %
        (mem_available_gb(), psi_io_pct()))
    if os.path.isfile(STATE):
        try:
            with open(STATE) as f:
                state = json.load(f)
            log("state last saved %s: %.1f GiB resident, %.1f GiB kept, over "
                "%d files" % (state.get("saved"), state.get("resident_gb", 0),
                              state.get("kept_gb", state.get("resident_gb", 0)),
                              len(state.get("files", ()))))
        except (OSError, ValueError):
            log("state file unreadable")
    else:
        log("no state yet")
    if not pack:
        return
    for path in pack_files(pack)[1][:3]:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        res = sum(r[1] for r in resident_ranges(path))
        log("  now resident: %.60s  %5.1f%%" %
            (os.path.basename(path), 100.0 * res / size if size else 0.0))


def daemon(interval, boot_restore, cap_gb, gen_life, floor_gb, psi_max, pack):
    try:
        os.nice(10)
    except OSError:
        pass
    if boot_restore and os.path.isfile(STATE):
        log("boot restore of the saved working set")
        restore_saved(False, floor_gb, psi_max)
    else:
        log("daemon up (no saved state yet - first snapshot coming)")
    while True:
        take_snapshot(pack, merge=True, cap_gb=cap_gb, gen_life=gen_life)
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["status", "snapshot", "restore", "daemon"])
    ap.add_argument("config", nargs="?",
                    help="the config .json the setup runs (its --pack is kept "
                         "warm); default: the maya-*.json beside the repo root")
    ap.add_argument("--pack", help="the pack directory (instead of a config)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the MemAvailable / io-PSI guards")
    ap.add_argument("--interval", type=int, default=900,
                    help="daemon snapshot period, seconds (default 900)")
    ap.add_argument("--no-boot-restore", action="store_true",
                    help="daemon: skip the initial restore")
    ap.add_argument("--cap-gb", type=float, default=MERGE_CAP_GB_DEFAULT,
                    help="snapshot merge's kept-set ceiling in GiB (default 120)")
    ap.add_argument("--gen-life", type=int, default=MERGE_GEN_LIFE_DEFAULT,
                    help="snapshots an unseen range survives the merge "
                         "(default 8; 0 keeps every generation)")
    ap.add_argument("--floor-gb", type=float, default=FLOOR_GB_DEFAULT,
                    help="hold restore I/O off below this much MemAvailable, "
                         "GB (default 24)")
    ap.add_argument("--psi-max", type=float, default=PSI_MAX_DEFAULT,
                    help="hold restore I/O off above this /proc/pressure/io "
                         "some avg10 percent (default 12)")
    a = ap.parse_args()
    if not os.path.isdir("/proc/self"):
        log("! Linux only (needs /proc, mincore, posix_fadvise)")
        sys.exit(3)
    pack = resolve_pack(a.config, a.pack)
    if pack and not os.path.isdir(pack):
        log("! pack directory not found: %s" % pack)
        pack = None
    if a.mode in ("snapshot", "daemon") and not pack:
        sys.exit(2)
    if a.mode == "status":
        status(pack)
    elif a.mode == "snapshot":
        take_snapshot(pack, cap_gb=a.cap_gb, gen_life=a.gen_life)
    elif a.mode == "restore":
        sys.exit(restore_saved(a.force, a.floor_gb, a.psi_max))
    elif a.mode == "daemon":
        daemon(a.interval, not a.no_boot_restore, a.cap_gb, a.gen_life,
               a.floor_gb, a.psi_max, pack)


if __name__ == "__main__":
    main()
