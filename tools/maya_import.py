"""maya.py --import-model <folder>: model files downloaded some other way (a download manager, a browser, a copy from
another PC, a Hugging Face cache) are found by what they are, not by what they are called.

A file matches a published file when its size is the published size (files whose size is not listed: a GGUF) and its
sha256 is the published one - whatever its name or folder.  Matched files are moved (same drive) or copied (another
drive, or a Hugging Face cache link) into the folder layout the setup uses, and recorded as verified.  Nothing that did
not match is touched, and nothing is deleted: a move leaves no old copy, a copy leaves the original.

The catalog is a list of dicts: {"label": the model's folder (relative to the models folder), "rel": the file's place
(relative to the models folder), "size": bytes or None, "sha": [lowercase sha256, ...]}.  No GPU, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

VERIFIED = ".maya-verified.json"          # the same record maya.py keeps: a file whose sha256 matched, by size and time
MIN_UNLISTED = 1 << 20                     # a GGUF the table lists by sha256 only is read from this size on
SHARED_SMALL = 16 << 20                    # a file below this that several models share is only placed beside a model


def sha256_of(path: Path, progress=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
            if progress:
                progress(len(b))
    return h.hexdigest()


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def files_under(folder: Path) -> list:
    """Every file under folder (links followed), each real file once: (the path it was found by, its real path).  A
    link (a Hugging Face cache's snapshot -> blob) is found by its link's name; Maya's own bookkeeping is skipped."""
    seen, out = {}, []
    for root, dirs, names in os.walk(folder, followlinks=True):
        dirs.sort()
        for n in sorted(names):
            p = Path(root) / n
            if n == VERIFIED or n.endswith((".maya-import", ".incomplete", ".part", ".aria2", ".td", ".xltd", ".!ut")):
                continue                       # (a download that is not finished is not a file yet)
            try:
                st = p.stat()
            except OSError:
                continue
            key = (st.st_dev, st.st_ino) if st.st_ino else (str(p.resolve()),)
            if not p.is_file():
                continue
            real = p.resolve()
            via_link = os.path.normcase(os.path.abspath(p)) != os.path.normcase(str(real))
            if key in seen:                    # (the same file twice: a link and its target - the link is kept, so the
                i = seen[key]                  #  file is copied, never moved out from under a Hugging Face cache)
                if via_link and out[i][0] == out[i][1]:
                    out[i] = (p, real, st.st_size)
                continue
            seen[key] = len(out)
            out.append((p, real, st.st_size))
    return out


def find(folder: Path, catalog: list, progress=None, hashing=None) -> dict:
    """Match the folder's files to the catalog: {"found": [(found path, real path, size, [entries])], "unmatched":
    [(path, size)], "shared": [(path, [labels])]}.  progress(text): a line before a big file is read."""
    by_size, sha_only = {}, []
    for e in catalog:
        (by_size.setdefault(e["size"], []) if e["size"] is not None else sha_only).append(e)
    by_sha = {}
    for e in catalog:
        for s in e["sha"]:
            by_sha.setdefault(s, []).append(e)
    found, unmatched = [], []
    for path, real, size in files_under(folder):
        if size == 0:
            continue
        candidates = size in by_size or (sha_only and size >= MIN_UNLISTED and is_gguf(real))
        if not candidates:
            unmatched.append((path, size))
            continue
        if progress and size >= (256 << 20):
            progress(f"  reading {path.name} ({size / 1e9:.1f} GB) ...")
        digest = sha256_of(real, hashing)
        hits = [e for e in by_sha.get(digest, []) if e["size"] in (None, size)]
        if hits:
            found.append((path, real, size, hits))
        else:
            unmatched.append((path, size))
    return {"found": found, "unmatched": unmatched}


def choose(found: list) -> tuple:
    """Where each found file goes: ([(found path, real path, size, [entries to place])], [(path, [labels])] for the
    files shared by several models that no model of the folder claims).  A hash several models list (a shared vision
    file) goes to the models the folder holds other files of; with none it is only placed when exactly one lists it."""
    labels_with_big = {e["label"] for _, _, size, hits in found if size >= SHARED_SMALL and len(hits) == 1
                       for e in hits}
    placed, shared, taken = [], [], set()
    for path, real, size, hits in found:
        every = hits
        if len({e["label"] for e in hits}) > 1:
            hits = [e for e in hits if e["label"] in labels_with_big]
        if not hits:
            shared.append((path, sorted({e["label"] for e in every})))
            continue
        key = tuple(sorted(e["rel"] for e in hits))
        if key in taken:                       # (a second copy of the same file in the folder)
            continue
        taken.add(key)
        placed.append((path, real, size, hits))
    return placed, shared


def same_drive(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_dev == b.stat().st_dev
    except OSError:
        return False


def record(dest: Path) -> None:
    """dest's sha256 matched: remember it as maya.py does (its size and modification time), so it is not read again."""
    book_path = dest.parent / VERIFIED
    try:
        book = json.loads(book_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        book = {}
    st = dest.stat()
    book[dest.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    book_path.write_text(json.dumps(book, indent=1), encoding="utf-8")


def place(path: Path, real: Path, size: int, entries: list, models: Path, copy: bool, say) -> dict:
    """Put one found file at its place(s) in the models folder: moved when it can be (same drive, no link), else
    copied when copy says so (the original stays).  Returns {"moved": n, "copied": n, "skipped": [text]}."""
    out = {"moved": 0, "copied": 0, "skipped": []}
    first = None                               # where the file went first: further places are copied from it
    for e in entries:
        dest = models / e["rel"]
        if dest.exists():
            try:
                same = dest.samefile(real)
            except OSError:
                same = False
            if same:
                record(dest)
                continue
            out["skipped"].append(f"{e['rel']} is there already ({dest.stat().st_size:,} bytes); not replaced "
                                  f"(move it away to use {path.name})")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        source = first or real
        linked = os.path.normcase(os.path.abspath(path)) != os.path.normcase(str(real))   # (reached through a link)
        can_move = first is None and not linked and same_drive(real, dest.parent)
        if not can_move and not copy and first is None:
            out["skipped"].append(f"{path} is on another drive (or a link into a cache): pass --copy to copy it "
                                  f"into {dest.parent} (the original stays)")
            continue
        if can_move:
            os.replace(real, dest)
            out["moved"] += 1
        else:
            tmp = dest.with_name(dest.name + ".maya-import")
            try:
                shutil.copyfile(source, tmp)
                if tmp.stat().st_size != size:
                    raise OSError("the copy is not as big as the original")
                os.replace(tmp, dest)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                out["skipped"].append(f"{path.name}: not copied to {dest} ({exc})")
                continue
            out["copied"] += 1
        record(dest)
        first = first or dest
    return out


def report_lines(catalog: list, placed: list, models: Path) -> tuple:
    """(have, missing): per model the files found in this run or there before -> [(label, [present names], [missing
    names])] for the models at least one file of which was found."""
    got = {}
    for _, _, _, hits in placed:
        for e in hits:
            got.setdefault(e["label"], set()).add(e["rel"])
    have, missing = [], []
    for label in sorted(got):
        names = {e["rel"]: e for e in catalog if e["label"] == label}
        absent = sorted(Path(r).name for r in names if r not in got[label] and not (models / r).exists())
        have.append((label, sorted(Path(r).name for r in got[label])))
        missing.append((label, absent))
    return have, missing
