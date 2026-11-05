#!/usr/bin/env python3
"""Check internal links and anchors in the repository's markdown.

Every relative link must point at a file that exists, and every `#fragment`
must match a heading (GitHub's slug rules) or an explicit `<a name="...">`
anchor in the target file. External links are not fetched: the check is
deterministic and runs offline in CI.

    python tools/check_markdown_links.py [--root .]

Exit code 1 on a broken link or anchor; 0 otherwise. Translation heading
parity is reported as a warning only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

SKIP_DIRS = {".git", "ref", "third_party", "node_modules", "build"}


def markdown_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def anchors(path: str) -> set[str]:
    out: set[str] = set()
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r'<a name="([^"]+)"', text):
        out.add(m.group(1))
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            continue
        heading = re.sub(r'<a name="[^"]+"></a>', "", m.group(2))
        heading = re.sub(r"[`*_\[\]()]", "", heading)
        slug = heading.strip().lower()
        slug = re.sub(r"[^\w\- ]", "", slug)
        out.add(slug.replace(" ", "-"))
    return out


def check(root: str) -> list[tuple[str, str, str]]:
    problems: list[tuple[str, str, str]] = []
    cache: dict[str, set[str]] = {}
    for path in markdown_files(root):
        text = open(path, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\]\(([^)\s]+)", text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            rel, _, frag = target.partition("#")
            if not rel:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), rel))
            if not os.path.exists(resolved):
                problems.append((os.path.relpath(path, root), target, "missing file"))
                continue
            if frag and os.path.isfile(resolved) and resolved.endswith(".md"):
                if resolved not in cache:
                    cache[resolved] = anchors(resolved)
                if frag not in cache[resolved]:
                    problems.append((os.path.relpath(path, root), target, "missing anchor"))
    return problems


def heading_parity(root: str) -> list[str]:
    main = os.path.join(root, "README.md")
    if not os.path.exists(main):
        return []
    want = sum(1 for line in open(main, encoding="utf-8") if line.startswith("## "))
    notes = []
    for name in ("README.ja.md", "README.zh-CN.md", "README.pt-BR.md"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        have = sum(1 for line in open(path, encoding="utf-8") if line.startswith("## "))
        if have != want:
            notes.append(f"{name}: {have} level-2 headings, README.md has {want}")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()
    problems = check(args.root)
    for path, target, kind in problems:
        print(f"{path}: {kind}: {target}")
    for note in heading_parity(args.root):
        print(f"warning: translation parity: {note}")
    if problems:
        print(f"\n{len(problems)} broken internal link(s) or anchor(s)")
        return 1
    print(f"OK: no broken internal links or anchors ({len(list(markdown_files(args.root)))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
