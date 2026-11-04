# The README's download badges: each model's all-time Hugging Face downloads written as a small SVG into OUT_DIR, so the
# README shows files from this repository's `badges` branch and never waits on an outside badge service (GitHub's image
# proxy gives up after a few seconds, and a slow one showed as a bare link).  Run daily by .github/workflows/badges.yml.
import json, sys, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "badges")
# (file name, label, Hugging Face repo)
BADGES = [
    ("glm", "GLM-5.3-Flash Maya downloads", "peasantsmith/GLM-5.3-Flash-Maya-GGUF"),
    ("glm-m-derisked", "GLM-5.3-Flash Maya-M-Derisked downloads", "peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF"),
    ("deepseek", "DeepSeek-V4.1-Flash Maya downloads", "peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"),
    ("minimax", "MiniMax-M3.1-Flash Maya downloads", "peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF"),
]


def downloads(repo: str) -> int:
    url = f"https://huggingface.co/api/models/{repo}?expand=downloadsAllTime"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "maya-badges"}), timeout=60) as r:
        return int(json.load(r)["downloadsAllTime"])


def width(text: str) -> int:
    # Verdana 11px, roughly: narrow letters, wide capitals and digits (close enough for a badge's padding)
    w = 0.0
    for ch in text:
        w += 3.5 if ch in "ijlI.,:;|!' -" else 8.5 if ch in "MW" else 7.5 if ch.isupper() or ch.isdigit() else 6.5
    return int(w) + 10


def fit(w: int) -> str:
    # the text drawn exactly into its box whatever the viewer's font (the width above is an estimate)
    return f'textLength="{w - 10}" lengthAdjust="spacingAndGlyphs"'


def svg(label: str, value: str) -> str:
    lw, vw = width(label), width(value)
    t = lw + vw
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{t}" height="20" role="img" aria-label="{label}: {value}">'
            f'<title>{label}: {value}</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" '
            f'stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
            f'<clipPath id="r"><rect width="{t}" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)">'
            f'<rect width="{lw}" height="20" fill="#555"/><rect x="{lw}" width="{vw}" height="20" fill="#007ec6"/>'
            f'<rect width="{t}" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" '
            f'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
            f'<text x="{lw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3" {fit(lw)}>{label}</text>'
            f'<text x="{lw / 2:.1f}" y="14" {fit(lw)}>{label}</text>'
            f'<text x="{lw + vw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3" {fit(vw)}>{value}</text>'
            f'<text x="{lw + vw / 2:.1f}" y="14" {fit(vw)}>{value}</text></g></svg>\n')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, label, repo in BADGES:
        try:
            n = downloads(repo)
        except Exception as e:   # keep yesterday's badge rather than a broken one
            print(f"{repo}: {e}", file=sys.stderr)
            failed += 1
            continue
        (OUT / f"{name}.svg").write_text(svg(label, f"{n:,}"), encoding="utf-8")
        print(f"{name}: {n:,}")
    return 1 if failed == len(BADGES) else 0


if __name__ == "__main__":
    sys.exit(main())
