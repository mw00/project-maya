"""serve/pdf.py - the chat page's PDF attachments (#837): a PDF comes back as its text when it carries a text
layer, and as its first pages in pictures when it does not (a scanned one, a print of slides).

The work is poppler's when it is on PATH - pdftotext with -layout, and pdftoppm for the pages - looked up with
shutil.which at every call.  Maya installs nothing system-wide, so without poppler (most Windows and macOS PCs)
serve/pdf_text.py reads the PDF in pure Python instead: the text layer, and a scanned page's own picture (the JPEG
or image a scanner stored).  Only what neither can read gets an error, naming poppler as the way to read more.

    extract(data, want_images=..., text_limit=...) -> a JSON-ready dict, one of:
      {"kind": "text", "text", "pages", "truncated", "note"}                the text layer, capped at text_limit
      {"kind": "images", "pages", "images": [{"name", "data"}], "note"}     the first pages as PNG / JPEG data URLs,
                                                                            for the vision path like any attachment
      {"error": {"message"}}                                                encrypted, nothing readable, too slow

text_limit is the page's health.max_text_file (the server's text_file_limit), so an extracted text is never larger
than a file the reader could have attached by hand; the rest is cut and said so.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

from serve import pdf_text

TEXT_FLOOR = 200          # less than this many characters after stripping and the "text" is page furniture: the
                          # PDF is treated as scanned and its pages go to pictures instead
MAX_PAGES = 6             # a scanned PDF's pages per attachment: every one costs the sender's vision_max_tokens
DPI = 100                 # (pdftoppm's own default: enough for the encoders, and a page is ~1 MB of PNG at it)
TIMEOUT_S = 30            # one PDF is never worth holding the request worker longer than this
NEED_POPPLER = ("poppler on this PC reads more PDFs (Ubuntu/Debian: sudo apt install poppler-utils; Arch: sudo pacman "
                "-S poppler; macOS: brew install poppler; Windows: scoop install poppler), then restart Maya")


def _run(tool: str, args: list[str]) -> subprocess.CompletedProcess | None:
    """One poppler call, or None when it could not run; runs out of time raises TimeoutError (extract turns it into
    the error the page toasts)."""
    try:
        return subprocess.run([tool, *args], capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"{tool} took over {TIMEOUT_S} s with this PDF") from None
    except OSError:
        return None


def page_count(pdf: Path) -> int | None:
    """The PDF's page count by pdfinfo, when poppler carries it and it says; None otherwise (never fatal: the
    count only decorates the toast)."""
    info = shutil.which("pdfinfo")
    if info is None:
        return None
    try:
        r = _run(info, [str(pdf)])
    except TimeoutError:
        return None
    if r is None or r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split()[1])
            except (ValueError, IndexError):
                return None
    return None


def _text_of(pdftotext: str, pdf: Path, cap: int) -> str | None:
    """The PDF's text layer, -layout to keep columns and indentation; None when pdftotext cannot read the file.
    At most a little over `cap` characters are read - the caller cuts, and a bigger text than the cap is one
    keystroke of a PDF nobody asked to hold whole."""
    out = pdf.with_suffix(".txt")
    try:
        r = _run(pdftotext, ["-q", "-layout", str(pdf), str(out)])
        if r is None or r.returncode != 0:
            return None
        return out.read_bytes()[:cap + 4096].decode("utf-8", "replace")
    except OSError:
        return None      # (a broken poppler install, or a tempdir that vanished: not a readable PDF)
    finally:
        out.unlink(missing_ok=True)


def _pictures(pdftoppm: str, pdf: Path) -> list[bytes] | None:
    """The first MAX_PAGES pages as PNG bytes, in page order; None when pdftoppm cannot render the file."""
    base = pdf.with_name("page")
    # -l caps the work: pdftoppm numbers its output by the page's width (page-1.png, page-01.png at ten or more),
    # so the files are ordered by the number in their name, not by the name itself
    try:
        r = _run(pdftoppm, ["-png", "-r", str(DPI), "-f", "1", "-l", str(MAX_PAGES), str(pdf), str(base)])
        if r is None or r.returncode != 0:
            return None
        pages = sorted(pdf.parent.glob("page-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        return [p.read_bytes() for p in pages[:MAX_PAGES]]
    except (OSError, ValueError):
        return None




def _as_text(body: str, pages: int | None, text_limit: int) -> dict:
    """The text layer as the attachment, cut at the page's own text-file limit with the cut said."""
    cut = len(body) > text_limit
    out = {"kind": "text", "text": body[:text_limit], "pages": pages, "truncated": cut}
    if cut:
        out["note"] = f"cut at {text_limit} characters - the text-file limit for this model's context"
    return out


def _as_pictures(pages: list[tuple[str, bytes]], total: int | None) -> dict:
    """The pages' pictures as data URLs, in the shape an image attachment takes (the chat sends them onward as any
    other picture, through the vision path it already has)."""
    out = {"kind": "images", "pages": len(pages),
           "images": [{"name": f"page {i + 1}", "data": f"data:{mime};base64," + base64.b64encode(p).decode()}
                      for i, (mime, p) in enumerate(pages)]}
    if total and total > len(pages):
        out["note"] = f"the first {len(pages)} of {total} pages"
    return out


def _extract_into(pdf: Path, data: bytes, *, want_images: bool, text_limit: int, pdftotext: str | None,
                  pdftoppm: str | None, text_only: str | None = None) -> dict:
    """The route out of the tempdir that holds the PDF: text, pages or the error that says why neither came out.
    poppler first when it is there; serve/pdf_text.py where it is not, or where it read nothing.  text_only: the
    model's name when it has no vision tower at all (GLM-5.3) - the error then says so, not "images are off"."""
    body, total, problem = (_text_of(pdftotext, pdf, text_limit) or "").strip() if pdftotext else "", None, None
    if len(body) < TEXT_FLOOR:
        try:
            pages = pdf_text.text_pages(data)
            total = len(pages)
            own = "\n\n".join(p for p in pages if p).strip()
            if len(own) > len(body):
                body = own
        except pdf_text.PdfError as e:
            problem = str(e)
    if len(body) >= TEXT_FLOOR:
        return _as_text(body, page_count(pdf) or total, text_limit)
    # too little text to be the attachment: the pages are, when they can go; else whatever little text there
    # is still beats an error (a PDF with a stray page number says more than nothing)
    if want_images:
        pics = [("image/png", p) for p in (_pictures(pdftoppm, pdf) or [])] if pdftoppm else []
        if not pics and problem is None:
            try:
                pics = pdf_text.page_pictures(data, MAX_PAGES)
            except pdf_text.PdfError as e:
                problem = str(e)
        if pics:
            return _as_pictures(pics, page_count(pdf) or total)
    if body:
        return _as_text(body, page_count(pdf) or total, text_limit)
    if problem:
        return {"error": {"message": problem}}
    if not want_images and text_only:
        return {"error": {"message": f"this PDF has no text layer (a scanned one), and {text_only} reads text only: "
                                     "it has no vision tower to read its pages as pictures - attach the document's "
                                     "text instead"}}
    if not want_images:
        return {"error": {"message": "this PDF has no text layer, and its pages cannot be sent as pictures: this "
                                     "model runs without images (Models › Engine settings turns them on when it can)"}}
    return {"error": {"message": "nothing readable came out of this PDF" +
                                 (f" - {NEED_POPPLER}" if pdftotext is None or pdftoppm is None else "")}}


def extract(data: bytes, *, want_images: bool, text_limit: int, text_only: str | None = None) -> dict:
    """A PDF's bytes -> the chat page's attachment: text when it carries a text layer worth sending, the first
    pages as pictures when it is scanned and the model reads pictures (want_images), an error saying why when
    neither applies (text_only: the name of a model with no vision tower, GLM-5.3, for its words).  Temporary files
    live in one tempdir, gone at the exit."""
    if not data.lstrip()[:5] == b"%PDF-":
        return {"error": {"message": "this is not a PDF"}}
    pdftotext, pdftoppm = shutil.which("pdftotext"), shutil.which("pdftoppm")
    with tempfile.TemporaryDirectory(prefix="maya-pdf-") as d:
        pdf = Path(d) / "in.pdf"
        try:
            pdf.write_bytes(data)
        except OSError as e:
            return {"error": {"message": f"the PDF could not be written to a temporary file ({e})"}}
        try:
            return _extract_into(pdf, data, want_images=want_images, text_limit=text_limit,
                                 pdftoppm=pdftoppm, pdftotext=pdftotext,
                                 **({"text_only": text_only} if text_only else {}))
        except TimeoutError as e:
            return {"error": {"message": f"{e}: the PDF was dropped, try a smaller one"}}
