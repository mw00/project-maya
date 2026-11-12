"""serve/attach.py - any file a chat attaches, read into what the model takes: text, or pictures for a model that
reads them.  The web chat sends here what the browser cannot read itself (POST /attach); the APIs' own file parts
(OpenAI "file" / Responses "input_file" / Anthropic "document") go through expand_request() before the template.

    extract(name, data, want_images=..., text_limit=...) -> a JSON-ready dict, one of:
      {"kind": "text", "text", "format", "truncated", "note", "pages"}   the file's text, capped at text_limit
      {"kind": "images", "format", "pages", "images": [{"name", "data"}], "note"}   a scanned PDF's pages
      {"error": {"message"}}                                             what it is and why it was not read

What is read, all with the standard library: PDFs (serve/pdf.py: poppler when installed, else pure Python), Word /
Excel / PowerPoint (.docx .xlsx .pptx and their macro twins), OpenDocument (.odt .ods .odp), EPUB, RTF, and any
file that is text in UTF-8, UTF-16 or a Latin code page whatever its name.  Old binary Office files, archives,
audio and executables get a message that says what to send instead.  ZIP members are read with a size cap (a zip
bomb stops there) and XML through the standard parser, which resolves no external entities.
"""
from __future__ import annotations

import base64
import binascii
import csv
import io
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

from serve import pdf

MAX_BYTES = 64 << 20          # the largest attachment POST /attach takes (the page checks it before the read)
MEMBER_MAX = 64 << 20         # the most one ZIP member may inflate to
TOTAL_MAX = 256 << 20         # ... and all of them together

READS = ("text and code files, PDFs, Word, Excel and PowerPoint documents, OpenDocument files, EPUB and RTF")


class Unreadable(ValueError):
    pass


def _ext(name: str) -> str:
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.rsplit(".", 1)[-1] if "." in base else ""


# ------------------------------------------------------------------ plain text in any common encoding
def as_text(data: bytes) -> str | None:
    """The bytes as text when they are text (UTF-8 / UTF-16 with or without BOM, else a Latin code page with almost
    nothing but printable characters), else None."""
    if not data:
        return ""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    head = data[:65536]
    if b"\x00" in head:
        # UTF-16 without a BOM: every other byte zero in ASCII-heavy text
        for enc, zeros in (("utf-16-le", head[1::2]), ("utf-16-be", head[0::2])):
            if zeros.count(0) > 0.9 * len(zeros):
                try:
                    return data[:len(data) // 2 * 2].decode(enc)
                except UnicodeDecodeError:
                    return None
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start >= len(data) - 3:                     # a multi-byte character cut by a truncation
            return data[:e.start].decode("utf-8")
    bad = sum(1 for b in head if b < 32 and b not in (9, 10, 12, 13)) + head.count(0x7F)
    if bad > len(head) * 0.002:
        return None
    return data.decode("cp1252", "replace")


# ------------------------------------------------------------------ ZIP-based documents
class _Zip:
    def __init__(self, data: bytes):
        try:
            self.z = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, ValueError) as e:
            raise Unreadable(f"the file is a damaged ZIP container ({e})") from None
        self.names = set(self.z.namelist())
        self.read_total = 0

    def read(self, name: str) -> bytes | None:
        if name not in self.names:
            return None
        info = self.z.getinfo(name)
        if info.flag_bits & 0x1:
            raise Unreadable("the document is encrypted (password-protected): save an unprotected copy and attach "
                             "that")
        with self.z.open(info) as f:
            out = f.read(MEMBER_MAX + 1)
        self.read_total += len(out)
        if len(out) > MEMBER_MAX or self.read_total > TOTAL_MAX:
            raise Unreadable("the document unpacks to more than Maya reads from one attachment")
        return out

    def xml(self, name: str):
        raw = self.read(name)
        if raw is None:
            return None
        try:
            return ET.fromstring(raw)
        except ET.ParseError as e:
            raise Unreadable(f"{name} inside the document is not valid XML ({e})") from None


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _rels(z: _Zip, part: str) -> dict[str, str]:
    """An OOXML part's relationships: id -> the target part's path in the ZIP."""
    folder, base = posixpath.split(part)
    root = z.xml(posixpath.join(folder, "_rels", base + ".rels"))
    out = {}
    if root is not None:
        for r in root:
            target = r.get("Target") or ""
            if r.get("TargetMode") == "External":
                continue
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
            out[r.get("Id")] = path
    return out


def _docx(z: _Zip) -> str:
    root = z.xml("word/document.xml")
    if root is None:
        raise Unreadable("a Word document without its text part (word/document.xml)")
    out: list[str] = []

    def para(p) -> str:
        bits = []
        for el in p.iter():
            tag = _local(el.tag)
            if tag == "t":                                # (deleted text, delText, is left out)
                bits.append(el.text or "")
            elif tag == "tab":
                bits.append("\t")
            elif tag in ("br", "cr"):
                bits.append("\n")
        return "".join(bits)

    def block(el) -> None:
        for child in el:
            tag = _local(child.tag)
            if tag == "p":
                out.append(para(child))
            elif tag == "tbl":
                for tr in child.iter():
                    if _local(tr.tag) == "tr":
                        cells = [" ".join(para(p) for p in tc.iter() if _local(p.tag) == "p").strip()
                                 for tc in tr if _local(tc.tag) == "tc"]
                        out.append(" | ".join(cells))
                out.append("")
            elif tag in ("sdt", "sdtContent", "customXml", "body"):
                block(child)

    block(root)
    return "\n".join(out)


def _xlsx(z: _Zip) -> str:
    shared = []
    ss = z.xml("xl/sharedStrings.xml")
    if ss is not None:
        for si in ss:
            if _local(si.tag) == "si":
                shared.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
    book = z.xml("xl/workbook.xml")
    if book is None:
        raise Unreadable("an Excel workbook without its sheet list (xl/workbook.xml)")
    rels = _rels(z, "xl/workbook.xml")
    out = []
    for sh in book.iter():
        if _local(sh.tag) != "sheet":
            continue
        rid = next((v for k, v in sh.attrib.items() if _local(k) == "id"), None)
        root = z.xml(rels.get(rid, ""))
        if root is None:
            continue
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for row in root.iter():
            if _local(row.tag) != "row":
                continue
            cells: list[str] = []
            for c in row:
                if _local(c.tag) != "c":
                    continue
                ref = re.match(r"([A-Z]+)", c.get("r") or "")
                if ref:                                   # empty cells skipped in the XML keep their column
                    col = 0
                    for ch in ref.group(1):
                        col = col * 26 + ord(ch) - 64
                    cells.extend([""] * max(0, col - 1 - len(cells)))
                kind, val = c.get("t"), None
                for el in c:
                    if _local(el.tag) == "v":
                        val = el.text or ""
                    elif _local(el.tag) == "is":
                        val = "".join(t.text or "" for t in el.iter() if _local(t.tag) == "t")
                if kind == "s" and val is not None:
                    try:
                        val = shared[int(val)]
                    except (ValueError, IndexError):
                        pass
                elif kind == "b" and val is not None:
                    val = "TRUE" if val == "1" else "FALSE"
                cells.append(val or "")
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                w.writerow(cells)
        out.append(f"## Sheet: {sh.get('name') or '?'}\n{buf.getvalue().rstrip()}")
    return "\n\n".join(out)


def _pptx(z: _Zip) -> str:
    pres = z.xml("ppt/presentation.xml")
    rels = _rels(z, "ppt/presentation.xml")
    order = []
    if pres is not None:
        for el in pres.iter():
            if _local(el.tag) == "sldId":
                rid = next((v for k, v in el.attrib.items() if _local(k) == "id" and k != "id"), None)
                if rid in rels:
                    order.append(rels[rid])
    if not order:
        order = sorted((n for n in z.names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                       key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
    out = []
    for k, part in enumerate(order, 1):
        root = z.xml(part)
        if root is None:
            continue
        paras = []
        for p in root.iter():
            if _local(p.tag) == "p":
                paras.append("".join(t.text or "" for t in p.iter() if _local(t.tag) == "t"))
        out.append(f"## Slide {k}\n" + "\n".join(s for s in paras if s.strip()))
    return "\n\n".join(out)


def _odf(z: _Zip) -> str:
    root = z.xml("content.xml")
    if root is None:
        raise Unreadable("an OpenDocument file without its content.xml")
    out: list[str] = []

    def text(el) -> str:
        bits = [el.text or ""]
        for c in el:
            tag = _local(c.tag)
            if tag == "s":
                bits.append(" " * int(next((v for k, v in c.attrib.items() if _local(k) == "c"), "1") or 1))
            elif tag == "tab":
                bits.append("\t")
            elif tag == "line-break":
                bits.append("\n")
            else:
                bits.append(text(c))
            bits.append(c.tail or "")
        return "".join(bits)

    def walk(el) -> None:
        for c in el:
            tag = _local(c.tag)
            if tag in ("p", "h"):
                out.append(text(c))
            elif tag == "table":
                name = next((v for k, v in c.attrib.items() if _local(k) == "name"), "")
                if name:
                    out.append(f"## {name}")
                for row in c.iter():
                    if _local(row.tag) == "table-row":
                        cells = []
                        for cell in row:
                            if _local(cell.tag) not in ("table-cell", "covered-table-cell"):
                                continue
                            rep = next((v for k, v in cell.attrib.items() if _local(k) == "number-columns-repeated"),
                                       "1")
                            val = "\n".join(text(p) for p in cell if _local(p.tag) in ("p", "h"))
                            cells.extend([val] * min(int(rep) if rep.isdigit() else 1, 64 if val else 1))
                        while cells and not cells[-1]:
                            cells.pop()
                        if cells:
                            out.append("\t".join(cells))
                out.append("")
            elif tag == "page":                            # a presentation's slide
                out.append(f"## {next((v for k, v in c.attrib.items() if _local(k) == 'name'), 'Slide')}")
                walk(c)
            else:
                walk(c)

    walk(root)
    return "\n".join(out)


class _Html(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote",
             "pre", "table", "ul", "ol", "dt", "dd", "hr", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(markup: str) -> str:
    p = _Html()
    p.feed(markup)
    p.close()
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.out))).strip()


def _epub(z: _Zip) -> str:
    cont = z.xml("META-INF/container.xml")
    opf = next((el.get("full-path") for el in cont.iter() if _local(el.tag) == "rootfile"), None) if cont is not None \
        else None
    root = z.xml(opf) if opf else None
    if root is None:
        raise Unreadable("an EPUB without its package file")
    folder = posixpath.dirname(opf)
    items = {el.get("id"): el.get("href") for el in root.iter() if _local(el.tag) == "item"}
    out = []
    for el in root.iter():
        if _local(el.tag) == "itemref" and el.get("idref") in items:
            raw = z.read(posixpath.normpath(posixpath.join(folder, items[el.get("idref")] or "")))
            if raw:
                out.append(html_text(raw.decode("utf-8", "replace")))
    return "\n\n".join(s for s in out if s)


def _rtf(data: bytes) -> str:
    """RTF to plain text: control words dropped, \\par and \\line as newlines, \\'hh and \\uN characters kept, the
    font / colour / style tables and other destinations skipped."""
    s, out, i = data.decode("latin-1"), [], 0
    stack, skip, uc = [], False, 1
    n = len(s)
    skip_dest = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "headerl", "headerr",
                 "footerl", "footerr", "listtable", "listoverridetable", "rsidtbl", "themedata", "colorschememapping",
                 "latentstyles", "datastore", "xmlnstbl", "generator", "fldinst", "object", "filetbl"}
    while i < n:
        c = s[i]
        if c == "{":
            stack.append((skip, uc))
            i += 1
        elif c == "}":
            skip, uc = stack.pop() if stack else (False, 1)
            i += 1
        elif c == "\\":
            m = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)", re.S).match(s, i)
            if not m:
                i += 1
                continue
            i = m.end()
            word, arg, hexc, sym = m.groups()
            if hexc and not skip:
                out.append(bytes([int(hexc, 16)]).decode("cp1252", "replace"))
            elif sym is not None:
                if sym == "*":
                    skip = True
                elif not skip and sym in "{}\\":
                    out.append(sym)
                elif not skip and sym == "~":
                    out.append(" ")
                elif not skip and sym in "\n\r":
                    out.append("\n")
            elif word:
                if word in skip_dest:
                    skip = True
                elif word == "uc" and arg:
                    uc = int(arg)
                elif skip:
                    pass
                elif word in ("par", "line", "row", "sect", "page"):
                    out.append("\n")
                elif word in ("tab", "cell"):
                    out.append("\t")
                elif word == "u" and arg:
                    out.append(chr(int(arg) % 65536))
                    i += uc                                  # the fallback characters after \uN
        elif c in "\r\n":
            i += 1
        else:
            j = i
            while j < n and s[j] not in "{}\\\r\n":
                j += 1
            if not skip:
                out.append(s[i:j])
            i = j
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


ZIP_KINDS = (  # (detector, format label, reader)
    (lambda z: "word/document.xml" in z.names, "Word document", _docx),
    (lambda z: "xl/workbook.xml" in z.names, "Excel workbook", _xlsx),
    (lambda z: "ppt/presentation.xml" in z.names, "PowerPoint presentation", _pptx),
    (lambda z: "content.xml" in z.names and (z.read("mimetype") or b"").startswith(b"application/vnd.oasis"),
     "OpenDocument file", _odf),
    (lambda z: (z.read("mimetype") or b"").strip() == b"application/epub+zip" or "META-INF/container.xml" in z.names,
     "EPUB book", _epub),
)

# what Maya does not read, by its first bytes: (magic, what it is, what to send instead)
NOT_READ = (
    (b"\xd0\xcf\x11\xe0", "an old binary Office file (.doc, .xls or .ppt)",
     "save it as .docx, .xlsx, .pptx or PDF and attach that"),
    (b"Rar!", "a RAR archive", "attach the files inside it"),
    (b"7z\xbc\xaf", "a 7-Zip archive", "attach the files inside it"),
    (b"\x1f\x8b", "a gzip archive", "attach the files inside it"),
    (b"MZ", "a Windows program", "attach its source or a text description"),
    (b"\x7fELF", "a Linux program", "attach its source or a text description"),
    (b"ID3", "an audio file", "none of Maya's models hear audio - attach a transcript"),
    (b"fLaC", "an audio file", "none of Maya's models hear audio - attach a transcript"),
    (b"OggS", "an audio file", "none of Maya's models hear audio - attach a transcript"),
    (b"GGUF", "a model file", "it is not something a chat reads"),
)


def _result(text: str, fmt: str, text_limit: int) -> dict:
    text = text.strip()
    if not text:
        return {"error": {"message": f"this {fmt} has no text in it"}}
    cut = len(text) > text_limit
    out = {"kind": "text", "format": fmt, "text": text[:text_limit], "truncated": cut}
    if cut:
        out["note"] = f"cut at {text_limit} characters - the text-file limit for this model's context"
    return out


def extract(name: str, data: bytes, *, want_images: bool, text_limit: int, text_only: str | None = None) -> dict:
    """Any attached file -> the chat page's attachment (see the module's doc).  text_only: the name of a model with
    no vision tower at all (GLM-5.3): its errors say it reads text only, never "when its images are on"."""
    try:
        return _extract(name, data, want_images=want_images, text_limit=text_limit, text_only=text_only)
    except Unreadable as e:
        return {"error": {"message": str(e)}}
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError) as e:
        return {"error": {"message": f"the file could not be read ({e})"}}


def _extract(name: str, data: bytes, *, want_images: bool, text_limit: int, text_only: str | None = None) -> dict:
    ext = _ext(name)
    if data.lstrip()[:5] == b"%PDF-" or (ext == "pdf" and b"%PDF-" in data[:1024]):
        out = pdf.extract(data[data.find(b"%PDF-"):], want_images=want_images, text_limit=text_limit,
                          **({"text_only": text_only} if text_only else {}))
        if "error" not in out:
            out["format"] = "PDF"
        return out
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
        z = _Zip(data)
        for detect, fmt, read in ZIP_KINDS:
            if detect(z):
                return _result(read(z), fmt, text_limit)
        raise Unreadable(f"{name or 'this file'} is a ZIP archive: attach the files inside it")
    if data.startswith(b"{\\rtf"):
        return _result(_rtf(data), "RTF document", text_limit)
    for magic, what, instead in NOT_READ:
        if data.startswith(magic):
            raise Unreadable(f"{name or 'this file'} is {what}: {instead}")
    if data[4:8] == b"ftyp" or data.startswith((b"\x1a\x45\xdf\xa3", b"RIFF")):
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            if text_only:
                raise Unreadable(f"{name or 'this file'} is a picture, and {text_only} reads text only: it has no "
                                 "vision tower to read it - describe the picture in text instead")
            raise Unreadable(f"{name or 'this file'} is a picture: attach it as one (the model reads pictures only "
                             "when its images are on)")
        if text_only:
            raise Unreadable(f"{name or 'this file'} is a video or audio file, and {text_only} reads text only")
        raise Unreadable(f"{name or 'this file'} is a video or audio file: videos go to the models that read them "
                         "(attach them as videos), and none of the models hear audio")
    text = as_text(data)
    if text is None:
        raise Unreadable(f"{name or 'this file'} is not a kind of file Maya reads ({'.' + ext if ext else 'binary'})"
                         f" - attach {READS}{'' if text_only else ', or pictures'}")
    return _result(text, "text file", text_limit)


# ------------------------------------------------------------------ the APIs' file parts
FILE_PARTS = ("file", "input_file", "document")


def _data_url(s: str) -> tuple[str, bytes] | None:
    m = re.match(r"data:([^;,]*)(;[^,]*)?,(.*)$", s or "", re.S)
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(3), validate=False) if ";base64" in (m.group(2) or "") else \
            m.group(3).encode()
    except (binascii.Error, ValueError):
        return None
    return m.group(1), raw


def _part_file(part: dict) -> tuple[str, bytes | None, str | None]:
    """(file name, bytes, ready text) of an OpenAI / Responses / Anthropic file part."""
    kind = part.get("type")
    if kind == "document":                                   # Anthropic
        src = part.get("source") or {}
        name = part.get("title") or "document"
        st = src.get("type")
        if st == "base64":
            try:
                return name, base64.b64decode(src.get("data") or ""), None
            except (binascii.Error, ValueError):
                raise ValueError("document.source.data is not base64") from None
        if st == "text":
            return name, None, str(src.get("data") or "")
        if st == "content":
            blocks = src.get("content")
            if isinstance(blocks, str):
                return name, None, blocks
            return name, None, "\n".join(b.get("text", "") for b in blocks or [] if isinstance(b, dict))
        raise ValueError(f"document.source.type {st!r}: Maya reads base64, text and content documents (not URLs "
                         "or file ids)")
    f = part.get("file") if kind == "file" and isinstance(part.get("file"), dict) else part   # OpenAI nests it
    name = f.get("filename") or "file"
    raw = f.get("file_data")
    if not raw:
        raise ValueError(f"{kind}: Maya reads files sent inline as file_data (not file_id or file_url)")
    parsed = _data_url(raw)
    if parsed is None:
        try:
            return name, base64.b64decode(raw), None
        except (binascii.Error, ValueError):
            raise ValueError(f"{kind}.file_data is neither a data: URL nor base64") from None
    return name, parsed[1], None


def file_block(name: str, text: str) -> str:
    """A file's text in the message the way the web chat writes it: its name, then the text fenced with more
    backticks than it holds itself."""
    longest = max([2] + [len(m) for m in re.findall(r"`+", text)])
    fence = "`" * (longest + 1)
    return f"File: {name}\n{fence}\n{text}\n{fence}"


def _image_part(api: str, url: str) -> dict:
    if api == "responses":
        return {"type": "input_image", "image_url": url}
    if api == "anthropic":
        mime, _, data = url[5:].partition(";base64,")
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
    return {"type": "image_url", "image_url": {"url": url}}


def expand_content(content, api: str, *, want_images: bool, text_limit: int, text_only: str | None = None):
    """A message's content with every file part replaced by its text part (and a scanned PDF's page pictures);
    content without file parts comes back as it was.  An unreadable file is a ValueError (the request's 400).
    text_only: as extract's (GLM-5.3: the errors say it reads text only)."""
    if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") in FILE_PARTS
                                                for p in content):
        return content
    text_type = "input_text" if api == "responses" else "text"
    out = []
    for part in content:
        if not (isinstance(part, dict) and part.get("type") in FILE_PARTS):
            out.append(part)
            continue
        name, raw, text = _part_file(part)
        if raw is not None:
            got = extract(name, raw, want_images=want_images, text_limit=text_limit,
                          **({"text_only": text_only} if text_only else {}))
            if "error" in got:
                raise ValueError(f"the attached file {name}: {got['error']['message']}")
            if got["kind"] == "images":
                out.append({"type": text_type, "text": f"File: {name} (its first {got['pages']} pages as pictures)"})
                out.extend(_image_part(api, im["data"]) for im in got["images"])
                continue
            text = got["text"]
        out.append({"type": text_type, "text": file_block(name, text[:text_limit])})
    return out


def expand_request(req: dict, api: str, *, want_images: bool, text_limit: int, text_only: str | None = None) -> None:
    """Every message's file parts read in place (chat completions and Anthropic: messages; Responses: input).
    text_only: as extract's."""
    if not isinstance(req, dict):
        return
    items = req.get("input") if api == "responses" else req.get("messages")
    if not isinstance(items, list):
        return
    for m in items:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            m["content"] = expand_content(m["content"], api, want_images=want_images, text_limit=text_limit,
                                          **({"text_only": text_only} if text_only else {}))
