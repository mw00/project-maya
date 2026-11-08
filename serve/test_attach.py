"""serve/test_attach.py - any file the chat attaches (serve/attach.py, serve/pdf_text.py, #837): Office, OpenDocument,
EPUB and RTF documents come back as their text, text in other encodings as text, a PDF is read without poppler (text
layer, ToUnicode fonts, object streams, a scanned page's own picture), what is not read says what it is, and the
APIs' file parts (OpenAI file, Responses input_file, Anthropic document) reach the model as text.  Every fixture is
written by the test itself; no GPU, no poppler.

    python -m unittest serve.test_attach -v
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import sys
import unittest
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import attach, pdf_text    # noqa: E402
from serve.frontend import ChatTemplate    # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 512 * 1024
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
REL = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'


def _zip(files: dict[str, str | bytes], stored_first: str | None = None) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        if stored_first:
            z.writestr(zipfile.ZipInfo(stored_first), files.pop(stored_first), compress_type=zipfile.ZIP_STORED)
        for name, body in files.items():
            z.writestr(name, body)
    return b.getvalue()


def docx() -> bytes:
    body = (f'<w:document {W}><w:body>'
            '<w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p>'
            '<w:p><w:r><w:t xml:space="preserve">Revenue grew </w:t></w:r><w:r><w:t>12%</w:t></w:r>'
            '<w:del><w:r><w:delText>deleted words</w:delText></w:r></w:del><w:r><w:tab/><w:t>tabbed</w:t></w:r></w:p>'
            '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Sales</w:t></w:r>'
            '</w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>North</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r>'
            '<w:t>42</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
            '<w:p><w:r><w:t>Done.</w:t></w:r></w:p></w:body></w:document>')
    return _zip({"[Content_Types].xml": "<Types/>", "word/document.xml": body})


def xlsx() -> bytes:
    return _zip({
        "[Content_Types].xml": "<Types/>",
        "xl/workbook.xml": f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" {R}>'
                           '<sheets><sheet name="Budget" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": f'<Relationships {REL}><Relationship Id="rId1" Type="x" '
                                      'Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                '<si><t>Item</t></si><si><t>Cost, EUR</t></si><si><t>Rent</t></si></sst>',
        "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                    '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
                                    '</row><row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>950.5</v></c>'
                                    '<c r="D2" t="b"><v>1</v></c><c r="E2" t="inlineStr"><is><t>note</t></is></c>'
                                    '</row></sheetData></worksheet>'})


def pptx() -> bytes:
    slide = ('<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
             'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody>'
             '<a:p><a:r><a:t>{0}</a:t></a:r></a:p><a:p><a:r><a:t>{1}</a:t></a:r></a:p></p:txBody></p:sp>'
             '</p:spTree></p:cSld></p:sld>')
    return _zip({
        "[Content_Types].xml": "<Types/>",
        "ppt/presentation.xml": f'<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                                f'{R}><p:sldIdLst><p:sldId id="256" r:id="rId3"/><p:sldId id="257" r:id="rId2"/>'
                                '</p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": f'<Relationships {REL}><Relationship Id="rId2" Type="x" '
                                           'Target="slides/slide1.xml"/><Relationship Id="rId3" Type="x" '
                                           'Target="slides/slide2.xml"/></Relationships>',
        "ppt/slides/slide1.xml": slide.format("Second in order", "B point"),
        "ppt/slides/slide2.xml": slide.format("Title slide", "A point")})


ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"')


def odt() -> bytes:
    return _zip({"mimetype": "application/vnd.oasis.opendocument.text",
                 "content.xml": f'<office:document-content {ODF_NS}><office:body><office:text>'
                                '<text:h>Heading</text:h><text:p>Hello<text:s text:c="2"/>world<text:tab/>x</text:p>'
                                '<table:table table:name="T1"><table:table-row><table:table-cell><text:p>a</text:p>'
                                '</table:table-cell><table:table-cell table:number-columns-repeated="2"><text:p>b'
                                '</text:p></table:table-cell></table:table-row></table:table>'
                                '</office:text></office:body></office:document-content>'},
                stored_first="mimetype")


def epub() -> bytes:
    return _zip({"mimetype": "application/epub+zip",
                 "META-INF/container.xml": '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles>'
                                           '</container>',
                 "OEBPS/book.opf": '<package><manifest><item id="c1" href="ch1.xhtml"/><item id="c2" href="ch2.xhtml"/>'
                                   '</manifest><spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>',
                 "OEBPS/ch1.xhtml": "<html><head><title>x</title><style>p{}</style></head><body><p>Chapter one"
                                    "</p></body></html>",
                 "OEBPS/ch2.xhtml": "<html><body><h1>Opening</h1><p>Call me &amp; friends.</p></body></html>"},
                stored_first="mimetype")


def ex(name: str, data: bytes, **kw) -> dict:
    return attach.extract(name, data, want_images=kw.get("want_images", True), text_limit=kw.get("limit", LIMIT))


class Documents(unittest.TestCase):
    def test_word(self):
        r = ex("report.docx", docx())
        self.assertEqual((r["kind"], r["format"]), ("text", "Word document"))
        self.assertIn("Revenue grew 12%\ttabbed", r["text"])
        self.assertIn("Region | Sales\nNorth | 42", r["text"])
        self.assertNotIn("deleted words", r["text"])
        self.assertTrue(r["text"].startswith("Quarterly report"))

    def test_excel(self):
        r = ex("budget.xlsx", xlsx())
        self.assertEqual(r["format"], "Excel workbook")
        self.assertIn("## Sheet: Budget", r["text"])
        self.assertIn('Item,"Cost, EUR"', r["text"])
        self.assertIn("Rent,,950.5,TRUE,note", r["text"])      # (B2 empty: its column kept)

    def test_powerpoint_in_presentation_order(self):
        r = ex("deck.pptx", pptx())
        self.assertEqual(r["format"], "PowerPoint presentation")
        self.assertLess(r["text"].index("Title slide"), r["text"].index("Second in order"))
        self.assertIn("## Slide 1\nTitle slide\nA point", r["text"])

    def test_opendocument(self):
        r = ex("notes.odt", odt())
        self.assertEqual(r["format"], "OpenDocument file")
        self.assertIn("Heading\nHello  world\tx", r["text"])
        self.assertIn("## T1\na\tb\tb", r["text"])

    def test_epub_in_spine_order(self):
        r = ex("book.epub", epub())
        self.assertEqual(r["format"], "EPUB book")
        self.assertLess(r["text"].index("Opening"), r["text"].index("Chapter one"))
        self.assertIn("Call me & friends.", r["text"])
        self.assertNotIn("p{}", r["text"])

    def test_rtf(self):
        rtf = (rb"{\rtf1\ansi{\fonttbl{\f0 Arial;}}{\colortbl;\red0\green0\blue0;}"
               rb"\f0 Caf\'e9 au lait\par Second \b bold\b0  line \u8364? euro\par}")
        r = ex("memo.rtf", rtf)
        self.assertEqual(r["format"], "RTF document")
        self.assertEqual(r["text"], "Café au lait\nSecond bold line € euro")

    def test_cut_at_the_limit(self):
        r = ex("report.docx", docx(), limit=20)
        self.assertEqual((len(r["text"]), r["truncated"]), (20, True))
        self.assertIn("cut at 20", r["note"])


class Text(unittest.TestCase):
    def test_any_name_utf8(self):
        r = ex("Notes.weird", "plain words, ünïcode".encode())
        self.assertEqual((r["kind"], r["text"]), ("text", "plain words, ünïcode"))

    def test_utf16_with_and_without_bom(self):
        self.assertEqual(ex("a.txt", "héllo wörld".encode("utf-16"))["text"], "héllo wörld")
        self.assertEqual(ex("a.log", "plain ascii log line".encode("utf-16-le"))["text"], "plain ascii log line")

    def test_latin1(self):
        self.assertEqual(ex("old.csv", "naïve;café\n".encode("cp1252"))["text"], "naïve;café")

    def test_binary_says_what_it_is(self):
        self.assertIn("old binary Office file", ex("x.doc", b"\xd0\xcf\x11\xe0" + bytes(500))["error"]["message"])
        self.assertIn("ZIP archive", ex("x.zip", _zip({"a.bin": b"\0\1"}))["error"]["message"])
        self.assertIn("audio", ex("x.mp3", b"ID3\x04" + bytes(100))["error"]["message"])
        m = ex("blob.dat", bytes(range(256)) * 8)["error"]["message"]
        self.assertIn("not a kind of file Maya reads (.dat)", m)
        self.assertIn("Word, Excel and PowerPoint", m)

    def test_empty_document_says_so(self):
        self.assertIn("no text", ex("e.docx", _zip({"word/document.xml": f"<w:document {W}/>"}))["error"]["message"])

    def test_zip_bomb_capped(self):
        big = _zip({"word/document.xml": f"<w:document {W}><w:body><w:p><w:r><w:t>" + "a" * 2000 + "</w:t></w:r>"
                    "</w:p></w:body></w:document>"})
        with mock.patch.object(attach, "MEMBER_MAX", 1000):
            self.assertIn("unpacks to more", ex("b.docx", big)["error"]["message"])


# ------------------------------------------------------------------ PDFs without poppler
def _pdf(objs: list[bytes], root: int = 1, trailer_extra: str = "") -> bytes:
    out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root {root} 0 R {trailer_extra}>>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _stream(body: bytes, extra: str = "", flate: bool = True) -> bytes:
    data = zlib.compress(body) if flate else body
    return (f"<< /Length {len(data)} {'/Filter /FlateDecode ' if flate else ''}{extra}>>\nstream\n".encode() + data +
            b"\nendstream")


TOUNICODE = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n1 begincodespacerange <0000> <FFFF> "
             b"endcodespacerange\n2 beginbfchar\n<0003> <0020>\n<0010> <00E9>\nendbfchar\n1 beginbfrange\n"
             b"<0024> <003D> <0041>\nendbfrange\nendcmap end end")


class PurePdf(unittest.TestCase):
    """serve/pdf_text.py on hand-written PDFs: what a Word or LaTeX export looks like inside, in miniature."""

    def test_identity_font_through_tounicode(self):
        # glyph ids 0x24.. are A.. (bfrange), 0x03 a space, 0x10 an e-acute; Flate content, a hex TJ with kerning
        content = b"BT /F1 12 Tf 72 700 Td [<0024002500260003>-50<0027>] TJ 0 -14 Td <00280010> Tj ET"
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>",
                    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                    b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
                    b"<< /Type /Font /Subtype /Type0 /Encoding /Identity-H /ToUnicode 6 0 R >>",
                    _stream(content), _stream(TOUNICODE)])
        self.assertEqual(pdf_text.text_pages(pdf), ["ABC D\nEé"])

    def test_object_stream_and_differences(self):
        # the page tree and the font live in a compressed object stream (PDF 1.5); the font remaps code 1 to "fi"
        inner = [b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                 b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
                 b"<< /Type /Font /Subtype /Type1 /Encoding << /Differences [1 /fi 2 /eacute] >> >>"]
        heads, body, off = [], b"", 0
        for k, o in enumerate(inner):
            heads.append(f"{k + 2} {off}")
            body += o + b" "
            off = len(body)
        head = " ".join(heads).encode() + b" "
        objstm = _stream(head + body, f"/Type /ObjStm /N 3 /First {len(head)} ")
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>", b"null", b"null", b"null",
                    _stream(b"BT /F1 10 Tf (\\001nal caf\\002) Tj T* (next) ' ET"), objstm])
        pdf = pdf.replace(b"2 0 obj\nnull", b"% 2 obj\nnull").replace(b"3 0 obj\nnull", b"% 3 obj\nnull")
        pdf = pdf.replace(b"4 0 obj\nnull", b"% 4 obj\nnull")
        self.assertEqual(pdf_text.text_pages(pdf), ["final café\nnext"])

    def test_scanned_page_pictures(self):
        jpeg = b"\xff\xd8\xff\xe0 a jpeg stand-in \xff\xd9"
        gray = bytes(range(16)) * 16                       # 16 x 16, 8-bit
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>",
                    b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
                    b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 5 0 R >> >> /Contents 7 0 R >>",
                    b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 6 0 R >> >> /Contents 7 0 R >>",
                    _stream(jpeg, "/Type /XObject /Subtype /Image /Width 8 /Height 8 /ColorSpace /DeviceRGB "
                                  "/BitsPerComponent 8 /Filter /DCTDecode ", flate=False),
                    _stream(gray, "/Type /XObject /Subtype /Image /Width 16 /Height 16 /ColorSpace /DeviceGray "
                                  "/BitsPerComponent 8 "),
                    _stream(b"q 612 0 0 792 0 0 cm /Im0 Do Q")])
        pics = pdf_text.page_pictures(pdf, 6)
        self.assertEqual([m for m, _ in pics], ["image/jpeg", "image/png"])
        self.assertEqual(pics[0][1], jpeg)
        png = pics[1][1]
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (16, 16))
        idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
        self.assertEqual(zlib.decompress(idat), b"".join(b"\0" + gray[r * 16:(r + 1) * 16] for r in range(16)))
        # through serve/pdf.py with no poppler on PATH: the pages as data URLs
        with mock.patch.object(shutil, "which", return_value=None):
            r = attach.extract("scan.pdf", pdf, want_images=True, text_limit=LIMIT)
        self.assertEqual((r["kind"], r["pages"], r["format"]), ("images", 2, "PDF"))
        self.assertTrue(r["images"][0]["data"].startswith("data:image/jpeg;base64,"))

    def test_encrypted_says_so(self):
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [] /Count 0 >>",
                    b"<< /Filter /Standard /V 2 >>"], trailer_extra="/Encrypt 3 0 R ")
        with mock.patch.object(shutil, "which", return_value=None):
            r = attach.extract("locked.pdf", pdf, want_images=True, text_limit=LIMIT)
        self.assertIn("encrypted", r["error"]["message"])

    def test_inline_image_skipped(self):
        content = b"BT /F1 12 Tf (before) Tj ET BI /W 2 /H 2 /BPC 8 /CS /G ID \x00(\x29)Tj EI BT (after) Tj ET"
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                    b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
                    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", _stream(content)])
        self.assertEqual(pdf_text.text_pages(pdf), ["before after"])


# ------------------------------------------------------------------ the APIs' file parts
class ApiParts(unittest.TestCase):
    def kw(self, **k):
        return {"want_images": True, "text_limit": LIMIT, **k}

    def test_openai_file_part(self):
        url = "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," + \
            base64.b64encode(docx()).decode()
        out = attach.expand_content([{"type": "text", "text": "Summarise"},
                                     {"type": "file", "file": {"filename": "r.docx", "file_data": url}}],
                                    "openai", **self.kw())
        self.assertEqual(out[0], {"type": "text", "text": "Summarise"})
        self.assertEqual(out[1]["type"], "text")
        self.assertTrue(out[1]["text"].startswith("File: r.docx\n```\nQuarterly report"))

    def test_responses_input_file(self):
        out = attach.expand_content([{"type": "input_file", "filename": "n.txt",
                                      "file_data": base64.b64encode(b"some notes").decode()}],
                                    "responses", **self.kw())
        self.assertEqual(out, [{"type": "input_text", "text": "File: n.txt\n```\nsome notes\n```"}])

    def test_anthropic_documents(self):
        out = attach.expand_content([{"type": "document", "title": "a.txt",
                                      "source": {"type": "text", "media_type": "text/plain", "data": "x ``` y"}}],
                                    "anthropic", **self.kw())
        self.assertEqual(out[0]["text"], "File: a.txt\n````\nx ``` y\n````")

    def test_anthropic_scanned_pdf_pages_as_image_blocks(self):
        jpeg = b"\xff\xd8\xff\xe0 stand-in \xff\xd9"
        pdf = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                    b"<< /Type /Page /Parent 2 0 R /Resources << /XObject << /I 4 0 R >> >> /Contents 5 0 R >>",
                    _stream(jpeg, "/Subtype /Image /Width 4 /Height 4 /Filter /DCTDecode ", flate=False),
                    _stream(b"/I Do")])
        with mock.patch.object(shutil, "which", return_value=None):
            out = attach.expand_content([{"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf).decode()}}],
                "anthropic", **self.kw())
        self.assertEqual(out[1], {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                              "data": base64.b64encode(jpeg).decode()}})

    def test_unreadable_and_by_reference_are_errors(self):
        with self.assertRaisesRegex(ValueError, "file_data"):
            attach.expand_content([{"type": "file", "file": {"file_id": "file-1"}}], "openai", **self.kw())
        with self.assertRaisesRegex(ValueError, "the attached file x.bin"):
            attach.expand_content([{"type": "file", "file": {"filename": "x.bin", "file_data":
                                    base64.b64encode(bytes(range(256))).decode()}}], "openai", **self.kw())

    def test_content_without_files_untouched(self):
        content = [{"type": "text", "text": "hi"}]
        self.assertIs(attach.expand_content(content, "openai", **self.kw()), content)


class Route(unittest.TestCase):
    """POST /attach and the APIs' file parts against the mock engine (no GPU, no vision)."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, path, body: bytes, headers: dict):
        r = urllib.request.Request(self.base + path, method="POST", headers=headers, data=body)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_attach_a_word_document(self):
        s, j = self.call("/attach", docx(), {"Content-Type": "application/octet-stream",
                                             "X-File-Name": "Q3%20report%C3%A9.docx"})
        self.assertEqual((s, j["kind"], j["format"]), (200, "text", "Word document"))
        self.assertIn("North | 42", j["text"])

    def test_foreign_page_refused(self):
        s, j = self.call("/attach", b"hi", {"Content-Type": "application/octet-stream", "Origin": "http://evil.example"})
        self.assertEqual(s, 403)

    def test_form_types_refused(self):
        s, _ = self.call("/attach", b"a=b", {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(s, 415)

    def test_health_says_videos(self):
        with urllib.request.urlopen(self.base + "/health", timeout=30) as resp:
            self.assertIs(json.loads(resp.read())["videos"], False)        # (the mock engine has no vision)

    def test_chat_completion_with_a_file_part(self):
        url = "data:text/plain;base64," + base64.b64encode(b"the secret word is pineapple").decode()
        body = json.dumps({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is the word?"},
            {"type": "file", "file": {"filename": "w.txt", "file_data": url}}]}]}).encode()
        s, j = self.call("/v1/chat/completions", body, {"Content-Type": "application/json"})
        self.assertEqual(s, 200, j)
        s, j = self.call("/api/tokens", json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
                         {"Content-Type": "application/json"})
        self.assertEqual(s, 200)

    def test_bad_file_part_is_a_400(self):
        body = json.dumps({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "file", "file": {"file_id": "file-abc"}}]}]}).encode()
        s, j = self.call("/v1/chat/completions", body, {"Content-Type": "application/json"})
        self.assertEqual(s, 400)
        self.assertIn("file_data", json.dumps(j))


if __name__ == "__main__":
    unittest.main()
