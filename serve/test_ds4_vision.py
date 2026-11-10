"""DeepSeek-V4.1's pictures: the resize plan (DeepSeek's image_processor.py), the spans, the GENI file, the template.

    python -m unittest serve.test_ds4_vision -v
"""
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path

from serve import ds4_vision as DV

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("DS4_MODEL") or Path.home() / "models/DSV4")   # (the DeepSeek model's folder)
PACK = MODEL / "maya-ds4/tokenizer"
TEMPLATE = ROOT / "tools/ds4/chat_template.jinja"


class Plan(unittest.TestCase):
    def test_reference_plans(self):
        # (width, height) -> DeepSeek's plan_image_grid (n_h, n_w, best_height, best_width), from the reference code
        cases = {(800, 600): (15, 20, 602, 812), (100, 100): (13, 13, 546, 546), (4000, 3000): (27, 36, 1134, 1512),
                 (5000, 100): (3, 120, 112, 5012), (100, 5000): (120, 3, 5012, 112),
                 (1920, 1080): (23, 41, 966, 1708), (1036, 714): (17, 25, 714, 1036), (20000, 30): (1, 477, 42, 20006)}
        g = DV.Geometry()
        for (w, h), want in cases.items():
            with self.subTest(size=(w, h)):
                got = DV.plan(w, h, g)
                self.assertEqual(got, want)
                self.assertLessEqual(DV.span_tokens(got[0], got[1]), g.max_tokens)

    def test_prepare(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no Pillow")
        im = Image.new("RGBA", (300, 200), (255, 0, 0, 0))   # transparent: white, not black
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        pic = DV.prepare(buf.getvalue(), DV.Geometry())
        n_h, n_w, bh, bw = DV.plan(300, 200, DV.Geometry())
        self.assertEqual((pic.width, pic.height, pic.n_h, pic.n_w), (bw, bh, n_h, n_w))
        self.assertEqual(len(pic.rgb), bw * bh * 3)
        mid = ((bh // 2) * bw + bw // 2) * 3
        self.assertEqual(pic.rgb[mid:mid + 3], b"\xff\xff\xff")


class Spans(unittest.TestCase):
    def test_expand_and_file(self):
        g = DV.Geometry()
        v = DV.Vision(g, Path(tempfile.mkdtemp()), lambda s: b"")
        a = DV.Picture(b"\x01" * (28 * 42 * 3), 42, 28, 1, 1)
        b = DV.Picture(b"\x02" * (42 * 84 * 3), 84, 42, 1, 2)
        tok = g.image_token
        ids, starts = v.expand([5, tok, 6, tok, 7], [a, b])
        self.assertEqual(starts, [1, 1 + a.tokens + 1])
        self.assertEqual(ids, [5] + [tok] * a.tokens + [6] + [tok] * b.tokens + [7])
        with self.assertRaises(ValueError):
            v.expand([5, tok, 6], [a, b])
        with self.assertRaises(ValueError):
            v.expand([5, tok, tok, 6], [a])
        path = v.write(v.dir / "x.dsi", [a, b], starts)
        data = path.read_bytes()
        self.assertEqual(data[:8], struct.pack("<4si", b"DSI1", 2))
        self.assertEqual(struct.unpack_from("<qii", data, 8), (1, 42, 28))
        off = 8 + 16 + len(a.rgb)
        self.assertEqual(struct.unpack_from("<qii", data, off), (starts[1], 84, 42))
        self.assertEqual(len(data), off + 16 + len(b.rgb))


@unittest.skipUnless((PACK / "vocab.json").exists(), "no DeepSeek pack")
class Template(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        self.tok = ST.Tokenizer.from_dir(PACK)
        self.tpl = ChatTemplate(TEMPLATE)

    def test_deepseek_test_case(self):
        """encoding/tests/test_input_5.json -> test_output_5.txt (two pictures between text blocks)."""
        ref = MODEL / "encoding/tests/test_output_5.txt"
        if not ref.exists():
            self.skipTest("no encoding tests")
        msgs = [{"role": "system", "content": "You are a helpful vision assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "请按“第一张、第二张”的顺序回答：第一张图"}, {"type": "image", "source": "a"},
                    {"type": "text", "text": "和第二张图"}, {"type": "image", "source": "b"},
                    {"type": "text", "text": "中分别是什么食材？它们通常食用的部位分别是什么？"}]}]
        self.assertEqual(self.tpl.render(msgs, reasoning_effort="max"), ref.read_text(encoding="utf-8").rstrip("\n"))

    def test_typed_placeholder_is_text(self):
        """The placeholder typed in a message is plain text, not a picture (the server's literal marks)."""
        from serve.frontend import literal_tags, mark_literals, unmark_literals
        tags = literal_tags(self.tok.special_tokens)
        msgs = [{"role": "user", "content": "what is " + DV.PLACEHOLDER + "?"}]
        marked, _, changed = mark_literals(msgs, None, tags)
        self.assertTrue(changed)
        prompt, plain = unmark_literals(self.tpl.render(marked), tags)
        self.assertNotIn(DV.Geometry().image_token, self.tok.encode(prompt, parse_special=True, plain=plain))
        self.assertIn(DV.Geometry().image_token, self.tok.encode(prompt, parse_special=True))


class ImageUUIDs(unittest.TestCase):
    def test_uuid_reuse_miss_and_lru(self):
        from unittest.mock import patch
        loads = []
        def load(source):
            loads.append(source)
            return source.encode()
        with tempfile.TemporaryDirectory() as directory:
            vision = DV.Vision(DV.Geometry(), Path(directory), load)
            vision.IMAGE_IDS_MAX = 2
            with DV.image_scope(("test", "a")), patch.object(DV, "prepare", side_effect=lambda data, g: data):
                first = vision.picture({"source": "red", "uuid": "a"})
                self.assertEqual(vision.picture({"source": "blue", "uuid": "a"}), first)
                self.assertEqual(vision.picture({"source": None, "uuid": "a"}), first)
                self.assertEqual(loads, ["red"])
                vision.picture({"source": "blue", "uuid": "b"})
                vision.picture({"source": None, "uuid": "a"})
                vision.picture({"source": "green", "uuid": "c"})
                with self.assertRaisesRegex(DV.ImageCacheMiss, r"Cache miss for image at index 3 but data is not provided\."):
                    vision.picture({"source": None, "uuid": "b", "index": 3})
                self.assertEqual(len(vision.image_ids), 2)
                self.assertEqual(vision.picture("red"), first)

    def test_openai_uuid_roundtrip_and_error(self):
        from serve.frontend import openai_to_messages, deepseek_images
        from serve.ds41_encoding import _extract_image
        from serve.server import error_body
        req = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}, "uuid": "a"},
            {"type": "image_url", "image_url": None, "uuid": "a"}]}]}
        messages, _, _ = openai_to_messages(req, deepseek=True)
        images = deepseek_images(messages, with_ids=True)
        self.assertEqual(images, [{"source": "data:image/png;base64,abc", "uuid": "a", "index": 0},
                                  {"source": "", "uuid": "a", "index": 1}])
        self.assertEqual(_extract_image(req["messages"][0]["content"][1])["uuid"], "a")
        with self.assertRaises(ValueError):
            _extract_image({"type": "image_url", "image_url": None})
        message = "Cache miss for image at index 1 but data is not provided."
        self.assertEqual(error_body(DV.ImageCacheMiss(message)), {"error": {
            "type": "BadRequestError", "message": message, "param": None, "code": 400}})


class ImageUUIDHTTP(unittest.TestCase):
    def test_register_reuse_and_miss_before_streaming(self):
        import http.client
        import json
        import threading
        from unittest.mock import patch
        from serve.server import serve
        from serve.test_ds41_conformance import ServiceIntegration

        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegration().service("</think>OK")
            loads = []

            def load(source):
                loads.append(source)
                return source.encode()

            geometry = DV.Geometry(image_token=service.tok.special_tokens[DV.PLACEHOLDER])
            service.vision = DV.Vision(geometry, Path(directory), load)
            picture = DV.Picture(bytes(42 * 42 * 3), 42, 42, 1, 1)
            httpd = serve(service, port=0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()

            session = None
            def post(parts, stream):
                nonlocal session
                connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
                try:
                    request = {"model": "default", "stream": stream, "max_tokens": 16,
                               "chat_template_kwargs": {"enable_thinking": False},
                               "messages": [{"role": "user", "content": parts}]}
                    headers = {"Content-Type": "application/json"}
                    if session:
                        headers["X-Maya-Image-Session"] = session
                    connection.request("POST", "/v1/chat/completions", json.dumps(request), headers)
                    response = connection.getresponse()
                    session = response.getheader("X-Maya-Image-Session")
                    return response.status, response.getheader("Content-Type"), response.read().decode()
                finally:
                    connection.close()

            try:
                with patch.object(DV, "prepare", return_value=picture):
                    for stream in (False, True):
                        with self.subTest(stream=stream):
                            uid = "frame-" + str(stream)
                            full = {"type": "image_url", "image_url": {"url": "test-image"}, "uuid": uid}
                            known = {"type": "image_url", "image_url": None, "uuid": uid}
                            self.assertEqual(post([full], stream)[0], 200)
                            loads_before = list(loads)
                            self.assertEqual(post([known], stream)[0], 200)
                            self.assertEqual(loads, loads_before)
                            unknown = {"type": "image_url", "image_url": None, "uuid": "absent"}
                            status, content_type, raw = post([full, unknown], stream)
                            self.assertEqual(status, 400)
                            self.assertIn("application/json", content_type)
                            self.assertEqual(json.loads(raw), {"error": {"type": "BadRequestError",
                                "message": "Cache miss for image at index 1 but data is not provided.",
                                "param": None, "code": 400}})
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)


class ImageUUIDIsolationHTTP(unittest.TestCase):
    def setUp(self):
        import threading
        from unittest.mock import patch
        from serve.server import serve
        from serve.test_ds41_conformance import ServiceIntegration
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = ServiceIntegration().service("</think>OK")
        self.loads = []
        def load(source):
            self.loads.append(source)
            return source.encode()
        g = DV.Geometry(image_token=self.service.tok.special_tokens[DV.PLACEHOLDER])
        self.service.vision = DV.Vision(g, Path(self.directory.name), load)
        def prepare(data, geometry):
            return DV.Picture(bytes([data[0]]) * (42 * 42 * 3), 42, 42, 1, 1)
        self.patch = patch.object(DV, "prepare", side_effect=prepare)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.httpd = serve(self.service, port=0)
        self.addCleanup(self.stop)

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, source=None, key=None, session=None, stream=False, extra=None, path="/v1/chat/completions"):
        import http.client
        import json
        headers = {"Content-Type": "application/json"}
        if self.service.cors_origins:
            headers["Origin"] = "https://client.example"
        if key is not None:
            headers.update(key)
        if session is not None:
            headers["X-Maya-Image-Session"] = session
        req = {"stream": stream, "max_tokens": 16,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": source} if source else None, "uuid": "same-id"}]}]}
        req.update(extra or {})
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            c.request("POST", path, json.dumps(req), headers)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read().decode()
        finally:
            c.close()

    def assert_miss(self, result):
        import json
        status, headers, raw = result
        self.assertEqual(status, 400)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(json.loads(raw)["error"], {"type": "BadRequestError",
                         "message": "Cache miss for image at index 0 but data is not provided.",
                         "param": None, "code": 400})

    def test_keys_cannot_resolve_or_overwrite_each_other(self):
        self.service.api_key = ["key-a", "key-b"]
        a = {"Authorization": "Bearer key-a"}
        b = {"x-api-key": "key-b"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.service.vision.image_ids.clear()
                self.assertEqual(self.post("red", a, stream=stream)[0], 200)
                before = list(self.loads)
                self.assert_miss(self.post(key=b, stream=stream,
                    extra={"_image_scope": ["api-key", "key-a"],
                           "chat_template_kwargs": {"_image_scope": "key-a"}}, session="a" * 64))
                self.assertEqual(self.loads, before)
                self.assertEqual(self.post("blue", b, stream=stream)[0], 200)
                self.assertEqual(self.post(key=a, stream=stream)[0], 200)
                self.assertEqual(self.post(key=b, stream=stream)[0], 200)
        pictures = list(self.service.vision.image_ids.values())
        self.assertEqual(len(pictures), 2)
        self.assertEqual({p.rgb[0] for p in pictures}, {ord("r"), ord("b")})
        # Same authenticated key through the other header resolves the same entry.
        self.assertEqual(self.post(key={"x-api-key": "key-a"})[0], 200)
        self.assertEqual(self.post(key={"Authorization": "Bearer key-b"})[0], 200)
        self.assertEqual(self.post(key={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.post()[0], 401)
        self.assertEqual(len(self.service.vision.image_ids), 2)

    def test_no_key_conversations_and_new_connections_are_isolated(self):
        self.service.cors_origins = ["https://client.example"]
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, headers, _ = self.post("red", stream=stream)
                self.assertEqual(status, 200)
                session_a = headers["X-Maya-Image-Session"]
                self.assertRegex(session_a, r"^[0-9a-f]{64}$")
                self.assertIn("X-Maya-Image-Session", headers["Access-Control-Expose-Headers"])
                self.assertEqual(self.post(session=session_a, stream=stream)[0], 200)
                self.assert_miss(self.post(stream=stream))
                self.assert_miss(self.post(session="b" * 64, stream=stream,
                    extra={"conversation_id": session_a, "_image_scope": session_a}))
                status, headers, _ = self.post("blue", stream=stream)
                self.assertEqual(status, 200)
                session_b = headers["X-Maya-Image-Session"]
                self.assertNotEqual(session_a, session_b)
                self.assertEqual(self.post(session=session_b, stream=stream)[0], 200)
                self.assertEqual(self.post(session=session_a, stream=stream)[0], 200)
        self.assertEqual(len(self.service.vision.image_ids), 4)

    def test_count_tokens_cannot_register_for_another_key(self):
        self.service.api_key = ["key-a", "key-b"]
        a, b = {"x-api-key": "key-a"}, {"x-api-key": "key-b"}
        self.assertEqual(self.post("red", a, path="/api/tokens")[0], 200)
        self.assert_miss(self.post(key=b))
        self.assertEqual(self.post(key=a)[0], 200)

    def test_context_isolation_on_concurrent_requests(self):
        from concurrent.futures import ThreadPoolExecutor
        self.service.api_key = ["key-a", "key-b"]
        a, b = {"x-api-key": "key-a"}, {"x-api-key": "key-b"}
        self.assertEqual(self.post("red", a)[0], 200)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [(key, pool.submit(self.post, key=key)) for key in [a, b] * 4]
            for key, future in futures:
                if key == a:
                    self.assertEqual(future.result()[0], 200)
                else:
                    self.assert_miss(future.result())
        self.assertEqual(len(self.service.vision.image_ids), 1)


class ImageUUIDScope(unittest.TestCase):
    def test_resumable_stream_keeps_its_image_namespace(self):
        import threading
        from unittest.mock import patch
        from serve.server import StreamJob
        with tempfile.TemporaryDirectory() as directory:
            v = DV.Vision(DV.Geometry(), Path(directory), lambda source: source.encode())
            def chunks():
                try:
                    value = v.picture({"source": None, "uuid": "id"}).decode()
                except DV.ImageCacheMiss:
                    value = "miss"
                yield {"result": value}
            with patch.object(DV, "prepare", side_effect=lambda data, g: data), \
                    patch("serve.server.job_file", return_value=None):
                with DV.image_scope(("key", "a")):
                    v.picture({"source": "red", "uuid": "id"})
                    a = StreamJob("scope-a", chunks(), threading.Event())
                with DV.image_scope(("key", "b")):
                    b = StreamJob("scope-b", chunks(), threading.Event())
                for job, result in [(a, "red"), (b, "miss")]:
                    with job.cv:
                        self.assertTrue(job.cv.wait_for(lambda: job.done, timeout=3))
                    self.assertEqual(job.items, [{"result": result}])
                self.assertIsNone(DV.IMAGE_SCOPE.get())

    def test_unscoped_direct_calls_fail_closed_and_scope_resets(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            v = DV.Vision(DV.Geometry(), Path(directory), lambda source: source.encode())
            with patch.object(DV, "prepare", side_effect=lambda data, g: data):
                v.picture({"source": "red", "uuid": "id"})
                self.assertEqual(len(v.image_ids), 0)
                with self.assertRaises(DV.ImageCacheMiss):
                    v.picture({"source": None, "uuid": "id"})
                with DV.image_scope(("key", "a")):
                    v.picture({"source": "red", "uuid": "id"})
                    with DV.image_scope(("key", "b")):
                        with self.assertRaises(DV.ImageCacheMiss):
                            v.picture({"source": None, "uuid": "id"})
                    self.assertEqual(v.picture({"source": None, "uuid": "id"}), b"red")
                self.assertIsNone(DV.IMAGE_SCOPE.get())
                with self.assertRaises(DV.ImageCacheMiss):
                    v.picture({"source": None, "uuid": "id"})


if __name__ == "__main__":
    unittest.main()
