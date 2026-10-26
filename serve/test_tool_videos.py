"""Videos ride the image pipeline: OpenAI and Anthropic video parts survive conversion, rendering and prompt
preparation - every frame of a video becomes an independent image between the video's markers.

    python -m unittest serve.test_tool_videos -v
"""
import contextlib
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, anthropic_to_messages, media_of, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


def image(data="aW1hZ2U="):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def video(data="dmlkZW8="):
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}}


def anthropic_video(data="dmlkZW8="):
    return {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": data}}


class ToolVideos(unittest.TestCase):
    def test_openai_video_url_part(self):
        messages, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "watch"}, video()]}]})
        self.assertEqual(messages[-1]["content"], [
            {"type": "text", "text": "watch"},
            {"type": "video", "source": "data:video/mp4;base64,dmlkZW8="}])
        self.assertEqual(media_of(messages), [("video", "data:video/mp4;base64,dmlkZW8=")])

    def test_openai_video_url_shorthand(self):
        messages, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "video_url", "video_url": "clip.mp4"}]}]})
        self.assertEqual(media_of(messages), [("video", "clip.mp4")])

    def test_anthropic_video_source(self):
        messages, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": [anthropic_video()]}]})
        self.assertEqual(media_of(messages), [("video", "data:video/mp4;base64,dmlkZW8=")])

    def test_media_prompt_order(self):
        messages, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [video(), {"type": "text", "text": "next"}, image()]},
            {"role": "assistant", "content": [{"type": "video_url", "video_url": {"url": "clip.mp4"}}]}]})
        self.assertEqual(media_of(messages), [("video", "data:video/mp4;base64,dmlkZW8="),
                                              ("image", "data:image/png;base64,aW1hZ2U="),
                                              ("video", "clip.mp4")])

    def test_templates_render_and_prepare_in_media_order(self):
        class Tokenizer(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + ["<|begin_of_image|>", "<|image|>", "<|end_of_image|>",
                                                 "<|begin_of_video|>", "<|video|>", "<|end_of_video|>",
                                                 "<|video_pad|>"]

        class Vision:
            owed = False

            def __init__(self, directory):
                self.dir = directory
                self.sources = []

            def active(self):
                return contextlib.nullcontext()

            def _record(self, source):
                self.sources.append(source)
                path = self.dir / f"media-{len(self.sources)}.sve"
                path.write_bytes(bytes([len(self.sources)]))
                return path

            def encode(self, source):
                return self._record(source), 3

            def encode_video(self, source):
                return self._record(source), [2, 5]

        req = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": "watch"}, video()]},
            {"role": "assistant", "content": [{"type": "text", "text": "sure"}]},
            {"role": "user", "content": [image(), {"type": "text", "text": "compare"}]},
        ]}
        messages, tools, kwargs = openai_to_messages(req)
        for template_path, pad, start, end in (
                ("serve/chat_template.jinja", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"),
                ("serve/fixtures/glm_tool_images.jinja", "<|image|>", "<|begin_of_image|>", "<|end_of_image|>")):
            with self.subTest(template=template_path), tempfile.TemporaryDirectory() as directory:
                template = ChatTemplate(ROOT / template_path)
                rendered = template.render(messages, tools=tools, **kwargs)
                self.assertIn("watch", rendered)
                self.assertIn(start + pad + end, rendered)
                tok = Tokenizer()
                vision = Vision(Path(directory))
                svc = Service(MockEngine(tok, "ok", max_context=8192), tok, template, vision=vision)
                ids, _, _ = svc.prepare(messages, tools, dict(kwargs), 10)
                self.assertEqual(vision.sources, ["data:video/mp4;base64,dmlkZW8=",
                                                  "data:image/png;base64,aW1hZ2U="])
                expanded = tok.decode(ids)
                # the video's frames, every frame an independent image between its markers, before the image
                self.assertIn(start + pad * 2 + end, expanded)
                self.assertIn(start + pad * 5 + end, expanded)
                self.assertLess(expanded.index(start + pad * 2 + end), expanded.index(start + pad * 5 + end))
                self.assertEqual(svc.embeddings.path.read_bytes(), b"\x01\x02")


class VideoGuards(unittest.TestCase):
    """What reaches ffmpeg: real video containers only, ffmpeg itself there, a long video refused before decoding."""

    def vision(self, directory, data):
        import threading
        from serve.server import Vision
        v = object.__new__(Vision)
        v.dir, v.lock, v.cache, v.proc = Path(directory), threading.Lock(), {}, None
        v.load = lambda source: data
        v._start = lambda: (_ for _ in ()).throw(AssertionError("the encoder must not start"))
        return v

    def test_containers(self):
        from serve.server import Vision
        self.assertEqual(Vision.video_container(b"\0\0\0\x18ftypisom" + b"\0" * 8), "mp4")
        self.assertEqual(Vision.video_container(b"\x1a\x45\xdf\xa3" + b"\0" * 8), "webm")
        self.assertEqual(Vision.video_container(b"RIFF\0\0\0\0AVI LIST"), "avi")
        for bad in (b"#EXTM3U\n#EXT-X-VERSION:3\nfile:///etc/passwd\n", b"ffconcat version 1.0\nfile '/x'\n",
                    b"\x89PNG\r\n\x1a\n", b""):
            self.assertIsNone(Vision.video_container(bad))

    def test_a_playlist_never_reaches_ffmpeg(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d, patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            v = self.vision(d, b"#EXTM3U\nfile:///etc/passwd\n")
            with self.assertRaisesRegex(ValueError, "MP4, MOV, WebM, MKV or AVI"):
                v.encode_video("data:video/mp4;base64,AA==")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_no_ffmpeg(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d, patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "need ffmpeg and ffprobe"):
                self.vision(d, b"\0\0\0\x18ftypisom").encode_video("clip.mp4")

    def test_too_long(self):
        from unittest.mock import patch
        from serve.server import Vision
        with tempfile.TemporaryDirectory() as d, patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                patch.object(Vision, "video_seconds", staticmethod(lambda path: 600.0)), \
                patch.dict("os.environ", {"STRATA_VIDEO_MAX_FRAMES": "240"}):
            with self.assertRaisesRegex(ValueError, r"600 s long: 2400 frames"):
                self.vision(d, b"\0\0\0\x18ftypisom").encode_video("clip.mp4")
            self.assertEqual(list(Path(d).iterdir()), [])   # (the temporary file removed)


if __name__ == "__main__":
    unittest.main()
