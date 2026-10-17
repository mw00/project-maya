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


if __name__ == "__main__":
    unittest.main()
