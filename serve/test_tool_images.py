"""Anthropic Read tool images survive conversion, rendering and prompt preparation.

    python -m unittest serve.test_tool_images -v
"""
import contextlib
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, anthropic_to_messages, images_of
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


def image(data="Ymx1ZQ=="):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def request(content):
    return {"messages": [
        {"role": "user", "content": "Read the image"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read",
                                           "input": {"file_path": "blue.png"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": content}]},
    ]}


class ToolImages(unittest.TestCase):
    def test_image_only_result(self):
        messages, _, _ = anthropic_to_messages(request([image()]))
        self.assertEqual(messages[-1], {"role": "tool", "content": [
            {"type": "image", "source": "data:image/png;base64,Ymx1ZQ=="}]})
        self.assertEqual(images_of(messages), ["data:image/png;base64,Ymx1ZQ=="])

    def test_text_only_result_stays_a_string(self):
        for content in ("plain", [{"type": "text", "text": "plain"}]):
            with self.subTest(content=content):
                messages, _, _ = anthropic_to_messages(request(content))
                self.assertEqual(messages[-1], {"role": "tool", "content": "plain"})
                self.assertEqual(images_of(messages), [])

    def test_text_and_image_order(self):
        messages, _, _ = anthropic_to_messages(request([{"type": "text", "text": "before"}, image(),
                                                       {"type": "text", "text": "after"}]))
        self.assertEqual(messages[-1]["content"], [{"type": "text", "text": "before"},
            {"type": "image", "source": "data:image/png;base64,Ymx1ZQ=="},
            {"type": "text", "text": "after"}])

    def test_templates_render_and_prepare_in_image_order(self):
        class Tokenizer(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + ["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"]

        class Vision:
            owed = False

            def __init__(self, directory):
                self.dir = directory
                self.sources = []

            def active(self):
                return contextlib.nullcontext()

            def encode(self, source):
                self.sources.append(source)
                path = self.dir / f"image-{len(self.sources)}.sve"
                path.write_bytes(bytes([len(self.sources)]))
                return path, len(self.sources) + 2

        req = request([{"type": "text", "text": "before"}, image()])
        req["messages"].append({"role": "user", "content": [image("cmVk"), {"type": "text", "text": "compare"}]})
        messages, tools, kwargs = anthropic_to_messages(req)
        for template_path, pad, start, end in (
                ("serve/chat_template.jinja", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"),
                ("serve/fixtures/glm_tool_images.jinja", "<|image|>", "<|begin_of_image|>", "<|end_of_image|>")):
            with self.subTest(template=template_path), tempfile.TemporaryDirectory() as directory:
                template = ChatTemplate(ROOT / template_path)
                rendered = template.render(messages, tools=tools, **kwargs)
                response = rendered.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
                self.assertIn("before", response)
                self.assertIn(start + pad + end, response)
                tok = Tokenizer()
                vision = Vision(Path(directory))
                svc = Service(MockEngine(tok, "ok", max_context=8192), tok, template, vision=vision)
                ids, _, _ = svc.prepare(messages, tools, dict(kwargs), 10)
                self.assertEqual(vision.sources, ["data:image/png;base64,Ymx1ZQ==", "data:image/png;base64,cmVk"])
                expanded = tok.decode(ids)
                self.assertIn(start + pad * 3 + end, expanded)
                self.assertIn(start + pad * 4 + end, expanded)
                self.assertLess(expanded.index(pad * 3), expanded.index(pad * 4))
                self.assertEqual(svc.embeddings.path.read_bytes(), b"\x01\x02")


if __name__ == "__main__":
    unittest.main()
