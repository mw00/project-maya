"""Synthetic GLM Chat continuation contracts; NOT native generation/business evidence.

Requires MAYA_AGENT_TOKENIZER pointing at an archived real GLM pack tokenizer.
Missing assets fail explicitly, never silently skip or substitute a mock tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, openai_to_messages, parse_tool_call
from tools.strata_tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"value": {"type": "number"}}}}}]


def lcp(left: list[int], right: list[int]) -> int:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


class AgentPrefixTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = os.environ.get("MAYA_AGENT_TOKENIZER")
        if not directory:
            raise RuntimeError("MAYA_AGENT_TOKENIZER must name a real archived GLM tokenizer")
        cls.assets = Path(directory)
        required = ("vocab.json", "merges.txt", "tokenizer.json", "token_type.json", "chat_template.jinja")
        missing = [name for name in required if not (cls.assets / name).is_file()]
        if missing:
            raise RuntimeError(f"real tokenizer assets missing: {missing}")
        cls.tok = Tokenizer.from_dir(cls.assets)
        if cls.tok.pre not in ("glm4", "glm5"):
            raise RuntimeError(f"not a GLM tokenizer: {cls.tok.pre}")
        cls.template = ChatTemplate(cls.assets / "chat_template.jinja", glm_dsa=True)
        cls.metrics: list[dict] = []
        cls.hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            *(cls.assets / name for name in required), ROOT / "serve/frontend.py",
            ROOT / "tools/strata_tokenizer.py", Path(__file__))}

    @classmethod
    def tearDownClass(cls) -> None:
        print("AGENT_PREFIX_EVIDENCE=" + json.dumps({"synthetic": True, "native_ids": False,
              "source_sha256": cls.hashes, "comparisons": cls.metrics}, sort_keys=True))

    def render(self, history: list[dict]) -> str:
        req = {"messages": [{"role": "user", "content": "Look up one value."}] + history,
               "tools": TOOLS, "tool_choice": "auto", "reasoning_effort": "low"}
        messages, tools, kwargs = openai_to_messages(req, glm_dsa=True)
        return self.template.render(messages, tools, **kwargs)

    def call_history(self, raw_call: str, reasoning: str | None = "plan", content: str = "") -> list[dict]:
        call = parse_tool_call(raw_call[len("<tool_call>"):-len("</tool_call>")], TOOLS[0]["function"])
        assistant = {"role": "assistant", "content": content,
                     "tool_calls": [call.to_openai()]}
        if reasoning is not None:
            assistant["reasoning_content"] = reasoning
        return [assistant, {"role": "tool", "tool_call_id": call.id, "content": "found"}]

    def compare(self, generated_text: str, history: list[dict], terminal: str = "", *, matches: bool) -> None:
        # This constructs a SYNTHETIC token path from raw fixture text, not observed target IDs.
        prompt = self.tok.encode(self.render([]), parse_special=True)
        generated = self.tok.encode(generated_text + terminal, parse_special=True)
        candidate = prompt + generated
        following = self.tok.encode(self.render(history), parse_special=True)
        common = lcp(candidate, following)
        self.metrics.append({"test": self.id(), "candidate_tokens": len(candidate),
                             "next_tokens": len(following), "lcp": common,
                             "exact_prefix": common == len(candidate)})
        self.assertEqual(common == len(candidate), matches)

    def test_stable_single_tool_chat_is_exact_append(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
        self.compare("plan</think>" + call, self.call_history(call), "<|observation|>", matches=True)

    def test_trailing_assistant_whitespace_safely_misses(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
        self.compare("plan</think>answer  " + call,
                     self.call_history(call, content="answer  "), matches=False)

    def test_exponent_argument_normalization_safely_misses(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1e0</arg_value></tool_call>"
        self.compare("plan</think>" + call, self.call_history(call), matches=False)

    def test_missing_reasoning_safely_misses(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
        self.compare("plan</think>" + call, self.call_history(call, reasoning=None), matches=False)

    def test_committed_user_terminator_before_tool_result_safely_misses(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
        self.compare("plan</think>" + call, self.call_history(call), "<|user|>", matches=False)

    def test_committed_eos_before_tool_result_safely_misses(self) -> None:
        call = "<tool_call>lookup<arg_key>value</arg_key><arg_value>1</arg_value></tool_call>"
        self.compare("plan</think>" + call, self.call_history(call), "<|endoftext|>", matches=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
