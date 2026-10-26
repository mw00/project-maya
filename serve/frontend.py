"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    either template's XML form - GLM's (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`)
    or Qwen's (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`) - never
    emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import copy
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------------------------------------ template
class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        # GLM-5.x: the template knows reasoning_effort "low" and "high" only - anything else, an absent one included,
        # is "Max", its most verbose - and it always opens <think> (no enable_thinking).  The server's levels are
        # GLM's own (low / high / max, or thinking off: the think block closed the way the template writes a turn
        # without reasoning).  Another model's template gets them in its own words (Qwen: low / medium / xhigh).
        self.glm_effort = "reasoning_effort in ['low', 'high']" in self.source
        # DeepSeek-V4.1 (tools/ds4/chat_template.jinja): the levels by their own names (its budgets 50 / 75 / 100), tool
        # calls in DSML (OutputParser(dsml=True) reads them)
        self.deepseek = "｜DSML｜" in self.source

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        if self.deepseek:
            # Pinned, reviewed upstream encoder: keep its roles, tool ordering,
            # namespace semantics, tasks and thinking policy in one implementation.
            from serve.ds41_encoding import encode_messages
            history = copy.deepcopy(messages)
            schemas = [t if "function" in t else {"type": "function", "function": t}
                       for t in tools or []]
            if schemas:
                if not history or history[0].get("role") != "system":
                    history.insert(0, {"role": "system", "content": ""})
                history[0]["tools"] = schemas
            options = deepseek_effort_kwargs(kwargs.get("reasoning_effort"))
            thinking = kwargs.get("enable_thinking") is not False and options.get("enable_thinking") is not False
            prompt = encode_messages(history, thinking_mode="thinking" if thinking else "chat",
                                     reasoning_effort=options.get("reasoning_effort"),
                                     drop_thinking=not deepseek_keep_thinking())
            # The reference encoder always opens a generation turn after a user,
            # tool result or late system message. Expose Jinja's opt-out without
            # changing the pinned encoder or removing explicit task tokens.
            if not add_generation_prompt and history:
                last = history[-1]
                opens_turn = last.get("role") in ("user", "tool") or (
                    last.get("role") == "system" and len(history) > 1)
                header = "<｜Assistant｜>" + ("<think>" if thinking else "</think>")
                if opens_turn and last.get("task") is None and prompt.endswith(header):
                    prompt = prompt[:-len(header)]
            return prompt
        off = False
        if self.glm_effort:
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            kwargs["reasoning_effort"] = "low" if off else level if level in ("low", "high") else "max"
        elif kwargs.get("reasoning_effort") in QWEN_EFFORT and not self.deepseek:
            kwargs["reasoning_effort"] = QWEN_EFFORT[kwargs["reasoning_effort"]]
        if add_generation_prompt and not self.glm_effort:
            # The parser treats a generated turn as reasoning unless explicitly disabled. Pass
            # that same default to templates which use enable_thinking; templates which ignore it
            # remain unchanged.
            kwargs["enable_thinking"] = kwargs.get("enable_thinking") is not False
        out = self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                   **kwargs)
        if off and add_generation_prompt and out.endswith("<think>"):
            out += "</think>"
        return out


# ------------------------------------------------------------------------------------------------ requests
def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   ("text", "input_text", None))


IMAGE_PARTS = ("image_url", "input_image", "image")
VIDEO_PARTS = ("video_url", "input_video", "video")

# Thinking levels: GLM-5.x's own - Off, Low, High and Max (its template's default, the most thorough).  Clients spell
# them many ways and everything maps onto those four: OpenAI's "medium" (GLM has none) is High, "xhigh" is Max.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max", "maximum": "max"}
QWEN_EFFORT = {"low": "low", "high": "medium", "max": "xhigh"}   # the same levels in a Qwen3.x template's words


def effort_level(value) -> str | None:
    """A client's spelling -> the level's name (none / low / high / max); None when none is given or it is unknown."""
    if value is None or value == "":
        return None
    if value is False:
        return "none"
    key = str(value).strip().lower()
    return (EFFORT[key] or "none") if key in EFFORT else None


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, high or max")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def deepseek_effort_kwargs(value) -> dict:
    """DeepSeek's 1..100 budget, retaining the existing named effort aliases."""
    if type(value) is int:
        if not 1 <= value <= 100:
            raise ValueError("DeepSeek reasoning_effort must be in [1, 100]")
        return {"reasoning_effort": value}
    if value is not None and value is not False and not isinstance(value, str):
        raise ValueError("DeepSeek reasoning_effort must be an integer in [1, 100] or a named effort")
    return effort_kwargs(value)


def deepseek_keep_thinking() -> bool:
    """MAYA_DS4_KEEP_THINKING=1: an earlier turn's reasoning stays in the prompt for the turns after it, where
    DeepSeek's own serving drops it at the next user turn - so a tool-less chat's next turn starts with the prompt the
    model already holds (with MAYA_DS4_SLOT_KEEP, with its state), instead of one that differs at every assistant turn."""
    return os.environ.get("STRATA_DS4_KEEP_THINKING", "0") not in ("", "0")


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K Low, under 8K High, else Max."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "high" if n < 8192 else "max"}


def _has_media(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS + VIDEO_PARTS
                                             for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _video_source(part: dict) -> str:
    """A video part's source as one string, the same forms as an image's: OpenAI
    {"type": "video_url", "video_url": {"url": ...}} (or "video_url": "..."), Responses-style
    {"type": "input_video", "video_url": ...}; Anthropic: {"type": "video", "source": {...}}."""
    if part.get("type") == "video":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'video/mp4')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("video_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no media (unchanged behaviour), otherwise the
    template's list form - text items and image / video items, in order - whose media items carry their source."""
    if not _has_media(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in VIDEO_PARTS:
            items.append({"type": "video", "source": _video_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


def media_of(messages: list[dict]) -> list[tuple[str, str]]:
    """The (kind, source) of the conversation's media, "image" or "video", in prompt order - the order the
    template renders their markers in, so the order their embeddings are read in."""
    return [(item["type"], item["source"]) for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") in ("image", "video")]


def deepseek_images(messages: list[dict]) -> list[str]:
    """Image sources in the reference encoder's final tool-result/prompt order."""
    from serve.ds41_encoding import (_extract_image, _is_image_block, merge_tool_messages,
                                     sort_tool_results_by_call_order)
    ordered = sort_tool_results_by_call_order(merge_tool_messages(messages))
    images = []

    def collect(blocks):
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            if _is_image_block(block):
                images.append(_extract_image(block))
            elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                collect(block["content"])

    for message in ordered:
        # Collect only images: textual special literals are protected later by
        # render_ids, and must not be rejected as forged image placeholders here.
        blocks = message.get("content_blocks") if "content_blocks" in message else message.get("content")
        if isinstance(blocks, list):
            collect(blocks)
    sources = []
    for item in images:
        source = item.get("source")
        if isinstance(source, str):
            sources.append(source)
        elif isinstance(source, dict):
            sources.append(_image_source({"type": "image", "source": source}))
        else:
            sources.append(item.get("url") or item.get("data") or "")
    return sources


# A special token's text written INSIDE a message is that text, not the token.  The rendered prompt is encoded with the
# special literals parsed (and GLM's type-4 ones - <think>, </think>, <tool_call>, <tool_response>, <arg_key>, ... - are
# matched even without), so a user quoting "</think>", a file an agent reads that holds "<|user|>", or a tool result
# that prints "</tool_response>" handed the model a real end of thinking, a new turn or the end of the result.  Before
# the template is rendered every such literal in the messages and tools is swapped for a private-use character; the
# server turns them back into their text after rendering and encodes those spans as ordinary text
# (Tokenizer.encode(..., plain=spans)).  Only the literals the template itself writes stay special tokens.
LITERAL_MARK0 = 0xF0E00
# typed on purpose: GLM's "/nothink" in a user message turns thinking off for that turn, as the model was trained
KEPT_LITERALS = {"/nothink"}
# a client replaying the model's own raw call text in an assistant message sends the tokens the model wrote
CALL_TAGS = {"<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}


def literal_tags(specials) -> dict[str, str]:
    """{literal: its mark} for the tokenizer's special literals (both kinds), longest first as the tokenizer matches
    them - a literal inside a longer one is not marked before it.  {} without any (a tokenizer that has none)."""
    names = sorted({s for s in specials or () if s and s not in KEPT_LITERALS}, key=lambda s: (-len(s), s))
    return {s: chr(LITERAL_MARK0 + k) for k, s in enumerate(names)}


def _mark(text: str, tags: dict[str, str]) -> str:
    for tag, mark in tags.items():
        if tag in text:
            text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str]):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str]) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


def mark_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str]):
    """-> (messages, tools, changed): every special literal in the messages' and tools' text swapped for its mark.
    Nothing to mark: the same objects back and False - such a prompt renders and encodes exactly as before.
    An assistant message keeps the call tags it wrote (its raw call text, replayed) and, when its content opens with a
    whole <think>...</think> block (clients that send the reasoning inline), that one block's two tags: the template
    takes the reasoning apart at them.  Its structured tool_calls are data: everything in them is marked."""
    if not tags or (not _has_tag(messages, tags) and not _has_tag(tools, tags)):
        return messages, tools, False
    own = {t: v for t, v in tags.items() if t not in CALL_TAGS}
    out = []
    for m in messages:
        m = dict(m)
        assistant = m.get("role") == "assistant"
        content = m.get("content")
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, own if assistant and k != "tool_calls" else tags)
        if assistant and isinstance(content, str) and content.lstrip().startswith("<think>") and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = content[:i] + _mark(content[i:j], own) + "</think>" + _mark(content[j + len("</think>"):], own)
        out.append(m)
    return out, _mark_deep(tools, tags), True


def unmark_literals(prompt: str, tags: dict[str, str]) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with the marks turned back into their literals' text, and the (start, end) spans of those
    literals in it - the spans the tokenizer encodes as ordinary text."""
    marks = {v: k for k, v in tags.items()}
    if not marks:
        return prompt, []
    out, spans, pos, n = [], [], 0, 0
    for m in re.finditer("[" + "".join(map(re.escape, marks)) + "]", prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    if not spans:
        return prompt, []
    out.append(prompt[pos:])
    return "".join(out), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def stable_tool_order(tools: list[dict] | None) -> list[dict] | None:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list at the start of the prompt: one swap moved the shared prefix back to the system
    prompt and the whole conversation was read again.  A tool set's first request is left as sent (its prompt is what
    the client asked for); later ones with the same names in another order are put back in that first order.
    Duplicate or missing names, or fewer than two tools: as sent."""
    if not tools or len(tools) < 2:
        return tools
    names = [t.get("name") if isinstance(t, dict) else None for t in tools]
    if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[t["name"]])


def openai_to_messages(req: dict, deepseek: bool = False) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    if deepseek:
        from serve.ds41_encoding import _tool_name_for_encoding, tools_from_openai_format
        messages = copy.deepcopy(req.get("messages", []))
        for message in messages:
            if message.get("role") == "developer":
                message["role"] = "system"
            if _has_media(message.get("content")):
                message["content"] = _parts_of(message["content"])
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if "arguments" not in function:
                    continue
                original = arguments = function["arguments"]
                # Decode before literal marking: JSON Unicode escapes can hide
                # special tokens until the reference encoder loads arguments.
                # Match its tolerance for twice-encoded JSON and its fallback.
                for _ in range(2):
                    if not isinstance(arguments, str):
                        break
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        break
                function["arguments"] = arguments if isinstance(arguments, dict) else {"arguments": original}
        # Unlike the legacy adapter, preserve tool IDs, namespaces, content_blocks,
        # mid-system roles, latest_reminder and task fields for the reference encoder.
        declared = copy.deepcopy(req.get("tools") or [])
        tools = tools_from_openai_format(declared) if declared else None
        if req.get("tool_choice") == "none":
            tools = None
            for message in messages:
                message.pop("tools", None)
        kwargs = {}
        reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
        effort = req.get("reasoning_effort", reasoning.get("effort"))
        kwargs.update(deepseek_effort_kwargs(effort))
        for key, value in (req.get("chat_template_kwargs") or {}).items():
            if key == "enable_thinking" and not value:
                kwargs = {"enable_thinking": False}
            elif key == "reasoning_effort" and "enable_thinking" not in kwargs:
                kwargs.update(deepseek_effort_kwargs(value))
        choice = req.get("tool_choice")
        if tools and choice == "required":
            kwargs["_force_tool"] = ""
        elif tools and isinstance(choice, dict) and choice.get("type") == "function":
            function = dict(choice.get("function") or {})
            if choice.get("namespace") is not None:
                function["namespace"] = choice["namespace"]
            kwargs["_force_tool"] = _tool_name_for_encoding(function) if function.get("name") else ""
        return messages, tools, kwargs
    messages = []
    for m in req.get("messages", []):
        role = m.get("role")
        if role == "developer":
            role = "system"
        out = {"role": role, "content": _parts_of(m.get("content")) if role in ("user", "tool", "assistant") else _text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", c)
                args = fn.get("arguments")
                if isinstance(args, str):               # the template requires a mapping, not a JSON string
                    args = json.loads(args) if args.strip() else {}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            out["tool_calls"] = calls
        if role == "assistant" and not out["content"] and not out.get("tool_calls") and not out.get("reasoning_content"):
            continue   # an empty turn is not rendered: a history of them teaches the model to answer with nothing
        messages.append(out)
    tools = stable_tool_order([t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
                               for t in req.get("tools") or []] or None)
    tc = req.get("tool_choice")
    if tc == "none":   # the tools stay declared to the client, but none is offered to the model
        tools = None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    # "required" / a named function: the answer starts with the call (Service.prepare; from Strata #790)
    if tools and tc == "required":
        kwargs["_force_tool"] = ""
    elif tools and isinstance(tc, dict) and tc.get("type") == "function" and (tc.get("function") or {}).get("name"):
        kwargs["_force_tool"] = tc["function"]["name"]
    return _late_system_to_user(messages), tools, kwargs


def anthropic_to_messages(req: dict, deepseek: bool = False) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs)."""
    messages = []
    system = req.get("system")
    if system:
        text = "\n\n".join(_text_of([part]) for part in system) if deepseek and isinstance(system, list) else _text_of(system)
        messages.append({"role": "system", "content": text})
    for m in req.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_media(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        if deepseek and m.get("role") == "user":
            # The reference encoder sorts only tool-result positions by the IDs
            # of the preceding assistant calls, retaining surrounding text.
            messages.append({"role": "user", "content_blocks": copy.deepcopy(content or [])})
            continue
        text, reasoning, calls = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                call = {"function": {"name": block.get("name"), "arguments": block.get("input") or {}}}
                if deepseek:
                    call.update(id=block.get("id", ""), type="function")
                calls.append(call)
            elif kind == "tool_result":
                messages.append({"role": "tool", "content": _parts_of(block.get("content"))})
        if text or calls or reasoning:
            out = {"role": m["role"], "content": ("\n\n" if deepseek else "").join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""),
              "parameters": t.get("input_schema", {})} for t in req.get("tools") or []] or None
    if not deepseek:
        tools = stable_tool_order(tools)
    if isinstance(req.get("tool_choice"), dict) and req["tool_choice"].get("type") == "none":
        tools = None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort or (deepseek and effort is not None and effort != ""):
        kwargs.update(deepseek_effort_kwargs(effort) if deepseek else effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    tc = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    if tools and tc.get("type") == "any":            # a call is required: the answer starts with it
        kwargs["_force_tool"] = ""
    elif tools and tc.get("type") == "tool" and tc.get("name"):
        kwargs["_force_tool"] = tc["name"]
    return messages if deepseek else _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])
    namespace: str | None = None

    def to_openai(self, arguments: str | None = None) -> dict:
        """Keep qualified names for internal routing; expose the reference API shape."""
        namespace = self.namespace
        call = {"id": self.id, "type": "function", "function": {
            "name": self.name.split("::", 1)[-1] if namespace is not None else self.name,
            "arguments": json.dumps(self.arguments, ensure_ascii=False) if arguments is None else arguments}}
        if namespace is not None:
            call["namespace"] = namespace
        return call


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None
    complete: bool = True          # false when the model never closed the call's syntax


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"
FUNC_START = "<function="                       # Qwen's form: <function=NAME><parameter=P>V</parameter>...</function>
ARG_KEY, ARG_KEY_END = "<arg_key>", "</arg_key>"   # GLM's form: NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
ARG_VAL, ARG_VAL_END = "<arg_value>", "</arg_value>"


def _arg_value(value: str, declared) -> object:
    """A parameter's text as the client gets it: as written when the schema says string, else JSON-decoded when it
    parses (numbers, booleans, objects, arrays), else as written."""
    if declared == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def call_name(body: str) -> str:
    """The tool name of a call body (the text between <tool_call> and </tool_call>), in either form."""
    body = body.strip()
    if body.startswith(FUNC_START):
        return body[len(FUNC_START):].split(">", 1)[0]
    return body.split("<", 1)[0].strip()


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """A call body -> ToolCall, in GLM's form (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`, what GLM-5.x
    writes: issue #5) or Qwen's (`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>`).  Values are
    JSON-decoded when the tool's schema says the parameter is not a string (or, without a schema, when they parse as
    JSON objects/arrays/numbers/booleans)."""
    body = body.strip()
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    if not body.startswith(FUNC_START):
        name = call_name(body)
        rest = body[len(body.split("<", 1)[0]):]
        if not name or not (rest == "" or rest.lstrip().startswith(ARG_KEY)):
            raise ValueError("malformed tool call: " + body[:80])
        args = {}
        while ARG_KEY in rest:
            rest = rest[rest.index(ARG_KEY) + len(ARG_KEY):]
            end = rest.find(ARG_KEY_END)
            if end < 0:
                raise ValueError("malformed tool call: " + body[:80])
            key, rest = rest[:end].strip(), rest[end + len(ARG_KEY_END):]
            v0 = rest.find(ARG_VAL)
            if v0 < 0 or rest[:v0].strip():
                raise ValueError("malformed tool call: " + body[:80])
            rest = rest[v0 + len(ARG_VAL):]
            end = rest.find(ARG_VAL_END)
            value = rest[:end] if end >= 0 else rest
            rest = rest[end + len(ARG_VAL_END):] if end >= 0 else ""
            args[key] = _arg_value(value, (props.get(key) or {}).get("type"))
        return ToolCall(name=name, arguments=args)
    if ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len(FUNC_START):body.index(">")]
    rest = body[body.index(">") + 1:]
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        pname = rest[:rest.index(">")]
        rest = rest[rest.index(">") + 1:]
        end = rest.find("</parameter>")
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len("</parameter>"):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        declared = (props.get(pname) or {}).get("type")
        if declared == "string":
            args[pname] = value
        else:
            try:
                args[pname] = json.loads(value)
            except ValueError:
                args[pname] = value
    return ToolCall(name=name, arguments=args)


DSML = "｜DSML｜"
_DSML_ATTR = re.compile(r'(\w+)="([^"]*)"')


class DsmlRewriter:
    """DeepSeek-V4.1's tool calls (DSML: `<｜DSML｜ calls>` around `<｜DSML｜ invoke name="N">` blocks of
    `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`) rewritten, as they stream, into GLM's form
    (`<tool_call>N<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`) for OutputParser.  A tag split across
    deltas is held until it is whole; the whitespace between the tags of a calls block is dropped.  A value is passed
    as written; string="true" is JSON-quoted so the parser preserves the model's
    string-vs-JSON flag independently of the tool's schema."""

    def __init__(self):
        self.buf = ""
        self.in_calls = self.in_param = False
        self.parameter_text = ""
        self.parameter_string = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            i = self.buf.find("<")
            if i < 0:
                out.append(self._plain(self.buf))
                self.buf = ""
                break
            if i > 0:
                out.append(self._plain(self.buf[:i]))
                self.buf = self.buf[i:]
            opener = next((o for o in ("<" + DSML, "</" + DSML) if self.buf.startswith(o)), None)
            if opener is None:
                if any(o.startswith(self.buf) for o in ("<" + DSML, "</" + DSML)):
                    break                                   # could still become a tag: hold it
                out.append(self._plain("<"))
                self.buf = self.buf[1:]
                continue
            j = self.buf.find(">")
            if j < 0:
                break                                       # the tag is not whole yet
            tag, self.buf = self.buf[:j + 1], self.buf[j + 1:]
            out.append(self._tag(tag, opener.startswith("</"), tag[len(opener):-1]))
        return "".join(out)

    def finish(self) -> str:
        rest, self.buf = self.buf, ""
        return self._plain(rest)

    @staticmethod
    def _esc(s: str) -> str:
        """s as JSON string content - character by character, so pieces of a value escape to the whole's escape
        (issue #403) - with `<` as \\u003c (see _tag)."""
        return json.dumps(s, ensure_ascii=False)[1:-1].replace("<", "\\u003c")

    def _plain(self, s: str) -> str:
        if self.in_param:
            if self.parameter_string:    # a string value goes out as it is written (issue #403: a long file's
                return self._esc(s)      # content held to its closing tag left clients without data for minutes)
            self.parameter_text += s
            return ""
        return "" if self.in_calls and not self.in_param and not s.strip() else s

    def _tag(self, tag: str, closing: bool, body: str) -> str:
        kind = body.split(None, 1)[0] if body.strip() else ""
        if self.in_param and not (closing and kind == "parameter"):
            # Like the reference decoder, only the parameter's closing tag is
            # structural here. Other DSML tags can be literal argument data.
            return self._plain(tag)
        attrs = dict(_DSML_ATTR.findall(body))
        if kind == "calls":
            self.in_calls = not closing
            return ""
        if kind == "invoke":
            return "</tool_call>" if closing else CALL_START + attrs.get("name", "")
        if kind == "parameter":
            self.in_param = not closing
            if closing:
                # A JSON string can contain a literal legacy parser delimiter
                # (e.g. a tool writing an XML example). JSON's Unicode escape
                # keeps those values intact across this internal rewrite.
                # (a string value: its quotes around the pieces _plain sent as they came)
                value = '"' if self.parameter_string else self.parameter_text.replace("<", "\\u003c")
                self.parameter_text = ""
                return value + ARG_VAL_END
            self.parameter_string = attrs.get("string") == "true"
            self.parameter_text = ""
            return ARG_KEY + attrs.get("name", "") + ARG_KEY_END + ARG_VAL + ('"' if self.parameter_string else "")
        return tag


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 dsml: bool = False, strict_tools: bool = False, single_tool: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.dsml = DsmlRewriter() if dsml else None      # DeepSeek-V4.1's tool calls, read as GLM's (DsmlRewriter)
        self.buf = ""
        self.lead = False
        self.schemas = {} if dsml else {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        self.strict_tools, self.single_tool, self.finished = strict_tools, single_tool, False
        # the answer text that has gone out, followed to tell a call from a quoted one (Strata #1058): the open code
        # fence, the current line, and the backticks of the current paragraph
        self.fence, self.line, self.ticks = "", "", 0
        self._reset_scan()

    def _track(self, text: str) -> str:
        """Follow answer text that goes out (see fence above).  Returns the text."""
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line, self.line = self.line + part, ""
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
            else:
                self.line += part
        return text

    def _in_code(self) -> bool:
        """The answer so far leaves the next character inside a code fence or inline code."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _reset_scan(self):
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}
        self.glm = False             # the call is in GLM's form (NAME<arg_key>..), not Qwen's (<function=NAME>..)
        self.vend = "</parameter>"   # what ends the value being streamed

    def _start_call(self, name: str, out: list, args) -> None:
        self.scall = ToolCall(name=name, arguments={})
        if self.dsml is not None and "::" in name:
            self.scall.namespace = name.split("::", 1)[0]
        props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
        self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
        out.append(Event("tool_start", call=self.scall))
        args("{")

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                lead = rest.lstrip()
                if not lead or FUNC_START.startswith(lead):
                    return out          # which form it is: not known yet
                if lead.startswith(FUNC_START):
                    a = rest.find(FUNC_START)
                    b = rest.find(">", a + len(FUNC_START))
                    if b < 0:
                        return out
                    self._start_call(rest[a + len(FUNC_START):b], out, args)
                    self.sp += b + 1
                    self.ss = "between"
                else:
                    # GLM: the name runs to the first tag (or the line's end); until one has arrived it may go on
                    b = min((i for i in (rest.find("<"), rest.find("\n", len(rest) - len(lead))) if i >= 0),
                            default=-1)
                    if b < 0 or not rest[:b].strip():
                        return out      # (no name before a tag: malformed - the whole call's parse says so)
                    self._start_call(rest[:b].strip(), out, args)
                    self.sp += b
                    self.glm, self.vend, self.ss = True, ARG_VAL_END, "between"
            elif self.ss == "between" and self.glm:
                stripped = rest.lstrip()
                if not stripped.startswith(ARG_KEY):
                    return out          # the next pair still arriving (or the call's end: the caller closes it)
                k1 = stripped.find(ARG_KEY_END)
                if k1 < 0:
                    return out
                after = stripped[k1 + len(ARG_KEY_END):]
                v0 = len(after) - len(after.lstrip())
                if not after[v0:].startswith(ARG_VAL):
                    return out
                pname = stripped[len(ARG_KEY):k1].strip()
                args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                self.sfirst = False
                self.sp += (len(rest) - len(stripped)) + k1 + len(ARG_KEY_END) + v0 + len(ARG_VAL)
                if self.sdeclared.get(pname) == "string":
                    args('"')
                    self.ss, self.sval_started = "str", True     # GLM's values are written as they are
                elif self.dsml is not None:   # DeepSeek's: a string="true" value arrives JSON-quoted, piece by piece
                    self.ss, self.sval_started = "jstr", False
                else:
                    self.ss = "raw"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    self.sp += b + 1
                    if self.sdeclared.get(pname) == "string":
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    args("}")
                    self.sp += len("</function>")
                    self.ss = "done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                end = rest.find(self.vend)
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n") and not self.glm:
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(self.vend)
                    self.ss = "between"
                    continue
                safe = len(rest) - self._hold(rest, (self.vend,))
                if safe > 0 and rest[safe - 1] == "\n" and not self.glm:   # may be the newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "jstr":
                # DsmlRewriter's value: JSON string content between quotes as it is written (its closing quote comes
                # with the end tag), or a string="false" value whole.  A value that is not complete yet is a string's:
                # its pieces go out as they are (issue #403); one that arrives complete takes the raw path.
                end = rest.find(self.vend)
                if end >= 0 and self.sval_started:
                    args(rest[:end])                 # (its last piece and the closing quote)
                    self.sp += end + len(self.vend)
                    self.ss = "between"
                    continue
                if end >= 0:
                    self.ss = "raw"
                    continue
                if not rest or (not self.sval_started and rest[0] != '"'):
                    return out                       # (a string="false" value: complete at its end tag)
                args(rest)
                self.sp += len(rest)
                self.sval_started = True
                return out
            elif self.ss == "raw":
                end = rest.find(self.vend)
                if end < 0:
                    return out
                value = rest[:end]
                if not self.glm:
                    if value.startswith("\n"):
                        value = value[1:]
                    if value.endswith("\n"):
                        value = value[:-1]
                try:
                    v = json.loads(value)
                except ValueError:
                    v = value
                args(json.dumps(v, ensure_ascii=False))
                self.sp += end + len(self.vend)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """A streamed call that ended without a clean </function>: close its JSON so clients can still parse it."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str" or (self.ss == "jstr" and self.sval_started):
            tail += '"'
        elif self.ss in ("raw", "jstr"):
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def feed(self, delta: str) -> list[Event]:
        if self.finished:
            return []
        if self.dsml is not None:
            delta = self.dsml.feed(delta)
        return self._feed(delta)

    def _feed(self, delta: str) -> list[Event]:
        if self.finished:
            return []
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning":
                i = self.buf.find(THINK_END)
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out.append(Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0
            elif self.state == "content":
                if self.lead and self.dsml is None:              # legacy GLM/Qwen newline policy
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # Strata #1058: a <tool_call> inside a code fence or inline code of the answer is a quoted example -
                # text, never a call (an example `rm -rf` must not run).  The text before it decides, so streamed and
                # whole outputs agree; one at the top level, mid-sentence too, is a call as before.
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if self.dsml is not None:
                    prefix = self.buf[:i]
                    if prefix.endswith("\n\n"):
                        prefix = prefix[:-2]
                    if prefix:
                        out.append(Event("content", self._track(prefix)))
                elif i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            else:
                i = self.buf.find(CALL_END)
                complete = True
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        if self.strict_tools and self.scall is not None:
                            if self.glm:
                                complete = self.ss == "between" and not self.buf[self.sp:].strip()
                            else:
                                complete = self.ss == "done" and not self.buf[self.sp:].strip()
                        if complete or not self.strict_tools:
                            out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(CALL_END):]
                call = parse_tool_call(body, self.schemas.get(call_name(body)))
                if self.dsml is not None and "::" in call.name:
                    call.namespace = call.name.split("::", 1)[0]
                if self.scall is not None:
                    call.id = self.scall.id
                out.append(Event("tool_call", call=call, complete=complete))
                self._reset_scan()
                self.state, self.lead = "content", True
                if self.single_tool and complete:
                    self.buf, self.finished = "", True
                    return out

    def finish(self) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content)."""
        if self.finished:
            return []
        out = self._feed(self.dsml.finish()) if self.dsml is not None else []
        if self.finished:
            return out
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            if not self.strict_tools:
                out += self._close_scan()
            out.append(Event("tool_call", call=self.scall, complete=not self.strict_tools))
            self.buf = ""
            self._reset_scan()
            return out
        if self.buf:
            if self.strict_tools and self.state == "call":
                out.append(Event("tool_incomplete", complete=False))
                self.buf = ""
                return out
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else CALL_START + self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out
