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
import uuid
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

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        off = False
        if self.glm_effort:
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            kwargs["reasoning_effort"] = "low" if off else level if level in ("low", "high") else "max"
        elif kwargs.get("reasoning_effort") in QWEN_EFFORT:
            kwargs["reasoning_effort"] = QWEN_EFFORT[kwargs["reasoning_effort"]]
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


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K Low, under 8K High, else Max."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "high" if n < 8192 else "max"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


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


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
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
    tools = [t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
             for t in req.get("tools") or []] or None
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


def anthropic_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs)."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for m in req.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": block.get("input") or {}}})
            elif kind == "tool_result":
                messages.append({"role": "tool", "content": _parts_of(block.get("content"))})
        if text or calls or reasoning:
            out = {"role": m["role"], "content": "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in req.get("tools") or []] or None
    if isinstance(req.get("tool_choice"), dict) and req["tool_choice"].get("type") == "none":
        tools = None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    tc = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    if tools and tc.get("type") == "any":            # a call is required: the answer starts with it
        kwargs["_force_tool"] = ""
    elif tools and tc.get("type") == "tool" and tc.get("name"):
        kwargs["_force_tool"] = tc["name"]
    return _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


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


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
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
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
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
                if self.lead:                                   # newlines right after </think> or a call
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
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            else:
                i = self.buf.find(CALL_END)
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(CALL_END):]
                call = parse_tool_call(body, self.schemas.get(call_name(body)))
                if self.scall is not None:
                    call.id = self.scall.id
                out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True

    def finish(self) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content)."""
        out = []
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            out += self._close_scan()
            out.append(Event("tool_call", call=self.scall))
            self.buf = ""
            self._reset_scan()
            return out
        if self.buf:
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else CALL_START + self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out
