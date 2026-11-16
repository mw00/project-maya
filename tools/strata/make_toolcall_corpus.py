#!/usr/bin/env python3
"""tools/strata/make_toolcall_corpus.py - writes the B12.7 tool-call corpus data/glm-dsa/toolcalls/*.json.

50 cases in five groups (BUILD-PLAN B12.7): 14 single calls, 10 parallel calls, 10 nested JSON arguments (objects,
arrays of objects, enums, numbers, booleans), 8 Chinese arguments, 8 replies to a tool error.  Each file is one OpenAI
Chat Completions request plus what a correct answer looks like:

    {"id": "single_01", "group": "single", "api": "openai" | "anthropic",
     "request": {"messages": [...], "tools": [...], "tool_choice": ...},
     "expect": {"calls": [{"name": "get_weather", "args": {"city": "Lisbon"}}],   # names (and arg subsets) expected
                "min_calls": 1, "max_calls": 1, "no_call": false, "answer_any": ["..."]}}

tools/strata/soak_run.py sends them (`toolcalls` mode and inside the soak) and checks two things.  "parse" is the
M5 gate (100%): every call has a declared name and JSON arguments that satisfy the tool's schema, and the call count
fits min/max.  "match" is reported only: the expected names and argument values.  An error-reply case must not
repeat the failed call verbatim.  The anthropic cases go to /v1/messages with the same tools in Anthropic form.

The files are generated, never edited by hand: `--check` verifies that the committed files equal the generator's
output (tools/strata/test_soak_run.py runs it).

    python3 tools/strata/make_toolcall_corpus.py --out data/glm-dsa/toolcalls
    python3 tools/strata/make_toolcall_corpus.py --check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "glm-dsa" / "toolcalls"


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer"}
B = {"type": "boolean"}

WEATHER = fn("get_weather", "Current weather for a city.",
             {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"])
CURRENCY = fn("convert_currency", "Convert an amount of money between two ISO 4217 currencies.",
              {"amount": N, "from": S, "to": S}, ["amount", "from", "to"])
SEARCH = fn("search_web", "Search the web and return the top results.",
            {"query": S, "max_results": I}, ["query"])
READ_FILE = fn("read_file", "Read a text file from the user's project.", {"path": S}, ["path"])
CALENDAR = fn("create_event", "Create a calendar event.",
              {"title": S, "start": {"type": "string", "description": "ISO 8601 date-time"},
               "duration_minutes": I, "attendees": {"type": "array", "items": S}}, ["title", "start"])
EMAIL = fn("send_email", "Send an e-mail.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"])
STOCK = fn("get_stock_price", "Latest price of a stock ticker.", {"ticker": S}, ["ticker"])
TIMEZONE = fn("get_time", "Current local time in an IANA time zone.", {"timezone": S}, ["timezone"])
TRANSLATE = fn("translate", "Translate text into a target language.", {"text": S, "target_language": S},
               ["text", "target_language"])
RUN_SQL = fn("run_sql", "Run a read-only SQL query on the analytics database.", {"query": S}, ["query"])
ORDER = fn("create_order", "Place an order in the shop.", {
    "customer": {"type": "object", "properties": {"name": S, "email": S,
                                                  "address": {"type": "object", "properties": {
                                                      "street": S, "city": S, "postcode": S, "country": S},
                                                      "required": ["street", "city", "country"]}},
                 "required": ["name", "address"]},
    "items": {"type": "array", "items": {"type": "object", "properties": {"sku": S, "quantity": I},
                                         "required": ["sku", "quantity"]}},
    "express": B}, ["customer", "items"])
FILTER = fn("query_products", "Search the product catalogue with structured filters.", {
    "filters": {"type": "object", "properties": {
        "category": S, "price": {"type": "object", "properties": {"min": N, "max": N}},
        "tags": {"type": "array", "items": S}, "in_stock": B}},
    "sort": {"type": "object", "properties": {"field": {"type": "string", "enum": ["price", "rating", "name"]},
                                              "order": {"type": "string", "enum": ["asc", "desc"]}}},
    "limit": I}, ["filters"])
CHART = fn("plot_chart", "Draw a chart from data series.", {
    "kind": {"type": "string", "enum": ["line", "bar", "pie"]}, "title": S,
    "series": {"type": "array", "items": {"type": "object", "properties": {
        "label": S, "values": {"type": "array", "items": N}}, "required": ["label", "values"]}}},
    ["kind", "series"])
CONFIG = fn("update_settings", "Update application settings; only the given keys change.", {
    "settings": {"type": "object", "properties": {
        "theme": {"type": "string", "enum": ["light", "dark", "system"]}, "font_size": I,
        "notifications": {"type": "object", "properties": {"email": B, "push": B, "quiet_hours": {
            "type": "object", "properties": {"start": S, "end": S}}}}}}}, ["settings"])
ROUTE = fn("plan_route", "Plan a route through waypoints.", {
    "waypoints": {"type": "array", "items": {"type": "object", "properties": {"lat": N, "lon": N, "name": S},
                                             "required": ["lat", "lon"]}},
    "mode": {"type": "string", "enum": ["driving", "walking", "cycling"]}, "avoid": {"type": "array", "items": S}},
    ["waypoints"])

ALL_SIMPLE = [WEATHER, CURRENCY, SEARCH, READ_FILE, CALENDAR, EMAIL, STOCK, TIMEZONE, TRANSLATE, RUN_SQL]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def case(cid: str, group: str, messages: list, tools: list, calls: list, *, min_calls: int | None = None,
         max_calls: int | None = None, no_call: bool = False, api: str = "openai", answer_any: list | None = None,
         tool_choice=None, must_not_repeat: dict | None = None) -> dict:
    req: dict = {"messages": messages, "tools": tools}
    if tool_choice is not None:
        req["tool_choice"] = tool_choice
    exp: dict = {"calls": calls, "min_calls": len(calls) if min_calls is None else min_calls,
                 "max_calls": max(len(calls), 1) + 2 if max_calls is None else max_calls, "no_call": no_call}
    if answer_any:
        exp["answer_any"] = answer_any
    if must_not_repeat:
        exp["must_not_repeat"] = must_not_repeat
    return {"id": cid, "group": group, "api": api, "request": req, "expect": exp}


def single() -> list[dict]:
    rows = [
        ("What's the weather like in Lisbon right now?", [WEATHER], "get_weather", {"city": "Lisbon"}),
        ("How much is 120 euros in Japanese yen?", [CURRENCY, WEATHER], "convert_currency",
         {"amount": 120, "from": "EUR", "to": "JPY"}),
        ("Search the web for the release date of Python 3.14.", [SEARCH, READ_FILE], "search_web", {}),
        ("Open src/main.rs and tell me what it does.", [READ_FILE, SEARCH], "read_file", {"path": "src/main.rs"}),
        ("Book a meeting called 'Design review' on 2026-11-03 at 14:00 for 45 minutes.", [CALENDAR, EMAIL],
         "create_event", {"title": "Design review", "duration_minutes": 45}),
        ("Email bob@example.com with subject 'Lunch' and say I am running ten minutes late.", [EMAIL, CALENDAR],
         "send_email", {"to": "bob@example.com", "subject": "Lunch"}),
        ("What is NVDA trading at?", [STOCK, SEARCH], "get_stock_price", {"ticker": "NVDA"}),
        ("What time is it in Tokyo?", [TIMEZONE, WEATHER], "get_time", {}),
        ("Translate 'the invoice is overdue' into German.", [TRANSLATE, SEARCH], "translate",
         {"target_language": None}),
        ("How many orders did we get in September 2026? Use the analytics database.", [RUN_SQL, SEARCH], "run_sql", {}),
        ("Weather in Reykjavik in fahrenheit, please.", ALL_SIMPLE, "get_weather", {"city": "Reykjavik",
                                                                                   "unit": "fahrenheit"}),
        ("Find three recent articles about solid-state batteries.", ALL_SIMPLE, "search_web", {}),
    ]
    out = []
    for i, (text, tools, name, args) in enumerate(rows, 1):
        out.append(case(f"single_{i:02d}", "single", [user(text)], tools,
                        [{"name": name, "args": {k: v for k, v in args.items() if v is not None}}]))
    # forced choice and an Anthropic-dialect single call
    out.append(case("single_13", "single", [user("Lisbon.")], [WEATHER, TIMEZONE],
                    [{"name": "get_weather", "args": {"city": "Lisbon"}}],
                    tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    out.append(case("single_14", "single", [user("What's the weather in Nairobi?")], [WEATHER, CURRENCY],
                    [{"name": "get_weather", "args": {"city": "Nairobi"}}], api="anthropic"))
    return out


def parallel() -> list[dict]:
    rows = [
        ("What's the weather in Paris and in Rome?", [WEATHER],
         [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Rome"})]),
        ("Convert 100 USD to EUR and 100 USD to GBP.", [CURRENCY],
         [("convert_currency", {"amount": 100, "from": "USD", "to": "EUR"}),
          ("convert_currency", {"amount": 100, "from": "USD", "to": "GBP"})]),
        ("Get the prices of AAPL, MSFT and GOOG.", [STOCK],
         [("get_stock_price", {"ticker": "AAPL"}), ("get_stock_price", {"ticker": "MSFT"}),
          ("get_stock_price", {"ticker": "GOOG"})]),
        ("What time is it in New York and what's the weather there?", [TIMEZONE, WEATHER],
         [("get_time", {}), ("get_weather", {})]),
        ("Read both README.md and CHANGELOG.md.", [READ_FILE],
         [("read_file", {"path": "README.md"}), ("read_file", {"path": "CHANGELOG.md"})]),
        ("Translate 'good morning' into French and into Spanish.", [TRANSLATE],
         [("translate", {"text": "good morning"}), ("translate", {"text": "good morning"})]),
        ("Weather in Oslo, Helsinki, Stockholm and Copenhagen?", [WEATHER],
         [("get_weather", {"city": c}) for c in ("Oslo", "Helsinki", "Stockholm", "Copenhagen")]),
        ("Search for 'llama.cpp MTP' and also look up TSLA's price.", ALL_SIMPLE,
         [("search_web", {}), ("get_stock_price", {"ticker": "TSLA"})]),
        ("Email alice@example.com and carol@example.com the subject 'Standup moved' saying it is now at 10:30.",
         [EMAIL], [("send_email", {"to": "alice@example.com"}), ("send_email", {"to": "carol@example.com"})]),
    ]
    out = []
    for i, (text, tools, calls) in enumerate(rows, 1):
        out.append(case(f"parallel_{i:02d}", "parallel", [user(text)], tools,
                        [{"name": n, "args": a} for n, a in calls], min_calls=2))
    out.append(case("parallel_10", "parallel", [user("Weather in Madrid and in Porto?")], [WEATHER],
                    [{"name": "get_weather", "args": {"city": "Madrid"}},
                     {"name": "get_weather", "args": {"city": "Porto"}}], min_calls=2, api="anthropic"))
    return out


def nested() -> list[dict]:
    rows = [
        ("Order 2 units of SKU A-100 and 1 of SKU B-7 for Jane Doe, jane@example.com, 12 Rose Street, Leeds, LS1 4AB, "
         "United Kingdom. Express shipping.", [ORDER], "create_order", {"express": True}),
        ("Ship one SKU Z-9 to Pedro Alves at Rua Augusta 100, Lisbon, Portugal.", [ORDER], "create_order", {}),
        ("Show me laptops between 800 and 1500 dollars that are in stock, tagged 'ultrabook' or 'oled', sorted by rating "
         "descending, top 5.", [FILTER], "query_products", {"limit": 5}),
        ("Find kitchen products under 30 dollars, cheapest first.", [FILTER], "query_products", {}),
        ("Plot a bar chart titled 'Revenue' with two series: 2025 = 10, 12, 15 and 2026 = 11, 14, 19.", [CHART],
         "plot_chart", {"kind": "bar", "title": "Revenue"}),
        ("Make a line chart of temperature: label 'Berlin' values 3.5, 4.1, 7.8, 12.2.", [CHART], "plot_chart",
         {"kind": "line"}),
        ("Switch to dark theme, font size 16, turn off push notifications and set quiet hours from 22:00 to 07:00.",
         [CONFIG], "update_settings", {}),
        ("Only enable email notifications; leave everything else alone.", [CONFIG], "update_settings", {}),
        ("Plan a cycling route from (38.7223, -9.1393) via (38.6979, -9.2065) named 'Belem' avoiding highways.",
         [ROUTE], "plan_route", {"mode": "cycling"}),
        ("Walk me through three points: Tower (51.5081, -0.0759), Bridge (51.5055, -0.0754), Market (51.5054, -0.0910).",
         [ROUTE, WEATHER], "plan_route", {}),
    ]
    return [case(f"nested_{i:02d}", "nested", [user(t)], tools, [{"name": n, "args": a}])
            for i, (t, tools, n, a) in enumerate(rows, 1)]


def chinese() -> list[dict]:
    rows = [
        ("北京现在天气怎么样？", [WEATHER], "get_weather", {}),
        ("帮我把 500 人民币换算成美元。", [CURRENCY], "convert_currency", {"amount": 500}),
        ("搜索一下“量子计算最新进展”。", [SEARCH], "search_web", {}),
        ("给 zhang.wei@example.com 发一封邮件，主题是“周报”，内容写“本周进度正常，详见附件”。", [EMAIL],
         "send_email", {"to": "zhang.wei@example.com"}),
        ("在 2026-12-01 上午九点创建一个叫“年度计划讨论”的会议，参加人有李雷和韩梅梅。", [CALENDAR],
         "create_event", {}),
        ("把“我们明天上午十点开会”翻译成英文。", [TRANSLATE], "translate", {}),
        ("给王芳下单：两件 SKU C-3，地址是上海市黄浦区南京东路 100 号，中国。", [ORDER], "create_order", {}),
        ("上海和广州的天气分别怎么样？", [WEATHER], "get_weather", {}),
    ]
    out = []
    for i, (t, tools, n, a) in enumerate(rows, 1):
        c = case(f"chinese_{i:02d}", "chinese", [user(t)], tools, [{"name": n, "args": a}],
                 min_calls=2 if i == 8 else 1)
        c["expect"]["cjk_args"] = True     # at least one string argument keeps Chinese characters (not escaped away)
        if i == 2:                         # an amount and two currency codes: no CJK argument
            c["expect"]["cjk_args"] = False
        out.append(c)
    return out


def error_replies() -> list[dict]:
    rows = [
        (WEATHER, {"city": "Lisbn"}, "Weather in Lisbn?", '{"error": "unknown city: Lisbn"}'),
        (CURRENCY, {"amount": 50, "from": "USD", "to": "XYZ"}, "Convert 50 USD to XYZ.",
         '{"error": "unsupported currency: XYZ"}'),
        (READ_FILE, {"path": "secret/keys.txt"}, "Read secret/keys.txt.", '{"error": "permission denied"}'),
        (STOCK, {"ticker": "APPL"}, "What's APPL at?", '{"error": "unknown ticker APPL; did you mean AAPL?"}'),
        (RUN_SQL, {"query": "SELEC count(*) FROM orders"}, "Count the orders.",
         '{"error": "syntax error at or near \\"SELEC\\""}'),
        (SEARCH, {"query": "weather"}, "Search for the weather.", '{"error": "rate limited, retry after 2 seconds"}'),
        (EMAIL, {"to": "bob@", "subject": "Hi", "body": "Hello"}, "Email bob saying hello.",
         '{"error": "invalid address: bob@"}'),
        (TIMEZONE, {"timezone": "Mars/Olympus"}, "Time on Mars/Olympus?", '{"error": "unknown time zone"}'),
    ]
    out = []
    for i, (tool, args, text, err) in enumerate(rows, 1):
        name = tool["function"]["name"]
        msgs = [user(text),
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": f"call_{i}", "content": err}]
        out.append(case(f"error_{i:02d}", "error_reply", msgs, [tool], [], min_calls=0, max_calls=2,
                        must_not_repeat={"name": name, "args": args}))
    return out


def corpus() -> list[dict]:
    out = single() + parallel() + nested() + chinese() + error_replies()
    assert len(out) == 50, len(out)
    assert len({c["id"] for c in out}) == 50
    return out


def render(c: dict) -> str:
    return json.dumps(c, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="verify the committed files instead of writing")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    cases = corpus()
    if a.check:
        bad = []
        names = {f"{c['id']}.json" for c in cases}
        for c in cases:
            p = out / f"{c['id']}.json"
            if not p.exists() or p.read_text(encoding="utf-8") != render(c):
                bad.append(p.name)
        extra = sorted(p.name for p in out.glob("*.json") if p.name not in names)
        if bad or extra:
            print(f"make_toolcall_corpus: differs: {bad}; not generated: {extra}", file=sys.stderr)
            return 1
        print(f"make_toolcall_corpus: {len(cases)} cases match")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for c in cases:
        (out / f"{c['id']}.json").write_text(render(c), encoding="utf-8")
    print(f"make_toolcall_corpus: wrote {len(cases)} cases to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
