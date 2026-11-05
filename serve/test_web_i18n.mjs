// Dependency-free checks for serve/web/i18n.js. Run: node serve/test_web_i18n.mjs
import assert from "node:assert/strict";
import {readFile, readdir} from "node:fs/promises";

const saved = new Map();
globalThis.localStorage = {
  getItem: (key) => saved.get(key) || null,
  setItem: (key, value) => saved.set(key, value),
  removeItem: (key) => saved.delete(key),
};
globalThis.matchMedia = () => ({addEventListener() {}});
globalThis.document = {documentElement: {}, hidden: false, addEventListener() {}};

const {languageChoice, setLanguage, t, tr} = await import("./web/i18n.js");
const {S} = await import("./web/state.js");
const {Toasts} = await import("./web/ui.js");
const {Answer, localizeAnswerMeta} = await import("./web/chat-view.js");
const {streamingMarkdown, patchHtml} = await import("./web/md.js");
const {api} = await import("./web/lib.js");
const {phaseText} = await import("./web/server.js");
const stateModule = await import("./web/state.js");
const {exportChat} = await import("./web/chats.js");
await import("./web/settings.js"); // New grouped Settings must retain valid cross-page imports.
const {freeAdvice} = await import("./web/monitor.js");
assert.equal(languageChoice(), "en");
assert.equal(t("new-chat", "New chat"), "New chat");

setLanguage("zh-CN");
assert.equal(languageChoice(), "zh-CN");
assert.equal(saved.get("maya.language"), "zh-CN");
assert.equal(document.documentElement.lang, "zh-CN");
assert.equal(t("new-chat", "New chat"), "新建对话");
assert.equal(t("settings.connect", "Connect your tools"), "连接你的工具");
assert.equal(t("missing-key", "English fallback"), "English fallback");

setLanguage("en");
assert.equal(languageChoice(), "en");
assert.equal(saved.has("maya.language"), false);
assert.equal(document.documentElement.lang, "en");
assert.equal(t("missing-key", "English fallback"), "English fallback");

// A non-empty toast reaches the map callback in Toasts(). This guards against the callback's toast object
// shadowing the translation helper.
S.toasts = [{id: 1, kind: "success", title: "Saved", text: "", action: null}];
assert.ok(Toasts());
S.toasts = [];

// Existing answer components opt into a normal Preact update when only the language changes, so their action labels
// update without replacing their model text or interrupting the in-place streaming path.
const message = {id: "answer-1", text: "kept", reasoning: "", tools: []};
const answer = new Answer();
answer.props = {m: message, lang: "en"};
assert.equal(answer.shouldComponentUpdate({m: message, lang: "zh-CN"}), true);
let repaint = false;
answer.update = (props, force) => { repaint = props.lang === "zh-CN" && force; };
answer.props = {m: message, lang: "zh-CN"};
answer.componentDidUpdate({m: message, lang: "en"});
assert.equal(repaint, true);

// Every static source phrase used by the localized pages has a real catalog entry, and named placeholders match.
setLanguage("zh-CN");
const webRoot = new URL("./web/", import.meta.url);
const catalogs = (await readdir(webRoot)).filter((name) => /^i18n-.+\.js$/.test(name));
const placeholders = (text) => [...new Set([...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]))].sort();
for (const name of catalogs) {
  const {default: phrases} = await import(new URL(name, webRoot));
  for (const [english, chinese] of Object.entries(phrases)) {
    assert.deepEqual(placeholders(chinese), placeholders(english), `${name}: placeholder mismatch for ${english}`);
  }
}
const missing = [];
for (const name of ["settings.js", "server.js", "monitor.js", "ui.js", "turbo.js", "md.js", "models.js", "sidebar.js", "chat-view.js", "chats.js", "state.js", "lib.js"]) {
  const source = await readFile(new URL(name, webRoot), "utf8");
  for (const match of source.matchAll(/\btr\(\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g)) {
    const literal = match[1];
    const english = literal[0] === '"' ? JSON.parse(literal)
      : JSON.parse('"' + literal.slice(1,-1).replace(/\\'/g,"'").replace(/"/g,'\\"') + '"');
    if (t(english, "__MISSING__") === "__MISSING__") missing.push(`${name}: ${english}`);
  }
}
assert.deepEqual(missing, [], "Static UI phrases missing from zh-CN catalog");
for (const label of ["Overview", "Engine", "Storage", "Its folder", "The pack Maya built", "A model file",
  "Its image encoder", "Its DFlash2 drafter", "The folder", "The pack", "The image encoder", "The DFlash2 drafter",
  "Appearance", "Global instructions", "Integrations", "Usage", "Privacy", "About", "peak this request", "peak last request"])
  assert.notEqual(t(label, "__MISSING__"), "__MISSING__", `Missing upstream UI label: ${label}`);
const gib = 2 ** 30;
const tpPair = [{index:0,free:gib,tp_room:gib,tp_group:0,slots:1000,parts:{pool:10*gib,drafter:gib}},
  {index:1,free:9*gib,tp_room:gib,tp_group:0,slots:1000,parts:{pool:10*gib}}];
const tpAdvice = freeAdvice([tpPair[1]], tpPair, {engine_kind:"ds4"}, () => "RAW_GPU_1", () => "RAW_POOL", () => 999);
assert.match(tpAdvice, /RAW_GPU_1.*8.*闲置/);
assert.match(tpAdvice, /MAYA_DS4_DRAFT_DEV/);
assert.doesNotMatch(tpAdvice, /999/); // A full TP partner cannot grow the pool on restart.
assert.equal(tr("How {model} writes new answers. Each model keeps its own.", {model:"RAW_MODEL_<>&"}),
  "RAW_MODEL_<>& 生成新回答的方式。每个模型分别保存设置。");
assert.equal(t("toString", "English fallback"), "English fallback");
setLanguage("en");
assert.equal(tr("How {model} writes new answers. Each model keeps its own.", {model:"Raw model"}),
  "How Raw model writes new answers. Each model keeps its own.");

// A long streaming answer keeps cached HTML. Its code actions must follow locale changes
// without changing the model's own text or leaving old English actions in the prefix.
const longAnswer = {text: "```html\n<p>RAW_CODE_COPY_RUNNING</p>\n```\n\n" + "Raw model prose. ".repeat(450) + "\n\nTail"};
const rawText = longAnswer.text;
assert.match(streamingMarkdown(longAnswer), /aria-label="Copy the code"/);
setLanguage("zh-CN");
const zhMarkdown = streamingMarkdown(longAnswer);
assert.match(zhMarkdown, /aria-label="复制代码"/);
assert.doesNotMatch(zhMarkdown, /aria-label="Copy the code"/);
assert.match(zhMarkdown, /RAW_CODE_COPY_RUNNING/);
assert.equal(longAnswer.text, rawText);
assert.equal(Object.keys(longAnswer).includes("_md"), false);
setLanguage("en");
assert.match(streamingMarkdown(longAnswer), /aria-label="Copy the code"/);

// Code actions with the same number of buttons still need a label update. Minimal DOM
// doubles exercise patchHtml's existing code-preserving branch without adding a DOM dependency.
let replacement = false;
let wrapPressed = null;
const newActions = {token: "new", querySelector: () => ({setAttribute: (key,value) => { if (key === "aria-pressed") wrapPressed = value; }})};
const oldActions = {children: [1,2,3], isEqualNode: () => false, replaceWith: (node) => { replacement = node === newActions; }};
const code = {innerHTML: "RAW_CODE"};
const pre = {scrollHeight: 100, scrollTop: 0, clientHeight: 100};
const label = {textContent: "code"};
const currentBlock = {nodeType:1, dataset:{lang:""}, classList:{contains:()=>true}, isEqualNode:()=>false, matches:()=>true,
  querySelector:(selector)=>({pre,code,".code__actions":oldActions,".code__lang":label}[selector])};
const nextBlock = {nodeType:1, dataset:{lang:""}, matches:()=>true,
  querySelector:(selector)=>({code,".code__actions":newActions,".code__lang":{textContent:"代码"}}[selector])};
const savedCreate = document.createElement;
document.createElement = () => ({content:{childNodes:[nextBlock]}, set innerHTML(value) {}});
patchHtml({childNodes:[currentBlock]}, "fixture");
document.createElement = savedCreate;
assert.equal(replacement, true);
assert.equal(code.innerHTML, "RAW_CODE");
assert.equal(label.textContent, "代码");
assert.equal(wrapPressed, "true");

// API auth guidance is UI-owned; arbitrary backend diagnostics remain byte-for-byte.
const savedFetch = globalThis.fetch;
setLanguage("zh-CN");
globalThis.fetch = async () => ({ok:false,status:401,json:async()=>({error:{message:"RAW_AUTH"}})});
await assert.rejects(api("fixture"), /此服务器需要 API 密钥/);
globalThis.fetch = async () => ({ok:false,status:503,json:async()=>({error:{message:"RAW_DETAIL_COPY_RUNNING"}})});
await assert.rejects(api("fixture"), (error) => error.message === "RAW_DETAIL_COPY_RUNNING");
globalThis.fetch = savedFetch;
for (const phase of ["thinking", "answering", "reading the prompt", "tool call complete"])
  assert.match(phaseText(phase), /[\u4e00-\u9fff]/);
assert.equal(phaseText("writing a tool call: RAW_TOOL_Copy"), "正在编写工具调用：RAW_TOOL_Copy");
assert.equal(phaseText("RAW_PHASE"), "RAW_PHASE");
assert.deepEqual(localizeAnswerMeta("128 tokens · 16.0 tok/s · read 2,048 new prompt tokens in 2.0 s (2,048 reused) · 3 tool calls"),
  ["128 个 Token", "16.0 tok/s", "在 2.0 秒内读取 2,048 个新提示 Token（复用 2,048 个）", "3 次工具调用"]);
assert.deepEqual(localizeAnswerMeta("Stopped"), ["已停止"]);
assert.deepEqual(localizeAnswerMeta("Stopped at the limit of 3 tool rounds (mcp.max_rounds)"), ["在 3 轮工具调用上限处停止（mcp.max_rounds）"]);
assert.deepEqual(localizeAnswerMeta("RAW_DIAGNOSTIC · Running Copy"), ["RAW_DIAGNOSTIC", "Running Copy"]);
for (const family of ["glm", "deepseek", "minimax", "qwen"]) {
  stateModule.setFamily(family);
  for (const level of stateModule.LEVELS) for (const label of [level.name, level.desc])
    assert.notEqual(t(label, "__MISSING__"), "__MISSING__", `Missing ${family} thinking label: ${label}`);
  for (const title of Object.values(stateModule.THINK_TITLES[family]))
    assert.notEqual(t(title, "__MISSING__"), "__MISSING__", `Missing ${family} thinking tooltip: ${title}`);
}
stateModule.setFamily("");
for (const label of Object.values(stateModule.PRESET_NAMES))
  assert.notEqual(t(label, "__MISSING__"), "__MISSING__", `Missing preset: ${label}`);

// A partially written answer still exports its model text when a connection error has a UI label.
let exportedBlob;
const createObjectURL = URL.createObjectURL;
URL.createObjectURL = (blob) => { exportedBlob = blob; return "blob:fixture"; };
document.createElement = () => ({click() {}});
exportChat({title:"RAW_EXPORT_TITLE", updated:Date.now(), messages:[{role:"assistant", text:"RAW_PARTIAL_ANSWER", error:"RAW_ERROR",
  errorUI:{key:"the connection closed early"}, reasoning:"RAW_REASONING"}]});
const exported = await exportedBlob.text();
assert.match(exported, /RAW_PARTIAL_ANSWER/);
assert.match(exported, /RAW_REASONING/);
assert.doesNotMatch(exported, /连接提前关闭/);
URL.createObjectURL = createObjectURL;
document.createElement = savedCreate;
setLanguage("en");
assert.deepEqual(localizeAnswerMeta("Stopped"), ["Stopped"]);

console.log("i18n persistence, locale switching, source coverage, placeholders, and English fallback OK");
