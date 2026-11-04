// serve/web/chats.js - the chats (kept in this browser), their folders and pins, and the chat itself: sending, an answer
// streamed (and taken back up from the server after a reload or a dropped connection), stopping, editing, regenerating,
// attachments, the context meter and the Markdown export.
import {SPRITE, api, coarse, ctxfmt, db, download, fmt, headers, store, uid} from "./lib.js";
import {P, PENALTIES, S, confirmDialog, defaultSystem, emit, levelFor, modelId, promptDialog, toast} from "./state.js";
import {mcpOn} from "./server.js";

// ------------------------------------------------------------------ the chats' list
let dbBroken = false;                 // IndexedDB unavailable (some private windows): localStorage, without pictures
function saveFolders() { store.set("folders", S.folders); store.set("folders-open", S.folderOpen); }
export const titleOf = (msgs) => {
  const u = msgs.find((m) => m.role === "user" && (m.text || (m.files || []).length));
  const t = u ? (u.text || (u.files || []).map((f) => f.name).join(", ")) : "";
  return t.replace(/\s+/g, " ").trim().slice(0, 60) || "New chat";
};
export const chatTitle = (c) => c.title || (c.messages.length ? titleOf(c.messages) : "New chat");
export function newChatObj() {
  return {id: uid(), title: "", created: Date.now(), updated: Date.now(), messages: [], system: defaultSystem(), thinking: P.thinking,
          thinkingFor: modelId()};
}
// the thinking level this chat asks for: its own (the box's speedometer), else the Settings default
// (a level the running model does not have - a chat from MiniMax's "Auto" on GLM - is the default's)
// the chat's own level - set under another model (a switch since), this model's default instead: a level does not
// carry over; a chat from before levels were tied to a model takes the nearest this one has
export const thinkingOf = (c = S.current) => {
  if (!c || !c.thinking) return P.thinking;
  const f = c.thinkingFor;
  if (f && !f.startsWith("|") && S.health.model && f !== modelId()) return P.thinking;
  return levelFor(c.thinking) || P.thinking;
};
export function setThinking(level) {
  const c = S.current;
  c.thinking = level;
  c.thinkingFor = modelId();
  if (c.messages.length) persist(c);
  emit();
  updateCtxMeter();
}

const slim = (c) => ({...c, messages: c.messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name}))}))});
export async function persist(c = S.current) {
  if (!c || !c.messages.length) return;
  if (!S.chats.includes(c)) S.chats.unshift(c);
  const plain = JSON.parse(JSON.stringify(c));    // no transient fields (the streaming cache is not enumerable)
  try {
    if (dbBroken) throw new Error("no IndexedDB");
    await db.put(plain);
  } catch (e) {
    dbBroken = true;
    try { localStorage.setItem("maya.chats-fallback", JSON.stringify(S.chats.slice(0, 30).map(slim))); }
    catch (e2) { /* full: the newest chats only next time */ }
  }
}
export function saveChat(c = S.current) {
  if (!c) return;
  c.updated = Date.now();
  if (!c.title || c.title === "New chat") c.title = titleOf(c.messages);
  S.chats.sort((a, b) => b.updated - a.updated);
  persist(c);
  emit();
}
export async function loadChats() {
  let chats;
  try { chats = await db.all(); }
  catch (e) { dbBroken = true; chats = store.get("chats-fallback", []); }
  if (!chats.length) {                            // the single chat of earlier versions becomes the first one
    const old = store.get("chat", []);
    if (old.length) {
      const c = newChatObj();
      c.messages = old;
      c.title = titleOf(old);
      chats.push(c);
      S.chats = chats;
      await persist(c);
    }
  }
  try { localStorage.removeItem("maya.chat"); localStorage.removeItem("strata.chat"); } catch (e) { /* ignore */ }
  for (const c of chats) for (const m of c.messages) if (!m.id) m.id = uid();
  chats.sort((a, b) => b.updated - a.updated);
  S.chats = chats;
  const want = store.get("current-chat", null);
  S.current = chats.find((c) => c.id === want) || chats[0] || newChatObj();
  emit();
}
export const chatById = (id) => S.chats.find((c) => c.id === id);

// an answer being written goes on in its own chat while another one is open (and shows again when its chat is)
export function openChat(c) {
  if (S.current === c) { S.view = "chat"; S.drawer = false; emit(); focusInput(); return; }
  S.current = c;
  store.set("current-chat", c.id);
  if (c.folder && !S.folderOpen[c.folder]) { S.folderOpen = {...S.folderOpen, [c.folder]: true}; saveFolders(); }
  cancelEdit();
  S.view = "chat";
  S.drawer = false;
  S.ctx = {tokens: null, approx: false, why: ""};
  emit();
  scrollDown(true);
  updateCtxMeter();
  resumePending();
}
export function newChat() {
  const c = S.current;
  if (c && !c.messages.length && c.system === defaultSystem()) {
    S.view = "chat"; S.drawer = false; emit(); focusInput(); return;
  }
  openChat(newChatObj());
  focusInput();
}
export async function deleteChat(c) {
  if (S.busy && S.busy.chat === c) { toast("warn", "Still writing", "Stop the answer first."); return; }
  S.chats = S.chats.filter((x) => x !== c);
  try { if (!dbBroken) await db.del(c.id); } catch (e) { /* ignore */ }
  if (dbBroken) {
    try { localStorage.setItem("maya.chats-fallback", JSON.stringify(S.chats.slice(0, 30).map(slim))); } catch (e) { /* ignore */ }
  }
  if (S.current === c) openChat(S.chats[0] || newChatObj());
  focusInput();
  emit();
  toast("info", "Chat deleted", chatTitle(c), 6000, {label: "Undo", run: () => {
    S.chats.push(c);
    S.chats.sort((a, b) => b.updated - a.updated);
    persist(c);
    emit();
  }});
}
export async function renameChat(c) {
  const v = await promptDialog({title: "Rename this chat", label: "Name", value: chatTitle(c), okLabel: "Rename"});
  if (v == null || !v.trim()) return;
  c.title = v.trim().slice(0, 80);
  persist(c);
  emit();
}
export async function chatInstructions(c) {
  const v = await promptDialog({title: "Instructions for this chat", label: "The model reads them before the chat",
    value: c.system || "", placeholder: "e.g. Answer briefly. Write code in TypeScript.", multiline: true,
    note: "Settings › Instructions has the ones new chats start with."});
  if (v == null) return;
  c.system = v;
  persist(c);
  emit();
  updateCtxMeter();
  toast("success", "Instructions saved", v.trim() ? "The next answer reads them first." : "This chat has none now.");
}
export function pinChat(c, on) {
  c.pinned = on || undefined;
  persist(c);
  emit();
}

// ------------------------------------------------------------------ folders ({id, name}, in this browser)
export async function newFolder(c = null) {
  const name = await promptDialog({title: "New folder", label: "Name", placeholder: "e.g. Maya engine", okLabel: "Create"});
  if (name == null || !name.trim()) return null;
  const f = {id: uid(), name: name.trim().slice(0, 60)};
  S.folders = [...S.folders, f];
  S.folderOpen = {...S.folderOpen, [f.id]: true};
  saveFolders();
  if (c) moveToFolder(c, f.id);
  emit();
  return f;
}
export function moveToFolder(c, id) {
  c.folder = id || undefined;
  if (id) S.folderOpen = {...S.folderOpen, [id]: true};
  saveFolders();
  persist(c);
  emit();
}
export function toggleFolder(id) {
  S.folderOpen = {...S.folderOpen, [id]: !S.folderOpen[id]};
  saveFolders();
  emit();
}
export async function renameFolder(f) {
  const v = await promptDialog({title: "Rename the folder", label: "Name", value: f.name, okLabel: "Rename"});
  if (v == null || !v.trim()) return;
  f.name = v.trim().slice(0, 60);
  S.folders = [...S.folders];
  saveFolders();
  emit();
}
export async function deleteFolder(f) {
  const inIt = S.chats.filter((c) => c.folder === f.id);
  const ok = await confirmDialog(`Delete the folder “${f.name}”?`,
    inIt.length ? `<p>Its ${inIt.length} chat${inIt.length > 1 ? "s stay" : " stays"}: they move back to the list.</p>`
                : "<p>It is empty.</p>", "Delete folder", true);
  if (!ok) return;
  for (const c of inIt) { c.folder = undefined; persist(c); }
  S.folders = S.folders.filter((x) => x !== f);
  delete S.folderOpen[f.id];
  saveFolders();
  emit();
}

// ------------------------------------------------------------------ the chat as the API sends it
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
export function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
// `light`: pictures as placeholders (the context meter's count needs only where they are)
export function apiMessages(chat, msgs = chat.messages, light = false) {
  const out = [];
  if (chat.system && chat.system.trim()) out.push({role: "system", content: chat.system.trim()});
  for (const m of msgs) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: light ? "data:," : i.url}}))] : text});
    } else if (!m.error) {
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and their
// results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // (an answer stopped before its text - while the model thought - goes back empty: without it the model reads two
  //  questions in a row and answers the one that was stopped)
  if (!ran.length) return m.text ? [{role: "assistant", content: m.text}] : m.stopped ? [{role: "assistant", content: ""}] : [];
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

// ------------------------------------------------------------------ the context meter
// How much of the model's context this chat takes: while an answer is written, the request's own prompt tokens plus the
// tokens written so far (the server's live count); otherwise the chat with what is being typed, counted by the server with
// the model's own template and tokenizer (POST /api/tokens).  "≈" only when a count can't be exact.
let ctxKey = null, ctxTimer = null, ctxSeq = 0;
function draftChat() {
  const c = S.current, base = S.editIndex != null ? c.messages.slice(0, S.editIndex) : c.messages;
  const text = S.draft.trim();
  if (!text && !S.attachments.length) return base;
  return [...base, {role: "user", text, images: S.attachments.filter((a) => a.kind === "image"),
                    files: S.attachments.filter((a) => a.kind === "file")}];
}
export function estimateTokens(msgs, c = S.current) {   // an older server without /api/tokens: about 3.5 characters a token
  let chars = (c.system || "").length;
  for (const m of msgs) chars += m.role === "user" ? userText(m).length : (m.text || "").length;
  return Math.round(chars / 3.5) + 8 * msgs.length;
}
// while an answer is written in the open chat: the server's live count (every poll)
export function liveCtx() {
  const live = S.metrics && S.metrics.live;
  if (!S.busy || S.busy.chat !== S.current || !live || live.state === "idle" || live.prompt_tokens == null) return null;
  return {tokens: live.prompt_tokens + (live.generated || 0), approx: false, why: "This request"};
}
export function updateCtxMeter() {
  clearTimeout(ctxTimer);
  if (liveCtx()) return;
  const msgs = draftChat();
  if (!msgs.length) { ctxKey = null; S.ctx = {tokens: null, approx: false, why: ""}; emit(); return; }
  ctxTimer = setTimeout(async () => {
    const c = S.current;
    const body = {messages: apiMessages(c, msgs, true), reasoning_effort: thinkingOf(c)};
    if (mcpOn()) body.strata_mcp = true;
    const key = JSON.stringify(body);
    if (key === ctxKey && S.ctx.tokens != null) return;
    const seq = ++ctxSeq;
    let tokens = null, approx = false;
    try {
      const j = await api("api/tokens", {method: "POST", body});
      tokens = j.tokens;
      approx = !!j.approximate;
    } catch (e) {
      tokens = estimateTokens(msgs);
      approx = true;
    }
    if (seq !== ctxSeq || liveCtx()) return;      // typed on, or an answer started meanwhile
    ctxKey = key;
    S.ctx = {tokens, approx, why: "This chat"};
    emit();
  }, 400);
}
// what the meter shows now: {tokens, approx, why, max, pct, level, title} or null
export function ctxMeter() {
  const max = S.health.max_context, c = liveCtx() || S.ctx;
  if (!max || c.tokens == null) return null;
  const pct = Math.min(100, (100 * c.tokens) / max);
  return {...c, max, pct, level: pct >= 95 ? "danger" : pct >= 80 ? "warn" : "",
          title: `${c.why}: ${fmt(c.tokens)} of the ${fmt(max)}-token context (${fmt(pct, pct < 10 ? 1 : 0)}%)` +
                 (c.approx ? " - approximate (images count once they are read)" : "") +
                 (pct >= 95 ? ". Nearly full: start a new chat, or raise the context size in Settings." : "")};
}

// ------------------------------------------------------------------ the view's hooks (the message list registers them)
export const painters = new Map();     // a message -> repaint its answer (while it streams)
export const chatView = {el: null, follow: true, held: false, lastTop: 0, onJump: null};
export function scrollDown(force) {
  const s = chatView.el;
  if (force) { chatView.follow = true; chatView.held = false; }
  if (!s) return;
  if (chatView.follow && !chatView.held) { s.scrollTop = s.scrollHeight; chatView.lastTop = s.scrollTop; }
  if (chatView.onJump) chatView.onJump();
}
export const focusInput = () => setTimeout(() => { const t = document.getElementById("composer-input"); if (t && !coarse()) t.focus(); }, 0);

// ------------------------------------------------------------------ an answer, streamed
const TOOL_ROUND_SEP = "\n\n";
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function answerMeta(m, n, secs) {
  const parts = [];
  if (n) {
    const t = m.timings || {};
    const r = t.predicted_per_second || (secs > 0.25 ? n / secs : null);
    parts.push(`${fmt(n)} tokens${r ? ` · ${fmt(r, 1)} tok/s` : ""}`);
    if (t.prompt_n != null && t.prompt_ms) {
      parts.push(`read ${fmt(t.prompt_n)} new prompt tokens in ${fmt(t.prompt_ms / 1000, 1)} s` +
                 (t.cache_n ? ` (${fmt(t.cache_n)} reused)` : ""));
    }
  }
  if (m.stopped) parts.push("stopped");
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) parts.push(`${ran} tool call${ran > 1 ? "s" : ""}`);
  if (m.limit) parts.push(`stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`);
  const s = parts.join(" · ");   // ("Stopped", not "stopped", when nothing comes before it: a stop has no usage)
  return s && s[0].toUpperCase() + s.slice(1);
}
const sleepOrVisible = (ms) => new Promise((ok) => {
  const go = () => { document.removeEventListener("visibilitychange", go); clearTimeout(tm); ok(); };
  const tm = setTimeout(go, ms);
  document.addEventListener("visibilitychange", go);
});
const setHint = (t) => { S.hint = t || ""; emit(); };

// One answer, streamed: a new one (POST), or - after a page reload - the one still running on the server (resumeJob)
export async function answer(resumeJob = null) {
  const chat = S.current, messages = chat.messages;
  let m;
  if (resumeJob) {
    m = messages[messages.length - 1];
    Object.assign(m, {text: "", reasoning: "", tools: [], error: undefined, meta: undefined, stopped: undefined});
  } else {
    m = {id: uid(), role: "assistant", text: "", reasoning: "", time: Date.now()};
    messages.push(m);
  }
  const controller = new AbortController();
  S.busy = {controller, msg: m, chat, job: resumeJob || undefined};
  emit();
  scrollDown(true);

  let body = null;
  if (!resumeJob) {
    body = {model: S.health.model, messages: apiMessages(chat, messages.slice(0, -1)), stream: true, reasoning_effort: thinkingOf(chat)};
    if (P.temperature > 0) {
      // (top-k and min-p go even when off - 0: no top-k cut, no min-p - else the run config's or the shared ones applied
      //  while the page said "off")
      Object.assign(body, {temperature: +P.temperature, top_p: +P.top_p, top_k: Math.max(0, +P.top_k || 0),
                           min_p: Math.max(0, +P.min_p || 0)});
    } else {
      body.temperature = 0;
    }
    // the penalties the user set (blank: none sent - no model has any as a standard)
    for (const k of PENALTIES) if (P[k] !== "" && P[k] != null && Number.isFinite(+P[k])) body[k] = +P[k];
    if (P.seed) body.seed = +P.seed;
    if (P.max) body.max_tokens = +P.max;
    if (mcpOn()) body.strata_mcp = true;           // this server may run MCP tools for it
    // the server keeps the answer running if this connection drops (a phone's screen sleeping mid-answer, or a reload):
    // the page reconnects to it by id and carries on from the last chunk it read (strata_resume)
    body.strata_resume = true;
  }
  let firstAt = null, thinkStart = null, usage = null, pending = false, lastPaint = 0, jobId = resumeJob, nRecv = 0, finished = false;
  let ended = false;                               // a repaint queued while streaming must not land after the end
  const paint = () => {
    pending = false;
    if (ended || chat !== S.current) return;
    const f = painters.get(m);
    if (!f) return;
    lastPaint = performance.now();
    f();
    scrollDown();
  };
  const schedule = () => {                         // a long answer repaints at most ten times a second
    if (pending) return;
    pending = true;
    const wait = m.text.length > 20000 ? Math.max(0, 100 - (performance.now() - lastPaint)) : 0;
    setTimeout(() => requestAnimationFrame(paint), wait);
  };
  const consume = async (r) => {
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") { finished = true; continue; }
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        nRecv++;
        if (j.id && !jobId) {
          jobId = j.id;
          S.busy.job = jobId;
          m.job = jobId;
          m.pending = true;
          saveChat(chat);                                     // a reload can take this answer back up
        }
        if (j.error) { finished = true; throw new Error(j.error.message || "the engine reported an error"); }
        if (j.usage) usage = j.usage;
        if (j.timings) m.timings = j.timings;
        if (j.strata_mcp) { onTool(m, j.strata_mcp); emit(); }
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += TOOL_ROUND_SEP;
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += TOOL_ROUND_SEP;
          m.text += d.content;
        }
        schedule();
      }
    }
  };
  // reconnect to the running answer by id (every 2 s for up to 10 minutes, at once when the page is shown again)
  const follow = async (gone) => {
    const t0 = Date.now();
    setHint(resumeJob ? "Taking the answer back up…" : "Connection lost - reconnecting to the answer…");
    for (;;) {
      if (controller.signal.aborted) throw new DOMException("stopped", "AbortError");
      if (Date.now() - t0 > 600000) throw gone;
      try {
        const r2 = await fetch(`v1/strata/stream?id=${encodeURIComponent(jobId)}&from=${nRecv}`,
                               {headers: headers(false), signal: controller.signal});
        if (r2.status === 404) throw gone;
        if (r2.ok) {
          setHint("");
          await consume(r2);
          if (finished) return;
        }
      } catch (e2) {
        if (e2.name === "AbortError" || e2 === gone) throw e2;
        /* still offline: try again */
      }
      await sleepOrVisible(2000);
    }
  };
  const failed = async (r) => {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    if (r.status === 401) msg = "This server needs an API key: add it in Settings › Connect your tools.";
    return new Error(msg);
  };
  try {
    if (resumeJob) {
      await follow(new Error("This answer was cut off when the page was reloaded, and the server no longer has it."));
    } else {
      const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                     signal: controller.signal});
      if (!r.ok) throw await failed(r);
      try {
        await consume(r);
        if (!finished) throw new Error("the connection closed early");
      } catch (e) {
        if (e.name === "AbortError" || finished || !jobId) throw e;
        await follow(e);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (S.hint && !S.reload) S.hint = "";
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (usage) m.usage = {prompt: usage.prompt_tokens, completion: usage.completion_tokens,
                        cached: (usage.prompt_tokens_details || {}).cached_tokens || 0};
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  m.meta = answerMeta(m, n, firstAt ? (performance.now() - firstAt) / 1000 : 0);
  delete m.pending;
  delete m.job;
  if (m._md) m._md = null;
  ended = true;
  S.busy = null;
  saveChat(chat);
  emit();
  if (chat === S.current) { scrollDown(); updateCtxMeter(); }
  resumePending();                                 // the open chat's answer cut off by a reload, now that none runs
}
// after a reload: an answer the page was streaming when it went away is taken back up from the server
export function resumePending() {
  const ms = S.current.messages, m = ms[ms.length - 1];
  if (!S.busy && m && m.role === "assistant" && m.pending && m.job) answer(m.job);
}

export async function send() {
  const text = S.draft.trim();
  if (S.reload) { toast("info", "The model is reloading", "Send it once the new context size is loaded."); return; }
  if ((!text && !S.attachments.length) || S.busy) return;
  const c = S.current;
  if (S.editIndex != null) c.messages.splice(S.editIndex);       // the edited message and every answer after it
  c.messages.push({id: uid(), role: "user", text, images: S.attachments.filter((a) => a.kind !== "file"),
                   files: S.attachments.filter((a) => a.kind === "file"), time: Date.now()});
  S.editIndex = null;
  S.attachments = [];
  S.draft = "";
  S.view = "chat";
  saveChat(c);
  store.set("current-chat", c.id);
  focusInput();
  await answer();
}
export function regenerate() {
  const ms = S.current.messages;
  if (S.busy || S.reload) return;
  while (ms.length && ms[ms.length - 1].role === "assistant") ms.pop();
  if (!ms.length) return;
  answer();
}
export function editMessage(i) {
  if (S.busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const m = S.current.messages[i];
  S.editIndex = i;
  S.draft = m.text || "";
  S.attachments = [...(m.images || []).filter((a) => a.url).map((a) => ({...a, kind: "image"})),
                   ...(m.files || []).filter((f) => f.text != null).map((f) => ({...f, kind: "file"}))];
  emit();
  focusInput();
  updateCtxMeter();
}
export function cancelEdit(clear = true) {
  if (S.editIndex == null) return;
  S.editIndex = null;
  if (clear) { S.draft = ""; S.attachments = []; }
  emit();
  updateCtxMeter();
}
export function stopAnswer() {
  const b = S.busy;
  if (!b) return;
  // the server runs a dropped answer on, so Stop says so explicitly (then the stream ends with it)
  if (b.job) fetch("v1/strata/cancel", {method: "POST", headers: headers(true), body: JSON.stringify({id: b.job})}).catch(() => {});
  b.controller.abort();
}

// ------------------------------------------------------------------ attachments: the + menu, a drop, a pasted picture
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
// the server says how big a text file the context holds (health.max_text_file: about 4 bytes a token); 512 KB before it did
const textLimit = () => S.health.max_text_file || 512 * 1024;
const sizeLabel = (n) => n >= 1048576 ? `${(n / 1048576).toFixed(n >= 10485760 ? 0 : 1)} MB` : `${Math.round(n / 1024)} KB`;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
const attached = (a) => { S.attachments = [...S.attachments, a]; emit(); updateCtxMeter(); };
export function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!S.health.images) { toast("warn", "Images are off", "This model runs without images: Models › Engine settings turns them on when it can."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => attached({kind: "image", name: f.name || "pasted image", url: r.result});
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${S.health.images ? " or images" : ""}.`); continue; }
    if (f.size > textLimit()) {
      const ctx = S.health.max_context;
      toast("warn", "File too large", `${f.name} (${sizeLabel(f.size)}) is over the limit for this model's ${ctx ? ctxfmt(ctx) + "-token " : ""}context ` +
        `(${sizeLabel(textLimit())}).${ctx ? " A larger context size in Settings raises it; or attach a part of the file." : ""}`);
      continue;
    }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attached({kind: "file", name: f.name, text});
    };
    r.readAsText(f);
  }
}
export function removeAttachment(i) {
  S.attachments = S.attachments.filter((_, k) => k !== i);
  emit();
  updateCtxMeter();
}

// ------------------------------------------------------------------ export (Markdown)
export function exportChat(c = S.current) {
  if (!c.messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const head = `# ${chatTitle(c)}\n\n` + (c.system ? `> Instructions: ${c.system.replace(/\n/g, " ")}\n\n` : "");
  const md = head + c.messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${S.health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  download(`maya-chat-${new Date(c.updated || Date.now()).toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`, md, "text/markdown");
}

// ------------------------------------------------------------------ clear this browser
export async function clearBrowser() {
  const ok = await confirmDialog("Clear this browser's chats and settings?",
    "<p>Every chat, its folders, the chat settings, the instructions and the API key kept in this browser are removed. " +
    "The model, its files and the server's own settings are not touched.</p>", "Clear everything", true);
  if (!ok) return;
  try { await db.clear(); } catch (e) { /* nothing stored there */ }
  try {
    for (const k of Object.keys(localStorage)) if ((k.startsWith("maya.") || k.startsWith("strata.")) && k !== "maya.theme" && k !== "maya.scheme") localStorage.removeItem(k);
  } catch (e) { /* ignore */ }
  location.reload();
}

export const ctxLabel = () => (S.health.max_context ? ctxfmt(S.health.max_context) : "");
export {SPRITE};
