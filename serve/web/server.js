// serve/web/server.js - what the page asks the server: its health (the model, its defaults), the live metrics every
// second, the MCP servers, the context size and port (each a restart), the update, the shared settings and the report.
import {api, ctxfmt, esc, fmt, headers} from "./lib.js";
import {tr} from "./i18n.js";
import {DEFAULTS, P, PENALTIES, S, THINKING, confirmDialog, emit, loadPrefs, modelId, newModelCheck, setFamily, toast} from "./state.js";

// Only phases produced by this server are UI labels. Future/third-party phases stay verbatim.
export function phaseText(phase) {
  const labels = {thinking: "Thinking", answering: "Answering", "reading the prompt": "Reading the prompt",
                  "tool call complete": "Tool call complete"};
  if (Object.hasOwn(labels, phase)) return tr(labels[phase]);
  const tool = /^writing a tool call: (.+)$/.exec(phase || "");
  return tool ? tr("Writing a tool call: {tool}", {tool: tool[1]}) : phase;
}

// ------------------------------------------------------------------ health: the model and its recommended settings
let onHealth = () => {};
export const whenHealth = (f) => { onHealth = f; };
export async function loadHealth() {
  try {
    const h = await (await fetch("health")).json();
    S.health = h;
    // the server's recommended settings for this model (its run config) replace the built-in defaults, and are what the
    // chat uses until its user picks settings of their own
    const d = h.defaults || {};
    setFamily(h.model);                            // (its thinking levels: MiniMax's are not GLM's)
    if (THINKING.includes(d.reasoning_effort)) DEFAULTS.thinking = d.reasoning_effort;   // (the run config's level)
    if (d.temperature !== undefined) DEFAULTS.temperature = d.temperature;
    if (d.top_p !== undefined) DEFAULTS.top_p = d.top_p;
    if (d.temperature !== undefined || d.top_k !== undefined) DEFAULTS.top_k = d.top_k !== undefined ? d.top_k : 0;
    if (d.temperature !== undefined || d.min_p !== undefined) DEFAULTS.min_p = d.min_p !== undefined ? d.min_p : 0;
    const was = P.thinking;                        // (the level the empty chat on screen took from the defaults)
    const switched = newModelCheck();              // another model than last time: its own standard parameters
    loadPrefs();                                   // each model keeps its own (DeepSeek's, MiniMax's are not GLM's)
    // ... made before the model was known: it starts at this model's default, as a change in Settings moves it
    if (S.current && !S.current.messages.length && (S.current.thinking === was || switched)) {
      S.current.thinking = P.thinking;
      S.current.thinkingFor = modelId();
    }
    if (switched) toast("info", tr("{model}: its own settings", {model:h.model_label || h.model}), tr("A new model starts at its standard parameters and thinking level; change them in Settings › Chat defaults."), 6000);
    emit();
    onHealth();
    if (h.status === "reloading") watchReload();
    if (h.status === "updating" && !S.updateWatch) loadUpdate().then((u) => u && u.state && watchUpdate(u.state.to));
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ the live metrics (the Monitor, the status, the meter)
let fails = 0, keyWarned = false, mcpTick = 0;
export const reqAll = {on: false};       // the request table: the last 12, or every one the server keeps
export async function poll() {
  try {
    const r = await fetch(reqAll.on ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      S.conn = "key";
      if (!keyWarned) { keyWarned = true; toast("warn", tr("API key needed"), tr("This server needs a key: add it in Settings › Connect your tools."), 6000); }
    } else if (r.ok) {
      S.metrics = await r.json();
      S.conn = "ok";
      fails = 0;
      const rl = S.metrics.reload;
      if (rl && (rl.state === "waiting" || rl.state === "running")) watchReload();
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++fails >= 3) S.conn = "down";
  }
  if (!document.hidden) emit();
  if ((S.view === "monitor" || S.settings === "tools") && ++mcpTick % 10 === 0) loadMcp();   // (they change rarely: every 10 s)
  setTimeout(poll, document.hidden ? 5000 : 1000);               // a hidden tab looks five times less often
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) emit(); });

// ------------------------------------------------------------------ Monitor figures that are computed (unit-tested)
// The Speculation card: which drafters the engine runs with (the server's engine.speculation, empty: none) and how many of
// the tokens they drafted the model kept - live while an answer is written (the engine's STAT draft_acc / draft_off, the
// drafts since the request began), else the last answer's (its DONE line).  Three states, never a made-up 0%:
//   off      no drafter is configured, or the running request asked for none (spec=1)  -> "off", "off"
//   waiting  drafting is on, nothing drafted yet                                       -> the mode, "-"
//   kept     drafting is on and drafts were offered                                    -> the mode, "62% (124 of 200)"
export function speculationView(m, now = Date.now() / 1000) {
  const eng = (m && m.engine) || {}, live = (m && m.live) || {}, tn = m && m.tiers && m.tiers.now, last = ((m && m.requests) || [])[0];
  const modes = Array.isArray(eng.speculation) ? eng.speculation : null;
  if (modes == null) return {state: "none", name: null, text: ""};                  // (a server that does not say)
  const running = live.state === "generating";
  const reqOff = running ? !!live.spec_off : !!(last && last.spec_off);
  const off = modes.length === 0 || (running && !!live.spec_off);
  if (off) return {state: "off", name: "off", text: "off", pct: null};
  // the figures: this answer's while it is written (a reading from before it began is the last answer's: not this one's),
  // else the last answer's
  let acc = null, offered = null, pct = null;
  const since = running && live.elapsed_s != null ? now - live.elapsed_s - 1 : null;
  if (running) {
    if (tn && tn.time && (since == null || tn.time >= since) && now - tn.time <= 30) {
      if (tn.draft_off != null) { offered = +tn.draft_off; acc = +tn.draft_acc || 0; }
      else if (tn.drafts != null) pct = 100 * tn.drafts;                              // (an engine that gives the share only)
    }
  } else if (last && !reqOff && last.drafts_offered != null) {
    offered = +last.drafts_offered; acc = +last.drafts_accepted || 0;
  }
  const name = modes.join(" + ");
  if (reqOff && !running) return {state: "off", name, text: "off", pct: null};      // (the last answer asked for no drafting)
  if (offered != null && offered > 0) pct = 100 * acc / offered;
  if (pct == null) return {state: "waiting", name, text: "\u2014", pct: null};
  const text = offered != null && offered > 0 ? `${fmt(pct, 0)}% (${tr("{accepted} of {offered}", {accepted:fmt(acc),offered:fmt(offered)})})` : `${fmt(pct, 0)}%`;
  return {state: "kept", name, text, pct, accepted: acc, offered,
          short: `${fmt(pct, 0)}%`, detail: offered != null && offered > 0 ? tr("{accepted} of {offered} drafted", {accepted:fmt(acc),offered:fmt(offered)}) : ""};
}

// "Where the experts live": the experts in VRAM, in RAM and on the SSD only, each with its bytes (the engine's
// vram_gb / ram_gb / disk_gb; a count alone when an engine reports no bytes)
export function tierShares(n, eng = {}) {
  const ds4 = eng.engine_kind === "ds4";
  const vu = n.vram_used || 0, ru = n.ram_used || 0;
  const total = (ds4 ? vu + (n.cpu_n != null ? n.cpu_n : +eng.cpu_experts || 0) : eng.experts) || Math.max(1, vu + ru);
  const disk = Math.max(0, total - vu - ru);
  const gbOf = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
  return {vu, ru, total, disk, vramGb: gbOf(n.vram_gb), ramGb: gbOf(n.ram_gb), diskGb: gbOf(n.disk_gb)};
}

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request, "strata_mcp":
// true, which only this page sends); Settings lists the servers and what they offer.
let mcpRetry = null;
export async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    S.mcp = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  emit();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((S.mcp.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
export const mcpOn = () => P.mcp !== false && S.mcp.tools > 0;

// ------------------------------------------------------------------ the context size (a reload of the model)
export const GLM_KV_BYTES = 12288;   // GLM's attention cache per token (KV streaming keeps all of it in RAM)
export const CTX_STOPS = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288, 786432, 1048576];
export async function loadCtxInfo() {
  try {
    const r = await fetch("api/context", {headers: headers()});
    S.ctxInfo = r.ok ? await r.json() : null;
  } catch (e) { S.ctxInfo = null; }
  emit();
  return S.ctxInfo;
}
// the Context size slider's stops (Settings > Chat defaults and the Models page's Engine settings): CTX_STOPS within a model's
// range ({min, max}: GET /api/context, or /api/models' context_range), and the sizes given (the one now, a draft's)
export function ctxStopsIn(r, ...sizes) {
  if (!r) return [];
  const own = sizes.map(Number).filter((n) => n > 0);
  return [...new Set([...CTX_STOPS.filter((v) => v >= r.min && v <= r.max), ...own])].sort((a, b) => a - b);
}
export const ctxStops = () => (S.ctxInfo ? ctxStopsIn(S.ctxInfo, S.ctxInfo.context) : []);
// what a context size costs here: the engine's own measurement of the current size, scaled
export function ctxEstimate(n) {
  const c = S.ctxInfo;
  // a streamed cache - KV streaming - keeps a fixed window in VRAM and every position in pinned RAM (maya-mm3): its RAM
  // scales (the engine's own count at this size, kv_ram_gb), against the RAM free now
  if (c && c.kv_resident && c.kv_ram_gb && c.context) {
    const est = (c.kv_ram_gb / c.context) * n;
    return {ram: true, gb: est, delta: est - c.kv_ram_gb, free: c.ram_free_gb != null ? +c.ram_free_gb : null};
  }
  // (... without its RAM count, scaling its fixed VRAM window by the context would mislead)
  if (!c || !c.kv_gb || !c.kv_ctx || c.kv_resident) return null;
  const perTok = c.kv_gb / c.kv_ctx;
  const est = perTok * n, now = perTok * c.context;
  const expertGb = c.vram_gb && c.vram_slots ? +c.vram_gb / c.vram_slots : null;
  return {gb: est, delta: est - now, experts: expertGb ? Math.round((est - now) / expertGb) : null, vram: +c.vram_gb || null};
}
// the Context size slider's line under it (Settings > Chat defaults and the Models page alike): what n tokens of context cost.
// `live`: the running engine's own measurement scaled (ctxEstimate); else from the cache's bytes per token - `ram`: a
// streamed cache, every position in pinned RAM and `win` of them in VRAM (`whole`: a context up to `win` is not streamed,
// GLM's), `vram`: a cache wholly in VRAM.  `now`: the size it has now (the change in GB is shown against it).
export function ctxHint(n, {live = false, now = null, win = 0, whole = false, ram = 0, vram = 0} = {}) {
  const e = live ? ctxEstimate(n) : null, same = now == null || n === now;
  const more = (d) => (same || !d ? "." : tr(" · {gb} GB {direction} than now.", {gb:fmt(Math.abs(d),1),direction:d>0 ? tr("more") : tr("less")}));
  const W = ctxfmt(win || (S.ctxInfo && S.ctxInfo.kv_resident) || 0);
  if (e && e.ram) return tr("≈ {gb} GB of RAM for the context (kept pinned; {window} positions of it also in VRAM)", {gb:fmt(e.gb,1),window:W}) + more(e.delta);
  if (e) return tr("≈ {gb} GB of GPU memory for the context", {gb:fmt(e.gb,1)}) + (same ? "." : e.experts
    ? tr(" · about {count} {direction} experts held in VRAM than now{slower}.", {count:fmt(Math.abs(e.experts)),direction:e.experts>0 ? tr("fewer") : tr("more"),slower:e.experts>0 ? tr(" (answers a little slower)") : ""}) : ".");
  const gb = (bytes, k) => (bytes * k) / 2 ** 30;
  if (ram && win) {
    if (whole && n <= win) return tr("Up to {window} the whole cache stays in VRAM.", {window:W});
    return tr("≈ {gb} GB of RAM for the context (kept pinned; {window} positions of it also in VRAM)", {gb:fmt(gb(ram,n),1),window:W}) +
      (now != null && whole && now <= win ? "." : more(gb(ram, n - (now || 0))));
  }
  if (vram) return tr("≈ {gb} GB of GPU memory for the context, taken from the experts", {gb:fmt(gb(vram,n),1)}) + more(gb(vram, n - (now || 0)));
  return tr("How much text the model holds at once: the chat, files and tool output. More context takes GPU memory from the experts, so answers get a little slower.");
}
// how long loading the model takes here: as long as its last load (GET /api/context's load_s), else the usual 1-3 minutes
export function loadTime() {
  const s = S.ctxInfo && +S.ctxInfo.load_s;
  if (!(s > 0)) return tr("about 1–3 minutes");
  const m = Math.round(s / 60);
  return m <= 1 ? tr("about a minute, as its last load took") : tr("about {minutes} minutes, as its last load took", {minutes:m});
}
export async function confirmContext(n, chatTokens) {
  const c = S.ctxInfo, e = ctxEstimate(n);
  const parts = [tr("<p>The engine restarts with a <b>{context}</b>-token context (now {now}). That takes {time}.</p>", {context:esc(ctxfmt(n)),now:esc(ctxfmt(c.context)),time:esc(loadTime())}),
                 tr("<ul><li>Your chats stay here; the next answer reads the chat again.</li>") +
                 tr("<li>Requests from other apps get a “try again” until the model is back.</li>") +
                 tr("<li>If the new size does not fit, the current one comes back by itself.</li>") +
                 (S.busy ? tr("<li>The answer being written finishes first.</li>") : "") + "</ul>"];
  if (e && e.ram) parts.push(tr('<p class="muted">Its cost here: about {gb} GB of RAM, kept pinned for the context.</p>', {gb:esc(fmt(e.gb,1))}));
  else if (e) parts.push(tr('<p class="muted">Its cost here: about {gb} GB of GPU memory', {gb:esc(fmt(e.gb,1))}) +
                    (e.experts ? tr(", {count} {direction} experts in VRAM than now.", {count:esc(fmt(Math.abs(e.experts))),direction:e.experts>0 ? tr("fewer") : tr("more")}) : ".") + "</p>");
  if (chatTokens > n) parts.push(tr('<p class="warn-text">The open chat holds about {tokens} tokens: more than the new size, so it will not fit. Start a new chat after the reload, or keep a larger context.</p>', {tokens:esc(fmt(chatTokens))}));
  const ok = await confirmDialog(tr("Reload the model with a {context} context?", {context:ctxfmt(n)}), parts.join(""), tr("Reload with {context}", {context:ctxfmt(n)}));
  if (!ok) return false;
  try {
    await api("api/context", {method: "POST", body: {max_context: n}});
  } catch (err) {
    toast("error", tr("The context did not change"), err.message, 6000);
    return false;
  }
  watchReload(n);
  return true;
}
// a reload in progress: the message box waits, the status says so, and the page follows it to the end
export async function watchReload(to) {
  if (S.reload) return;
  S.reload = {to: to || (S.ctxInfo && S.ctxInfo.reload && S.ctxInfo.reload.to) || null};
  S.hint = S.reload.to ? {key: "Reloading the model with a {context} context…", values: {context: ctxfmt(S.reload.to)}}
                       : {key: "Reloading the model…"};
  emit();
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 2000));
    const info = await loadCtxInfo();
    const rl = info && info.reload;
    if (!info) continue;                           // the server is busy restarting: ask again
    if (!S.reload.to && rl) S.reload.to = rl.to;
    if (rl && (rl.state === "waiting" || rl.state === "running")) { emit(); continue; }
    S.reload = null;
    S.hint = "";
    await loadHealth();
    if (rl && rl.state === "failed") toast("error", tr("The context did not change"), rl.error || tr("the engine did not start with it"), 9000);
    else toast("success", tr("Model reloaded"), tr("It runs with a {context} context now.", {context:ctxfmt(info.context)}), 5000);
    // (the same setting on the Models page: its Engine settings read the run config again)
    try { window.dispatchEvent(new Event("maya:context")); } catch (e) { /* (no window: a test) */ }
    emit();
    return;
  }
}

// ------------------------------------------------------------------ the thinking budget (applies at once, no restart)
export async function loadThinkBudget() {
  try { S.thinkBudget = await api("api/thinking-budget"); } catch (e) { S.thinkBudget = null; }
  emit();
  return S.thinkBudget;
}
export const budgetError = (text) => {
  const t = String(text).trim();
  if (!/^\d+$/.test(t)) return tr("Use a whole number of tokens.");
  const n = +t;
  return n === 0 || (n >= 256 && n <= 262144) ? "" : tr("Use 0 (no limit) or 256 to 262144 tokens.");
};
export async function saveThinkBudget(text) {
  const bad = budgetError(text);
  if (bad) { toast("error", tr("The thinking budget did not change"), bad, 6000); return false; }
  try {
    S.thinkBudget = await api("api/thinking-budget", {method: "POST", body: {budget: +String(text).trim()}});
    emit();
    toast("success", tr("Thinking budget saved"), +text ? tr("Thinking stops at {count} tokens, from the next answer.", {count:fmt(+text)}) : tr("Thinking has no limit, from the next answer."), 5000);
    return true;
  } catch (e) {
    toast("error", tr("The thinking budget did not change"), e.message, 6000);
    return false;
  }
}

// ------------------------------------------------------------------ the port (Maya starts again on it)
export async function loadPortInfo() {
  try {
    const r = await fetch("api/port", {headers: headers()});
    S.portInfo = r.ok ? await r.json() : null;
  } catch (e) { S.portInfo = null; }
  emit();
  return S.portInfo;
}
export async function confirmPort(n) {
  const now = S.portInfo.port, url = `${location.protocol}//${location.hostname}:${n}${location.pathname}`;
  if (n < 1024 || n > 65535) { toast("error", tr("The port did not change"), tr("Use a number from 1024 to 65535."), 6000); return false; }
  const ok = await confirmDialog(tr("Move Maya to port {port}?", {port:n}),
    tr("<p>Maya listens on port <b>{port}</b> now. ", {port:esc(String(now))}) + (S.portInfo.restarts
      ? tr("Saving starts it again on the new port: the model loads again, {time}.</p>", {time:esc(loadTime())}) +
        tr("<ul><li>This page moves to <b>") + esc(url) + tr("</b> when Maya is back.</li>") +
        tr("<li>Apps that use the API need the new port.</li>") + (S.busy ? tr("<li>The answer being written finishes first.</li>") : "") + "</ul>"
      : tr("The new port is saved and applies from the next start.</p>")), tr("Use port {port}", {port:n}));
  if (!ok) return false;
  try {
    const out = await api("api/port", {method: "POST", body: {port: n}});
    if (!out.restarts) { toast("success", tr("Port saved"), tr("Maya uses port {port} from its next start.", {port:n}), 6000); loadPortInfo(); return true; }
  } catch (err) {
    toast("error", tr("The port did not change"), err.message, 6000);
    return false;
  }
  S.hint = {key: "Moving Maya to port {port}…", values: {port:n}};
  emit();
  toast("info", tr("Starting again"), tr("Maya moves to port {port}; this page follows when it is back.", {port:n}), 8000);
  // the new address answers once Maya is back (no-cors: another port is another origin; an answer at all is enough)
  for (let i = 0; i < 300; i++) {
    await new Promise((res) => setTimeout(res, 2000));
    try {
      await fetch(`${location.protocol}//${location.hostname}:${n}/health`, {mode: "no-cors", cache: "no-store"});
      location.href = url;
      return true;
    } catch (e) { /* not up yet */ }
  }
  S.hint = "";
  emit();
  toast("error", tr("Maya is not back yet"), tr("Open {url} when it is.", {url}), 30000);
  return true;
}

// ------------------------------------------------------------------ "Use these for other apps too" (GET/POST /settings)
export async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) { const j = await r.json(); S.shared = {on: !!j.shared, now: j.defaults || null}; emit(); }
  } catch (e) { /* an older server: the switch just stays off */ }
}
export function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking, temperature: +s.temperature};
  // (top-k and min-p even when off: 0 - an app's request then gets no cut, not the run config's)
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p, top_k: Math.max(0, +s.top_k || 0), min_p: Math.max(0, +s.min_p || 0)});
  for (const k of PENALTIES) if (s[k] !== "" && s[k] != null && Number.isFinite(+s[k])) d[k] = +s[k];   // (only those set)
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  return d;
}
export async function saveShared(on) {
  const j = await api("settings", {method: "POST", body: {defaults: on ? sharedDefaults(P) : null}});
  S.shared = {on: !!j.shared, now: j.defaults || null};
  emit();
}
let sharedTimer = null;
// the chat settings changed: the server's copy follows (when they are shared)
export function syncShared() {
  if (!S.shared.on) return;
  clearTimeout(sharedTimer);
  sharedTimer = setTimeout(() => saveShared(true).catch((e) => toast("error", tr("Not saved for other apps"), e.message, 6000)), 600);
}

// ------------------------------------------------------------------ updates
// the server asks GitHub for the latest release (at most every six hours) and says whether this folder can update itself
// (a git checkout without local changes, started by maya.sh / START-MAYA.bat); else the steps by hand
export async function loadUpdate(force = false) {
  try {
    const r = await fetch(force ? "api/update?force=1" : "api/update", {headers: headers()});
    S.update = r.ok ? await r.json() : null;
  } catch (e) { S.update = null; }
  emit();
  return S.update;
}
export async function startUpdate() {
  const v = S.update && S.update.latest && S.update.latest.version;
  if (!v) return;
  const ok = await confirmDialog(tr("Update Project Maya to v{version}?", {version:v}),
    tr("<p>Only Maya's code is updated: <b>the model is not downloaded again</b>, and its files, your settings and the chats in this browser stay.</p><p>Maya finishes the answer it is writing, downloads the new code (git), compiles only the engine files that changed (usually a few minutes; the terminal shows the progress) and loads the model again. Apps using the API are asked to try again meanwhile.</p>"), tr("Update to v{version}", {version:v}));
  if (!ok) return;
  try {
    await api("api/update", {method: "POST", body: {}});
  } catch (e) {
    toast("error", tr("The update did not start"), e.message, 9000);
    loadUpdate();
    return;
  }
  watchUpdate(v);
}
// until the server answers with the new version (then the page reloads for its new files), or says it failed
export async function watchUpdate(to) {
  if (S.updateWatch) return;
  S.updateWatch = {to, down: false};
  emit();
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 3000));
    let h = null;
    try {
      const r = await fetch("health", {cache: "no-store"});
      h = r.ok ? await r.json() : null;
    } catch (e) { h = null; }
    if (!h) {                                      // stopped: compiling, or loading the model
      if (!S.updateWatch.down) { S.updateWatch.down = true; emit(); }
      continue;
    }
    if (h.version === to) {
      toast("success", tr("Updated to v{version}", {version:to}), tr("Reloading the page…"), 4000);
      setTimeout(() => location.reload(), 1200);
      return;
    }
    if (h.status === "updating") continue;
    const restarted = S.updateWatch.down;
    S.updateWatch = null;
    await loadUpdate();
    if (!restarted) toast("error", tr("The update did not happen"), (S.update && S.update.state && S.update.state.error) || "", 9000);
    else toast("warn", tr("Maya started again on the same version"), tr("The update's files did not arrive; the terminal shows why."), 9000);
    return;
  }
}

// ------------------------------------------------------------------ the troubleshooting report (GET /api/report)
export async function fetchReport() {
  const r = await fetch("api/report", {headers: headers()});
  if (!r.ok) throw new Error(r.status === 401 ? tr("this server needs its API key (Settings › Connect your tools)") : `HTTP ${r.status}`);
  return r.text();
}
