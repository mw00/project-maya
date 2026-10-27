// serve/web/server.js - what the page asks the server: its health (the model, its defaults), the live metrics every
// second, the MCP servers, the context size and port (each a restart), the update, the shared settings and the report.
import {api, ctxfmt, esc, fmt, headers} from "./lib.js";
import {DEFAULTS, P, S, THINKING, confirmDialog, emit, loadPrefs, setFamily, toast} from "./state.js";

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
    const was = P.thinking;                        // (the level the empty chat on screen took from the defaults)
    loadPrefs();                                   // each model keeps its own (DeepSeek's, MiniMax's are not GLM's)
    // ... made before the model was known: it starts at this model's default, as a change in Settings moves it
    if (S.current && !S.current.messages.length && S.current.thinking === was) S.current.thinking = P.thinking;
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
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it in Settings › Connect your tools.", 6000); }
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
export const CTX_STOPS = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288, 786432, 1048576];
export async function loadCtxInfo() {
  try {
    const r = await fetch("api/context", {headers: headers()});
    S.ctxInfo = r.ok ? await r.json() : null;
  } catch (e) { S.ctxInfo = null; }
  emit();
  return S.ctxInfo;
}
export function ctxStops() {
  const c = S.ctxInfo;
  if (!c) return [];
  let stops = CTX_STOPS.filter((v) => v >= c.min && v <= c.max);
  if (!stops.includes(c.context)) stops = [...stops, c.context].sort((a, b) => a - b);
  return stops;
}
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
// how long loading the model takes here: as long as its last load (GET /api/context's load_s), else the usual 1-3 minutes
export function loadTime() {
  const s = S.ctxInfo && +S.ctxInfo.load_s;
  if (!(s > 0)) return "about 1–3 minutes";
  const m = Math.round(s / 60);
  return `about ${m <= 1 ? "a minute" : `${m} minutes`}, as its last load took`;
}
export async function confirmContext(n, chatTokens) {
  const c = S.ctxInfo, e = ctxEstimate(n);
  const parts = [`<p>The engine restarts with a <b>${esc(ctxfmt(n))}</b>-token context (now ${esc(ctxfmt(c.context))}). ` +
                 `That takes ${esc(loadTime())}.</p>`,
                 "<ul><li>Your chats stay here; the next answer reads the chat again.</li>" +
                 "<li>Requests from other apps get a “try again” until the model is back.</li>" +
                 "<li>If the new size does not fit, the current one comes back by itself.</li>" +
                 (S.busy ? "<li>The answer being written finishes first.</li>" : "") + "</ul>"];
  if (e && e.ram) parts.push(`<p class="muted">Its cost here: about ${esc(fmt(e.gb, 1))} GB of RAM, kept pinned for the context.</p>`);
  else if (e) parts.push(`<p class="muted">Its cost here: about ${esc(fmt(e.gb, 1))} GB of GPU memory` +
                    (e.experts ? `, ${esc(fmt(Math.abs(e.experts)))} ${e.experts > 0 ? "fewer" : "more"} experts in VRAM than now.` : ".") + "</p>");
  if (chatTokens > n) parts.push(`<p class="warn-text">The open chat holds about ${esc(fmt(chatTokens))} tokens: more than the new size, so it will not fit. Start a new chat after the reload, or keep a larger context.</p>`);
  const ok = await confirmDialog(`Reload the model with a ${ctxfmt(n)} context?`, parts.join(""), `Reload with ${ctxfmt(n)}`);
  if (!ok) return false;
  try {
    await api("api/context", {method: "POST", body: {max_context: n}});
  } catch (err) {
    toast("error", "The context did not change", err.message, 6000);
    return false;
  }
  watchReload(n);
  return true;
}
// a reload in progress: the message box waits, the status says so, and the page follows it to the end
export async function watchReload(to) {
  if (S.reload) return;
  S.reload = {to: to || (S.ctxInfo && S.ctxInfo.reload && S.ctxInfo.reload.to) || null};
  S.hint = `Reloading the model${S.reload.to ? ` with a ${ctxfmt(S.reload.to)} context` : ""}…`;
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
    if (rl && rl.state === "failed") toast("error", "The context did not change", rl.error || "the engine did not start with it", 9000);
    else toast("success", "Model reloaded", `It runs with a ${ctxfmt(info.context)} context now.`, 5000);
    emit();
    return;
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
  if (n < 1024 || n > 65535) { toast("error", "The port did not change", "Use a number from 1024 to 65535.", 6000); return false; }
  const ok = await confirmDialog(`Move Maya to port ${n}?`,
    `<p>Maya listens on port <b>${esc(String(now))}</b> now. ` + (S.portInfo.restarts
      ? `Saving starts it again on the new port: the model loads again, ${esc(loadTime())}.</p>` +
        "<ul><li>This page moves to <b>" + esc(url) + "</b> when Maya is back.</li>" +
        "<li>Apps that use the API need the new port.</li>" + (S.busy ? "<li>The answer being written finishes first.</li>" : "") + "</ul>"
      : "The new port is saved and applies from the next start.</p>"), `Use port ${n}`);
  if (!ok) return false;
  try {
    const out = await api("api/port", {method: "POST", body: {port: n}});
    if (!out.restarts) { toast("success", "Port saved", `Maya uses port ${n} from its next start.`, 6000); loadPortInfo(); return true; }
  } catch (err) {
    toast("error", "The port did not change", err.message, 6000);
    return false;
  }
  S.hint = `Moving Maya to port ${n}…`;
  emit();
  toast("info", "Starting again", `Maya moves to port ${n}; this page follows when it is back.`, 8000);
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
  toast("error", "Maya is not back yet", `Open ${url} when it is.`, 30000);
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
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p});
  if (+s.temperature > 0 && +s.top_k > 0) d.top_k = +s.top_k;
  if (+s.temperature > 0 && +s.min_p > 0) d.min_p = +s.min_p;
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
  sharedTimer = setTimeout(() => saveShared(true).catch((e) => toast("error", "Not saved for other apps", e.message, 6000)), 600);
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
  const ok = await confirmDialog(`Update Project Maya to v${v}?`,
    "<p>Only Maya's code is updated: <b>the model is not downloaded again</b>, and its files, your settings and the " +
    "chats in this browser stay.</p><p>Maya finishes the answer it is writing, downloads the new code (git), compiles " +
    "only the engine files that changed (usually a few minutes; the terminal shows the progress) and loads the model " +
    "again. Apps using the API are asked to try again meanwhile.</p>", `Update to v${v}`);
  if (!ok) return;
  try {
    await api("api/update", {method: "POST", body: {}});
  } catch (e) {
    toast("error", "The update did not start", e.message, 9000);
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
      toast("success", `Updated to v${to}`, "Reloading the page…", 4000);
      setTimeout(() => location.reload(), 1200);
      return;
    }
    if (h.status === "updating") continue;
    const restarted = S.updateWatch.down;
    S.updateWatch = null;
    await loadUpdate();
    if (!restarted) toast("error", "The update did not happen", (S.update && S.update.state && S.update.state.error) || "", 9000);
    else toast("warn", "Maya started again on the same version", "The update's files did not arrive; the terminal shows why.", 9000);
    return;
  }
}

// ------------------------------------------------------------------ the troubleshooting report (GET /api/report)
export async function fetchReport() {
  const r = await fetch("api/report", {headers: headers()});
  if (!r.ok) throw new Error(r.status === 401 ? "this server needs its API key (Settings › Connect your tools)" : `HTTP ${r.status}`);
  return r.text();
}
