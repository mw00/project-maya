// serve/web/app.js - the Project Maya web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor is built on the server's own /metrics (hardware, requests) and the engine's STAT lines (expert tiers).
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K", 1048576 -> "1M" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1048576 === 0 ? `${fmt(n / 1048576)}M` : n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
// an age in seconds as people say it: "12 s", "4 min", "1 h 25 min"
function ago(s) {
  if (s < 90) return `${fmt(s)} s`;
  if (s < 5400) return `${fmt(Math.round(s / 60))} min`;
  return `${fmt(Math.floor(s / 3600))} h ${fmt(Math.round((s % 3600) / 60))} min`;
}

const store = {
  get(k, d) {
    try {
      let v = localStorage.getItem("maya." + k);
      if (v === null) v = localStorage.getItem("strata." + k);
      return v === null ? d : JSON.parse(v);
    } catch (e) { return d; }
  },
  set(k, v) { try { localStorage.setItem("maya." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ------------------------------------------------------------------ the chats' store (IndexedDB: no 5 MB cap, pictures and
// attached files kept, so a reopened chat sends exactly what it sent before)
const db = (() => {
  let opening = null;
  const open = () => opening || (opening = new Promise((ok, bad) => {
    try {
      const r = indexedDB.open("maya", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("chats", {keyPath: "id"});
      r.onsuccess = () => ok(r.result);
      r.onerror = () => bad(r.error);
    } catch (e) { bad(e); }
  }));
  const run = async (mode, fn) => {
    const d = await open();
    return new Promise((ok, bad) => {
      const t = d.transaction("chats", mode);
      const req = fn(t.objectStore("chats"));
      t.oncomplete = () => ok(req ? req.result : undefined);
      t.onerror = () => bad(t.error);
      t.onabort = () => bad(t.error);
    });
  };
  return {
    all: () => run("readonly", (s) => s.getAll()),
    put: (c) => run("readwrite", (s) => s.put(c)),
    del: (id) => run("readwrite", (s) => s.delete(id)),
    clear: () => run("readwrite", (s) => s.clear()),
  };
})();

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary st-btn--sm toast-action";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn, quiet = false) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    const was = use.getAttribute("href");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", was), 1500);
  }
  if (!quiet) toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ focus: a dialog or drawer keeps the keyboard inside it
function focusables(root) {
  return [...root.querySelectorAll('button, [href], input, select, textarea, iframe, [tabindex]:not([tabindex="-1"])')]
    .filter((e) => !e.disabled && !e.hidden && e.offsetParent !== null);
}
function trapTab(e, root) {
  const f = focusables(root);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

// ------------------------------------------------------------------ the dialog (confirmations, the HTML preview)
let modalReturn = null, modalOnClose = null;
function openModal({title, body, actions = [], wide = false, onClose = null}) {
  $("modal-title").textContent = title;
  const b = $("modal-body");
  b.innerHTML = "";
  if (typeof body === "string") b.innerHTML = body; else b.appendChild(body);
  const a = $("modal-actions");
  a.innerHTML = "";
  for (const act of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `st-btn ${act.primary ? "st-btn--primary" : act.danger ? "st-btn--danger" : "st-btn--secondary"}`;
    btn.textContent = act.label;
    btn.onclick = () => { closeModal(true); if (act.run) act.run(); };
    a.appendChild(btn);
  }
  a.hidden = !actions.length;
  $("modal-box").classList.toggle("modal__box--wide", wide);
  modalReturn = document.activeElement;
  modalOnClose = onClose;
  $("modal").hidden = false;
  (a.querySelector(".st-btn--primary") || $("modal-close")).focus();
}
function closeModal(byAction = false) {          // byAction: one of its buttons answered (no "closed" callback)
  if ($("modal").hidden) return;
  $("modal").hidden = true;
  $("modal-body").innerHTML = "";                // an iframe preview stops with it
  const f = modalOnClose;
  modalOnClose = null;
  if (f && !byAction) f();
  if (modalReturn && modalReturn.focus) modalReturn.focus();
}
$("modal-close").onclick = () => closeModal();
$("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });
function confirmDialog(title, html, okLabel, danger = false) {
  return new Promise((ok) => {
    let answered = false;
    openModal({title, body: html, onClose: () => { if (!answered) ok(false); },
               actions: [{label: "Cancel", run: () => { answered = true; ok(false); }},
                         {label: okLabel, primary: !danger, danger, run: () => { answered = true; ok(true); }}]});
  });
}

// ------------------------------------------------------------------ theme and tabs
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("maya.ui-theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = t === "dark" ? "#090a14" : "#f4f4fa";
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "dark", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("maya.ui-theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  document.body.dataset.tab = tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadUpdate();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
// the composer's hint line: a passing note (reconnecting, reloading) or the keyboard hint
let hintNote = "";
function setHint(t) { hintNote = t || ""; refreshHint(); }
function refreshHint() {
  $("composer-hint").textContent = hintNote || (busy && busy.chat !== current ? "An answer is being written in another chat" :
                                                busy || reloadWatch ? "" : "Shift+Enter: new line");
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Kept in this browser only."); };

let health = {model: "maya", images: false, max_context: 0, version: null};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    // the server's recommended settings for this model (its run config) replace the built-in Chat defaults, and
    // are what the Chat uses until its user applies settings of their own
    const d = health.defaults || {};
    if (THINKING.includes(d.reasoning_effort)) DEFAULTS.thinking = d.reasoning_effort;   // the server sends GLM's names
    if (d.temperature !== undefined) DEFAULTS.temperature = d.temperature;
    if (d.top_p !== undefined) DEFAULTS.top_p = d.top_p;
    if (d.temperature !== undefined || d.top_k !== undefined) DEFAULTS.top_k = d.top_k !== undefined ? d.top_k : 0;
    if (!store.get("sampling", null)) settings = {...DEFAULTS};
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = health.mcp_tools ? "Runs on this machine. Nothing leaves it, unless an MCP tool you turned on reaches the web."
                                                       : "Runs on this machine. Nothing leaves it.";
    renderModelLine();
    updateCtxMeter();
    if (health.status === "reloading") watchReload();
    if (health.status === "updating" && !updateWatch) loadUpdate().then((u) => u && u.state && watchUpdate(u.state.to));
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}
function renderModelLine() {
  const ctx = health.max_context ? `${ctxfmt(health.max_context)} context` : "";
  const ver = health.version ? `v${health.version}` : "";
  $("side-model").innerHTML = `<b>${esc(health.model)}</b>${esc([ctx, ver].filter(Boolean).join(" · "))}`;
  $("chat-empty-model").textContent = [health.model, ctx, ver ? `Project Maya ${ver}` : ""].filter(Boolean).join(" · ");
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "gpu", label: "GPU active", icon: "gpu", unit: "%", series: "gpu_util", max: 100,
   tip: "NVIDIA's counter: the share of time any GPU kernel is running. While Maya answers it reads 100% - between "
        + "bursts of work the GPU waits (inside a kernel) for experts from RAM and the CPU. Power shows how hard it "
        + "really works."},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "ram", label: "System RAM", icon: "memory", unit: "GB", series: "ram_used", tone: "info"},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"${m.tip ? ` title="${esc(m.tip)}"` : ""}><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    <span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>
  </div></div>`).join("");

// the GLM engine's expert tiers: where each routed expert came from while the model writes
// (speed, time per token and the VRAM hit are the Throughput card's - not repeated here)
const TIERS = [
  {key: "ram_fetch", label: "From RAM", icon: "link", unit: "/tok", digits: 2, tone: "info"},
  {key: "disk", label: "From SSD", icon: "disk", unit: "/tok", digits: 2, tone: "warn"},
  {key: "promo", label: "Promoted", icon: "bolt", unit: "/tok", digits: 2},
];
$("tiers").innerHTML = TIERS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    <span class="st-metric__value" id="tv-${m.key}">–</span>
    <span class="st-metric__sub" id="ts-${m.key}"></span>
    <svg class="st-metric__spark" id="tp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>
  </div></div>`).join("");
const TIER_SUB = {ram_fetch: "experts pulled over PCIe from RAM", disk: "experts read from the SSD", promo: "experts moved into VRAM"};
function renderTiers(t, eng) {
  const card = $("tiers-card");
  if (!t || !t.now) { card.hidden = true; return; }
  card.hidden = false;
  const n = t.now, h = t.history || {};
  const age = n.time ? Math.max(0, Date.now() / 1000 - n.time) : null;
  const stale = age != null && age > 30;               // a reading from the last answer, not live
  card.classList.toggle("is-stale", stale);
  for (const m of TIERS) {
    const v = n[m.key];
    const shown = v == null ? null : fmt(v, m.digits);
    $(`tv-${m.key}`).innerHTML = shown == null ? "–" : `${esc(shown)}<small>${esc(m.unit)}</small>`;
    $(`ts-${m.key}`).textContent = (stale ? "last answer: " : "") + (TIER_SUB[m.key] || "");
    spark(`tp-${m.key}`, (h[m.key] || []).map((x) => (x == null ? 0 : x)), m.max);
  }
  const vu = n.vram_used || 0, ru = n.ram_used || 0;
  const total = (eng && eng.experts) || Math.max(1, vu + ru);
  const disk = Math.max(0, total - vu - ru);
  $("tierbar-vram").style.width = `${(100 * vu) / total}%`;
  $("tierbar-ram").style.width = `${(100 * ru) / total}%`;
  $("tierbar-disk").style.width = `${(100 * disk) / total}%`;
  $("tier-vram-text").textContent = `${fmt(vu)} experts · ${fmt(n.vram_gb, 1)} GB`;
  $("tier-ram-text").textContent = `${fmt(ru)} experts · ${fmt(n.ram_gb, 1)} GB`;
  $("tier-disk-text").textContent = `${fmt(disk)} experts`;
  $("tiers-sub").textContent = age == null ? "" : stale ? `last reading ${ago(age)} ago, from the last answer`
                                                        : `live · ${fmt(total)} routed experts`;
  // advice from the live numbers (only while they are fresh)
  let tip = "";
  if (!stale && n.disk != null && n.disk >= 3) {
    tip = "Many experts are read from the SSD for every token: more RAM, a smaller model or a shorter context makes answers faster.";
  } else if (!stale && n.vram_hit != null && n.vram_hit < 0.85 && (n.ram_fetch || 0) >= 4) {
    tip = "Most misses come from RAM over PCIe: more VRAM (or a second GPU) keeps more experts on the card.";
  }
  $("tier-advice").hidden = !tip;
  $("tier-advice").textContent = tip;
}

// a sparkline; false when there is nothing to draw (no data, or only zeros: idle)
function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  const line = svg.querySelector(".line"), area = svg.querySelector(".area");
  if (v.length < 2 || v.every((x) => x === 0)) { line.setAttribute("d", ""); area.setAttribute("d", ""); return false; }
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  line.setAttribute("d", d);
  area.setAttribute("d", `${d}L100,32L0,32Z`);
  return true;
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      if (!document.hidden) render(lastMetrics);
      const rl = lastMetrics.reload;
      if (rl && (rl.state === "waiting" || rl.state === "running")) watchReload();
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, document.hidden ? 5000 : 1000);                // a hidden tab looks five times less often
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && lastMetrics) render(lastMetrics); });

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (reloadWatch) {
    setPill("reading", "Reloading");
  } else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading · ${pct}%` : "Reading");
  } else if (live.state === "generating") {
    setPill("generating", live.tok_s ? `${fmt(live.tok_s, 1)} tok/s` : "Writing");
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0 && !reloadWatch) setPill("queued", `${live.queued} queued`);
  liveCtx();                                     // the context meter follows the answer being written
  if (tab === "monitor") {
    renderBanners(m);
    renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
    renderTiers(m.tiers, eng);
    renderHero(live, h, last, eng, m.tiers, m.totals);
    renderGpus(hw);
  }
  if (tab === "about") renderAbout(eng, hw, st, m.storage);
}

// problems worth a line at the top of the Monitor
function renderBanners(m) {
  const out = [];
  const rl = m.reload;
  if (rl && (rl.state === "waiting" || rl.state === "running")) {
    out.push(["info", `Reloading the model with a ${ctxfmt(rl.to)} context${rl.state === "waiting" ? ", after the answer being written" : ""}. Requests wait until it is back.`]);
  } else if (rl && rl.state === "failed" && rl.ended && Date.now() / 1000 - rl.ended < 600) {
    out.push(["error", `The context change did not work: ${rl.error || "the engine did not start"}. It runs with ${ctxfmt(rl.context)} again.`]);
  }
  const gpus = (m.hardware || {}).gpus || [{index: null, temp: (m.hardware || {}).gpu_temp}];
  for (const g of gpus) {
    if (g.temp != null && g.temp >= 87) out.push(["warn", `${g.index == null ? "The GPU" : `GPU ${g.index}`} is at ${fmt(g.temp)} °C: it may slow itself down to cool off.`]);
  }
  $("banners").innerHTML = out.map(([k, t]) => `<div class="banner banner--${k}" role="${k === "error" ? "alert" : "status"}">${icon(k === "info" ? "info" : k === "error" ? "error" : "warning", "st-icon st-icon--sm")}<span>${esc(t)}</span></div>`).join("");
}

// the Monitor's hero: the speed (now, else the last request), its sparkline and four headline numbers
function renderHero(live, h, last, eng, tiers, totals) {
  const gen = live.state === "generating";
  const speed = gen ? live.tok_s : last ? last.decode_tok_s : null;
  $("hero-speed").textContent = speed == null ? "–" : fmt(speed, 1);
  $("hero-speed-sub").textContent = "decode · " + (gen ? "writing now" : live.state === "reading" ? "reading the prompt"
                                    : last ? "last request" : "waiting for a request");
  // prefill: the engine's rate over the tokens it read (not the reused prefix), live while a request runs
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last ? (last.prefill_tok_s != null ? last.prefill_tok_s
                          : last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null)
                : null;
  $("hero-prefill").textContent = prefill == null ? "–" : fmt(prefill);
  $("hero-prefill-sub").textContent = "prefill · " + (live.state === "reading" ? "reading now" : gen ? "this request"
                                      : last ? "last request" : "waiting");
  const a = spark("hero-spark", h.tok_s);
  const b = spark("hero-spark-prefill", h.prefill_tok_s_mean);
  $("hero-spark-wrap").classList.toggle("is-empty", !a && !b);
  const tn = tiers && tiers.now;
  const fresh = tn && tn.time && Date.now() / 1000 - tn.time <= 30;
  const ms = gen && tn && tn.ms_tok ? tn.ms_tok : speed ? 1000 / speed : null;
  $("hero-ms").innerHTML = ms == null ? "–" : `${fmt(ms, 1)}<small>ms</small>`;
  $("hero-hit").innerHTML = fresh && tn.vram_hit != null ? `${fmt(100 * tn.vram_hit, 1)}<small>%</small>`
                          : last && last.hit_rate != null ? `${fmt(100 * last.hit_rate, 1)}<small>%</small>` : "–";
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  $("hero-ctx-k").textContent = live.state !== "idle" ? "Context of this request" : "Context of the last request";
  $("hero-ctx").innerHTML = ctx ? `${kfmt(used)}<small>/ ${ctxfmt(ctx)}</small>` : "–";
  $("hero-req").textContent = totals && totals.requests != null ? fmt(totals.requests) : "–";
}

// five GPUs or more: one row each (up to four, the hardware cards list every GPU's own value - each figure once)
const GPU_TABLE_FROM = 5;
function renderGpus(hw) {
  const gpus = hw.gpus || [];
  $("gpus-card").hidden = gpus.length < GPU_TABLE_FROM;
  if (gpus.length < GPU_TABLE_FROM) return;
  $("gpus-sub").textContent = `${gpus.length} cards`;
  $("gpus-body").innerHTML = gpus.map((g) => {
    const gen = g.pcie_gen_max || g.pcie_gen;
    const pcie = gen ? `Gen${gen}${g.pcie_width ? ` x${g.pcie_width}` : ""}${g.pcie_rx_mb != null ? ` · ${fmt(g.pcie_rx_mb, g.pcie_rx_mb < 10 ? 1 : 0)} MB/s` : ""}` : "–";
    return `<tr><td>GPU ${esc(g.index)}</td><td class="num">${g.util == null ? "–" : `${fmt(g.util)}%`}</td>` +
      `<td class="num">${g.mem_used == null ? "–" : `${gb(g.mem_used)}${g.mem_total ? ` / ${gb(g.mem_total, 0)}` : ""} GB`}</td>` +
      `<td class="num">${g.power == null ? "–" : `${fmt(g.power)}${g.power_limit ? ` / ${fmt(g.power_limit)}` : ""} W`}</td>` +
      `<td class="num">${g.temp == null ? "–" : `${fmt(g.temp)} °C`}</td><td class="num">${esc(pcie)}</td></tr>`;
  }).join("");
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
const API_NAMES = {web: "Chat", openai: "OpenAI", anthropic: "Anthropic"};
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const prog = $("state-progress"), badge = $("state-badge");
  const took = (s) => (s == null ? "" : s < 90 ? ` · ${fmt(s)} s` : ` · ${fmt(Math.floor(s / 60))} min ${fmt(s % 60)} s`);
  let label = "Idle", state = "idle", detail = "waiting for a request", pct = 0;
  if (reloadWatch) {
    label = "Reloading";
    state = "reading";
    prog.dataset.tone = "info";
    detail = `a ${ctxfmt(reloadWatch.to)} context`;
  } else if (live.state === "reading") {
    label = "Reading";
    state = "reading";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} prompt tokens`;
    } else {
      detail = `${fmt(live.prompt_tokens)} prompt tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Writing";
    state = "generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens${took(live.elapsed_s)}`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last answer: ${fmt(last.output_tokens)} tokens${took(last.duration_s)}`;
  }
  if (live.queued > 0) label = `${label} · ${live.queued} queued`;
  badge.textContent = label;
  badge.dataset.s = state;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the hardware cards: the total (or the mean / the hottest) and its history; with two to four GPUs the line under it
  // has every GPU's own value (from five, the GPUs table above has them and the cards keep their totals)
  const gpus = hw.gpus || [];
  const multi = gpus.length > 1, each = multi && gpus.length < GPU_TABLE_FROM;
  const per = (f) => gpus.map((g) => `GPU ${g.index} ${f(g)}`.replace(/ /g, " ")).join(" · ");   // wraps between GPUs only
  const rate = (mb) => (mb == null ? "–" : mb >= 1000 ? `${fmt(mb / 1024, 1)} GB/s` : `${fmt(mb, mb < 10 ? 1 : 0)} MB/s`);
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            each ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : multi ? `mean of ${gpus.length} cards` : st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            each ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                 : (multi ? `${gpus.length} cards · ` : "") + (eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            each ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : multi ? `hottest of ${gpus.length} cards` : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W",
            each ? per((g) => (g.power == null ? "–" : `${fmt(g.power)}${g.power_limit ? `/${fmt(g.power_limit)}` : ""} W`))
                 : hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit${multi ? ` (all ${gpus.length} cards)` : ""}` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  const link = (g) => `Gen${g.pcie_gen_max || g.pcie_gen || "?"}${g.pcie_width ? ` x${g.pcie_width}` : ""}`;
  const sameLink = gpus.every((g) => link(g) === link(gpus[0]));
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            each ? per((g) => `${sameLink ? "" : `${link(g)} `}${rate(g.pcie_rx_mb)}`)
                 : hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${rate(hw.gpu_pcie_rx_mb)}` +
                   (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  for (const k of ["gpu", "vram", "temp", "power", "pcie"]) $(`ms-${k}`).classList.toggle("is-each", each);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);
  setMetric("ram", hw.ram_used == null ? null : gb(hw.ram_used), hw.ram_total ? `/ ${gb(hw.ram_total, 0)} GB` : "GB",
            eng.ram_gb ? `${fmt(+eng.ram_gb, 1)} GB pinned for experts` : "");
  spark("sp-ram", h.ram_used, hw.ram_total);

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badges = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                    disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badges[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const api = r.api ? ` <span class="chip chip--xs">${esc(API_NAMES[r.api] || r.api)}</span>` : "";
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`;
      const pre = r.prefill_tok_s != null ? r.prefill_tok_s
                : r.prompt_ms > 0 && r.prompt_tokens > (r.reused || 0) ? (r.prompt_tokens - (r.reused || 0)) / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${api}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(pre)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// the troubleshooting report: what a GitHub issue needs (GET /api/report)
async function fetchReport() {
  const r = await fetch("api/report", {headers: headers()});
  if (!r.ok) throw new Error(r.status === 401 ? "this server needs its API key (About > Settings)" : `HTTP ${r.status}`);
  return r.text();
}
$("report-copy").onclick = async () => {
  try { await copyText(await fetchReport(), null, true); toast("success", "Report copied", "Paste it into a GitHub issue - read it first."); }
  catch (e) { toast("error", "No report", e.message, 5000); }
};
$("report-show").onclick = async () => {
  const pre = $("report-pre");
  if (!pre.hidden) { pre.hidden = true; $("report-show").textContent = "Show engine log"; return; }
  try { pre.textContent = await fetchReport(); pre.hidden = false; $("report-show").textContent = "Hide"; }
  catch (e) { toast("error", "No report", e.message, 5000); }
};

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// "Tesla V100 + Tesla V100" (the server's list) -> "2 × Tesla V100"; mixed cards stay listed
function gpuNames(name, count) {
  const parts = String(name).split(" + ");
  if (parts.length > 1) return parts.every((p) => p === parts[0]) ? `${parts.length} × ${parts[0]}` : parts.join(" + ");
  return count > 1 ? `${count} × ${name}` : name;
}
let snippetsKey = "";
function renderAbout(eng, hw, st, storage) {
  const ver = eng.maya_version || health.version;
  const kvPerTok = eng.kv_gb && eng.kv_ctx ? (eng.kv_gb * 1073741824) / eng.kv_ctx : null;
  let kv;
  if (eng.engine_kind === "glm-fast") {
    const k = eng.kv === "int8" ? "8-bit (INT8)" : "16-bit";
    kv = eng.kv_resident   // KV streaming (--kv-resident): the cache's VRAM is its window, not a cost per token
      ? `${k} attention cache, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM`
      : `${k} attention cache, in VRAM${kvPerTok ? ` (about ${fmt(kvPerTok / 1024)} KB a token, ${fmt(eng.kv_gb, 1)} GB at this size)` : ""}`;
  } else {
    const k = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit", f32: "32-bit"}[eng.kv] || eng.kv;
    kv = k ? `${k}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null;
  }
  facts($("facts-engine"), [
    ["Version", ver ? `Project Maya v${ver}${eng.version ? ` · engine ${eng.version}` : ""}` : null],
    ["Model", eng.model],
    ["Engine", eng.engine_kind ? `${eng.engine_kind === "glm-fast" ? "GLM fast path" : eng.engine_kind}` : null],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens (Settings changes it)` : null],
    ["Trained for", eng.trained_context ? `up to ${fmt(eng.trained_context)} tokens` : null],
    ["KV cache", kv],
    ["Experts", eng.experts ? `${fmt(eng.experts)} routed; room for ${fmt(eng.expert_slots || 0)} in VRAM` +
                 `${eng.ram_slots ? ` and ${fmt(eng.ram_slots)} in RAM` : ""}, the rest read from the SSD` : null],
    ["Speculation", eng.mtp == null ? null : +eng.mtp ? "MTP: the model's draft block proposes the next token, the full model checks it"
                  : "off (one GPU, or no draft block in this model file)"],
    ["Images", eng.images ? "on" : "off"],
  ]);
  const gpus = hw.gpus || [];
  const pcie = gpus.length > 1 ? gpus.map((g) => `GPU ${g.index}: Gen${g.pcie_gen_max || g.pcie_gen || "?"}${g.pcie_width ? ` x${g.pcie_width}` : ""}`).join(" · ")
             : (hw.gpu_pcie_gen_max || hw.gpu_pcie_gen) ? `Gen${hw.gpu_pcie_gen_max || hw.gpu_pcie_gen}${hw.gpu_pcie_width ? ` x${hw.gpu_pcie_width}` : ""}` : null;
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${gpuNames(st.gpu_name, gpus.length)}${hw.gpu_mem_total ? ` · ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : "not readable (NVML)"],
    ["PCIe", pcie],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
    ["Model folder", storage ? `${storage.path} · ${gb(storage.free, 0)} GB free of ${gb(storage.total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
  const key = !!health.api_key;
  const k = `${base}|${eng.model}|${key}`;
  if (k !== snippetsKey) {                         // rebuilt only when it changes (keeps a selection or a copy click)
    snippetsKey = k;
    const auth = key ? "<your API key>" : "maya";
    $("snippets").innerHTML =
      `<div class="snippet"><div class="snippet__title">Claude Code</div>${codeBlock("bash",
        `ANTHROPIC_BASE_URL=${base} ANTHROPIC_AUTH_TOKEN=${auth} claude`)}</div>` +
      `<div class="snippet"><div class="snippet__title">OpenAI-compatible apps and SDKs</div>${codeBlock("bash",
        `OPENAI_BASE_URL=${base}/v1\nOPENAI_API_KEY=${key ? "<your API key>" : "any-text"}\nmodel: ${eng.model}`)}</div>` +
      `<div class="snippet"><div class="snippet__title">A first request (curl)</div>${codeBlock("bash",
        `curl ${base}/v1/chat/completions -H "Content-Type: application/json" ${key ? '-H "Authorization: Bearer <your API key>" ' : ""}\\\n  -d '{"model": "${eng.model}", "messages": [{"role": "user", "content": "Hello!"}]}'`)}</div>`;
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
// ------------------------------------------------------------------ About > Updates
// the server asks GitHub for the latest release (at most every six hours) and says whether this folder can update
// itself (a git checkout without local changes, started by maya.sh / START-MAYA.bat); else the steps by hand
let update = null, updateWatch = null;
async function loadUpdate(force = false) {
  try {
    const r = await fetch(force ? "api/update?force=1" : "api/update", {headers: headers()});
    update = r.ok ? await r.json() : null;
  } catch (e) { update = null; }
  renderUpdate();
  return update;
}
function renderUpdate() {
  const el = $("update-card"), u = update;
  $("about-dot").hidden = !(u && u.newer) || !!updateWatch;
  if (!updateWatch && (!u || !u.enabled)) { el.hidden = true; return; }
  const latest = (u && u.latest) || {};
  const notes = latest.url ? `<a class="st-btn st-btn--secondary" href="${esc(latest.url)}" target="_blank" rel="noopener">What's new</a>` : "";
  const again = (label) => `<button class="st-btn st-btn--secondary" type="button" data-update="check">${icon("refresh")}${label}</button>`;
  const when = u && u.checked ? new Date(u.checked * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "";
  const state = u && u.state;
  let tone = "", ic = "info", title, text = "", how = "", acts = "";
  if (updateWatch || (state && state.state === "updating")) {
    const to = updateWatch ? updateWatch.to : state.to;
    tone = "accent"; ic = "download";
    title = `Updating to v${to}…`;
    text = updateWatch && updateWatch.down
      ? "Maya is starting again: it compiles what changed in the engine and loads the model (a few minutes; the terminal shows the progress). This page reloads when it is back."
      : "Maya finishes the answer it is writing, then downloads the update.";
  } else if (state && state.state === "failed") {
    tone = "error"; ic = "error";
    title = "The update did not happen";
    text = state.error || "";
    acts = notes + (u.can_update ? `<button class="st-btn st-btn--primary" type="button" data-update="go">Try again</button>` : "");
  } else if (u.newer) {
    tone = "accent"; ic = "download";
    title = `Project Maya v${latest.version} is out`;
    text = latest.summary || "";
    if (!u.can_update) how = `Update by hand: ${u.by_hand}. This page can't, because ${u.blocker}.`;
    acts = notes + (u.can_update ? `<button class="st-btn st-btn--primary" type="button" data-update="go">${icon("download")}Update to v${esc(latest.version)}</button>` : "");
  } else if (u.error) {
    ic = "warning";
    title = "Couldn't check for updates";
    text = `${u.error}. Project Maya v${u.current || "?"} is running.`;
    acts = again("Try again");
  } else if (u.latest) {
    tone = "ok"; ic = "check";
    title = `Up to date: v${u.current} is the latest release`;
    text = `Checked on GitHub${when ? ` at ${when}` : ""}.`;
    acts = again("Check again");
  } else {
    title = "Checking for updates…";
  }
  el.hidden = false;
  el.dataset.tone = tone;
  el.innerHTML = `<span class="update-card__icon">${icon(ic)}</span>` +
    `<div class="update-card__text"><b>${esc(title)}</b>${text ? `<span>${esc(text)}</span>` : ""}` +
    `${how ? `<span class="update-card__how">${esc(how)}</span>` : ""}</div>` +
    (acts ? `<div class="update-card__acts">${acts}</div>` : "");
}
$("update-card").addEventListener("click", (e) => {
  const b = e.target.closest("[data-update]");
  if (!b) return;
  if (b.dataset.update === "check") { b.disabled = true; loadUpdate(true); }
  else startUpdate();
});
async function startUpdate() {
  const v = update && update.latest && update.latest.version;
  if (!v) return;
  const ok = await confirmDialog(`Update Project Maya to v${v}?`,
    "<p>Only Maya's code is updated: <b>the model is not downloaded again</b>, and its files, your settings and the " +
    "chats in this browser stay.</p><p>Maya finishes the answer it is writing, downloads the new code (git), compiles " +
    "only the engine files that changed (usually a few minutes; the terminal shows the progress) and loads the model " +
    "again. Apps using the API are asked to try again meanwhile.</p>", `Update to v${v}`);
  if (!ok) return;
  let r, d = {};
  try {
    r = await fetch("api/update", {method: "POST", headers: headers(true), body: "{}"});
    d = await r.json().catch(() => ({}));
  } catch (e) {
    toast("error", "The update did not start", "The server did not answer.", 9000);
    return;
  }
  if (!r.ok) {
    toast("error", "The update did not start", (d.error && d.error.message) || `HTTP ${r.status}`, 9000);
    loadUpdate();
    return;
  }
  watchUpdate(v);
}
// until the server answers with the new version (then the page reloads for its new files), or says it failed
async function watchUpdate(to) {
  if (updateWatch) return;
  updateWatch = {to, down: false};
  renderUpdate();
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 3000));
    let h = null;
    try {
      const r = await fetch("health", {cache: "no-store"});
      h = r.ok ? await r.json() : null;
    } catch (e) { h = null; }
    if (!h) {                                      // stopped: compiling, or loading the model
      if (!updateWatch.down) { updateWatch.down = true; renderUpdate(); }
      continue;
    }
    if (h.version === to) {
      toast("success", `Updated to v${to}`, "Reloading the page…", 4000);
      setTimeout(() => location.reload(), 1200);
      return;
    }
    if (h.status === "updating") continue;
    const restarted = updateWatch.down;
    updateWatch = null;
    await loadUpdate();
    if (!restarted) toast("error", "The update did not happen", (update && update.state && update.state.error) || "", 9000);
    else toast("warn", "Maya started again on the same version", "The update's files did not arrive; the terminal shows why.", 9000);
    return;
  }
}

$("clear-data").onclick = async () => {
  const ok = await confirmDialog("Clear this browser's chats and settings?",
    "<p>Every chat, the Chat settings, the instructions and the API key kept in this browser are removed. The model, " +
    "its files and the server's own settings are not touched.</p>", "Clear everything", true);
  if (!ok) return;
  try { await db.clear(); } catch (e) { /* nothing stored there */ }
  try {
    for (const k of Object.keys(localStorage)) if ((k.startsWith("maya.") || k.startsWith("strata.")) && k !== "maya.ui-theme") localStorage.removeItem(k);
  } catch (e) { /* ignore */ }
  location.reload();
};

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ syntax highlighting (small, local, no library)
const KW = (s) => new Set(s.split(/\s+/));
const HL_FAMILY = {};
for (const [fam, names] of Object.entries({
  c: "js javascript jsx mjs cjs ts typescript tsx java c h cpp cc cxx hpp c++ cu cuh cuda cs csharp go golang rust rs kotlin kt swift php dart scala glsl hlsl wgsl zig",
  py: "py python python3", sh: "sh bash shell zsh console ps1 powershell bat cmd", json: "json jsonc json5",
  html: "html htm xhtml xml svg vue", css: "css scss less", sql: "sql", yaml: "yaml yml toml ini cfg conf env dockerfile",
})) for (const n of names.split(" ")) HL_FAMILY[n] = fam;
const STR = /"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?/y;
const NUM = /\b(?:0x[\da-fA-F_]+|0b[01_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)[a-zA-Z]*\b/y;
const WORD = /[A-Za-z_$][\w$]*/y;
const HL = {
  c: {rules: [[/\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [/`(?:\\[\s\S]|[^`\\])*`?/y, "s"], [STR, "s"],
              [/#[ \t]*[a-z]+/y, "k"], [NUM, "n"], [WORD, "w"]],
      kw: KW(`abstract as async await break case catch class const constexpr continue debugger default defer delete do else enum
        export extends extern false final finally fn for from func function get go goto if impl implements import in inline
        instanceof interface let loop match mod module move mut namespace new nil null nullptr of operator override package
        private protected pub public readonly ref register return self Self sizeof static struct super switch template this
        throw throws trait true try type typedef typename typeof union unsafe use using var virtual void volatile where while
        with yield auto bool char double float int long short signed unsigned void string number boolean any never unknown
        __global__ __device__ __host__ __shared__ __restrict__ i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 usize isize`)},
  py: {rules: [[/#[^\n]*/y, "c"], [/[rbfuRBFU]{0,2}(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$))/y, "s"],
               [/[rbfuRBFU]{0,2}(?:"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?)/y, "s"], [/@[\w.]+/y, "f"], [NUM, "n"], [WORD, "w"]],
       kw: KW(`and as assert async await break case class continue def del elif else except False finally for from global if
         import in is lambda match None nonlocal not or pass raise return True try while with yield self cls print len range`)},
  sh: {rules: [[/(?:^|(?<=\s))#[^\n]*/y, "c"], [STR, "s"], [/\$\{[^}\n]*\}?|\$[\w@#?*!$-]+/y, "v"], [NUM, "n"], [/[A-Za-z_][\w.-]*/y, "w"]],
       kw: KW(`if then else elif fi for while until do done case esac in function return export local set unset echo printf cd
         sudo source alias exit read shift trap eval exec`)},
  json: {rules: [[/"(?:\\[\s\S]|[^"\\\n])*"(?=\s*:)/y, "a"], [STR, "s"], [/-?\d+\.?\d*(?:[eE][+-]?\d+)?/y, "n"], [/[a-z]+/y, "w"]],
         kw: KW("true false null")},
  html: {rules: [[/<!--[\s\S]*?(?:-->|$)/y, "c"], [/<\/?[A-Za-z][\w:-]*/y, "k"], [/[A-Za-z_:][\w:.-]*(?==)/y, "a"], [STR, "s"],
                 [/&#?\w+;/y, "n"]], kw: KW("")},
  css: {rules: [[/\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [STR, "s"], [/#[\da-fA-F]{3,8}\b/y, "n"], [/@[\w-]+/y, "k"],
                [/-?\d+\.?\d*(?:px|em|rem|%|vh|vw|s|ms|deg|fr)?/y, "n"], [/[\w-]+(?=\s*:[^:{;]*[;}])/y, "a"]], kw: KW("")},
  sql: {rules: [[/--[^\n]*|\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [STR, "s"], [NUM, "n"], [WORD, "w"]],
        kw: KW(`select from where and or not insert into values update set delete create table index view drop alter add join
          left right inner outer on group by order having limit offset as distinct union all null is in like between case
          when then else end primary key foreign references default exists with returning asc desc count sum avg min max`),
        ci: true},
  yaml: {rules: [[/#[^\n]*/y, "c"], [/[\w.-]+(?=[ \t]*[:=])/y, "a"], [STR, "s"], [NUM, "n"], [/[a-z]+/y, "w"]],
         kw: KW("true false null yes no on off")},
};
function highlight(code, lang) {
  const fam = HL_FAMILY[(lang || "").toLowerCase()];
  if (!fam || code.length > 200000) return esc(code);
  const R = HL[fam];
  let out = "", plain = "", i = 0;
  while (i < code.length) {
    let tok = null, cls = null;
    for (const [re, c] of R.rules) {
      re.lastIndex = i;
      const m = re.exec(code);
      if (m && m[0].length) { tok = m[0]; cls = c; break; }
    }
    if (tok === null) { plain += code[i++]; continue; }
    if (cls === "w") {
      const w = R.ci ? tok.toLowerCase() : tok;
      cls = R.kw.has(w) ? "k" : code[i + tok.length] === "(" ? "f" : null;
    }
    if (plain) { out += esc(plain); plain = ""; }
    out += cls ? `<span class="tk-${cls}">${esc(tok)}</span>` : esc(tok);
    i += tok.length;
  }
  return out + esc(plain);
}

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
const CODE_EXT = {python: "py", py: "py", javascript: "js", js: "js", jsx: "jsx", typescript: "ts", ts: "ts", tsx: "tsx",
  html: "html", htm: "html", css: "css", scss: "scss", json: "json", bash: "sh", sh: "sh", shell: "sh", zsh: "sh",
  c: "c", h: "h", cpp: "cpp", "c++": "cpp", hpp: "hpp", cuda: "cu", cu: "cu", rust: "rs", rs: "rs", go: "go", java: "java",
  kotlin: "kt", swift: "swift", sql: "sql", yaml: "yaml", yml: "yml", toml: "toml", xml: "xml", svg: "svg", markdown: "md",
  md: "md", ps1: "ps1", powershell: "ps1", php: "php", ruby: "rb", rb: "rb", lua: "lua", dockerfile: "Dockerfile"};
const isHtmlCode = (lang, code) => ["html", "htm", "xhtml"].includes((lang || "").toLowerCase()) ||
                                   (!lang && /^\s*(<!doctype html|<html[\s>])/i.test(code));
function codeBlock(lang, code, complete = true) {
  const l = (lang || "").trim().split(/\s+/)[0] || "";
  const btn = (attr, name, title) => `<button class="st-btn st-btn--icon" ${attr} aria-label="${title}" title="${title}">${icon(name)}</button>`;
  const actions = (complete && isHtmlCode(l, code) ? btn("data-code-preview", "eye", "Preview this page") : "") +
    btn("data-code-wrap", "wrap", "Wrap long lines") + btn("data-code-download", "download", "Download as a file") +
    btn("data-code-copy", "copy", "Copy");
  return `<div class="st-code" data-lang="${esc(l)}"><div class="st-code__head"><span>${esc(l || "code")}</span>` +
    `<span class="st-code__actions">${actions}</span></div><pre><code>${highlight(code, l)}</code></pre></div>`;
}
// a list, nested by indentation: [{indent, ordered, start, text}]
function renderList(items) {
  let pos = 0;
  const build = (indent) => {
    const first = items[pos];
    const tag = first.ordered ? "ol" : "ul";
    let html = `<${tag}${first.ordered && first.start && first.start !== 1 ? ` start="${first.start}"` : ""}>`;
    while (pos < items.length && items[pos].indent >= indent) {
      const it = items[pos];
      if (it.indent > indent) {                     // a deeper level: inside the item before it
        const sub = build(it.indent);
        html = html.endsWith("</li>") ? `${html.slice(0, -5)}${sub}</li>` : `${html}<li>${sub}</li>`;
        continue;
      }
      html += `<li>${inline(it.text)}</li>`;
      pos++;
    }
    return `${html}</${tag}>`;
  };
  let out = "";
  while (pos < items.length) out += build(items[pos].indent);
  return out;
}
const LIST_RE = /^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$/;
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [];
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<div class="table-scroll"><table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table></div>");
      continue;
    }
    if (LIST_RE.test(l)) {
      flushPara();
      const items = [];
      while (i < lines.length) {
        const L = lines[i], mm = L.match(LIST_RE);
        if (mm) { items.push({indent: mm[1].replace(/\t/g, "    ").length, ordered: !mm[2], start: mm[3] ? +mm[3] : null, text: mm[4]}); i++; continue; }
        if (L.trim() && /^\s{2,}\S/.test(L) && items.length) { items[items.length - 1].text += " " + L.trim(); i++; continue; }
        break;
      }
      i--;
      out.push(renderList(items));
      continue;
    }
    para.push(l);
  }
  flushPara();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest, false); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}
// While a long answer streams, the part before its last blank line (outside a code block) is rendered once and kept;
// only the tail is rendered again each frame (a 14K-token answer stays smooth).
function streamingMarkdown(m) {
  const t = m.text;
  if (t.length < 6000) return markdown(t);
  const c = m._md || (Object.defineProperty(m, "_md", {value: {upto: 0, html: ""}, enumerable: false, writable: true}), m._md);
  let cut = t.lastIndexOf("\n\n", t.length - 3);
  while (cut > c.upto && ((t.slice(0, cut).match(/(^|\n)```/g) || []).length % 2)) cut = t.lastIndexOf("\n\n", cut - 1);
  if (cut > c.upto) { c.html += markdown(t.slice(c.upto, cut)); c.upto = cut; }
  return c.html + markdown(t.slice(c.upto));
}

// the code blocks' buttons (chat answers and About's snippets)
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-code-copy], [data-code-wrap], [data-code-download], [data-code-preview]");
  if (!b) return;
  const block = b.closest(".st-code"), code = block.querySelector("pre").textContent, lang = block.dataset.lang || "";
  if (b.hasAttribute("data-code-copy")) copyText(code, b);
  else if (b.hasAttribute("data-code-wrap")) block.classList.toggle("is-wrapped");
  else if (b.hasAttribute("data-code-download")) {
    const ext = CODE_EXT[lang.toLowerCase()] || (isHtmlCode(lang, code) ? "html" : "txt");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([code], {type: "text/plain"}));
    a.download = ext === "Dockerfile" ? ext : `maya-code.${ext}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } else if (b.hasAttribute("data-code-preview")) previewHtml(code);
});
// an HTML answer in a sandboxed frame: scripts run, but cut off from this page (no same-origin, no storage)
function previewHtml(code) {
  const wrap = document.createElement("div");
  wrap.className = "preview";
  wrap.innerHTML = `<p class="muted small">Runs in a sandbox, cut off from this page. Scripts it loads from the web (a
    three.js CDN, say) come from the internet.</p>`;
  const f = document.createElement("iframe");
  f.className = "preview__frame";
  f.setAttribute("sandbox", "allow-scripts allow-pointer-lock allow-modals");
  f.setAttribute("title", "Preview");
  f.srcdoc = code;
  wrap.appendChild(f);
  openModal({title: "Preview", body: wrap, wide: true});
}

// ------------------------------------------------------------------ Chats (many, kept in this browser)
// thinking: GLM's own levels - none (Off), low, high, max.  The page before v1.0.18 saved Qwen-era names, where its
// "Medium" was GLM's High and its "High" GLM's Max: they are translated once (thinking_v 2)
const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, max: "", seed: "", show: true,
                  mcp: true, system: "", sys_default: false, thinking_v: 2};
const THINKING = ["none", "low", "high", "max"];
const glmLevel = (v, v2) => (v2 ? v : {medium: "high", high: "max", xhigh: "max"}[v] || v);
let settings = {...DEFAULTS, ...store.get("sampling", {})};
if (settings.thinking_v !== 2) { settings.thinking = glmLevel(settings.thinking, false); settings.thinking_v = 2; }
if (!THINKING.includes(settings.thinking)) settings.thinking = "high";
let chats = [];                       // {id, title, created, updated, messages, system}, newest first
let current = null;                   // the open chat
let messages = [];                    // current.messages
let attachments = [];                 // {kind: "image" | "file", name, url | text}
let busy = null;                      // {controller, msg, chat}
let editIndex = null;                 // the user message being edited (its answers are replaced on send)
let dbBroken = false;                 // IndexedDB unavailable (some private windows): localStorage, without pictures

const titleOf = (msgs) => {
  const u = msgs.find((m) => m.role === "user" && (m.text || (m.files || []).length));
  const t = u ? (u.text || (u.files || []).map((f) => f.name).join(", ")) : "";
  return t.replace(/\s+/g, " ").trim().slice(0, 60) || "New chat";
};
function newChatObj() {
  return {id: uid(), title: "", created: Date.now(), updated: Date.now(), messages: [],
          system: settings.sys_default ? settings.system || "" : ""};
}
const slim = (c) => ({...c, messages: c.messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name}))}))});
async function persist(c = current) {
  if (!c || !c.messages.length) return;
  if (!chats.includes(c)) chats.unshift(c);
  const plain = JSON.parse(JSON.stringify(c));    // no transient fields (the streaming cache is not enumerable)
  try {
    if (dbBroken) throw new Error("no IndexedDB");
    await db.put(plain);
  } catch (e) {
    dbBroken = true;
    try { localStorage.setItem("maya.chats-fallback", JSON.stringify(chats.slice(0, 30).map(slim))); }
    catch (e2) { /* full: the newest chats only next time */ }
  }
}
function saveChat(c = current) {
  if (!c) return;
  c.updated = Date.now();
  if (!c.title || c.title === "New chat") c.title = titleOf(c.messages);
  chats.sort((a, b) => b.updated - a.updated);
  persist(c);
  renderChatList();
}
async function loadChats() {
  try { chats = await db.all(); }
  catch (e) { dbBroken = true; chats = store.get("chats-fallback", []); }
  if (!chats.length) {                            // the single chat of earlier versions becomes the first one
    const old = store.get("chat", []);
    if (old.length) {
      const c = newChatObj();
      c.messages = old;
      c.title = titleOf(old);
      chats.push(c);
      await persist(c);
    }
  }
  try { localStorage.removeItem("maya.chat"); localStorage.removeItem("strata.chat"); } catch (e) { /* ignore */ }
  chats.sort((a, b) => b.updated - a.updated);
  const want = store.get("current-chat", null);
  current = chats.find((c) => c.id === want) || chats[0] || newChatObj();
  messages = current.messages;
}
// an answer being written goes on in its own chat while another one is open (and shows again when its chat is)
function openChat(c) {
  current = c;
  messages = c.messages;
  store.set("current-chat", c.id);
  cancelEdit();
  renderChat();
  renderChatList();
  refreshHint();
  showChats(false);
  if (tab !== "chat") showTab("chat");
  resumePending();
}
function newChat() {
  if (!messages.length && current.system === (settings.sys_default ? settings.system || "" : "")) {
    showChats(false); showTab("chat"); $("input").focus(); return;
  }
  openChat(newChatObj());
  $("input").focus();
}
async function deleteChat(c) {
  if (busy && busy.chat === c) { toast("warn", "Still writing", "Stop the answer first."); return; }
  chats = chats.filter((x) => x !== c);
  try { if (!dbBroken) await db.del(c.id); } catch (e) { /* ignore */ }
  if (dbBroken) persist(chats[0]);
  if (current === c) { current = chats[0] || newChatObj(); messages = current.messages; renderChat(); }
  renderChatList();
  toast("info", "Chat deleted", c.title || "", 6000, {label: "Undo", run: () => { chats.push(c); saveChat(c); }});
}
function relTime(t) {
  const d = new Date(t), now = new Date();
  if (d.toDateString() === now.toDateString()) return d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], {month: "short", day: "numeric"});
}
function renderChatList() {
  const q = $("chats-search").value.trim().toLowerCase();
  const list = $("chats-list");
  let items = chats;
  if (!chats.includes(current)) items = [current, ...chats];            // the new, still empty chat
  if (q) items = items.filter((c) => (c.title || "").toLowerCase().includes(q) ||
                                     c.messages.some((m) => (m.text || "").toLowerCase().includes(q)));
  list.innerHTML = items.length ? "" : `<p class="muted small chats__empty">${q ? "No chat matches." : "No chats yet."}</p>`;
  for (const c of items) {
    const row = document.createElement("div");
    row.className = "chat-item" + (c === current ? " is-active" : "");
    row.setAttribute("role", "listitem");
    row.innerHTML = `<button type="button" class="chat-item__open"><span class="chat-item__title"></span>` +
      `<span class="chat-item__time muted"></span></button>` +
      `<span class="chat-item__actions"><button type="button" class="st-btn st-btn--icon" data-act="rename" aria-label="Rename" title="Rename">${icon("edit")}</button>` +
      `<button type="button" class="st-btn st-btn--icon" data-act="delete" aria-label="Delete" title="Delete">${icon("trash")}</button></span>`;
    row.querySelector(".chat-item__title").textContent = c.title || (c.messages.length ? titleOf(c.messages) : "New chat");
    row.querySelector(".chat-item__time").textContent = c.messages.length ? relTime(c.updated) : "";
    row.querySelector(".chat-item__open").onclick = () => { if (c !== current) openChat(c); else showChats(false); };
    row.querySelector('[data-act="delete"]').onclick = () => deleteChat(c);
    row.querySelector('[data-act="rename"]').onclick = () => renameChat(row, c);
    if (!c.messages.length) row.querySelector(".chat-item__actions").hidden = true;
    list.appendChild(row);
  }
}
function renameChat(row, c) {
  const t = row.querySelector(".chat-item__title");
  const input = document.createElement("input");
  input.className = "st-input chat-item__rename";
  input.value = c.title || titleOf(c.messages);
  t.replaceWith(input);
  input.focus();
  input.select();
  const done = (save) => {
    if (save && input.value.trim()) { c.title = input.value.trim().slice(0, 80); persist(c); }
    renderChatList();
  };
  input.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); done(true); } if (e.key === "Escape") { e.stopPropagation(); done(false); } };
  input.onblur = () => done(true);
  input.onclick = (e) => e.stopPropagation();
}
$("chats-search").addEventListener("input", renderChatList);
$("chats-new").onclick = newChat;
// narrow screens: the chats slide in from the left
function showChats(open) {
  document.body.classList.toggle("chats-open", open);
  $("chats-scrim").hidden = !open;
  $("chats-toggle").setAttribute("aria-expanded", String(open));
}
$("chats-toggle").onclick = () => showChats(!document.body.classList.contains("chats-open"));
$("chats-scrim").onclick = () => showChats(false);

// ------------------------------------------------------------------ Chat
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.innerHTML = `<span></span><button class="st-btn st-btn--icon" data-msg-edit aria-label="Edit and send again" title="Edit">${icon("edit")}</button>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy" title="Copy">${icon("copy")}</button>`;
    meta.firstChild.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button>` +
      `<button class="st-btn st-btn--icon" data-msg-regen aria-label="Write this answer again" title="Regenerate">${icon("refresh")}</button></div>`;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m, streaming) {
  if (!m.tools || !m.tools.length) return streaming ? streamingMarkdown(m) : markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
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
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m, streaming);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
  // regenerate: the last answer only, when nothing is running
  el.querySelector("[data-msg-regen]").hidden = streaming || !!busy || +el.dataset.i !== messages.length - 1;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
  updateCtxMeter();
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; updateJump(); }
function updateJump() { $("jump-latest").hidden = nearBottom() || !messages.length; }
$("chat-scroll").addEventListener("scroll", updateJump, {passive: true});
$("jump-latest").onclick = () => scrollDown(true);

$("chat").addEventListener("click", (e) => {
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  const me = e.target.closest("[data-msg-edit]");
  if (me) { editMessage(+me.closest(".st-msg").dataset.i); return; }
  const mr = e.target.closest("[data-msg-regen]");
  if (mr) { regenerate(); return; }
  const sg = e.target.closest(".suggestion");
  if (sg) { $("input").value = sg.dataset.q; autosize(); updateCtxMeter(); $("input").focus(); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

// the chat as the API sends it; `light`: pictures as placeholders (the context meter's count needs only where they are)
function apiMessages(msgs = messages, light = false) {
  const out = [];
  if (current.system && current.system.trim()) out.push({role: "system", content: current.system.trim()});
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
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  if (!ran.length) return m.text ? [{role: "assistant", content: m.text}] : [];
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

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on || !!reloadWatch;
  refreshHint();
}

// the context meter: how much of the model's context this chat takes - while an answer is written, the request's own
// prompt tokens plus the tokens written so far (the server's live count); otherwise the chat with what is being typed,
// counted by the server with the model's own template and tokenizer (POST /api/tokens).  "≈" only when a count can't be
// exact (pictures, or an older server: then a length estimate).
const tokfmt = (n) => (n == null ? "–" : n < 1000 ? fmt(n) : `${fmt(n / 1024, n < 10240 ? 1 : 0)}K`);   // as 128K = 131,072
let ctxShown = {key: null, tokens: null, approx: false};
let ctxTimer = null, ctxSeq = 0;
function draftChat() {
  const base = editIndex != null ? messages.slice(0, editIndex) : messages;
  const text = $("input").value.trim();
  if (!text && !attachments.length) return base;
  return [...base, {role: "user", text, images: attachments.filter((a) => a.kind === "image"),
                    files: attachments.filter((a) => a.kind === "file")}];
}
function estimateTokens(msgs) {              // an older server without /api/tokens: about 3.5 characters a token
  let chars = (current.system || "").length;
  for (const m of msgs) chars += m.role === "user" ? userText(m).length : (m.text || "").length;
  return Math.round(chars / 3.5) + 8 * msgs.length;
}
function showCtx(tokens, approx, why) {
  const max = health.max_context, meter = $("ctx-meter");
  if (!max || tokens == null) { meter.hidden = true; return; }
  const pct = Math.min(100, (100 * tokens) / max);
  meter.hidden = false;
  $("ctx-ring").setAttribute("stroke-dasharray", `${Math.max(pct, 1.5).toFixed(1)} 100`);
  meter.dataset.level = pct >= 95 ? "danger" : pct >= 80 ? "warn" : "";
  $("ctx-text").textContent = `${approx ? "≈" : ""}${tokfmt(tokens)} / ${ctxfmt(max)}`;
  meter.title = `${why}: ${fmt(tokens)} of the ${fmt(max)}-token context (${fmt(pct, pct < 10 ? 1 : 0)}%)` +
    (approx ? " - approximate (pictures count once they are read)" : "") +
    (pct >= 95 ? ". Nearly full: start a new chat, or raise the context size in Settings." : "");
}
// while an answer is written in the open chat: the server's live count (every poll)
function liveCtx() {
  const live = lastMetrics && lastMetrics.live;
  if (!busy || busy.chat !== current || !live || live.state === "idle" || live.prompt_tokens == null) return false;
  showCtx(live.prompt_tokens + (live.generated || 0), false, "This request");
  return true;
}
function updateCtxMeter() {
  clearTimeout(ctxTimer);
  if (liveCtx()) return;
  const msgs = draftChat();
  if (!msgs.length) { ctxShown = {key: null, tokens: null, approx: false}; $("ctx-meter").hidden = true; return; }
  if (ctxShown.tokens != null) showCtx(ctxShown.tokens, ctxShown.approx, "This chat");   // the last count meanwhile
  ctxTimer = setTimeout(async () => {
    const body = {messages: apiMessages(msgs, true), reasoning_effort: settings.thinking};
    if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;
    const key = JSON.stringify(body);
    if (key === ctxShown.key && ctxShown.tokens != null) return;
    const seq = ++ctxSeq;
    let tokens = null, approx = false;
    try {
      const r = await fetch("api/tokens", {method: "POST", headers: headers(true), body: key});
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      tokens = j.tokens;
      approx = !!j.approximate;
    } catch (e) {
      tokens = estimateTokens(msgs);
      approx = true;
    }
    if (seq !== ctxSeq || liveCtx()) return;      // typed on, or an answer started meanwhile
    ctxShown = {key, tokens, approx};
    showCtx(tokens, approx, "This chat");
  }, 400);
}

function answerMeta(m, n, secs) {
  const parts = [];
  if (n) {
    const t = m.timings || {};
    const rate = t.predicted_per_second || (secs > 0.25 ? n / secs : null);
    parts.push(`${fmt(n)} tokens${rate ? ` · ${fmt(rate, 1)} tok/s` : ""}`);
    if (t.prompt_n != null && t.prompt_ms) {
      parts.push(`read ${fmt(t.prompt_n)} new prompt tokens in ${fmt(t.prompt_ms / 1000, 1)} s` +
                 (t.cache_n ? ` (${fmt(t.cache_n)} reused)` : ""));
    }
  }
  if (m.stopped) parts.push("stopped");
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) parts.push(`${ran} tool call${ran > 1 ? "s" : ""}`);
  if (m.limit) parts.push(`stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`);
  return parts.join(" · ") || (m.stopped ? "Stopped" : "");
}

const sleepOrVisible = (ms) => new Promise((ok) => {
  const go = () => { document.removeEventListener("visibilitychange", go); clearTimeout(tm); ok(); };
  const tm = setTimeout(go, ms);
  document.addEventListener("visibilitychange", go);
});

// One answer, streamed: a new one (POST), or - after a page reload - the one still running on the server (resumeJob)
async function answer(resumeJob = null) {
  let m;
  if (resumeJob) {
    m = messages[messages.length - 1];
    Object.assign(m, {text: "", reasoning: "", tools: [], error: undefined, meta: undefined, stopped: undefined});
  } else {
    m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
    messages.push(m);
  }
  const chat = current;
  renderChat();
  // the answer on the screen: none while another chat is open (renderChat draws it anew when its chat is opened again)
  const view = () => (chat === current ? $("chat").querySelector(`.st-msg[data-i="${chat.messages.indexOf(m)}"]`) : null);
  const controller = new AbortController();
  busy = {controller, msg: m, chat};
  setBusy(true);

  let body = null;
  if (!resumeJob) {
    body = {model: health.model, messages: apiMessages(), stream: true, reasoning_effort: settings.thinking};
    if (settings.temperature > 0) {
      Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p});
      if (+settings.top_k > 0) body.top_k = +settings.top_k;   // 0: no top-k cut
      if (+settings.min_p > 0) body.min_p = +settings.min_p;
    } else {
      body.temperature = 0;
    }
    if (settings.seed) body.seed = +settings.seed;
    if (settings.max) body.max_tokens = +settings.max;
    if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it
    // the server keeps the answer running if this connection drops (a phone's screen sleeping mid-answer, or a reload):
    // the page reconnects to it by id and carries on from the last chunk it read (strata_resume)
    body.strata_resume = true;
  }
  let firstAt = null, thinkStart = null, usage = null, pending = false, lastPaint = 0, jobId = resumeJob, nRecv = 0, finished = false;
  let ended = false;                               // a repaint queued while streaming must not land after the end
  const paint = () => {
    pending = false;
    const el = ended ? null : view();
    if (!el) return;
    lastPaint = performance.now();
    updateAssistant(el, m, true);
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
          busy.job = jobId;
          m.job = jobId;
          m.pending = true;
          saveChat(chat);                                     // a reload can take this answer back up
        }
        if (j.error) { finished = true; throw new Error(j.error.message || "the engine reported an error"); }
        if (j.usage) usage = j.usage;
        if (j.timings) m.timings = j.timings;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
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
    if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
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
  setHint("");
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
  busy = null;
  setBusy(false);
  if (chat === current) {
    const el = view();
    if (el) updateAssistant(el, m, false);
    scrollDown();
    updateCtxMeter();
  } else {                                         // another chat is open: its last answer can be written again now
    const last = $("chat").lastElementChild;
    if (last && last.classList.contains("st-msg--assistant")) updateAssistant(last, messages[+last.dataset.i], false);
  }
  saveChat(chat);
  resumePending();                                 // the open chat's answer cut off by a reload, now that none runs
}
// after a reload: an answer the page was streaming when it went away is taken back up from the server
function resumePending() {
  const m = messages[messages.length - 1];
  if (!busy && m && m.role === "assistant" && m.pending && m.job) answer(m.job);
}

async function send() {
  const text = $("input").value.trim();
  if (reloadWatch) { toast("info", "The model is reloading", "Send it once the new context size is loaded."); return; }
  if ((!text && !attachments.length) || busy) return;
  if (editIndex != null) messages.splice(editIndex);       // the edited message and every answer after it
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  cancelEdit(false);
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  saveChat();
  await answer();
}
function regenerate() {
  if (busy || reloadWatch) return;
  while (messages.length && messages[messages.length - 1].role === "assistant") messages.pop();
  if (!messages.length) return;
  answer();
}
function editMessage(i) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const m = messages[i];
  editIndex = i;
  $("input").value = m.text || "";
  attachments = [...(m.images || []).filter((a) => a.url).map((a) => ({...a, kind: "image"})),
                 ...(m.files || []).filter((f) => f.text != null).map((f) => ({...f, kind: "file"}))];
  renderAttachments();
  $("edit-banner").hidden = false;
  autosize();
  $("input").focus();
  updateCtxMeter();
}
function cancelEdit(clear = true) {
  if (editIndex == null) return;
  editIndex = null;
  $("edit-banner").hidden = true;
  if (clear) { $("input").value = ""; attachments = []; renderAttachments(); autosize(); }
}
$("edit-cancel").onclick = () => cancelEdit();

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
function stopAnswer() {
  if (!busy) return;
  // the server runs a dropped answer on, so Stop says so explicitly (then the stream ends with it)
  if (busy.job) fetch("v1/strata/cancel", {method: "POST", headers: headers(true), body: JSON.stringify({id: busy.job})})
    .catch(() => {});
  busy.controller.abort();
}
$("stop-btn").onclick = stopAnswer;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !matchMedia("(pointer: coarse)").matches) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", () => { autosize(); updateCtxMeter(); });

$("new-btn").onclick = newChat;
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const head = `# ${current.title || titleOf(messages)}\n\n` + (current.system ? `> Instructions: ${current.system.replace(/\n/g, " ")}\n\n` : "");
  const md = head + messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `maya-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); updateCtxMeter(); };
    c.appendChild(x);
    box.appendChild(c);
  });
  updateCtxMeter();
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the context size (a reload of the model)
const CTX_STOPS = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288, 786432, 1048576];
let ctxInfo = null;                   // GET /api/context: the range, the current size, its measured cost, a reload's state
let ctxStops = [];
let reloadWatch = null;               // {to}: a reload in progress (the page waits for it)
async function loadCtxInfo() {
  try {
    const r = await fetch("api/context", {headers: headers()});
    ctxInfo = r.ok ? await r.json() : null;
  } catch (e) { ctxInfo = null; }
  if (ctxInfo) {
    ctxStops = CTX_STOPS.filter((v) => v >= ctxInfo.min && v <= ctxInfo.max);
    if (!ctxStops.includes(ctxInfo.context)) ctxStops = [...ctxStops, ctxInfo.context].sort((a, b) => a - b);
  }
  return ctxInfo;
}
// what a context size costs here: the engine's own measurement of the current size, scaled
function ctxEstimate(n) {
  // (a streamed cache - KV streaming - keeps a fixed window in VRAM: scaling it by the context would mislead)
  if (!ctxInfo || !ctxInfo.kv_gb || !ctxInfo.kv_ctx || ctxInfo.kv_resident) return null;
  const perTok = ctxInfo.kv_gb / ctxInfo.kv_ctx;
  const est = perTok * n, now = perTok * ctxInfo.context;
  const expertGb = ctxInfo.vram_gb && ctxInfo.vram_slots ? +ctxInfo.vram_gb / ctxInfo.vram_slots : null;
  return {gb: est, delta: est - now, experts: expertGb ? Math.round((est - now) / expertGb) : null, vram: +ctxInfo.vram_gb || null};
}
function renderCtxField(n) {
  const field = $("ctx-field");
  field.hidden = !ctxInfo;
  if (!ctxInfo) return;
  const v = n || ctxInfo.context;
  $("s-ctx").max = String(ctxStops.length - 1);
  $("s-ctx").value = String(Math.max(0, ctxStops.indexOf(v)));
  $("s-ctx").disabled = !!reloadWatch;
  outputsCtx();
}
function ctxValue() { return ctxStops[+$("s-ctx").value] || (ctxInfo && ctxInfo.context); }
function outputsCtx() {
  if (!ctxInfo) return;
  const v = ctxValue(), now = ctxInfo.context;
  $("o-ctx").textContent = `${ctxfmt(v)} tokens${v === now ? " · now" : ""}`;
  const e = ctxEstimate(v);
  let text = "How much text the model holds at once: the chat, files, tool output.";
  if (reloadWatch) text = `Reloading with a ${ctxfmt(reloadWatch.to)} context…`;
  else if (e) {
    text = `≈ ${fmt(e.gb, 1)} GB of GPU memory for the context` + (v === now ? "." :
      e.experts ? ` · about ${fmt(Math.abs(e.experts))} ${e.experts > 0 ? "fewer" : "more"} experts held in VRAM than now` +
                  `${e.experts > 0 ? " (answers a little slower)" : ""}.` : ".");
  }
  $("ctx-est").textContent = text;
  const warns = [];
  if (v > 262144) warns.push("Beyond 256K is untested on this engine.");
  if (e && e.vram && e.gb > e.vram * 0.6) warns.push("That leaves little GPU memory for experts: answers get much slower, and it may not start (the current size then comes back).");
  if (v !== now) warns.push("Saving reloads the model (1-3 minutes).");
  $("ctx-warn").hidden = !warns.length;
  $("ctx-warn").textContent = warns.join(" ");
}
$("s-ctx").oninput = outputsCtx;
async function confirmContext(n) {
  const e = ctxEstimate(n);
  const tokens = ctxShown.tokens != null ? ctxShown.tokens : estimateTokens(messages);
  const parts = [`<p>The engine restarts with a <b>${esc(ctxfmt(n))}</b>-token context (now ${esc(ctxfmt(ctxInfo.context))}). ` +
                 "That takes about 1-3 minutes.</p>",
                 "<ul><li>Your chat stays here; the next answer reads it again.</li>" +
                 "<li>Requests from other apps get a “try again” until the model is back.</li>" +
                 "<li>If the new size does not fit, the current one comes back by itself.</li>" +
                 (busy ? "<li>The answer being written finishes first.</li>" : "") + "</ul>"];
  if (e) parts.push(`<p class="muted">Its cost here: about ${esc(fmt(e.gb, 1))} GB of GPU memory` +
                    (e.experts ? `, ${esc(fmt(Math.abs(e.experts)))} ${e.experts > 0 ? "fewer" : "more"} experts in VRAM than now.` : ".") + "</p>");
  if (tokens > n) parts.push(`<p class="ctx-warn">This chat holds about ${esc(fmt(tokens))} tokens: more than the new size, so it will not fit. Start a new chat after the reload, or keep a larger context.</p>`);
  const ok = await confirmDialog(`Reload the model with a ${ctxfmt(n)} context?`, parts.join(""), `Reload with ${ctxfmt(n)}`);
  if (!ok) return false;
  try {
    const r = await fetch("api/context", {method: "POST", headers: headers(true), body: JSON.stringify({max_context: n})});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (err) { /* not json */ }
      throw new Error(msg);
    }
  } catch (err) {
    toast("error", "The context did not change", err.message, 6000);
    return false;
  }
  watchReload(n);
  return true;
}
// a reload in progress: the composer waits, the pill says so, and the page follows it to the end
async function watchReload(to) {
  if (reloadWatch) return;
  reloadWatch = {to: to || (ctxInfo && ctxInfo.reload && ctxInfo.reload.to) || null};
  setBusy(!!busy);
  setHint(`Reloading the model${reloadWatch.to ? ` with a ${ctxfmt(reloadWatch.to)} context` : ""}…`);
  if (lastMetrics) render(lastMetrics);
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 2000));
    const info = await loadCtxInfo();
    const rl = info && info.reload;
    if (!info) continue;                           // the server is busy restarting: ask again
    if (!reloadWatch.to && rl) reloadWatch.to = rl.to;
    if (rl && (rl.state === "waiting" || rl.state === "running")) { if (lastMetrics) render(lastMetrics); continue; }
    reloadWatch = null;
    setHint("");
    setBusy(!!busy);
    await loadHealth();
    if (rl && rl.state === "failed") toast("error", "The context did not change", rl.error || "the engine did not start with it", 9000);
    else toast("success", "Model reloaded", `It runs with a ${ctxfmt(info.context)} context now.`, 5000);
    if ($("drawer").dataset.open === "true") renderCtxField();
    if (lastMetrics) render(lastMetrics);
    return;
  }
}

// ------------------------------------------------------------------ the settings drawer
let drawerReturn = null, drawerSnapshot = null;
function formState() {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  return {thinking: sel ? sel.dataset.v : DEFAULTS.thinking, temperature: +$("s-temp").value, top_p: +$("s-topp").value,
          top_k: +$("s-topk").value, min_p: +$("s-minp").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
          show: $("s-show").getAttribute("aria-checked") === "true", mcp: $("s-mcp").getAttribute("aria-checked") === "true",
          share: $("s-share").getAttribute("aria-checked") === "true", system: $("s-system").value,
          sys_default: $("s-sys-default").getAttribute("aria-checked") === "true", ctx: ctxInfo ? ctxValue() : null};
}
async function openDrawer() {
  drawerReturn = document.activeElement;
  $("drawer").dataset.open = "true";
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").hidden = false;
  $("dirty-bar").hidden = true;
  loadDrawer();
  await Promise.all([loadShared(), loadMcp(), loadCtxInfo()]);
  renderCtxField();
  loadDrawer();
  drawerSnapshot = JSON.stringify(formState());
  $("drawer-close").focus();
}
function closeDrawer(force = false) {
  if ($("drawer").dataset.open !== "true") return;
  if (!force && drawerSnapshot && JSON.stringify(formState()) !== drawerSnapshot) {
    $("dirty-bar").hidden = false;                 // changes not saved: ask first
    $("dirty-save").focus();
    return;
  }
  $("drawer").dataset.open = "false";
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").hidden = true;
  $("dirty-bar").hidden = true;
  if (drawerReturn && drawerReturn.focus) drawerReturn.focus();
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) {
    b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
    b.tabIndex = b.dataset.v === s.thinking ? 0 : -1;
  }
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k; $("s-minp").value = s.min_p || 0;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  $("s-system").value = current.system || "";
  $("s-sys-default").setAttribute("aria-checked", String(!!settings.sys_default));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false, sharedNow = null;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) { const j = await r.json(); sharedOn = !!j.shared; sharedNow = j.defaults || null; }
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  renderShareSub();
}
function renderShareSub() {
  const d = sharedOn && sharedNow ? sharedNow : null;
  if (!d || !Object.keys(d).length) { $("share-sub").textContent = "API clients get these settings for anything they don't set themselves"; return; }
  const names = {reasoning_effort: "thinking", temperature: "temperature", top_p: "top-p", top_k: "top-k", min_p: "min-p", seed: "seed", max_tokens: "max tokens"};
  $("share-sub").textContent = "API clients get now: " + Object.entries(d).map(([k, v]) => `${names[k] || k} ${v}`).join(", ");
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking, temperature: +s.temperature};
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p});
  if (+s.temperature > 0 && +s.top_k > 0) d.top_k = +s.top_k;
  if (+s.temperature > 0 && +s.min_p > 0) d.min_p = +s.min_p;
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  const j = await r.json();
  sharedOn = !!j.shared;
  sharedNow = j.defaults || null;
}
const PRESETS = {precise: {temperature: 0}, balanced: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0},
                 creative: {temperature: 1.0, top_p: 0.95, top_k: 0, min_p: 0.05}};
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away", low: "a short think", high: "thinks it through",
                                       max: "the most thorough, the longest"}[sel.dataset.v] +
    (sel.dataset.v === DEFAULTS.thinking ? " (default)" : "") : "";
  $("o-topk").textContent = +$("s-topk").value > 0 ? $("s-topk").value : "off";
  $("o-minp").textContent = +$("s-minp").value > 0 ? (+$("s-minp").value).toFixed(2) : "off";
  for (const id of ["s-topp", "s-topk", "s-minp"]) $(id).disabled = t === 0;
  // the preset these values match, if any
  const f = {temperature: t, top_p: +$("s-topp").value, top_k: +$("s-topk").value, min_p: +$("s-minp").value};
  for (const b of $("s-preset").children) {
    const p = PRESETS[b.dataset.p];
    b.setAttribute("aria-pressed", String(Object.entries(p).every(([k, v]) => (k === "temperature" || t > 0) && Math.abs(f[k] - v) < 1e-9)));
  }
}
for (const b of $("s-preset").children) b.onclick = () => {
  const p = PRESETS[b.dataset.p];
  if (p.temperature !== undefined) $("s-temp").value = p.temperature;
  if (p.top_p !== undefined) $("s-topp").value = p.top_p;
  if (p.top_k !== undefined) $("s-topk").value = p.top_k;
  if (p.min_p !== undefined) $("s-minp").value = p.min_p;
  outputs();
};
function selectThinking(b) {
  for (const x of $("s-thinking").children) { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; }
  outputs();
}
for (const b of $("s-thinking").children) b.onclick = () => selectThinking(b);
// a radio group: the arrow keys move the choice
$("s-thinking").addEventListener("keydown", (e) => {
  if (!["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(e.key)) return;
  e.preventDefault();
  const items = [...$("s-thinking").children];
  const i = items.findIndex((x) => x.getAttribute("aria-checked") === "true");
  const next = items[(i + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length];
  selectThinking(next);
  next.focus();
});
for (const id of ["s-temp", "s-topp", "s-topk", "s-minp"]) $(id).oninput = outputs;
for (const id of ["s-show", "s-mcp", "s-share", "s-sys-default"]) {
  $(id).onclick = () => $(id).setAttribute("aria-checked", String($(id).getAttribute("aria-checked") !== "true"));
}
$("s-reset").onclick = () => { loadDrawer({...DEFAULTS, system: current.system}); renderCtxField(); };
async function applySettings() {
  const f = formState();
  settings = {thinking: f.thinking, temperature: f.temperature, top_p: f.top_p, top_k: f.top_k, min_p: f.min_p, max: f.max,
              seed: f.seed, show: f.show, mcp: f.mcp, sys_default: f.sys_default,
              system: f.sys_default ? f.system : settings.system || ""};
  store.set("sampling", settings);
  current.system = f.system;
  if (messages.length) saveChat();
  closeDrawer(true);
  let note = settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "";
  if (f.share || sharedOn) {
    try {
      await saveShared(f.share, settings);
      note = f.share ? "Other apps (API clients) use these settings from their next request." : "Other apps use their own settings again.";
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
      note = null;
    }
  }
  if (note !== null) toast("success", "Settings saved", note);
  if (ctxInfo && f.ctx && f.ctx !== ctxInfo.context) await confirmContext(f.ctx);
}
$("s-apply").onclick = applySettings;
$("dirty-save").onclick = applySettings;
$("dirty-discard").onclick = () => closeDrawer(true);
$("sampling-btn").onclick = openDrawer;
$("drawer-close").onclick = () => closeDrawer();
$("scrim").onclick = () => closeDrawer();

// ------------------------------------------------------------------ the keyboard
document.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    if (!$("modal").hidden) trapTab(e, $("modal-box"));
    else if ($("drawer").dataset.open === "true") trapTab(e, $("drawer"));
    return;
  }
  if (e.key === "Escape") {
    if (!$("modal").hidden) { closeModal(); return; }
    if ($("drawer").dataset.open === "true") { closeDrawer(); return; }
    if (document.body.classList.contains("chats-open")) { showChats(false); return; }
    if (editIndex != null) { cancelEdit(); return; }
    if (busy && busy.chat === current) { stopAnswer(); return; }
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "O" || e.key === "o")) { e.preventDefault(); newChat(); }
});

// ------------------------------------------------------------------ start
setBusy(false);
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
showTab(location.hash.slice(1) || "chat");
loadChats().then(() => {
  renderChat();
  renderChatList();
  return loadHealth();
}).then(loadMcp).then(() => {
  loadUpdate();                                    // (the About tab's dot when a new version is out)
  resumePending();
  if (startQuestion) { if (messages.length) openChat(newChatObj()); $("input").value = startQuestion; send(); }
});
poll();
