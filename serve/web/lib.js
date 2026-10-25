// serve/web/lib.js - small helpers every part of the page uses: formatting, this browser's storage (localStorage for
// settings, IndexedDB for the chats), the server's address and key, the clipboard and file downloads.

export const SPRITE = "web/sprite.svg";

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
export const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–"
  : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
export const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K", 1048576 -> "1M" (powers of two), else like kfmt
export const ctxfmt = (n) => (n && n % 1048576 === 0 ? `${fmt(n / 1048576)}M` : n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
// a token count as the context meter shows it (128K = 131,072)
export const tokfmt = (n) => (n == null ? "–" : n < 1000 ? fmt(n) : `${fmt(n / 1024, n < 10240 ? 1 : 0)}K`);
export const GIB = 1073741824;
export const gb = (b, d = 1) => (b == null ? "–" : fmt(b / GIB, d));   // memory: binary GB, as Windows shows it
export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
// an age in seconds as people say it: "12 s", "4 min", "1 h 25 min"
export function ago(s) {
  if (s < 90) return `${fmt(s)} s`;
  if (s < 5400) return `${fmt(Math.round(s / 60))} min`;
  return `${fmt(Math.floor(s / 3600))} h ${fmt(Math.round((s % 3600) / 60))} min`;
}
export const timeStr = (t) => new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
export const rate = (mb) => (mb == null ? "–" : mb >= 1000 ? `${fmt(mb / 1024, 1)} GB/s` : `${fmt(mb, mb < 10 ? 1 : 0)} MB/s`);

// settings in localStorage ("maya.<key>", JSON); the names of releases before Maya ("strata.<key>") are still read
export const store = {
  get(k, d) {
    try {
      let v = localStorage.getItem("maya." + k);
      if (v === null) v = localStorage.getItem("strata." + k);
      return v === null ? d : JSON.parse(v);
    } catch (e) { return d; }
  },
  set(k, v) { try { localStorage.setItem("maya." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// the chats' store (IndexedDB: no 5 MB cap, pictures and attached files kept, so a reopened chat sends exactly what it
// sent before)
export const db = (() => {
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

// the server: relative addresses (the page also works behind a proxy under a path), and the API key when it has one
export function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
// a JSON request; an error carries the server's own message
export async function api(path, {method = "GET", body, signal} = {}) {
  const r = await fetch(path, {method, headers: headers(body !== undefined), signal,
                               body: body === undefined ? undefined : JSON.stringify(body)});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    if (r.status === 401) msg = "This server needs its API key: add it in Settings › Connect your tools.";
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

export async function writeClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
}

export function download(name, text, type = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// a chat's age in the sidebar's groups
export function dayGroup(t, now = new Date()) {
  const d = new Date(t);
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (t >= start) return "Today";
  if (t >= start - 86400000) return "Yesterday";
  if (t >= start - 7 * 86400000) return "Previous 7 days";
  if (t >= start - 30 * 86400000) return "Previous 30 days";
  return d.getFullYear() === now.getFullYear() ? d.toLocaleDateString([], {month: "long"})
                                               : d.toLocaleDateString([], {month: "long", year: "numeric"});
}
export function relTime(t) {
  const d = new Date(t), now = new Date();
  if (d.toDateString() === now.toDateString()) return timeStr(t);
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], {month: "short", day: "numeric"});
}

// the model family's logo (web/logo-*.svg: lobe-icons' marks, MIT) and its name as people write it:
// "glm-5.3-flash" -> "GLM 5.3 Flash", "deepseek-v4.1-flash" -> "DeepSeek V4.1 Flash", "qwen3.8-flash-next" -> "Qwen3.8 Flash Next"
const MODEL_LOGOS = [[/glm/i, "glm", true], [/deepseek/i, "deepseek"], [/minimax/i, "minimax"], [/qwen/i, "qwen"]];
export function modelLogo(name) {
  const m = MODEL_LOGOS.find(([re]) => re.test(name || ""));
  return m ? {src: `web/logo-${m[1]}.svg`, mono: !!m[2]} : null;
}
const NAME_WORDS = {glm: "GLM", deepseek: "DeepSeek", minimax: "MiniMax", qwen: "Qwen"};
export function modelName(id) {
  return String(id || "").split(/[-_\s]+/).filter(Boolean).map((w) => {
    const lw = w.toLowerCase(), m = lw.match(/^([a-z]+)(\d.*)$/);
    if (NAME_WORDS[lw]) return NAME_WORDS[lw];
    if (m && NAME_WORDS[m[1]]) return NAME_WORDS[m[1]] + m[2];   // (a brand with its version: qwen3.8)
    if (/^[a-z]\d/.test(lw)) return lw.toUpperCase();             // (a version: v4.1, m3.1)
    return lw.charAt(0).toUpperCase() + lw.slice(1);
  }).join(" ");
}

// the platform's modifier key, as the shortcuts are shown
export const MOD = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘" : "Ctrl";
export const isPhone = () => matchMedia("(max-width: 767px)").matches;
export const coarse = () => matchMedia("(pointer: coarse)").matches;
