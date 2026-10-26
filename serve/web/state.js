// serve/web/state.js - the page's one state object, how the view learns that it changed, and the small things every part
// uses: toasts, dialogs (confirmations, a text to type, the HTML preview), the theme and the chat settings.
import {useEffect, useState} from "./preact.js";
import {store, writeClipboard} from "./lib.js";

export const S = {
  // what the server says
  health: {model: "maya", images: false, max_context: 0, version: null, api_key: false, mcp_tools: false},
  metrics: null,            // GET /metrics, every second
  conn: "connecting",       // "ok", "connecting", "down" (three failed polls), "key" (an API key is needed)
  mcp: {servers: [], tools: 0},
  ctxInfo: null,            // GET /api/context
  portInfo: null,           // GET /api/port
  update: null,             // GET /api/update
  updateWatch: null,        // {to, down}: an update in progress
  reload: null,             // {to}: a context reload in progress (the composer waits)
  shared: {on: false, now: null},   // GET /settings: these settings as every API client's defaults

  // the view
  view: "chat",             // "chat", "monitor" or "models"
  sideOpen: store.get("side-open", true),   // the sidebar on a wide screen
  drawer: false,            // the chats drawer on a phone
  settings: null,           // the Settings dialog's open section, or null
  search: false,            // the search dialog
  dialog: null,             // a confirmation or a question
  toasts: [],

  // the chats (chats.js)
  chats: [], current: null, folders: store.get("folders", []), folderOpen: store.get("folders-open", {}),
  busy: null, editIndex: null, attachments: [], draft: "", hint: "",
  ctx: {tokens: null, approx: false, why: ""},
};

// ------------------------------------------------------------------ change -> one repaint (batched)
const subs = new Set();
let queued = false;
export function emit() {
  if (queued) return;
  queued = true;
  queueMicrotask(() => { queued = false; for (const f of subs) f(); });
}
export function useStore() {
  const [, force] = useState(0);
  useEffect(() => {
    const f = () => force((n) => n + 1);
    subs.add(f);
    return () => subs.delete(f);
  }, []);
  return S;
}
export const set = (patch) => { Object.assign(S, patch); emit(); };

// ------------------------------------------------------------------ toasts
let toastId = 0;
export function toast(kind, title, text = "", ms = 3500, action = null) {
  const t = {id: ++toastId, kind, title, text, action};
  S.toasts = [...S.toasts, t].slice(-4);
  emit();
  setTimeout(() => dismissToast(t.id), ms);
}
export function dismissToast(id) { S.toasts = S.toasts.filter((t) => t.id !== id); emit(); }
export async function copyText(text, quiet = false) {
  await writeClipboard(text);
  if (!quiet) toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ dialogs
// {title, body (HTML text or a node), actions: [{label, primary, danger, run}], wide, onClose}
let dialogReturn = null;
export function openDialog(d) {
  dialogReturn = document.activeElement;
  S.dialog = d;
  emit();
}
export function closeDialog(byAction = false) {
  const d = S.dialog;
  if (!d) return;
  S.dialog = null;
  emit();
  if (d.onClose && !byAction) d.onClose();
  setTimeout(() => {   // the keyboard goes back where it was, else into the dialog still open (Settings)
    if (dialogReturn && dialogReturn.focus && document.contains(dialogReturn)) dialogReturn.focus();
    else { const b = document.querySelector('.dialog [aria-current="page"]'); if (b) b.focus(); }
  }, 0);
}
export function confirmDialog(title, body, okLabel, danger = false) {
  return new Promise((ok) => {
    let answered = false;
    openDialog({title, body, onClose: () => { if (!answered) ok(false); },
                actions: [{label: "Cancel", run: () => { answered = true; ok(false); }},
                          {label: okLabel, primary: !danger, danger, run: () => { answered = true; ok(true); }}]});
  });
}
// a text to type: resolves with it, or null
export function promptDialog({title, label = "", value = "", placeholder = "", okLabel = "Save", multiline = false, note = ""}) {
  return new Promise((ok) => {
    let answered = false;
    openDialog({title, prompt: {label, value, placeholder, multiline, note}, onClose: () => { if (!answered) ok(null); },
                actions: [{label: "Cancel", run: () => { answered = true; ok(null); }},
                          {label: okLabel, primary: true, run: (v) => { answered = true; ok(v); }}]});
  });
}

// ------------------------------------------------------------------ the theme: dark, light, or the system's
export function themeChoice() { try { return localStorage.getItem("maya.ui-theme") || "dark"; } catch (e) { return "dark"; } }
export function applyTheme() {
  const c = themeChoice();
  const t = c === "system" ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : c;
  document.documentElement.dataset.theme = t;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = t === "dark" ? "#19191d" : "#ffffff";
}
export function setTheme(c) {
  try { localStorage.setItem("maya.ui-theme", c); } catch (e) { /* ignore */ }
  applyTheme();
  emit();
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { if (themeChoice() === "system") applyTheme(); });

// ------------------------------------------------------------------ the chat settings (each model keeps its own)
// thinking: GLM's own levels - none (Off), low, high, max.  The page before v1.0.18 saved Qwen-era names, where its
// "Medium" was GLM's High and its "High" GLM's Max: they are translated once (thinking_v 2)
export const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, max: "", seed: "", show: true,
                         mcp: true, system: "", sys_default: false, thinking_v: 2};
export const THINKING = ["none", "low", "high", "max"];
export const LEVELS = [
  {id: "none", name: "Off", desc: "Answers right away"},
  {id: "low", name: "Low", desc: "A short think"},
  {id: "high", name: "High", desc: "Thinks it through"},
  {id: "max", name: "Max", desc: "The most thorough, and the longest"},
];
export const THINK_TITLES = {
  glm: {none: "No thinking: it answers right away", low: "GLM's Low: a short think", high: "GLM's High: thinks it through",
        max: "GLM's Max: the most thorough, and the longest"},
  deepseek: {none: "No thinking: it answers right away", low: "DeepSeek's Low: reasoning effort 50",
             high: "DeepSeek's High: reasoning effort 75", max: "DeepSeek's Max: reasoning effort 100, the most thorough"},
};
export const PRESETS = {precise: {temperature: 0}, balanced: {temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0},
                        creative: {temperature: 1.0, top_p: 0.95, top_k: 0, min_p: 0.05}};
export const PRESET_NAMES = {precise: "Precise", balanced: "Balanced", creative: "Creative"};
export function presetOf(p) {
  const t = +p.temperature;
  for (const [k, v] of Object.entries(PRESETS)) {
    if (Object.entries(v).every(([kk, vv]) => (kk === "temperature" || t > 0) && Math.abs(+p[kk] - vv) < 1e-9)) return k;
  }
  return null;
}
const glmLevel = (v) => ({medium: "high", high: "max", xhigh: "max"}[v] || v);

export const prefsKey = () => (/deepseek/i.test(S.health.model || "") ? "sampling_deepseek" : "sampling");
export let P = null;                     // the running model's chat settings
export function loadPrefs(key = prefsKey()) {
  const p = {...DEFAULTS, ...store.get(key, {})};
  if (p.thinking_v !== 2) { p.thinking = glmLevel(p.thinking); p.thinking_v = 2; }
  if (!THINKING.includes(p.thinking)) p.thinking = DEFAULTS.thinking;
  P = p;
  return p;
}
export function savePrefs(patch) {
  P = {...P, ...patch};
  store.set(prefsKey(), P);
  emit();
}
loadPrefs("sampling");
// the instructions new chats start with
export const defaultSystem = () => (P.sys_default ? P.system || "" : "");
