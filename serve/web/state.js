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
// (saved as "maya.theme"; the old dashboard's "maya.ui-theme" is not read - dark is the default for everyone)
export function themeChoice() { try { return localStorage.getItem("maya.theme") || "dark"; } catch (e) { return "dark"; } }
export function applyTheme() {
  const c = themeChoice();
  const t = c === "system" ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : c;
  document.documentElement.dataset.theme = t;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = t === "dark" ? "#19191d" : "#ffffff";
}
export function setTheme(c) {
  try { localStorage.setItem("maya.theme", c); localStorage.removeItem("maya.ui-theme"); } catch (e) { /* ignore */ }
  applyTheme();
  emit();
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { if (themeChoice() === "system") applyTheme(); });
// the colours, in either theme: "default" (neutral greys, violet the accent) or "violet" (violet-tinted surfaces and a soft
// glow, the earlier dashboard's look) - saved as "maya.scheme" in this browser; <html data-scheme="violet"> (tokens.css)
export function schemeChoice() { try { return localStorage.getItem("maya.scheme") === "violet" ? "violet" : "default"; } catch (e) { return "default"; } }
export function applyScheme() {
  if (schemeChoice() === "violet") document.documentElement.dataset.scheme = "violet";
  else delete document.documentElement.dataset.scheme;
}
export function setScheme(c) {
  try { if (c === "violet") localStorage.setItem("maya.scheme", c); else localStorage.removeItem("maya.scheme"); } catch (e) { /* ignore */ }
  applyScheme();
  emit();
}

// ------------------------------------------------------------------ the chat settings (each model keeps its own)
// thinking: GLM's own levels - none (Off), low, high, max - and DeepSeek's by the same names; MiniMax-M3.1's: Off, Auto
// (no effort named: its template's default, the default here) and its card's five efforts, each in its template's
// adaptive mode (the model decides whether to think).  The page before v1.0.18 saved Qwen-era names, where its
// "Medium" was GLM's High and its "High" GLM's Max: they are translated once (thinking_v 2)
export const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, max: "", seed: "", show: true,
                         mcp: true, system: "", sys_default: false, thinking_v: 2};
const THINK_LEVELS = {glm: ["none", "low", "high", "max"], deepseek: ["none", "low", "high", "max"],
                      minimax: ["none", "auto", "low", "medium", "high", "xhigh", "max"]};
const THINK_DEFAULT = {glm: "high", deepseek: "high", minimax: "auto"};   // (the run config's level wins: loadHealth)
const LEVEL = {none: ["Off", "Answers right away"], auto: ["Auto", "It decides whether to think"], low: ["Low", "A short think"],
               medium: ["Medium", "A moderate think"], high: ["High", "Thinks it through"], xhigh: ["XHigh", "Thinks hard"],
               max: ["Max", "The most thorough, and the longest"]};
const LEVEL_MINIMAX = {low: "Effort low", medium: "Effort medium", high: "Effort high", xhigh: "Effort extra-high",
                       max: "Effort max, the most thorough"};   // (each: it decides whether to think, at that effort)
export const THINK_TITLES = {
  glm: {none: "No thinking: it answers right away", low: "GLM's Low: a short think", high: "GLM's High: thinks it through",
        max: "GLM's Max: the most thorough, and the longest"},
  deepseek: {none: "No thinking: it answers right away", low: "DeepSeek's Low: reasoning effort 50",
             high: "DeepSeek's High: reasoning effort 75", max: "DeepSeek's Max: reasoning effort 100, the most thorough"},
  minimax: {none: "No thinking: it answers right away", auto: "MiniMax's default: it decides whether to think, no effort set",
            low: "Effort low: it decides whether to think, and keeps it short", medium: "Effort medium: it decides whether to think",
            high: "Effort high: it decides whether to think, and thinks it through",
            xhigh: "Effort extra-high: it decides whether to think, and thinks hard",
            max: "Effort max: it decides whether to think - the most thorough, and the longest"},
};
// the running model's family, its thinking levels (THINKING, LEVELS: live bindings) and their titles
export const familyOf = (model = S.health.model) =>
  (/deepseek/i.test(model || "") ? "deepseek" : /minimax/i.test(model || "") ? "minimax" : "glm");
export let THINKING = THINK_LEVELS.glm;
export let LEVELS = [];
export function setFamily(model) {
  const f = familyOf(model);
  THINKING = THINK_LEVELS[f];
  LEVELS = THINKING.map((id) => ({id, name: LEVEL[id][0], desc: (f === "minimax" && LEVEL_MINIMAX[id]) || LEVEL[id][1]}));
  DEFAULTS.thinking = THINK_DEFAULT[f];
  return f;
}
setFamily("");
export const thinkTitles = () => THINK_TITLES[familyOf()];
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

export const prefsKey = () => ({deepseek: "sampling_deepseek", minimax: "sampling_minimax"}[familyOf()] || "sampling");
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
