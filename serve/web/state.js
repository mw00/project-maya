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
  savedSheet: false,        // the phone's Saved sheet (what the API would have charged)
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
// (the penalties: blank, none sent - no model has any as a standard; its user may set them)
export const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, max: "", seed: "", show: true,
                         mcp: true, system: "", sys_default: false, thinking_v: 2,
                         repetition_penalty: "", frequency_penalty: "", presence_penalty: ""};
export const PENALTIES = ["repetition_penalty", "frequency_penalty", "presence_penalty"];
// CYBER-FROST-3.8-PS (Qwen3.8's template, thinking always on): its three reasoning efforts, low / medium / xhigh
const THINK_LEVELS = {glm: ["none", "low", "high", "max"], deepseek: ["none", "low", "high", "max"],
                      minimax: ["none", "auto", "low", "medium", "high", "xhigh", "max"], qwen: ["low", "medium", "xhigh"]};
const THINK_DEFAULT = {glm: "max", deepseek: "high", minimax: "auto", qwen: "xhigh"};   // (the run config's level wins: loadHealth)
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
  qwen: {low: "Reasoning effort low: a short think", medium: "Reasoning effort medium: thinks it through",
         xhigh: "Reasoning effort xhigh (its default): checks its assumptions and alternatives - the most thorough"},
};
// the running model's family, its thinking levels (THINKING, LEVELS: live bindings) and their titles
export const familyOf = (model = S.health.model) =>
  (/deepseek/i.test(model || "") ? "deepseek" : /minimax/i.test(model || "") ? "minimax"
    : /cyber-frost|qwen/i.test(model || "") ? "qwen" : "glm");
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

// a level this model does not offer (a chat started on another model): the nearest it has - MiniMax's XHigh is Max
// elsewhere, its Medium High (as the server maps them), Auto the model's own default - not the default whatever it was
const NEAREST = {xhigh: ["max", "high"], medium: ["high", "low"], max: ["xhigh", "high"], high: ["medium", "max"],
                 low: ["low"], none: ["none"]};
export function levelFor(level) {
  if (THINKING.includes(level)) return level;
  return (NEAREST[level] || []).find((l) => THINKING.includes(l)) || null;
}

export const prefsKey = () => ({deepseek: "sampling_deepseek", minimax: "sampling_minimax", qwen: "sampling_qwen"}[familyOf()] ||
  "sampling");
export let P = null;                     // the running model's chat settings: its defaults, then its user's own choices
// stored: only what the user chose (a preset, a slider, a level) - the model's own defaults fill the rest, so they apply
// until its user picks something.  Before (prefs_v 3) a change saved every value as it was then, the defaults too:
// those that equal this model's defaults are not a choice and follow the model from now on
const CHOSEN = ["thinking", "temperature", "top_p", "top_k", "min_p", ...PENALTIES];
export function loadPrefs(key = prefsKey()) {
  let own = store.get(key, {}) || {};
  if (own.prefs_v !== 3 && S.health && S.health.model) {   // (once, against this model's defaults - not the built-in)
    own = {...own, prefs_v: 3};
    for (const k of CHOSEN) if (k in own && String(own[k]) === String(DEFAULTS[k])) delete own[k];
    store.set(key, own);
  }
  const p = {...DEFAULTS, ...own};
  if (p.thinking_v !== 2) { p.thinking = glmLevel(p.thinking); p.thinking_v = 2; }
  if (!THINKING.includes(p.thinking)) p.thinking = levelFor(p.thinking) || DEFAULTS.thinking;
  P = p;
  return p;
}
export function savePrefs(patch) {
  P = {...P, ...patch};
  const key = prefsKey();
  store.set(key, {...(store.get(key, {}) || {}), ...patch, prefs_v: 3});
  emit();
}
// the model running: its id and its files (the label: "GLM-5.3 · Maya-L") - another one is a model switch
export const modelId = () => `${S.health.model || ""}|${S.health.model_label || ""}`;
// a model switch (the Models page's, or another model found when the page loads - a setup that changed it): the
// user's sampling choices and level do not carry over - each model starts at its own standard parameters (the owner,
// 2026-10-27); within one model they stay, across reloads.  True when it was a switch.
export function newModelCheck() {
  const id = modelId(), was = store.get("model", null);
  store.set("model", id);
  if (!S.health.model || was === null || was === id) return false;
  let keys = [];                                  // (every model's stored settings: "maya.sampling", "maya.sampling_<family>")
  try { keys = Object.keys(localStorage).filter((k) => k.startsWith("maya.sampling")).map((k) => k.slice(5)); } catch (e) { /* none */ }
  for (const key of new Set([prefsKey(), ...keys])) {
    const own = {...(store.get(key, {}) || {})};
    for (const k of CHOSEN) delete own[k];
    store.set(key, own);
  }
  return true;
}

// back to the model's own for these settings (Settings' Reset): no longer the user's
export function resetPrefs(keys) {
  const key = prefsKey(), own = {...(store.get(key, {}) || {})};
  for (const k of keys) delete own[k];
  store.set(key, {...own, prefs_v: 3});
  P = {...P, ...Object.fromEntries(keys.map((k) => [k, DEFAULTS[k]]))};
  emit();
}
loadPrefs("sampling");
// the instructions new chats start with
export const defaultSystem = () => (P.sys_default ? P.system || "" : "");
