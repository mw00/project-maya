// serve/web/models.js - the Models page: the models set up on this machine (the run configs the setup wrote, GET
// /api/models), the running one marked; switching to another (POST /api/models/switch: Maya starts again with it);
// each one's sheet - Overview (its facts), Engine (its engine settings: POST /api/models/settings, a dry run first for
// the command they make; the running model's context size alone reloads in place, as Settings > Chat defaults does) and
// Storage (what deleting it frees and what stays because another model uses it); deleting one (POST /api/models/delete:
// off the list, and with its downloaded files - ticked - the dialog naming them and the GB freed; the running one only
// after a switch to another); and the files in the models folders no model uses (POST /api/models/leftovers deletes one).
import {html, useEffect, useRef} from "./preact.js";
import {api, ctxfmt, esc, fmt, isPhone, modelName, relTime} from "./lib.js";
import {S, confirmDialog, copyText, emit, toast} from "./state.js";
import {estimateTokens, newChat} from "./chats.js";
import {setSide, showView} from "./sidebar.js";
import {GLM_KV_BYTES, confirmContext, ctxHint, ctxStopsIn, loadCtxInfo} from "./server.js";
import {CtxSlider, Icon, ModelLogo, Seg, Switch, trapTab} from "./ui.js";
import {tr, languageChoice} from "./i18n.js";

// the page's own state (beside S: nothing else reads it but the model menu's count).  sheet: the model sheet open
// ({file, tab: overview / engine / storage, from: the button}); plans: each model's files, as deleting it would go
// (the delete's dry run); want: the sheet to open on the running model once the list is read (Settings' link)
export const MS = {data: null, err: null, add: false, edit: null, draft: null, tab: 0, preview: null, sheet: null, plans: {},
                   want: null, check: "", saving: false, restart: null, del: null, gone: null, cat: null, catErr: null, loadBusy: null};

export async function loadModels() {
  try {
    MS.data = await api("api/models");
    MS.err = null;
    MS.plans = {};                                     // (read again when a Storage tab or a Delete asks)
    saidDeleted(MS.data.deleted);
    if (MS.want) {
      const m = MS.data.models.find((x) => x.running), tab = MS.want;
      MS.want = null;
      if (m) openSheet(m, tab);
    }
  } catch (e) {
    MS.err = e.message;
  }
  emit();
  return MS.data;
}

// Settings > Chat defaults' link: the Models page, the running model's sheet on its Engine settings
export function openModelSheet(tab = "engine") {
  const m = MS.data && MS.data.models.find((x) => x.running);
  if (m) openSheet(m, tab);
  else MS.want = tab;
  showView("models");
}

// the model that was running, deleted by the server after the switch (said once a start, not on every visit)
function saidDeleted(x) {
  if (!x || !x.time) return;
  try {
    if (sessionStorage.getItem("maya.deleted") === String(x.time)) return;
    sessionStorage.setItem("maya.deleted", String(x.time));
  } catch (e) { /* (no storage: said again on the next visit) */ }
  if (x.error) toast("error", tr("{file} was not deleted", {file: x.file}), x.error, 12000);
  else MS.gone = {name: x.model ? `${modelName(x.model)} ${x.quant || ""}`.trim() : x.deleted, files: x.files, freed: x.freed,
                  folder: null, users: x.drafter_users || []};
}

const dateLocale = () => languageChoice() === "zh-CN" ? "zh-CN" : "en";
const titleOf = (m) => `${modelName(m.model)} ${m.quant}`;
const DRAFTS = {mtp: "MTP", dflash2: "DFlash2", dspark: "DSpark", none: "off"};
const size = (b) => (b >= 1e8 ? `${fmt(b / 1e9, b < 1e10 ? 1 : 0)} GB` : null);
const setUpOn = (w) => {
  const d = w ? new Date(String(w).replace(" ", "T")) : null;
  return d && !Number.isNaN(d.getTime()) ? tr("set up {date}", {date: d.toLocaleDateString(dateLocale(), {month: "short", day: "numeric"})}) : null;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// a GPU as the Monitor names it: "RTX 5060 Ti", not "NVIDIA GeForce RTX 5060 Ti"
const shortGpu = (n) => (n || "").replace(/^NVIDIA /, "").replace(/^GeForce /, "").replace(/^AMD /, "").replace(/^Radeon /, "");
// DeepSeek's --tp choices: 1, 2, the calibration's 4, 8 and 16 up to the GPUs picked, all of them as one group, and the
// config's own (serve/models.py takes 1 up to the GPUs picked, 2 on one GPU)
const tpChoices = (n, ...own) => [...new Set([1, 2, ...[4, 8, 16].filter((t) => t <= n), Math.max(1, n), ...own.map(Number)])]
  .filter((t) => t >= 1).sort((a, b) => a - b);

// ------------------------------------------------------------------ Maya starting again (a switch, or new settings)
async function watchRestart(r) {
  MS.restart = r;
  emit();
  let down = false;
  for (let i = 0; i < 450; i++) {                      // (15 minutes: a big model's first load can take a while)
    await new Promise((ok) => setTimeout(ok, 2000));
    let h = null;
    try {
      const res = await fetch("health", {cache: "no-store"});
      h = res.ok ? await res.json() : null;
    } catch (e) { h = null; }
    if (!h) { down = true; continue; }
    if (down && h.status === "ok") {                   // back: the page starts over with the model that runs now
      toast("success", r.why === "switch" ? tr("{model} is running", {model: r.name}) : tr("Maya is back"), tr("Reloading the page…"), 3000);
      setTimeout(() => location.reload(), 900);
      return;
    }
  }
  MS.restart = null;
  emit();
  toast("error", tr("Maya is not back yet"), tr("The terminal it runs in shows why. This page reconnects when it is."), 30000);
}

async function switchTo(m) {
  const run = (MS.data.models || []).find((x) => x.running);
  const restarts = MS.data.restarts;
  const ok = await confirmDialog(tr("Switch to {model}?", {model: titleOf(m)}),
    (restarts
      ? tr("<p>Maya stops <b>{running}</b> and loads <b>{model}</b> from <code>{file}</code>. That takes a few minutes.</p><ul><li>Your chats stay here. The chat settings start at {model}'s own standard parameters and thinking level: a model's are not another's.</li><li>Apps using the API get a “try again” until it is back, then the model name <code>{modelId}</code>.</li>{busy}</ul>", {running: esc(run ? titleOf(run) : tr("the model running now")), model: esc(titleOf(m)), file: esc(m.file), modelId: esc(m.model), busy: S.busy ? tr("<li>The answer being written finishes first.</li>") : ""})
      : tr("<p><b>{model}</b> becomes the model Maya starts next. This server was not started by maya.sh or START-MAYA.bat, so it cannot start again by itself: stop it and start Maya again.</p>", {model: esc(titleOf(m))})),
    restarts ? tr("Switch model") : tr("Start it next"));
  if (!ok) return;
  try {
    const out = await api("api/models/switch", {method: "POST", body: {file: m.file}});
    if (!out.restarts) {
      toast("success", tr("Saved as the next model"), tr("{model} starts the next time Maya starts (./maya.sh or START-MAYA.bat).", {model: titleOf(m)}), 8000);
      loadModels();
      return;
    }
  } catch (e) {
    toast("error", tr("The model did not change"), e.message, 8000);
    return;
  }
  watchRestart({why: "switch", file: m.file, name: titleOf(m)});
}

// Unload: the engine ends and the GPU has its memory back (POST /models/unload); Load starts it again (/models/load, a
// few minutes).  Any request to the API loads it too.
async function loadOrUnload(op) {
  MS.loadBusy = op === "load" ? "load" : null;
  emit();
  try {
    await api(`models/${op}`, {method: "POST", body: {}});
  } catch (e) {
    toast("error", op === "load" ? tr("The model did not load") : tr("The model did not unload"), e.message, 8000);
  }
  MS.loadBusy = null;
  await loadModels();
}

// ------------------------------------------------------------------ a model's sheet: Overview, Engine, Storage
// (the engine settings a draft while the sheet is open: Cancel, Close or Esc drop it)
const SHEET_TABS = [{id: "overview", label: "Overview"}, {id: "engine", label: "Engine"}, {id: "storage", label: "Storage"}];
function openSheet(m, tab) {
  MS.sheet = {file: m.file, tab: tab === "engine" && !m.ok ? "overview" : tab, from: document.activeElement};
  if (m.ok) openEditor(m);
  if (MS.sheet.tab === "storage") loadPlan(m);
  emit();
}
function closeSheet() {
  const s = MS.sheet;
  if (!s || MS.saving) return;
  MS.sheet = null;
  closeEditor();
  if (s.from && s.from.focus && document.contains(s.from)) s.from.focus();
}
function sheetTab(m, tab) {
  MS.sheet.tab = tab;
  if (tab === "storage") loadPlan(m);
  emit();
}

// a model's files as deleting it would go (the delete's dry run: nothing deleted), for its Storage tab and the dialog
async function loadPlan(m, again = false) {
  if (MS.plans[m.file] && !again) return MS.plans[m.file];
  MS.plans[m.file] = {loading: true, items: [], shared: [], kept: []};
  let p;
  try {
    p = await api("api/models/delete", {method: "POST", body: {file: m.file, files: true, dry_run: true}});
  } catch (e) {
    p = {ok: false, why: e.message, items: [], shared: [], kept: []};
  }
  MS.plans[m.file] = p;
  emit();
  return p;
}

// ------------------------------------------------------------------ the engine settings (the sheet's Engine tab, a draft)
function openEditor(m) {
  MS.edit = m.file;
  MS.draft = JSON.parse(JSON.stringify(m.settings));
  MS.tab = 0;
  MS.preview = m.command;
  MS.check = "";
  if (m.running) loadCtxInfo();                     // (the running engine's own measurement: the Context size's line)
  emit();
}
function closeEditor() {
  MS.edit = MS.draft = MS.preview = null;
  MS.check = "";
  emit();
}

// the draft as the server takes it: "" -> the default (null), a number typed -> a number
const NUMS = ["context", "parallel", "reserve_mib", "expert_cache", "prefill", "kv_resident", "tp", "prefill_chunk", "cpu_threads"];
const DECS = ["reserve_gb", "cache_gb", "dspark_conf"];   // (a decimal: 0.7, 1.5)
function toSend(d) {
  const out = {...d};
  for (const k of [...NUMS, ...DECS]) {
    if (!(k in out)) continue;
    const v = out[k];
    out[k] = v === "" || v == null ? null : (DECS.includes(k) ? decimal : whole)(v) ? Number(v) : v;
  }
  out.env = Object.fromEntries((d.envRows || Object.entries(d.env || {})).filter(([k]) => String(k).trim()));
  delete out.envRows;
  return out;
}

let checkTimer = null;
function setDraft(patch) {
  Object.assign(MS.draft, patch);
  emit();
  clearTimeout(checkTimer);
  const file = MS.edit;
  checkTimer = setTimeout(async () => {               // the server checks them and says the command they make
    try {
      const out = await api("api/models/settings", {method: "POST", body: {file, settings: toSend(MS.draft), dry_run: true}});
      if (MS.edit !== file) return;
      MS.preview = out.command;
      MS.check = "";
    } catch (e) {
      if (MS.edit !== file) return;
      MS.check = e.message;
    }
    emit();
  }, 300);
}

async function saveSettings(m) {
  if (MS.saving) return;
  // the running model's, where Maya starts again by itself: the reload is said and confirmed first, as a switch is
  if (m.running && MS.data.restarts) {
    const ok = await confirmDialog(tr("Restart {model} with these engine settings?", {model: titleOf(m)}),
      tr("<p>Maya saves them into <code>{file}</code> and loads the model again with them. That takes a few minutes.</p><ul><li>Your chats stay here; the next answer reads the chat again.</li><li>Apps using the API get a “try again” until it is back.</li>{busy}</ul>", {file: esc(m.file), busy: S.busy ? tr("<li>The answer being written finishes first.</li>") : ""}), tr("Save and restart"));
    if (!ok) return;
  }
  MS.saving = true;
  emit();
  try {
    const out = await api("api/models/settings", {method: "POST", body: {file: m.file, settings: toSend(MS.draft)}});
    MS.saving = false;
    closeSheet();
    if (out.restarts) { watchRestart({why: "settings", file: m.file, name: titleOf(m)}); return; }
    toast("success", tr("Engine settings saved"), m.running
      ? tr("Saved into {file}. They apply from Maya's next start (./maya.sh or START-MAYA.bat starts it again by itself).", {file: m.file})
      : tr("Saved into {file}. {model} starts with them when you switch to it.", {file: m.file, model: titleOf(m)}), 8000);
    loadModels();
  } catch (e) {
    MS.check = e.message;
    toast("error", tr("Not saved"), e.message, 8000);
  } finally {
    MS.saving = false;
    emit();
  }
}

// the running model's context size alone: reloaded in place (POST /api/context, saved into its config), the same as
// Settings > Chat defaults > Context size - one setting in two places
async function reloadContext(n) {
  const tokens = S.ctx && S.ctx.tokens != null ? S.ctx.tokens : S.current ? estimateTokens(S.current.messages) : 0;
  if (await confirmContext(n, tokens)) closeSheet();
}

const whole = (v) => v === "" || v == null || /^\s*\d+\s*$/.test(String(v));
const decimal = (v) => v === "" || v == null || /^\s*(\d+\.?\d*|\.\d+)\s*$/.test(String(v));

function Row({label, flag, hint, changed, children, below, warn}) {
  return html`<div class="mdl-field">
    <div class="mdl-field__top">
      <span class="mdl-field__text"><span class="mdl-field__name"><span>${label}</span>${flag ? html`<code>${flag}</code>` : null}
        ${changed ? html`<i class="mdl-dot" role="img" aria-label=${tr("changed")} title=${tr("Changed, not saved yet")}></i>` : null}</span>
        ${hint ? html`<span class="mdl-field__hint">${hint}</span>` : null}</span>
      ${children ? html`<span class="mdl-field__ctl">${children}</span>` : null}
    </div>
    ${below}
    ${warn ? html`<div class="mdl-warn">${warn}</div>` : null}
  </div>`;
}
const Num = ({label, value, ph, unit, on}) => html`<span class="mdl-num">
  <input class=${`input input--num${whole(value) ? "" : " is-bad"}`} inputmode="numeric" aria-label=${label} value=${value ?? ""}
    placeholder=${ph == null ? ph : tr(ph)} onInput=${(e) => on(e.target.value)}/>${unit ? html`<span>${tr(unit)}</span>` : null}</span>`;
const Dec = ({label, value, ph, unit, on}) => html`<span class="mdl-num">
  <input class=${`input input--num${decimal(value) ? "" : " is-bad"}`} inputmode="decimal" aria-label=${label} value=${value ?? ""}
    placeholder=${ph == null ? ph : tr(ph)} onInput=${(e) => on(e.target.value.replace(",", "."))}/>${unit ? html`<span>${tr(unit)}</span>` : null}</span>`;
// a drafter's confidence floor (DeepSeek's and MiniMax's --dspark-conf), shown while it drafts
const draftConf = (d, ch, dflt) => (d.draft === "dspark" ? html`<${Row} label=${tr("Draft confidence")} flag="--dspark-conf" changed=${ch("dspark_conf")}
  hint=${tr("A draft is checked only while the drafter's confidence in it is at least this, from 0 to 1. Lower checks more drafts. Empty: the engine's {default}.", {default:dflt})}
  warn=${decimal(d.dspark_conf) && !(+d.dspark_conf > 1) ? null : tr("A number from 0 to 1, or empty.")}>
  <${Dec} label=${tr("Draft confidence")} value=${d.dspark_conf} ph=${String(dflt)} on=${(v) => setDraft({dspark_conf: v})}/><//>` : null);
function Stepper({label, value, min, max, on}) {
  const v = Number(value) || min;
  return html`<span class="mdl-step" role="group" aria-label=${label}>
    <button type="button" aria-label=${tr("Fewer")} disabled=${v <= min} onClick=${() => on(v - 1)}><${Icon} name="minus" sm/></button>
    <b aria-live="polite">${v}</b>
    <button type="button" aria-label=${tr("More")} disabled=${v >= max} onClick=${() => on(v + 1)}><${Icon} name="plus" sm/></button></span>`;
}
// a choice of "auto" and a number of one's own (the expert cache, the prompt chunk)
function AutoOr({label, k, d, dflt, unit, subLabel}) {
  const set = d[k] != null && d[k] !== "";
  return {ctl: html`<${Seg} label=${label} value=${set ? "set" : "auto"} options=${[{id: "auto", label: tr("Auto")}, {id: "set", label: tr("Set")}]}
      onChange=${(v) => setDraft({[k]: v === "auto" ? null : String(dflt)})}/>`,
    below: set ? html`<div class="mdl-sub"><span>${subLabel}</span><${Num} label=${subLabel} value=${d[k]} ph=${String(dflt)} unit=${unit}
      on=${(v) => setDraft({[k]: v})}/></div>` : null};
}

// the Context size: Settings > Chat defaults's slider (ui.js' CtxSlider) over this model's own range (up to the context it was
// trained for: GET /api/models' context_range), with the same line under it (server.js' ctxHint). `cost`: the cache's
// bytes per token as the draft sets it (ctxHint's win / whole / ram / vram); `keys`: the settings that change it - while
// none does, the running model's line is the engine's own measurement, as in Settings
function contextRow(m, d, ch, cost, keys = [], extra = null) {
  const s = m.settings, r = m.context_range || {min: 4096, max: 262144, tested: 262144};
  const n = Number(d.context), now = Number(s.context), stops = ctxStopsIn(r, now, n);
  const warn = [n !== now && n > r.tested ? tr("Beyond {context} is untested on this engine.", {context:ctxfmt(r.tested)}) : null, extra].filter(Boolean).join(" ");
  return html`<${Row} label=${tr("Context size")} flag="--max-context" changed=${ch("context")} warn=${warn || null}
    hint=${m.running ? tr("The same setting as Settings › Chat defaults › Context size: changed alone, the model reloads with it in place.") : null}
    below=${html`<${CtxSlider} id=${`mdl-ctx-${m.file}`} label="" value=${n} now=${now} stops=${stops}
      hint=${ctxHint(n, {live: m.running && !ch(...keys), now, ...cost})} onChange=${(v) => setDraft({context: v})}/>`}/>`;
}

function fieldsOf(m, d) {
  const s = m.settings, ch = (...ks) => ks.some((k) => !same(d[k], s[k])), glm = m.family === "glm";
  const per = Math.max(1, Number(d.parallel) || 1);   // (several answers at once: a cache each)
  const context = contextRow(m, d, ch, !glm ? {} : d.kv_streaming
    ? {win: Number(d.kv_resident) || 32768, whole: true, ram: GLM_KV_BYTES * per} : {vram: GLM_KV_BYTES * per},
    ["kv_streaming", "kv_resident", "parallel"]);
  const images = html`<${Row} label=${tr("Images")} flag=${glm ? "vision" : "--vision"} changed=${ch("images")}
    hint=${m.choices.images ? tr("Reads images in chats. Off leaves its memory to the experts.")
      : glm && m.backend === "hip" ? tr("The setup does not build GLM's image encoder for AMD GPUs (HIP) yet: this model reads text only.")
      : glm ? tr("This model was set up without its image encoder: the setup adds it.") : tr("This pack has no vision tower.")}>
    <${Switch} label=${tr("Images")} on=${d.images} disabled=${!m.choices.images && !d.images} onChange=${(v) => setDraft({images: v})}/><//>`;
  const gpus = html`<${Row} label=${tr("GPUs")} flag="gpu" changed=${ch("gpus")} hint=${tr("The engine splits the model across the GPUs picked. At least one.")}
    below=${html`<div class="mdl-gpus">${(MS.data.gpus || []).map((g) => {
      const on = d.gpus.includes(g.index);
      return html`<button type="button" role="checkbox" aria-checked=${String(on)} class="mdl-gpu"
        onClick=${() => { if (on && d.gpus.length === 1) return; setDraft({gpus: on ? d.gpus.filter((x) => x !== g.index) : [...d.gpus, g.index].sort((a, b) => a - b)}); }}>
        <span class="mdl-gpu__box">${on ? html`<${Icon} name="check" sm/>` : null}</span>
        <span><b>GPU ${g.index}</b><span>${[shortGpu(g.name), g.vram_gb ? `${fmt(g.vram_gb)} GB` : null].filter(Boolean).join(" · ") || tr("on this PC")}</span></span></button>`;
    })}</div>`}/>`;
  const reserve = html`<${Row} label=${tr("VRAM kept free")} flag=${glm ? "--vram-reserve-mib" : "--reserve-mib"} changed=${ch("reserve_mib")}
    hint=${glm ? tr("For the driver, the desktop and other programs. Empty: 700 MiB.") : tr("Per GPU, for the driver, the desktop and other programs. Empty: the engine's own.")}
    warn=${whole(d.reserve_mib) ? null : tr("A whole number of MiB, or empty for the default.")}>
    <${Num} label=${tr("VRAM kept free")} value=${d.reserve_mib} ph=${glm ? "700" : "default"} unit="MiB / GPU" on=${(v) => setDraft({reserve_mib: v})}/><//>`;
  if (m.family === "minimax") return mm3Fields(m, d, s, ch, images, gpus);
  if (m.family === "qwen") return qwenFields(m, d, s, ch, gpus);
  if (glm) {
    const ec = AutoOr({label: tr("Expert cache"), k: "expert_cache", d, dflt: 1200, unit: "experts", subLabel: tr("Experts in the first GPU's pool")});
    const pf = AutoOr({label: tr("Prompt chunk"), k: "prefill", d, dflt: 8192, unit: "tokens", subLabel: tr("Tokens a prompt is read in at a time")});
    const dflash = m.choices.draft.includes("dflash2");
    return [
      {label: tr("Model"), rows: [context,
        html`<${Row} label=${tr("Answers at once")} flag="--parallel" changed=${ch("parallel")}
          hint=${tr("Up to 8. Each extra conversation keeps its own context in VRAM, which the expert cache gives up.")}>
          <${Stepper} label=${tr("Answers at once")} value=${d.parallel} min=${1} max=${8} on=${(v) => setDraft({parallel: v})}/><//>`,
        html`<${Row} label=${tr("Drafting")} flag="--dflash" changed=${ch("draft")}
          hint=${dflash ? tr("MTP is built into the model. DFlash2 is a separate drafter, often faster for one answer at a time.")
            : tr("MTP is built into the model. DFlash2 needs its drafter: the setup's --draft dflash2 downloads it (0.7 GB).")}
          warn=${d.draft === "dflash2" && d.parallel > 1 ? tr("DFlash2 drafts for one answer at a time: with {count} at once the engine drafts with MTP.", {count:d.parallel}) : null}>
          <${Seg} label=${tr("Drafting")} value=${d.draft} options=${[{id: "mtp", label: "MTP"}, ...(dflash ? [{id: "dflash2", label: "DFlash2"}] : [])]}
            onChange=${(v) => setDraft({draft: v})}/><//>`,
        images]},
      {label: tr("GPUs and memory"), rows: [gpus, reserve,
        html`<${Row} label=${tr("Expert cache")} flag="--expert-cache" changed=${ch("expert_cache")}
          hint=${tr("Auto takes the VRAM left free, less the reserve. A number caps the first GPU's pool at that many experts.")}
          below=${ec.below} warn=${whole(d.expert_cache) ? null : tr("A whole number, or Auto.")}>${ec.ctl}<//>`]},
      {label: tr("Prompts and cache"), rows: [
        html`<${Row} label=${tr("Prompt chunk")} flag="--prefill" changed=${ch("prefill")}
          hint=${tr("Auto takes the largest chunk the expert pool can lend, up to 8,192 tokens.")}
          below=${pf.below} warn=${whole(d.prefill) ? null : tr("A whole number, or Auto.")}>${pf.ctl}<//>`,
        html`<${Row} label=${tr("KV streaming")} flag="--kv-streaming" changed=${ch("kv_streaming", "kv_resident")}
          hint=${tr("The attention cache lives in RAM, 12 KB a token, with a window of it in VRAM. Off: all of it in VRAM.")}
          below=${d.kv_streaming ? html`<div class="mdl-sub"><span>${tr("Positions each layer keeps in VRAM (--kv-resident)")}</span>
            <${Num} label=${tr("Positions each layer keeps in VRAM")} value=${d.kv_resident} ph="32768" unit="tokens" on=${(v) => setDraft({kv_resident: v})}/></div>` : null}
          warn=${whole(d.kv_resident) ? null : tr("A whole number, or empty for 32,768.")}>
          <${Switch} label=${tr("KV streaming")} on=${d.kv_streaming} onChange=${(v) => setDraft({kv_streaming: v})}/><//>`]},
    ];
  }
  const dspark = m.choices.draft.includes("dspark");
  return [
    {label: tr("Model"), rows: [context,
      html`<${Row} label=${tr("Answers at once")} flag="--parallel" changed=${ch("parallel")}
        hint=${tr("Up to 8. Each extra conversation keeps its own context's caches in VRAM, which the experts give up. An answer written alone still drafts; several at once are decoded together.")}
        warn=${d.parallel > 1 && d.context * d.parallel > 1048576 ? tr("{count} conversations of {context} each: a smaller context leaves the experts more VRAM.", {count:d.parallel,context:ctxfmt(d.context)}) : null}>
        <${Stepper} label=${tr("Answers at once")} value=${d.parallel} min=${1} max=${8} on=${(v) => setDraft({parallel: v})}/><//>`,
      html`<${Row} label=${tr("Drafting")} flag="--dspark" changed=${ch("draft")}
        hint=${dspark ? tr("DSpark drafts a few tokens at a time for the model to check. The speed calibration picks the faster on this PC.")
          : tr("This pack has no DSpark drafter.")}>
        <${Seg} label=${tr("Drafting")} value=${d.draft} options=${[...(dspark ? [{id: "dspark", label: "DSpark"}] : []), {id: "none", label: tr("Off")}]}
          onChange=${(v) => setDraft({draft: v})}/><//>`,
      images]},
    {label: tr("GPUs and memory"), rows: [gpus,
      html`<${Row} label=${tr("Tensor parallel")} flag="--tp" changed=${ch("tp")}
        hint=${tr("How many GPUs share each layer: equal GPUs, in groups of this size. On one GPU, 1. The calibration (--calibrate) picks the faster on this PC.")}
        warn=${Number(d.tp) === 2 && d.gpus.length < 2 ? tr("With one GPU picked, 2 runs its layers alone and reads prompts slower: use 1.")
          : Number(d.tp) > Math.max(2, d.gpus.length) ? tr("A group of {count} needs at least {count} GPUs picked.", {count:d.tp}) : null}>
        <${Seg} label=${tr("Tensor parallel")} value=${Number(d.tp)} options=${tpChoices(d.gpus.length, s.tp, d.tp).map((n) => ({id: n, label: String(n)}))}
          onChange=${(v) => setDraft({tp: v})}/><//>`,
      reserve,
      html`<${Row} label=${tr("Experts VRAM does not hold")} flag="--mmap-experts" changed=${ch("mmap")}
        hint=${tr("Auto copies them into RAM when it holds them, else reads them from the SSD in place.")}>
        <${Seg} label=${tr("Experts VRAM does not hold")} value=${d.mmap} options=${[{id: "auto", label: tr("Auto")}, {id: "on", label: tr("From the files")}, {id: "off", label: tr("In RAM")}]}
          onChange=${(v) => setDraft({mmap: v})}/><//>`]},
    {label: tr("Speed"), rows: [
      html`<${Row} label=${tr("Prompt chunk")} flag="--prefill-chunk" changed=${ch("prefill_chunk")} hint=${tr("Empty: the engine picks.")}
        warn=${whole(d.prefill_chunk) ? null : tr("A whole number, or empty.")}>
        <${Num} label=${tr("Prompt chunk")} value=${d.prefill_chunk} ph=${tr("auto")} unit="tokens" on=${(v) => setDraft({prefill_chunk: v})}/><//>`,
      html`<${Row} label=${tr("CPU threads")} flag="--cpu-threads" changed=${ch("cpu_threads")} hint=${tr("Set by the calibration (--calibrate) on this PC. Empty: the engine picks.")}
        warn=${whole(d.cpu_threads) ? null : tr("A whole number, or empty.")}>
        <${Num} label=${tr("CPU threads")} value=${d.cpu_threads} ph=${tr("auto")} unit=${tr("threads")} on=${(v) => setDraft({cpu_threads: v})}/><//>`,
      html`<${Row} label=${tr("Prompt lookup")} flag="--suffix-draft" changed=${ch("prompt_lookup")}
        hint=${tr("Where the answer repeats earlier text (code edits, quotes), the words that followed it there are checked as drafts: the same answer, often much faster. Off: --suffix-draft 0.")}>
        <${Switch} label=${tr("Prompt lookup")} on=${d.prompt_lookup} onChange=${(v) => setDraft({prompt_lookup: v})}/><//>`,
      draftConf(d, ch, 0.5)].filter(Boolean)},
  ];
}

// MiniMax's (maya-mm3: it stops at a flag it does not know, so only its own are offered).  Its KV cache's pinned RAM a
// context token, by format (maya.py's MM3_KV_BYTES)
const MM3_KV = [{id: "bf16", label: "BF16", bytes: 122880}, {id: "int8", label: "INT8", bytes: 63360},
                {id: "fp8", label: "FP8", bytes: 63360}, {id: "fp4", label: "FP4", bytes: 34560}];
function mm3Fields(m, d, s, ch, images, gpus) {
  const dspark = m.choices.draft.includes("dspark"), win = Number(d.kv_resident) || 32768;
  const kv = MM3_KV.find((f) => f.id === d.kv_cache);
  const kvb = kv ? kv.bytes : 63360;                 // (a format of the config's own the page does not list: INT8's size)
  const tps = [...new Set([0, 1, 2, 4].filter((t) => t <= Math.max(1, d.gpus.length)).concat([Number(s.tp), Number(d.tp)]))].sort((a, b) => a - b);
  return [
    {label: tr("Model"), rows: [
      contextRow(m, d, ch, d.parallel > 1 ? {vram: d.parallel * kvb} : d.kv_streaming ? {win, ram: kvb} : {vram: kvb},
        ["kv_cache", "kv_streaming", "kv_resident", "parallel"],
        d.parallel > 1 ? tr("{count} answers at once: each keeps a whole cache of this size in VRAM. The engine shortens a context that does not fit.", {count:d.parallel}) : null),
      html`<${Row} label=${tr("Answers at once")} flag="--parallel" changed=${ch("parallel")}
        hint=${tr("Up to 8. Each extra conversation keeps its own context's cache in VRAM, which the experts give up. With more than one the answers are decoded together: no KV streaming and no DSpark.")}
        warn=${d.parallel > 1 && d.context * d.parallel > 1048576 ? tr("{count} conversations of {context} each: a smaller context leaves the experts more VRAM.", {count:d.parallel,context:ctxfmt(d.context)}) : null}>
        <${Stepper} label=${tr("Answers at once")} value=${d.parallel} min=${1} max=${8} on=${(v) => setDraft({parallel: v})}/><//>`,
      html`<${Row} label=${tr("Drafting")} flag="--dspark" changed=${ch("draft")}
        hint=${dspark ? tr("DSpark drafts a few tokens at a time for the model to check: the same answer. The speed calibration picks the faster on this PC.")
          : tr("This pack has no DSpark drafter.")}
        warn=${d.draft === "dspark" && d.parallel > 1 ? tr("DSpark drafts for one answer at a time: with {count} at once the engine decodes without it.", {count:d.parallel}) : null}>
        <${Seg} label=${tr("Drafting")} value=${d.draft} options=${[...(dspark ? [{id: "dspark", label: "DSpark"}] : []), {id: "none", label: tr("Off")}]}
          onChange=${(v) => setDraft({draft: v})}/><//>`,
      images]},
    {label: tr("GPUs and memory"), rows: [gpus,
      html`<${Row} label=${tr("Tensor parallel")} flag="--tp" changed=${ch("tp")}
        hint=${tr("Off splits the layers across the GPUs. 1, 2 or 4: each layer's work shared by a group of that many GPUs; the setup picks 4 or 2 when the GPUs make whole groups.")}
        warn=${Number(d.tp) > d.gpus.length ? tr("A group of {count} needs at least {count} GPUs picked.", {count:d.tp}) : null}>
        <${Seg} label=${tr("Tensor parallel")} value=${Number(d.tp)} options=${tps.map((n) => ({id: n, label: n ? String(n) : tr("Off")}))}
          onChange=${(v) => setDraft({tp: v})}/><//>`,
      html`<${Row} label=${tr("VRAM kept free")} flag="--reserve-gb" changed=${ch("reserve_gb")}
        hint=${tr("Per GPU, beyond the engine's plan: for the driver, the desktop and other programs. Empty: 1 GB, 0.3 GB with tensor parallel.")}
        warn=${decimal(d.reserve_gb) ? null : tr("A number of GB, or empty for the default.")}>
        <${Dec} label=${tr("VRAM kept free")} value=${d.reserve_gb} ph=${tr("default")} unit="GB / GPU" on=${(v) => setDraft({reserve_gb: v})}/><//>`,
      Number(d.tp) > 0 ? html`<${Row} label=${tr("Expert cache")} flag="--cache-gb" changed=${ch("cache_gb")}
        hint=${tr("With tensor parallel: each GPU's expert cache, at least this. Where memory is tight the engine makes it larger. Empty: 1.5 GB.")}
        warn=${decimal(d.cache_gb) ? null : tr("A number of GB, or empty for the default.")}>
        <${Dec} label=${tr("Expert cache")} value=${d.cache_gb} ph="1.5" unit="GB / GPU" on=${(v) => setDraft({cache_gb: v})}/><//>` : null].filter(Boolean)},
    {label: tr("Prompts and cache"), rows: [
      html`<${Row} label=${tr("KV cache")} flag="--kv-cache" changed=${ch("kv_cache")}
        hint=${tr("The attention cache's format: BF16 is the model's own, INT8 the engine's default, FP8 and FP4 smaller.")}>
        <${Seg} label=${tr("KV cache")} value=${d.kv_cache} options=${(kv ? MM3_KV : [...MM3_KV, {id: d.kv_cache, label: d.kv_cache}]).map((f) => ({id: f.id, label: f.label}))}
          onChange=${(v) => setDraft({kv_cache: v})}/><//>`,
      html`<${Row} label=${tr("KV streaming")} flag="--kv-streaming" changed=${ch("kv_streaming", "kv_resident")}
        hint=${d.parallel > 1 ? tr("Off while several answers are written at once (each conversation's whole cache is in VRAM); the choice below returns with one.")
          : tr("The attention cache lives in pinned RAM, with each layer's newest positions in VRAM. Off: all of it in VRAM.")}
        below=${d.kv_streaming ? html`<div class="mdl-sub"><span>${tr("Positions each layer keeps in VRAM (--kv-resident)")}</span>
          <${Num} label=${tr("Positions each layer keeps in VRAM")} value=${d.kv_resident} ph="32768" unit="tokens" on=${(v) => setDraft({kv_resident: v})}/></div>` : null}
        warn=${whole(d.kv_resident) ? null : tr("A whole number, or empty for 32,768.")}>
        <${Switch} label=${tr("KV streaming")} on=${d.kv_streaming} onChange=${(v) => setDraft({kv_streaming: v})}/><//>`]},
    {label: tr("Speed"), rows: [
      html`<${Row} label=${tr("CPU threads")} flag="--cpu-threads" changed=${ch("cpu_threads")} hint=${tr("The CPU lane's threads, which compute the experts VRAM does not hold. Empty: the engine picks.")}
        warn=${whole(d.cpu_threads) ? null : tr("A whole number, or empty.")}>
        <${Num} label=${tr("CPU threads")} value=${d.cpu_threads} ph=${tr("auto")} unit=${tr("threads")} on=${(v) => setDraft({cpu_threads: v})}/><//>`,
      html`<${Row} label=${tr("Prompt chunk")} flag="--chunk" changed=${ch("prefill_chunk")}
        hint=${tr("The prompt tokens one pass reads at most. Empty: the size this PC's prompt calibration picked.")}
        warn=${whole(d.prefill_chunk) ? null : tr("A whole number, or empty.")}>
        <${Num} label=${tr("Prompt chunk")} value=${d.prefill_chunk} ph=${tr("auto")} unit="tokens" on=${(v) => setDraft({prefill_chunk: v})}/><//>`,
      draftConf(d, ch, 0.7)].filter(Boolean)},
  ];
}

// CYBER-FROST's (the `maya` engine's Qwen path; pictures through maya-vision and the base model's mmproj).  Its KV cache's VRAM a context token, by format: 12 attention
// layers' K and V plus the sparse attention's pooled keys (maya.py's QWEN_KV_BYTES for FP16)
const QWEN_KV = [{id: "fp16", label: "FP16", bytes: 26112}, {id: "int8", label: "INT8", bytes: 14208},
                 {id: "k8v4", label: "K8 V4", bytes: 11328}, {id: "q4_0", label: "Q4_0", bytes: 8448}];
function qwenFields(m, d, s, ch, gpus) {
  const mtp = m.choices.draft.includes("mtp");
  const kv = QWEN_KV.find((f) => f.id === d.kv) || QWEN_KV[0];
  const ec = AutoOr({label: tr("Expert cache"), k: "expert_cache", d, dflt: 8000, unit: "experts", subLabel: tr("Experts in the GPU's pool")});
  const pf = AutoOr({label: tr("Prompt chunk"), k: "prefill", d, dflt: 4096, unit: "tokens", subLabel: tr("Tokens a prompt is read in at a time")});
  return [
    {label: tr("Model"), rows: [
      contextRow(m, d, ch, {vram: kv.bytes}, ["kv"]),
      html`<${Row} label=${tr("Images")} flag="vision" changed=${ch("images")}
        hint=${m.choices.images ? tr("Reads images in chats: the image encoder starts when a picture arrives, no VRAM kept for it. Off reads text only.")
          : tr("This model was set up without its image encoder: the setup adds it.")}>
        <${Switch} label=${tr("Images")} on=${d.images} disabled=${!m.choices.images && !d.images} onChange=${(v) => setDraft({images: v})}/><//>`,
      html`<${Row} label=${tr("Drafting")} flag="--mtp-max-t" changed=${ch("draft")}
        hint=${mtp ? tr("The model's own MTP block drafts a few tokens for the model to check: the same answer. Off keeps it loaded with its drafts off. The setup measures which is faster on this PC.")
          : tr("This pack has no MTP drafter: the setup builds it from the model file.")}>
        <${Seg} label=${tr("Drafting")} value=${d.draft} options=${[...(mtp ? [{id: "mtp", label: "MTP"}] : []), {id: "none", label: tr("Off")}]}
          onChange=${(v) => setDraft({draft: v})}/><//>`]},
    {label: tr("GPUs and memory"), rows: [gpus,
      html`<${Row} label=${tr("VRAM kept free")} flag="--vram-reserve-mib" changed=${ch("reserve_mib")}
        hint=${tr("For the driver, the desktop and other programs. Empty: 700 MiB.")}
        warn=${whole(d.reserve_mib) ? null : tr("A whole number of MiB, or empty for the default.")}>
        <${Num} label=${tr("VRAM kept free")} value=${d.reserve_mib} ph="700" unit="MiB" on=${(v) => setDraft({reserve_mib: v})}/><//>`,
      html`<${Row} label=${tr("Expert cache")} flag="--expert-cache" changed=${ch("expert_cache")}
        hint=${tr("Auto takes the VRAM left free, less the reserve. A number caps the GPU's pool at that many experts.")}
        below=${ec.below} warn=${whole(d.expert_cache) ? null : tr("A whole number, or Auto.")}>${ec.ctl}<//>`,
      html`<${Row} label=${tr("Experts in RAM")} flag="MAYA_EXPERT_RAM_GB" changed=${ch("ram_gb")}
        hint=${tr("The experts the GPU does not hold stay in RAM, the most-used first, as far as the free RAM goes; the rest are read from the SSD. A number caps that RAM.")}
        warn=${whole(d.ram_gb) ? null : tr("A whole number of GB, or empty for all the engine can take.")}>
        <${Num} label=${tr("Experts in RAM")} value=${d.ram_gb} ph=${tr("auto")} unit="GB" on=${(v) => setDraft({ram_gb: v})}/><//>`]},
    {label: tr("Prompts and cache"), rows: [
      html`<${Row} label=${tr("Prompt chunk")} flag="--prefill" changed=${ch("prefill")}
        hint=${tr("Auto takes the largest chunk the expert pool can lend.")}
        below=${pf.below} warn=${whole(d.prefill) ? null : tr("A whole number, or Auto.")}>${pf.ctl}<//>`,
      html`<${Row} label=${tr("KV cache")} flag="--kv" changed=${ch("kv")}
        hint=${tr("The attention cache's format: INT8 the setup's (the model card's own), FP16 exact, K8 V4 and Q4_0 smaller - more VRAM for experts at long contexts.")}>
        <${Seg} label=${tr("KV cache")} value=${kv.id} options=${QWEN_KV.map((f) => ({id: f.id, label: f.label}))}
          onChange=${(v) => setDraft({kv: v})}/><//>`]},
  ];
}

// the command as "--flag value" pairs, one a line; the pairs the saved command does not have are the changes
function pairs(args) {
  const out = [];
  for (let i = 0; i < args.length; i += 1) {
    const a = String(args[i]), v = args[i + 1];
    if (v != null && a.startsWith("--") && !String(v).startsWith("--")) { out.push([a, String(v)]); i += 1; } else out.push([a, null]);
  }
  return out;
}
function Command({c, saved}) {
  if (!c) return null;
  const was = new Set(pairs(saved ? saved.args : []).map((p) => p.join(" ")));
  const envWas = (saved && saved.env) || {};       // (a variable is new when the saved command has it otherwise)
  const segs = [];
  for (const [k, v] of Object.entries(c.env || {})) segs.push({t: `${k}=${v} \\\n`, cls: saved && envWas[k] !== v ? "is-new" : "c-env"});
  segs.push({t: c.exe, cls: "c-exe"});
  for (const [a, v] of pairs(c.args)) {
    const isNew = saved && !was.has([a, v].join(" "));
    segs.push({t: " \\\n  ", cls: ""}, {t: a, cls: isNew ? "is-new" : "c-flag"});
    if (v != null) segs.push({t: ` ${v}`, cls: isNew ? "is-new" : ""});
  }
  return html`<pre class="mdl-cmd">${segs.map((x) => html`<span class=${x.cls}>${x.t}</span>`)}</pre>`;
}

function Editor({m}) {
  const d = MS.draft, s = m.settings;
  if (!d) return null;
  const rows = d.envRows || Object.entries(d.env || {});
  const setEnv = (r) => setDraft({envRows: r});
  const tabs = [...fieldsOf(m, d), {label: tr("Advanced"), adv: true}];
  const tab = Math.min(MS.tab, tabs.length - 1);
  const sent = toSend(d);
  const keys = Object.keys(s);
  const changed = keys.filter((k) => !same(sent[k], s[k]));
  const envChanged = !same(sent.env, s.env);
  const bad = Object.entries(d).some(([k, v]) => (NUMS.includes(k) && !whole(v)) || (DECS.includes(k) && !decimal(v)));
  const canSave = changed.length > 0 && !bad && !MS.check && !MS.saving && !MS.restart;
  const restarts = m.running && MS.data.restarts;
  // (the running model's context size alone, on a server that reloads it in place: Settings' reload, not a restart)
  const ctxOnly = m.running && changed.length === 1 && changed[0] === "context" && !envChanged && !!S.ctxInfo && !S.reload;
  const tabDot = (t) => (t.adv ? envChanged : t.rows.some((r) => r && r.props && r.props.changed));
  const tabWarn = (t) => !t.adv && t.rows.some((r) => r && r.props && r.props.warn);
  const keysOf = (e) => {
    if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
    e.preventDefault();
    MS.tab = (tab + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    emit();
    setTimeout(() => { const b = e.currentTarget && e.currentTarget.querySelector('[aria-selected="true"]'); if (b) b.focus(); }, 0);
  };
  const id = `mdl-${m.file.replace(/[^a-z0-9]/gi, "-")}`;
  return html`<div class="mdl-edit" role="region" aria-label=${tr("Engine settings for {model}", {model: titleOf(m)})}>
    <div class="mdl-edit__head"><h3>${tr("Engine settings")}</h3><code>${m.file}</code></div>
    <div class="mdl-tabs" role="tablist" aria-label=${tr("Engine settings")} onKeyDown=${keysOf}>
      ${tabs.map((t, i) => html`<button type="button" role="tab" id=${`${id}-t${i}`} aria-controls=${`${id}-p`} aria-selected=${String(i === tab)}
        tabIndex=${i === tab ? 0 : -1} onClick=${() => { MS.tab = i; emit(); }}>${t.label}
        ${tabWarn(t) ? html`<i class="mdl-dot mdl-dot--warn" role="img" aria-label=${tr("needs a look")}></i>`
          : tabDot(t) ? html`<i class="mdl-dot" role="img" aria-label=${tr("changed")}></i>` : null}</button>`)}
    </div>
    <div class="mdl-panel" role="tabpanel" id=${`${id}-p`} aria-labelledby=${`${id}-t${tab}`}>
      ${tabs[tab].adv ? html`
        <${Row} label=${tr("Environment")} flag="env" changed=${envChanged} hint=${html`${tr("Variables the engine starts with, as the setup's")} <code>--env KEY=VALUE</code> ${tr("writes them.")}`}
          below=${html`<div class="mdl-env">${rows.map(([k, v], i) => html`<div class="mdl-env__row" key=${i}>
              <input class="input mdl-mono" aria-label=${tr("Variable name")} placeholder=${tr("NAME")} value=${k}
                onInput=${(e) => setEnv(rows.map((r, j) => (j === i ? [e.target.value.trim(), r[1]] : r)))}/>
              <span aria-hidden="true">=</span>
              <input class="input mdl-mono" aria-label=${tr("Value of {variable}", {variable: k || tr("the variable")})} placeholder=${tr("value")} value=${v}
                onInput=${(e) => setEnv(rows.map((r, j) => (j === i ? [r[0], e.target.value] : r)))}/>
              <button type="button" class="icon-btn icon-btn--sm" aria-label=${tr("Remove {variable}", {variable: k || tr("this variable")})}
                onClick=${() => setEnv(rows.filter((_, j) => j !== i))}><${Icon} name="close" sm/></button></div>`)}
            <button type="button" class="btn btn--ghost btn--sm mdl-link" onClick=${() => setEnv([...rows, ["", ""]])}><${Icon} name="plus" sm/>${tr("Add a variable")}</button></div>`}/>
        <div class="mdl-field mdl-field--last"><span class="mdl-field__text"><span class="mdl-field__name"><span>${tr("The engine's command")}</span></span>
          <span class="mdl-field__hint">${tr("What Maya runs from this config, changes in purple. A setup run again keeps these settings.")}</span></span>
          <${Command} c=${MS.preview} saved=${m.command}/></div>`
      : tabs[tab].rows}
    </div>
    ${MS.check ? html`<div class="mdl-warn mdl-warn--err" role="alert">${MS.check}</div>` : null}
    ${changed.length ? html`<p class="mdl-note">${ctxOnly
      ? tr("The model reloads with a {context} context in place (a few minutes), as Settings › Chat defaults does; {file} keeps it for the next start too.", {context:ctxfmt(sent.context),file:m.file})
      : restarts
      ? tr("Maya starts again with these settings after the answer being written; the model loads again (a few minutes). The settings before stay in {file}.prev.", {file: m.file})
      : m.running ? tr("Saved into {file}; they apply from Maya's next start (this server was not started by maya.sh or START-MAYA.bat).", {file: m.file})
      : tr("Saved into {file}; {model} starts with them when you switch to it.", {file: m.file, model: titleOf(m)})}</p>` : null}
    <div class="mdl-edit__foot">
      <span class=${`mdl-count${changed.length ? " is-on" : ""}`}>${changed.length ? tr(changed.length === 1 ? "{count} change" : "{count} changes", {count: changed.length}) : tr("No changes")}</span>
      <button type="button" class="btn btn--ghost btn--sm mdl-link" onClick=${() => { MS.draft = JSON.parse(JSON.stringify(m.setup)); setDraft({}); }}
        disabled=${same(toSend(d), toSend(m.setup))}><${Icon} name="undo" sm/>${tr("Reset to the setup's values")}</button>
      <span class="mdl-grow"></span>
      <button type="button" class="btn btn--secondary" onClick=${closeSheet}>${tr("Cancel")}</button>
      ${ctxOnly ? html`<button type="button" class="btn btn--primary" disabled=${!canSave} onClick=${() => reloadContext(sent.context)}>
          ${tr("Reload with {context}", {context:ctxfmt(sent.context)})}</button>`
        : html`<button type="button" class="btn btn--primary" disabled=${!canSave} onClick=${() => saveSettings(m)}>
        ${MS.saving ? tr("Saving…") : restarts ? tr("Save and restart") : tr("Save")}</button>`}
    </div>
  </div>`;
}

// ------------------------------------------------------------------ deleting a model (POST /api/models/delete)
// MS.del: the dialog ({m, plan: what its files are (a dry run), files: the box ticked, then: the model to switch to
// when m is the one running, busy, err, from: the button}); MS.gone: what was deleted, said at the top of the page
const bytes = (b) => (b >= 1e9 ? `${fmt(b / 1e9, b < 1e10 ? 1 : 0)} GB` : b >= 1e6 ? `${fmt(b / 1e6)} MB`
  : b >= 1e3 ? `${fmt(b / 1e3)} KB` : `${fmt(b || 0)} bytes`);
const nameOf = (file) => { const x = MS.data && MS.data.models.find((m) => m.file === file); return x ? titleOf(x) : file; };
const and = (xs) => (xs.length > 1
  ? languageChoice() === "zh-CN" ? `${xs.slice(0, -1).join("、")}和${xs[xs.length - 1]}` : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`
  : xs[0] || "");
// the models the running one can be switched to before it is deleted (the most recently used first)
const startable = () => (MS.data ? MS.data.models.filter((x) => !x.running && x.ok) : []);

// why a model's Delete is off, said beside it (null: it can be deleted). The running one: only by switching to another
// model first (Maya deletes it at the next start, once nothing runs from it); never while it is loaded.
function deleteBlock(m) {
  if (MS.restart) return tr("Maya is starting again: delete when it is back.");
  if (!m.running) return null;
  if (!MS.data.restarts) return tr("It is the model running now. This server cannot switch by itself: start Maya with another model, then delete this one.");
  if (!startable().length) return tr("It is the model running now, and no other model here can start instead: set up another one first.");
  return null;
}

async function askDelete(m) {
  if (deleteBlock(m)) return;
  const d = {m, plan: MS.plans[m.file] && !MS.plans[m.file].loading ? MS.plans[m.file] : null, files: false,
             then: m.running ? (startable()[0] || {}).file || null : null, busy: false, err: null, from: document.activeElement};
  d.files = !!(d.plan && d.plan.ok && d.plan.items.length);   // (Delete frees the disk: its files ticked, named first)
  MS.del = d;
  emit();
  if (d.plan) return;
  const p = await loadPlan(m, true);
  if (MS.del !== d) return;
  d.plan = p;
  d.files = !!(p.ok && p.items.length);
  emit();
}

function closeDelete() {
  const d = MS.del;
  if (!d || d.busy) return;
  MS.del = null;
  emit();
  if (d.from && d.from.focus && document.contains(d.from)) d.from.focus();
}

async function deleteModel() {
  const d = MS.del;
  if (!d || d.busy || !d.plan) return;
  d.busy = true;
  d.err = null;
  emit();
  try {
    const out = await api("api/models/delete", {method: "POST", body: {file: d.m.file, files: d.files, then: d.then}});
    if (out.deleting) {                              // (the running one: Maya switches, the next start deletes it)
      MS.del = null;
      MS.sheet = null;
      closeEditor();
      watchRestart({why: "switch", file: out.to, name: nameOf(out.to)});
      return;
    }
    MS.gone = {name: titleOf(d.m), files: d.files, freed: out.freed, folder: d.m.folder, users: []};
    MS.del = null;
    if (MS.sheet && MS.sheet.file === d.m.file) { MS.sheet = null; closeEditor(); }
    await loadModels();
    const h = document.querySelector(".mdl__page h1");
    if (h) { h.tabIndex = -1; h.focus(); }         // (its card is gone: the keyboard goes to the page's top)
  } catch (e) {
    d.busy = false;
    d.err = tr("It was not deleted: {error}", {error: e.message});
    emit();
  }
}

// its files, said: "Its 3 GGUF files, the pack Maya built from them, its image encoder and its DFlash2 drafter (0.7 GB)."
function filesSaid(p) {
  const parts = [];
  if (p.gguf) parts.push(p.gguf === 1 ? tr("its GGUF file") : tr("its {count} GGUF files", {count: p.gguf}));
  if (p.pack) parts.push(p.gguf ? tr("the pack Maya built from them") : tr("the pack Maya built"));
  if (p.vision) parts.push(tr("its image encoder"));
  if (p.drafter != null) parts.push(tr("its DFlash2 drafter ({size})", {size: bytes(p.drafter)}));
  if (languageChoice() === "zh-CN") return parts.length ? `${parts.join("、")}。` : "";
  const said = and(parts);
  return said ? said[0].toUpperCase() + said.slice(1) + "." : "";
}

// what deleting it deletes (each folder or file, its size) and what stays because another model uses it
const GOES = {folder: "Its folder", pack: "The pack Maya built", gguf: "A model file", vision: "Its image encoder", drafter: "Its DFlash2 drafter"};
const STAYS = {folder: "The folder", pack: "The pack", vision: "The image encoder", drafter: "The DFlash2 drafter"};
const baseName = (p) => String(p).split(/[\\/]/).filter(Boolean).pop() || String(p);
function PlanList({p}) {
  const shared = p.shared || [];
  return html`${p.items.length ? html`<ul class="mdl-files" aria-label=${tr("Deleted with it")}>${p.items.map((i) => html`<li key=${i.path}>
      <${Icon} name=${i.what === "folder" ? "folder" : "disk"} sm/>
      <span class="mdl-files__text"><span><b>${baseName(i.path)}</b> · ${Object.hasOwn(GOES, i.what) ? tr(GOES[i.what]) : i.what}</span><code>${i.path}</code></span>
      <b class="mdl-files__size">${bytes(i.bytes)}</b></li>`)}</ul>` : null}
    ${shared.length ? html`<div class="mdl-files__cap">${tr(shared.length > 1 ? "Kept: another model uses them" : "Kept: another model uses it")}</div>
      <ul class="mdl-files mdl-files--kept" aria-label=${tr("Kept, another model uses them")}>${shared.map((s) => {
        const users = and((s.users || []).map(nameOf));
        return html`<li key=${`${s.what}:${s.path}`}><${Icon} name=${s.what === "folder" ? "folder" : "disk"} sm/>
          <span class="mdl-files__text"><span><b>${s.what === "gguf" ? tr(s.files === 1 ? "{count} model file" : "{count} model files", {count:s.files}) : Object.hasOwn(STAYS, s.what) ? tr(STAYS[s.what]) : s.what}</b>
            ${" "}· ${tr((s.users || []).length > 1 ? (s.what === "gguf" && s.files > 1 ? "{models} use them too" : "{models} use it too") : (s.what === "gguf" && s.files > 1 ? "{models} uses them too" : "{models} uses it too"), {models:users})}</span><code>${s.path}</code></span>
          <span class="mdl-files__size">${tr("stays")}</span></li>`;
      })}</ul>` : null}
    ${p.linked ? html`<p class="mdl-del__note">${tr(p.linked === 1 ? "{count} file in it is a hard link of another model's copy: deleting it frees nothing, and that copy stays (not counted above)." : "{count} files in it are hard links of another model's copy: deleting them frees nothing, and that copy stays (not counted above).", {count:p.linked})}</p>` : null}
    ${(p.kept || []).map((k) => html`<p class="mdl-del__note">${k}</p>`)}`;
}

function DeleteDialog() {
  const box = useRef(null);
  useEffect(() => {   // (it destroys something: the keyboard starts on Cancel, so a stray Enter does not confirm it)
    setTimeout(() => { const b = box.current && box.current.querySelector(".dialog__actions .btn--secondary"); if (b) b.focus(); }, 0);
  }, []);
  const d = MS.del;
  if (!d) return null;
  const m = d.m, p = d.plan;
  const can = !!(p && p.ok && p.items.length);
  const files = can && d.files;
  const to = m.running ? startable() : [];
  const keys = (e) => {
    if (e.key === "Tab") trapTab(e, box.current);
    else if (e.key === "Escape") { e.stopPropagation(); closeDelete(); }
  };
  return html`<div class="dialog-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) closeDelete(); }}>
    <div ref=${box} class="dialog mdl-del" role="dialog" aria-modal="true" aria-labelledby="mdl-del-title" onKeyDown=${keys}>
      <div class="dialog__head"><h2 id="mdl-del-title">${tr("Delete {model}?", {model: titleOf(m)})}</h2>
        <button type="button" class="icon-btn dialog__close" aria-label=${tr("Close")} onClick=${closeDelete}><${Icon} name="close"/></button></div>
      <div class="dialog__body">
        <p>${tr("Maya takes it off this list: its config")} <code>${m.file}</code>${m.script ? html`${tr("and its start script")} <code>${m.script}</code>` : null}. ${tr("Your chats stay.")}</p>
        ${m.running ? html`<div class="mdl-del__to" role="radiogroup" aria-labelledby="mdl-del-to">
          <p id="mdl-del-to">${tr("It is running now, so it is not deleted while it is loaded: Maya switches to another model first (a few minutes), then deletes it:")}</p>
          ${to.map((x) => html`<button type="button" role="radio" aria-checked=${String(d.then === x.file)} key=${x.file} disabled=${d.busy}
            class=${`mdl-del__box mdl-del__box--radio${d.then === x.file ? " is-on" : ""}`} onClick=${() => { d.then = x.file; emit(); }}>
            <span class="mdl-del__check" aria-hidden="true"></span>
            <span class="mdl-del__text"><span class="mdl-del__top"><b>${titleOf(x)}</b></span><code>${x.file}</code></span></button>`)}</div>` : null}
        ${!p ? html`<p class="mdl-del__note" role="status">${tr("Looking at its files…")}</p>`
          : can ? html`<button type="button" role="checkbox" aria-checked=${String(files)} class=${`mdl-del__box${files ? " is-on" : ""}`}
              disabled=${d.busy} onClick=${() => { d.files = !d.files; emit(); }}>
              <span class="mdl-del__check" aria-hidden="true">${files ? html`<${Icon} name="check" sm/>` : null}</span>
              <span class="mdl-del__text">
                <span class="mdl-del__top"><b>${tr("Also delete its downloaded files")}</b><b class="mdl-del__size">${bytes(p.bytes)}</b></span>
                <span>${filesSaid(p)}${(p.shared || []).length ? "" : ` ${tr("No other model uses them.")}`}</span></span></button>`
          : html`<p class="mdl-del__note">${p.ok ? tr("Its files are not on this PC any more.") : tr("Its files stay: {reason}.", {reason:p.why})}</p>`}
        ${p && p.ok ? html`<${PlanList} p=${p}/>` : null}
        ${files ? html`<div class="mdl-warn">${tr(p.gguf ? "This cannot be undone. To use this model again, the setup downloads its {size} again." : "This cannot be undone. To use this model again, the setup builds its pack again.", {size:bytes(p.bytes)})}</div>` : null}
        ${d.err ? html`<div class="mdl-warn mdl-warn--err" role="alert">${d.err}</div>` : null}
      </div>
      <div class="dialog__actions">
        <button type="button" class="btn btn--secondary" disabled=${d.busy} onClick=${closeDelete}>${tr("Cancel")}</button>
        <button type="button" class="btn btn--danger" disabled=${d.busy || !p || (m.running && !d.then)} onClick=${deleteModel}>
          ${d.busy ? (m.running ? tr("Switching…") : tr("Deleting…")) : m.running ? (files ? tr("Switch, then delete it and {size}", {size: bytes(p.bytes)}) : tr("Switch, then delete it from the list")) : (files ? tr("Delete it and {size}", {size: bytes(p.bytes)}) : tr("Delete it from the list"))}</button>
      </div>
    </div></div>`;
}

// ------------------------------------------------------------------ the sheet
function Overview({m}) {
  const about = [m.backend === "hip" ? "HIP" : "CUDA", size(m.size), setUpOn(m.written)].filter(Boolean).join(" · ");
  const facts = [["Config", m.path, true], ["Start script", m.script ? `./${m.script}` : "–", !!m.script], ["Model folder", m.folder || "–", true],
                 ["Port", m.port ? String(m.port) : "8080"], ["API model name", m.model, true],
                 ["Last used", m.running ? tr("running now") : m.used ? tr(relTime(m.used * 1000, dateLocale())) : tr("not yet")]];
  return html`<div class="mdl-sheet__sec">
    ${m.missing ? html`<div class="mdl-warn">${tr("It can't start: {reason}. Run the setup again to repair it (", {reason:m.missing})}<code>./maya.sh --setup</code>).</div>` : null}
    <dl class="mdl-facts">${[["Setup", about], ...facts].map(([k, v, mono]) => html`<dt>${tr(k)}</dt><dd class=${mono ? "mdl-mono-dd" : ""}>${v}</dd>`)}</dl>
  </div>`;
}

function Storage({m}) {
  const p = MS.plans[m.file], why = deleteBlock(m), L = MS.data && MS.data.leftovers;
  const toLeft = () => {
    closeSheet();
    setTimeout(() => { const el = document.getElementById("mdl-left"); if (el) { el.scrollIntoView({block: "start", behavior: "smooth"}); el.focus(); } }, 0);
  };
  return html`<div class="mdl-sheet__sec">
    ${!p || p.loading ? html`<p class="mdl-del__note" role="status">${tr("Looking at its files…")}</p>`
      : !p.ok ? html`<p class="mdl-del__note">${tr("Deleting it takes it off the list only. Its files stay: {reason}.", {reason:p.why})}</p>`
      : html`<p class="mdl-sheet__lead">${p.items.length > 1 ? html`${tr("Deleting it with its files frees ")}<b>${bytes(p.bytes)}</b>${tr(" on the disk:")}`
          : p.items.length ? tr("Deleting it with its files frees this on the disk:")
          : tr("Its files are not on this PC any more: deleting it takes it off the list.")}</p>
        <${PlanList} p=${p}/>`}
    <div class="mdl-sheet__del">
      <button type="button" class="btn btn--danger" aria-haspopup="dialog" disabled=${!!why} aria-describedby=${why ? "mdl-del-why" : undefined}
        onClick=${() => askDelete(m)}><${Icon} name="trash" sm/>${tr("Delete {model}", {model:titleOf(m)})}</button>
      ${why ? html`<span id="mdl-del-why" class="mdl-del__note">${why}</span>` : null}
    </div>
    ${L && L.items && L.items.length ? html`<button type="button" class="btn btn--ghost btn--sm mdl-link" onClick=${toLeft}>
      ${tr("Files no model uses: {size}", {size:bytes(L.bytes)})}<${Icon} name="chevron-right" sm/></button>` : null}
  </div>`;
}

function Sheet() {
  const box = useRef(null);
  const s = MS.sheet, m = s && MS.data && MS.data.models.find((x) => x.file === s.file);
  useEffect(() => {
    setTimeout(() => { const b = box.current && box.current.querySelector('.mdl-sheet__tabs [aria-selected="true"]'); if (b) b.focus(); }, 0);
  }, []);
  if (!m) return null;
  const tabs = SHEET_TABS.map((t) => ({...t, off: t.id === "engine" && !m.ok}));
  const keys = (e) => {
    if (MS.del) return;                              // (the delete dialog over it has the keyboard)
    if (e.key === "Tab") trapTab(e, box.current);
    else if (e.key === "Escape" && !e.defaultPrevented) { e.stopPropagation(); closeSheet(); }
  };
  const tabKeys = (e) => {
    if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
    e.preventDefault();
    const on = tabs.filter((t) => !t.off), i = on.findIndex((t) => t.id === s.tab);
    sheetTab(m, on[(i + (e.key === "ArrowRight" ? 1 : -1) + on.length) % on.length].id);
    setTimeout(() => { const b = box.current && box.current.querySelector('.mdl-sheet__tabs [aria-selected="true"]'); if (b) b.focus(); }, 0);
  };
  const r = MS.restart, stopping = m.running && r && r.why === "switch", restarting = m.running && r && r.why === "settings";
  const pill = m.running ? html`<span class=${`pill pill--${stopping ? "warn" : restarting || MS.loadBusy ? "info" : MS.data.loaded ? "accent" : "warn"}`}>
      <i class="status__dot"></i>${tr(stopping ? "Stopping…" : restarting ? "Restarting…" : MS.loadBusy ? "Loading…" : MS.data.loaded ? "Running" : "Unloaded")}</span>`
    : !m.ok ? html`<span class="pill pill--warn">${tr(m.missing && m.missing.startsWith("its model files") ? "Files missing" : "Incomplete")}</span>` : null;
  return html`<div class="dialog-scrim mdl-sheet-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) closeSheet(); }}>
    <div ref=${box} class="dialog mdl-sheet" role="dialog" aria-modal="true" aria-labelledby="mdl-sheet-title" onKeyDown=${keys}>
      <div class="dialog__head mdl-sheet__head"><${ModelLogo} model=${m.model} size=${40}/>
        <span class="mdl-sheet__name"><h2 id="mdl-sheet-title">${modelName(m.model)}</h2><span class="mdl-badge">${m.quant}</span></span>
        ${pill}<button type="button" class="icon-btn dialog__close" aria-label=${tr("Close")} onClick=${closeSheet}><${Icon} name="close"/></button></div>
      <div class="mdl-tabs mdl-sheet__tabs" role="tablist" aria-label=${`${titleOf(m)}`} onKeyDown=${tabKeys}>
        ${tabs.map((t) => html`<button type="button" role="tab" id=${`mdl-sheet-t-${t.id}`} aria-controls="mdl-sheet-p" aria-selected=${String(t.id === s.tab)}
          tabIndex=${t.id === s.tab ? 0 : -1} disabled=${t.off} title=${t.off ? tr("It can't start: its files are missing") : undefined}
          onClick=${() => sheetTab(m, t.id)}>${tr(t.label)}</button>`)}</div>
      <div class="dialog__body mdl-sheet__body" role="tabpanel" id="mdl-sheet-p" aria-labelledby=${`mdl-sheet-t-${s.tab}`}>
        ${s.tab === "engine" ? html`<${Editor} m=${m} key=${m.file}/>` : s.tab === "storage" ? html`<${Storage} m=${m}/>` : html`<${Overview} m=${m}/>`}
      </div>
    </div></div>`;
}

// ------------------------------------------------------------------ downloading (GET /api/catalog, POST /api/pull)
// MS.cat: {models: what the setup can download, pull: the download running (or the last one)}.  The download is the
// setup's own (curl with resume, the sha256 check, the pack, the config), run by the server: it goes on if the page closes.
const pulling = (p) => !!p && (p.state === "starting" || p.state === "downloading");
let pullTimer = null;

export async function loadCatalog() {
  try { MS.cat = await api("api/catalog"); MS.catErr = null; } catch (e) { MS.catErr = e.message; }
  emit();
  watchPull();
}

function watchPull() {
  if (pullTimer || !pulling(MS.cat && MS.cat.pull)) return;
  pullTimer = setTimeout(async () => {
    pullTimer = null;
    try {
      const was = MS.cat.pull, out = await api("api/pull/status");
      MS.cat.pull = out.pull;
      if (was && out.pull && was.state !== out.pull.state && out.pull.state === "success") {
        toast("success", tr("Download finished"), tr("{model} is on this PC and in the list.", {model: out.pull.model}), 8000);
        await loadModels();
        loadCatalog();
        return;
      }
      if (out.pull && out.pull.state === "error") toast("error", tr("The download stopped"), out.pull.error || "", 10000);
    } catch (e) { /* the server is busy or restarting: asked again */ }
    emit();
    watchPull();
  }, 2000);
}

async function startPull(c) {
  try {
    const out = await api("api/pull", {method: "POST", body: {model: c.id, detach: true}});
    MS.cat.pull = out.pull;
  } catch (e) {
    toast("error", tr("The download did not start"), e.message, 8000);
  }
  emit();
  watchPull();
}

async function cancelPull() {
  if (!(await confirmDialog(tr("Cancel the download?"), `<p>${tr("What is downloaded stays; the next download continues from it.")}</p>`, tr("Cancel the download")))) return;
  try { await api("api/pull/cancel", {method: "POST", body: {}}); } catch (e) { toast("error", tr("Not cancelled"), e.message, 6000); }
  loadCatalog();
}

function Download() {
  const C = MS.cat, p = C && C.pull, busy = pulling(p);
  const row = (c) => {
    const mine = p && p.model === c.id && (busy || p.state === "error");
    const pct = mine && p.percent != null ? p.percent : null;
    return html`<div class="mdl-dl" key=${c.id}>
      <span class="mdl-dl__name"><b>${c.name}</b><span>${[c.group === "other" && c.by ? tr("by {maker}", {maker:c.by}) : "",
        c.size_gb ? tr("about {size} GB", {size:fmt(c.size_gb, 0)}) : tr("size shown when it starts"), c.note || ""].filter(Boolean).join(" · ")}</span></span>
      ${c.installed ? html`<span class="pill pill--accent">${tr("Installed")}</span>`
        : mine && busy ? html`<button type="button" class="btn btn--ghost btn--sm" onClick=${cancelPull}>${tr("Cancel")}</button>`
        : html`<button type="button" class="btn btn--secondary btn--sm" disabled=${busy} onClick=${() => startPull(c)}><${Icon} name="download" sm/>${tr("Download")}</button>`}
      ${mine ? html`<div class="mdl-dl__bar" role=${busy ? "progressbar" : "alert"} aria-valuemin="0" aria-valuemax="100" aria-valuenow=${pct == null ? undefined : pct}
          aria-label=${tr("Downloading {model}", {model: c.name})}><i style=${`width:${pct == null ? 4 : pct}%`}></i></div>
        <span class="mdl-dl__txt">${p.state === "error" ? tr("It stopped: {error}", {error: p.error})
          : p.total ? tr("{done} of {total}{percent}{speed}", {done: bytes(p.completed), total: bytes(p.total), percent: pct != null ? ` (${pct}%)` : "", speed: p.bytes_per_s ? ` · ${fmt(p.bytes_per_s / 1e6, 0)} MB/s` : ""})
          : p.message || tr("The setup is starting…")}</span>` : null}
    </div>`;
  };
  return html`<section class="card mdl-add" aria-label=${tr("Download a model")}>
    <b>${tr("Download a model")}</b>
    <span>${tr("The setup's own download: it resumes where it stopped, checks every file's sha256, then builds the pack and writes the model's config, which appears in the list.")}</span>
    ${MS.catErr ? html`<div class="mdl-warn mdl-warn--err" role="alert">${MS.catErr}</div>` : !C ? html`<span role="status">${tr("Reading the catalog…")}</span>`
      : html`${C.models.filter((c) => c.group !== "other").map(row)}
        ${C.models.some((c) => c.group === "other") ? html`<b class="mdl-add__group">${tr("Other quants")}</b>
          <span>${tr("Files by other makers that the setup also downloads and runs: the same download, checks and pack.")}</span>
          ${C.models.filter((c) => c.group === "other").map(row)}` : null}`}
  </section>`;
}

// ------------------------------------------------------------------ files no model uses (POST /api/models/leftovers)
async function deleteLeftover(x) {
  const what = [x.pack ? tr("a pack Maya built") : null, x.hub ? tr("a download from Hugging Face") : null].filter(Boolean);
  const ok = await confirmDialog(tr("Delete {name}?", {name: x.name}),
    `<p>${tr("Maya deletes")} ${tr(x.folder ? "this folder and everything in it" : "this file")}: <b>${esc(bytes(x.bytes))}</b> ${tr("at")} <code>${esc(x.path)}</code>.</p><p>${what.length ? `${tr("It holds")} ${esc(and(what))}. ` : `${tr("Maya's setup may not have made it.")} `}${tr("No model on this list runs from it. This cannot be undone.")}</p>`, tr("Delete {size}", {size: bytes(x.bytes)}), true);
  if (!ok) return;
  try {
    const out = await api("api/models/leftovers", {method: "POST", body: {path: x.path}});
    toast("success", tr("{name} is deleted", {name: x.name}), tr("{size} freed on the disk.", {size: bytes(out.freed)}), 6000);
  } catch (e) {
    toast("error", tr("{name} was not deleted", {name: x.name}), e.message, 10000);
  }
  loadModels();
}

function Leftovers({L}) {
  if (!L || !L.items || !L.items.length) return null;
  return html`<section class="card mdl-left" id="mdl-left" tabIndex="-1" aria-labelledby="mdl-left-title">
    <div class="mdl-left__head"><b id="mdl-left-title">${tr("Files no model uses")}</b><b class="mdl-del__size">${bytes(L.bytes)}</b></div>
    <span class="mdl-left__sub">${tr("In {folders}, and no model on this list runs from them: a model deleted by hand, or set up again somewhere else. Maya may not have downloaded all of them, so look before you delete.", {folders: and(L.folders)})}</span>
    <ul class="mdl-left__list">${L.items.map((x) => html`<li key=${x.path}>
      <span class="mdl-left__text"><span class="mdl-left__name"><${Icon} name=${x.folder ? "folder" : "disk"} sm/><b>${x.name}</b>
        ${x.pack ? html`<span class="mdl-badge">${tr("Maya pack")}</span>` : null}${x.hub ? html`<span class="mdl-badge">Hugging Face</span>` : null}</span>
        <code>${x.path}</code><span>${bytes(x.bytes)} · ${tr("changed")} ${tr(relTime(x.modified * 1000, dateLocale()))}</span></span>
      <button type="button" class="btn btn--ghost btn--sm" aria-label=${tr("Delete {name}", {name: x.name})} disabled=${!!MS.restart}
        onClick=${() => deleteLeftover(x)}><${Icon} name="trash" sm/>${tr("Delete")}</button></li>`)}</ul>
  </section>`;
}

// ------------------------------------------------------------------ a model's card
function Card({m}) {
  const s = m.settings, r = MS.restart, sheet = MS.sheet && MS.sheet.file === m.file ? MS.sheet.tab : null;
  const stopping = m.running && r && r.why === "switch", restarting = m.running && r && r.why === "settings";
  const loading = r && r.why === "switch" && r.file === m.file;
  const chips = [[tr("Context"), ctxfmt(s.context)], [tr("Draft"), DRAFTS[s.draft] ? tr(DRAFTS[s.draft]) : s.draft], [tr("GPUs"), s.gpus.join(", ") || "–"],
                 [tr("Images"), s.images ? tr("on") : tr("off")], ...(s.parallel > 1 ? [[tr("At once"), String(s.parallel)]] : [])];
  const about = [m.backend === "hip" ? "HIP" : "CUDA", size(m.size), setUpOn(m.written)].filter(Boolean).join(" · ");
  const why = deleteBlock(m);
  return html`<section class=${`card mdl-card${m.running ? " is-running" : ""}${loading ? " is-loading" : ""}`} aria-label=${titleOf(m)}>
    <div class="mdl-card__head">
      <${ModelLogo} model=${m.model} size=${44}/>
      <span class="mdl-card__name"><span><b>${modelName(m.model)}</b><span class="mdl-badge">${m.quant}</span></span>
        <span class="mdl-card__about">${about}</span></span>
      ${m.running ? html`<span class=${`pill pill--${stopping ? "warn" : restarting || MS.loadBusy ? "info" : MS.data.loaded ? "accent" : "warn"}`} role="status">
          <i class=${`status__dot${stopping || restarting || MS.loadBusy ? " is-pulse" : ""}`}></i>${stopping ? tr("Stopping…") : restarting ? tr("Restarting…")
            : MS.loadBusy ? tr("Loading…") : MS.data.loaded ? tr("Running") : tr("Unloaded")}</span>`
        : loading ? html`<span class="pill pill--info" role="status"><i class="status__dot is-pulse"></i>${tr("Loading…")}</span>`
        : !m.ok ? html`<span class="pill pill--warn" title=${m.missing || ""}>${m.missing && m.missing.startsWith("its model files") ? tr("Files missing") : tr("Incomplete")}</span>`
        : html`<button type="button" class="btn btn--secondary btn--sm" disabled=${!!r} onClick=${() => switchTo(m)}><${Icon} name="swap" sm/>${tr("Switch to this")}</button>`}
    </div>
    <div class="mdl-chips">${chips.map(([k, v]) => html`<span class="mdl-chip"><span>${tr(k)}</span><b>${v}</b></span>`)}</div>
    ${m.missing && !m.running ? html`<div class="mdl-warn">${tr("It can't start: {reason}. Run the setup again to repair it (", {reason:m.missing})}<code>./maya.sh --setup</code>).</div>` : null}
    <div class="mdl-card__foot">
      <span class="mdl-card__file"><code>${m.file}</code><span>·</span><span>${m.running ? tr("running now") : m.used ? tr("last used {time}", {time:tr(relTime(m.used*1000,dateLocale()))}) : ""}</span></span>
      <span class="mdl-card__acts">
        ${m.running ? html`<button type="button" class="btn btn--ghost btn--sm" onClick=${() => showView("monitor")}><${Icon} name="activity" sm/>${tr("Monitor")}</button>` : null}
        ${m.running && MS.data.can_unload && !r ? html`<button type="button" class="btn btn--ghost btn--sm" disabled=${!!MS.loadBusy}
          onClick=${() => loadOrUnload(MS.data.loaded ? "unload" : "load")}><${Icon} name=${MS.data.loaded ? "stop" : "refresh"} sm/>${MS.data.loaded ? tr("Unload") : tr("Load")}</button>` : null}
        <button type="button" class=${`btn btn--ghost btn--sm${sheet === "engine" ? " is-on" : ""}`} aria-haspopup="dialog" disabled=${!m.ok}
          onClick=${() => openSheet(m, "engine")}><${Icon} name="sliders" sm/>${tr("Engine settings")}</button>
        <button type="button" class=${`btn btn--ghost btn--sm${sheet === "overview" ? " is-on" : ""}`} aria-haspopup="dialog"
          onClick=${() => openSheet(m, "overview")}><${Icon} name="info" sm/>${tr("Details")}</button>
        <button type="button" class=${`btn btn--ghost btn--sm${sheet === "storage" ? " is-on" : ""}`} aria-haspopup="dialog"
          onClick=${() => openSheet(m, "storage")}><${Icon} name="disk" sm/>${tr("Storage")}</button>
        <button type="button" class="btn btn--ghost btn--sm mdl-card__del" aria-haspopup="dialog" aria-label=${tr("Delete {model}", {model:titleOf(m)})}
          disabled=${!!why} title=${why || tr("Delete {model} and free its disk space", {model:titleOf(m)})} onClick=${() => askDelete(m)}><${Icon} name="trash" sm/>${tr("Delete")}</button>
      </span>
    </div>
  </section>`;
}

// ------------------------------------------------------------------ the page
export function ModelsPage() {
  useEffect(() => {
    loadModels();
    const again = () => loadModels();             // (a context size reloaded from Settings: the configs say it now)
    window.addEventListener("maya:context", again);
    return () => window.removeEventListener("maya:context", again);
  }, []);
  const phone = isPhone(), D = MS.data, r = MS.restart;
  const cmds = [[tr("Linux and macOS"), "./maya.sh --setup"], ["Windows", "START-MAYA.bat --setup"]];
  return html`<main class="mon mdl">
    ${phone ? html`<header class="chat-head chat-head--float">
        <button type="button" class="glass-btn" data-open-chats aria-label=${tr("Chats")} onClick=${() => { S.drawer = true; emit(); }}><${Icon} name="menu"/></button>
        <span class="model-btn model-btn--phone">${tr("Models")}</span>
        <button type="button" class="glass-btn" aria-label=${tr("New chat")} onClick=${newChat}><${Icon} name="new-chat"/></button></header>`
      : !S.sideOpen ? html`<header class="chat-head"><div class="chat-head__left">
        <button type="button" class="icon-btn" aria-label=${tr("Open the sidebar")} title=${tr("Open the sidebar")} onClick=${() => setSide(true)}><${Icon} name="panel-left"/></button>
        <button type="button" class="icon-btn" aria-label=${tr("New chat")} title=${tr("New chat")} onClick=${newChat}><${Icon} name="new-chat"/></button></div></header>` : null}
    <div class="mon__scroll"><div class="mon__page mdl__page">
      <div class="mon__head">
        <div><h1>${tr("Models")}</h1><p>${tr("The models set up on this machine. Maya runs one at a time.")}</p></div>
        <div class="mon__acts"><button type="button" class=${`btn btn--secondary btn--sm${MS.add ? " is-on" : ""}`} aria-expanded=${String(MS.add)}
          onClick=${() => { MS.add = !MS.add; emit(); if (MS.add) loadCatalog(); }}><${Icon} name="plus" sm/>${tr("Add a model")}</button></div>
      </div>
      ${MS.add || pulling(D && D.pull) ? html`<${Download}/>` : null}
      ${MS.add ? html`<section class="card mdl-add" aria-label=${tr("Run the setup yourself")}>
        <b>${tr("Or run the setup yourself")}</b>
        <span>${tr("It asks which model and quant, shows the download's size first, then builds the model's pack and writes its config, which appears here.")}</span>
        ${cmds.map(([os, c]) => html`<div class="mdl-add__cmd"><span class="mdl-add__os">${tr(os)}</span><code>${c}</code>
          <button type="button" class="icon-btn icon-btn--sm" aria-label=${tr("Copy {command}", {command:c})} onClick=${() => copyText(c)}><${Icon} name="copy" sm/></button></div>`)}
      </section>` : null}
      ${r ? html`<div class="callout callout--info" role="status"><${Icon} name="info" sm/><span>
          <b>${r.why === "switch" ? tr("Switching to {model}.", {model:r.name}) : tr("Restarting {model} with the new engine settings.", {model:r.name})}</b>
          ${" "}${tr("Maya finishes the answer it is writing, then loads the model (a few minutes). This page reloads when it is back.")}</span></div>` : null}
      ${MS.gone ? html`<div class="callout callout--ok" role="status"><${Icon} name="check" sm/><span>
          <b>${tr("{model} is deleted.", {model:MS.gone.name})}</b>${" "}${MS.gone.files ? tr("{bytes} freed on the disk.", {bytes:bytes(MS.gone.freed)})
            : MS.gone.folder ? tr("Its files stay in {folder}.", {folder:MS.gone.folder}) : ""}${MS.gone.users.length
            ? tr(MS.gone.users.length>1 ? " {models} draft with MTP now." : " {models} drafts with MTP now.", {models:and(MS.gone.users)}) : ""}</span>
          <button type="button" class="icon-btn icon-btn--sm" aria-label=${tr("Dismiss")} onClick=${() => { MS.gone = null; emit(); }}><${Icon} name="close" sm/></button></div>` : null}
      ${MS.err && !D ? html`<div class="card mon__wait">${MS.err}</div>`
        : !D ? html`<div class="card mon__wait">${tr("Reading the models…")}</div>`
        : !D.models.length ? html`<div class="card mon__wait">${tr("No model is set up in this folder yet. The setup writes one: ")}<code>./maya.sh --setup</code> (Windows: <code>START-MAYA.bat --setup</code>)</div>`
        : D.models.map((m) => html`<${Card} m=${m} key=${m.file}/>`)}
      ${D ? html`<${Leftovers} L=${D.leftovers}/>` : null}
      ${D && D.models.length ? html`<p class="mdl-foot"><${Icon} name="info" sm/><span>${D.restarts
        ? tr("Switching, or saving the running model's engine settings, starts Maya again through its start script, as a port change does. Each model keeps its own chat settings.")
        : tr("This server was not started by maya.sh or START-MAYA.bat, so it cannot start again by itself: a switch or new settings apply from Maya's next start.")}</span></p>` : null}
    </div></div>
    ${MS.sheet ? html`<${Sheet} key=${MS.sheet.file}/>` : null}
    ${MS.del ? html`<${DeleteDialog} key=${MS.del.m.file}/>` : null}
  </main>`;
}
