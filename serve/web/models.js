// serve/web/models.js - the Models page: the models set up on this machine (the run configs the setup wrote, GET
// /api/models), the running one marked; switching to another (POST /api/models/switch: Maya starts again with it) and
// each one's engine settings, edited in its card (POST /api/models/settings - a dry run first for the command they make).
import {html, useEffect, useRef} from "./preact.js";
import {api, ctxfmt, esc, fmt, isPhone, modelName, relTime} from "./lib.js";
import {S, confirmDialog, copyText, emit, toast} from "./state.js";
import {newChat} from "./chats.js";
import {setSide, showView} from "./sidebar.js";
import {Icon, ModelLogo, Seg, Switch} from "./ui.js";

// the page's own state (beside S: nothing else reads it but the model menu's count)
export const MS = {data: null, err: null, add: false, details: {}, edit: null, draft: null, tab: 0, preview: null,
                   check: "", saving: false, restart: null};

export async function loadModels() {
  try {
    MS.data = await api("api/models");
    MS.err = null;
  } catch (e) {
    MS.err = e.message;
  }
  emit();
  return MS.data;
}

const MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
const titleOf = (m) => `${modelName(m.model)} ${m.quant}`;
const DRAFTS = {mtp: "MTP", dflash2: "DFlash2", dspark: "DSpark", none: "off"};
const size = (b) => (b >= 1e8 ? `${fmt(b / 1e9, b < 1e10 ? 1 : 0)} GB` : null);
const setUpOn = (w) => {
  const d = w ? new Date(String(w).replace(" ", "T")) : null;
  return d && !Number.isNaN(d.getTime()) ? `set up ${d.toLocaleDateString([], {month: "short", day: "numeric"})}` : null;
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
      toast("success", r.why === "switch" ? `${r.name} is running` : "Maya is back", "Reloading the page…", 3000);
      setTimeout(() => location.reload(), 900);
      return;
    }
  }
  MS.restart = null;
  emit();
  toast("error", "Maya is not back yet", "The terminal it runs in shows why. This page reconnects when it is.", 30000);
}

async function switchTo(m) {
  const run = (MS.data.models || []).find((x) => x.running);
  const restarts = MS.data.restarts;
  const ok = await confirmDialog(`Switch to ${titleOf(m)}?`,
    (restarts
      ? `<p>Maya stops <b>${esc(run ? titleOf(run) : "the model running now")}</b> and loads <b>${esc(titleOf(m))}</b> from ` +
        `<code>${esc(m.file)}</code>. That takes a few minutes.</p><ul>` +
        "<li>Your chats stay here. Each model keeps its own chat settings.</li>" +
        `<li>Apps using the API get a “try again” until it is back, then the model name <code>${esc(m.model)}</code>.</li>` +
        (S.busy ? "<li>The answer being written finishes first.</li>" : "") + "</ul>"
      : `<p><b>${esc(titleOf(m))}</b> becomes the model Maya starts next. This server was not started by maya.sh or ` +
        "START-MAYA.bat, so it cannot start again by itself: stop it and start Maya again.</p>"),
    restarts ? "Switch model" : "Start it next");
  if (!ok) return;
  try {
    const out = await api("api/models/switch", {method: "POST", body: {file: m.file}});
    if (!out.restarts) {
      toast("success", "Saved as the next model", `${titleOf(m)} starts the next time Maya starts (./maya.sh or START-MAYA.bat).`, 8000);
      loadModels();
      return;
    }
  } catch (e) {
    toast("error", "The model did not change", e.message, 8000);
    return;
  }
  watchRestart({why: "switch", file: m.file, name: titleOf(m)});
}

// ------------------------------------------------------------------ the engine settings (one card's, as a draft)
function openEditor(m) {
  if (MS.edit === m.file) { closeEditor(); return; }
  MS.edit = m.file;
  MS.draft = JSON.parse(JSON.stringify(m.settings));
  MS.tab = 0;
  MS.preview = m.command;
  MS.check = "";
  emit();
}
function closeEditor() {
  MS.edit = MS.draft = MS.preview = null;
  MS.check = "";
  emit();
}

// the draft as the server takes it: "" -> the default (null), a number typed -> a number
const NUMS = ["context", "parallel", "reserve_mib", "expert_cache", "prefill", "kv_resident", "tp", "prefill_chunk", "cpu_threads"];
function toSend(d) {
  const out = {...d};
  for (const k of NUMS) {
    if (!(k in out)) continue;
    const v = out[k];
    out[k] = v === "" || v == null ? null : /^\s*\d+\s*$/.test(String(v)) ? Number(v) : v;
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
  MS.saving = true;
  emit();
  try {
    const out = await api("api/models/settings", {method: "POST", body: {file: m.file, settings: toSend(MS.draft)}});
    closeEditor();
    if (out.restarts) { watchRestart({why: "settings", file: m.file, name: titleOf(m)}); return; }
    toast("success", "Engine settings saved", m.running
      ? `Saved into ${m.file}. They apply from Maya's next start (./maya.sh or START-MAYA.bat starts it again by itself).`
      : `Saved into ${m.file}. ${titleOf(m)} starts with them when you switch to it.`, 8000);
    loadModels();
  } catch (e) {
    MS.check = e.message;
    toast("error", "Not saved", e.message, 8000);
  } finally {
    MS.saving = false;
    emit();
  }
}

const whole = (v) => v === "" || v == null || /^\s*\d+\s*$/.test(String(v));

function Row({label, flag, hint, changed, children, below, warn}) {
  return html`<div class="mdl-field">
    <div class="mdl-field__top">
      <span class="mdl-field__text"><span class="mdl-field__name"><span>${label}</span>${flag ? html`<code>${flag}</code>` : null}
        ${changed ? html`<i class="mdl-dot" role="img" aria-label="changed" title="Changed, not saved yet"></i>` : null}</span>
        ${hint ? html`<span class="mdl-field__hint">${hint}</span>` : null}</span>
      ${children ? html`<span class="mdl-field__ctl">${children}</span>` : null}
    </div>
    ${below}
    ${warn ? html`<div class="mdl-warn">${warn}</div>` : null}
  </div>`;
}
const Num = ({label, value, ph, unit, on}) => html`<span class="mdl-num">
  <input class=${`input input--num${whole(value) ? "" : " is-bad"}`} inputmode="numeric" aria-label=${label} value=${value ?? ""}
    placeholder=${ph} onInput=${(e) => on(e.target.value)}/>${unit ? html`<span>${unit}</span>` : null}</span>`;
function Stepper({label, value, min, max, on}) {
  const v = Number(value) || min;
  return html`<span class="mdl-step" role="group" aria-label=${label}>
    <button type="button" aria-label="Fewer" disabled=${v <= min} onClick=${() => on(v - 1)}><${Icon} name="minus" sm/></button>
    <b aria-live="polite">${v}</b>
    <button type="button" aria-label="More" disabled=${v >= max} onClick=${() => on(v + 1)}><${Icon} name="plus" sm/></button></span>`;
}
// a choice of "auto" and a number of one's own (the expert cache, the prompt chunk)
function AutoOr({label, k, d, dflt, unit, subLabel}) {
  const set = d[k] != null && d[k] !== "";
  return {ctl: html`<${Seg} label=${label} value=${set ? "set" : "auto"} options=${[{id: "auto", label: "Auto"}, {id: "set", label: "Set"}]}
      onChange=${(v) => setDraft({[k]: v === "auto" ? null : String(dflt)})}/>`,
    below: set ? html`<div class="mdl-sub"><span>${subLabel}</span><${Num} label=${subLabel} value=${d[k]} ph=${String(dflt)} unit=${unit}
      on=${(v) => setDraft({[k]: v})}/></div>` : null};
}

function fieldsOf(m, d) {
  const s = m.settings, ch = (...ks) => ks.some((k) => !same(d[k], s[k])), glm = m.family === "glm";
  const ctxs = [...new Set([...(m.contexts || MS.data.contexts || []), s.context, d.context].map(Number))].sort((a, b) => a - b);
  const context = html`<${Row} label="Context size" flag="--max-context" changed=${ch("context")}
    hint=${glm && d.kv_streaming ? (d.context <= (Number(d.kv_resident) || 32768) ? `Up to ${ctxfmt(Number(d.kv_resident) || 32768)} the whole cache stays in VRAM.`
      : `Above ${ctxfmt(Number(d.kv_resident) || 32768)} the cache streams from RAM: about ${fmt((d.context * 12288) / 1e9, 1)} GB of RAM at ${ctxfmt(d.context)}.`)
      : "More context takes GPU memory from the experts."}>
    <${Seg} label="Context size" value=${Number(d.context)} options=${ctxs.map((n) => ({id: n, label: ctxfmt(n)}))}
      onChange=${(v) => setDraft({context: v})}/><//>`;
  const images = html`<${Row} label="Images" flag=${glm ? "vision" : "--vision"} changed=${ch("images")}
    hint=${m.choices.images ? "Reads images in chats. Off leaves its memory to the experts."
      : glm ? "This model was set up without its image encoder: the setup adds it." : "This pack has no vision tower."}>
    <${Switch} label="Images" on=${d.images} disabled=${!m.choices.images && !d.images} onChange=${(v) => setDraft({images: v})}/><//>`;
  const gpus = html`<${Row} label="GPUs" flag="gpu" changed=${ch("gpus")} hint="The engine splits the model across the GPUs picked. At least one."
    below=${html`<div class="mdl-gpus">${(MS.data.gpus || []).map((g) => {
      const on = d.gpus.includes(g.index);
      return html`<button type="button" role="checkbox" aria-checked=${String(on)} class="mdl-gpu"
        onClick=${() => { if (on && d.gpus.length === 1) return; setDraft({gpus: on ? d.gpus.filter((x) => x !== g.index) : [...d.gpus, g.index].sort((a, b) => a - b)}); }}>
        <span class="mdl-gpu__box">${on ? html`<${Icon} name="check" sm/>` : null}</span>
        <span><b>GPU ${g.index}</b><span>${[shortGpu(g.name), g.vram_gb ? `${fmt(g.vram_gb)} GB` : null].filter(Boolean).join(" · ") || "on this PC"}</span></span></button>`;
    })}</div>`}/>`;
  const reserve = html`<${Row} label="VRAM kept free" flag=${glm ? "--vram-reserve-mib" : "--reserve-mib"} changed=${ch("reserve_mib")}
    hint=${glm ? "For the driver, the desktop and other programs. Empty: 700 MiB." : "Per GPU, for the driver, the desktop and other programs. Empty: the engine's own."}
    warn=${whole(d.reserve_mib) ? null : "A whole number of MiB, or empty for the default."}>
    <${Num} label="VRAM kept free" value=${d.reserve_mib} ph=${glm ? "700" : "default"} unit="MiB / GPU" on=${(v) => setDraft({reserve_mib: v})}/><//>`;
  if (m.family === "minimax") return mm3Fields(m, d, s, ch, ctxs, images, gpus);
  if (glm) {
    const ec = AutoOr({label: "Expert cache", k: "expert_cache", d, dflt: 1200, unit: "experts", subLabel: "Experts in the first GPU's pool"});
    const pf = AutoOr({label: "Prompt chunk", k: "prefill", d, dflt: 8192, unit: "tokens", subLabel: "Tokens a prompt is read in at a time"});
    const dflash = m.choices.draft.includes("dflash2");
    return [
      {label: "Model", rows: [context,
        html`<${Row} label="Answers at once" flag="--parallel" changed=${ch("parallel")}
          hint="Up to 8. Each extra conversation keeps its own context in VRAM, which the expert cache gives up.">
          <${Stepper} label="Answers at once" value=${d.parallel} min=${1} max=${8} on=${(v) => setDraft({parallel: v})}/><//>`,
        html`<${Row} label="Drafting" flag="--dflash" changed=${ch("draft")}
          hint=${dflash ? "MTP is built into the model. DFlash2 is a separate drafter, often faster for one answer at a time."
            : "MTP is built into the model. DFlash2 needs its drafter: the setup's --draft dflash2 downloads it (0.7 GB)."}
          warn=${d.draft === "dflash2" && d.parallel > 1 ? `DFlash2 drafts for one answer at a time: with ${d.parallel} at once the engine drafts with MTP.` : null}>
          <${Seg} label="Drafting" value=${d.draft} options=${[{id: "mtp", label: "MTP"}, ...(dflash ? [{id: "dflash2", label: "DFlash2"}] : [])]}
            onChange=${(v) => setDraft({draft: v})}/><//>`,
        images]},
      {label: "GPUs and memory", rows: [gpus, reserve,
        html`<${Row} label="Expert cache" flag="--expert-cache" changed=${ch("expert_cache")}
          hint="Auto takes the VRAM left free, less the reserve. A number caps the first GPU's pool at that many experts."
          below=${ec.below} warn=${whole(d.expert_cache) ? null : "A whole number, or Auto."}>${ec.ctl}<//>`]},
      {label: "Prompts and cache", rows: [
        html`<${Row} label="Prompt chunk" flag="--prefill" changed=${ch("prefill")}
          hint="Auto takes the largest chunk the expert pool can lend, up to 8,192 tokens."
          below=${pf.below} warn=${whole(d.prefill) ? null : "A whole number, or Auto."}>${pf.ctl}<//>`,
        html`<${Row} label="KV streaming" flag="--kv-streaming" changed=${ch("kv_streaming", "kv_resident")}
          hint="The attention cache lives in RAM, 12 KB a token, with a window of it in VRAM. Off: all of it in VRAM."
          below=${d.kv_streaming ? html`<div class="mdl-sub"><span>Positions each layer keeps in VRAM (--kv-resident)</span>
            <${Num} label="Positions each layer keeps in VRAM" value=${d.kv_resident} ph="32768" unit="tokens" on=${(v) => setDraft({kv_resident: v})}/></div>` : null}
          warn=${whole(d.kv_resident) ? null : "A whole number, or empty for 32,768."}>
          <${Switch} label="KV streaming" on=${d.kv_streaming} onChange=${(v) => setDraft({kv_streaming: v})}/><//>`]},
    ];
  }
  const dspark = m.choices.draft.includes("dspark");
  return [
    {label: "Model", rows: [context,
      html`<${Row} label="Drafting" flag="--dspark" changed=${ch("draft")}
        hint=${dspark ? "DSpark drafts a few tokens at a time for the model to check. The speed calibration picks the faster on this PC."
          : "This pack has no DSpark drafter."}>
        <${Seg} label="Drafting" value=${d.draft} options=${[...(dspark ? [{id: "dspark", label: "DSpark"}] : []), {id: "none", label: "Off"}]}
          onChange=${(v) => setDraft({draft: v})}/><//>`,
      images]},
    {label: "GPUs and memory", rows: [gpus,
      html`<${Row} label="Tensor parallel" flag="--tp" changed=${ch("tp")}
        hint="How many GPUs share each layer: equal GPUs, in groups of this size. On one GPU, 1. The calibration (--calibrate) picks the faster on this PC."
        warn=${Number(d.tp) === 2 && d.gpus.length < 2 ? "With one GPU picked, 2 runs its layers alone and reads prompts slower: use 1."
          : Number(d.tp) > Math.max(2, d.gpus.length) ? `A group of ${d.tp} needs at least ${d.tp} GPUs picked.` : null}>
        <${Seg} label="Tensor parallel" value=${Number(d.tp)} options=${tpChoices(d.gpus.length, s.tp, d.tp).map((n) => ({id: n, label: String(n)}))}
          onChange=${(v) => setDraft({tp: v})}/><//>`,
      reserve,
      html`<${Row} label="Experts VRAM does not hold" flag="--mmap-experts" changed=${ch("mmap")}
        hint="Auto copies them into RAM when it holds them, else reads them from the SSD in place.">
        <${Seg} label="Experts VRAM does not hold" value=${d.mmap} options=${[{id: "auto", label: "Auto"}, {id: "on", label: "From the files"}, {id: "off", label: "In RAM"}]}
          onChange=${(v) => setDraft({mmap: v})}/><//>`]},
    {label: "Speed", rows: [
      html`<${Row} label="Prompt chunk" flag="--prefill-chunk" changed=${ch("prefill_chunk")} hint="Empty: the engine picks."
        warn=${whole(d.prefill_chunk) ? null : "A whole number, or empty."}>
        <${Num} label="Prompt chunk" value=${d.prefill_chunk} ph="auto" unit="tokens" on=${(v) => setDraft({prefill_chunk: v})}/><//>`,
      html`<${Row} label="CPU threads" flag="--cpu-threads" changed=${ch("cpu_threads")} hint="Set by the calibration (--calibrate) on this PC. Empty: the engine picks."
        warn=${whole(d.cpu_threads) ? null : "A whole number, or empty."}>
        <${Num} label="CPU threads" value=${d.cpu_threads} ph="auto" unit="threads" on=${(v) => setDraft({cpu_threads: v})}/><//>`]},
  ];
}

// MiniMax's (maya-mm3: it stops at a flag it does not know, so only its own are offered).  Its KV cache's pinned RAM a
// context token, by format (maya.py's MM3_KV_BYTES)
const MM3_KV = [{id: "bf16", label: "BF16", bytes: 122880}, {id: "int8", label: "INT8", bytes: 63360},
                {id: "fp8", label: "FP8", bytes: 63360}, {id: "fp4", label: "FP4", bytes: 34560}];
function mm3Fields(m, d, s, ch, ctxs, images, gpus) {
  const dspark = m.choices.draft.includes("dspark"), win = Number(d.kv_resident) || 32768;
  const kv = MM3_KV.find((f) => f.id === d.kv_cache);
  const ram = kv ? `about ${fmt((d.context * kv.bytes) / 2 ** 30, 1)} GB of pinned RAM at ${ctxfmt(d.context)} (${kv.label})` : "pinned RAM";
  const tps = [...new Set([0, 1, 2, 4].filter((t) => t <= Math.max(1, d.gpus.length)).concat([Number(s.tp), Number(d.tp)]))].sort((a, b) => a - b);
  return [
    {label: "Model", rows: [
      html`<${Row} label="Context size" flag="--max-context" changed=${ch("context")}
        hint=${d.kv_streaming ? `Every position's cache sits in RAM: ${ram}, its newest ${ctxfmt(win)} in VRAM too.${d.context > 262144 ? " Past 256K it has not been run end to end." : ""}`
          : "The whole cache in VRAM: more context takes GPU memory from the experts."}>
        <${Seg} label="Context size" value=${Number(d.context)} options=${ctxs.map((n) => ({id: n, label: ctxfmt(n)}))}
          onChange=${(v) => setDraft({context: v})}/><//>`,
      html`<${Row} label="Drafting" flag="--dspark" changed=${ch("draft")}
        hint=${dspark ? "DSpark drafts a few tokens at a time for the model to check: the same answer. The speed calibration picks the faster on this PC."
          : "This pack has no DSpark drafter."}>
        <${Seg} label="Drafting" value=${d.draft} options=${[...(dspark ? [{id: "dspark", label: "DSpark"}] : []), {id: "none", label: "Off"}]}
          onChange=${(v) => setDraft({draft: v})}/><//>`,
      images]},
    {label: "GPUs and memory", rows: [gpus,
      html`<${Row} label="Tensor parallel" flag="--tp" changed=${ch("tp")}
        hint="Off splits the layers across the GPUs. 1, 2 or 4: each layer's work shared by a group of that many GPUs; the setup picks 4 or 2 when the GPUs make whole groups."
        warn=${Number(d.tp) > d.gpus.length ? `A group of ${d.tp} needs at least ${d.tp} GPUs picked.` : null}>
        <${Seg} label="Tensor parallel" value=${Number(d.tp)} options=${tps.map((n) => ({id: n, label: n ? String(n) : "Off"}))}
          onChange=${(v) => setDraft({tp: v})}/><//>`]},
    {label: "Prompts and cache", rows: [
      html`<${Row} label="KV cache" flag="--kv-cache" changed=${ch("kv_cache")}
        hint="The attention cache's format: BF16 is the model's own, INT8 the engine's default, FP8 and FP4 smaller.">
        <${Seg} label="KV cache" value=${d.kv_cache} options=${(kv ? MM3_KV : [...MM3_KV, {id: d.kv_cache, label: d.kv_cache}]).map((f) => ({id: f.id, label: f.label}))}
          onChange=${(v) => setDraft({kv_cache: v})}/><//>`,
      html`<${Row} label="KV streaming" flag="--kv-streaming" changed=${ch("kv_streaming", "kv_resident")}
        hint="The attention cache lives in pinned RAM, with each layer's newest positions in VRAM. Off: all of it in VRAM."
        below=${d.kv_streaming ? html`<div class="mdl-sub"><span>Positions each layer keeps in VRAM (--kv-resident)</span>
          <${Num} label="Positions each layer keeps in VRAM" value=${d.kv_resident} ph="32768" unit="tokens" on=${(v) => setDraft({kv_resident: v})}/></div>` : null}
        warn=${whole(d.kv_resident) ? null : "A whole number, or empty for 32,768."}>
        <${Switch} label="KV streaming" on=${d.kv_streaming} onChange=${(v) => setDraft({kv_streaming: v})}/><//>`]},
    {label: "Speed", rows: [
      html`<${Row} label="CPU threads" flag="--cpu-threads" changed=${ch("cpu_threads")} hint="The CPU lane's threads, which compute the experts VRAM does not hold. Empty: the engine picks."
        warn=${whole(d.cpu_threads) ? null : "A whole number, or empty."}>
        <${Num} label="CPU threads" value=${d.cpu_threads} ph="auto" unit="threads" on=${(v) => setDraft({cpu_threads: v})}/><//>`]},
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
  const ref = useRef(null);
  useEffect(() => { const el = ref.current; if (el && el.scrollIntoView) el.scrollIntoView({block: "nearest", behavior: "smooth"}); }, []);
  if (!d) return null;
  const rows = d.envRows || Object.entries(d.env || {});
  const setEnv = (r) => setDraft({envRows: r});
  const tabs = [...fieldsOf(m, d), {label: "Advanced", adv: true}];
  const tab = Math.min(MS.tab, tabs.length - 1);
  const sent = toSend(d);
  const keys = Object.keys(s);
  const changed = keys.filter((k) => !same(sent[k], s[k]));
  const envChanged = !same(sent.env, s.env);
  const bad = Object.entries(d).some(([k, v]) => NUMS.includes(k) && !whole(v));
  const canSave = changed.length > 0 && !bad && !MS.check && !MS.saving && !MS.restart;
  const restarts = m.running && MS.data.restarts;
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
  return html`<div class="mdl-edit" ref=${ref} role="region" aria-label=${`Engine settings for ${titleOf(m)}`}>
    <div class="mdl-edit__head"><h3>Engine settings</h3><code>${m.file}</code></div>
    <div class="mdl-tabs" role="tablist" aria-label="Engine settings" onKeyDown=${keysOf}>
      ${tabs.map((t, i) => html`<button type="button" role="tab" id=${`${id}-t${i}`} aria-controls=${`${id}-p`} aria-selected=${String(i === tab)}
        tabIndex=${i === tab ? 0 : -1} onClick=${() => { MS.tab = i; emit(); }}>${t.label}
        ${tabWarn(t) ? html`<i class="mdl-dot mdl-dot--warn" role="img" aria-label="needs a look"></i>`
          : tabDot(t) ? html`<i class="mdl-dot" role="img" aria-label="changed"></i>` : null}</button>`)}
    </div>
    <div class="mdl-panel" role="tabpanel" id=${`${id}-p`} aria-labelledby=${`${id}-t${tab}`}>
      ${tabs[tab].adv ? html`
        <${Row} label="Environment" flag="env" changed=${envChanged} hint=${html`Variables the engine starts with, as the setup's <code>--env KEY=VALUE</code> writes them.`}
          below=${html`<div class="mdl-env">${rows.map(([k, v], i) => html`<div class="mdl-env__row" key=${i}>
              <input class="input mdl-mono" aria-label="Variable name" placeholder="NAME" value=${k}
                onInput=${(e) => setEnv(rows.map((r, j) => (j === i ? [e.target.value.trim(), r[1]] : r)))}/>
              <span aria-hidden="true">=</span>
              <input class="input mdl-mono" aria-label=${`Value of ${k || "the variable"}`} placeholder="value" value=${v}
                onInput=${(e) => setEnv(rows.map((r, j) => (j === i ? [r[0], e.target.value] : r)))}/>
              <button type="button" class="icon-btn icon-btn--sm" aria-label=${`Remove ${k || "this variable"}`}
                onClick=${() => setEnv(rows.filter((_, j) => j !== i))}><${Icon} name="close" sm/></button></div>`)}
            <button type="button" class="btn btn--ghost btn--sm mdl-link" onClick=${() => setEnv([...rows, ["", ""]])}><${Icon} name="plus" sm/>Add a variable</button></div>`}/>
        <div class="mdl-field mdl-field--last"><span class="mdl-field__text"><span class="mdl-field__name"><span>The engine's command</span></span>
          <span class="mdl-field__hint">What Maya runs from this config, changes in purple. A setup run again keeps these settings.</span></span>
          <${Command} c=${MS.preview} saved=${m.command}/></div>`
      : tabs[tab].rows}
    </div>
    ${MS.check ? html`<div class="mdl-warn mdl-warn--err" role="alert">${MS.check}</div>` : null}
    ${changed.length ? html`<p class="mdl-note">${restarts
      ? `Maya starts again with these settings after the answer being written; the model loads again (a few minutes). The settings before stay in ${m.file}.prev.`
      : m.running ? `Saved into ${m.file}; they apply from Maya's next start (this server was not started by maya.sh or START-MAYA.bat).`
      : `Saved into ${m.file}; ${titleOf(m)} starts with them when you switch to it.`}</p>` : null}
    <div class="mdl-edit__foot">
      <span class=${`mdl-count${changed.length ? " is-on" : ""}`}>${changed.length ? `${changed.length} change${changed.length === 1 ? "" : "s"}` : "No changes"}</span>
      <button type="button" class="btn btn--ghost btn--sm mdl-link" onClick=${() => { MS.draft = JSON.parse(JSON.stringify(m.setup)); setDraft({}); }}
        disabled=${same(toSend(d), toSend(m.setup))}><${Icon} name="undo" sm/>Reset to the setup's values</button>
      <span class="mdl-grow"></span>
      <button type="button" class="btn btn--secondary" onClick=${closeEditor}>Cancel</button>
      <button type="button" class="btn btn--primary" disabled=${!canSave} onClick=${() => saveSettings(m)}>
        ${MS.saving ? "Saving…" : restarts ? "Save and restart" : "Save"}</button>
    </div>
  </div>`;
}

// ------------------------------------------------------------------ a model's card
function Card({m}) {
  const s = m.settings, open = !!MS.details[m.file], editing = MS.edit === m.file, r = MS.restart;
  const stopping = m.running && r && r.why === "switch", restarting = m.running && r && r.why === "settings";
  const loading = r && r.why === "switch" && r.file === m.file;
  const chips = [["Context", ctxfmt(s.context)], ["Draft", DRAFTS[s.draft] || s.draft], ["GPUs", s.gpus.join(", ") || "–"],
                 ["Images", s.images ? "on" : "off"], ...(s.parallel > 1 ? [["At once", String(s.parallel)]] : [])];
  const about = [m.backend === "hip" ? "HIP" : "CUDA", size(m.size), setUpOn(m.written)].filter(Boolean).join(" · ");
  const facts = [["Config", m.path, true], ["Start script", m.script ? `./${m.script}` : "–", !!m.script], ["Model folder", m.folder || "–", true],
                 ["Port", m.port ? String(m.port) : "8080"], ["API model name", m.model, true]];
  return html`<section class=${`card mdl-card${m.running ? " is-running" : ""}${loading ? " is-loading" : ""}`} aria-label=${titleOf(m)}>
    <div class="mdl-card__head">
      <${ModelLogo} model=${m.model} size=${44}/>
      <span class="mdl-card__name"><span><b>${modelName(m.model)}</b><span class="mdl-badge">${m.quant}</span></span>
        <span class="mdl-card__about">${about}</span></span>
      ${m.running ? html`<span class=${`pill pill--${stopping ? "warn" : restarting ? "info" : "accent"}`} role="status">
          <i class=${`status__dot${stopping || restarting ? " is-pulse" : ""}`}></i>${stopping ? "Stopping…" : restarting ? "Restarting…" : "Running"}</span>`
        : loading ? html`<span class="pill pill--info" role="status"><i class="status__dot is-pulse"></i>Loading…</span>`
        : !m.ok ? html`<span class="pill pill--warn" title=${m.missing || ""}>${m.missing && m.missing.startsWith("its model files") ? "Files missing" : "Incomplete"}</span>`
        : html`<button type="button" class="btn btn--secondary btn--sm" disabled=${!!r} onClick=${() => switchTo(m)}><${Icon} name="swap" sm/>Switch to this</button>`}
    </div>
    <div class="mdl-chips">${chips.map(([k, v]) => html`<span class="mdl-chip"><span>${k}</span><b>${v}</b></span>`)}</div>
    ${m.missing && !m.running ? html`<div class="mdl-warn">It can't start: ${m.missing}. Run the setup again to repair it (<code>./maya.sh --setup</code>).</div>` : null}
    <div class="mdl-card__foot">
      <span class="mdl-card__file"><code>${m.file}</code><span>·</span><span>${m.running ? "running now" : m.used ? `last used ${relTime(m.used * 1000)}` : ""}</span></span>
      <span class="mdl-card__acts">
        ${m.running ? html`<button type="button" class="btn btn--ghost btn--sm" onClick=${() => showView("monitor")}><${Icon} name="activity" sm/>Monitor</button>` : null}
        <button type="button" class=${`btn btn--ghost btn--sm${editing ? " is-on" : ""}`} aria-expanded=${String(editing)} disabled=${!m.ok}
          onClick=${() => openEditor(m)}><${Icon} name="sliders" sm/>Engine settings</button>
        <button type="button" class="btn btn--ghost btn--sm" aria-expanded=${String(open)} onClick=${() => { MS.details[m.file] = !open; emit(); }}>
          Details<${Icon} name="chevron" sm cls=${`chev${open ? "" : " is-closed"}`}/></button>
      </span>
    </div>
    ${open ? html`<dl class="mdl-facts">${facts.map(([k, v, mono]) => html`<dt>${k}</dt><dd style=${mono ? `font-family:${MONO}` : ""}>${v}</dd>`)}</dl>` : null}
    ${editing ? html`<${Editor} m=${m} key=${m.file}/>` : null}
  </section>`;
}

// ------------------------------------------------------------------ the page
export function ModelsPage() {
  useEffect(() => { loadModels(); }, []);
  const phone = isPhone(), D = MS.data, r = MS.restart;
  const cmds = [["Linux and macOS", "./maya.sh --setup"], ["Windows", "START-MAYA.bat --setup"]];
  return html`<main class="mon mdl">
    ${phone ? html`<header class="chat-head chat-head--float">
        <button type="button" class="glass-btn" aria-label="Chats" onClick=${() => { S.drawer = true; emit(); }}><${Icon} name="menu"/></button>
        <span class="model-btn model-btn--phone">Models</span>
        <button type="button" class="glass-btn" aria-label="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button></header>`
      : !S.sideOpen ? html`<header class="chat-head"><div class="chat-head__left">
        <button type="button" class="icon-btn" aria-label="Open the sidebar" title="Open the sidebar" onClick=${() => setSide(true)}><${Icon} name="panel-left"/></button>
        <button type="button" class="icon-btn" aria-label="New chat" title="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button></div></header>` : null}
    <div class="mon__scroll"><div class="mon__page mdl__page">
      <div class="mon__head">
        <div><h1>Models</h1><p>The models set up on this machine. Maya runs one at a time.</p></div>
        <div class="mon__acts"><button type="button" class=${`btn btn--secondary btn--sm${MS.add ? " is-on" : ""}`} aria-expanded=${String(MS.add)}
          onClick=${() => { MS.add = !MS.add; emit(); }}><${Icon} name="plus" sm/>Add a model</button></div>
      </div>
      ${MS.add ? html`<section class="card mdl-add" aria-label="Add a model">
        <b>Add a model</b>
        <span>Run the setup again and pick another model or quant. It shows the download's size first, then builds the model's pack and writes its config, which appears here.</span>
        ${cmds.map(([os, c]) => html`<div class="mdl-add__cmd"><span class="mdl-add__os">${os}</span><code>${c}</code>
          <button type="button" class="icon-btn icon-btn--sm" aria-label=${`Copy ${c}`} onClick=${() => copyText(c)}><${Icon} name="copy" sm/></button></div>`)}
      </section>` : null}
      ${r ? html`<div class="callout callout--info" role="status"><${Icon} name="info" sm/><span>
          <b>${r.why === "switch" ? `Switching to ${r.name}.` : `Restarting ${r.name} with the new engine settings.`}</b>
          ${" "}Maya finishes the answer it is writing, then loads the model (a few minutes). This page reloads when it is back.</span></div>` : null}
      ${MS.err && !D ? html`<div class="card mon__wait">${MS.err}</div>`
        : !D ? html`<div class="card mon__wait">Reading the models…</div>`
        : !D.models.length ? html`<div class="card mon__wait">No model is set up in this folder yet. The setup writes one: <code>./maya.sh --setup</code> (Windows: <code>START-MAYA.bat --setup</code>)</div>`
        : D.models.map((m) => html`<${Card} m=${m} key=${m.file}/>`)}
      ${D && D.models.length ? html`<p class="mdl-foot"><${Icon} name="info" sm/><span>${D.restarts
        ? "Switching, or saving the running model's engine settings, starts Maya again through its start script, as a port change does. Each model keeps its own chat settings."
        : "This server was not started by maya.sh or START-MAYA.bat, so it cannot start again by itself: a switch or new settings apply from Maya's next start."}</span></p>` : null}
    </div></div>
  </main>`;
}
