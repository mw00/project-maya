// serve/web/monitor.js - the Monitor: the model's speed and what it is doing now, where its experts live, what fills each
// GPU, the hardware, and the recent requests - all from the server's /metrics (every second) and the engine's STAT lines.
import {html, useRef, useState} from "./preact.js";
import {GIB, ago, ctxfmt, fmt, gb, isPhone, kfmt, rate, store} from "./lib.js";
import {S, copyText, emit, toast} from "./state.js";
import {fetchReport, reqAll} from "./server.js";
import {newChat} from "./chats.js";
import {setSide, status} from "./sidebar.js";
import {Icon} from "./ui.js";

// ------------------------------------------------------------------ a sparkline (nothing drawn when idle: all zeros)
function sparkPath(values, max) {
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2 || v.every((x) => x === 0)) return null;
  const top = Math.max(max || 0, ...v, 1e-9);
  const d = v.map((x, i) => `${i ? "L" : "M"}${((i / (v.length - 1)) * 100).toFixed(2)},${(30 - (x / top) * 26).toFixed(2)}`).join("");
  return {line: d, area: `${d}L100,32L0,32Z`};
}
const Spark = ({values, max, tone = "", cls = ""}) => {
  const p = sparkPath(values, max);
  return html`<svg class=${`spark ${cls}`} data-tone=${tone} viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
    ${p ? html`<path class="spark__area" d=${p.area}/><path class="spark__line" d=${p.line} vector-effect="non-scaling-stroke"/>` : null}</svg>`;
};
const Num = ({v, unit}) => html`<span class="stat__value">${v == null ? "–" : v}${v != null && unit ? html`<small>${unit}</small>` : null}</span>`;
const Badge = ({tone, children}) => html`<span class=${`badge badge--${tone}`}><i class="badge__dot"></i>${children}</span>`;

// ------------------------------------------------------------------ problems worth a line at the top
function Banners({m}) {
  const out = [];
  const rl = m.reload;
  if (rl && (rl.state === "waiting" || rl.state === "running")) {
    out.push(["info", `Reloading the model with a ${ctxfmt(rl.to)} context${rl.state === "waiting" ? ", after the answer being written" : ""}. Requests wait until it is back.`]);
  } else if (rl && rl.state === "failed" && rl.ended && Date.now() / 1000 - rl.ended < 600) {
    out.push(["danger", `The context change did not work: ${rl.error || "the engine did not start"}. It runs with ${ctxfmt(rl.context)} again.`]);
  }
  const eng = m.engine || {};
  if (eng.gpu_commit_capped_gb > 0) {   // (DeepSeek on a Windows APU: the page file kept experts off the GPU)
    out.push(["warn", `${fmt(eng.gpu_commit_capped_gb)} GB of experts stay off the GPU: Windows' page file is too small to hold them there. A page file of ${fmt(eng.pagefile_hint_gb)} GB keeps them on the GPU (System > About > Advanced system settings > Performance > Advanced > Virtual memory), then restart Maya.`]);
  }
  const gpus = (m.hardware || {}).gpus || [{index: null, temp: (m.hardware || {}).gpu_temp}];
  for (const g of gpus) {
    if (g.temp != null && g.temp >= 87) out.push(["warn", `${g.index == null ? "The GPU" : `GPU ${g.index}`} is at ${fmt(g.temp)} °C: it may slow itself down to cool off.`]);
  }
  if (!out.length) return null;
  return html`<div class="banners">${out.map(([k, t]) => html`<div class=${`callout callout--${k}`} role=${k === "danger" ? "alert" : "status"}>
    <${Icon} name=${k === "info" ? "info" : k === "danger" ? "error" : "warning"} sm/><span>${t}</span></div>`)}</div>`;
}

// ------------------------------------------------------------------ the speed, and what is happening now
function Throughput({m}) {
  const live = m.live || {}, h = m.history || {}, last = (m.requests || [])[0], eng = m.engine || {}, tn = m.tiers && m.tiers.now;
  const gen = live.state === "generating";
  const many = live.active > 1;                  // several answers at once (--parallel): their total, the graph's newest point
  const speed = many ? (h.tok_s || []).at(-1) : gen ? live.tok_s : last ? last.decode_tok_s : null;
  // prefill: the engine's rate over the tokens it read (not the reused prefix), live while a request runs
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
    : last ? (last.prefill_tok_s != null ? last.prefill_tok_s
              : last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null)
    : null;
  const fresh = tn && tn.time && Date.now() / 1000 - tn.time <= 30;
  const ms = gen && tn && tn.ms_tok ? tn.ms_tok : speed ? 1000 / speed : null;
  const hit = fresh && tn.vram_hit != null ? 100 * tn.vram_hit : last && last.hit_rate != null ? 100 * last.hit_rate : null;
  const a = sparkPath(h.tok_s), b = sparkPath(h.prefill_tok_s_mean);
  // (DSpark: DeepSeek's server says whether it drafts, MiniMax's engine itself - dspark=0 / 1; the engine's STAT lines the
  //  share of its drafts kept)
  const spec = eng.dspark != null ? (+eng.dspark ? "DSpark" : "Off")
    : eng.mtp == null ? (eng.dflash ? "DFlash2" : null) : +eng.mtp ? "MTP" : "Off";
  const kept = +eng.dspark && tn && tn.drafts != null ? `${fmt(100 * tn.drafts)}% kept` : "";
  return html`<section class="card card--wide" aria-label="Throughput">
    <div class="speeds">
      <div class="speed"><span class="speed__label"><i class="key key--line"></i>Decode · ${many ? `${live.writing} writing` : gen ? "writing now" : live.state === "reading" ? "reading the prompt" : last ? "last request" : "waiting for a request"}</span>
        <span class="speed__big num">${speed == null ? "–" : fmt(speed, 1)}<small>tok/s</small></span></div>
      <div class="speed" title="Prompt tokens read per second (the prefill), counting only the tokens read - not the ones reused from the conversation cache">
        <span class="speed__label"><i class="key key--dash"></i>Prefill · ${live.state === "reading" ? "reading now" : gen ? "this request" : last ? "last request" : "waiting"}</span>
        <span class="speed__mid num">${prefill == null ? "–" : fmt(prefill)}<small>tok/s</small></span></div>
    </div>
    <div class=${`spark-wrap${a || b ? "" : " is-empty"}`}>
      <svg class="spark spark--hero" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
        ${a ? html`<path class="spark__area" d=${a.area}/><path class="spark__line" d=${a.line} vector-effect="non-scaling-stroke"/>` : null}
        ${b ? html`<path class="spark__line spark__line--dash" d=${b.line} vector-effect="non-scaling-stroke"/>` : null}</svg>
      <span class="spark-wrap__empty">No activity in the last minute</span>
    </div>
    <div class="stats">
      <div class="stat"><span class="stat__label">Time per token</span><${Num} v=${ms == null ? null : fmt(ms, 1)} unit="ms"/></div>
      <div class="stat"><span class="stat__label">VRAM hit</span><${Num} v=${hit == null ? null : fmt(hit, 1)} unit="%"/></div>
      <div class="stat"><span class="stat__label">Requests served</span><${Num} v=${m.totals && m.totals.requests != null ? fmt(m.totals.requests) : null}/></div>
      <div class="stat" title=${kept ? "The share of the drafted tokens the model kept, in the last answer" : undefined}>
        <span class="stat__label">Speculation</span><${Num} v=${spec} unit=${kept}/></div>
    </div>
  </section>`;
}

function RightNow({m}) {
  const live = m.live || {}, last = (m.requests || [])[0], eng = m.engine || {};
  const took = (s) => (s == null ? "" : s < 90 ? ` · ${fmt(s, s < 10 ? 1 : 0)} s` : ` · ${fmt(Math.floor(s / 60))} min ${fmt(s % 60)} s`);
  let label = "Idle", tone = "muted", detail = "waiting for a request", pct = null, ptone = "accent";
  if (S.reload) {
    label = "Reloading"; tone = "info"; ptone = "info"; detail = `a ${ctxfmt(S.reload.to)} context`;
  } else if (live.state === "reading") {
    label = "Reading"; tone = "info"; ptone = "info";
    if (live.prompt_total) { pct = (100 * live.prompt_read) / live.prompt_total; detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} prompt tokens`; }
    else detail = `${fmt(live.prompt_tokens)} prompt tokens`;
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Writing"; tone = "accent";
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : null;
    detail = `${fmt(live.generated)} tokens${took(live.elapsed_s)}`;
  } else if (last) {
    detail = `last answer: ${fmt(last.output_tokens)} tokens${took(last.duration_s)}`;
  }
  const running = live.state !== "idle";
  return html`<section class="card card--side" aria-label="Right now">
    <span class="card__title">Right now</span>
    <div class="now"><${Badge} tone=${tone}>${label}<//><span class="now__detail num">${detail}</span></div>
    ${pct != null ? html`<div class="bar" data-tone=${ptone} role="progressbar" aria-valuenow=${Math.round(pct)} aria-valuemin="0" aria-valuemax="100"
      aria-label=${label}><span style=${`width:${pct}%`}></span></div>` : null}
    <dl class="kv">
      ${running && live.prompt_total ? html`<dt>Prompt read</dt><dd class="num">${fmt(live.prompt_read)} of ${fmt(live.prompt_total)}</dd>`
        : last ? html`<dt>Reused from the cache</dt><dd class="num">${fmt(last.reused || 0)} tokens</dd>` : null}
      <dt>Answers at once</dt><dd class="num">${fmt(live.active || 0)}</dd>
      <dt>Waiting</dt><dd class="num">${live.queued > 0 ? fmt(live.queued) : "none"}</dd>
    </dl>
  </section>`;
}

// ------------------------------------------------------------------ the context: what the conversation holds, of its size
// the prompt the conversation cache already held (reused), the prompt read new and what was written - of the request being
// answered, else of the last one; GLM's KV streaming keeps the first kv_resident positions of each layer in VRAM
function ContextCard({m}) {
  const live = m.live || {}, last = (m.requests || [])[0], eng = m.engine || {};
  const max = eng.max_context || 0;
  if (!max) return null;
  const running = live.state === "reading" || live.state === "generating";
  let prompt = 0, reused = null, written = 0, what = "";
  if (running) {
    prompt = live.prompt_tokens || 0; reused = live.prompt_reused; written = live.generated || 0;
    what = live.active > 1 ? `the newest of ${fmt(live.active)} answers being written` : "the request being answered";
  } else if (last) {
    prompt = last.prompt_tokens || 0; reused = last.reused; written = last.output_tokens || 0; what = "the last request";
  }
  const r = reused == null ? null : Math.min(reused, prompt);
  const segs = [...(r == null ? [] : [{k: "reused", label: "Reused from the cache", n: r}]),
                {k: "fresh", label: r == null ? "Prompt" : "Read new", n: prompt - (r || 0)}, {k: "written", label: "Written", n: written}];
  const used = Math.min(max, prompt + written), free = Math.max(0, max - used), pct = (100 * used) / max;
  const win = +eng.kv_resident || 0, showWin = win > 0 && win < max;
  const note = !running && !last ? "No conversation yet: the context fills as a chat goes on, with every message, file and tool result in it."
    : showWin ? (used > win ? `Beyond the first ${ctxfmt(win)} positions the cache streams from RAM (KV streaming): about ${fmt((used * 12288) / 1e9, 1)} GB of RAM for this conversation.`
                            : `Up to ${ctxfmt(win)} the whole cache stays in VRAM; beyond it, it streams from RAM (KV streaming).`)
    : "What the model holds at once: the chat, its files and tool results. A chat longer than this is refused: start a new one, or give the model a bigger context.";
  const aria = `${fmt(used)} of ${fmt(max)} tokens in use` + segs.map((s) => `, ${s.label.toLowerCase()} ${fmt(s.n)}`).join("");
  return html`<section class="card" aria-label="Context">
    <div class="card__head"><span class="card__title">Context</span><span class="card__sub">${what || ""}</span></div>
    <div class="ctx__nums"><span class="ctx__big num">${fmt(used)}<small>/ ${fmt(max)} tokens</small></span>
      <span class="ctx__pct num"><b>${pct > 0 && pct < 0.1 ? "< 0.1" : fmt(pct, pct > 0 && pct < 10 ? 1 : 0)}%</b> used · ${fmt(free)} left</span></div>
    <div class=${`ctx__track${showWin ? " ctx__track--win" : ""}`}>
      ${showWin ? html`<span class="ctx__win" style=${`left:${(100 * win) / max}%`}><span>VRAM window · ${ctxfmt(win)}</span></span>` : null}
      <div class="ctxbar" role="img" aria-label=${aria}>${segs.filter((s) => s.n > 0).map((s) => html`<span class=${`ctxbar__${s.k}`} style=${`flex-grow:${s.n}`}></span>`)}
        <span style=${`flex-grow:${free}`}></span></div></div>
    <div class="legend">${segs.map((s) => html`<span><i class=${`sw sw--${s.k}`}></i>${s.label} <b class="num">${fmt(s.n)}</b></span>`)}
      <span><i class="sw sw--free"></i>Free <b class="num">${fmt(free)}</b></span></div>
    <p class="card__note card__note--tip">${note} The size: Settings › Model, or the model's Engine settings on the Models page.</p>
  </section>`;
}

// ------------------------------------------------------------------ where the experts live (the engine's STAT lines)
// GLM's engine: where each routed expert came from while the model writes; DeepSeek's (maya-ds4): the experts the CPU lane
// computed, what the storage read, the cache's moves and the n-gram memory (Engram) a token; MiniMax's (maya-mm3): the
// experts its CPU lane computed from the pinned pool
const TIER_SETS = {
  glm: [
    {key: "ram_fetch", label: "From RAM", unit: "/tok", digits: 2, tone: "info", sub: "experts pulled over PCIe from RAM"},
    {key: "disk", label: "From SSD", unit: "/tok", digits: 2, tone: "warn", sub: "experts read from the SSD"},
    {key: "promo", label: "Promoted", unit: "/tok", digits: 2, sub: "experts moved into VRAM"},
  ],
  ds4: [
    {key: "ram_fetch", label: "From RAM", unit: "/tok", digits: 2, tone: "info", sub: "experts the CPU computed from RAM"},
    {key: "disk", label: "From SSD", unit: "/tok", digits: 2, tone: "warn", sub: "experts read from the SSD"},
    {key: "promo", label: "Promoted", unit: "/tok", digits: 2, sub: "experts moved into VRAM"},
    {key: "engram_ms", label: "N-gram memory", unit: " ms/tok", digits: 2, sub: "n-gram lookups"},
  ],
  mm3: [
    {key: "ram_fetch", label: "From RAM", unit: "/tok", digits: 2, tone: "info", sub: "experts the CPU computed from pinned RAM"},
    {key: "disk", label: "From SSD", unit: "/tok", digits: 2, tone: "warn", sub: "experts read from the SSD"},
    {key: "promo", label: "Promoted", unit: "/tok", digits: 2, sub: "experts copied into the VRAM cache"},
  ],
};
function Tiers({t, eng}) {
  if (!t || !t.now) return null;
  const ds4 = !!(eng && eng.engine_kind === "ds4"), mm3 = !!(eng && eng.engine_kind === "mm3");
  const set = TIER_SETS[ds4 ? "ds4" : mm3 ? "mm3" : "glm"];
  const n = t.now, h = t.history || {};
  const age = n.time ? Math.max(0, Date.now() / 1000 - n.time) : null;
  const stale = age != null && age > 30;               // a reading from the last answer, not live
  // VRAM, RAM and the rest on the SSD - DeepSeek's (maya-ds4) the same keys: its RAM the copies or the read-ahead
  const vu = n.vram_used || 0, ru = n.ram_used || 0;
  const total = (ds4 ? vu + (n.cpu_n != null ? n.cpu_n : +eng.cpu_experts || 0) : eng && eng.experts) || Math.max(1, vu + ru);
  const disk = Math.max(0, total - vu - ru);
  let tip = "";
  if (ds4) {
    if (!stale && n.disk_mb != null && n.disk_mb >= 500) tip = "Much of every token is read from the SSD: more RAM keeps more of the model's experts cached, and answers get faster.";
    else if (!stale && n.vram_hit != null && n.vram_hit < 0.5) tip = "Most experts run on the CPU: more VRAM (or a second GPU) keeps more of them on the card.";
  } else if (!stale && n.disk != null && n.disk >= 3) {
    tip = "Many experts are read from the SSD for every token: more RAM, a smaller model or a shorter context makes answers faster.";
  } else if (mm3) {   // (MiniMax: the CPU computes the experts VRAM does not hold - nothing crosses PCIe for them)
    if (!stale && n.vram_hit != null && n.vram_hit < 0.5) tip = "Most experts run on the CPU: more VRAM (or a second GPU) keeps more of them on the card.";
  } else if (!stale && n.vram_hit != null && n.vram_hit < 0.85 && (n.ram_fetch || 0) >= 4) {
    tip = "Most misses come from RAM over PCIe: more VRAM (or a second GPU) keeps more experts on the card.";
  }
  const pc = (x) => (100 * x) / total;
  return html`<section class=${`card${stale ? " is-stale" : ""}`} aria-label="Where the experts live">
    <div class="card__head"><span class="card__title">Where the experts live</span>
      <span class="card__sub">${age == null ? "" : stale ? `last reading ${ago(age)} ago, from the last answer` : `live · ${fmt(total)} routed experts`}</span></div>
    <div class="tierbar" role="img" aria-label=${`${fmt(pc(vu), 0)}% of the experts in VRAM, ${fmt(pc(ru), 0)}% in ${ds4 ? "RAM" : "pinned RAM"}, ${fmt(pc(disk), 0)}% on the SSD only`}>
      <span class="tierbar__vram" style=${`flex-grow:${vu}`}></span><span class="tierbar__ram" style=${`flex-grow:${ru}`}></span>
      <span class="tierbar__disk" style=${`flex-grow:${disk}`}></span></div>
    <div class="legend">
      <span><i class="sw sw--vram"></i>VRAM <b class="num">${fmt(vu)} · ${fmt(n.vram_gb, 1)} GB</b></span>
      <span><i class="sw sw--ram"></i>${ds4 ? "RAM" : "Pinned RAM"} <b class="num">${fmt(ru)} · ${fmt(n.ram_gb, 1)} GB</b></span>
      <span><i class="sw sw--disk"></i>SSD only <b class="num">${fmt(disk)}</b></span></div>
    <div class=${`tiles${set.length === 4 ? " tiles--4" : ""}`}>
      ${set.map((x) => {
        const v = n[x.key];
        let sub = x.sub, tip = null;
        if (x.key === "engram_ms" && eng && eng.engram_lookups) sub = `${fmt(+eng.engram_lookups)} n-gram lookups a token`;
        if (x.key === "engram_ms" && n.eg_lookup_rows > 0) {   // (its host cache, #522: the share found in RAM)
          sub += ` · ${fmt((100 * (n.eg_lookup_hits || 0)) / n.eg_lookup_rows, 0)}% from its RAM cache`;
          tip = `N-gram memory's RAM cache: ${fmt(n.eg_cache_rows || 0)} of ${fmt(n.eg_cache_cap || 0)} rows held; ` +
                `${fmt(n.eg_lookup_rows)} lookups since the start, ${fmt(n.eg_lookup_hits || 0)} found in RAM, ` +
                `${fmt(n.eg_lookup_reads || 0)} read from the SSD`;
        }
        if ((x.key === "disk" || x.key === "ram_fetch") && v == null && ds4) sub = "not measured on this system";
        return html`<div class="tile" title=${tip}><span class="stat__label">${x.label}</span>
          <${Num} v=${v == null ? null : fmt(v, x.digits)} unit=${x.unit}/>
          <span class="tile__sub">${stale ? "last answer: " : ""}${sub}</span>
          <${Spark} values=${(h[x.key] || []).map((y) => (y == null ? 0 : y))} tone=${x.tone || ""}/></div>`;
      })}
    </div>
    ${tip ? html`<p class="card__note">${tip}</p>` : null}
  </section>`;
}

// ------------------------------------------------------------------ what fills the GPUs (/metrics gpu_map)
// a row a GPU, its bar the whole card; with several, the parts' colours once above the rows with their GB over all of
// them, each row's load, power, temperature and PCIe link under its bar, and a row opened (a click) shows its parts with a
// sentence each - one GPU: its row, opened (the hardware cards have its readings).  No VRAM lines: NVML's memory, from five
const GPU_PARTS = [
  {key: "weights", label: "Model weights"}, {key: "kv", label: "KV cache"}, {key: "act", label: "Activations"},
  {key: "prompt", label: "Prompt buffers"}, {key: "tp", label: "Tensor parallel"},
  {key: "drafter", label: "Drafter"}, {key: "pool", label: "Expert pool"}, {key: "vision", label: "Vision encoder"},
  {key: "rest", label: "CUDA and the rest"}, {key: "other", label: "Other programs"},
];
const GPU_PLAIN_FROM = 5;
const gpuKey = (g) => String(g.index != null ? g.index : `d${g.dev}`);
const gpuLabel = (g) => `GPU ${g.index ?? g.dev}`;
const gpuName = (g) => (g.name || "").replace(/^NVIDIA /, "").replace(/^GeForce /, "").replace(/^AMD /, "").replace(/^Radeon /, "");
const gpuGb = (b) => `${gb(b, b < GIB ? 2 : 1)} GB`;
const gpuList = (gs) => gs.length === 1 ? gpuLabel(gs[0])
  : `GPUs ${gs.slice(0, -1).map((g) => g.index ?? g.dev).join(", ")} and ${gs[gs.length - 1].index ?? gs[gs.length - 1].dev}`;
// the layers a GPU holds: the GLM engine's VRAM lines say [first, end), DeepSeek's and MiniMax's (maya-ds4, maya-mm3)
// [first, last]
const dsparkEngine = (eng) => eng.engine_kind === "ds4" || eng.engine_kind === "mm3";
function gpuLayers(g, eng) {
  if (!g.layers) return "";
  const [a, b] = g.layers, last = dsparkEngine(eng) ? b : b - 1;
  return a < 0 || last < a ? "" : last > a ? `layers ${a}–${last}` : `layer ${a}`;
}
// the KV cache streams from RAM only beyond its VRAM window (DeepSeek's engine: a window as long as the context)
const kvWindow = (eng, ctx) => (+eng.kv_resident > 0 && +eng.kv_resident < ctx ? +eng.kv_resident : 0);
function gpuSegs(g, eng) {
  const p = g.parts || {}, draft = eng.dflash ? "DFlash2" : +eng.mtp ? "MTP" : dsparkEngine(eng) ? "DSpark" : "";
  return GPU_PARTS.filter((d) => (p[d.key] || 0) > 0 || d.key === "pool" || d.key === "kv")
    .map((d) => ({key: d.key, bytes: p[d.key] || 0, label: d.key === "drafter" && draft ? `Drafter (${draft})` : d.label}))
    .concat([{key: "free", label: "Free", bytes: g.free || 0}]);
}
// a GPU's NVML readings under its bar: load, power, temperature (warm from 80 °C, hot from 87) and PCIe link
function GpuStats({r}) {
  if (!r) return null;
  const gen = r.pcie_gen_max || r.pcie_gen, mb = r.pcie_rx_mb;
  const heat = r.temp >= 87 ? "is-hot" : r.temp >= 80 ? "is-warm" : "";
  return html`<span class="gpustat num">
    ${r.util != null ? html`<span><b>${fmt(r.util)}%</b> load</span>` : null}
    ${r.power != null ? html`<span><b>${fmt(r.power)}</b>${r.power_limit ? ` / ${fmt(r.power_limit)}` : ""} W</span>` : null}
    ${r.temp != null ? html`<span class=${heat}><b>${fmt(r.temp)}</b> °C</span>` : null}
    ${gen ? html`<span>PCIe <b>Gen${gen}</b>${r.pcie_width ? ` x${r.pcie_width}` : ""}${mb == null ? "" : ` · ${rate(mb)}`}</span>` : null}</span>`;
}
function gpuDetail(g, segs, m) {
  const eng = m.engine || {}, live = m.live || {}, last = (m.requests || [])[0];
  const ctx = g.ctx || eng.max_context || 0;
  const inUse = live.state !== "idle" ? (live.prompt_tokens || 0) + (live.generated || 0)
              : last ? (last.prompt_tokens || 0) + (last.output_tokens || 0) : 0;
  const sub = {
    weights: "attention, shared experts and the other dense tensors",
    kv: (kvWindow(eng, ctx) ? `${ctxfmt(kvWindow(eng, ctx))} of ${ctxfmt(ctx)} positions here, the rest in RAM` : `${ctxfmt(ctx)} positions`) +
        (inUse ? ` · ${kfmt(inUse)} in use` : "") + (g.kv_ram ? ` · ${gb(g.kv_ram)} GB pinned in RAM` : ""),
    act: "one token's working buffers",
    prompt: "a long prompt's chunks: their working buffers and the ring the CPU lane's experts stream through",
    tp: g.tp_n > 1 ? `the copies a group of ${g.tp_n} GPUs needs: the tensors each of them holds whole, and its layers' KV cache` : "",
    drafter: [g.drafter ? `weights ${gb(g.drafter, 2)} GB` : "", g.window ? `verify window ${gb(g.window, 2)} GB` : ""].filter(Boolean).join(" · "),
    pool: [g.slots ? `${fmt(g.slots)} expert slots` : "", g.borrow ? `prompts borrow up to ${gb(g.borrow)} GB of it` : ""].filter(Boolean).join(" · "),
    vision: g.vision_lent ? "the pool's tail, while it encodes" : g.vision_own ? "the vision tower, in the engine's process" : "its own process",
    rest: "the driver's reserve and context, conversation snapshots and small buffers",
    other: "processes that are not Maya's",
    free: (g.free || 0) < 1.5 * GIB ? "kept free for the driver and the sampler" : "unused",
    used: "",
  };
  return html`<div class="gpulegend">${segs.map((s) => html`<div class="gpulegend__item" data-k=${s.key}>
    <i class=${`sw${s.key === "free" ? " sw--free" : ""}`} style=${`--gm-c:var(--gm-${s.key})`}></i><span>${s.label}</span>
    <b class="num">${gpuGb(s.bytes)}</b><span class="gpulegend__sub">${sub[s.key] || ""}</span></div>`)}</div>`;
}
function GpuMap({m}) {
  const [open, setOpen] = useState(store.get("gpu-open", null));   // the row opened (several GPUs), remembered here
  const [focus, setFocus] = useState(null), [tip, setTip] = useState(null);
  const card = useRef(null);
  let map = m.gpu_map;
  const reads = (m.hardware || {}).gpus || [];
  const plain = !map || !map.length;   // (no VRAM lines: NVML's memory in use, from five GPUs)
  if (plain && reads.length >= GPU_PLAIN_FROM) {
    const names = ((m.hardware_static || {}).gpu_name || "").split(" + ");
    map = reads.map((r, i) => ({index: r.index, name: names.length === reads.length ? names[i] : "", total: r.mem_total || 0,
                                free: Math.max(0, (r.mem_total || 0) - (r.mem_used || 0)), parts: {used: r.mem_used || 0}}));
  }
  if (!map || !map.length) return null;
  const eng = m.engine || {}, live = m.live || {}, one = map.length === 1, fold = !one && !plain;
  const byIndex = Object.fromEntries(reads.map((r) => [String(r.index), r]));
  const rows = map.map((g) => ({g, key: gpuKey(g), read: one ? null : byIndex[String(g.index)],
                                segs: plain ? [{key: "used", label: "In use", bytes: g.parts.used}, {key: "free", label: "Free", bytes: g.free}] : gpuSegs(g, eng)}));
  const isOpen = (r) => !plain && (one || r.key === open);
  const sum = (f) => map.reduce((a, g) => a + (f(g) || 0), 0);
  const total = sum((g) => g.total), used = sum((g) => g.total - (g.free || 0));
  const reading = live.state === "reading";
  // the key: each part's GB over all the GPUs
  const tot = {}, label = {};
  for (const r of rows) for (const s of r.segs) { tot[s.key] = (tot[s.key] || 0) + s.bytes; label[s.key] = s.label; }
  // what is happening now: a prompt borrows the pools' tails, the vision encoder GPU 0's
  const vis = map.find((g) => g.vision_lent);
  const lentNow = reading ? sum((g) => !g.vision_lent && g.borrow > 0 && (g.parts || {}).pool > 0 ? Math.min(g.borrow, g.parts.pool) : 0) : 0;
  const now = vis ? `The vision encoder has ${gb(vis.parts.vision || vis.lendable)} GB of ${one ? "the" : `${gpuLabel(vis)}'s`} expert pool while it encodes.`
            : lentNow ? `Reading a prompt: the expert pool${one ? " lends" : "s lend"} it up to ${gb(lentNow)} GB (hatched).` : "";
  // advice: memory the expert pool could have at the next start
  const complete = eng.expert_slots && eng.experts && eng.expert_slots >= eng.experts;
  const more = (g, b) => (g.slots && g.parts.pool ? Math.max(0, Math.floor(b / (g.parts.pool / g.slots))) : 0);
  const where = (gs) => (one ? "this GPU" : gpuList(gs));
  const pools = (gs) => (one ? "the expert pool" : gs.length > 1 ? "their expert pools" : "its expert pool");
  const others = complete || plain ? [] : map.filter((g) => (g.parts.other || 0) >= 0.5 * GIB);
  const frees = complete || plain ? [] : map.filter((g) => !g.vision_lent && (g.free || 0) >= 2 * GIB);
  let advice = "";
  if (others.length) {
    const b = others.reduce((a, g) => a + g.parts.other, 0), n = others.reduce((a, g) => a + more(g, g.parts.other), 0);
    advice = `Other programs hold ${gb(b)} GB on ${where(others)}: with them closed, a restart gives ${pools(others)} about ${fmt(n)} more experts.`;
  } else if (frees.length) {
    const b = frees.reduce((a, g) => a + g.free, 0), n = frees.reduce((a, g) => a + more(g, g.free - GIB), 0);
    advice = `${gb(b)} GB of ${where(frees)} are free: a restart gives ${pools(frees)} about ${fmt(n)} more experts.`;
  }
  const pick = (k) => { const v = open === k ? null : k; setOpen(v); store.set("gpu-open", v); };
  const move = (e, r) => {
    const seg = e.target.closest(".gpubar__seg");
    const s = seg && r.segs.find((x) => x.key === seg.dataset.k);
    setFocus(s && s.key !== "free" ? s.key : null);
    if (!s) { setTip(null); return; }
    const c = card.current.getBoundingClientRect(), bar = seg.parentElement.getBoundingClientRect();
    setTip({x: Math.min(Math.max(e.clientX - c.left, 130), c.width - 130), y: bar.top - c.top,
            text: `${rows.length > 1 ? `${gpuLabel(r.g)} · ` : ""}${s.label} · ${gpuGb(s.bytes)} · ${fmt((100 * s.bytes) / (r.g.total || 1), 1)}% of the card`});
  };
  return html`<section ref=${card} class=${`card gpumap${one ? " is-one" : ""}`} aria-label=${plain ? "GPUs" : one ? "What fills the GPU" : "What fills the GPUs"}>
    <div class="card__head"><span class="card__title">${plain ? "GPUs" : one ? "What fills the GPU" : "What fills the GPUs"}</span>
      <span class="card__sub">${[one ? "" : `${map.length} GPUs`, `${gb(used)} of ${gb(total, 0)} GB in use`,
                                 map.some((g) => g.measured === "start") ? "as measured at the start" : ""].filter(Boolean).join(" · ")}</span></div>
    ${one || plain ? null : html`<div class="legend legend--sm" onPointerLeave=${() => setFocus(null)}>
      ${GPU_PARTS.map((d) => d.key).concat(["free"]).filter((k) => tot[k] > 0).map((k) => html`
        <span class="legend__item" onPointerOver=${() => setFocus(k === "free" ? null : k)}>
          <i class=${`sw${k === "free" ? " sw--free" : ""}`} style=${`--gm-c:var(--gm-${k})`}></i>${label[k]} <b class="num">${gpuGb(tot[k])}</b></span>`)}</div>`}
    <div class="gpurows" data-focus=${focus || undefined} onPointerLeave=${() => { setFocus(null); setTip(null); }}>
      ${rows.map((r) => {
        const g = r.g, p = g.parts || {}, o = isOpen(r);
        const lent = reading && !g.vision_lent && g.borrow > 0 && p.pool > 0 ? Math.min(100, (100 * g.borrow) / p.pool) : 0;
        const line = html`
          <span class="gpurow__name"><b>${gpuLabel(g)}</b><span>${[gpuName(g), gpuLayers(g, eng)].filter(Boolean).join(" · ")}</span></span>
          <span class="gpurow__track"><span class="gpubar" role="img" onPointerMove=${(e) => move(e, r)}
              aria-label=${`${gpuLabel(g)}: ` + r.segs.map((s) => `${s.label} ${gpuGb(s.bytes)}`).join(", ")}>
              ${r.segs.map((s) => html`<span class="gpubar__seg" data-k=${s.key} style=${`--gm-c:var(--gm-${s.key});flex-grow:${s.bytes / (g.total || 1)}`}>
                ${s.key === "pool" ? html`<i class="gpubar__lent" style=${`width:${lent}%`}></i>` : null}</span>`)}</span>
            ${one ? null : html`<${GpuStats} r=${r.read}/>`}</span>
          <span class="gpurow__used num"><b>${gb(Math.max(0, g.total - (g.free || 0)))}</b><span> / ${gb(g.total, 0)} GB</span>
            ${fold ? html`<${Icon} name="chevron" sm cls=${o ? "chev" : "chev is-closed"}/>` : null}</span>`;
        return html`<div class=${`gpurow${o ? " is-open" : ""}`} key=${r.key}>
          ${fold ? html`<button type="button" class="gpurow__line" aria-expanded=${String(o)} onClick=${() => pick(r.key)}>${line}</button>`
                 : html`<div class="gpurow__line">${line}</div>`}
          ${o ? html`<div class="gpurow__detail">${gpuDetail(g, r.segs, m)}</div>` : null}</div>`;
      })}
    </div>
    ${now ? html`<p class="card__note">${now}</p>` : null}
    ${advice ? html`<p class="card__note card__note--tip">${advice}</p>` : null}
    ${tip ? html`<div class="gputip" role="tooltip" style=${`left:${tip.x}px;top:${tip.y}px`}>${tip.text}</div>` : null}
  </section>`;
}

// ------------------------------------------------------------------ the hardware cards
const METRICS = [
  {key: "gpu", label: "GPU active", unit: "%", series: "gpu_util", max: 100,
   tip: "NVIDIA's counter: the share of time any GPU kernel is running. While Maya answers it reads 100% - between bursts of work the GPU waits (inside a kernel) for experts from RAM and the CPU. Power shows how hard it really works."},
  {key: "vram", label: "VRAM", series: "gpu_mem_used"},
  {key: "power", label: "Power", unit: "W", series: "gpu_power"},
  {key: "temp", label: "GPU temp", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "pcie", label: "PCIe", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", unit: "%", series: "cpu", max: 100},
  {key: "ram", label: "System RAM", series: "ram_used", tone: "info"},
  {key: "disk", label: "Disk read", series: "disk_read_mb", tone: "info"},
];
function Hardware({m, gpuEach}) {
  const hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  // the total (or the mean / the hottest) and its history; with several GPUs the line under it has every GPU's own value -
  // unless the GPU card's rows have them (gpuEach), each figure shown once
  const gpus = hw.gpus || [];
  const multi = gpus.length > 1, each = multi && !gpuEach;
  const per = (f) => gpus.map((g) => `GPU ${g.index} ${f(g)}`.replace(/ /g, " ")).join(" · ");
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  const link = (g) => `Gen${g.pcie_gen_max || g.pcie_gen || "?"}${g.pcie_width ? ` x${g.pcie_width}` : ""}`;
  const sameLink = gpus.every((g) => link(g) === link(gpus[0]));
  const big = hw.disk_read_mb != null && hw.disk_read_mb >= 1000;
  const v = {
    gpu: [hw.gpu_util == null ? null : fmt(hw.gpu_util), "%", each ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : multi ? `mean of ${gpus.length} cards` : st.gpu_name || ""],
    vram: [hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
           each ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`)) : (multi ? `${gpus.length} cards · ` : "") + (eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : "")],
    temp: [hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C", each ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : multi ? `hottest of ${gpus.length} cards` : ""],
    power: [hw.gpu_power == null ? null : fmt(hw.gpu_power), "W",
            each ? per((g) => (g.power == null ? "–" : `${fmt(g.power)}${g.power_limit ? `/${fmt(g.power_limit)}` : ""} W`))
                 : hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit${multi ? ` (all ${gpus.length} cards)` : ""}` : ""],
    pcie: [gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
           each ? per((g) => `${sameLink ? "" : `${link(g)} `}${rate(g.pcie_rx_mb)}`)
                : hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${rate(hw.gpu_pcie_rx_mb)}` + (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : "")],
    cpu: [hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : ""],
    ram: [hw.ram_used == null ? null : gb(hw.ram_used), hw.ram_total ? `/ ${gb(hw.ram_total, 0)} GB` : "GB", eng.ram_gb ? `${fmt(+eng.ram_gb, 1)} GB pinned for experts` : ""],
    disk: hw.disk_read_mb == null ? [null, "", st.psutil ? "" : "needs psutil (setup installs it)"]
      : [big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
         hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`],
  };
  const max = {vram: hw.gpu_mem_total, temp: 90, power: hw.gpu_power_limit, ram: hw.ram_total};
  return html`<h2 class="mon__h2">Hardware</h2>
    <div class="hw">${METRICS.map((x) => html`<div class="card hw__card" title=${x.tip}>
      <span class="stat__label">${x.label}</span><${Num} v=${v[x.key][0]} unit=${v[x.key][1]}/>
      <span class=${`hw__sub${each && ["gpu", "vram", "temp", "power", "pcie"].includes(x.key) ? " is-each" : ""}`}>${v[x.key][2]}</span>
      <${Spark} values=${h[x.series]} max=${x.max || max[x.key]} tone=${x.tone || ""}/></div>`)}</div>`;
}

// ------------------------------------------------------------------ recent requests
const API_NAMES = {web: "Chat", openai: "OpenAI", anthropic: "Anthropic"};
const FINISH = {stop: ["ok", "Done"], length: ["ok", "Max tokens"], cancel: ["warn", "Stopped"], disconnect: ["warn", "Closed"], error: ["danger", "Error"]};
function totalsLine(t, lg = {}) {
  if (!t || !t.requests) return "";
  // the loop guard: the loops it saw and what it did about them (/metrics' maya:loop_guard_* counters, here only; the
  // request rows say where), or that it is off
  const did = [[lg.penalised, "cut with a short penalty"], [lg.thinking_closed, "closed the thinking"],
               [lg.stopped, "ended the answer"]].filter(([n]) => n > 0).map(([n, s]) => `${fmt(n)} ${s}`);
  const loops = lg.detected ? ` · loop guard: ${fmt(lg.detected)} loop${lg.detected === 1 ? "" : "s"} seen${did.length ? ` (${did.join(", ")})` : ""}`
              : lg.on === false ? " · loop guard off" : "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  // (the requests' count is the hero's Requests served)
  return `Since ${since}: ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ${fmt(t.output_tokens)} written${oSpeed}${loops}`;
}
function Requests({m}) {
  const requests = m.requests || [], kept = m.requests_kept == null ? requests.length : m.requests_kept;
  const rows = requests.slice(0, reqAll.on ? requests.length : 12);
  return html`<section class="card" aria-label="Recent requests">
    <div class="card__head"><span class="card__title">Recent requests</span>
      ${kept > 12 ? html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => { reqAll.on = !reqAll.on; emit(); }}>
        ${reqAll.on ? "Show fewer" : `Show all ${kept}`}</button>` : null}</div>
    <div class=${`table-wrap${reqAll.on ? " is-all" : ""}`}><table class="table num">
      <thead><tr><th class="l">Time</th><th class="l">Status</th><th>Prompt</th><th title="Prompt tokens the conversation cache already held">Reused</th>
        <th title="Prompt tokens read per second (the reused ones not counted)">Prefill</th><th>Output</th>
        <th title="Answer tokens written per second">Decode</th><th title="While writing the answer: the share of the routed experts that were already in VRAM">VRAM hit</th><th>Took</th></tr></thead>
      <tbody>${rows.length ? rows.map((r) => {
        const [tone, text] = FINISH[r.finish] || ["muted", r.finish || "–"];
        const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
        const pre = r.prefill_tok_s != null ? r.prefill_tok_s
          : r.prompt_ms > 0 && r.prompt_tokens > (r.reused || 0) ? (r.prompt_tokens - (r.reused || 0)) / (r.prompt_ms / 1000) : null;
        return html`<tr><td class="l">${t}</td><td class="l nowrap"><span class=${`badge badge--${tone}`}>${text}</span>
            ${r.api ? html` <span class="muted small">${API_NAMES[r.api] || r.api}</span>` : null}
            ${r.loop_guard && r.loop_guard.length   /* (what the loop guard did in this answer: on hover) */
              ? html` <span class="badge badge--warn" title=${r.loop_guard.join("\n")}>Loop guard</span>` : null}</td>
          <td>${fmt(r.prompt_tokens)}</td><td>${fmt(r.reused)}</td><td>${fmt(pre)}</td><td>${fmt(r.output_tokens)}</td>
          <td class="strong">${fmt(r.decode_tok_s, 1)}</td><td>${r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`}</td>
          <td>${fmt(r.duration_s, 1)} s</td></tr>`;
      }) : html`<tr><td colspan="9" class="l muted">No requests yet</td></tr>`}</tbody>
    </table></div>
    ${m.totals && m.totals.requests ? html`<p class="card__note">${totalsLine(m.totals, m.loop_guard || {})}</p>` : null}
  </section>`;
}

// ------------------------------------------------------------------ the page
export function MonitorPage() {
  const m = S.metrics;
  const st = status(true), phone = isPhone();
  const copy = async () => {
    try { await copyText(await fetchReport(), true); toast("success", "Report copied", "Paste it into a GitHub issue - read it first."); }
    catch (e) { toast("error", "No report", e.message, 5000); }
  };
  // the GPU card's rows carry each GPU's readings (several GPUs): the hardware cards then keep their totals
  const reads = (m && (m.hardware || {}).gpus) || [];
  const rowsN = (m && m.gpu_map && m.gpu_map.length) || (reads.length >= GPU_PLAIN_FROM ? reads.length : 0);
  const gpuEach = rowsN > 1 && reads.length > 1;
  return html`<main class="mon">
    ${phone ? html`<header class="chat-head chat-head--float">
        <button type="button" class="glass-btn" aria-label="Chats" onClick=${() => { S.drawer = true; emit(); }}><${Icon} name="menu"/></button>
        <span class="model-btn model-btn--phone">Monitor</span>
        <button type="button" class="glass-btn" aria-label="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button></header>`
      : !S.sideOpen ? html`<header class="chat-head"><div class="chat-head__left">
        <button type="button" class="icon-btn" aria-label="Open the sidebar" title="Open the sidebar" onClick=${() => setSide(true)}><${Icon} name="panel-left"/></button>
        <button type="button" class="icon-btn" aria-label="New chat" title="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button></div></header>` : null}
    <div class="mon__scroll"><div class="mon__page">
      <div class="mon__head">
        <div><h1>Monitor</h1><p>The model, its experts and this machine, live.</p></div>
        <div class="mon__acts"><span class=${`pill pill--${st.tone}`} role="status"><i class="status__dot"></i>${st.text}</span>
          <button type="button" class="btn btn--secondary btn--sm" onClick=${copy}><${Icon} name="report" sm/>Copy report</button></div>
      </div>
      ${m ? html`
        <${Banners} m=${m}/>
        <div class="mon__top"><${Throughput} m=${m}/><${RightNow} m=${m}/></div>
        <${ContextCard} m=${m}/>
        <${Tiers} t=${m.tiers} eng=${m.engine || {}}/>
        <${GpuMap} m=${m}/>
        <${Hardware} m=${m} gpuEach=${gpuEach}/>
        <${Requests} m=${m}/>`
      : html`<div class="card mon__wait">${S.conn === "down" ? "The server is not reachable. It may be starting, or stopped."
                                        : S.conn === "key" ? "This server needs its API key: add it in Settings › Connect your tools."
                                        : "Waiting for the first reading…"}</div>`}
    </div></div>
  </main>`;
}
