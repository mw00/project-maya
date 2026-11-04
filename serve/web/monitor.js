// serve/web/monitor.js - the Monitor: the model's speed and what it is doing now, where its experts live, what fills each
// GPU, the hardware, and the recent requests - all from the server's /metrics (every second) and the engine's STAT lines.
import {html, useRef, useState} from "./preact.js";
import {GIB, ago, ctxfmt, fmt, gb, isPhone, kfmt, rate, store} from "./lib.js";
import {S, copyText, emit, toast} from "./state.js";
import {fetchReport, phaseText, reqAll, speculationView, tierShares} from "./server.js";
import {newChat} from "./chats.js";
import {setSide, status} from "./sidebar.js";
import {Icon} from "./ui.js";
import {languageChoice, tr} from "./i18n.js";

const dateLocale = () => languageChoice() === "zh-CN" ? "zh-CN" : "en";

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
    out.push(["info", `${tr("Reloading the model with a")} ${ctxfmt(rl.to)} ${tr("context")}${rl.state === "waiting" ? tr(", after the answer being written") : ""}${tr(". Requests wait until it is back.")}`]);
  } else if (rl && rl.state === "failed" && rl.ended && Date.now() / 1000 - rl.ended < 600) {
    out.push(["danger", `${tr("The context change did not work:")} ${rl.error || tr("the engine did not start")}${tr(". It runs with")} ${ctxfmt(rl.context)} ${tr("again.")}`]);
  }
  const eng = m.engine || {};
  if (eng.gpu_commit_capped_gb > 0) {   // (DeepSeek on a Windows APU: the page file kept experts off the GPU)
    out.push(["warn", `${fmt(eng.gpu_commit_capped_gb)} ${tr("GB of experts stay off the GPU: Windows' page file is too small to hold them there. A page file of")} ${fmt(eng.pagefile_hint_gb)} ${tr("GB keeps them on the GPU (System > About > Advanced system settings > Performance > Advanced > Virtual memory), then restart Maya.")}`]);
  }
  const gpus = (m.hardware || {}).gpus || [{index: null, temp: (m.hardware || {}).gpu_temp}];
  for (const g of gpus) {
    if (g.temp != null && g.temp >= 87) out.push(["warn", `${g.index == null ? tr("The GPU") : `GPU ${g.index}`} ${tr("is at")} ${fmt(g.temp)} °C${tr(": it may slow itself down to cool off.")}`]);
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
  // (the drafters on, and the share of their drafts the model kept: live from the engine's STAT lines, else the last answer's)
  const spec = speculationView(m);
  // the highest Decode figure since Maya started - live or a finished answer's (the server's, kept as each reading beats
  // it; from 32 tokens on: a few tokens' speed says little)
  const peak = m.totals && m.totals.peak_decode;
  const when = peak ? new Date(peak.time * 1000) : null;
  const at = when ? when.toLocaleString(dateLocale(), new Date().toDateString() === when.toDateString()
    ? {hour: "2-digit", minute: "2-digit"} : {weekday: "short", hour: "2-digit", minute: "2-digit"}) : "";
  return html`<section class="card card--wide" aria-label=${tr("Throughput")}>
    <div class="speeds">
      <div class="speed"><span class="speed__label"><i class="key key--line"></i>${tr("Decode")} · ${many ? `${live.writing} ${tr("writing")}` : gen ? tr("writing now") : live.state === "reading" ? tr("reading the prompt") : last ? tr("last request") : tr("waiting for a request")}</span>
        <div class="speed__row"><span class="speed__big num">${speed == null ? "–" : fmt(speed, 1)}<small>tok/s</small></span>
          ${/* the peak beside the live Decode (below it on a phone), small: a reference, not a second headline */ ""}
          <span class="speed__peak num" title=${peak ? tr("The highest decode speed since Maya started: {speed} tok/s at {time}, {tokens} tokens into the answer. Live speeds and finished answers count, from 32 tokens on.", {speed: fmt(peak.tok_s, 1), time: at, tokens: fmt(peak.tokens)})
              : tr("The highest decode speed since Maya started. Live speeds and finished answers count, from 32 tokens on.")}>${tr("Peak")} ${peak ? `${fmt(peak.tok_s, 1)} tok/s` : "–"}</span></div></div>
      <div class="speed" title=${tr("Prompt tokens read per second (the prefill), counting only the tokens read - not the ones reused from the conversation cache")}>
        <span class="speed__label"><i class="key key--dash"></i>${tr("Prefill")} · ${live.state === "reading" ? tr("reading now") : gen ? tr("this request") : last ? tr("last request") : tr("waiting")}</span>
        <span class="speed__mid num">${prefill == null ? "–" : fmt(prefill)}<small>tok/s</small></span></div>
    </div>
    <div class=${`spark-wrap${a || b ? "" : " is-empty"}`}>
      <svg class="spark spark--hero" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
        ${a ? html`<path class="spark__area" d=${a.area}/><path class="spark__line" d=${a.line} vector-effect="non-scaling-stroke"/>` : null}
        ${b ? html`<path class="spark__line spark__line--dash" d=${b.line} vector-effect="non-scaling-stroke"/>` : null}</svg>
      <span class="spark-wrap__empty">${tr("No activity in the last minute")}</span>
    </div>
    <div class="stats">
      <div class="stat"><span class="stat__label">${tr("Time per token")}</span><${Num} v=${ms == null ? null : fmt(ms, 1)} unit="ms"/></div>
      <div class="stat"><span class="stat__label">${tr("VRAM hit")}</span><${Num} v=${hit == null ? null : fmt(hit, 1)} unit="%"/></div>
      <div class="stat"><span class="stat__label">${tr("Requests served")}</span><${Num} v=${m.totals && m.totals.requests != null ? fmt(m.totals.requests) : null}/></div>
      <div class="stat stat--spec" title=${spec.state === "off" ? tr("No drafter is running for this answer: every token comes from a full model step")
          : tr("Drafting: a small drafter proposes tokens and the full model checks them. Shown: the drafts the model kept of those offered, live while an answer is written, else the last answer's. A dash: nothing drafted yet.")}>
        <span class="stat__label">${tr("Speculation")}</span>
        ${/* the share kept is the figure; the drafter (DSpark / MTP / DFlash2 - the prompt lookup beside it is not named) the note */ ""}
        <span class="stat__value num">${spec.state === "kept" ? spec.short : spec.state === "off" ? tr("off") : "–"}</span>
        ${spec.state === "none" || spec.state === "off" ? null : html`<span class="stat__note">${drafterName(spec.name)}${spec.state === "kept" ? ` · ${tr("drafts kept")}` : ""}</span>`}
        ${spec.state === "kept" && spec.detail ? html`<span class="stat__note num">${spec.detail}</span>` : null}</div>
    </div>
  </section>`;
}

// The Speculation note's drafter: the model drafter alone ("DSpark + Lookup" -> "DSpark"), the lookup only when it drafts alone
function drafterName(name) {
  const parts = String(name || "").split(" + ").filter((p) => p && p !== "Lookup");
  return parts.length ? parts.join(" + ") : name || "";
}

function RightNow({m}) {
  const live = m.live || {}, last = (m.requests || [])[0], eng = m.engine || {};
  const took = (s) => (s == null ? "" : s < 90 ? ` · ${fmt(s, s < 10 ? 1 : 0)} s` : ` · ${fmt(Math.floor(s / 60))} min ${fmt(s % 60)} s`);
  let label = tr("Idle"), tone = "muted", detail = tr("waiting for a request"), pct = null, ptone = "accent";
  if (S.reload) {
    label = tr("Reloading"); tone = "info"; ptone = "info"; detail = `${tr("a")} ${ctxfmt(S.reload.to)} ${tr("context")}`;
  } else if (live.state === "reading") {
    label = tr("Reading"); tone = "info"; ptone = "info";
    if (live.prompt_total) { pct = (100 * live.prompt_read) / live.prompt_total; detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} ${tr("prompt tokens")}`; }
    else detail = `${fmt(live.prompt_tokens)} ${tr("prompt tokens")}`;
  } else if (live.state === "generating") {
    label = live.phase ? phaseText(live.phase) : tr("Writing"); tone = "accent";
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : null;
    detail = `${fmt(live.generated)} ${tr("tokens")}${took(live.elapsed_s)}`;
  } else if (last) {
    detail = `${tr("last answer:")} ${fmt(last.output_tokens)} ${tr("tokens")}${took(last.duration_s)}`;
  }
  const running = live.state !== "idle";
  return html`<section class="card card--side" aria-label=${tr("Right now")}>
    <span class="card__title">${tr("Right now")}</span>
    <div class="now"><${Badge} tone=${tone}>${label}<//><span class="now__detail num">${detail}</span></div>
    ${pct != null ? html`<div class="bar" data-tone=${ptone} role="progressbar" aria-valuenow=${Math.round(pct)} aria-valuemin="0" aria-valuemax="100"
      aria-label=${label}><span style=${`width:${pct}%`}></span></div>` : null}
    <dl class="kv">
      ${running && live.prompt_total ? html`<dt>${tr("Prompt read")}</dt><dd class="num">${fmt(live.prompt_read)} ${tr("of")} ${fmt(live.prompt_total)}</dd>`
        : last ? html`<dt>${tr("Reused from the cache")}</dt><dd class="num">${fmt(last.reused || 0)} ${tr("tokens")}</dd>` : null}
      <dt>${tr("Answers at once")}</dt><dd class="num">${fmt(live.active || 0)}</dd>
      <dt>${tr("Waiting")}</dt><dd class="num">${live.queued > 0 ? fmt(live.queued) : tr("none")}</dd>
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
    what = live.active > 1 ? `${tr("the newest of")} ${fmt(live.active)} ${tr("answers being written")}` : tr("the request being answered");
  } else if (last) {
    prompt = last.prompt_tokens || 0; reused = last.reused; written = last.output_tokens || 0; what = tr("the last request");
  }
  const r = reused == null ? null : Math.min(reused, prompt);
  const segs = [...(r == null ? [] : [{k: "reused", label: "Reused from the cache", n: r}]),
                {k: "fresh", label: r == null ? "Prompt" : "Read new", n: prompt - (r || 0)}, {k: "written", label: "Written", n: written}];
  const used = Math.min(max, prompt + written), free = Math.max(0, max - used), pct = (100 * used) / max;
  const win = +eng.kv_resident || 0, showWin = win > 0 && win < max;
  // (the VRAM window is KV streaming's --kv-resident: Models > Engine settings > Prompts and cache, GLM and MiniMax alike)
  const W = ctxfmt(win), where = showWin
    ? tr("The window is a setting: Models › Engine settings › Prompts and cache › Positions each layer keeps in VRAM.")
    : tr("The size is a setting: Settings › Model › Context size.");
  const note = !running && !last ? tr("No conversation yet: the context fills as a chat goes on, with every message, file and tool result in it.")
    : showWin ? (used > win ? tr("This chat is past the {window} VRAM window: its cache is kept in RAM (about {ram} GB), {window} tokens of it in GPU memory and the rest copied in when needed.", {window: W, ram: fmt((used * 12288) / 1e9, 1)})
                            : tr("VRAM window: a chat up to {window} tokens keeps its whole cache in GPU memory. A longer one keeps it in RAM, {window} tokens of it in GPU memory and the rest copied in when needed.", {window: W}))
    : tr("What the model holds at once: the chat, its files and tool results. A chat longer than this is refused: start a new one, or give the model a bigger context.");
  const aria = tr("{used} of {max} tokens in use", {used: fmt(used), max: fmt(max)}) + segs.map((s) => `, ${tr(s.label)} ${fmt(s.n)}`).join("");
  return html`<section class="card" aria-label=${tr("Context")}>
    <div class="card__head"><span class="card__title">${tr("Context")}</span><span class="card__sub">${what || ""}</span></div>
    <div class="ctx__nums"><span class="ctx__big num">${fmt(used)}<small>/ ${fmt(max)} ${tr("tokens")}</small></span>
      <span class="ctx__pct num"><b>${pct > 0 && pct < 0.1 ? "< 0.1" : fmt(pct, pct > 0 && pct < 10 ? 1 : 0)}%</b> ${tr("used")} · ${fmt(free)} ${tr("left")}</span></div>
    <div class=${`ctx__track${showWin ? " ctx__track--win" : ""}`}>
      ${showWin ? html`<span class="ctx__win" style=${`left:${(100 * win) / max}%`}><span>${tr("VRAM window")} · ${ctxfmt(win)}</span></span>` : null}
      <div class="ctxbar" role="img" aria-label=${aria}>${segs.filter((s) => s.n > 0).map((s) => html`<span class=${`ctxbar__${s.k}`} style=${`flex-grow:${s.n}`}></span>`)}
        <span style=${`flex-grow:${free}`}></span></div></div>
    <div class="legend">${segs.map((s) => html`<span><i class=${`sw sw--${s.k}`}></i>${tr(s.label)} <b class="num">${fmt(s.n)}</b></span>`)}
      <span><i class="sw sw--free"></i>${tr("Free")} <b class="num">${fmt(free)}</b></span></div>
    <p class="card__note card__note--tip">${note} ${where}</p>
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
  // CYBER-FROST (the maya engine's Qwen path): the CPU computes the experts VRAM does not hold, from RAM or read from the
  // model file; the drafts' windows count too
  qwen: [
    {key: "ram_fetch", label: "From RAM", unit: "/tok", digits: 1, tone: "info", sub: "experts the CPU computed from RAM"},
    {key: "disk", label: "From SSD", unit: "/tok", digits: 2, tone: "warn", sub: "experts read from the model file"},
  ],
};
function Tiers({t, eng}) {
  if (!t || !t.now) return null;
  const ds4 = !!(eng && eng.engine_kind === "ds4"), mm3 = !!(eng && eng.engine_kind === "mm3");
  const qwen = !!(eng && eng.engine_kind === "qwen");
  const set = TIER_SETS[ds4 ? "ds4" : mm3 ? "mm3" : qwen ? "qwen" : "glm"];
  const n = t.now, h = t.history || {};
  const helper = ds4 && ((n.helper_held || 0) > 0 || (n.helper || 0) > 0 || +eng.helper_allocated_bytes > 0);
  const helperHeld = n.helper_held != null ? n.helper_held : +eng.helper_slots || 0;
  const age = n.time ? Math.max(0, Date.now() / 1000 - n.time) : null;
  const stale = age != null && age > 30;               // a reading from the last answer, not live
  // VRAM, RAM and the rest on the SSD - DeepSeek's (maya-ds4) the same keys: its RAM the copies or the read-ahead
  const {vu, ru, total, disk, vramGb, ramGb, diskGb} = tierShares(n, eng || {});
  let tip = "";
  if (ds4) {
    if (!stale && n.disk_mb != null && n.disk_mb >= 500) tip = tr("Much of every token is read from the SSD: more RAM keeps more of the model's experts cached, and answers get faster.");
    else if (!stale && n.vram_hit != null && n.vram_hit < 0.5)
      tip = helper && !n.helper_dead ? tr("Most selections miss the primary cache; the helper handles some before CPU fallback.")
                   : tr("Most experts run on the CPU: more VRAM (or a second GPU) keeps more of them on the card.");
  } else if (!stale && n.disk != null && n.disk >= 3) {
    tip = tr("Many experts are read from the SSD for every token: more RAM, a smaller model or a shorter context makes answers faster.");
  } else if (mm3 || qwen) {   // (MiniMax, CYBER-FROST: the CPU computes the experts VRAM does not hold)
    if (!stale && n.vram_hit != null && n.vram_hit < 0.5) tip = tr("Most experts run on the CPU: more VRAM (or a second GPU) keeps more of them on the card.");
  } else if (!stale && n.vram_hit != null && n.vram_hit < 0.85 && (n.ram_fetch || 0) >= 4) {
    tip = tr("Most misses come from RAM over PCIe: more VRAM (or a second GPU) keeps more experts on the card.");
  }
  const pc = (x) => (100 * x) / total;
  return html`<section class=${`card${stale ? " is-stale" : ""}`} aria-label=${tr("Where the experts live")}>
    <div class="card__head"><span class="card__title">${helper ? tr("Primary cache and RAM backing") : tr("Where the experts live")}</span>
      <span class="card__sub">${age == null ? "" : stale ? `${tr("last reading")} ${ago(age)} ${tr("ago, from the last answer")}` : `${tr("live")} · ${fmt(total)} ${tr("routed experts")}`}</span></div>
    <div class="tierbar" role="img" aria-label=${tr("{vram}% of the experts in VRAM, {ram}% in {ramKind}, {disk}% on the SSD only", {vram: fmt(pc(vu), 0), ram: fmt(pc(ru), 0), ramKind: tr(ds4 ? "RAM" : "pinned RAM"), disk: fmt(pc(disk), 0)})}>
      <span class="tierbar__vram" style=${`flex-grow:${vu}`}></span><span class="tierbar__ram" style=${`flex-grow:${ru}`}></span>
      <span class="tierbar__disk" style=${`flex-grow:${disk}`}></span></div>
    <div class="legend">
      <span><i class="sw sw--vram"></i>${tr(helper ? "Primary VRAM" : "VRAM")} <b class="num">${fmt(vu)} · ${fmt(vramGb, 1)} GB</b></span>
      <span><i class="sw sw--ram"></i>${tr(ds4 ? "RAM" : "Pinned RAM")} <b class="num">${fmt(ru)} · ${fmt(ramGb, 1)} GB</b></span>
      <span><i class="sw sw--disk"></i>${tr("SSD only")} <b class="num">${fmt(disk)}${diskGb == null ? "" : ` · ${fmt(diskGb, 1)} GB`}</b></span></div>
    <div class=${`tiles${set.length === 4 ? " tiles--4" : ""}`}>
      ${set.map((x) => {
        const v = n[x.key];
        let sub = x.sub, subText = null, tip = null;
        if (x.key === "engram_ms" && eng && eng.engram_lookups) subText = tr("{count} n-gram lookups a token", {count: fmt(+eng.engram_lookups)});
        if (x.key === "engram_ms" && n.eg_lookup_rows > 0) {   // (its host cache, #522: the share found in RAM)
          const cacheHit = tr("{percent}% from its RAM cache", {percent: fmt((100 * (n.eg_lookup_hits || 0)) / n.eg_lookup_rows, 0)});
          subText = subText ? `${subText} · ${cacheHit}` : cacheHit;
          tip = tr("N-gram memory's RAM cache: {held} of {capacity} rows held; {lookups} lookups since the start, {hits} found in RAM, {reads} read from the SSD", {
            held: fmt(n.eg_cache_rows || 0), capacity: fmt(n.eg_cache_cap || 0), lookups: fmt(n.eg_lookup_rows),
            hits: fmt(n.eg_lookup_hits || 0), reads: fmt(n.eg_lookup_reads || 0),
          });
        }
        if ((x.key === "disk" || x.key === "ram_fetch") && v == null && ds4) subText = tr("not measured on this system");
        return html`<div class="tile" title=${tip}><span class="stat__label">${tr(x.label)}</span>
          <${Num} v=${v == null ? null : fmt(v, x.digits)} unit=${x.unit}/>
          <span class="tile__sub">${stale ? `${tr("last answer:")} ` : ""}${subText || tr(sub)}</span>
          <${Spark} values=${(h[x.key] || []).map((y) => (y == null ? 0 : y))} tone=${x.tone || ""}/></div>`;
      })}
    </div>
    ${helper ? html`<p class="card__note">${n.helper_dead ? tr("Helper unavailable; its selections fall back to the CPU.")
      : tr("Helper GPU: {count} profiled cached experts{selections}.", {count: fmt(helperHeld), selections: n.helper == null ? "" : ` · ${tr("{count} selections / evaluated token", {count: fmt(n.helper, 2)})}`})}
      ${tr("The bar shows primary VRAM and RAM backing. The helper pool is separate; CPU backing remains available for fallback.")}</p>` : null}
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
  {key: "drafter", label: "Drafter"}, {key: "pool", label: "Expert pool"}, {key: "helper", label: "Helper experts"},
  {key: "vision", label: "Vision encoder"},
  {key: "rest", label: "CUDA and the rest"}, {key: "other", label: "Other programs"},
];
const GPU_PLAIN_FROM = 5;
const gpuKey = (g) => String(g.index != null ? g.index : `d${g.dev}`);
const gpuLabel = (g) => `GPU ${g.index ?? g.dev}`;
const gpuName = (g) => (g.name || "").replace(/^NVIDIA /, "").replace(/^GeForce /, "").replace(/^AMD /, "").replace(/^Radeon /, "");
const gpuGb = (b) => `${gb(b, b < GIB ? 2 : 1)} GB`;
const gpuList = (gs) => gs.length === 1 ? gpuLabel(gs[0])
  : tr("GPUs {indices} and {last}", {indices:gs.slice(0,-1).map((g)=>g.index ?? g.dev).join(", "),last:gs[gs.length-1].index ?? gs[gs.length-1].dev});
// the layers a GPU holds: the GLM engine's VRAM lines say [first, end), DeepSeek's and MiniMax's (maya-ds4, maya-mm3)
// [first, last]
const dsparkEngine = (eng) => eng.engine_kind === "ds4" || eng.engine_kind === "mm3";
function gpuLayers(g, eng) {
  if (!g.layers) return "";
  const [a, b] = g.layers, last = dsparkEngine(eng) ? b : b - 1;
  return a < 0 || last < a ? "" : last > a ? tr("layers {first}–{last}", {first:a,last}) : tr("layer {layer}", {layer:a});
}
// the KV cache streams from RAM only beyond its VRAM window (DeepSeek's engine: a window as long as the context)
const kvWindow = (eng, ctx) => (+eng.kv_resident > 0 && +eng.kv_resident < ctx ? +eng.kv_resident : 0);
function gpuSegs(g, eng) {
  const p = g.parts || {}, draft = eng.dflash ? "DFlash2" : +eng.mtp ? "MTP" : dsparkEngine(eng) ? "DSpark" : "";
  return GPU_PARTS.filter((d) => (p[d.key] || 0) > 0 || d.key === "pool" || d.key === "kv")
    .map((d) => ({key: d.key, bytes: p[d.key] || 0, label: d.label, draft: d.key === "drafter" ? draft : ""}))
    .concat([{key: "free", label: "Free", bytes: g.free || 0}]);
}
const gpuSegmentLabel = (s) => s.draft ? tr("Drafter ({draft})", {draft: s.draft}) : tr(s.label);
// a GPU's NVML readings under its bar: load, power, temperature (warm from 80 °C, hot from 87) and PCIe link
function GpuStats({r}) {
  if (!r) return null;
  const gen = r.pcie_gen_max || r.pcie_gen, mb = r.pcie_rx_mb;
  const heat = r.temp >= 87 ? "is-hot" : r.temp >= 80 ? "is-warm" : "";
  return html`<span class="gpustat num">
    ${r.util != null ? html`<span><b>${fmt(r.util)}%</b> ${tr("load")}</span>` : null}
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
    weights: tr("attention, shared experts and the other dense tensors"),
    kv: (kvWindow(eng, ctx) ? tr("{window} of {context} positions here, the rest in RAM", {window: ctxfmt(kvWindow(eng, ctx)), context: ctxfmt(ctx)}) : tr("{context} positions", {context: ctxfmt(ctx)})) +
        (inUse ? ` · ${tr("{count} in use", {count: kfmt(inUse)})}` : "") + (g.kv_ram ? ` · ${tr("{gb} GB pinned in RAM", {gb: gb(g.kv_ram)})}` : ""),
    act: tr("one token's working buffers"),
    prompt: tr("a long prompt's chunks: their working buffers and the ring the CPU lane's experts stream through"),
    tp: g.tp_n > 1 ? tr("the copies a group of {count} GPUs needs: the tensors each of them holds whole, and its layers' KV cache", {count: g.tp_n}) : "",
    drafter: [g.drafter ? tr("weights {gb} GB", {gb: gb(g.drafter, 2)}) : "", g.window ? tr("verify window {gb} GB", {gb: gb(g.window, 2)}) : ""].filter(Boolean).join(" · "),
    pool: [g.slots ? tr("{slots} expert slots", {slots: fmt(g.slots)}) : "", g.borrow ? tr("prompts borrow up to {gb} GB of it", {gb: gb(g.borrow)}) : ""].filter(Boolean).join(" · "),
    helper: tr("separate helper expert pool; CPU backing available for fallback"),
    vision: g.vision_lent ? tr("the pool's tail, while it encodes") : g.vision_own ? tr("the vision tower, in the engine's process") : tr("its own process"),
    rest: tr("the driver's reserve and context, conversation snapshots and small buffers"),
    other: tr("processes that are not Maya's"),
    free: (g.free || 0) < 1.5 * GIB ? tr("kept free for the driver and the sampler") : tr("unused"),
    used: "",
  };
  return html`<div class="gpulegend">${segs.map((s) => html`<div class="gpulegend__item" data-k=${s.key}>
    <i class=${`sw${s.key === "free" ? " sw--free" : ""}`} style=${`--gm-c:var(--gm-${s.key})`}></i><span>${gpuSegmentLabel(s)}</span>
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
  for (const r of rows) for (const s of r.segs) { tot[s.key] = (tot[s.key] || 0) + s.bytes; label[s.key] = s; }
  // what is happening now: a prompt borrows the pools' tails, the vision encoder GPU 0's
  const vis = map.find((g) => g.vision_lent);
  const lentNow = reading ? sum((g) => !g.vision_lent && g.borrow > 0 && (g.parts || {}).pool > 0 ? Math.min(g.borrow, g.parts.pool) : 0) : 0;
  const now = vis ? tr(one ? "The vision encoder has {gb} GB of the expert pool while it encodes." : "The vision encoder has {gb} GB of {gpu}'s expert pool while it encodes.", {gb: gb(vis.parts.vision || vis.lendable), gpu: gpuLabel(vis)})
            : lentNow ? tr(one ? "Reading a prompt: the expert pool lends it up to {gb} GB (hatched)." : "Reading a prompt: the expert pools lend it up to {gb} GB (hatched).", {gb: gb(lentNow)}) : "";
  // advice: memory the expert pool could have at the next start
  const complete = eng.expert_slots && eng.experts && eng.expert_slots >= eng.experts;
  const more = (g, b) => (g.slots && g.parts.pool ? Math.max(0, Math.floor(b / (g.parts.pool / g.slots))) : 0);
  const where = (gs) => (one ? tr("this GPU") : gpuList(gs));
  const pools = (gs) => (one ? tr("the expert pool") : gs.length > 1 ? tr("their expert pools") : tr("its expert pool"));
  // A dedicated helper (or drafter-only device) has no primary pool to grow on restart.
  const hasPrimaryPool = (g) => g.slots > 0 && (g.parts.pool || 0) > 0;
  const others = complete || plain ? [] : map.filter((g) => hasPrimaryPool(g) && (g.parts.other || 0) >= 0.5 * GIB);
  const frees = complete || plain ? [] : map.filter((g) => hasPrimaryPool(g) && !g.vision_lent && (g.free || 0) >= 2 * GIB);
  let advice = "";
  if (others.length) {
    const b = others.reduce((a, g) => a + g.parts.other, 0), n = others.reduce((a, g) => a + more(g, g.parts.other), 0);
    advice = tr("Other programs hold {gb} GB on {where}: with them closed, a restart gives {pools} about {count} more experts.", {gb: gb(b), where: where(others), pools: pools(others), count: fmt(n)});
  } else if (frees.length) {
    const b = frees.reduce((a, g) => a + g.free, 0), n = frees.reduce((a, g) => a + more(g, g.free - GIB), 0);
    advice = tr("{gb} GB of {where} are free: a restart gives {pools} about {count} more experts.", {gb: gb(b), where: where(frees), pools: pools(frees), count: fmt(n)});
  }
  const pick = (k) => { const v = open === k ? null : k; setOpen(v); store.set("gpu-open", v); };
  const move = (e, r) => {
    const seg = e.target.closest(".gpubar__seg");
    const s = seg && r.segs.find((x) => x.key === seg.dataset.k);
    setFocus(s && s.key !== "free" ? s.key : null);
    if (!s) { setTip(null); return; }
    const c = card.current.getBoundingClientRect(), bar = seg.parentElement.getBoundingClientRect();
    setTip({x: Math.min(Math.max(e.clientX - c.left, 130), c.width - 130), y: bar.top - c.top,
            text: tr("{gpu}{label} · {gb} · {pct}% of the card", {gpu: rows.length > 1 ? `${gpuLabel(r.g)} · ` : "", label: gpuSegmentLabel(s), gb: gpuGb(s.bytes), pct: fmt((100 * s.bytes) / (r.g.total || 1), 1)})});
  };
  const heading = plain ? "GPUs" : one ? "What fills the GPU" : "What fills the GPUs";
  return html`<section ref=${card} class=${`card gpumap${one ? " is-one" : ""}`} aria-label=${tr(heading)}>
    <div class="card__head"><span class="card__title">${tr(heading)}</span>
      <span class="card__sub">${[one ? "" : tr("{count} GPUs", {count: map.length}), tr("{used} of {total} GB in use", {used: gb(used), total: gb(total, 0)}),
                                 map.some((g) => g.measured === "start") ? tr("as measured at the start") : ""].filter(Boolean).join(" · ")}</span></div>
    ${one || plain ? null : html`<div class="legend legend--sm" onPointerLeave=${() => setFocus(null)}>
      ${GPU_PARTS.map((d) => d.key).concat(["free"]).filter((k) => tot[k] > 0).map((k) => html`
        <span class="legend__item" onPointerOver=${() => setFocus(k === "free" ? null : k)}>
          <i class=${`sw${k === "free" ? " sw--free" : ""}`} style=${`--gm-c:var(--gm-${k})`}></i>${gpuSegmentLabel(label[k])} <b class="num">${gpuGb(tot[k])}</b></span>`)}</div>`}
    <div class="gpurows" data-focus=${focus || undefined} onPointerLeave=${() => { setFocus(null); setTip(null); }}>
      ${rows.map((r) => {
        const g = r.g, p = g.parts || {}, o = isOpen(r);
        const lent = reading && !g.vision_lent && g.borrow > 0 && p.pool > 0 ? Math.min(100, (100 * g.borrow) / p.pool) : 0;
        const line = html`
          <span class="gpurow__name"><b>${gpuLabel(g)}</b><span>${[gpuName(g), gpuLayers(g, eng)].filter(Boolean).join(" · ")}</span></span>
          <span class="gpurow__track"><span class="gpubar" role="img" onPointerMove=${(e) => move(e, r)}
              aria-label=${tr("{gpu}: {parts}", {gpu: gpuLabel(g), parts: r.segs.map((s) => `${gpuSegmentLabel(s)} ${gpuGb(s.bytes)}`).join(", ")})}>
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
    gpu: [hw.gpu_util == null ? null : fmt(hw.gpu_util), "%", each ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : multi ? tr("mean of {count} cards", {count: gpus.length}) : st.gpu_name || ""],
    vram: [hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
           each ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`)) : [multi ? tr("{count} cards", {count: gpus.length}) : "", eng.expert_slots ? tr("{count} experts cached", {count: fmt(eng.expert_slots)}) : ""].filter(Boolean).join(" · ")],
    temp: [hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C", each ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : multi ? tr("hottest of {count} cards", {count: gpus.length}) : ""],
    power: [hw.gpu_power == null ? null : fmt(hw.gpu_power), "W",
            each ? per((g) => (g.power == null ? "–" : `${fmt(g.power)}${g.power_limit ? `/${fmt(g.power_limit)}` : ""} W`))
                 : hw.gpu_power_limit ? tr(multi ? "of {watts} W limit (all {count} cards)" : "of {watts} W limit", {watts: fmt(hw.gpu_power_limit), count: gpus.length}) : ""],
    pcie: [gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
           each ? per((g) => `${sameLink ? "" : `${link(g)} `}${rate(g.pcie_rx_mb)}`)
                : hw.gpu_pcie_rx_mb == null ? "" : [tr("to GPU {rate}", {rate: rate(hw.gpu_pcie_rx_mb)}), hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? tr("idle Gen{generation}", {generation: hw.gpu_pcie_gen}) : ""].filter(Boolean).join(" · ")],
    cpu: [hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? [st.cores ? tr("{count} cores", {count: st.cores}) : "", tr("{count} threads", {count: st.threads})].filter(Boolean).join(" · ") : ""],
    ram: [hw.ram_used == null ? null : gb(hw.ram_used), hw.ram_total ? `/ ${gb(hw.ram_total, 0)} GB` : "GB", eng.ram_gb ? tr("{gb} GB pinned for experts", {gb: fmt(+eng.ram_gb, 1)}) : ""],
    disk: hw.disk_read_mb == null ? [null, "", st.psutil ? "" : tr("needs psutil (setup installs it)")]
      : [big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
         hw.disk_write_mb == null ? "" : tr("write {rate} MB/s", {rate: fmt(hw.disk_write_mb, 1)})],
  };
  const max = {vram: hw.gpu_mem_total, temp: 90, power: hw.gpu_power_limit, ram: hw.ram_total};
  return html`<h2 class="mon__h2">${tr("Hardware")}</h2>
    <div class="hw">${METRICS.map((x) => html`<div class="card hw__card" title=${x.tip ? tr(x.tip) : ""}>
      <span class="stat__label">${tr(x.label)}</span><${Num} v=${v[x.key][0]} unit=${v[x.key][1]}/>
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
               [lg.stopped, "ended the answer"]].filter(([n]) => n > 0).map(([n, s]) => tr("{count} {action}", {count: fmt(n), action: tr(s)}));
  // (and across turns: a tool call made again and again with the same result, a note after it)
  const tools = [lg.tool_repeats ? tr("{count} repeated tool calls noted", {count: fmt(lg.tool_repeats)}) : "",
                 lg.tool_blocked ? tr("{count} blocked", {count: fmt(lg.tool_blocked)}) : "",
                 lg.kept_going ? tr("{count} kept going", {count: fmt(lg.kept_going)}) : "",
                 lg.collapsed ? tr("{count} prompts without the repeats", {count: fmt(lg.collapsed)}) : ""]
    .filter(Boolean).join(", ");
  const loops = lg.detected ? tr(" · loop guard: {count} loops seen{actions}{tools}", {count: fmt(lg.detected), actions: did.length ? ` (${did.join(", ")})` : "", tools: tools ? `, ${tools}` : ""})
              : tools ? tr(" · loop guard: {tools}", {tools}) : lg.on === false ? tr(" · loop guard off") : "";
  const since = new Date(t.since * 1000).toLocaleString(dateLocale(), {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? tr(" at {speed} tok/s", {speed: fmt(read / (t.prompt_ms / 1000))}) : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? tr(" at {speed} tok/s", {speed: fmt(t.output_tokens / (t.decode_ms / 1000), 1)}) : "";
  // (the requests' count is the hero's Requests served)
  return tr("Since {since}: {read} prompt tokens read{prefill} ({reused} reused) · {written} written{decode}{loops}", {since, read: fmt(read), prefill: pSpeed, reused: fmt(t.reused), written: fmt(t.output_tokens), decode: oSpeed, loops});
}
function Requests({m}) {
  const requests = m.requests || [], kept = m.requests_kept == null ? requests.length : m.requests_kept;
  const rows = requests.slice(0, reqAll.on ? requests.length : 12);
  return html`<section class="card" aria-label=${tr("Recent requests")}>
    <div class="card__head"><span class="card__title">${tr("Recent requests")}</span>
      ${kept > 12 ? html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => { reqAll.on = !reqAll.on; emit(); }}>
        ${reqAll.on ? tr("Show fewer") : tr("Show all {count}", {count: kept})}</button>` : null}</div>
    <div class=${`table-wrap${reqAll.on ? " is-all" : ""}`}><table class="table num">
      <thead><tr><th class="l">${tr("Time")}</th><th class="l">${tr("Status")}</th><th>${tr("Prompt")}</th><th title=${tr("Prompt tokens the conversation cache already held")}>${tr("Reused")}</th>
        <th title=${tr("Prompt tokens read per second (the reused ones not counted)")}>${tr("Prefill")}</th><th>${tr("Output")}</th>
        <th title=${tr("Answer tokens written per second")}>${tr("Decode")}</th><th title=${tr("While writing the answer: the share of the routed experts that were already in VRAM")}>${tr("VRAM hit")}</th><th>${tr("Took")}</th></tr></thead>
      <tbody>${rows.length ? rows.map((r) => {
        const [tone, text] = FINISH[r.finish] || ["muted", r.finish || "–"];
        const t = new Date(r.time * 1000).toLocaleTimeString(dateLocale(), {hour: "2-digit", minute: "2-digit", second: "2-digit"});
        const pre = r.prefill_tok_s != null ? r.prefill_tok_s
          : r.prompt_ms > 0 && r.prompt_tokens > (r.reused || 0) ? (r.prompt_tokens - (r.reused || 0)) / (r.prompt_ms / 1000) : null;
        return html`<tr><td class="l">${t}</td><td class="l nowrap"><span class=${`badge badge--${tone}`}>${FINISH[r.finish] ? tr(text) : text}</span>
            ${r.api ? html` <span class="muted small">${API_NAMES[r.api] ? tr(API_NAMES[r.api]) : r.api}</span>` : null}
            ${r.loop_guard && r.loop_guard.length   /* (what the loop guard did in this answer: on hover) */
              ? html` <span class="badge badge--warn" title=${r.loop_guard.join("\n")}>${tr("Loop guard")}</span>` : null}</td>
          <td>${fmt(r.prompt_tokens)}</td><td>${fmt(r.reused)}</td><td>${fmt(pre)}</td><td>${fmt(r.output_tokens)}</td>
          <td class="strong">${fmt(r.decode_tok_s, 1)}</td><td>${r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`}</td>
          <td>${fmt(r.duration_s, 1)} s</td></tr>`;
      }) : html`<tr><td colspan="9" class="l muted">${tr("No requests yet")}</td></tr>`}</tbody>
    </table></div>
    ${m.totals && m.totals.requests ? html`<p class="card__note">${totalsLine(m.totals, m.loop_guard || {})}</p>` : null}
  </section>`;
}

// ------------------------------------------------------------------ the page
function monitorStatusText(text) {
  const queued = /^(\d+) queued$/.exec(text);
  if (queued) return tr("{count} queued", {count: queued[1]});
  const idle = /^Idle(?: · (.*))?$/.exec(text);
  if (idle) return [tr("Idle"), idle[1]].filter(Boolean).join(" · ");
  return tr(text);
}
export function MonitorPage() {
  const m = S.metrics;
  const st = status(true), phone = isPhone();
  const copy = async () => {
    try { await copyText(await fetchReport(), true); toast("success", tr("Report copied"), tr("Paste it into a GitHub issue - read it first.")); }
    catch (e) { toast("error", tr("No report"), e.message, 5000); }
  };
  // the GPU card's rows carry each GPU's readings (several GPUs): the hardware cards then keep their totals
  const reads = (m && (m.hardware || {}).gpus) || [];
  const rowsN = (m && m.gpu_map && m.gpu_map.length) || (reads.length >= GPU_PLAIN_FROM ? reads.length : 0);
  const gpuEach = rowsN > 1 && reads.length > 1;
  return html`<main class="mon">
    ${phone ? html`<header class="chat-head chat-head--float">
        <button type="button" class="glass-btn" data-open-chats aria-label=${tr("Chats")} onClick=${() => { S.drawer = true; emit(); }}><${Icon} name="menu"/></button>
        <span class="model-btn model-btn--phone">${tr("Monitor")}</span>
        <button type="button" class="glass-btn" aria-label=${tr("New chat")} onClick=${newChat}><${Icon} name="new-chat"/></button></header>`
      : !S.sideOpen ? html`<header class="chat-head"><div class="chat-head__left">
        <button type="button" class="icon-btn" aria-label=${tr("Open the sidebar")} title=${tr("Open the sidebar")} onClick=${() => setSide(true)}><${Icon} name="panel-left"/></button>
        <button type="button" class="icon-btn" aria-label=${tr("New chat")} title=${tr("New chat")} onClick=${newChat}><${Icon} name="new-chat"/></button></div></header>` : null}
    <div class="mon__scroll"><div class="mon__page">
      <div class="mon__head">
        <div><h1>${tr("Monitor")}</h1><p>${tr("The model, its experts and this machine, live.")}</p></div>
        <div class="mon__acts"><span class=${`pill pill--${st.tone}`} role="status"><i class="status__dot"></i>${monitorStatusText(st.text)}</span>
          <button type="button" class="btn btn--secondary btn--sm" onClick=${copy}><${Icon} name="report" sm/>${tr("Copy report")}</button></div>
      </div>
      ${m ? html`
        <${Banners} m=${m}/>
        <div class="mon__top"><${Throughput} m=${m}/><${RightNow} m=${m}/></div>
        <${ContextCard} m=${m}/>
        <${Tiers} t=${m.tiers} eng=${m.engine || {}}/>
        <${GpuMap} m=${m}/>
        <${Hardware} m=${m} gpuEach=${gpuEach}/>
        <${Requests} m=${m}/>`
      : html`<div class="card mon__wait">${S.conn === "down" ? tr("The server is not reachable. It may be starting, or stopped.")
                                        : S.conn === "key" ? tr("This server needs its API key: add it in Settings › Connect your tools.")
                                        : tr("Waiting for the first reading…")}</div>`}
    </div></div>
  </main>`;
}
