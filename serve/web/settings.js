// serve/web/settings.js - the Settings dialog: General, Model (the context size and the port: each a restart), Chat
// defaults (sampling, thinking, shared with other apps), Instructions, Tools (MCP), Connect your tools, Data, About and
// updates.  A change applies at once; the two that restart the model ask first.
import {html, useEffect, useRef, useState} from "./preact.js";
import {MOD, ctxfmt, fmt, gb, modelName, store} from "./lib.js";
import {DEFAULTS, LEVELS, P, PRESETS, S, THINK_TITLES, copyText, emit, prefsKey, presetOf, savePrefs, setTheme, themeChoice,
        toast} from "./state.js";
import {confirmContext, confirmPort, ctxEstimate, ctxStops, fetchReport, loadCtxInfo, loadMcp, loadPortInfo, loadShared,
        loadUpdate, saveShared, startUpdate, syncShared} from "./server.js";
import {chatInstructions, clearBrowser, estimateTokens, updateCtxMeter} from "./chats.js";
import {codeBlock, onCodeClick} from "./md.js";
import {Icon, Kbd, ModelLogo, Seg, Switch, trapTab} from "./ui.js";

const SECTIONS = [
  {id: "general", label: "General", icon: "settings"},
  {id: "model", label: "Model", icon: "cpu"},
  {id: "chat", label: "Chat defaults", icon: "gauge"},
  {id: "instr", label: "Instructions", icon: "edit"},
  {id: "tools", label: "Tools", icon: "tool"},
  {id: "connect", label: "Connect your tools", icon: "link"},
  {id: "data", label: "Data", icon: "disk"},
  {id: "about", label: "About and updates", icon: "info"},
];

const Row = ({label, sub, children, htmlFor, cls = ""}) => html`<div class=${`set-row ${cls}`}>
  <span class="set-row__text">${htmlFor ? html`<label for=${htmlFor}>${label}</label>` : html`<span class="set-row__label">${label}</span>`}
    ${sub ? html`<span class="set-row__sub">${sub}</span>` : null}</span>
  <span class="set-row__ctl">${children}</span></div>`;

// ------------------------------------------------------------------ General
function General() {
  const keys = [["Send", "Enter"], ["New line", "Shift Enter"], ["New chat", `${MOD} Shift O`], ["Search chats", `${MOD} K`],
                ["Stop the answer", "Esc"], ["Settings", `${MOD} ,`], ["Show or hide the sidebar", `${MOD} Shift S`]];
  return html`<h3>General</h3>
    <${Row} label="Theme" sub="System follows your computer's light or dark setting.">
      <${Seg} label="Theme" value=${themeChoice()} onChange=${setTheme}
        options=${[{id: "system", label: "System"}, {id: "dark", label: "Dark"}, {id: "light", label: "Light"}]}/><//>
    <${Row} label="Show thinking as it's written" sub="It folds away into “Thought for …” once the answer starts.">
      <${Switch} label="Show thinking as it's written" on=${P.show} onChange=${(v) => savePrefs({show: v})}/><//>
    <div class="set-block"><span class="set-row__label">Keyboard shortcuts</span>
      <div class="keys">${keys.map(([w, k]) => html`<span class="keys__row"><span>${w}</span><${Kbd} k=${k}/></span>`)}</div></div>`;
}

// ------------------------------------------------------------------ Model: the context size and the port
function Model() {
  const c = S.ctxInfo, stops = ctxStops(), h = S.health, eng = (S.metrics && S.metrics.engine) || {};
  const [ctx, setCtx] = useState(null), [port, setPort] = useState("");
  useEffect(() => { if (S.portInfo) setPort(String(S.portInfo.port)); }, [S.portInfo && S.portInfo.port]);
  const v = ctx != null && stops[ctx] ? stops[ctx] : c && c.context;
  const changed = c && v !== c.context && !S.reload;
  const e = c ? ctxEstimate(v) : null;
  let est = "How much text the model holds at once: the chat, files and tool output. More context takes GPU memory from the experts, so answers get a little slower.";
  if (S.reload) est = `Reloading with a ${ctxfmt(S.reload.to)} context…`;
  else if (e) est = `≈ ${fmt(e.gb, 1)} GB of GPU memory for the context` + (v === c.context ? "." :
    e.experts ? ` · about ${fmt(Math.abs(e.experts))} ${e.experts > 0 ? "fewer" : "more"} experts held in VRAM than now${e.experts > 0 ? " (answers a little slower)" : ""}.` : ".");
  const warns = [];
  if (c && changed) {
    const tested = c.tested || 262144;   // (the largest context run end to end on this engine)
    if (v > tested) warns.push(`Beyond ${ctxfmt(tested)} is untested on this engine.`);
    if (e && e.vram && e.gb > e.vram * 0.6) warns.push("That leaves little GPU memory for experts: answers get much slower, and it may not start (the current size then comes back).");
    warns.push("Reloading takes 1–3 minutes; your chats stay, and other apps are asked to try again meanwhile.");
  }
  const pi = S.portInfo, pv = /^\d+$/.test(port.trim()) ? +port.trim() : null;
  const portChanged = pi && pv != null && pv !== pi.port;
  const engine = {"glm-fast": "GLM fast path", ds4: "DeepSeek-V4.1 (maya-ds4)"}[eng.engine_kind] || eng.engine_kind;
  const reload = async () => {
    const tokens = S.ctx.tokens != null ? S.ctx.tokens : estimateTokens(S.current.messages);
    if (await confirmContext(v, tokens)) setCtx(null);
  };
  return html`<h3>Model</h3>
    <div class="model-card"><${ModelLogo} model=${h.model} size=${36}/>
      <span><b>${modelName(h.model)}</b><span>${[h.model, engine, h.version ? `Project Maya v${h.version}` : ""].filter(Boolean).join(" · ")}</span></span></div>
    ${c ? html`<div class="set-block">
      <label for="set-ctx" class="set-split"><span>Context size</span>
        <b class="num">${ctxfmt(v)} tokens${v === c.context ? " · now" : ""}</b></label>
      <input id="set-ctx" type="range" class="range" min="0" max=${stops.length - 1} step="1" disabled=${!!S.reload}
        value=${Math.max(0, stops.indexOf(v))} onInput=${(ev) => setCtx(+ev.target.value)}/>
      <span class="set-split set-split--muted"><span>${ctxfmt(stops[0])}</span><span>${ctxfmt(stops[stops.length - 1])}</span></span>
      <span class="set-row__sub">${est}</span>
      ${changed ? html`<div class="callout callout--warn"><span>${warns.join(" ")}</span>
        <button type="button" class="btn btn--primary btn--sm" onClick=${reload}>Reload with ${ctxfmt(v)}</button></div>` : null}
    </div>` : html`<p class="set-note">This server does not say its context range, so it can't be changed from here.</p>`}
    ${pi && pi.port && pi.saved ? html`
      <${Row} label="Port" htmlFor="set-port" sub="Where this page and the API listen. A new port starts Maya again there, and this page follows.">
        <input id="set-port" class="input input--num" inputmode="numeric" value=${port} onInput=${(ev) => setPort(ev.target.value)}/><//>
      ${portChanged ? html`<div class="callout callout--warn">
        <span>${pv < 1024 || pv > 65535 ? "Use a number from 1024 to 65535."
          : pi.restarts ? `Maya starts again on port ${pv} (about 1–3 minutes). Apps that use the API need the new address too.`
                        : `Saved for the next start: this server was not started by Maya's start script, so it stays on ${pi.port} until then.`}</span>
        ${pv >= 1024 && pv <= 65535 ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${() => confirmPort(pv)}>Move to ${pv}</button>` : null}
      </div>` : null}` : null}`;
}

// ------------------------------------------------------------------ Chat defaults (sampling)
function ChatDefaults() {
  const t = +P.temperature, greedy = t === 0, preset = presetOf(P);
  const titles = THINK_TITLES[prefsKey() === "sampling_deepseek" ? "deepseek" : "glm"];
  const set = (patch) => { savePrefs(patch); syncShared(); };
  const sliders = [
    {k: "temperature", label: "Temperature", hint: "0 always picks the likeliest word", min: 0, max: 1.5, step: 0.05,
     shown: greedy ? "0 · exact" : t.toFixed(2)},
    {k: "top_p", label: "Top-p", hint: "keeps the likeliest words", min: 0.05, max: 1, step: 0.05, shown: (+P.top_p).toFixed(2), off: greedy},
    {k: "top_k", label: "Top-k", hint: "0 is off", min: 0, max: 64, step: 1, shown: +P.top_k > 0 ? String(P.top_k) : "off", off: greedy},
    {k: "min_p", label: "Min-p", hint: "drops far less likely words", min: 0, max: 0.5, step: 0.01,
     shown: +P.min_p > 0 ? (+P.min_p).toFixed(2) : "off", off: greedy},
  ];
  const share = async (on) => {
    try {
      await saveShared(on);
      toast("success", on ? "Shared with other apps" : "Not shared any more",
            on ? "API clients use these settings from their next request." : "Other apps use their own settings again.");
    } catch (e) { toast("error", "Not saved for other apps", e.message, 6000); }
  };
  const names = {reasoning_effort: "thinking", temperature: "temperature", top_p: "top-p", top_k: "top-k", min_p: "min-p", seed: "seed", max_tokens: "max tokens"};
  const sharedNow = S.shared.on && S.shared.now && Object.keys(S.shared.now).length
    ? "API clients get now: " + Object.entries(S.shared.now).map(([k, v]) => `${names[k] || k} ${v}`).join(", ")
    : "API clients get them for anything they don't set themselves.";
  return html`<h3>Chat defaults</h3>
    <p class="set-intro">How ${modelName(S.health.model)} writes new answers. Each model keeps its own.</p>
    <${Row} label="Presets"><${Seg} label="Presets" value=${preset} onChange=${(id) => set(PRESETS[id])}
      options=${[{id: "precise", label: "Precise", title: "Temperature 0: the same question gives the same answer"},
                 {id: "balanced", label: "Balanced", title: "Temperature 0.6, top-p 0.95, top-k 20"},
                 {id: "creative", label: "Creative", title: "Temperature 1.0, top-p 0.95, min-p 0.05"}]}/><//>
    <${Row} label="Thinking" sub="The level a new chat starts with. A chat changes its own with the speedometer in the message box.">
      <${Seg} label="Thinking" value=${P.thinking} onChange=${(id) => {
        set({thinking: id});
        if (S.current && !S.current.messages.length) S.current.thinking = id;   // (the new chat on screen starts there too)
      }}
        options=${LEVELS.map((l) => ({id: l.id, label: l.name, title: titles[l.id]}))}/><//>
    ${sliders.map((r) => html`<div class="set-slider">
      <label for=${`set-${r.k}`}><span>${r.label}</span><span class="set-row__sub">${r.hint}</span></label>
      <input id=${`set-${r.k}`} type="range" class="range" min=${r.min} max=${r.max} step=${r.step} value=${P[r.k]} disabled=${r.off}
        onInput=${(e) => set({[r.k]: +e.target.value})}/>
      <b class="num">${r.shown}</b></div>`)}
    <div class="set-pair">
      <label class="field"><span class="field__label">Max tokens</span>
        <input class="input" inputmode="numeric" placeholder="Until done" value=${P.max}
          onInput=${(e) => set({max: e.target.value.replace(/\D/g, "")})}/></label>
      <label class="field"><span class="field__label">Seed</span>
        <input class="input" inputmode="numeric" placeholder="Random" value=${P.seed}
          onInput=${(e) => set({seed: e.target.value.replace(/\D/g, "")})}/></label>
    </div>
    <${Row} label="Use these for other apps too" sub=${sharedNow}>
      <${Switch} label="Use these for other apps too" on=${S.shared.on} onChange=${share}/><//>
    <div class="set-foot"><button type="button" class="btn btn--secondary btn--sm" onClick=${() => {
      const d = {...DEFAULTS};
      set({thinking: d.thinking, temperature: d.temperature, top_p: d.top_p, top_k: d.top_k, min_p: d.min_p, max: "", seed: ""});
      toast("success", "Back to the defaults", "The model's recommended settings.");
    }}>Reset to the defaults</button></div>`;
}

// ------------------------------------------------------------------ Instructions
function Instructions() {
  const c = S.current;
  return html`<h3>Instructions</h3>
    <label for="set-instr" class="set-row__text"><span class="set-row__label">For new chats</span>
      <span class="set-row__sub">The model reads them before your first message. A chat can change its own from its ⋯ menu.</span></label>
    <textarea id="set-instr" class="input input--area" rows="8" placeholder="e.g. Answer briefly. Write code in TypeScript."
      value=${P.sys_default ? P.system : ""} onInput=${(e) => {
        savePrefs({system: e.target.value, sys_default: true});
        if (S.current && !S.current.messages.length) { S.current.system = e.target.value; updateCtxMeter(); }   // (the new chat on screen)
      }}></textarea>
    ${c && c.messages.length ? html`<div class="set-row">
      <span class="set-row__text"><span class="set-row__label">This chat</span>
        <span class="set-row__sub">${c.system && c.system.trim() ? `“${c.system.trim().slice(0, 90)}${c.system.trim().length > 90 ? "…" : ""}”` : "No instructions of its own."}</span></span>
      <span class="set-row__ctl"><button type="button" class="btn btn--secondary btn--sm" onClick=${() => chatInstructions(c)}>Edit</button></span></div>` : null}`;
}

// ------------------------------------------------------------------ Tools (MCP)
const MCP_STATE = {ready: ["ok", "Connected"], starting: ["info", "Starting"], failed: ["danger", "Failed"],
                   stopped: ["warn", "Stopped"], idle: ["muted", "Waiting"]};
function Tools() {
  const servers = S.mcp.servers || [];
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  return html`<h3>Tools</h3>
    <${Row} label="Let the model use MCP tools" sub="They run on this machine with your rights, when the model decides to call them. Only this page's chat uses them.">
      <${Switch} label="Let the model use MCP tools" on=${P.mcp !== false} onChange=${(v) => savePrefs({mcp: v})}/><//>
    ${servers.length ? html`<div class="set-block">
      <span class="set-caption">Servers in the run config · ${fmt(S.mcp.tools)} tools · ${ready.length} of ${servers.length} connected</span>
      ${servers.map((s) => {
        const [tone, label] = MCP_STATE[s.status] || ["muted", s.status];
        const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
        return html`<div class="mcp">
          <div class="mcp__head"><span class=${`badge badge--${tone}`}><i class="badge__dot"></i>${label}</span>
            <b>${s.name}</b><span class="muted">${s.transport} · ${fmt(s.tools.length)} tools${info}</span></div>
          ${s.error ? html`<div class="msg-error">${s.error}</div>` : null}
          ${s.tools.length ? html`<div class="mcp__tools">${s.tools.map((t) => html`<span class="tag" title=${t.description || ""}>${t.tool}</span>`)}</div>` : null}
        </div>`;
      })}</div>`
    : html`<p class="set-note">No MCP servers are set up. Add them to the run config (“mcpServers”, as Claude Desktop writes them)
        or start Maya with --mcp-config, and their tools appear here.</p>`}`;
}

// ------------------------------------------------------------------ Connect your tools
function Connect() {
  const base = location.origin, model = S.health.model, key = !!S.health.api_key;
  const [tab, setTab] = useState("claude");
  const [apikey, setKey] = useState(store.get("apikey", ""));
  const auth = key ? "<your API key>" : "maya";
  const examples = {
    claude: ["Claude Code", `ANTHROPIC_BASE_URL=${base} ANTHROPIC_AUTH_TOKEN=${auth} claude`],
    openai: ["OpenAI apps and SDKs", `OPENAI_BASE_URL=${base}/v1\nOPENAI_API_KEY=${key ? "<your API key>" : "any-text"}\nmodel: ${model}`],
    curl: ["A first request", `curl ${base}/v1/chat/completions -H "Content-Type: application/json" ${key ? '-H "Authorization: Bearer <your API key>" ' : ""}\\\n  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`],
  };
  const addrs = [["OpenAI base URL", `${base}/v1`], ["Anthropic base URL", base], ["Model name", model]];
  return html`<h3>Connect your tools</h3>
    <p class="set-intro">Any OpenAI- or Anthropic-compatible app or coding agent works with these.</p>
    ${addrs.map(([k, v]) => html`<div class="addr"><span>${k}</span><code>${v}</code>
      <button type="button" class="icon-btn icon-btn--sm" aria-label=${`Copy the ${k}`} title="Copy" onClick=${() => copyText(v)}><${Icon} name="copy" sm/></button></div>`)}
    <${Row} label="API key" htmlFor="set-key" sub=${key ? "This server needs one. It is kept in this browser only." : "Only if the server was started with one. Kept in this browser only."}>
      <input id="set-key" class="input input--key" type="password" autocomplete="off" placeholder="Not needed" value=${apikey}
        onInput=${(e) => { setKey(e.target.value); store.set("apikey", e.target.value.trim()); }}
        onChange=${() => toast("success", "API key saved", "Kept in this browser only.")}/><//>
    <div class="set-block">
      <div class="tabs" role="tablist" aria-label="Examples">${Object.entries(examples).map(([id, [label]]) => html`
        <button type="button" role="tab" aria-selected=${String(tab === id)} class="tabs__tab" onClick=${() => setTab(id)}>${label}</button>`)}</div>
      <div onClick=${onCodeClick} dangerouslySetInnerHTML=${{__html: codeBlock("bash", examples[tab][1])}}></div>
    </div>`;
}

// ------------------------------------------------------------------ Data
function Data() {
  const n = S.chats.length;
  return html`<h3>Data</h3>
    <p class="set-intro">Chats, images and attached files are kept in this browser only (${fmt(n)} chat${n === 1 ? "" : "s"} now). The server keeps none of them.</p>
    <${Row} label="Clear chats and settings" sub="Every chat and folder, these settings and the API key in this browser. The model, its files and the server's own settings are not touched.">
      <button type="button" class="btn btn--danger" onClick=${clearBrowser}>Clear…</button><//>`;
}

// ------------------------------------------------------------------ About and updates
function UpdateCard() {
  const u = S.update, w = S.updateWatch;
  if (!w && (!u || !u.enabled)) return null;
  const latest = (u && u.latest) || {}, state = u && u.state;
  const when = u && u.checked ? new Date(u.checked * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "";
  const notes = latest.url ? html`<a class="btn btn--secondary btn--sm" href=${latest.url} target="_blank" rel="noopener">What's new</a>` : null;
  const again = (label) => html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => loadUpdate(true)}>${label}</button>`;
  let tone = "", icon = "info", title, text = "", how = "", acts = null;
  if (w || (state && state.state === "updating")) {
    const to = w ? w.to : state.to;
    tone = "accent"; icon = "download";
    title = `Updating to v${to}…`;
    text = w && w.down ? "Maya is starting again: it compiles what changed in the engine and loads the model (a few minutes; the terminal shows the progress). This page reloads when it is back."
                       : "Maya finishes the answer it is writing, then downloads the update.";
  } else if (state && state.state === "failed") {
    tone = "danger"; icon = "error";
    title = "The update did not happen";
    text = state.error || "";
    acts = html`${notes}${u.can_update ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${startUpdate}>Try again</button>` : null}`;
  } else if (u.newer) {
    tone = "accent"; icon = "download";
    title = `Project Maya v${latest.version} is out`;
    text = latest.summary || "";
    if (!u.can_update) how = `Update by hand: ${u.by_hand}. This page can't, because ${u.blocker}.`;
    acts = html`${notes}${u.can_update ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${startUpdate}>Update to v${latest.version}</button>` : null}`;
  } else if (u.error) {
    icon = "warning";
    title = "Couldn't check for updates";
    text = `${u.error}. Project Maya v${u.current || "?"} is running.`;
    acts = again("Try again");
  } else if (u.latest) {
    tone = "ok"; icon = "check";
    title = "Up to date";
    text = `v${u.current} is the latest release. Checked on GitHub${when ? ` at ${when}` : ""}.`;
    acts = again("Check again");
  } else {
    title = "Checking for updates…";
  }
  return html`<div class=${`update${tone ? ` update--${tone}` : ""}`} aria-live="polite"><${Icon} name=${icon}/>
    <span class="update__text"><b>${title}</b>${text ? html`<span>${text}</span>` : null}${how ? html`<span class="update__how">${how}</span>` : null}</span>
    ${acts ? html`<span class="update__acts">${acts}</span>` : null}</div>`;
}
// "Tesla V100 + Tesla V100" (the server's list) -> "2 × Tesla V100"; mixed cards stay listed
function gpuNames(name, count) {
  const parts = String(name).split(" + ");
  if (parts.length > 1) return parts.every((p) => p === parts[0]) ? `${parts.length} × ${parts[0]}` : parts.join(" + ");
  return count > 1 ? `${count} × ${name}` : name;
}
function facts() {
  const m = S.metrics || {}, eng = m.engine || {}, hw = m.hardware || {}, st = m.hardware_static || {}, storage = m.storage;
  const ver = eng.maya_version || S.health.version;
  const kvPerTok = eng.kv_gb && eng.kv_ctx ? (eng.kv_gb * 1073741824) / eng.kv_ctx : null;
  let kv;
  if (eng.engine_kind === "glm-fast") {
    const k = eng.kv === "int8" ? "8-bit (INT8)" : "16-bit";
    kv = eng.kv_resident   // KV streaming (--kv-resident): the cache's VRAM is its window, not a cost per token
      ? `${k} attention cache, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM`
      : `${k} attention cache, in VRAM${kvPerTok ? ` (about ${fmt(kvPerTok / 1024)} KB a token, ${fmt(eng.kv_gb, 1)} GB at this size)` : ""}`;
  } else {
    const k = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit", bf16: "16-bit", f32: "32-bit",
               fp4: "4-bit FP4 as DeepSeek stores it (the recent-token window FP8)"}[eng.kv] || eng.kv;
    kv = k ? `${k}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null;
  }
  const gpus = hw.gpus || [];
  const pcie = gpus.length > 1 ? gpus.map((g) => `GPU ${g.index}: Gen${g.pcie_gen_max || g.pcie_gen || "?"}${g.pcie_width ? ` x${g.pcie_width}` : ""}`).join(" · ")
             : (hw.gpu_pcie_gen_max || hw.gpu_pcie_gen) ? `Gen${hw.gpu_pcie_gen_max || hw.gpu_pcie_gen}${hw.gpu_pcie_width ? ` x${hw.gpu_pcie_width}` : ""}` : null;
  return [
    ["Version", ver ? `Project Maya v${ver}${eng.version ? ` · engine ${eng.version}` : ""}` : null],
    ["Model", eng.model || S.health.model],
    ["Engine", eng.engine_kind ? ({"glm-fast": "GLM fast path", ds4: "DeepSeek-V4.1 (maya-ds4)"}[eng.engine_kind] || eng.engine_kind) : null],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["Trained for", eng.trained_context ? `up to ${fmt(eng.trained_context)} tokens` : null],
    ["KV cache", kv],
    ["Experts", eng.experts ? `${fmt(eng.experts)} routed; room for ${fmt(eng.expert_slots || 0)} in VRAM` +
                 `${eng.ram_slots ? ` and ${fmt(eng.ram_slots)} in RAM` : ""}, the rest read from the SSD` : null],
    ["Speculation", eng.mtp == null ? null : +eng.mtp ? "MTP: the model's draft block proposes the next token, the full model checks it"
                  : "off (one GPU, or no draft block in this model file)"],
    ["Images", eng.images != null ? (eng.images ? "On" : "Off") : (S.health.images ? "On" : "Off")],
    ["GPU", st.gpu_name ? `${gpuNames(st.gpu_name, gpus.length)}${hw.gpu_mem_total ? ` · ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : "not readable (NVML)"],
    ["PCIe", pcie],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
    ["Model folder", storage ? `${storage.path} · ${gb(storage.free, 0)} GB free of ${gb(storage.total, 0)} GB` : null],
  ].filter((r) => r[1] != null && r[1] !== "");
}
function About() {
  const [log, setLog] = useState(null);
  const copy = async () => {
    try { await copyText(await fetchReport(), true); toast("success", "Report copied", "Paste it into a GitHub issue - read it first."); }
    catch (e) { toast("error", "No report", e.message, 5000); }
  };
  const show = async () => {
    if (log != null) { setLog(null); return; }
    try { setLog(await fetchReport()); } catch (e) { toast("error", "No report", e.message, 5000); }
  };
  return html`<h3>About and updates</h3>
    <${UpdateCard}/>
    <dl class="facts">${facts().map(([k, v]) => html`<dt>${k}</dt><dd>${v}</dd>`)}</dl>
    <div class="set-row set-row--top">
      <span class="set-row__text"><span class="set-row__label">Troubleshooting</span>
        <span class="set-row__sub">A report for a GitHub issue: the versions, this PC, the engine's settings and the telling lines of its log, never the API key. Read it before you share it.</span></span>
      <span class="set-row__ctl set-row__ctl--two"><button type="button" class="btn btn--secondary btn--sm" onClick=${show}>${log != null ? "Hide" : "Show engine log"}</button>
        <button type="button" class="btn btn--primary btn--sm" onClick=${copy}><${Icon} name="report" sm/>Copy report</button></span></div>
    ${log != null ? html`<pre class="report">${log}</pre>` : null}`;
}

// ------------------------------------------------------------------ the dialog
const BODIES = {general: General, model: Model, chat: ChatDefaults, instr: Instructions, tools: Tools, connect: Connect, data: Data, about: About};
export function SettingsDialog() {
  const box = useRef(null), body = useRef(null);
  const sec = BODIES[S.settings] ? S.settings : "general";
  useEffect(() => {
    loadShared(); loadMcp(); loadCtxInfo(); loadPortInfo(); loadUpdate();
    const ret = document.activeElement;
    setTimeout(() => { const b = box.current && box.current.querySelector(".set-nav [aria-current=page]"); if (b) b.focus(); }, 0);
    return () => { if (ret && ret.focus && document.contains(ret)) ret.focus(); };
  }, []);
  useEffect(() => { if (body.current) body.current.scrollTop = 0; }, [sec]);
  const close = () => { S.settings = null; emit(); };
  const keys = (e) => {
    if (e.key === "Tab") trapTab(e, box.current);
    else if (e.key === "Escape" && !e.defaultPrevented) { e.stopPropagation(); close(); }
  };
  const Body = BODIES[sec];
  const newer = S.update && S.update.newer && !S.updateWatch;
  return html`<div class="dialog-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) close(); }}>
    <div ref=${box} class="dialog settings" role="dialog" aria-modal="true" aria-labelledby="settings-title" onKeyDown=${keys}>
      <div class="dialog__head"><h2 id="settings-title">Settings</h2>
        <button type="button" class="icon-btn" aria-label="Close" onClick=${close}><${Icon} name="close"/></button></div>
      <div class="settings__main">
        <nav class="set-nav" aria-label="Settings sections">${SECTIONS.map((s) => html`
          <button type="button" class="nav-item" aria-current=${s.id === sec ? "page" : undefined}
            onClick=${() => { S.settings = s.id; emit(); }}><${Icon} name=${s.icon}/><span>${s.label}</span>
            ${s.id === "about" && newer ? html`<i class="nav-item__dot" title="A new version is available"></i>` : null}</button>`)}</nav>
        <div class="set-body" ref=${body}><${Body}/></div>
      </div>
    </div></div>`;
}
