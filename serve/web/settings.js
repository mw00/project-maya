// serve/web/settings.js - the Settings dialog, its pages in five groups: General (Appearance; Chat defaults:
// the running model - its context size, also on the Models page's Engine settings, and the port - sampling, thinking,
// shared with other apps; Global instructions), Integrations (Connect your tools: the OpenAI and Anthropic base URLs, the
// model's name, the API key and the examples on one page, as in v1.1.52; MCP tools), Usage (the API prices the sidebar's
// Saved counts with), Privacy (Data) and About (updates).  A change applies at once; the two that restart the model (the
// context size, the port) ask first.  Each page is inset groups of rows (the grouped look); on a phone the dialog opens
// on the list of pages and a page has a way back to it.
import {html, useEffect, useRef, useState} from "./preact.js";
import {MOD, SUPPORT_URL, api, ctxfmt, fmt, gb, isPhone, modelName, store} from "./lib.js";
import {DEFAULTS, LEVELS, P, PENALTIES, PRESETS, S, confirmDialog, copyText, emit, presetOf, resetPrefs, savePrefs, schemeChoice, setScheme, setTheme,
        themeChoice, thinkTitles, toast} from "./state.js";
import {usd} from "./sidebar.js";
import {GLM_KV_BYTES, confirmContext, confirmPort, ctxEstimate, ctxHint, ctxStops, fetchReport, loadCtxInfo, loadMcp, loadPortInfo, loadShared,
        budgetError, loadThinkBudget, saveThinkBudget, loadTime, loadUpdate, saveShared, startUpdate, syncShared} from "./server.js";
import {chatInstructions, clearBrowser, estimateTokens, updateCtxMeter} from "./chats.js";
import {codeBlock, onCodeClick} from "./md.js";
import {openModelSheet} from "./models.js";
import {TurboButton} from "./turbo.js";
import {CtxSlider, Icon, Kbd, ModelLogo, Seg, Switch, trapTab} from "./ui.js";
import {languageChoice, setLanguage, t, tr} from "./i18n.js";

const GROUPS = [
  {label: "General", items: [{id: "general", label: "Appearance", icon: "settings"},
                             {id: "chat", label: "Chat defaults", icon: "gauge"},
                             {id: "instr", label: "Global instructions", icon: "edit"}]},
  {label: "Integrations", items: [{id: "connect", label: "Connect your tools", icon: "link"},
                                  {id: "tools", label: "MCP tools", icon: "tool"}]},
  {label: "Usage", items: [{id: "savings", label: "Usage and savings", icon: "dollar"}]},
  {label: "Privacy", items: [{id: "data", label: "Data and privacy", icon: "disk"}]},
  {label: "About", items: [{id: "about", label: "About and updates", icon: "info"}]},
];
// other names a link or a hash may say: the Model page before the groups (its context size and port are on Chat
// defaults now), and the revamp's API pages (one page again: Connect your tools)
const OLD_IDS = {model: "chat", connection: "connect", openai: "connect", anthropic: "connect"};


// a page's title and its line under it; an inset group of rows (an optional caption above it, a note under it)
const Head = ({title, intro, children}) => html`<div class="set-head"><div class="set-head__text"><h3>${title}</h3>
  ${intro ? html`<p class="set-intro">${intro}</p>` : null}</div>${children}</div>`;
const Group = ({cap, note, children, cls = ""}) => html`<section class="set-sec">${cap ? html`<span class="set-cap">${cap}</span>` : null}
  <div class=${`set-group ${cls}`}>${children}</div>${note ? html`<p class="set-foot-note">${note}</p>` : null}</section>`;

const Row = ({label, sub, children, htmlFor, cls = ""}) => html`<div class=${`set-row ${cls}`}>
  <span class="set-row__text">${htmlFor ? html`<label for=${htmlFor}>${label}</label>` : html`<span class="set-row__label">${tr(label)}</span>`}
    ${sub ? html`<span class="set-row__sub">${sub}</span>` : null}</span>
  <span class="set-row__ctl">${children}</span></div>`;

// the engines by their INFO engine_kind
const ENGINES = {"glm-fast": "GLM fast path", ds4: "DeepSeek-V4.1 (maya-ds4)", mm3: "MiniMax-M3.1 (maya-mm3)"};

// ------------------------------------------------------------------ General
function General() {
  const keys = [["Send", "Enter"], ["New line", "Shift Enter"], ["New chat", `${MOD} Shift O`], ["Search chats", `${MOD} K`],
                ["Stop the answer", "Esc"], ["Settings", `${MOD} ,`], ["Show or hide the sidebar", `${MOD} Shift S`]];
  return html`<${Head} title=${tr("Appearance")}/>
    <${Group}>
    <${Row} label=${t("language", "Language")} sub=${t("language-help", "The interface language is saved in this browser. Untranslated text stays in English.")}>
      <${Seg} label=${t("language", "Language")} value=${languageChoice()} onChange=${setLanguage}
        options=${[{id: "en", label: t("english", "English")}, {id: "zh-CN", label: t("simplified-chinese", "Simplified Chinese")}]}/><//>
    <${Row} label=${tr("Theme")} sub=${tr("System follows your computer's light or dark setting.")}>
      <${Seg} label=${tr("Theme")} value=${themeChoice()} onChange=${setTheme}
        options=${[{id: "system", label: tr("System")}, {id: "dark", label: tr("Dark")}, {id: "light", label: tr("Light")}]}/><//>
    <${Row} label=${tr("Colours")} sub=${tr("Default: neutral greys with violet as the accent. Violet: violet-tinted surfaces and a soft glow.")}>
      <${Seg} label=${tr("Colours")} value=${schemeChoice()} onChange=${setScheme}
        options=${[{id: "default", label: tr("Default")}, {id: "violet", label: tr("Violet")}]}/><//>
    <${Row} label=${tr("Show thinking as it's written")} sub=${tr("It folds away into “Thought for …” once the answer starts.")}>
      <${Switch} label=${tr("Show thinking as it's written")} on=${P.show} onChange=${(v) => savePrefs({show: v})}/><//>
    ${isPhone() ? null
      : html`<${Group} cap=${tr("Keyboard shortcuts")} cls="set-group--keys">
      ${keys.map(([w, k]) => html`<span class="keys__row"><span>${tr(w)}</span><${Kbd} k=${k}/></span>`)}<//>`}<//>`;
}

// ------------------------------------------------------------------ the running model and its context size (Chat defaults)
function ModelCtx() {
  const c = S.ctxInfo, stops = ctxStops(), h = S.health, eng = (S.metrics && S.metrics.engine) || {};
  const [ctx, setCtx] = useState(null);
  const v = ctx != null && stops[ctx] ? stops[ctx] : c && c.context;
  const changed = c && v !== c.context && !S.reload;
  const e = c ? ctxEstimate(v) : null;
  // (the same line as the Models page's slider: the engine's measurement; GLM's streamed cache: 12 KB a token in RAM)
  const glmWin = eng.engine_kind === "glm-fast" && c && +c.kv_resident > 0 ? +c.kv_resident : 0;
  const est = S.reload ? `${tr("Reloading with a")} ${ctxfmt(S.reload.to)} ${tr("context…")}`
    : ctxHint(v, {live: true, now: c && c.context, win: glmWin, whole: true, ram: glmWin ? GLM_KV_BYTES : 0});
  const warns = [];
  if (c && changed) {
    const tested = c.tested || 262144;   // (the largest context run end to end on this engine)
    if (v > tested) warns.push(`${tr("Beyond")} ${ctxfmt(tested)} ${tr("is untested on this engine.")}`);
    if (e && e.vram && e.gb > e.vram * 0.6) warns.push(tr("That leaves little GPU memory for experts: answers get much slower, and it may not start (the current size then comes back)."));
    // (8 GB kept for the system: the pinned cache comes out of the RAM the experts not in VRAM use)
    if (e && e.ram && e.free != null && e.delta > e.free - 8) warns.push(tr("That takes more RAM than is free now: fewer experts stay in RAM (answers slower), and it may not start (the current size then comes back)."));
    warns.push(`${tr("Reloading takes")} ${loadTime()}${tr("; your chats stay, and other apps are asked to try again meanwhile.")}`);
  }
  const engine = ENGINES[eng.engine_kind] || eng.engine_kind;
  const reload = async () => {
    const tokens = S.ctx.tokens != null ? S.ctx.tokens : estimateTokens(S.current.messages);
    if (await confirmContext(v, tokens)) setCtx(null);
  };
  return html`<${Group} cap=${tr("The running model")}>
    <div class="model-card"><${ModelLogo} model=${h.model} size=${36}/>
      <span><b>${modelName(h.model)}</b><span>${[h.model, engine, h.version ? `Project Maya v${h.version}` : ""].filter(Boolean).join(" · ")}</span></span></div>
    ${c ? html`<div class="set-block">
      <${CtxSlider} id="set-ctx" label=${tr("Context size")} value=${v} now=${c.context} stops=${stops} disabled=${!!S.reload} hint=${est}
        onChange=${(n) => setCtx(stops.indexOf(n))}/>
      ${changed ? html`<div class="callout callout--warn"><span>${warns.join(" ")}</span>
        <button type="button" class="btn btn--primary btn--sm" onClick=${reload}>${tr("Reload with")} ${ctxfmt(v)}</button></div>` : null}
    </div>` : html`<p class="set-note">${tr("This server does not say its context range, so it can't be changed from here.")}</p>`}
    ${c ? html`<${Row} label=${tr("Engine settings")} sub=${tr("The context size is on the Models page too, with this model's other engine settings: one setting, the same value.")}>
      <button type="button" class="btn btn--ghost btn--sm" onClick=${() => { S.settings = null; openModelSheet("engine"); }}>
        ${tr("Models")}<${Icon} name="chevron-right" sm/></button><//>` : null}
    <${Port}/><//>`;
}

// ------------------------------------------------------------------ the port (with the running model): a restart
function Port() {
  const [port, setPort] = useState("");
  useEffect(() => { if (S.portInfo) setPort(String(S.portInfo.port)); }, [S.portInfo && S.portInfo.port]);
  const pi = S.portInfo, pv = /^\d+$/.test(port.trim()) ? +port.trim() : null;
  const portChanged = pi && pv != null && pv !== pi.port;
  return html`${pi && pi.port && pi.saved ? html`
      <${Row} label=${tr("Port")} htmlFor="set-port" sub=${tr("Where this page and the API listen. A new port starts Maya again there, and this page follows.")}>
        <input id="set-port" class="input input--num" inputmode="numeric" value=${port} onInput=${(ev) => setPort(ev.target.value)}/><//>
      ${portChanged ? html`<div class="callout callout--warn">
        <span>${pv < 1024 || pv > 65535 ? tr("Use a number from 1024 to 65535.")
          : pi.restarts ? `${tr("Maya starts again on port")} ${pv} (${loadTime()})${tr(". Apps that use the API need the new address too.")}`
                        : `${tr("Saved for the next start: this server was not started by Maya's start script, so it stays on")} ${pi.port} ${tr("until then.")}`}</span>
        ${pv >= 1024 && pv <= 65535 ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${() => confirmPort(pv)}>${tr("Move to")} ${pv}</button>` : null}
      </div>` : null}` : null}`;
}

// ------------------------------------------------------------------ Thinking budget (the model's run config; applies at once)
function ThinkBudget() {
  const tb = S.thinkBudget, [text, setText] = useState("");
  useEffect(() => { if (tb) setText(String(tb.budget)); }, [tb && tb.budget]);
  if (!tb || !tb.supported) return null;
  const bad = budgetError(text), changed = text.trim() !== String(tb.budget);
  return html`<div class="set-block">
    <label class="field"><span class="field__label">${tr("Thinking budget")}</span>
      <span class="set-row__sub">${tr("Tokens, 0 = no limit. Thinking stops at this many tokens and the answer starts.")}${tb.saved ? "" : tr(" Not saved: this server runs without a config.")}</span>
      <span class="set-row__ctl"><input id="set-think-budget" class="input input--num" inputmode="numeric" value=${text}
        aria-invalid=${changed && bad ? "true" : "false"} aria-label=${tr("Thinking budget in tokens")}
        onInput=${(e) => setText(e.target.value.replace(/\D/g, ""))}
        onKeyDown=${(e) => { if (e.key === "Enter" && changed && !bad) saveThinkBudget(text); }}/>
      ${changed ? html`<button type="button" class="btn btn--primary btn--sm" disabled=${!!bad} onClick=${() => saveThinkBudget(text)}>${tr("Save")}</button>` : null}</span></label>
    ${changed && bad ? html`<span class="set-row__sub" role="alert">${tr(bad)}</span>` : null}</div>`;
}

// ------------------------------------------------------------------ Chat defaults (sampling)
function ChatDefaults() {
  const temperature = +P.temperature, greedy = temperature === 0, preset = presetOf(P);
  const titles = thinkTitles();
  const set = (patch) => { savePrefs(patch); syncShared(); };
  // the running model's recommended values (its run config: the model card's) say so
  const isDefault = (k) => !greedy && Math.abs(+P[k] - (+DEFAULTS[k] || 0)) < 1e-9;
  const sliders = [
    {k: "temperature", label: tr("Temperature"), hint: tr("0 always picks the likeliest word"), min: 0, max: 1.5, step: 0.05,
     shown: greedy ? tr("0 · exact") : temperature.toFixed(2)},
    {k: "top_p", label: "Top-p", hint: "keeps the likeliest words", min: 0.05, max: 1, step: 0.05, shown: (+P.top_p).toFixed(2), off: greedy},
    {k: "top_k", label: "Top-k", hint: tr("0 is off"), min: 0, max: 64, step: 1, shown: +P.top_k > 0 ? String(P.top_k) : tr("off"), off: greedy},
    {k: "min_p", label: "Min-p", hint: "drops far less likely words", min: 0, max: 0.5, step: 0.01,
     shown: +P.min_p > 0 ? (+P.min_p).toFixed(2) : tr("off"), off: greedy},
  ];
  const share = async (on) => {
    try {
      await saveShared(on);
      toast("success", on ? tr("Shared with other apps") : tr("Not shared any more"),
            on ? tr("API clients use these settings from their next request.") : tr("Other apps use their own settings again."));
    } catch (e) { toast("error", tr("Not saved for other apps"), e.message, 6000); }
  };
  const names = {reasoning_effort: "thinking", temperature: "temperature", top_p: "top-p", top_k: "top-k", min_p: "min-p", seed: "seed", max_tokens: "max tokens",
                 repetition_penalty: "repetition penalty", frequency_penalty: "frequency penalty", presence_penalty: "presence penalty"};
  const sharedNow = S.shared.on && S.shared.now && Object.keys(S.shared.now).length
    ? tr("API clients get now: ") + Object.entries(S.shared.now).map(([k, v]) => `${names[k] ? tr(names[k]) : k} ${v}`).join(", ")
    : tr("API clients get them for anything they don't set themselves.");
  return html`<${Head} title=${tr("Chat defaults")} intro=${tr("How {model} writes new answers. Each model keeps its own.", {model: modelName(S.health.model)})}>
      <${TurboButton}/><//>
    <${ModelCtx}/>
    <${Group} cap=${tr("Sampling")}>
    <${Row} label=${tr("Presets")}><${Seg} label=${tr("Presets")} value=${preset} onChange=${(id) => set(PRESETS[id])}
      options=${[{id: "precise", label: tr("Precise"), title: tr("Temperature 0: the same question gives the same answer")},
                 {id: "balanced", label: tr("Balanced"), title: tr("Temperature 0.6, top-p 0.95, top-k 20")},
                 {id: "creative", label: tr("Creative"), title: tr("Temperature 1.0, top-p 0.95, min-p 0.05")}]}/><//>
    <${Row} label=${tr("Thinking")} sub=${tr("The level a new chat starts with. A chat changes its own with the speedometer in the message box.")}>
      <${Seg} label=${tr("Thinking")} value=${P.thinking} cls=${LEVELS.length > 4 ? "seg--many" : ""} onChange=${(id) => {
        set({thinking: id});
        if (S.current && !S.current.messages.length) S.current.thinking = id;   // (the new chat on screen starts there too)
      }}
        options=${LEVELS.map((l) => ({id: l.id, label: tr(l.name), title: tr(titles[l.id])}))}/><//>
    ${sliders.map((r) => html`<div class="set-slider">
      <label for=${`set-${r.k}`}><span>${tr(r.label)}</span><span class="set-row__sub">${tr(r.hint)}${isDefault(r.k) ? tr(" · the model's default") : ""}</span></label>
      <input id=${`set-${r.k}`} type="range" class="range" min=${r.min} max=${r.max} step=${r.step} value=${P[r.k]} disabled=${r.off}
        onInput=${(e) => set({[r.k]: +e.target.value})}/>
      <b class="num">${r.shown}</b></div>`)}<//>
    <${Group} cap=${tr("Length and repeats")}>
    <div class="set-pair">
      <label class="field"><span class="field__label">${tr("Max tokens")}</span>
        <input class="input" inputmode="numeric" placeholder=${tr("Until done")} value=${P.max}
          onInput=${(e) => set({max: e.target.value.replace(/\D/g, "")})}/></label>
      <label class="field"><span class="field__label">${tr("Seed")}</span>
        <input class="input" inputmode="numeric" placeholder=${tr("Random")} value=${P.seed}
          onInput=${(e) => set({seed: e.target.value.replace(/\D/g, "")})}/></label>
    </div>
    <${ThinkBudget}/>
    <div class="set-block set-block--pens"><span class="set-row__sub">${tr("Penalties: none by default (no model has any as a standard). Set one to discourage repeats.")}</span>
    <div class="set-pair set-pair--3">
      ${[["repetition_penalty", tr("Repetition"), tr("1.0 off, e.g. 1.1")], ["frequency_penalty", tr("Frequency"), tr("0 off, e.g. 0.3")],
         ["presence_penalty", tr("Presence"), tr("0 off, e.g. 0.3")]].map(([k, label, ph]) => html`<label class="field"><span class="field__label">${tr(label)}</span>
        <input class="input" inputmode="decimal" placeholder=${tr(ph)} value=${P[k]} aria-label=${tr("{label} penalty", {label: tr(label)})}
          onInput=${(e) => set({[k]: e.target.value.replace(/[^\d.]/g, "")})}/></label>`)}
    </div></div><//>
    <${Group} cap=${tr("Other apps")} note=${tr("The OpenAI and Anthropic APIs both take them: a request's own settings still win.")}>
    <${Row} label=${tr("Use these for other apps too")} sub=${sharedNow}>
      <${Switch} label=${tr("Use these for other apps too")} on=${S.shared.on} onChange=${share}/><//><//>
    <div class="set-foot"><button type="button" class="btn btn--secondary btn--sm" onClick=${() => {
      resetPrefs(["thinking", "temperature", "top_p", "top_k", "min_p", "max", "seed", ...PENALTIES]);   // (the model's own again)
      syncShared();
      toast("success", tr("Back to the defaults"), tr("The model's recommended settings."));
    }}>${tr("Reset to the defaults")}</button></div>`;
}

// ------------------------------------------------------------------ Instructions
function Instructions() {
  const c = S.current;
  return html`<${Head} title=${tr("Global instructions")} intro=${tr("Every new chat starts with these: the model reads them before your first message. A chat can change its own from its ⋯ menu.")}/>
    <section class="set-sec"><label for="set-instr" class="set-cap">${tr("For new chats")}</label>
    <textarea id="set-instr" class="input input--area" rows="8" placeholder=${tr("e.g. Answer briefly. Write code in TypeScript.")}
      value=${P.sys_default ? P.system : ""} onInput=${(e) => {
        savePrefs({system: e.target.value, sys_default: true});
        if (S.current && !S.current.messages.length) { S.current.system = e.target.value; updateCtxMeter(); }   // (the new chat on screen)
      }}></textarea></section>
    ${c && c.messages.length ? html`<${Group}><div class="set-row">
      <span class="set-row__text"><span class="set-row__label">${tr("This chat")}</span>
        <span class="set-row__sub">${c.system && c.system.trim() ? `“${c.system.trim().slice(0, 90)}${c.system.trim().length > 90 ? "…" : ""}”` : tr("No instructions of its own.")}</span></span>
      <span class="set-row__ctl"><button type="button" class="btn btn--secondary btn--sm" onClick=${() => chatInstructions(c)}>${tr("Edit")}</button></span></div><//>` : null}`;
}

// ------------------------------------------------------------------ Tools (MCP)
const MCP_STATE = {ready: ["ok", "Connected"], starting: ["info", "Starting"], failed: ["danger", "Failed"],
                   stopped: ["warn", "Stopped"], idle: ["muted", "Waiting"]};
function Tools() {
  const servers = S.mcp.servers || [];
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  return html`<${Head} title=${tr("MCP tools")} intro=${tr("Local tool servers the model may call in this page's chat.")}/>
    <${Group}>
    <${Row} label=${tr("Let the model use MCP tools")} sub=${tr("They run on this machine with your rights, when the model decides to call them. Only this page's chat uses them.")}>
      <${Switch} label=${tr("Let the model use MCP tools")} on=${P.mcp !== false} onChange=${(v) => savePrefs({mcp: v})}/><//><//>
    ${servers.length ? html`<${Group} cap=${tr("Servers in the run config · {tools} tools · {ready} of {total} connected", {tools:fmt(S.mcp.tools),ready:ready.length,total:servers.length})}>
      ${servers.map((s) => {
        const [tone, label] = MCP_STATE[s.status] || ["muted", s.status];
        const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
        return html`<div class="mcp">
          <div class="mcp__head"><span class=${`badge badge--${tone}`}><i class="badge__dot"></i>${MCP_STATE[s.status] ? tr(label) : label}</span>
            <b>${s.name}</b><span class="muted">${tr("{transport} · {count} tools{info}", {transport:s.transport,count:fmt(s.tools.length),info})}</span></div>
          ${s.error ? html`<div class="msg-error">${s.error}</div>` : null}
          ${s.tools.length ? html`<div class="mcp__tools">${s.tools.map((t) => html`<span class="tag" title=${t.description || ""}>${t.tool}</span>`)}</div>` : null}
        </div>`;
      })}<//>`
    : html`<p class="set-note">${tr("No MCP servers are set up. Add them to the run config (“mcpServers”, as Claude Desktop writes them) or start Maya with --mcp-config, and their tools appear here.")}</p>`}`;
}

// ------------------------------------------------------------------ Connect your tools (as in v1.1.52: both APIs' base URLs,
// the model's name and the API key on one page, then the examples)
function Connect() {
  const base = location.origin, model = S.health.model, key = !!S.health.api_key;
  const [tab, setTab] = useState("claude");
  const [apikey, setKey] = useState(store.get("apikey", ""));
  const auth = key ? "<your API key>" : "maya";
  const examples = {
    claude: ["Claude Code", `ANTHROPIC_BASE_URL=${base} ANTHROPIC_AUTH_TOKEN=${auth} claude`],
    openai: [tr("OpenAI apps and SDKs"), `OPENAI_BASE_URL=${base}/v1\nOPENAI_API_KEY=${key ? "<your API key>" : "any-text"}\nmodel: ${model}`],
    curl: [tr("A first request"), `curl ${base}/v1/chat/completions -H "Content-Type: application/json" ${key ? '-H "Authorization: Bearer <your API key>" ' : ""}\\\n  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`],
  };
  const addrs = [["OpenAI base URL", `${base}/v1`], ["Anthropic base URL", base], ["Model name", model]];
  return html`<${Head} title=${tr("Connect your tools")} intro=${tr("Any OpenAI- or Anthropic-compatible app or coding agent works with these.")}/>
    <${Group} cls="set-group--connect">
    ${addrs.map(([k, v]) => html`<div class="addr"><span>${tr(k)}</span><code>${v}</code>
      <button type="button" class="icon-btn icon-btn--sm" aria-label=${tr("Copy the {label}", {label:tr(k)})} title=${tr("Copy")} onClick=${() => copyText(v)}><${Icon} name="copy" sm/></button></div>`)}
    <${Row} label=${tr("API key")} htmlFor="set-key" sub=${key ? tr("This server needs one. It is kept in this browser only.") : tr("Only if the server was started with one. Kept in this browser only.")}>
      <input id="set-key" class="input input--key" type="password" autocomplete="off" placeholder=${tr("Not needed")} value=${apikey}
        onInput=${(e) => { setKey(e.target.value); store.set("apikey", e.target.value.trim()); }}
        onChange=${() => toast("success", tr("API key saved"), tr("Kept in this browser only."))}/><//><//>
    <${Group}><div class="set-block">
      <div class="tabs" role="tablist" aria-label=${tr("Examples")}>${Object.entries(examples).map(([id, [label]]) => html`
        <button type="button" role="tab" aria-selected=${String(tab === id)} class="tabs__tab" onClick=${() => setTab(id)}>${label}</button>`)}</div>
      <div onClick=${onCodeClick} dangerouslySetInnerHTML=${{__html: codeBlock("bash", examples[tab][1])}}></div>
    </div><//>`;
}


// ------------------------------------------------------------------ Data
function Data() {
  const n = S.chats.length;
  return html`<${Head} title=${tr("Data and privacy")} intro=${tr(n === 1 ? "Chats, images and attached files are kept in this browser only ({count} chat now). The server keeps none of them. Model files are managed on Models." : "Chats, images and attached files are kept in this browser only ({count} chats now). The server keeps none of them. Model files are managed on Models.", {count:fmt(n)})}/>
    <${Group}>
    <${Row} label=${tr("Clear chats and settings")} sub=${tr("Every chat and folder, these settings and the API key in this browser. The model, its files and the server's own settings are not touched.")}>
      <button type="button" class="btn btn--danger" onClick=${clearBrowser}>${tr("Clear")}</button><//><//>`;
}

// ------------------------------------------------------------------ Savings: the API prices the sidebar's Saved counts with
// (serve/savings.py: built-in prices per model, or the ones set here; the file keeps tokens, so a change counts every total
// again)
const PRICE_FIELDS = [["input", "Input"], ["cached", "Cached input"], ["output", "Output"]];
const priceShow = (k, v) => (v == null ? "" : k !== "peak" && Number.isInteger(Math.round(v * 1e6) / 1e4) ? v.toFixed(2) : String(v));   // ($0.50)
const priceText = (x) => Object.fromEntries(["input", "cached", "output", "peak"].map((k) => [k, priceShow(k, x[k])]));
const priceNum = (v) => (/^\s*(\d+\.?\d*|\.\d+)\s*$/.test(v) ? Number(v) : NaN);
function PriceCard({p, onSaved}) {
  const [d, setD] = useState(priceText(p));
  useEffect(() => setD(priceText(p)), [JSON.stringify(p)]);
  const def = p.default, peak = !!(def && def.peak);
  const keys = PRICE_FIELDS.map(([k]) => k).concat(peak ? ["peak"] : []);
  const bad = keys.filter((k) => Number.isNaN(priceNum(d[k])) || (k === "peak" && (priceNum(d[k]) < 1 || priceNum(d[k]) > 10)));
  const now = priceText(p), dirty = keys.some((k) => d[k].trim() !== now[k]);
  const save = async (price) => {
    try {
      onSaved(await api("api/savings/prices", {method: "POST", body: {model: p.model, price}}));
      toast("success", tr("Prices saved"), tr("{model}: every total is counted again with them.", {model:modelName(p.model)}));
    } catch (e) { toast("error", tr("Not saved"), e.message, 6000); }
  };
  const id = `price-${p.model.replace(/[^a-z0-9]/gi, "-")}`;
  return html`<section class="price" aria-label=${tr("{model}'s prices", {model:modelName(p.model)})}>
    <div class="price__head"><${ModelLogo} model=${p.model} size=${32}/>
      <span><b>${modelName(p.model)}</b><span>${p.custom ? tr("Your prices") + (def ? tr(" (built in: {api}'s)", {api:def.api}) : "")
        : def ? tr("{api}'s prices, built in (Oct 2026)", {api:def.api}) : tr("No API price yet: give it one to count what it saves")}</span></span>
      ${p.running ? html`<span class="pill pill--accent">${tr("Running")}</span>` : null}</div>
    <div class="price__grid">
      ${PRICE_FIELDS.map(([k, label]) => html`<label class="field" for=${`${id}-${k}`}><span class="field__label">${tr(label)}</span>
        <span class="price__in"><span aria-hidden="true">$</span><input id=${`${id}-${k}`} class=${`input input--num${bad.includes(k) && d[k] !== "" ? " is-bad" : ""}`}
          inputmode="decimal" value=${d[k]} placeholder=${def ? priceShow(k, def[k]) : "0.00"} onInput=${(e) => setD({...d, [k]: e.target.value})}/></span></label>`)}
      ${peak ? html`<label class="field" for=${`${id}-peak`}><span class="field__label">${tr("Peak hours")}</span>
        <span class="price__in"><span aria-hidden="true">×</span><input id=${`${id}-peak`} class=${`input input--num${bad.includes("peak") && d.peak !== "" ? " is-bad" : ""}`}
          inputmode="decimal" value=${d.peak} placeholder=${String(def.peak)} onInput=${(e) => setD({...d, peak: e.target.value})}/></span></label>` : null}
    </div>
    <div class="price__foot">
      <span>${tr("Dollars per 1M tokens")}${peak ? tr(". DeepSeek's peak hours, weekdays 01–04 and 06–10 UTC, cost this many times as much") : ""}.</span>
      ${p.custom && def ? html`<button type="button" class="btn btn--ghost btn--sm" onClick=${() => save(null)}>${tr("Use the built-in prices")}</button>` : null}
      ${dirty ? html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => setD(now)}>${tr("Cancel")}</button>` : null}
      <button type="button" class="btn btn--primary btn--sm" disabled=${!dirty || bad.length > 0}
        onClick=${() => save(Object.fromEntries(keys.map((k) => [k, priceNum(d[k])])))}>${tr("Save")}</button>
    </div>
  </section>`;
}
function Savings() {
  const [r, setR] = useState(null), [err, setErr] = useState("");
  useEffect(() => { api("api/savings").then(setR).catch((e) => setErr(e.message)); }, []);
  const reset = async () => {
    if (!await confirmDialog(tr("Start over from $0?"), tr("<p>Every model's count of tokens and what they would have cost goes back to zero. The prices stay. This cannot be undone.</p>"), tr("Start over"), true)) return;
    try { setR(await api("api/savings/reset", {method: "POST", body: {}})); toast("success", tr("Started over"), tr("Maya counts from $0 again.")); }
    catch (e) { toast("error", tr("Not reset"), e.message, 6000); }
  };
  const since = r && r.since ? new Date(r.since * 1000).toLocaleDateString(languageChoice(), {month: "short", day: "numeric"}) : "";
  return html`<${Head} title=${tr("Usage and savings")} intro=${tr("The sidebar's Saved line: what each model's own API would have charged for the same tokens. Change a price when the API's changes, or give one to a model without it. Maya keeps the tokens, so every total is counted again with the new prices.")}/>
    ${err ? html`<p class="set-note">${err}</p>` : !r ? html`<p class="set-note">${tr("Reading the prices…")}</p>`
      : !(r.price_list || []).length ? html`<p class="set-note">${tr("Nothing is counted on this server: it runs without a model config.")}</p>`
      : html`<div class="prices">${r.price_list.map((p) => html`<${PriceCard} key=${p.model} p=${p} onSaved=${setR}/>`)}</div>
        <${Row} label=${tr("Start over from $0")} sub=${tr("Every model's count of tokens: {saved} saved{since}. The prices stay.", {saved:usd(r.usd),since:since ? tr(" since {date}", {date:since}) : ""})}>
          <button type="button" class="btn btn--danger" onClick=${reset}>${tr("Start over")}</button><//><//>`}`;
}

// ------------------------------------------------------------------ About and updates
// supporting Maya (GitHub Sponsors): one warm line, never a popup - on About, and as a row of its own under an update's
// button (a little more visible there, still only where the update shows)
const SupportLine = ({cls = "set-support"}) => html`<p class=${cls}><${Icon} name="heart" sm/><span>${tr("Maya is free and open, made by one person with help from a small community. If it makes your work a little easier,")}${" "}<a class="support-link" href=${SUPPORT_URL}
  target="_blank" rel="noopener" title=${tr("Sponsor Maya on GitHub")}>${tr("sponsoring it")}</a>${tr(", in any amount, helps it keep growing. Thank you.")}</span></p>`;
function UpdateCard() {
  const u = S.update, w = S.updateWatch;
  if (!w && (!u || !u.enabled)) return null;
  const latest = (u && u.latest) || {}, state = u && u.state;
  const when = u && u.checked ? new Date(u.checked * 1000).toLocaleTimeString(languageChoice(), {hour: "2-digit", minute: "2-digit"}) : "";
  const notes = latest.url ? html`<a class="btn btn--secondary btn--sm" href=${latest.url} target="_blank" rel="noopener">${tr("What's new")}</a>` : null;
  const again = (label) => html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => loadUpdate(true)}>${label}</button>`;
  let tone = "", icon = "info", title, text = "", how = "", acts = null;
  if (w || (state && state.state === "updating")) {
    const to = w ? w.to : state.to;
    tone = "accent"; icon = "download";
    title = tr("Updating to v{version}…", {version:to});
    text = w && w.down ? tr("Maya is starting again: it compiles what changed in the engine and loads the model (a few minutes; the terminal shows the progress). This page reloads when it is back.")
                       : tr("Maya finishes the answer it is writing, then downloads the update.");
  } else if (state && state.state === "failed") {
    tone = "danger"; icon = "error";
    title = tr("The update did not happen");
    text = state.error || "";
    acts = html`${notes}${u.can_update ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${startUpdate}>${tr("Try again")}</button>` : null}`;
  } else if (u.newer) {
    tone = "accent"; icon = "download";
    title = tr("Project Maya v{version} is out", {version:latest.version});
    text = latest.summary || "";
    if (!u.can_update) how = tr("Update by hand: {command}. This page can't, because {reason}.", {command:u.by_hand,reason:u.blocker});
    acts = html`${notes}${u.can_update ? html`<button type="button" class="btn btn--primary btn--sm" onClick=${startUpdate}>${tr("Update to v{version}", {version:latest.version})}</button>` : null}`;
  } else if (u.error) {
    icon = "warning";
    title = tr("Couldn't check for updates");
    text = tr("{error}. Project Maya v{version} is running.", {error:u.error,version:u.current || "?"});
    acts = again(tr("Try again"));
  } else if (u.latest) {
    tone = "ok"; icon = "check";
    title = tr("Up to date");
    text = tr("v{version} is the latest release. Checked on GitHub{when}.", {version:u.current,when:when ? tr(" at {time}", {time:when}) : ""});
    acts = again(tr("Check again"));
  } else {
    title = tr("Checking for updates…");
  }
  return html`<div class=${`update${tone ? ` update--${tone}` : ""}`} aria-live="polite"><${Icon} name=${icon}/>
    <span class="update__text"><b>${title}</b>${text ? html`<span>${text}</span>` : null}${how ? html`<span class="update__how">${how}</span>` : null}</span>
    ${acts ? html`<span class="update__acts">${acts}</span>` : null}
    ${u && u.newer && !w && !(state && state.state) ? html`<${SupportLine} cls="update__support"/>` : null}</div>`;
}
// "Tesla V100 + Tesla V100" (the server's list) -> "2 × Tesla V100"; mixed cards stay listed
function gpuNames(name, count) {
  const parts = String(name).split(" + ");
  if (parts.length > 1) {   // (each model once, with its count: "NVIDIA GeForce RTX 3090 + 8 × NVIDIA GeForce RTX 5060 Ti")
    const n = new Map();
    for (const p of parts) n.set(p, (n.get(p) || 0) + 1);
    return [...n].map(([p, k]) => (k > 1 ? `${k} × ${p}` : p)).join(" + ");
  }
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
      ? tr("{format} attention cache, streamed: {positions} positions per layer in VRAM, the rest in RAM", {format:tr(k),positions:fmt(eng.kv_resident)})
      : tr("{format} attention cache, in VRAM{cost}", {format:tr(k),cost:kvPerTok ? tr(" (about {kb} KB a token, {gb} GB at this size)", {kb:fmt(kvPerTok/1024),gb:fmt(eng.kv_gb,1)}) : ""});
  } else {
    const k = (eng.engine_kind === "mm3"   // (MiniMax's: maya-mm3's KV formats)
      ? {bf16: "16-bit (BF16)", int8: "8-bit (INT8)", fp8: "8-bit (FP8)", fp4: "4-bit (FP4)", nvfp4: "4-bit (NVFP4)",
         mxfp4: "4-bit (MXFP4)", q4: "4-bit"}
      : {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit", bf16: "16-bit", f32: "32-bit",
         fp4: "4-bit FP4 as DeepSeek stores it (the recent-token window FP8)"})[eng.kv] || eng.kv;
    // (DeepSeek's engine says a window as long as its context: all of it in VRAM)
    const win = +eng.kv_resident > 0 && +eng.kv_resident < (eng.max_context || Infinity) ? +eng.kv_resident : 0;
    kv = k ? (win ? tr("{format}, streamed: {positions} positions per layer in VRAM, the rest in RAM", {format:tr(k),positions:fmt(win)}) : tr("{format}, all in VRAM", {format:tr(k)})) : null;
  }
  const gpus = hw.gpus || [];
  const link = (g) => `Gen${g.pcie_gen_max || g.pcie_gen || "?"}${g.pcie_width ? ` x${g.pcie_width}` : ""}`;
  const pcie = gpus.length > 1 ? (gpus.every((g) => link(g) === link(gpus[0])) ? tr("{link} on all {count} GPUs", {link:link(gpus[0]),count:gpus.length})
                                  : gpus.map((g) => `GPU ${g.index}: ${link(g)}`).join(" · "))
             : (hw.gpu_pcie_gen_max || hw.gpu_pcie_gen) ? `Gen${hw.gpu_pcie_gen_max || hw.gpu_pcie_gen}${hw.gpu_pcie_width ? ` x${hw.gpu_pcie_width}` : ""}` : null;
  return [
    [tr("Version"), ver ? tr("Project Maya v{version}{engine}", {version:ver,engine:eng.version ? tr(" · engine {version}", {version:eng.version}) : ""}) : null],
    [tr("Model"), eng.model || S.health.model],
    [tr("Engine"), eng.engine_kind ? (ENGINES[eng.engine_kind] ? tr(ENGINES[eng.engine_kind]) : eng.engine_kind) : null],
    [tr("Context"), eng.max_context ? tr("{count} tokens", {count:fmt(eng.max_context)}) : null],
    [tr("Trained for"), eng.trained_context ? tr("up to {count} tokens", {count:fmt(eng.trained_context)}) : null],
    [tr("KV cache"), kv],
    [tr("Experts"), eng.experts ? tr("{count} routed; room for {vram} in VRAM{ram}{rest}", {count:fmt(eng.experts),vram:fmt(eng.expert_slots || 0),
      ram:eng.ram_slots ? tr(" and {count} in RAM", {count:fmt(eng.ram_slots)}) : "",rest:+eng.expert_slots+(+eng.ram_slots||0)>=+eng.experts ? "" : tr(", the rest read from the SSD")}) : null],
    [tr("Speculation"), eng.dspark != null
                  ? (+eng.dspark ? tr("DSpark: its drafter proposes a few tokens at a time, the full model checks them") +
                                  (m.tiers && m.tiers.now && m.tiers.now.drafts != null ? tr(" ({percent}% of them kept in the last answer)", {percent:fmt(100*m.tiers.now.drafts)}) : "")
                                : tr("off (the config's --dspark, or no DSpark drafter in this pack)"))
                  : eng.mtp == null ? null : +eng.mtp ? tr("MTP: the model's draft block proposes the next token, the full model checks it")
                  : tr("off (one GPU, or no draft block in this model file)")],
    [tr("Images"), eng.images != null ? (eng.images ? tr("On") : tr("Off")) : (S.health.images ? tr("On") : tr("Off"))],
    ["GPU", st.gpu_name ? `${gpuNames(st.gpu_name, gpus.length)}${hw.gpu_mem_total ? ` · ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : tr("not readable (NVML)")],
    ["PCIe", pcie],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? tr(", {count} threads", {count:st.threads}) : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
    [tr("Model folder"), storage ? tr("{path} · {free} GB free of {total} GB", {path:storage.path,free:gb(storage.free,0),total:gb(storage.total,0)}) : null],
  ].filter((r) => r[1] != null && r[1] !== "");
}
function About() {
  const [log, setLog] = useState(null);
  const copy = async () => {
    try { await copyText(await fetchReport(), true); toast("success", tr("Report copied"), tr("Paste it into a GitHub issue - read it first.")); }
    catch (e) { toast("error", tr("No report"), e.message, 5000); }
  };
  const show = async () => {
    if (log != null) { setLog(null); return; }
    try { setLog(await fetchReport()); } catch (e) { toast("error", tr("No report"), e.message, 5000); }
  };
  const newer = S.update && S.update.newer && !S.updateWatch;   // (then the ask is beside the update instead)
  return html`<${Head} title=${tr("About and updates")}/>
    <${UpdateCard}/>
    ${newer ? null : html`<${SupportLine}/>`}
    <${Group} cap=${tr("This server")}><dl class="facts">${facts().map(([k, v]) => html`<div class="facts__row"><dt>${k}</dt><dd>${v}</dd></div>`)}</dl><//>
    <${Group}>
    <div class="set-row set-row--top">
      <span class="set-row__text"><span class="set-row__label">${tr("Troubleshooting")}</span>
        <span class="set-row__sub">${tr("A report for a GitHub issue: the versions, this PC, the engine's settings and the telling lines of its log, never the API key. Read it before you share it.")}</span></span>
      <span class="set-row__ctl set-row__ctl--two"><button type="button" class="btn btn--secondary btn--sm" onClick=${show}>${log != null ? tr("Hide") : tr("Show engine log")}</button>
        <button type="button" class="btn btn--primary btn--sm" onClick=${copy}><${Icon} name="report" sm/>${tr("Copy report")}</button></span></div>
    ${log != null ? html`<pre class="report">${log}</pre>` : null}<//>`;
}

// ------------------------------------------------------------------ the dialog
const BODIES = {general: General, chat: ChatDefaults, instr: Instructions, connect: Connect,
                tools: Tools, savings: Savings, data: Data, about: About};
export function SettingsDialog() {
  const box = useRef(null), body = useRef(null);
  const want = OLD_IDS[S.settings] || S.settings;
  const sec = BODIES[want] ? want : "general";
  // a phone: the list of pages first (as opened from the sidebar or Ctrl+,), a page when a link names one
  const [list, setList] = useState(() => isPhone() && (!S.settings || S.settings === "general"));
  useEffect(() => {
    loadShared(); loadMcp(); loadCtxInfo(); loadPortInfo(); loadThinkBudget(); loadUpdate();
    const ret = document.activeElement;
    setTimeout(() => { const b = box.current && box.current.querySelector(".set-nav [aria-current=page]"); if (b) b.focus(); }, 0);
    return () => { if (ret && ret.focus && document.contains(ret)) ret.focus(); };
  }, []);
  useEffect(() => { if (body.current) body.current.scrollTop = 0; }, [sec]);
  const close = () => { S.settings = null; emit(); };
  const pick = (id) => { S.settings = id; setList(false); emit(); };
  const keys = (e) => {
    if (e.key === "Tab") trapTab(e, box.current);
    else if (e.key === "Escape" && !e.defaultPrevented) { e.stopPropagation(); close(); }
  };
  const Body = BODIES[sec];
  const newer = S.update && S.update.newer && !S.updateWatch;
  return html`<div class="dialog-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) close(); }}>
    <div ref=${box} class=${`dialog settings${list ? " is-list" : ""}`} role="dialog" aria-modal="true" aria-labelledby="settings-title" onKeyDown=${keys}>
      <div class="dialog__head">${list ? null : html`<button type="button" class="btn btn--ghost btn--sm set-back" onClick=${() => setList(true)}>
          <${Icon} name="chevron-right" sm/>${tr("Settings")}</button>`}<h2 id="settings-title">${tr("Settings")}</h2>
        <button type="button" class="icon-btn" aria-label=${tr("Close")} onClick=${close}><${Icon} name="close"/></button></div>
      <div class="settings__main">
        <nav class="set-nav" aria-label=${tr("Settings sections")}>${GROUPS.map((g) => html`
          <span class="set-nav__group">${tr(g.label)}</span>
          <div class="set-nav__items">${g.items.map((s) => html`
          <button type="button" class="nav-item" aria-current=${s.id === sec && !list ? "page" : undefined}
            onClick=${() => pick(s.id)}><${Icon} name=${s.icon}/><span>${tr(s.label)}</span>
            ${s.id === "about" && newer ? html`<i class="nav-item__dot" title=${tr("A new version is available")}></i>` : null}
            <${Icon} name="chevron-right" sm cls="set-nav__chev"/></button>`)}</div>`)}</nav>
        <div class="set-body" ref=${body}><div class="set-page"><${Body}/></div></div>
      </div>
    </div></div>`;
}
