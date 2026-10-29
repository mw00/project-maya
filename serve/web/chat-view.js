// serve/web/chat-view.js - the chat: its header (the model menu, the context meter, the chat's ⋯ menu), the messages
// (an answer is painted in place while it streams), the message box with its thinking speedometer, and a new chat's
// greeting: the model that answers it and the chats to pick up again.
import {Component, html, useEffect, useLayoutEffect, useRef, useState} from "./preact.js";
import {SPRITE, coarse, ctxfmt, esc, fmt, isPhone, modelName, relTime, timeStr, tokfmt} from "./lib.js";
import {LEVELS, P, PRESET_NAMES, S, copyText, emit, presetOf, savePrefs, thinkTitles} from "./state.js";
import {mcpOn} from "./server.js";
import {addFiles, cancelEdit, chatTitle, chatView, ctxMeter, editMessage, focusInput, newChat, openChat, painters, regenerate,
        removeAttachment, scrollDown, send, setThinking, stopAnswer, thinkingOf, updateCtxMeter} from "./chats.js";
import {markdown, onCodeClick, patchHtml, streamingMarkdown} from "./md.js";
import {ChatMenuItems, openSettings, setSide, showView, status, usd} from "./sidebar.js";
import {MS, loadModels} from "./models.js";
import {Gauge, Icon, Menu, MenuItem, MenuSep, ModelLogo, useDismiss} from "./ui.js";

// ------------------------------------------------------------------ the header
function MeterRing({c}) {
  const pct = Math.max(c.pct, 1.5);
  return html`<svg class="meter__ring" viewBox="0 0 20 20" aria-hidden="true"><circle class="meter__track" cx="10" cy="10" r="8"/>
    <circle class="meter__fill" cx="10" cy="10" r="8" pathLength="100" stroke-dasharray=${`${pct.toFixed(1)} 100`}/></svg>`;
}
function Meter() {
  const c = ctxMeter();
  if (!c) return null;
  return html`<span class=${`meter${c.level ? ` meter--${c.level}` : ""}`} title=${c.title} role="img" aria-label=${c.title}>
    <${MeterRing} c=${c}/>${c.approx ? "≈" : ""}${tokfmt(c.tokens)} / ${ctxfmt(c.max)}</span>`;
}

function ModelMenu({open, close, anchor}) {
  const h = S.health, preset = presetOf(P), c = ctxMeter();
  useEffect(() => { if (open && !MS.data) loadModels(); }, [open]);   // (the Models row says how many are set up)
  const go = (s) => () => { close(); openSettings(s); };
  return html`<${Menu} open=${open} onClose=${close} anchor=${anchor} label="Model" align="left" cls="model-menu" width=${340}>
    <div class="model-menu__card"><${ModelLogo} model=${h.model} size=${36}/>
      <span><b>${modelName(h.model, !!h.model_label)}</b><span>${h.model_label || h.model} · on this machine${h.version ? ` · v${h.version}` : ""}</span></span></div>
    <${MenuSep}/>
    ${c && isPhone() ? html`<div class="model-menu__meter"><${MeterRing} c=${c}/><span>${c.why}: ${c.approx ? "≈" : ""}${fmt(c.tokens)} of ${fmt(c.max)} tokens</span></div>` : null}
    <button type="button" role="menuitem" class="menu__item" onClick=${go("model")}><span class="menu__label">Context size</span>
      <span class="menu__sub">${h.max_context ? ctxfmt(h.max_context) : "–"}</span><${Icon} name="chevron" sm cls="chev is-closed"/></button>
    <button type="button" role="menuitem" class="menu__item" onClick=${go("chat")}><span class="menu__label">Sampling</span>
      <span class="menu__sub">${preset ? PRESET_NAMES[preset] : "Custom"}</span><${Icon} name="chevron" sm cls="chev is-closed"/></button>
    <${MenuSep}/>
    <button type="button" role="menuitem" class="menu__item" onClick=${() => { close(); showView("models"); }}>
      <span class="menu__label">Models</span><span class="menu__sub">${MS.data ? `${MS.data.models.length} set up` : "Switch, settings"}</span>
      <${Icon} name="chevron" sm cls="chev is-closed"/></button>
  <//>`;
}

// (empty: a new chat - its model card says which model answers, so no model button or context meter up here)
function Header({empty}) {
  const [model, setModel] = useState(false), [opts, setOpts] = useState(false);
  const mBtn = useRef(null), oBtn = useRef(null);
  const name = modelName(S.health.model, true);   // (the short form here: "Qwen-3.8-Flash-Next"; the menu's line below it has the full one)
  if (isPhone()) {
    return html`<header class="chat-head chat-head--float">
      <button type="button" class="glass-btn" aria-label="Chats" onClick=${() => { S.drawer = true; emit(); }}><${Icon} name="menu"/></button>
      ${empty ? html`<span></span>` : html`<button type="button" ref=${mBtn} class="model-btn model-btn--phone" aria-haspopup="menu" aria-expanded=${String(model)}
        onClick=${() => setModel(!model)}>${name}<${Icon} name="chevron" sm/></button>`}
      <button type="button" class="glass-btn" aria-label="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button>
      ${empty ? null : html`<${ModelMenu} open=${model} close=${() => setModel(false)} anchor=${mBtn}/>`}
    </header>`;
  }
  return html`<header class="chat-head">
    <div class="chat-head__left">
      ${S.sideOpen ? null : html`
        <button type="button" class="icon-btn" aria-label="Open the sidebar" title="Open the sidebar" onClick=${() => setSide(true)}><${Icon} name="panel-left"/></button>
        <button type="button" class="icon-btn" aria-label="New chat" title="New chat" onClick=${newChat}><${Icon} name="new-chat"/></button>`}
      ${empty ? null : html`<span class="anchor">
        <button type="button" ref=${mBtn} class=${`model-btn${model ? " is-open" : ""}`} aria-haspopup="menu" aria-expanded=${String(model)}
          title=${S.health.model} onClick=${() => setModel(!model)}>${name}<${Icon} name="chevron" sm/></button>
        <${ModelMenu} open=${model} close=${() => setModel(false)} anchor=${mBtn}/>
      </span>`}
    </div>
    <div class="chat-head__right">
      ${empty ? null : html`<${Meter}/>`}
      <span class="anchor">
        <button type="button" ref=${oBtn} class="icon-btn" aria-label="Chat options" title="Chat options" aria-haspopup="menu"
          aria-expanded=${String(opts)} onClick=${() => setOpts(!opts)}><${Icon} name="ellipsis"/></button>
        <${Menu} open=${opts} onClose=${() => setOpts(false)} anchor=${oBtn} label="Chat options" width=${250}>
          <${ChatMenuItems} c=${S.current} close=${() => { setOpts(false); if (oBtn.current) oBtn.current.focus(); }}/>
        <//>
      </span>
    </div>
  </header>`;
}

// ------------------------------------------------------------------ messages
function UserMsg({m, i}) {
  const [copied, setCopied] = useState(false);
  const copy = () => { copyText(m.text, true); setCopied(true); setTimeout(() => setCopied(false), 1500); };
  return html`<div class="you" data-id=${m.id}>
    ${(m.files || []).length || (m.images || []).length ? html`<div class="you__files">
      ${(m.files || []).map((f) => html`<span class="chip"><${Icon} name="attach" sm/>${f.name}</span>`)}
      ${(m.images || []).map((im) => im.url ? html`<img src=${im.url} alt=${im.name || "picture"}/>`
                                            : html`<span class="chip"><${Icon} name="image" sm/>${im.name || "picture"}</span>`)}</div>` : null}
    ${m.text ? html`<div class="you__bubble">${m.text}</div>` : null}
    <div class="you__actions"><span class="you__time">${m.time ? timeStr(m.time) : ""}</span>
      <button type="button" class="icon-btn icon-btn--sm" aria-label="Edit and send again" title="Edit" onClick=${() => editMessage(i)}><${Icon} name="edit" sm/></button>
      <button type="button" class="icon-btn icon-btn--sm" aria-label="Copy" title="Copy" onClick=${copy}><${Icon} name=${copied ? "check" : "copy"} sm/></button>
    </div></div>`;
}

// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and the
// result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["badge--info", "Writing"], running: ["badge--accent", "Running"], done: ["", "Done"],
                    error: ["badge--danger", "Error"], skipped: ["badge--warn", "Not run"]};
const ico = (name) => `<svg class="st-icon st-icon--sm" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool__label">Arguments</div><pre class="tool__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="tool" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${ico("tool")}<span class="tool__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="tool__server">${esc(t.server)}</span>` : "") +
    `<span class="tool__preview">${esc(preview.slice(0, 200))}</span>` +
    `<span class="badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="tool__ms">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${ico("chevron")}</summary><div class="tool__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m, streaming) {
  if (!m.tools || !m.tools.length) return streaming ? streamingMarkdown(m) : markdown(m.text || "");
  let out = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) out += markdown(m.text.slice(pos, at));
    pos = at;
    out += toolHtml(t, k);
  });
  return out + markdown(m.text.slice(pos));
}

// An answer: painted in place (it repaints up to every frame while it streams), not re-rendered by the page
class Answer extends Component {
  constructor() {
    super();
    this.sig = "";
    this.paint = () => this.update(this.props, true);
    this.onToggle = () => {
      const d = this.det;
      if (d._auto) { d._auto = false; return; }          // our own open/close, not the reader's
      d.dataset.touched = "1";
      const body = d.querySelector(".think__body");
      if (d.open && body.dataset.pending) { body.textContent = this.props.m.reasoning; delete body.dataset.pending; }
    };
    this.copy = () => {
      copyText(this.props.m.text, true);
      const use = this.base.querySelector("[data-a=copy] use");
      use.setAttribute("href", `${SPRITE}#i-check`);
      setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
    };
  }
  componentDidMount() { painters.set(this.props.m, this.paint); this.update(this.props, true); }
  componentWillUnmount() { if (painters.get(this.props.m) === this.paint) painters.delete(this.props.m); }
  shouldComponentUpdate(np) {
    if (np.m !== this.props.m) return true;
    this.update(np, false);
    return false;
  }
  componentDidUpdate(pp) {
    if (pp.m === this.props.m) return;
    if (painters.get(pp.m) === this.paint) painters.delete(pp.m);
    painters.set(this.props.m, this.paint);
    this.update(this.props, true);
  }
  update({m, last}, force) {
    if (!this.base) return;
    const streaming = !!S.busy && S.busy.msg === m;
    const sig = [m.text.length, (m.reasoning || "").length, (m.tools || []).map((t) => `${t.state}${t.open ? 1 : 0}${t.result != null ? 1 : 0}`).join(),
                 m.error, m.meta, m.stopped, streaming, last, !!S.busy, P.show].join("|");
    if (!force && sig === this.sig) return;
    this.sig = sig;
    const det = this.det;
    if (m.reasoning) {
      det.hidden = false;
      const thinkingNow = streaming && !m.text;
      det.querySelector(".think__title").textContent = thinkingNow ? "Thinking…" :
        m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
      det.classList.toggle("is-live", thinkingNow);
      const body = det.querySelector(".think__body");
      if (det.open || thinkingNow) body.textContent = m.reasoning;
      else body.dataset.pending = "1";
      // open while it streams (if wanted), closed once the answer starts - unless the reader toggled it themselves
      if (thinkingNow && P.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
      if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
    } else {
      det.hidden = true;
    }
    const body = this.body;
    if (m.error) {
      body.innerHTML = `<div class="msg-error"></div>`;
      body.firstChild.textContent = m.error;
    } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
      body.innerHTML = m.reasoning ? "" : `<p class="answer__wait"><span class="dot-pulse"></span></p>`;
    } else {
      patchHtml(body, answerHtml(m, streaming));
    }
    body.classList.toggle("is-streaming", streaming && !!m.text);
    const acts = this.acts;
    acts.querySelector(".answer__meta").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
    acts.querySelector("[data-a=copy]").hidden = streaming || !m.text;
    acts.querySelector("[data-a=regen]").hidden = streaming || !!S.busy || !last;   // the last answer only, when nothing runs
    acts.hidden = streaming;
  }
  render() {
    return html`<article class="answer" data-id=${this.props.m.id}>
      <details class="think" hidden ref=${(e) => { this.det = e; }} onToggle=${this.onToggle}>
        <summary><span class="think__title"></span><${Icon} name="chevron" sm cls="chev"/></summary>
        <div class="think__body"></div></details>
      <div class="answer__body prose" ref=${(e) => { this.body = e; }}></div>
      <div class="answer__actions" ref=${(e) => { this.acts = e; }}>
        <button type="button" class="icon-btn icon-btn--sm" data-a="copy" aria-label="Copy the answer" title="Copy" onClick=${this.copy}><${Icon} name="copy" sm/></button>
        <button type="button" class="icon-btn icon-btn--sm" data-a="regen" aria-label="Write this answer again" title="Regenerate" onClick=${regenerate}><${Icon} name="refresh" sm/></button>
        <span class="answer__meta"></span></div>
    </article>`;
  }
}

function Thread() {
  const ms = S.current.messages;
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const click = (e) => {
    onCodeClick(e);
    const sum = e.target.closest(".tool > summary");
    if (!sum) return;
    e.preventDefault();
    const id = sum.closest(".answer").dataset.id, m = ms.find((x) => x.id === id);
    const t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    const f = painters.get(m);
    if (f) f();
  };
  return html`<div class="thread" onClick=${click}>
    ${ms.map((m, i) => m.role === "user" ? html`<${UserMsg} key=${m.id} m=${m} i=${i}/>`
                                          : html`<${Answer} key=${m.id} m=${m} last=${i === ms.length - 1} busy=${!!S.busy}/>`)}
  </div>`;
}

// ------------------------------------------------------------------ the message box
let measureCtx = null;
function textWidth(t, el) {
  measureCtx = measureCtx || document.createElement("canvas").getContext("2d");
  const cs = getComputedStyle(el);
  measureCtx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  return measureCtx.measureText(t).width;
}

// the thinking level: a thick bar hovering over the chat by the message box (above it; below it on a new chat's centred
// one), one segment a level - the model's own: GLM's Off, Low, High and Max - filled up to the one set.  A tap, a click
// or a drag along it sets another (one slider: the arrow keys, Home and End too).  On a phone it spans the message box.
function ThinkingBar({bar, below}) {
  const level = thinkingOf(), n = LEVELS.length, li = Math.max(0, LEVELS.findIndex((l) => l.id === level));
  const titles = thinkTitles();
  const track = useRef(null), dragging = useRef(false);
  const set = (i) => { const l = LEVELS[Math.min(n - 1, Math.max(0, i))]; if (l.id !== thinkingOf()) setThinking(l.id); };
  const from = (x) => { const r = track.current.getBoundingClientRect(); return Math.floor(((x - r.left) / r.width) * n); };
  const down = (e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); set(from(e.clientX)); };
  const move = (e) => { if (dragging.current) set(from(e.clientX)); };
  const up = () => { dragging.current = false; };
  const keys = (e) => {
    const to = {ArrowRight: li + 1, ArrowUp: li + 1, ArrowLeft: li - 1, ArrowDown: li - 1, Home: 0, End: n - 1}[e.key];
    if (to == null) return;
    e.preventDefault();
    set(to);
  };
  const cls = `think-bar${isPhone() ? "" : " think-bar--pop"}${below && !isPhone() ? " think-bar--below" : ""}${n > 5 ? " think-bar--many" : ""}`;
  return html`<div ref=${bar} class=${cls} style=${`--n:${n}`}>
    <div ref=${track} class="think-bar__track" role="slider" tabIndex="0" aria-label="Thinking level, for this chat" aria-valuemin="0"
      aria-valuemax=${n - 1} aria-valuenow=${li} aria-valuetext=${`${LEVELS[li].name}: ${LEVELS[li].desc}`}
      style=${`--fill:${((li + 1) * 100) / n}%`}
      onPointerDown=${down} onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up} onKeyDown=${keys}>
      <span class="think-bar__fill"></span>
      ${LEVELS.map((l, i) => html`<span class=${`think-bar__seg${i <= li ? " is-on" : ""}${i === li ? " is-sel" : ""}`}
        title=${titles[l.id]}>${l.name}</span>`)}
    </div>
  </div>`;
}
function ThinkingPopover({open, close, anchor, below}) {
  const ref = useRef(null);
  useDismiss(open, close, ref, anchor);
  useEffect(() => { if (open && ref.current) { const b = ref.current.querySelector('[role="slider"]'); if (b) b.focus({preventScroll: true}); } }, [open]);
  return open ? html`<${ThinkingBar} bar=${ref} below=${below}/>` : null;
}

function PlusMenu({open, close, anchor, pickFiles}) {
  const servers = (S.mcp.servers || []).length;
  return html`<${Menu} open=${open} onClose=${close} anchor=${anchor} label="Add" align="left" width=${270} cls="plus-menu">
    <${MenuItem} icon="attach" label="Select file" onClick=${() => { close(); pickFiles(); }}/>
    ${servers ? html`<${MenuSep}/>
      <button type="button" role="menuitemcheckbox" aria-checked=${String(P.mcp !== false)} class="menu__item"
        onClick=${() => savePrefs({mcp: P.mcp === false})}>
        <${Icon} name="tool" sm/><span class="menu__label">MCP tools<span class="menu__hint">${S.mcp.tools ? `${fmt(S.mcp.tools)} tools the model may call` : "no server connected yet"}</span></span>
        <span class="switch switch--sm" aria-hidden="true" data-on=${String(P.mcp !== false)}><span class="switch__knob"></span></span></button>` : null}
  <//>`;
}

function Composer({center}) {
  const ta = useRef(null), box = useRef(null), file = useRef(null), plusBtn = useRef(null), thinkBtn = useRef(null);
  const [tall, setTall] = useState(false), [plus, setPlus] = useState(false), [think, setThink] = useState(false);
  const compactW = useRef(0);
  const text = S.draft;
  const mine = S.busy && S.busy.chat === S.current;
  const canSend = (text.trim() || S.attachments.length) && !S.busy && !S.reload;
  useLayoutEffect(() => {
    const t = ta.current;
    if (!t) return;
    if (!tall) compactW.current = t.clientWidth - 8;
    const need = S.attachments.length > 0 || text.includes("\n") || (!!text && textWidth(text, t) > compactW.current);
    if (need !== tall) { setTall(need); return; }
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, Math.round(innerHeight * 0.4))}px`;
  }, [text, S.attachments.length, tall]);   // (not on every repaint: the chat below must not jump while it streams)
  const keydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !coarse()) { e.preventDefault(); send(); }
  };
  const paste = (e) => {
    if (!S.health.images) return;
    const files = [...((e.clipboardData || {}).files || [])].filter((f) => f.type.startsWith("image/"));
    if (files.length) { e.preventDefault(); addFiles(files); }
  };
  const level = thinkingOf(), lv = LEVELS.find((l) => l.id === level) || LEVELS[2];
  return html`<div class="composer-wrap">
    ${S.editIndex != null ? html`<div class="edit-bar"><${Icon} name="edit" sm/><span>Editing a message: sending it replaces the answers after it.</span>
      <button type="button" class="btn btn--secondary btn--sm" onClick=${() => cancelEdit()}>Cancel</button></div>` : null}
    <form ref=${box} class=${`composer${tall ? " is-tall" : ""}`} onSubmit=${(e) => { e.preventDefault(); send(); }}>
      ${S.attachments.length ? html`<div class="composer__files">${S.attachments.map((a, i) => html`
        <span class="chip chip--file">${a.kind === "image" && a.url ? html`<img src=${a.url} alt=""/>` : html`<${Icon} name=${a.kind === "file" ? "attach" : "image"} sm/>`}
          <span class="chip__name">${a.name}</span>
          <button type="button" class="chip__x" aria-label=${`Remove ${a.name}`} onClick=${() => removeAttachment(i)}><${Icon} name="close" sm/></button></span>`)}</div>` : null}
      <span class="anchor composer__plus">
        <button type="button" ref=${plusBtn} class="icon-btn icon-btn--round" aria-label="Add files, images and tools" title="Add files, images and tools"
          aria-haspopup="menu" aria-expanded=${String(plus)} onClick=${() => setPlus(!plus)}><${Icon} name="plus"/></button>
        <${PlusMenu} open=${plus} close=${() => setPlus(false)} anchor=${plusBtn} pickFiles=${() => file.current.click()}/>
      </span>
      <input type="file" ref=${file} multiple hidden onChange=${(e) => { addFiles(e.target.files); e.target.value = ""; focusInput(); }}/>
      <textarea id="composer-input" ref=${ta} class="composer__input" rows="1" placeholder="Ask anything" aria-label="Ask anything"
        value=${text} onInput=${(e) => { S.draft = e.target.value; emit(); updateCtxMeter(); }} onKeyDown=${keydown} onPaste=${paste}></textarea>
      <div class="composer__tools">
        <span class="anchor">
          <button type="button" ref=${thinkBtn} class=${`think-chip${think ? " is-open" : ""}`} aria-haspopup="dialog" aria-expanded=${String(think)}
            aria-label=${`Thinking: ${lv.name}. Change it`} title="How long the model thinks" onClick=${() => setThink(!think)}>
            <${Gauge} level=${level}/><span>${lv.name}</span></button>
        </span>
        ${mine ? html`<button type="button" class="send send--stop" aria-label="Stop the answer (Esc)" title="Stop (Esc)" onClick=${stopAnswer}>
                   <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg></button>`
               : html`<button type="submit" class="send" aria-label="Send" title="Send (Enter)" disabled=${!canSend}><${Icon} name="arrow-up"/></button>`}
      </div>
      <${ThinkingPopover} open=${think} close=${() => setThink(false)} anchor=${thinkBtn} below=${center}/>
    </form>
    ${center ? null : html`<p class="composer-note">${note()}</p>`}
  </div>`;
}
function note() {
  if (S.hint) return S.hint;
  if (S.busy && S.busy.chat !== S.current) return "An answer is being written in another chat.";
  return S.health.mcp_tools && mcpOn() ? "Runs on this machine. Nothing leaves it, unless an MCP tool you turned on reaches the web."
                                       : "Runs on this machine. Nothing leaves it.";
}

// ------------------------------------------------------------------ a new chat: the model, the box, the chats to pick up
// the model that answers here: ready or busy, its context, the last answer's speed, images, what it saved - and the Models
// page, to switch to another
function ModelCard() {
  const h = S.health || {}, m = S.metrics, st = status(true);
  const ready = S.conn === "ok" && st.text === "Idle";
  const last = m && (m.requests || [])[0], sv = m && m.savings;
  const facts = [h.max_context ? ["Context", ctxfmt(h.max_context)] : null,
                 last && last.decode_tok_s ? ["Last answer", `${fmt(last.decode_tok_s, 1)} tok/s`] : null,
                 h.model ? ["Images", h.images ? "on" : "off"] : null,
                 sv && sv.priced ? ["Saved", usd(sv.usd)] : null].filter(Boolean);
  return html`<section class="model-card" aria-label="The model">
    <${ModelLogo} model=${h.model} size=${44}/>
    <div class="model-card__text">
      <div class="model-card__name"><b title=${h.model || ""}>${h.model_label || modelName(h.model) || "The model"}</b>
        <span class=${`pill pill--${ready ? "ok" : st.tone}`} role="status"><i class="status__dot"></i>${ready ? "Ready" : st.text}</span></div>
      ${facts.length ? html`<div class="model-card__facts num">${facts.map(([k, v]) => html`<span>${k}<b>${v}</b></span>`)}</div>` : null}
    </div>
    <button type="button" class="btn btn--secondary btn--sm" onClick=${() => showView("models")}><${Icon} name="swap" sm/>Switch model</button>
  </section>`;
}

// the chats to pick up again: the last three written in (a folder's name with its chats), newest first
function recentChats() {
  return S.chats.filter((c) => c !== S.current && c.messages.length)
    .sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, 3);
}
function Recent() {
  const list = recentChats();
  const folder = (c) => (c.folder && (S.folders.find((f) => f.id === c.folder) || {}).name) || "";
  return html`<section class="recent" aria-labelledby="recent-title"><h2 id="recent-title">Pick up where you left off</h2>
    ${list.length ? html`<div class="recent__list">${list.map((c) => html`<button type="button" class="recent__chat" key=${c.id} onClick=${() => openChat(c)}>
      <span class="recent__meta"><${Icon} name=${folder(c) ? "folder" : "chat"} sm/><span>${[folder(c), relTime(c.updated)].filter(Boolean).join(" · ")}</span></span>
      <b>${chatTitle(c)}</b></button>`)}</div>`
      : html`<p class="recent__none"><${Icon} name="chat" sm/><span>Your last three chats show here. They stay in this browser.</span></p>`}
  </section>`;
}

// a new chat: what are we building, the model that answers it, the box, and the chats to pick up again
function Greeting({phone}) {
  return html`<div class=${`greeting${phone ? " greeting--phone" : ""}`}>
    <h1>What are we building today?</h1>
    <${ModelCard}/>
    ${phone ? null : html`<${Composer} center/>`}
    <${Recent}/>
  </div>`;
}

// ------------------------------------------------------------------ the page
export function ChatPage() {
  const scroller = useRef(null);
  const [jump, setJump] = useState(false), [drop, setDrop] = useState(false);
  const empty = !S.current.messages.length && S.editIndex == null;
  const phone = isPhone();
  useEffect(() => {
    const s = scroller.current;
    chatView.el = s;
    // (the arrow: whenever the chat does not follow and the end is out of sight)
    chatView.onJump = () => setJump(!!s && S.current.messages.length > 0
      && s.scrollHeight - s.scrollTop - s.clientHeight > (chatView.follow && !chatView.held ? 120 : 8));
    return () => { if (chatView.el === s) chatView.el = null; };
  });
  useEffect(() => { scrollDown(true); }, [S.current]);
  // the chat follows a streaming answer down until its reader takes over - a wheel or a swipe up, or a finger (or the
  // mouse button) held on it - and again once they are back at the end or press the arrow.  (Only the reader's own
  // input stops it: a smooth scroll's first small steps used to count as "still at the end" and pulled it back down.)
  useEffect(() => {
    const s = scroller.current, gap = () => s.scrollHeight - s.scrollTop - s.clientHeight;
    const jump = () => { if (chatView.onJump) chatView.onJump(); };
    let downAt = 0, downTop = 0;
    const hold = () => { chatView.held = true; downAt = performance.now(); downTop = s.scrollTop; jump(); };
    const release = () => {
      if (!chatView.held) return;
      chatView.held = false;
      const tap = performance.now() - downAt < 350 && Math.abs(s.scrollTop - downTop) < 4;   // (a tap on a button)
      if (gap() < 8 || (tap && chatView.follow)) scrollDown(true);
      else { chatView.follow = false; jump(); }
    };
    const wheel = (e) => { if (e.deltaY < 0) { chatView.follow = false; jump(); } };
    const opts = {passive: true}, ends = ["touchend", "touchcancel", "mouseup"];
    s.addEventListener("wheel", wheel, opts);
    s.addEventListener("touchstart", hold, opts);
    s.addEventListener("mousedown", hold, opts);
    for (const t of ends) addEventListener(t, release, opts);
    return () => {
      s.removeEventListener("wheel", wheel, opts);
      s.removeEventListener("touchstart", hold, opts);
      s.removeEventListener("mousedown", hold, opts);
      for (const t of ends) removeEventListener(t, release, opts);
      chatView.held = false;
    };
  }, []);
  const onScroll = () => {
    const s = scroller.current, gap = s.scrollHeight - s.scrollTop - s.clientHeight;
    if (s.scrollTop < chatView.lastTop - 2 && gap >= 8) chatView.follow = false;             // moved up: keys, scrollbar
    else if (s.scrollTop > chatView.lastTop && gap < 8 && !chatView.held) chatView.follow = true;   // back down at the end
    chatView.lastTop = s.scrollTop;
    chatView.onJump();
  };
  const files = (e) => [...((e.dataTransfer || {}).types || [])].includes("Files");
  return html`<main class=${`chat${empty ? " chat--empty" : ""}`}
    onDragOver=${(e) => { if (!files(e)) return; e.preventDefault(); setDrop(true); }}
    onDragLeave=${(e) => { if (e.target === e.currentTarget || !e.currentTarget.contains(e.relatedTarget)) setDrop(false); }}
    onDrop=${(e) => { setDrop(false); if (!e.dataTransfer || !e.dataTransfer.files.length) return; e.preventDefault(); addFiles(e.dataTransfer.files); focusInput(); }}>
    <${Header} empty=${empty}/>
    <div class="chat__scroll" ref=${scroller} onScroll=${onScroll}>
      ${empty ? html`<${Greeting} phone=${phone}/>` : html`<${Thread}/>`}
    </div>
    ${jump && !empty ? html`<button type="button" class="jump" onClick=${() => scrollDown(true)} aria-label="Scroll to the latest">
      <${Icon} name="arrow-down" sm/></button>` : null}
    ${empty && !phone ? html`<p class="composer-note composer-note--foot">${note()}</p>` : html`<div class="chat__dock"><${Composer}/></div>`}
    ${drop ? html`<div class="drop" aria-hidden="true"><${Icon} name="attach" lg/><span>Drop files${S.health.images ? " or images" : ""} to attach them</span></div>` : null}
  </main>`;
}
