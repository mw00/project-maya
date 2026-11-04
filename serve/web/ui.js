// serve/web/ui.js - the small pieces every view is built from: icons, menus, switches, segmented choices, the dialog,
// the toasts, the model's logo and the thinking speedometer.
import {html, useEffect, useLayoutEffect, useRef, useState} from "./preact.js";
import {SPRITE, ctxfmt, modelLogo} from "./lib.js";
import {S, closeDialog, dismissToast} from "./state.js";

export const Icon = ({name, sm, lg, cls = ""}) => html`
  <svg class=${`st-icon${sm ? " st-icon--sm" : ""}${lg ? " st-icon--lg" : ""}${cls ? " " + cls : ""}`} aria-hidden="true">
    <use href=${`${SPRITE}#i-${name}`}/></svg>`;

// the Maya mark on its gradient (the only place the gradient is used)
export const MayaMark = ({size = 28}) => html`
  <span class="maya-mark" style=${`width:${size}px;height:${size}px`} aria-hidden="true"><svg viewBox="0 0 24 24"
    style=${`width:${Math.round(size * 0.64)}px;height:${Math.round(size * 0.64)}px`}><use href=${`${SPRITE}#i-maya`}/></svg></span>`;

export function ModelLogo({model, size = 32, cls = ""}) {
  const l = modelLogo(model);
  return html`<span class=${`model-logo ${cls}`} style=${`width:${size}px;height:${size}px`} aria-hidden="true">
    ${l ? html`<img src=${l.src} alt="" class=${l.mono ? "is-mono" : ""} width=${Math.round(size * 0.56)} height=${Math.round(size * 0.56)}/>`
        : html`<${Icon} name="cpu" sm/>`}</span>`;
}

// the thinking level as a speedometer: its needle points further right the longer the model thinks
const NEEDLE = {none: -70, auto: -47, low: -25, medium: 0, high: 25, xhigh: 47, max: 70};   // (MiniMax's seven too)
export const Gauge = ({level, cls = ""}) => html`
  <svg class=${`gauge ${cls}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true"><path d="M3.34 19a10 10 0 1 1 17.32 0"/>
    <path d="M12 14V8.5" transform=${`rotate(${NEEDLE[level] ?? 25} 12 14)`}/></svg>`;

export const Kbd = ({k}) => html`<kbd class="kbd">${k}</kbd>`;

// ------------------------------------------------------------------ closing a popup: a click outside it, or Escape
export function useDismiss(open, close, ...refs) {
  useEffect(() => {
    if (!open) return;
    const down = (e) => { if (!refs.some((r) => r.current && r.current.contains(e.target))) close(); };
    // Escape closes it and gives the keyboard back to the button that opened it (refs[1], by convention)
    const key = (e) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      close();
      const a = refs[1];
      if (a && a.current && a.current.focus) a.current.focus();
    };
    const t = setTimeout(() => document.addEventListener("pointerdown", down, true), 0);
    document.addEventListener("keydown", key, true);
    return () => { clearTimeout(t); document.removeEventListener("pointerdown", down, true); document.removeEventListener("keydown", key, true); };
  }, [open]);
}
// a menu's items: the arrow keys move between them, Home and End jump
export function menuKeys(e) {
  const items = [...e.currentTarget.querySelectorAll('[role^="menuitem"]:not([disabled])')];
  const i = items.indexOf(document.activeElement);
  let n = null;
  if (e.key === "ArrowDown") n = items[(i + 1) % items.length];
  else if (e.key === "ArrowUp") n = items[(i - 1 + items.length) % items.length];
  else if (e.key === "Home") n = items[0];
  else if (e.key === "End") n = items[items.length - 1];
  if (n) { e.preventDefault(); n.focus(); }
}
// a menu opened from a button, placed on the screen itself (a scrolling list or a dialog never cuts it off): below its
// button, or above when there is more room there; focus goes to its first item
export function Menu({open, onClose, anchor, cls = "", label, children, align = "right", width}) {
  const ref = useRef(null);
  const [pos, setPos] = useState(null);
  useDismiss(open, onClose, ref, anchor);
  useLayoutEffect(() => {
    if (!open) { setPos(null); return; }
    const place = () => {
      if (!ref.current || !anchor || !anchor.current) return;
      const a = anchor.current.getBoundingClientRect(), m = ref.current;
      const w = m.offsetWidth, h = m.scrollHeight;
      const below = innerHeight - a.bottom - 12, above = a.top - 12;
      const up = below < Math.min(h, 360) && above > below;
      const left = Math.max(8, Math.min(align === "left" ? a.left : a.right - w, innerWidth - w - 8));
      setPos({left, up, y: up ? innerHeight - a.top + 4 : a.bottom + 4, max: Math.max(160, (up ? above : below) - 4)});
    };
    place();
    addEventListener("resize", place);
    addEventListener("scroll", place, true);
    return () => { removeEventListener("resize", place); removeEventListener("scroll", place, true); };
  }, [open]);
  const placed = !!pos;
  useEffect(() => {
    if (!placed || !ref.current) return;
    const first = ref.current.querySelector('[role^="menuitem"]:not([disabled])');
    if (first) first.focus({preventScroll: true});
  }, [placed]);
  if (!open) return null;
  const style = (width ? `width:${width}px;` : "") + (pos
    ? `left:${pos.left}px;right:auto;${pos.up ? `bottom:${pos.y}px;top:auto` : `top:${pos.y}px;bottom:auto`};max-height:${pos.max}px`
    : "left:0;top:0;visibility:hidden");
  return html`<div ref=${ref} role="menu" aria-label=${label} onKeyDown=${menuKeys}
    class=${`menu ${pos && pos.up ? "menu--up" : ""} ${cls}`} style=${style}>${children}</div>`;
}
export const MenuItem = ({icon, label, onClick, danger, checked, sub, disabled, kbd}) => html`
  <button type="button" role=${checked != null ? "menuitemradio" : "menuitem"} aria-checked=${checked != null ? String(!!checked) : undefined}
    class=${`menu__item${danger ? " is-danger" : ""}`} onClick=${onClick} disabled=${disabled}>
    ${icon ? html`<${Icon} name=${icon} sm/>` : null}<span class="menu__label">${label}</span>
    ${sub ? html`<span class="menu__sub">${sub}</span>` : null}${kbd ? html`<${Kbd} k=${kbd}/>` : null}
    ${checked ? html`<${Icon} name="check" sm cls="menu__check"/>` : null}</button>`;
export const MenuSep = () => html`<div class="menu__sep" role="separator"></div>`;
export const MenuLabel = ({children}) => html`<div class="menu__heading">${children}</div>`;

// ------------------------------------------------------------------ inputs
export const Switch = ({on, onChange, label, disabled}) => html`
  <button type="button" role="switch" class="switch" aria-checked=${String(!!on)} aria-label=${label} disabled=${disabled}
    onClick=${() => onChange(!on)}><span class="switch__knob"></span></button>`;
// a few choices side by side (a radio group: the arrow keys move the choice)
export function Seg({options, value, onChange, label, cls = ""}) {
  const keys = (e) => {
    if (!["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const root = e.currentTarget;
    const i = options.findIndex((o) => o.id === value), d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1;
    const n = options[i < 0 ? 0 : (i + d + options.length) % options.length];
    onChange(n.id);
    setTimeout(() => { const b = root.querySelector('[aria-checked="true"]'); if (b) b.focus(); }, 0);
  };
  const none = !options.some((o) => o.id === value);   // (custom values: the first choice takes the Tab stop)
  return html`<div class=${`seg ${cls}`} role="radiogroup" aria-label=${label} onKeyDown=${keys}>
    ${options.map((o) => html`<button type="button" role="radio" aria-checked=${String(o.id === value)} title=${o.title}
      tabIndex=${o.id === value || (none && o === options[0]) ? 0 : -1} onClick=${() => onChange(o.id)}>${o.label}</button>`)}</div>`;
}
// the Context size slider (Settings > Model, and each model's Engine settings on the Models page): one stop per size in
// `stops` (server.js' ctxStopsIn), the size picked and "now" when it is the one the model has, the range's ends, and
// `hint` (server.js' ctxHint) under it.  `label`: the name beside the size (empty where the row around it names it).
export const CtxSlider = ({id, label = "Context size", value, now, stops, disabled, hint, onChange}) => html`<div class="ctx-slider">
  <label for=${id} class="set-split"><span>${label}</span><b class="num">${ctxfmt(value)} tokens${value === now ? " · now" : ""}</b></label>
  <input id=${id} type="range" class="range" min="0" max=${stops.length - 1} step="1" disabled=${disabled} aria-label="Context size"
    aria-valuetext=${`${ctxfmt(value)} tokens`} value=${Math.max(0, stops.indexOf(value))} onInput=${(ev) => onChange(stops[+ev.target.value])}/>
  <span class="set-split set-split--muted"><span>${ctxfmt(stops[0])}</span><span>${ctxfmt(stops[stops.length - 1])}</span></span>
  ${hint ? html`<span class="set-row__sub">${hint}</span>` : null}</div>`;

// ------------------------------------------------------------------ the dialog (confirmations, a text to type, the preview)
function focusables(root) {
  return [...root.querySelectorAll('button, [href], input, select, textarea, iframe, [tabindex]:not([tabindex="-1"])')]
    .filter((e) => !e.disabled && !e.hidden && e.offsetParent !== null);
}
export function trapTab(e, root) {
  const f = focusables(root);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}
export function DialogHost() {
  const d = S.dialog;
  const box = useRef(null), nodeHost = useRef(null), input = useRef(null);
  const [value, setValue] = useState("");
  useEffect(() => {
    if (!d) return;
    if (d.prompt) setValue(d.prompt.value || "");
    if (d.node && nodeHost.current) { nodeHost.current.innerHTML = ""; nodeHost.current.appendChild(d.node); }
    setTimeout(() => {
      if (input.current) { input.current.focus(); input.current.select && input.current.select(); return; }
      // a confirmation that destroys something starts on Cancel (a stray Enter must not confirm it)
      const danger = (d.actions || []).some((a) => a.danger);
      const b = box.current && (danger ? box.current.querySelector(".dialog__actions .btn--secondary")
                                       : box.current.querySelector(".btn--primary") || box.current.querySelector(".dialog__close"));
      if (b) b.focus();
    }, 0);
  }, [d]);
  if (!d) return null;
  const act = (a) => { closeDialog(true); if (a.run) a.run(d.prompt ? value : undefined); };
  const primary = (d.actions || []).find((a) => a.primary);
  const keys = (e) => {
    if (e.key === "Tab") trapTab(e, box.current);
    else if (e.key === "Escape") { e.stopPropagation(); closeDialog(); }
    else if (e.key === "Enter" && d.prompt && !d.prompt.multiline && primary) { e.preventDefault(); act(primary); }
  };
  return html`<div class="dialog-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) closeDialog(); }}>
    <div ref=${box} class=${`dialog${d.wide ? " dialog--wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby="dialog-title" onKeyDown=${keys}>
      <div class="dialog__head"><h2 id="dialog-title">${d.title}</h2>
        <button type="button" class="icon-btn dialog__close" aria-label="Close" onClick=${() => closeDialog()}><${Icon} name="close"/></button></div>
      <div class="dialog__body">
        ${d.body ? html`<div class="prose" dangerouslySetInnerHTML=${{__html: d.body}}></div>` : null}
        ${d.node ? html`<div ref=${nodeHost}></div>` : null}
        ${d.prompt ? html`<label class="field">${d.prompt.label ? html`<span class="field__label">${d.prompt.label}</span>` : null}
          ${d.prompt.multiline
            ? html`<textarea ref=${input} class="input input--area" rows="7" placeholder=${d.prompt.placeholder} value=${value}
                onInput=${(e) => setValue(e.target.value)}></textarea>`
            : html`<input ref=${input} class="input" placeholder=${d.prompt.placeholder} value=${value}
                onInput=${(e) => setValue(e.target.value)}/>`}
          ${d.prompt.note ? html`<span class="field__note">${d.prompt.note}</span>` : null}</label>` : null}
      </div>
      ${(d.actions || []).length ? html`<div class="dialog__actions">${d.actions.map((a) => html`
        <button type="button" class=${`btn ${a.primary ? "btn--primary" : a.danger ? "btn--danger" : "btn--secondary"}`}
          onClick=${() => act(a)}>${a.label}</button>`)}</div>` : null}
    </div></div>`;
}

// ------------------------------------------------------------------ toasts
const TOAST_ICON = {info: "info", success: "check", warn: "warning", error: "error"};
export const Toasts = () => html`<div class="toasts" aria-live="polite">${S.toasts.map((t) => html`
  <div key=${t.id} class=${`toast toast--${t.kind}`} role=${t.kind === "error" ? "alert" : "status"}>
    <${Icon} name=${TOAST_ICON[t.kind] || "info"}/>
    <div class="toast__text"><b>${t.title}</b>${t.text ? html`<span>${t.text}</span>` : null}</div>
    ${t.action ? html`<button type="button" class="btn btn--secondary btn--sm" onClick=${() => { t.action.run(); dismissToast(t.id); }}>${t.action.label}</button>` : null}
    <button type="button" class="icon-btn icon-btn--sm toast__x" aria-label="Dismiss" title="Dismiss"
      onClick=${() => dismissToast(t.id)}><${Icon} name="close" sm/></button>
  </div>`)}</div>`;
