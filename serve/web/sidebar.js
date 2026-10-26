// serve/web/sidebar.js - the sidebar (new chat, search, Models, Monitor, Settings; the chats: pinned, in folders, then by day;
// the model's status), a chat's ⋯ menu, and the search dialog.
import {html, useEffect, useMemo, useRef, useState} from "./preact.js";
import {MOD, ctxfmt, dayGroup, fmt, modelName, relTime, store} from "./lib.js";
import {S, emit} from "./state.js";
import {chatInstructions, chatTitle, deleteChat, exportChat, moveToFolder, newChat, newFolder, openChat, pinChat,
        renameChat, renameFolder, deleteFolder, toggleFolder} from "./chats.js";
import {Icon, Kbd, MayaMark, Menu, MenuItem, MenuLabel, MenuSep, ModelLogo, trapTab} from "./ui.js";

export const showView = (v) => { S.view = v; S.drawer = false; emit(); };
export const openSettings = (section = "general") => { S.settings = section; S.drawer = false; emit(); };
export const setSide = (open) => { S.sideOpen = open; store.set("side-open", open); emit(); };

// ------------------------------------------------------------------ the model's status, in words and a tone
export function status() {
  const m = S.metrics, live = (m && m.live) || {};
  const ctx = S.health.max_context ? `${ctxfmt(S.health.max_context)} context` : "";
  if (S.conn === "down") return {tone: "danger", text: "Server not reachable"};
  if (S.conn === "key") return {tone: "warn", text: "API key needed"};
  if (S.updateWatch) return {tone: "info", text: "Updating"};
  if (S.reload) return {tone: "info", text: "Reloading"};
  if (!m) return {tone: "muted", text: "Connecting…"};
  if (live.queued > 0) return {tone: "warn", text: `${live.queued} queued`};
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    return {tone: "info", text: pct != null ? `Reading · ${pct}%` : "Reading"};
  }
  if (live.state === "generating") {
    const many = live.active > 1 ? ` · ${live.active} at once` : "";   // (--parallel: answers written together)
    return {tone: "accent", text: (live.tok_s ? `Writing · ${fmt(live.tok_s, 1)} tok/s` : "Writing") + many};
  }
  return {tone: "muted", text: ["Idle", ctx].filter(Boolean).join(" · ")};
}

// ------------------------------------------------------------------ a chat's ⋯ menu (in the sidebar and the header)
export function ChatMenuItems({c, close}) {
  const run = (f) => () => { close(); f(); };
  if (!c.messages.length) {
    return html`<${MenuItem} icon="notebook-pen" label="Instructions for this chat" onClick=${run(() => chatInstructions(c))}/>`;
  }
  return html`
    <${MenuItem} icon=${c.pinned ? "pin-off" : "pin"} label=${c.pinned ? "Unpin" : "Pin"} onClick=${run(() => pinChat(c, !c.pinned))}/>
    <${MenuSep}/>
    <${MenuLabel}>Move to folder<//>
    ${S.folders.map((f) => html`<${MenuItem} key=${f.id} icon="folder" label=${f.name} checked=${c.folder === f.id}
      onClick=${run(() => moveToFolder(c, c.folder === f.id ? null : f.id))}/>`)}
    <${MenuItem} icon="folder-plus" label="New folder" onClick=${run(() => newFolder(c))}/>
    ${c.folder ? html`<${MenuItem} icon="folder-minus" label="Take out of the folder" onClick=${run(() => moveToFolder(c, null))}/>` : null}
    <${MenuSep}/>
    <${MenuItem} icon="edit" label="Rename" onClick=${run(() => renameChat(c))}/>
    <${MenuItem} icon="notebook-pen" label="Instructions" onClick=${run(() => chatInstructions(c))}/>
    <${MenuItem} icon="download" label="Export as Markdown" onClick=${run(() => exportChat(c))}/>
    <${MenuItem} icon="trash" label="Delete" danger onClick=${run(() => deleteChat(c))}/>`;
}

function ChatRow({c, indent}) {
  const [open, setOpen] = useState(false);
  const btn = useRef(null);
  const close = () => { setOpen(false); if (btn.current) btn.current.focus(); };   // (a dialog then returns here)
  const cur = c === S.current && S.view === "chat";
  const writing = S.busy && S.busy.chat === c;
  return html`<div class=${`row${cur ? " is-current" : ""}${open ? " is-menu" : ""}`}>
    <button type="button" class=${`row__open${indent ? " row__open--in" : ""}`} title=${chatTitle(c)} aria-current=${cur ? "page" : undefined}
      onClick=${() => openChat(c)}>${chatTitle(c)}</button>
    ${writing ? html`<span class="row__live" title="An answer is being written" aria-label="Writing"></span>`
      : c.pinned ? html`<span class="row__pin" title="Pinned"><${Icon} name="pin" sm/></span>` : null}
    <button type="button" ref=${btn} class="row__more" aria-label=${`Options for ${chatTitle(c)}`} aria-haspopup="menu"
      aria-expanded=${String(open)} onClick=${() => setOpen(!open)}><${Icon} name="ellipsis" sm/></button>
    <${Menu} open=${open} onClose=${() => setOpen(false)} anchor=${btn} label="Chat options" width=${240}>
      <${ChatMenuItems} c=${c} close=${close}/>
    <//>
  </div>`;
}

function FolderRow({f, n}) {
  const [open, setOpen] = useState(false);
  const btn = useRef(null);
  const expanded = !!S.folderOpen[f.id];
  return html`<div class=${`row row--folder${open ? " is-menu" : ""}`}>
    <button type="button" class="row__open row__folder" aria-expanded=${String(expanded)} onClick=${() => toggleFolder(f.id)}>
      <${Icon} name="chevron" sm cls=${expanded ? "chev" : "chev is-closed"}/><${Icon} name="folder" sm/>
      <span class="row__name">${f.name}</span><span class="row__count">${n}</span></button>
    <button type="button" ref=${btn} class="row__more" aria-label=${`Options for the folder ${f.name}`} aria-haspopup="menu"
      aria-expanded=${String(open)} onClick=${() => setOpen(!open)}><${Icon} name="ellipsis" sm/></button>
    <${Menu} open=${open} onClose=${() => setOpen(false)} anchor=${btn} label="Folder options" width=${220}>
      <${MenuItem} icon="edit" label="Rename" onClick=${() => { setOpen(false); btn.current.focus(); renameFolder(f); }}/>
      <${MenuItem} icon="trash" label="Delete folder" danger onClick=${() => { setOpen(false); btn.current.focus(); deleteFolder(f); }}/>
    <//>
  </div>`;
}

function ChatList() {
  const chats = S.chats;
  const pinned = chats.filter((c) => c.pinned);
  const byFolder = {};
  for (const c of chats) if (c.folder && !c.pinned) (byFolder[c.folder] = byFolder[c.folder] || []).push(c);
  const known = new Set(S.folders.map((f) => f.id));
  const rest = chats.filter((c) => !c.pinned && !(c.folder && known.has(c.folder)));
  const groups = [];
  for (const c of rest) {
    const g = dayGroup(c.updated);
    if (!groups.length || groups[groups.length - 1].label !== g) groups.push({label: g, chats: []});
    groups[groups.length - 1].chats.push(c);
  }
  return html`<div class="side__list" aria-label="Chats">
    ${pinned.length ? html`<div class="side__label">Pinned</div>${pinned.map((c) => html`<${ChatRow} key=${c.id} c=${c}/>`)}` : null}
    <div class="side__label side__label--row"><span>Folders</span>
      <button type="button" class="icon-btn icon-btn--sm" aria-label="New folder" title="New folder" onClick=${() => newFolder()}>
        <${Icon} name="plus" sm/></button></div>
    ${S.folders.length ? null : html`<div class="side__note">Keep chats together: add a folder with +.</div>`}
    ${S.folders.map((f) => {
      const inIt = byFolder[f.id] || [];
      return html`<${FolderRow} key=${f.id} f=${f} n=${inIt.length}/>
        ${S.folderOpen[f.id] ? (inIt.length ? inIt.map((c) => html`<${ChatRow} key=${c.id} c=${c} indent/>`)
                                             : html`<div class="side__note side__note--in">Empty. Move a chat here from its ⋯ menu.</div>`) : null}`;
    })}
    ${groups.map((g) => html`<div class="side__label">${g.label}</div>${g.chats.map((c) => html`<${ChatRow} key=${c.id} c=${c}/>`)}`)}
    ${chats.length ? null : html`<div class="side__note">Your chats appear here. They stay in this browser.</div>`}
  </div>`;
}

function StatusCard() {
  const st = status();
  const name = modelName(S.health.model);
  return html`<div class="side__foot">
    <button type="button" class="status-card" onClick=${() => showView("monitor")} aria-label=${`${name}, ${st.text}. Open the Monitor`}>
      <${ModelLogo} model=${S.health.model}/>
      <span class="status-card__text"><b title=${S.health.model}>${name}</b>
        <span class=${`status status--${st.tone}`}><i class="status__dot"></i>${st.text}</span></span>
    </button></div>`;
}

export function Sidebar({drawer}) {
  const newer = S.update && S.update.newer && !S.updateWatch;
  const head = useRef(null);
  useEffect(() => {   // the phone's drawer: the keyboard goes into it, and back to the Chats button when it closes
    if (!drawer) return;
    const ret = document.activeElement;
    const b = head.current && head.current.querySelector("button");
    if (b) b.focus();
    return () => { if (ret && ret.focus && document.contains(ret)) ret.focus(); };
  }, [drawer]);
  return html`<nav class=${`side${drawer ? " side--drawer" : ""}`} aria-label="Maya" role=${drawer ? "dialog" : undefined}
    aria-modal=${drawer ? "true" : undefined}>
    <div class="side__head" ref=${head}>
      <span class="side__brand"><${MayaMark}/><span>Maya</span></span>
      ${drawer
        ? html`<button type="button" class="icon-btn" aria-label="Close the chats" onClick=${() => { S.drawer = false; emit(); }}><${Icon} name="close"/></button>`
        : html`<button type="button" class="icon-btn" aria-label="Close the sidebar" title="Close the sidebar" onClick=${() => setSide(false)}>
            <${Icon} name="panel-left"/></button>`}
    </div>
    <button type="button" class="nav-item" onClick=${newChat}><${Icon} name="new-chat"/><span>New chat</span><${Kbd} k=${`${MOD} ⇧ O`}/></button>
    <button type="button" class="nav-item" onClick=${() => { S.search = true; S.drawer = false; emit(); }}>
      <${Icon} name="search"/><span>Search chats</span><${Kbd} k=${`${MOD} K`}/></button>
    <button type="button" class=${`nav-item${S.view === "models" ? " is-current" : ""}`} aria-current=${S.view === "models" ? "page" : undefined}
      onClick=${() => showView("models")}><${Icon} name="boxes"/><span>Models</span></button>
    <button type="button" class=${`nav-item${S.view === "monitor" ? " is-current" : ""}`} aria-current=${S.view === "monitor" ? "page" : undefined}
      onClick=${() => showView("monitor")}><${Icon} name="activity"/><span>Monitor</span></button>
    <button type="button" class="nav-item" onClick=${() => openSettings(newer ? "about" : "general")}>
      <${Icon} name="settings"/><span>Settings</span>${newer ? html`<i class="nav-item__dot" title="A new version is available"></i>` : null}
      <${Kbd} k=${`${MOD} ,`}/></button>
    <${ChatList}/>
    <${StatusCard}/>
  </nav>`;
}

// ------------------------------------------------------------------ search (Ctrl K)
function snippet(c, q) {
  const title = chatTitle(c);
  if (q) {
    for (const m of c.messages) {
      const t = (m.text || "").replace(/\s+/g, " "), i = t.toLowerCase().indexOf(q);
      if (i >= 0 && t.trim() !== title) return (i > 30 ? "…" : "") + t.slice(Math.max(0, i - 30), i + 90);
    }
  }
  const a = c.messages.find((m) => m.role === "assistant" && m.text);
  return [relTime(c.updated), a ? a.text.replace(/[#*`>|_-]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 90) : ""].filter(Boolean).join(" · ");
}
export function SearchDialog() {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const box = useRef(null), input = useRef(null);
  useEffect(() => { setTimeout(() => input.current && input.current.focus(), 0); }, []);
  const ql = q.trim().toLowerCase();
  const found = useMemo(() => (ql ? S.chats.filter((c) => chatTitle(c).toLowerCase().includes(ql) ||
    c.messages.some((m) => (m.text || "").toLowerCase().includes(ql))) : S.chats).slice(0, 50), [ql, S.chats.length]);
  const close = () => { S.search = false; emit(); };
  const go = (c) => { close(); openChat(c); };
  const keys = (e) => {
    if (e.key === "Escape") { e.stopPropagation(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setSel(Math.min(found.length - 1, sel + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel(Math.max(0, sel - 1)); }
    else if (e.key === "Enter" && found[sel]) { e.preventDefault(); go(found[sel]); }
    else if (e.key === "Tab") trapTab(e, box.current);
  };
  useEffect(() => {
    const el = box.current && box.current.querySelector(`[data-i="${sel}"]`);
    if (el) el.scrollIntoView({block: "nearest"});
  }, [sel]);
  return html`<div class="dialog-scrim dialog-scrim--top" onPointerDown=${(e) => { if (e.target === e.currentTarget) close(); }}>
    <div ref=${box} class="dialog search" role="dialog" aria-modal="true" aria-label="Search chats" onKeyDown=${keys}>
      <div class="search__bar"><${Icon} name="search"/>
        <input ref=${input} class="search__input" placeholder="Search chats" aria-label="Search chats" value=${q}
          role="combobox" aria-expanded="true" aria-controls="search-results" aria-activedescendant=${found[sel] ? `sr-${sel}` : undefined}
          onInput=${(e) => { setQ(e.target.value); setSel(0); }}/>
        <button type="button" class="icon-btn" aria-label="Close" onClick=${close}><${Icon} name="close"/></button></div>
      <div class="search__results" id="search-results" role="listbox" aria-label="Chats">
        ${!ql ? html`<button type="button" class="search__item search__item--new" onClick=${() => { close(); newChat(); }}>
          <${Icon} name="new-chat"/><span>New chat</span></button>` : null}
        ${found.map((c, i) => html`<button type="button" key=${c.id} id=${`sr-${i}`} data-i=${i} role="option" aria-selected=${String(i === sel)}
          class=${`search__item${i === sel ? " is-sel" : ""}`} onClick=${() => go(c)} onPointerMove=${() => sel !== i && setSel(i)}>
          <${Icon} name=${c.pinned ? "pin" : c.folder ? "folder" : "chat"}/>
          <span class="search__text"><b>${chatTitle(c)}</b><span>${snippet(c, ql)}</span></span></button>`)}
        ${found.length ? null : html`<p class="search__none">${ql ? "No chat matches." : "No chats yet."}</p>`}
      </div>
    </div></div>`;
}
