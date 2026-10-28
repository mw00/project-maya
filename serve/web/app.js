// serve/web/app.js - the Project Maya web app: the chat, the Monitor and Settings, built with Preact + htm (web/preact.js,
// local: the page needs nothing from the network).  The state lives in state.js; the server's side in server.js; the chats
// in chats.js.
import {html, render, useEffect, useRef, useState} from "./preact.js";
import {isPhone} from "./lib.js";
import {S, applyScheme, applyTheme, closeDialog, emit, useStore} from "./state.js";
import {loadHealth, loadMcp, loadUpdate, poll, whenHealth} from "./server.js";
import {cancelEdit, chatView, focusInput, loadChats, newChat, newChatObj, openChat, resumePending, send, stopAnswer,
        updateCtxMeter} from "./chats.js";
import {DialogHost, Toasts, trapTab} from "./ui.js";
import {SavedSheet, SearchDialog, Sidebar, openSettings, setSide, showView} from "./sidebar.js";
import {ChatPage} from "./chat-view.js";
import {SettingsDialog} from "./settings.js";
import {MonitorPage} from "./monitor.js";
import {ModelsPage} from "./models.js";

applyTheme();
applyScheme();

function App() {
  useStore();
  const phone = isPhone();
  useEffect(() => {
    // the address follows the view: #monitor, #models, #settings (an old #about opens Settings' About)
    const want = S.view === "monitor" ? "#monitor" : S.view === "models" ? "#models" : S.settings ? "#settings" : "";
    if (location.hash !== want) history.replaceState(null, "", want || location.pathname + location.search);
  });
  const page = S.view === "monitor" ? html`<${MonitorPage}/>` : S.view === "models" ? html`<${ModelsPage}/>` : html`<${ChatPage}/>`;
  return html`<div class=${`shell${S.sideOpen && !phone ? "" : " shell--closed"}`}>
    ${phone ? html`<${PhonePush} page=${page}/>` : S.sideOpen ? html`<${Sidebar}/>` : null}
    ${phone ? null : page}
    ${S.settings ? html`<${SettingsDialog}/>` : null}
    ${S.search ? html`<${SearchDialog}/>` : null}
    ${S.savedSheet && phone ? html`<${SavedSheet}/>` : null}
    <${DialogHost}/>
    <${Toasts}/>
  </div>`;
}

// a phone: the chats slide in from the left and push the page over (as ChatGPT's app), its edge left in view, dimmed; a
// tap there, a swipe to the left, Escape or × closes them.  Both stay in the page, so they slide back as well.
function PhonePush({page}) {
  const open = !!S.drawer;
  const [drag, setDrag] = useState(0);
  const touch = useRef(null);
  const close = () => { S.drawer = false; emit(); };
  const start = (e) => {
    touch.current = open && e.touches.length === 1 ? {x: e.touches[0].clientX, y: e.touches[0].clientY, on: false} : null;
  };
  const move = (e) => {
    const t = touch.current;
    if (!t) return;
    const dx = e.touches[0].clientX - t.x, dy = e.touches[0].clientY - t.y;
    if (!t.on) {
      if (Math.abs(dy) > 10 && Math.abs(dy) > Math.abs(dx)) { touch.current = null; return; }   // (a scroll)
      if (dx > -10) return;
      t.on = true;
    }
    setDrag(Math.min(0, dx));
  };
  const end = () => {
    const t = touch.current;
    touch.current = null;
    if (!t || !t.on) return;
    if (drag < -70) close();
    setDrag(0);
  };
  return html`<div class=${`push${open ? " is-open" : ""}${drag ? " is-drag" : ""}`} style=${drag ? `--push-drag:${drag}px` : ""}
      onTouchStart=${start} onTouchMove=${move} onTouchEnd=${end} onTouchCancel=${end}>
    <${Sidebar} drawer open=${open}/>
    <div class="push__main" inert=${open}>${page}</div>
    <button type="button" class="push__dim" tabindex="-1" aria-hidden="true" onClick=${close}></button>
  </div>`;
}

// ------------------------------------------------------------------ the keyboard
document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.shiftKey && (e.key === "O" || e.key === "o")) { e.preventDefault(); newChat(); return; }
  if (mod && !e.shiftKey && (e.key === "k" || e.key === "K")) { e.preventDefault(); S.search = !S.search; emit(); return; }
  if (mod && e.key === ",") { e.preventDefault(); if (S.settings) S.settings = null; else openSettings(); emit(); return; }
  if (mod && e.shiftKey && (e.key === "S" || e.key === "s") && !isPhone()) { e.preventDefault(); setSide(!S.sideOpen); return; }
  if (e.key === "Tab" && S.drawer) { const d = document.querySelector(".side--drawer"); if (d) trapTab(e, d); return; }
  if (e.key !== "Escape" || e.defaultPrevented) return;
  // (menus, popovers and dialogs with the keyboard inside them close themselves first; this is Escape anywhere else)
  if (S.dialog) { closeDialog(); return; }
  if (S.search) { S.search = false; emit(); return; }
  if (S.settings) { S.settings = null; emit(); return; }
  if (S.drawer) { S.drawer = false; emit(); return; }
  if (S.editIndex != null) { cancelEdit(); return; }
  if (S.busy && S.busy.chat === S.current && S.view === "chat") stopAnswer();
});

// a phone turned into a tablet (or a window resized past the breakpoint) gets the other layout
let wasPhone = isPhone();
addEventListener("resize", () => { if (isPhone() !== wasPhone) { wasPhone = isPhone(); S.drawer = false; S.savedSheet = false; emit(); } });

// the address: #monitor, #models, #settings, #about
function route() {
  const h = location.hash.slice(1);
  if (h === "monitor" || h === "models") showView(h);
  else if (h === "settings" || h === "about") { S.view = S.view === "monitor" || S.view === "models" ? S.view : "chat"; openSettings(h === "about" ? "about" : "general"); }
  else if (!h && (S.view === "monitor" || S.view === "models")) showView("chat");
}
addEventListener("hashchange", route);

// ------------------------------------------------------------------ start
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
S.current = newChatObj();
render(html`<${App}/>`, document.getElementById("app"));
route();
whenHealth(() => updateCtxMeter());
loadChats().then(() => { if (S.view === "chat" && !S.settings) focusInput(); return loadHealth(); }).then(loadMcp).then(() => {
  loadUpdate();                                    // (Settings' dot when a new version is out)
  resumePending();
  if (startQuestion) {
    if (S.current.messages.length) openChat(newChatObj());
    S.draft = startQuestion;
    send();
  }
});
poll();
window.maya = {S, chatView};   // for a look from the browser's console
