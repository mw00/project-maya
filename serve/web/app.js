// serve/web/app.js - the Project Maya web app: the chat, the Monitor and Settings, built with Preact + htm (web/preact.js,
// local: the page needs nothing from the network).  The state lives in state.js; the server's side in server.js; the chats
// in chats.js.
import {html, render, useEffect} from "./preact.js";
import {isPhone} from "./lib.js";
import {S, applyScheme, applyTheme, closeDialog, emit, useStore} from "./state.js";
import {loadHealth, loadMcp, loadUpdate, poll, whenHealth} from "./server.js";
import {cancelEdit, chatView, focusInput, loadChats, newChat, newChatObj, openChat, resumePending, send, stopAnswer,
        updateCtxMeter} from "./chats.js";
import {DialogHost, Toasts, trapTab} from "./ui.js";
import {SearchDialog, Sidebar, openSettings, setSide, showView} from "./sidebar.js";
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
  return html`<div class=${`shell${S.sideOpen && !phone ? "" : " shell--closed"}`}>
    ${phone ? null : S.sideOpen ? html`<${Sidebar}/>` : null}
    ${phone && S.drawer ? html`<div class="drawer-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) { S.drawer = false; emit(); } }}>
      <${Sidebar} drawer/></div>` : null}
    ${S.view === "monitor" ? html`<${MonitorPage}/>` : S.view === "models" ? html`<${ModelsPage}/>` : html`<${ChatPage}/>`}
    ${S.settings ? html`<${SettingsDialog}/>` : null}
    ${S.search ? html`<${SearchDialog}/>` : null}
    <${DialogHost}/>
    <${Toasts}/>
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
addEventListener("resize", () => { if (isPhone() !== wasPhone) { wasPhone = isPhone(); S.drawer = false; emit(); } });

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
