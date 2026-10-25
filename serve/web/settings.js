// serve/web/settings.js - the Settings dialog.  (The redesign's next step fills in its sections.)
import {html} from "./preact.js";
import {S, emit} from "./state.js";
import {Icon} from "./ui.js";

export function SettingsDialog() {
  const close = () => { S.settings = null; emit(); };
  return html`<div class="dialog-scrim" onPointerDown=${(e) => { if (e.target === e.currentTarget) close(); }}>
    <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title"
      onKeyDown=${(e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } }}>
      <div class="dialog__head"><h2 id="settings-title">Settings</h2>
        <button type="button" class="icon-btn" aria-label="Close" onClick=${close}><${Icon} name="close"/></button></div>
      <div class="dialog__body"><p>The new Settings arrive in the next step of the redesign.</p></div>
    </div></div>`;
}
