// serve/web/account.js - the server's users, as the page sees them (the server's side: serve/users.py): who this page
// is (GET /auth/me), the sign-in a server with users opens on, and Settings' Account pages - Your account (the password,
// signing out), API keys (each user's own; an admin sees everyone's) and Users (an admin's: add, role, disable, delete).
// What a standard user may not do is hidden with state.js' isAdmin(); the server refuses it whatever the page shows.
import {html, useEffect, useRef, useState} from "./preact.js";
import {api, fmt, headers, relTime, store} from "./lib.js";
import {S, confirmDialog, copyText, emit, isAdmin, promptDialog, toast} from "./state.js";
import {Icon, MayaMark, Menu, MenuItem, MenuSep, Seg} from "./ui.js";

// ------------------------------------------------------------------ who this page is
function setMe(me) {
  S.me = {users: !!me.users, available: !!me.available, user: me.user || null};
  store.set("me", S.me);                           // (a name and a role, never a key: the page opens as it was)
  emit();
}
const ask = async () => {
  const r = await fetch("auth/me", {headers: headers()});
  if (r.status === 404) return {users: false, available: false, user: null};   // (a server without users at all)
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
};
export async function loadMe() {
  try {
    let me = await ask();
    if (store.get("session", "") && (!me.users || !me.user)) {
      // the session ended (or the server has no users any more): the page's API key, when it has one, may still get in
      store.set("session", "");
      me = await ask();
    }
    const was = S.me;
    setMe(me);
    // signed out while the page was open (the session ended, the user was disabled): start again at the sign-in
    if (was.users && was.user && me.users && !me.user) location.reload();
  } catch (e) {
    setTimeout(loadMe, 2000);                      // (the server is starting: it says who we are when it is up)
  }
  return S.me;
}
export const needsSignIn = () => S.me.users && !S.me.user;
// the server said 401 to a page that was let in: ask again who we are (at most every few seconds)
let asked = 0;
export function recheckMe() {
  if (Date.now() - asked < 5000) return;
  asked = Date.now();
  loadMe();
}

async function signIn(name, password) {
  const out = await api("auth/login", {method: "POST", body: {name, password}});
  store.set("session", out.token);
  setMe({users: true, available: true, user: out.user});
  location.reload();                               // (everything the page asked before it was let in: asked again)
}
export async function signOut() {
  const via = S.me.user && S.me.user.via;
  try { if (via === "session") await api("auth/logout", {method: "POST", body: {}}); } catch (e) { /* signed out anyway */ }
  store.set("session", "");
  if (via !== "session") store.set("apikey", "");  // (let in by a key: the page forgets it)
  setMe({...S.me, user: null});
  location.reload();
}

// ------------------------------------------------------------------ the sign-in (instead of the app, until let in)
export function Login() {
  const [name, setName] = useState(""), [pw, setPw] = useState(""), [key, setKey] = useState("");
  const [byKey, setByKey] = useState(false), [err, setErr] = useState(""), [busy, setBusy] = useState(false);
  const first = useRef(null);
  useEffect(() => { if (first.current) first.current.focus(); }, [byKey]);
  const submit = async (e) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setErr("");
    try {
      if (byKey) {
        store.set("apikey", key.trim());
        const me = await ask();
        if (!me.user) { store.set("apikey", ""); throw new Error("This server does not know that key."); }
        setMe(me);
        location.reload();
      } else {
        await signIn(name.trim(), pw);
      }
    } catch (e2) {
      setErr(e2.message || "Could not sign in.");
      setBusy(false);
    }
  };
  const model = S.health.model_label || S.health.model;
  return html`<main class="login">
    <form class="login__card" onSubmit=${submit} aria-labelledby="login-title">
      <div class="login__brand"><${MayaMark} size=${44}/>
        <h1 id="login-title">Sign in to Maya</h1>
        <p>${model && model !== "maya" ? `${model}, on this server.` : "This server has users: sign in to chat."}</p></div>
      ${byKey ? html`
        <label class="field"><span class="field__label">API key</span>
          <input ref=${first} class="input" type="password" autocomplete="off" placeholder="sk-maya-…" value=${key}
            onInput=${(e) => setKey(e.target.value)}/>
          <span class="field__note">One of your own keys, or the key in the server's config. Kept in this browser.</span></label>`
      : html`
        <label class="field"><span class="field__label">Name</span>
          <input ref=${first} class="input" name="username" autocomplete="username" autocapitalize="none" spellcheck="false"
            value=${name} onInput=${(e) => setName(e.target.value)}/></label>
        <label class="field"><span class="field__label">Password</span>
          <input class="input" type="password" name="password" autocomplete="current-password" value=${pw}
            onInput=${(e) => setPw(e.target.value)}/></label>`}
      ${err ? html`<div class="msg-error" role="alert">${err}</div>` : null}
      <button type="submit" class="btn btn--primary login__go" disabled=${busy || (byKey ? !key.trim() : !name.trim() || !pw)}>
        ${busy ? "Signing in…" : "Sign in"}</button>
      <button type="button" class="btn btn--ghost btn--sm" onClick=${() => { setByKey(!byKey); setErr(""); }}>
        ${byKey ? "Sign in with a name and password" : "Use an API key instead"}</button>
      <p class="login__note">No account? The admin of this server adds users.</p>
    </form></main>`;
}

// ------------------------------------------------------------------ Settings' pages: which ones this user has
export function accountPages() {
  const me = S.me, u = me.user;
  if (!me.available) return [];                    // (a server that keeps no users: nothing to show)
  const pages = [];
  if (u && u.id != null) pages.push({id: "account", label: "Your account", icon: "user"}, {id: "keys", label: "API keys", icon: "key"});
  if (isAdmin()) pages.push({id: "users", label: "Users", icon: "users"});
  return pages;
}
const ROLE = {admin: ["accent", "Admin"], user: ["muted", "Standard"]};
const RoleBadge = ({role}) => html`<span class=${`badge badge--${(ROLE[role] || ROLE.user)[0]}`}>${(ROLE[role] || [0, role])[1]}</span>`;
const when = (t) => (t ? relTime(t * 1000) : "never");
const PW_NOTE = "8 characters or more.";

// ------------------------------------------------------------------ Your account
export function AccountPage({ui: {Head, Group, Row}}) {
  const u = S.me.user || {};
  const [cur, setCur] = useState(""), [pw, setPw] = useState(""), [again, setAgain] = useState(""), [busy, setBusy] = useState(false);
  const bad = pw && pw.length < 8 ? "The new password is shorter than 8 characters." : again && pw !== again ? "The two new passwords differ." : "";
  const save = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api("auth/password", {method: "POST", body: {current: cur, new: pw}});
      setCur(""); setPw(""); setAgain("");
      toast("success", "Password changed", "Other browsers you were signed in on are signed out. Your API keys stay.", 6000);
    } catch (err) { toast("error", "The password did not change", err.message, 6000); }
    setBusy(false);
  };
  const out = async () => {
    if (await confirmDialog("Sign out?", "<p>This browser forgets who you are. Your chats stay in this browser, and your API keys keep working.</p>", "Sign out")) signOut();
  };
  return html`<${Head} title="Your account" intro="Who you are on this server."/>
    <${Group}>
      <${Row} label=${u.name} sub=${u.role === "admin" ? "An admin: everything, including the models, the server's settings and the users."
                                                      : "A standard user: the chat, the APIs and your own API keys. The models and the server's settings are an admin's."}>
        <${RoleBadge} role=${u.role}/><//>
      <${Row} label="Sign out" sub=${u.via === "session" ? "On this browser only." : "This page is let in by an API key: it forgets the key."}>
        <button type="button" class="btn btn--secondary btn--sm" onClick=${out}><${Icon} name="log-out" sm/>Sign out</button><//><//>
    <${Group} cap="Change your password"><form class="set-block" onSubmit=${save}>
      <input type="text" name="username" autocomplete="username" value=${u.name} hidden readonly/>
      <label class="field"><span class="field__label">Current password</span>
        <input class="input acct__pw" type="password" autocomplete="current-password" value=${cur} onInput=${(e) => setCur(e.target.value)}/></label>
      <div class="set-pair acct__pair">
        <label class="field"><span class="field__label">New password</span>
          <input class="input" type="password" autocomplete="new-password" value=${pw} onInput=${(e) => setPw(e.target.value)}/></label>
        <label class="field"><span class="field__label">Once more</span>
          <input class="input" type="password" autocomplete="new-password" value=${again} onInput=${(e) => setAgain(e.target.value)}/></label>
      </div>
      <span class="set-row__sub" role=${bad ? "alert" : undefined}>${bad || PW_NOTE}</span>
      <div><button type="submit" class="btn btn--primary btn--sm" disabled=${busy || !cur || pw.length < 8 || pw !== again}>Change password</button></div>
    </form><//>`;
}

// ------------------------------------------------------------------ API keys
export function KeysPage({ui: {Head, Group}}) {
  const admin = isAdmin(), mine = S.me.user && S.me.user.id;
  const [all, setAll] = useState(false), [keys, setKeys] = useState(null), [err, setErr] = useState("");
  const [name, setName] = useState(""), [made, setMade] = useState(null), [busy, setBusy] = useState(false);
  const load = (every = all) => api(every ? "api/keys?all=1" : "api/keys").then((j) => { setKeys(j.keys); setErr(""); })
    .catch((e) => setErr(e.message));
  useEffect(() => { load(); }, [all]);
  const create = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      const out = await api("api/keys", {method: "POST", body: {name: name.trim()}});
      setMade(out);
      setName("");
      load();
    } catch (e2) { toast("error", "No key made", e2.message, 6000); }
    setBusy(false);
  };
  const revoke = async (k) => {
    const whose = k.user_id === mine ? "" : ` (${k.user}'s)`;
    if (!await confirmDialog(`Delete the key “${k.name}”${whose}?`, "<p>Apps that use it are refused from now on. This cannot be undone.</p>", "Delete the key", true)) return;
    try {
      await api("api/keys/delete", {method: "POST", body: {id: k.id}});
      if (made && made.key.id === k.id) setMade(null);
      load();
    } catch (e) { toast("error", "Not deleted", e.message, 6000); }
  };
  return html`<${Head} title="API keys" intro="Keys for your apps and coding agents. A key works as you do, with your role; the addresses to use it with are on Connect your tools.">
      ${admin ? html`<${Seg} label="Whose keys" value=${all ? "all" : "mine"} onChange=${(v) => { setKeys(null); setAll(v === "all"); }}
        options=${[{id: "mine", label: "Mine"}, {id: "all", label: "Everyone's"}]}/>` : null}<//>
    ${made ? html`<div class="callout callout--ok keymade" role="status">
      <span><b>Copy “${made.key.name}” now.</b> It is shown this once: Maya keeps only a fingerprint of it.</span>
      <code class="keymade__key">${made.secret}</code>
      <span class="keymade__acts"><button type="button" class="btn btn--primary btn--sm" onClick=${() => copyText(made.secret)}><${Icon} name="copy" sm/>Copy</button>
        <button type="button" class="btn btn--ghost btn--sm" onClick=${() => setMade(null)}>Done</button></span></div>` : null}
    <${Group} cap="A new key"><form class="set-block acct__new" onSubmit=${create}>
      <input class="input" placeholder="What it is for, e.g. Laptop or Claude Code" maxlength="64" aria-label="The key's name" value=${name}
        onInput=${(e) => setName(e.target.value)}/>
      <button type="submit" class="btn btn--primary btn--sm" disabled=${busy}><${Icon} name="plus" sm/>Create key</button></form><//>
    ${err ? html`<p class="set-note">${err}</p>` : !keys ? html`<p class="set-note">Reading the keys…</p>`
      : !keys.length ? html`<p class="set-note">${all ? "Nobody has a key yet." : "You have no keys yet."}</p>`
      : html`<${Group} cap=${`${fmt(keys.length)} key${keys.length === 1 ? "" : "s"}`}>${keys.map((k) => html`<div class="acct-row" key=${k.id}>
          <span class="acct-row__text"><b>${k.name}${all ? html` <span class="muted">· ${k.user}</span>` : null}</b>
            <span><code>${k.hint}</code> · made ${when(k.created)} · last used ${when(k.last_used)}</span></span>
          <button type="button" class="btn btn--danger btn--sm" onClick=${() => revoke(k)}>Delete</button></div>`)}<//>`}`;
}

// ------------------------------------------------------------------ Users (an admin's)
function NewUser({first, onMade}) {
  const [name, setName] = useState(""), [pw, setPw] = useState(""), [role, setRole] = useState(first ? "admin" : "user");
  const [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const add = async (e) => {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      const out = await api("api/users", {method: "POST", body: {name: name.trim(), password: pw, role}});
      if (first) {                                 // from here the server asks everyone who they are: sign in as them
        toast("success", "Users are on", `Signing in as ${out.user.name}…`);
        await signIn(out.user.name, pw);
        return;
      }
      toast("success", `${out.user.name} can sign in`, "Give them the name and the password; they can change the password in Settings.", 6000);
      setName(""); setPw(""); setRole("user");
      onMade();
    } catch (e2) { setErr(e2.message); }
    setBusy(false);
  };
  return html`<form class="set-block" onSubmit=${add}>
    <div class="set-pair acct__pair">
      <label class="field"><span class="field__label">Name</span>
        <input class="input" autocomplete="off" autocapitalize="none" spellcheck="false" maxlength="32" placeholder="e.g. anna" value=${name}
          onInput=${(e) => setName(e.target.value)}/></label>
      <label class="field"><span class="field__label">Password</span>
        <input class="input" type="password" autocomplete="new-password" placeholder=${PW_NOTE} value=${pw}
          onInput=${(e) => setPw(e.target.value)}/></label>
    </div>
    <div class="acct__new">
      ${first ? html`<span class="set-row__sub">The first user is an admin.</span>`
        : html`<${Seg} label="Role" value=${role} onChange=${setRole}
            options=${[{id: "user", label: "Standard", title: "The chat, the APIs and their own API keys"},
                       {id: "admin", label: "Admin", title: "Everything: the models, the server's settings, the users"}]}/>`}
      <button type="submit" class="btn btn--primary btn--sm" disabled=${busy || name.trim().length < 2 || pw.length < 8}>
        <${Icon} name="plus" sm/>${first ? "Add the admin and sign in" : "Add user"}</button></div>
    ${err ? html`<div class="msg-error" role="alert">${err}</div>` : null}
  </form>`;
}

function UserRow({u, me, change}) {
  const [open, setOpen] = useState(false);
  const btn = useRef(null);
  const run = (f) => () => { setOpen(false); f(); };
  const self = u.id === me;
  const update = (body, done) => change("api/users/update", {id: u.id, ...body}, done);
  const reset = async () => {
    const pw = await promptDialog({title: `A new password for ${u.name}`, label: "New password", okLabel: "Set password",
                                   note: `${PW_NOTE} ${u.name} is signed out everywhere; their API keys keep working.`});
    if (pw == null) return;
    update({password: pw}, [`${u.name} has a new password`, "Tell them; they can change it in Settings."]);
  };
  const del = async () => {
    if (!await confirmDialog(`Delete ${u.name}?`, `<p>The account and its ${fmt(u.keys)} API key${u.keys === 1 ? "" : "s"} are deleted: apps that use them are refused from now on. This cannot be undone.</p>`, "Delete the user", true)) return;
    change("api/users/delete", {id: u.id}, [`${u.name} is deleted`, ""]);
  };
  return html`<div class=${`acct-row${u.disabled ? " is-off" : ""}`}>
    <span class="acct-row__text"><b>${u.name}${self ? html` <span class="muted">· you</span>` : null} <${RoleBadge} role=${u.role}/>
        ${u.disabled ? html` <span class="badge badge--warn">Disabled</span>` : null}</b>
      <span>${fmt(u.keys)} key${u.keys === 1 ? "" : "s"} · ${fmt(u.requests)} request${u.requests === 1 ? "" : "s"}${u.requests
        ? ` (${fmt(u.prompt_tokens)} tokens read, ${fmt(u.output_tokens)} written)` : ""} · last seen ${when(u.last_seen)}</span></span>
    ${self ? null   /* (your own role and password: Your account; another admin changes the rest) */ : html`
    <button type="button" ref=${btn} class="icon-btn icon-btn--sm" aria-label=${`Options for ${u.name}`} aria-haspopup="menu"
      aria-expanded=${String(open)} onClick=${() => setOpen(!open)}><${Icon} name="ellipsis" sm/></button>
    <${Menu} open=${open} onClose=${() => setOpen(false)} anchor=${btn} label=${`Options for ${u.name}`} width=${250}>
      ${u.role === "admin"
        ? html`<${MenuItem} icon="user" label="Make a standard user" onClick=${run(() => update({role: "user"}, [`${u.name} is a standard user`, ""]))}/>`
        : html`<${MenuItem} icon="users" label="Make an admin" onClick=${run(() => update({role: "admin"}, [`${u.name} is an admin`, ""]))}/>`}
      <${MenuItem} icon="key" label="Set a new password" onClick=${run(reset)}/>
      <${MenuItem} icon=${u.disabled ? "check" : "stop"} label=${u.disabled ? "Enable" : "Disable"}
        onClick=${run(() => update({disabled: !u.disabled}, u.disabled ? [`${u.name} can sign in again`, ""]
          : [`${u.name} is disabled`, "They are signed out, and their API keys are refused until enabled again."]))}/>
      <${MenuSep}/>
      <${MenuItem} icon="trash" label="Delete" danger onClick=${run(del)}/>
    <//>`}
  </div>`;
}

export function UsersPage({ui: {Head, Group}}) {
  const [list, setList] = useState(null), [err, setErr] = useState("");
  const on = S.me.users, me = S.me.user && S.me.user.id;
  const load = () => api("api/users").then((j) => { setList(j.users); setErr(""); }).catch((e) => setErr(e.message));
  useEffect(() => { if (on) load(); }, [on]);
  const change = async (path, body, [title, text]) => {
    try {
      await api(path, {method: "POST", body});
      toast("success", title, text, 5000);
    } catch (e) { toast("error", "Not changed", e.message, 6000); }
    load();
  };
  if (!on) {
    const open = !S.health.api_key;
    return html`<${Head} title="Users" intro="Give everyone who uses this server a sign-in and API keys of their own."/>
      <div class="callout callout--info"><span>This server has no users yet${open ? ": everyone who reaches it can do everything."
        : ": everyone who has its API key can do everything."} Add the first one and that changes at once:
        the page asks everyone to sign in, and the API takes only a user's key${open ? "" : " (or the config's key, which stays an admin's)"}.
        Standard users chat and call the APIs; only admins change the models, the server's settings and the users.</span></div>
      <${Group} cap="The first user"><${NewUser} first onMade=${() => {}}/><//>`;
  }
  const admins = (list || []).filter((u) => u.role === "admin" && !u.disabled).length;
  return html`<${Head} title="Users" intro="Who may use this server. An admin does everything; a standard user chats, calls the APIs and has API keys of their own."/>
    <${Group} cap="Add a user"><${NewUser} onMade=${load}/><//>
    ${err ? html`<p class="set-note">${err}</p>` : !list ? html`<p class="set-note">Reading the users…</p>`
      : html`<${Group} cap=${`${fmt(list.length)} user${list.length === 1 ? "" : "s"} · ${fmt(admins)} admin${admins === 1 ? "" : "s"}`}>
          ${list.map((u) => html`<${UserRow} key=${u.id} u=${u} me=${me} change=${change}/>`)}<//>`}`;
}
