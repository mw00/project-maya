"""serve/users.py - the server's users (opt-in): accounts in a SQLite file beside the run config, each an admin or a
standard user with API keys of their own; the web page signs in with a name and a password.

No user in the file: the server is as before - the config's "api_key" (or none) opens everything.  With one or more,
every request needs a user's API key, a signed-in page's session, or the config's "api_key" (which stays an admin's
key, so the tools set up before keep working).  A standard user may chat, call the model's APIs and manage their own
keys; what changes the server for everyone (models, the context size, the port, updates, shared settings, the users)
needs an admin.  `may()` is the one list of what a standard user may call: a route it does not name is an admin's, so
a route added later is closed to them until it is listed.

No secret is stored: a password as scrypt with its salt, a key or a session as its SHA-256 (each is 240 random bits,
so a fast hash is enough).  A key is shown once, when it is made.

    python -m unittest serve.test_users -v
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

FILE = "maya-users.db"
ROLES = ("admin", "user")
KEY_PREFIX = "sk-maya-"           # an API key (the "sk-" some OpenAI clients expect a key to start with)
SESSION_PREFIX = "ss-maya-"       # a signed-in page's session
SESSION_S = 30 * 86400            # a session ends this long after its last use
MAX_KEYS = 25                     # a user's keys at most
TOUCH_S = 60                      # "last used" / "last seen" are written at most this often
SCRYPT_N = 1 << 14                # scrypt's cost (r=8, p=1: 16 MB and ~50 ms a password)
FAILS_FREE = 5                    # wrong passwords before a name has to wait ...
FAIL_WAIT_S, FAIL_WAIT_MAX_S = 30, 900   # ... this long, doubling with each further one

NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  password TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
  disabled INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  last_seen REAL,
  requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS api_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  hint TEXT NOT NULL,
  created REAL NOT NULL,
  last_used REAL);
CREATE TABLE IF NOT EXISTS sessions (
  hash TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created REAL NOT NULL,
  expires REAL NOT NULL);
"""


class UserError(ValueError):
    """What a caller did wrong, with the HTTP status that says so."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclasses.dataclass(frozen=True)
class Caller:
    """Who sent a request.  `via`: "key" (a user's API key), "session" (a signed-in page), "config" (the config's
    api_key: an admin without an account) or "open" (a server with no key and no users: everyone is the admin)."""
    id: int | None
    name: str
    role: str
    via: str
    key_id: int | None = None

    @property
    def admin(self) -> bool:
        return self.role == "admin"

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "via": self.via}


# ------------------------------------------------------------------------------------------------ what a user may call
USER_GET = frozenset({
    "/auth/me", "/api/keys",
    "/v1/models", "/models", "/models/sse", "/props", "/slots", "/v1/status", "/status", "/v1/strata/stream",
    "/api/tags", "/api/ps", "/api/version",
    # the page: the model's state, its defaults and what the chat shows of them (reading only).  Not /mcp: the MCP
    # tools run on the server's PC with its rights, so the chat offers them to an admin alone (server.py's _openai)
    "/metrics", "/settings", "/api/context", "/api/thinking-budget", "/api/turbo", "/api/savings", "/api/update"})
USER_POST = frozenset({
    "/auth/logout", "/auth/password", "/api/keys", "/api/keys/delete",
    "/v1/chat/completions", "/v1/responses", "/v1/messages", "/api/chat", "/api/generate", "/api/show",
    "/api/tokens", "/tokenize", "/detokenize", "/attach", "/pdf", "/v1/strata/cancel"})
USER_PREFIXES = ("/v1/responses/",)     # a stored response: its owner's alone (the server checks the owner)


def may(role: str, method: str, path: str) -> bool:
    """Whether `role` may call `method path` (the path without its query or a trailing slash).  An admin: anything.
    A standard user: the lists above, and nothing else."""
    if role == "admin":
        return True
    if path.startswith(USER_PREFIXES):
        return method in ("GET", "POST", "DELETE")
    return path in (USER_GET if method == "GET" else USER_POST if method == "POST" else ())


# ------------------------------------------------------------------------------------------------ secrets
def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=8, p=1, dklen=32)
    return f"scrypt${SCRYPT_N}${salt.hex()}${h.hex()}"


def check_password(password, stored: str) -> bool:
    try:
        kind, n, salt, want = stored.split("$")
        got = hashlib.scrypt(str(password).encode(), salt=bytes.fromhex(salt), n=int(n), r=8, p=1, dklen=32)
    except (ValueError, TypeError):
        return False
    return kind == "scrypt" and hmac.compare_digest(got.hex(), want)


def clean_name(name) -> str:
    n = str(name or "").strip().lower()
    if not NAME_RE.fullmatch(n):
        raise UserError("a user name is 2 to 32 characters: letters, digits, '.', '_' or '-', starting with a letter "
                        "or a digit")
    return n


def clean_password(password) -> str:
    if not isinstance(password, str) or not 8 <= len(password) <= 256:
        raise UserError("a password is 8 to 256 characters")
    return password


def clean_role(role) -> str:
    if role not in ROLES:
        raise UserError('a role is "admin" or "user"')
    return role


# ------------------------------------------------------------------------------------------------ the store
class UserStore:
    """The users, their keys and sessions in one SQLite file.  One connection behind a lock: the writes are rare (a
    sign-in, a new key, a request's count) and the reads are single indexed rows."""

    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.RLock()
        new = not os.path.exists(self.path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        if new:
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)          # (hashes only, but still nobody else's to read)
        self._count = self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self._touched: dict = {}                    # ("key" | "user", id) -> when its "last used" was written
        self._fails: dict = {}                      # a name -> [wrong passwords in a row, may try again at]
        self._dummy = None                          # a hash to check against when the name is unknown (same time)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @contextlib.contextmanager
    def _tx(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
            self._count = self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def active(self) -> bool:
        """One user or more: the server asks everyone who they are."""
        return self._count > 0

    # -------------------------------------------------------------------------------------------- users
    @staticmethod
    def _user(r, keys=None) -> dict:
        d = {k: r[k] for k in ("id", "name", "role", "created", "last_seen", "requests", "prompt_tokens",
                               "output_tokens")}
        d["disabled"] = bool(r["disabled"])
        if keys is not None:
            d["keys"] = keys
        return d

    def users(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT u.*, (SELECT COUNT(*) FROM api_keys k WHERE k.user_id = u.id) AS n_keys "
                                   "FROM users u ORDER BY u.name").fetchall()
        return [self._user(r, r["n_keys"]) for r in rows]

    def user(self, user_id) -> dict:
        with self.lock:
            r = self.db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if r is None:
            raise UserError("no such user", 404)
        return self._user(r)

    def create_user(self, name, password, role="user") -> dict:
        name, password, role = clean_name(name), clean_password(password), clean_role(role)
        with self._tx() as db:
            if not self._count and role != "admin":
                raise UserError("the first user must be an admin")
            if db.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise UserError(f"a user named {name} exists already", 409)
            cur = db.execute("INSERT INTO users (name, password, role, created) VALUES (?, ?, ?, ?)",
                             (name, hash_password(password), role, time.time()))
            uid = cur.lastrowid
        return self.user(uid)

    def _other_admins(self, db, user_id) -> int:
        return db.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND disabled = 0 AND id != ?",
                          (user_id,)).fetchone()[0]

    def update_user(self, user_id, role=None, disabled=None, password=None) -> dict:
        """A user's role, whether they are disabled, or a new password an admin gives them.  The last admin who can
        sign in stays one.  A disabled user, or one with a new password, is signed out everywhere."""
        if role is not None:
            clean_role(role)
        if password is not None:
            clean_password(password)
        if disabled is not None and not isinstance(disabled, bool):
            raise UserError('"disabled" is true or false')
        with self._tx() as db:
            r = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if r is None:
                raise UserError("no such user", 404)
            loses = (role == "user" and r["role"] == "admin") or (disabled is True and not r["disabled"])
            if loses and r["role"] == "admin" and not r["disabled"] and not self._other_admins(db, user_id):
                raise UserError("this is the only admin: make another user an admin first", 409)
            if role is not None:
                db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            if disabled is not None:
                db.execute("UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id))
            if password is not None:
                db.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(password), user_id))
            if disabled or password is not None:
                db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return self.user(user_id)

    def delete_user(self, user_id) -> None:
        """The user, with their keys and sessions.  Not the last admin who can sign in."""
        with self._tx() as db:
            r = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if r is None:
                raise UserError("no such user", 404)
            if r["role"] == "admin" and not r["disabled"] and not self._other_admins(db, user_id):
                raise UserError("this is the only admin: make another user an admin first", 409)
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def change_password(self, user_id, current, new, keep: str = "") -> None:
        """The user's own: the current one proves it is them; every other session of theirs ends (`keep`: the one
        that asked, by its token)."""
        clean_password(new)
        with self._tx() as db:
            r = db.execute("SELECT password FROM users WHERE id = ?", (user_id,)).fetchone()
            if r is None or not check_password(current, r["password"]):
                raise UserError("the current password is wrong", 403)
            db.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new), user_id))
            db.execute("DELETE FROM sessions WHERE user_id = ? AND hash != ?", (user_id, digest(keep) if keep else ""))

    def note_usage(self, user_id, prompt_tokens: int, output_tokens: int) -> None:
        """A finished request of this user's: their count (the Users page)."""
        with self.lock:
            self.db.execute("UPDATE users SET requests = requests + 1, prompt_tokens = prompt_tokens + ?, "
                            "output_tokens = output_tokens + ?, last_seen = ? WHERE id = ?",
                            (int(prompt_tokens), int(output_tokens), time.time(), user_id))

    # -------------------------------------------------------------------------------------------- signing in
    def _wait(self, name: str, now: float) -> int:
        f = self._fails.get(name)
        return max(0, int(f[1] - now + 0.999)) if f else 0

    def _failed(self, name: str, now: float) -> None:
        if len(self._fails) > 4096:                 # (names nobody has: not kept for ever)
            self._fails = {k: v for k, v in self._fails.items() if v[1] > now}
        n = self._fails.get(name, [0, 0.0])[0] + 1
        wait = min(FAIL_WAIT_MAX_S, FAIL_WAIT_S * 2 ** (n - FAILS_FREE)) if n >= FAILS_FREE else 0
        self._fails[name] = [n, now + wait]

    def login(self, name, password) -> tuple[str, dict]:
        """(a new session's token, the user) for the right name and password; a UserError says only that one of the
        two is wrong, and a name with many wrong passwords in a row has to wait."""
        name, now = str(name or "").strip().lower()[:64], time.time()
        with self.lock:
            wait = self._wait(name, now)
            if wait:
                raise UserError(f"too many wrong passwords: try again in {wait} s", 429)
            r = self.db.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
            if self._dummy is None:
                self._dummy = hash_password(secrets.token_urlsafe(8))
        good = check_password(password, r["password"] if r is not None else self._dummy)   # (not under the lock: 50 ms)
        with self.lock:
            if r is None or not good or r["disabled"]:
                self._failed(name, now)
                raise UserError("wrong name or password", 401)
            self._fails.pop(name, None)
            token = SESSION_PREFIX + secrets.token_urlsafe(30)
            self.db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            self.db.execute("INSERT INTO sessions (hash, user_id, created, expires) VALUES (?, ?, ?, ?)",
                            (digest(token), r["id"], now, now + SESSION_S))
            self.db.execute("UPDATE users SET last_seen = ? WHERE id = ?", (now, r["id"]))
        return token, self._user(r)

    def logout(self, token: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM sessions WHERE hash = ?", (digest(token or ""),))

    def resolve(self, token: str) -> Caller | None:
        """The user a key or a session belongs to, or None: unknown, ended, or its user disabled."""
        if not token or not self._count:
            return None
        h, now = digest(token), time.time()
        with self.lock:
            if token.startswith(KEY_PREFIX):
                r = self.db.execute("SELECT k.id AS kid, u.id, u.name, u.role, u.disabled FROM api_keys k "
                                    "JOIN users u ON u.id = k.user_id WHERE k.hash = ?", (h,)).fetchone()
                if r is None or r["disabled"]:
                    return None
                if now - self._touched.get(("key", r["kid"]), 0) > TOUCH_S:
                    self._touched[("key", r["kid"])] = now
                    self.db.execute("UPDATE api_keys SET last_used = ? WHERE id = ?", (now, r["kid"]))
                who = Caller(r["id"], r["name"], r["role"], "key", r["kid"])
            elif token.startswith(SESSION_PREFIX):
                r = self.db.execute("SELECT s.expires, u.id, u.name, u.role, u.disabled FROM sessions s "
                                    "JOIN users u ON u.id = s.user_id WHERE s.hash = ? AND s.expires > ?",
                                    (h, now)).fetchone()
                if r is None or r["disabled"]:
                    return None
                if r["expires"] - now < SESSION_S / 2:      # used: it lasts SESSION_S from here again
                    self.db.execute("UPDATE sessions SET expires = ? WHERE hash = ?", (now + SESSION_S, h))
                who = Caller(r["id"], r["name"], r["role"], "session")
            else:
                return None
            if now - self._touched.get(("user", who.id), 0) > TOUCH_S:
                self._touched[("user", who.id)] = now
                self.db.execute("UPDATE users SET last_seen = ? WHERE id = ?", (now, who.id))
        return who

    # -------------------------------------------------------------------------------------------- API keys
    @staticmethod
    def _key(r) -> dict:
        return {k: r[k] for k in ("id", "name", "hint", "created", "last_used", "user_id", "user")}

    def keys(self, user_id=None) -> list[dict]:
        """A user's keys, or everyone's (None), the newest first; never the keys themselves."""
        q = "SELECT k.*, u.name AS user FROM api_keys k JOIN users u ON u.id = k.user_id"
        with self.lock:
            rows = self.db.execute(q + (" WHERE k.user_id = ?" if user_id is not None else "") +
                                   " ORDER BY k.created DESC, k.id DESC",
                                   (user_id,) if user_id is not None else ()).fetchall()
        return [self._key(r) for r in rows]

    def create_key(self, user_id, name="") -> tuple[dict, str]:
        """(the key's record, the key): the only time the key itself is known."""
        name = str(name or "").strip()[:64] or "API key"
        secret = KEY_PREFIX + secrets.token_urlsafe(30)
        with self._tx() as db:
            if db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
                raise UserError("no such user", 404)
            if db.execute("SELECT COUNT(*) FROM api_keys WHERE user_id = ?", (user_id,)).fetchone()[0] >= MAX_KEYS:
                raise UserError(f"a user has {MAX_KEYS} keys at most: delete one first", 409)
            cur = db.execute("INSERT INTO api_keys (user_id, name, hash, hint, created) VALUES (?, ?, ?, ?, ?)",
                             (user_id, name, digest(secret), f"{secret[:12]}…{secret[-4:]}", time.time()))
            kid = cur.lastrowid
        return next(k for k in self.keys(user_id) if k["id"] == kid), secret

    def delete_key(self, key_id, user_id=None) -> None:
        """The key (it stops working at once).  `user_id`: only when it is that user's."""
        with self._tx() as db:
            cur = db.execute("DELETE FROM api_keys WHERE id = ?" + (" AND user_id = ?" if user_id is not None else ""),
                             (key_id,) + ((user_id,) if user_id is not None else ()))
            if not cur.rowcount:
                raise UserError("no such key", 404)


def store_path(config=None, env=None) -> Path | None:
    """Where the users are kept: beside the run config, or MAYA_USERS_DB (a file; "off", "none" or "0": no users on
    this server).  None: nowhere (no config and no MAYA_USERS_DB)."""
    v = ((os.environ if env is None else env).get("MAYA_USERS_DB") or "").strip()
    if v.lower() in ("off", "none", "0"):
        return None
    if v:
        return Path(v).expanduser()
    return Path(config).resolve().parent / FILE if config else None


def identify(store: UserStore | None, config_keys: list, given: str) -> Caller | None:
    """Who a request's key (Authorization: Bearer, or x-api-key) says it is; None: nobody this server lets in."""
    if given and any(hmac.compare_digest(given.encode(), k.encode()) for k in config_keys or []):
        return Caller(None, "", "admin", "config")
    if store is not None and store.active():
        return store.resolve(given)
    return None if config_keys else Caller(None, "", "admin", "open")


# ------------------------------------------------------------------------------------------------ the routes
# The server's handler passes itself (`h`): its _json, _body and _own_page are the ones every dashboard route uses,
# h.caller is who _authorized() found.

def _local(h) -> bool:
    ip = h.client_address[0] if h.client_address else ""
    return ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def _token(h) -> str:
    auth = h.headers.get("Authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else h.headers.get("x-api-key", "")


def _need_store(svc) -> UserStore:
    store = getattr(svc, "users", None)
    if store is None:
        raise UserError("this server keeps no users: it runs without a run config (MAYA_USERS_DB names a file to keep "
                        "them in)")
    return store


def _json_body(h, what: str):
    """The request's JSON object, or None when it was refused (and answered): a form or another site's page cannot
    send application/json, as for the other dashboard routes."""
    body = h._body()
    if not h._own_page(what):
        return None
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise UserError("the request body is not JSON") from None
    if not isinstance(req, dict):
        raise UserError("the request body must be a JSON object")
    return req


def handle_get(h, svc, path: str) -> None:
    """GET /auth/me (anyone: who this page is, and whether the server has users), /api/keys (the caller's keys;
    ?all=1: everyone's, an admin's), /api/users (an admin's)."""
    store = getattr(svc, "users", None)
    if path == "/auth/me":
        who = h._identify()
        h._json(200, {"users": bool(store is not None and store.active()), "available": store is not None,
                      "user": who.public() if who else None})
        return
    if not h._authorized():
        return
    try:
        store = _need_store(svc)
        if path == "/api/users":
            h._json(200, {"users": store.users()})
        elif "all=1" in h.path.partition("?")[2]:
            if not h.caller.admin:
                raise UserError("every user's keys are an admin's to see", 403)
            h._json(200, {"keys": store.keys()})
        else:
            h._json(200, {"keys": store.keys(h.caller.id) if h.caller.id is not None else []})
    except UserError as e:
        h._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})


def _id(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UserError('"id" is the number of a user or a key')
    return value


def handle_post(h, svc, path: str) -> None:
    """POST /auth/login (anyone), /auth/logout, /auth/password, /api/keys, /api/keys/delete (the caller's own),
    /api/users, /api/users/update, /api/users/delete (an admin's; _authorized() has checked the role)."""
    try:
        req = _json_body(h, "accounts can be changed")
        if req is None:
            return
        store = _need_store(svc)
        who = getattr(h, "caller", None)
        if path == "/auth/login":
            if not store.active():
                raise UserError("this server has no users to sign in as")
            token, user = store.login(req.get("name"), req.get("password"))
            print(f"[maya] {user['name']} signed in", flush=True)
            h._json(200, {"token": token, "user": {**{k: user[k] for k in ("id", "name", "role")}, "via": "session"}})
        elif path == "/auth/logout":
            store.logout(_token(h))
            h._json(200, {"ok": True})
        elif path == "/auth/password":
            if who.id is None:
                raise UserError("the config's API key has no password: sign in as a user")
            store.change_password(who.id, req.get("current"), req.get("new"), keep=_token(h))
            h._json(200, {"ok": True})
        elif path == "/api/keys":
            # an admin may make one for another user ("user_id"); everyone else, and by default, their own
            uid = _id(req["user_id"]) if who.admin and req.get("user_id") is not None else who.id
            if uid is None:
                raise UserError("the config's API key is not a user: sign in as a user to make keys of your own")
            key, secret = store.create_key(uid, req.get("name"))
            print(f"[maya] a new API key for {key['user']} ({key['name']})", flush=True)
            h._json(200, {"key": key, "secret": secret})
        elif path == "/api/keys/delete":
            if not who.admin and who.id is None:
                raise UserError("no such key", 404)
            store.delete_key(_id(req.get("id")), None if who.admin else who.id)
            h._json(200, {"ok": True})
        elif path == "/api/users":
            if not store.active() and who.via == "open" and not _local(h):
                # nobody has a key yet: only this PC may make the first user, as for downloading models
                raise UserError("set an API key in the config, or open this page on the PC Maya runs on, to add the "
                                "first user", 403)
            user = store.create_user(req.get("name"), req.get("password"), req.get("role") or "user")
            print(f"[maya] new user: {user['name']} ({user['role']})", flush=True)
            h._json(200, {"user": user})
        elif path == "/api/users/update":
            if who.id is not None and req.get("id") == who.id and (req.get("disabled") is True or
                                                                   req.get("role") == "user"):
                raise UserError("ask another admin: you cannot disable yourself or give up your own role", 409)
            h._json(200, {"user": store.update_user(_id(req.get("id")), role=req.get("role"),
                                                    disabled=req.get("disabled"), password=req.get("password"))})
        elif path == "/api/users/delete":
            if who.id is not None and req.get("id") == who.id:
                raise UserError("ask another admin: you cannot delete yourself", 409)
            store.delete_user(_id(req.get("id")))
            h._json(200, {"ok": True})
        else:
            h._json(404, {"error": {"message": "not found"}})
    except Exception as e:  # noqa: BLE001 - a UserError, or the server's BadBody (a body that cannot be read)
        if not isinstance(getattr(e, "status", None), int):
            raise
        h._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})
