"""serve/test_users.py - the server's users (serve/users.py): the store (passwords, keys, sessions, the last admin),
what a standard user may call, and the routes on a running server.  Against the mock engine (no GPU, no pack).

    python -m unittest serve.test_users -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import users  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PW = "correct horse"


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "maya-users.db"
        self.s = users.UserStore(self.path)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_the_first_user_is_an_admin(self):
        self.assertFalse(self.s.active())
        with self.assertRaises(users.UserError):
            self.s.create_user("bob", PW, "user")
        a = self.s.create_user("Alice", PW, "admin")           # (names are lower case)
        self.assertEqual((a["name"], a["role"], a["disabled"]), ("alice", "admin", False))
        self.assertTrue(self.s.active())
        self.assertEqual(self.s.create_user("bob", PW)["role"], "user")

    def test_names_and_passwords(self):
        for bad in ("", "a", "-ab", "a b", "x" * 33, "né", None):
            with self.assertRaises(users.UserError, msg=repr(bad)):
                self.s.create_user(bad, PW, "admin")
        for bad in ("short", "", None, 12345678, "x" * 257):
            with self.assertRaises(users.UserError, msg=repr(bad)):
                self.s.create_user("alice", bad, "admin")
        with self.assertRaises(users.UserError):
            self.s.create_user("alice", PW, "root")
        self.s.create_user("alice", PW, "admin")
        with self.assertRaises(users.UserError) as e:
            self.s.create_user("ALICE", PW, "admin")
        self.assertEqual(e.exception.status, 409)

    def test_no_secret_is_stored(self):
        a = self.s.create_user("alice", PW, "admin")
        _, key = self.s.create_key(a["id"], "laptop")
        token, _ = self.s.login("alice", PW)
        self.s.close()
        raw = self.path.read_bytes()
        for secret in (PW, key, token, key[len(users.KEY_PREFIX):], token[len(users.SESSION_PREFIX):]):
            self.assertNotIn(secret.encode(), raw)
        self.s = users.UserStore(self.path)                    # ... and it is all there after a restart
        self.assertEqual(self.s.resolve(key).name, "alice")
        self.assertEqual(self.s.resolve(token).via, "session")

    def test_login(self):
        a = self.s.create_user("alice", PW, "admin")
        token, u = self.s.login(" Alice ", PW)
        self.assertEqual(u["id"], a["id"])
        who = self.s.resolve(token)
        self.assertEqual((who.id, who.name, who.role, who.via), (a["id"], "alice", "admin", "session"))
        for name, pw in (("alice", "wrong password"), ("nobody", PW), ("alice", None)):
            with self.assertRaises(users.UserError) as e:
                self.s.login(name, pw)
            self.assertEqual((e.exception.status, str(e.exception)), (401, "wrong name or password"))
        self.s.logout(token)
        self.assertIsNone(self.s.resolve(token))

    def test_wrong_passwords_have_to_wait(self):
        self.s.create_user("alice", PW, "admin")
        for _ in range(users.FAILS_FREE):
            with self.assertRaises(users.UserError) as e:
                self.s.login("alice", "wrong password")
            self.assertEqual(e.exception.status, 401)
        with self.assertRaises(users.UserError) as e:           # (the right one too: it has to wait)
            self.s.login("alice", PW)
        self.assertEqual(e.exception.status, 429)
        self.s._fails["alice"][1] = 0                           # the wait is over
        self.s.login("alice", PW)
        self.assertNotIn("alice", self.s._fails)

    def test_a_session_ends(self):
        self.s.create_user("alice", PW, "admin")
        token, _ = self.s.login("alice", PW)
        self.s.db.execute("UPDATE sessions SET expires = 1")
        self.assertIsNone(self.s.resolve(token))

    def test_keys(self):
        a = self.s.create_user("alice", PW, "admin")
        b = self.s.create_user("bob", PW)
        rec, key = self.s.create_key(b["id"], "  laptop  ")
        self.assertTrue(key.startswith(users.KEY_PREFIX))
        self.assertEqual((rec["name"], rec["user"], rec["last_used"]), ("laptop", "bob", None))
        self.assertNotIn(key[14:-4], rec["hint"])
        who = self.s.resolve(key)
        self.assertEqual((who.name, who.role, who.via, who.key_id), ("bob", "user", "key", rec["id"]))
        self.assertIsNotNone(self.s.keys(b["id"])[0]["last_used"])
        self.assertEqual(self.s.keys(a["id"]), [])
        self.assertEqual(len(self.s.keys()), 1)
        for wrong in ("", "sk-maya-nope", key + "x", key.replace("sk-", "ss-"), "anything"):
            self.assertIsNone(self.s.resolve(wrong))
        with self.assertRaises(users.UserError):                # another user's key is not theirs to delete
            self.s.delete_key(rec["id"], a["id"])
        self.s.delete_key(rec["id"], b["id"])
        self.assertIsNone(self.s.resolve(key))

    def test_a_user_has_so_many_keys(self):
        a = self.s.create_user("alice", PW, "admin")
        for _ in range(users.MAX_KEYS):
            self.s.create_key(a["id"])
        with self.assertRaises(users.UserError) as e:
            self.s.create_key(a["id"])
        self.assertEqual(e.exception.status, 409)

    def test_a_disabled_user_is_out(self):
        self.s.create_user("alice", PW, "admin")
        b = self.s.create_user("bob", PW)
        _, key = self.s.create_key(b["id"])
        token, _ = self.s.login("bob", PW)
        self.s.update_user(b["id"], disabled=True)
        self.assertIsNone(self.s.resolve(key))
        self.assertIsNone(self.s.resolve(token))
        with self.assertRaises(users.UserError):
            self.s.login("bob", PW)
        self.s.update_user(b["id"], disabled=False)
        self.assertEqual(self.s.resolve(key).name, "bob")       # the key works again; the session is gone
        self.assertIsNone(self.s.resolve(token))

    def test_the_last_admin_stays(self):
        a = self.s.create_user("alice", PW, "admin")
        b = self.s.create_user("bob", PW)
        for change in ({"role": "user"}, {"disabled": True}):
            with self.assertRaises(users.UserError) as e:
                self.s.update_user(a["id"], **change)
            self.assertEqual(e.exception.status, 409)
        with self.assertRaises(users.UserError):
            self.s.delete_user(a["id"])
        self.s.update_user(b["id"], role="admin")
        self.s.update_user(b["id"], disabled=True)              # (a disabled admin does not count)
        with self.assertRaises(users.UserError):
            self.s.delete_user(a["id"])
        self.s.update_user(b["id"], disabled=False)
        self.s.delete_user(a["id"])
        self.assertEqual([u["name"] for u in self.s.users()], ["bob"])

    def test_deleting_a_user_deletes_their_keys(self):
        self.s.create_user("alice", PW, "admin")
        b = self.s.create_user("bob", PW)
        _, key = self.s.create_key(b["id"])
        token, _ = self.s.login("bob", PW)
        self.s.delete_user(b["id"])
        self.assertIsNone(self.s.resolve(key))
        self.assertIsNone(self.s.resolve(token))
        self.assertEqual(self.s.keys(), [])
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)

    def test_passwords_change(self):
        a = self.s.create_user("alice", PW, "admin")
        here, _ = self.s.login("alice", PW)
        there, _ = self.s.login("alice", PW)
        with self.assertRaises(users.UserError) as e:
            self.s.change_password(a["id"], "not the password", "another password")
        self.assertEqual(e.exception.status, 403)
        self.s.change_password(a["id"], PW, "another password", keep=here)
        self.assertIsNotNone(self.s.resolve(here))              # this page stays signed in, the others do not
        self.assertIsNone(self.s.resolve(there))
        with self.assertRaises(users.UserError):
            self.s.login("alice", PW)
        self.s.login("alice", "another password")
        self.s.update_user(a["id"], password="a third password")   # an admin's reset: signed out everywhere
        self.assertIsNone(self.s.resolve(here))

    def test_usage(self):
        a = self.s.create_user("alice", PW, "admin")
        self.s.note_usage(a["id"], 100, 20)
        self.s.note_usage(a["id"], 5, 1)
        u = self.s.user(a["id"])
        self.assertEqual((u["requests"], u["prompt_tokens"], u["output_tokens"]), (2, 105, 21))


class May(unittest.TestCase):
    def test_an_admin_may_anything(self):
        for method, path in (("POST", "/api/models/delete"), ("GET", "/api/users"), ("DELETE", "/api/delete"),
                             ("POST", "/a/route/added/later")):
            self.assertTrue(users.may("admin", method, path))

    def test_a_user_may_chat(self):
        for method, path in (("POST", "/v1/chat/completions"), ("POST", "/v1/messages"), ("POST", "/v1/responses"),
                             ("GET", "/v1/responses/resp_1"), ("DELETE", "/v1/responses/resp_1"),
                             ("POST", "/v1/responses/resp_1/cancel"), ("GET", "/v1/models"), ("POST", "/api/chat"),
                             ("POST", "/attach"), ("GET", "/metrics"), ("POST", "/api/keys"), ("GET", "/api/keys"),
                             ("POST", "/auth/password")):
            self.assertTrue(users.may("user", method, path), f"{method} {path}")

    def test_a_user_may_not_change_the_server(self):
        for method, path in (("POST", "/api/context"), ("POST", "/api/port"), ("POST", "/api/update"),
                             ("POST", "/api/turbo"), ("POST", "/api/thinking-budget"), ("POST", "/settings"),
                             ("POST", "/api/models/switch"), ("POST", "/api/models/delete"), ("POST", "/api/pull"),
                             ("POST", "/api/delete"), ("DELETE", "/api/delete"), ("POST", "/models/unload"),
                             ("POST", "/api/savings/reset"), ("GET", "/api/models"), ("GET", "/api/report"),
                             ("GET", "/api/port"), ("GET", "/metrics/prometheus"), ("GET", "/api/users"),
                             ("GET", "/mcp"),
                             ("POST", "/api/users"), ("POST", "/api/users/update"), ("POST", "/api/users/delete"),
                             ("POST", "/a/route/added/later"), ("GET", "/a/route/added/later"),
                             ("PUT", "/v1/chat/completions")):
            self.assertFalse(users.may("user", method, path), f"{method} {path}")


class Identify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = users.UserStore(Path(self.tmp.name) / "u.db")

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_no_users(self):
        """As before there were users: no key - everyone; a key - who has it."""
        for store in (None, self.s):
            self.assertEqual(users.identify(store, [], "").via, "open")
            self.assertEqual(users.identify(store, [], "whatever").role, "admin")
            self.assertIsNone(users.identify(store, ["k1", "k2"], ""))
            self.assertIsNone(users.identify(store, ["k1", "k2"], "k3"))
            self.assertEqual(users.identify(store, ["k1", "k2"], "k2").via, "config")

    def test_users(self):
        a = self.s.create_user("alice", PW, "admin")
        _, key = self.s.create_key(a["id"])
        self.assertIsNone(users.identify(self.s, [], ""))       # no longer open
        self.assertIsNone(users.identify(self.s, [], "wrong"))
        self.assertEqual(users.identify(self.s, [], key).name, "alice")
        who = users.identify(self.s, ["cfg"], "cfg")            # the config's key: still an admin's
        self.assertEqual((who.role, who.via, who.id), ("admin", "config", None))

    def test_where_the_users_are_kept(self):
        self.assertIsNone(users.store_path(None, {}))
        self.assertEqual(users.store_path("/x/maya-m.json", {}).name, users.FILE)
        self.assertEqual(users.store_path("/x/maya-m.json", {}).parent, Path("/x").resolve())
        self.assertIsNone(users.store_path("/x/maya-m.json", {"MAYA_USERS_DB": "off"}))
        self.assertEqual(users.store_path(None, {"MAYA_USERS_DB": "/y/u.db"}), Path("/y/u.db"))


class Routes(unittest.TestCase):
    CONFIG_KEY = ""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.svc.users = users.UserStore(Path(self.tmp.name) / "maya-users.db")
        self.svc.api_key = self.CONFIG_KEY
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.users.close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, key=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        if key:
            h["Authorization"] = "Bearer " + key
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                raw = resp.read()
                try:
                    return resp.status, json.loads(raw or b"null")
                except ValueError:                              # (the report, Prometheus' text)
                    return resp.status, raw
        except urllib.error.HTTPError as e:
            with e:
                raw = e.read()
                try:
                    return e.code, json.loads(raw)
                except ValueError:
                    return e.code, raw

    def people(self):
        """An admin and a standard user, each signed in and with a key: (admin session, user session, user key, ids)."""
        s, j = self.req("POST", "/api/users", {"name": "alice", "password": PW, "role": "admin"}, key=self.CONFIG_KEY)
        self.assertEqual(s, 200, j)
        s, j = self.req("POST", "/auth/login", {"name": "alice", "password": PW})
        self.assertEqual(s, 200, j)
        admin = j["token"]
        s, j = self.req("POST", "/api/users", {"name": "bob", "password": PW}, key=admin)
        self.assertEqual((s, j["user"]["role"]), (200, "user"))
        bob_id = j["user"]["id"]
        s, j = self.req("POST", "/auth/login", {"name": "bob", "password": PW})
        user = j["token"]
        s, j = self.req("POST", "/api/keys", {"name": "laptop"}, key=user)
        self.assertEqual(s, 200, j)
        return admin, user, j["secret"], bob_id

    def chat(self, key=None):
        return self.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                         "max_tokens": 8}, key=key)


class OpenServer(Routes):
    """No API key in the config: open until its first user is made."""

    def test_open_until_the_first_user(self):
        s, j = self.req("GET", "/auth/me")
        self.assertEqual((s, j["users"], j["available"], j["user"]["via"]), (200, False, True, "open"))
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.req("GET", "/metrics")[0], 200)
        s, j = self.req("POST", "/auth/login", {"name": "alice", "password": PW})
        self.assertEqual(s, 400)                                # nobody to sign in as
        s, j = self.req("POST", "/api/users", {"name": "bob", "password": PW, "role": "user"})
        self.assertEqual(s, 400)                                # the first one is an admin
        s, j = self.req("POST", "/api/users", {"name": "alice", "password": PW, "role": "admin"})
        self.assertEqual((s, j["user"]["name"]), (200, "alice"))
        self.assertNotIn("password", j["user"])
        # ... and from here everyone is asked who they are
        s, j = self.req("GET", "/auth/me")
        self.assertEqual((j["users"], j["user"]), (True, None))
        s, j = self.chat()
        self.assertEqual((s, j["error"]["type"]), (401, "authentication_error"))
        for path in ("/metrics", "/v1/models", "/api/users", "/api/keys", "/settings", "/status"):
            self.assertEqual(self.req("GET", path)[0], 401, path)
        self.assertEqual(self.req("POST", "/api/users", {"name": "eve", "password": PW, "role": "admin"})[0], 401)
        self.assertEqual(self.req("GET", "/health")[0], 200)    # (the page still loads: it has to show the sign-in)

    def test_roles(self):
        admin, user, key, bob_id = self.people()
        for who in (user, key):                                 # a session or a key: the same user
            s, j = self.req("GET", "/auth/me", key=who)
            self.assertEqual((j["user"]["name"], j["user"]["role"]), ("bob", "user"))
            self.assertEqual(self.chat(who)[0], 200)
            self.assertEqual(self.req("GET", "/v1/models", key=who)[0], 200)
            for method, path, body in (("GET", "/api/users", None), ("GET", "/api/models", None),
                                       ("GET", "/api/report", None), ("GET", "/api/port", None),
                                       ("GET", "/metrics/prometheus", None),
                                       ("POST", "/api/context", {"max_context": 16384}),
                                       ("POST", "/settings", {"defaults": {"temperature": 2}}),
                                       ("POST", "/api/turbo", {"on": True}), ("POST", "/api/update", {}),
                                       ("POST", "/api/savings/reset", {}), ("POST", "/models/unload", {}),
                                       ("POST", "/api/users", {"name": "eve", "password": PW, "role": "admin"}),
                                       ("POST", "/api/users/update", {"id": bob_id, "role": "admin"}),
                                       ("POST", "/api/users/delete", {"id": 1}),
                                       ("DELETE", "/api/delete", None)):
                s, j = self.req(method, path, body, key=who)
                self.assertEqual((s, j["error"]["type"]), (403, "permission_error"), f"{method} {path}")
        self.assertEqual(self.svc.engine.max_context, 32768)    # (nothing of that happened)
        self.assertEqual([u["role"] for u in self.svc.users.users()], ["admin", "user"])
        # the admin may
        s, j = self.req("GET", "/api/users", key=admin)
        self.assertEqual([(u["name"], u["role"], u["keys"]) for u in j["users"]],
                         [("alice", "admin", 0), ("bob", "user", 1)])
        self.assertNotIn("password", json.dumps(j))
        for path in ("/api/models", "/api/port", "/api/report", "/metrics/prometheus"):
            self.assertNotIn(self.req("GET", path, key=admin)[0], (401, 403), path)

    def test_keys_are_their_owners(self):
        admin, user, key, bob_id = self.people()
        s, j = self.req("GET", "/api/keys", key=user)
        self.assertEqual([(k["name"], k["user"]) for k in j["keys"]], [("laptop", "bob")])
        self.assertNotIn(key, json.dumps(j))                    # (the key itself: only when it was made)
        kid = j["keys"][0]["id"]
        self.assertEqual(self.req("GET", "/api/keys", key=admin)[1]["keys"], [])
        self.assertEqual(self.req("GET", "/api/keys?all=1", key=user)[0], 403)
        self.assertEqual(len(self.req("GET", "/api/keys?all=1", key=admin)[1]["keys"]), 1)
        # a user's key is made for themselves, whatever they ask
        s, j = self.req("POST", "/api/keys", {"name": "x", "user_id": 1}, key=user)
        self.assertEqual(j["key"]["user"], "bob")
        s, j = self.req("POST", "/api/keys", {"name": "mine"}, key=admin)
        mine = j["key"]["id"]
        self.assertEqual(self.req("POST", "/api/keys/delete", {"id": mine}, key=user)[0], 404)
        self.assertEqual(self.req("POST", "/api/keys/delete", {"id": "1"}, key=user)[0], 400)
        self.assertEqual(self.req("POST", "/api/keys/delete", {"id": kid}, key=admin)[0], 200)   # an admin: anyone's
        self.assertEqual(self.chat(key)[0], 401)

    def test_admins_manage_users(self):
        admin, user, key, bob_id = self.people()
        s, j = self.req("POST", "/api/users/update", {"id": bob_id, "disabled": True}, key=admin)
        self.assertEqual((s, j["user"]["disabled"]), (200, True))
        self.assertEqual(self.chat(key)[0], 401)
        self.assertEqual(self.chat(user)[0], 401)
        self.assertEqual(self.req("POST", "/auth/login", {"name": "bob", "password": PW})[0], 401)
        self.req("POST", "/api/users/update", {"id": bob_id, "disabled": False, "password": "a new password"},
                 key=admin)
        self.assertEqual(self.req("POST", "/auth/login", {"name": "bob", "password": PW})[0], 401)
        self.assertEqual(self.req("POST", "/auth/login", {"name": "bob", "password": "a new password"})[0], 200)
        # not themselves, and not the last admin
        me = self.req("GET", "/auth/me", key=admin)[1]["user"]["id"]
        self.assertEqual(self.req("POST", "/api/users/update", {"id": me, "role": "user"}, key=admin)[0], 409)
        self.assertEqual(self.req("POST", "/api/users/update", {"id": me, "disabled": True}, key=admin)[0], 409)
        self.assertEqual(self.req("POST", "/api/users/delete", {"id": me}, key=admin)[0], 409)
        self.assertEqual(self.req("POST", "/api/users/delete", {"id": 999}, key=admin)[0], 404)
        self.assertEqual(self.req("POST", "/api/users/delete", {"id": bob_id}, key=admin)[0], 200)
        self.assertEqual(self.chat(key)[0], 401)

    def test_sign_out_and_password(self):
        admin, user, key, bob_id = self.people()
        s, j = self.req("POST", "/auth/password", {"current": "wrong one", "new": "another password"}, key=user)
        self.assertEqual(s, 403)
        s, j = self.req("POST", "/auth/password", {"current": PW, "new": "short"}, key=user)
        self.assertEqual(s, 400)
        s, j = self.req("POST", "/auth/password", {"current": PW, "new": "another password"}, key=user)
        self.assertEqual(s, 200, j)
        self.assertEqual(self.req("GET", "/auth/me", key=user)[1]["user"]["name"], "bob")   # still signed in
        self.assertEqual(self.req("POST", "/auth/logout", {}, key=user)[0], 200)
        self.assertIsNone(self.req("GET", "/auth/me", key=user)[1]["user"])
        self.assertEqual(self.chat(user)[0], 401)
        self.assertEqual(self.chat(key)[0], 200)                # (the key is not the session: it stays)

    def test_not_from_another_site(self):
        self.people()
        evil = {"Origin": "http://evil.example"}
        s, j = self.req("POST", "/auth/login", {"name": "alice", "password": PW}, headers=evil)
        self.assertEqual(s, 403)
        r = urllib.request.Request(self.base + "/auth/login", method="POST",      # a form's body: not JSON
                                   data=json.dumps({"name": "alice", "password": PW}).encode(),
                                   headers={"Content-Type": "text/plain"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(r, timeout=30)
        self.assertEqual(e.exception.code, 415)
        e.exception.close()

    def test_requests_are_their_users(self):
        admin, user, key, bob_id = self.people()
        self.assertEqual(self.chat(key)[0], 200)
        self.assertEqual(self.chat(key)[0], 200)
        self.assertEqual(self.chat(admin)[0], 200)
        rows = self.req("GET", "/metrics", key=admin)[1]["requests"]
        self.assertEqual(sorted(r["user"] for r in rows), ["alice", "bob", "bob"])
        mine = self.req("GET", "/metrics?requests=all", key=user)[1]
        self.assertEqual(([r["user"] for r in mine["requests"]], mine["requests_kept"]), (["bob", "bob"], 2))
        by_name = {u["name"]: u for u in self.req("GET", "/api/users", key=admin)[1]["users"]}
        self.assertEqual((by_name["bob"]["requests"], by_name["alice"]["requests"]), (2, 1))
        self.assertGreater(by_name["bob"]["prompt_tokens"], 0)
        self.assertGreater(by_name["bob"]["output_tokens"], 0)
        # what is being written now is not a standard user's to read
        with self.svc.status_lock:
            self.svc.status["tail"] = "someone's answer"
        self.assertNotIn("tail", self.req("GET", "/status", key=user)[1])
        self.assertEqual(self.req("GET", "/status", key=admin)[1]["tail"], "someone's answer")

    def test_mcp_tools_are_an_admins(self):
        """The MCP tools run on the server's PC with its rights: a standard user's chat is not offered them."""
        admin, user, key, bob_id = self.people()

        class Hub:
            asked = 0

            def wait(self, s):
                pass

            def template_tools(self, exclude=()):
                Hub.asked += 1
                return []

            def status(self):
                return {"servers": [], "tools": 0}

        self.svc.mcp = Hub()
        try:
            body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "strata_mcp": True}
            for who in (user, key):
                self.assertEqual(self.req("POST", "/v1/chat/completions", body, key=who)[0], 200)
                self.assertEqual(self.req("GET", "/mcp", key=who)[0], 403)
            self.assertEqual(Hub.asked, 0)
            self.assertEqual(self.req("POST", "/v1/chat/completions", body, key=admin)[0], 200)
            self.assertEqual(Hub.asked, 1)
            self.assertEqual(self.req("GET", "/mcp", key=admin)[0], 200)
        finally:
            self.svc.mcp = None

    def test_stored_responses_are_their_users(self):
        admin, user, key, bob_id = self.people()
        s, j = self.req("POST", "/v1/responses", {"input": "hi", "max_output_tokens": 8, "store": True}, key=key)
        self.assertEqual(s, 200, j)
        rid = j["id"]
        self.assertEqual(self.req("GET", f"/v1/responses/{rid}", key=user)[0], 200)   # the user's, by key or session
        self.assertEqual(self.req("GET", f"/v1/responses/{rid}", key=admin)[0], 404)  # nobody else's, an admin's neither


class KeyedServer(Routes):
    """An API key in the config: it stays an admin's key once there are users."""
    CONFIG_KEY = "config-key"

    def test_the_configs_key_makes_the_first_user(self):
        self.assertEqual(self.req("POST", "/api/users", {"name": "alice", "password": PW, "role": "admin"})[0], 401)
        s, j = self.req("GET", "/auth/me", key=self.CONFIG_KEY)
        self.assertEqual((j["users"], j["user"]["via"], j["user"]["role"]), (False, "config", "admin"))
        admin, user, key, bob_id = self.people()
        # the tools set up with it keep working, as an admin
        self.assertEqual(self.chat(self.CONFIG_KEY)[0], 200)
        self.assertEqual(self.req("GET", "/api/users", key=self.CONFIG_KEY)[0], 200)
        self.assertEqual(self.req("POST", "/api/users/update", {"id": bob_id, "role": "admin"},
                                  key=self.CONFIG_KEY)[1]["user"]["role"], "admin")
        # it is no user: no keys of its own, no password
        self.assertEqual(self.req("GET", "/api/keys", key=self.CONFIG_KEY)[1]["keys"], [])
        self.assertEqual(self.req("POST", "/api/keys", {"name": "x"}, key=self.CONFIG_KEY)[0], 400)
        self.assertEqual(self.req("POST", "/auth/password", {"current": "a", "new": "b" * 9},
                                  key=self.CONFIG_KEY)[0], 400)
        s, j = self.req("POST", "/api/keys", {"name": "for bob", "user_id": bob_id}, key=self.CONFIG_KEY)
        self.assertEqual((s, j["key"]["user"]), (200, "bob"))   # ... but it makes one for a user
        self.assertEqual(self.chat("wrong")[0], 401)
        self.assertEqual(self.chat()[0], 401)


class Chats(Routes):
    """A signed-in user's chats and folders are kept on the server: theirs alone, on every device."""
    CONFIG_KEY = "config-key"

    @staticmethod
    def a_chat(cid="c1", text="hello", updated=1000.0):
        return {"id": cid, "title": text, "created": 1.0, "updated": updated,
                "messages": [{"id": "m1", "role": "user", "text": text}]}

    def test_each_users_chats_are_their_own(self):
        admin, user, key, bob_id = self.people()
        self.assertEqual(self.req("GET", "/api/chats", key=user), (200, {"chats": []}))
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat("c1", "bob's"), key=user)[0], 200)
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat("c1", "alice's"), key=admin)[0], 200)   # (same id)
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat("c2", "alice's other", 2000.0), key=admin)[0], 200)
        s, j = self.req("GET", "/api/chats", key=user)
        self.assertEqual([c["title"] for c in j["chats"]], ["bob's"])
        s, j = self.req("GET", "/api/chats", key=admin)             # an admin too: their own, not everyone's
        self.assertEqual([c["title"] for c in j["chats"]], ["alice's other", "alice's"])   # (the newest first)
        self.assertEqual(self.req("GET", "/api/chats?index=1", key=admin)[1],
                         {"index": [{"id": "c2", "updated": 2000.0}, {"id": "c1", "updated": 1000.0}]})
        self.assertEqual(self.req("GET", "/api/chats?id=c2", key=admin)[1]["chat"]["title"], "alice's other")
        self.assertEqual(self.req("GET", "/api/chats?id=c2", key=user)[0], 404)
        # the user's API key is the user too
        self.assertEqual([c["title"] for c in self.req("GET", "/api/chats", key=key)[1]["chats"]], ["bob's"])
        # a save replaces the chat (the last one wins); a delete is the caller's own
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat("c1", "bob's, longer", 3000.0), key=user)[0], 200)
        self.assertEqual(self.req("GET", "/api/chats", key=user)[1]["chats"][0]["title"], "bob's, longer")
        self.assertEqual(self.req("POST", "/api/chats/delete", {"id": "c1"}, key=user)[0], 200)
        self.assertEqual(self.req("GET", "/api/chats", key=user)[1]["chats"], [])
        self.assertEqual(len(self.req("GET", "/api/chats", key=admin)[1]["chats"]), 2)
        self.assertEqual(self.req("POST", "/api/chats/delete", {"all": True}, key=admin)[1]["deleted"], 2)
        self.assertEqual(self.req("GET", "/api/chats", key=admin)[1]["chats"], [])

    def test_a_chat_comes_back_as_it_was_sent(self):
        admin, user, key, bob_id = self.people()
        chat = self.a_chat("k9-Zx_1", "héllo 世界 \U0001F600")
        chat["messages"].append({"id": "m2", "role": "assistant", "text": "cut inside an emoji: \ud83d",   # half a pair
                                 "images": [{"name": "a.png", "url": "data:image/png;base64," + "A" * 4000}]})
        self.assertEqual(self.req("POST", "/api/chats", chat, key=user)[0], 200)
        self.assertEqual(self.req("GET", "/api/chats", key=user)[1]["chats"], [chat])
        self.assertEqual(self.req("GET", "/api/chats?id=k9-Zx_1", key=user)[1]["chat"], chat)

    def test_what_is_not_a_chat_is_refused(self):
        admin, user, key, bob_id = self.people()
        for bad in ({"updated": 1, "messages": []}, {"id": "a b", "updated": 1, "messages": []},
                    {"id": "x" * 65, "updated": 1, "messages": []}, {"id": 7, "updated": 1, "messages": []},
                    {"id": "c1", "messages": []}, {"id": "c1", "updated": "now", "messages": []},
                    {"id": "c1", "updated": True, "messages": []}, {"id": "c1", "updated": 1}):
            self.assertEqual(self.req("POST", "/api/chats", bad, key=user)[0], 400, bad)
        self.assertEqual(self.req("POST", "/api/chats", [self.a_chat()], key=user)[0], 400)
        self.assertEqual(self.req("GET", "/api/chats", key=user)[1]["chats"], [])
        # nobody is not asked in, and the config's key is no user: its page keeps its chats in the browser
        self.assertEqual(self.req("GET", "/api/chats")[0], 401)
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat())[0], 401)
        self.assertEqual(self.req("GET", "/api/chats", key=self.CONFIG_KEY)[0], 400)
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat(), key=self.CONFIG_KEY)[0], 400)
        self.assertEqual(self.req("GET", "/api/workspace", key=self.CONFIG_KEY)[0], 400)
        # another site's page cannot write them
        self.assertEqual(self.req("POST", "/api/chats", self.a_chat(), key=user,
                                  headers={"Origin": "http://evil.example"})[0], 403)
        # a page opened by one user, the browser since signed in as another: its chats are not the new one's
        self.assertEqual(self.req("POST", f"/api/chats?as={bob_id}", self.a_chat(), key=user)[0], 200)
        self.assertEqual(self.req("POST", f"/api/chats?as={bob_id}", self.a_chat("c2"), key=admin)[0], 409)
        self.assertEqual(self.req("GET", f"/api/chats?as={bob_id}", key=admin)[0], 409)
        self.assertEqual(self.req("POST", f"/api/chats/delete?as={bob_id}", {"all": True}, key=admin)[0], 409)
        self.assertEqual(self.req("POST", f"/api/workspace?as={bob_id}", {"folders": []}, key=admin)[0], 409)
        self.assertEqual(self.req("GET", "/api/chats", key=admin)[1]["chats"], [])
        self.assertEqual(len(self.req("GET", f"/api/chats?as={bob_id}", key=user)[1]["chats"]), 1)

    def test_an_account_has_a_size(self):
        admin, user, key, bob_id = self.people()
        os.environ["MAYA_CHATS_QUOTA_MIB"] = "1"
        try:
            big = {**self.a_chat("big", "x" * (700 << 10)), "title": "big"}
            self.assertEqual(self.req("POST", "/api/chats", big, key=user)[0], 200)
            s, j = self.req("POST", "/api/chats", {**big, "id": "big2"}, key=user)
            self.assertEqual(s, 413)
            self.assertIn("1 MiB", j["error"]["message"])
            self.assertEqual(self.req("POST", "/api/chats", big, key=user)[0], 200)        # (itself again: replaced)
            self.assertEqual(self.req("POST", "/api/chats", {**big, "id": "big2"}, key=admin)[0], 200)   # (alice's own room)
        finally:
            del os.environ["MAYA_CHATS_QUOTA_MIB"]

    def test_folders(self):
        admin, user, key, bob_id = self.people()
        self.assertEqual(self.req("GET", "/api/workspace", key=user), (200, {"workspace": None}))
        w = {"folders": [{"id": "f1", "name": "Work"}], "folderOpen": {"f1": True}}
        self.assertEqual(self.req("POST", "/api/workspace", w, key=user)[0], 200)
        self.assertEqual(self.req("GET", "/api/workspace", key=user)[1]["workspace"], w)
        self.assertEqual(self.req("GET", "/api/workspace", key=admin)[1]["workspace"], None)
        self.assertEqual(self.req("POST", "/api/workspace", {"folders": []}, key=user)[0], 200)
        self.assertEqual(self.req("GET", "/api/workspace", key=user)[1]["workspace"], {"folders": []})
        self.assertEqual(self.req("POST", "/api/workspace", {"folders": ["x" * (300 << 10)]}, key=user)[0], 413)

    def test_a_deleted_user_takes_their_chats_along(self):
        admin, user, key, bob_id = self.people()
        self.req("POST", "/api/chats", self.a_chat(), key=user)
        self.req("POST", "/api/workspace", {"folders": []}, key=user)
        self.assertEqual(self.req("POST", "/api/users/delete", {"id": bob_id}, key=admin)[0], 200)
        db = self.svc.users.db
        self.assertEqual(db.execute("SELECT COUNT(*) FROM chats").fetchone()[0], 0)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0], 0)

    def test_a_users_file_from_before_gets_the_tables(self):
        with tempfile.TemporaryDirectory() as d:
            con = sqlite3.connect(str(Path(d) / "u.db"))
            con.executescript(users.SCHEMA.split("CREATE TABLE IF NOT EXISTS chats")[0])
            con.execute("INSERT INTO users (name, password, role, created) VALUES ('old', 'x', 'admin', 0)")
            con.commit()
            con.close()
            s = users.UserStore(Path(d) / "u.db")
            s.put_chat(1, self.a_chat())
            self.assertEqual(s.chat_index(1), [{"id": "c1", "updated": 1000.0}])
            s.close()


class NoStore(Routes):
    """A server without a users' file (no run config): as it always was."""

    def setUp(self):
        super().setUp()
        self.svc.users.close()
        self.svc.users = None

    def tearDown(self):
        self.svc.users = users.UserStore(Path(self.tmp.name) / "maya-users.db")
        super().tearDown()

    def test_as_before(self):
        s, j = self.req("GET", "/auth/me")
        self.assertEqual((j["users"], j["available"], j["user"]["via"]), (False, False, "open"))
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.req("GET", "/metrics")[1]["requests"][0]["user"], None)
        self.assertEqual(self.req("POST", "/api/users", {"name": "alice", "password": PW, "role": "admin"})[0], 400)
        self.assertEqual(self.req("GET", "/api/users")[0], 400)
        self.svc.api_key = "k1,k2"
        s, j = self.chat()
        self.assertEqual((s, j["error"]["message"]), (401, "missing or wrong API key"))
        self.assertEqual(self.chat("k2")[0], 200)


class Schema(unittest.TestCase):
    def test_a_role_is_one_of_two(self):
        with tempfile.TemporaryDirectory() as d:
            s = users.UserStore(Path(d) / "u.db")
            with self.assertRaises(sqlite3.IntegrityError):
                s.db.execute("INSERT INTO users (name, password, role, created) VALUES ('x', 'y', 'root', 0)")
            s.close()


if __name__ == "__main__":
    unittest.main()
