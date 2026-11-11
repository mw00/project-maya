"""tools/maya_names.py - Maya's settings are MAYA_*; the older STRATA_* names still work as fallbacks: the environment
and a config's "env" take either, the code reads MAYA_* (MAYA_X first, then STRATA_X), on disk and on screen they are MAYA_*.

    alias_env(os.environ)    every STRATA_X that has no MAYA_X also sets MAYA_X (MAYA_X wins), so what a program reads and
                             what it starts (the engines) find the setting under the name they read
    env_in(cfg)              a config as read from disk: its "env" under the names the code reads
    env_out(cfg)             a config about to be written: its "env" under the names users see
"""
from __future__ import annotations

import os

OLD, NEW = "STRATA_", "MAYA_"


def shown(name: str) -> str:
    """A setting's name as users see it."""
    return NEW + name[len(OLD):] if name.startswith(OLD) else name


def read_as(name: str) -> str:
    """A setting's name as the code reads it (MAYA_*; the older STRATA_* name is the fallback)."""
    return shown(name)


def alias_env(env) -> None:
    """In place (os.environ too): every STRATA_X that has no MAYA_X sets MAYA_X (the MAYA_ one counts when both are set)."""
    for k, v in list(env.items()):
        if k.startswith(OLD) and shown(k) not in env:
            env[shown(k)] = v


def both_names(name: str) -> tuple[str, str]:
    """(MAYA_X, STRATA_X) for a setting under either name; a name with neither prefix is its own twice."""
    m = shown(name)
    return m, (OLD + m[len(NEW):] if m.startswith(NEW) else m)


def lookup(env, name: str, default=None):
    """A setting from a dict/environment by either of its names (the MAYA_ one first).  A config in memory has its
    settings under the names the code reads (env_in), the shell and the dashboard under the names users see - code
    that reads a setting from either must not care which."""
    for k in both_names(name):
        v = env.get(k)
        if v is not None:
            return v
    return default


def export_env(target, env) -> None:
    """Put a config's "env" into `target` (a child's environment, os.environ) under BOTH names, so the program that
    reads a setting as getenv("MAYA_X") and the one that reads getenv("STRATA_X") each find it (issue #686: env_in
    leaves a config with the STRATA_ name only, and an engine that reads the MAYA_ name alone never saw it)."""
    for k, v in sorted((env or {}).items(), key=lambda kv: str(kv[0]).startswith(NEW)):   # (a MAYA_ key last: it wins)
        for name in {str(k), *both_names(str(k))}:
            target[name] = str(v)


def env_in(cfg):
    """A config as read from disk -> its "env" under the names the code reads (in place; returned).  Where a config
    has a setting under both names, the MAYA_ one counts."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if isinstance(e, dict):
        out = {shown(k): v for k, v in e.items() if not k.startswith(NEW)}
        out.update({k: v for k, v in e.items() if k.startswith(NEW)})
        cfg["env"] = out
    return cfg


def env_out(cfg):
    """A config about to be written -> a copy with its "env" under the names users see."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if not isinstance(e, dict):
        return cfg
    return {**cfg, "env": {shown(k): v for k, v in e.items()}}


alias_env(os.environ)   # (importing this module is enough: an older STRATA_X in the environment also counts as MAYA_X)
