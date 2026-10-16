"""Validated server-side limits for Responses state, jobs and event retention."""
import os

DEFAULTS = {
    "CACHE_ENTRIES": 128, "CACHE_BYTES": 64 * 1024 * 1024, "CACHE_TTL_S": 3600,
    "ENTRY_BYTES": 64 * 1024 * 1024, "OWNER_ENTRIES": 0, "OWNER_BYTES": 0,
    "ASSEMBLY_BYTES": 64 * 1024 * 1024,
    "JOB_WORKERS": 2, "JOB_ENTRIES": 16, "JOB_OWNER_ENTRIES": 4,
    "JOB_TIMEOUT_S": 300, "JOB_TTL_S": 900, "JOB_INPUT_BYTES": 2 * 1024 * 1024,
    "EVENT_BYTES": 2 * 1024 * 1024, "EVENT_COUNT": 4096,
    "STREAM_READERS": 32, "STREAM_OWNER_READERS": 8, "STREAM_TIMEOUT_S": 5,
}
PREFIX = "STRATA_GLM_RESPONSES_"
ENV_KEYS = tuple(PREFIX + name for name in DEFAULTS)


def limits_from_env():
    limits = {}
    for name, default in DEFAULTS.items():
        raw = os.environ.get(PREFIX + name, str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{PREFIX}{name} must be an integer") from None
        if value < 0 or (value == 0 and name not in ("OWNER_ENTRIES", "OWNER_BYTES")):
            raise ValueError(f"{PREFIX}{name} must be positive (owner quotas may be zero to disable)")
        limits[name] = value
    if limits["JOB_WORKERS"] > 32 or limits["JOB_ENTRIES"] > 4096:
        raise ValueError("Responses supports at most 32 background workers and 4096 job records")
    if limits["EVENT_BYTES"] < 4096:
        raise ValueError("Responses event buffers must be at least 4096 bytes")
    return limits
