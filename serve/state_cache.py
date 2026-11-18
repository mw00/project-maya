"""Native conversation-state diagnostics, separate from decode-rate readings."""

import math
import time

COUNTERS = {
    "hits",
    "misses",
    "restored_tokens",
    "admitted",
    "rejected",
    "evicted",
    "promoted",
    "demoted",
    "read_failures",
    "write_failures",
    "calibration_samples",
    "ram_hits",
    "disk_hits",
    "publishes",
    "publish_ms",
    "read_bytes",
    "reused_bytes",
    "hashed_bytes",
    "restores",
    "restore_ms",
}
GAUGES = {
    "units",
    "ram_bytes",
    "disk_bytes",
    "logical_bytes",
    "watermark",
    "cost_d",
    "cost_e",
    "pending_writes",
    "unique_bytes",
    "shared_bytes",
    "pinned_units",
    "pending_updates",
}


def parse(line):
    result = {}
    for item in line.split()[1:]:
        name, sep, raw = item.partition("=")
        if not sep or name not in COUNTERS | GAUGES:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            result[name] = value
    if result:
        result["time"] = time.time()
    return result


def metrics(reading):
    rows = []
    for name in sorted(COUNTERS | GAUGES):
        value = reading.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            counter = name in COUNTERS
            key = f"maya:state_cache_{name}" + ("_total" if counter else "")
            rows.append(
                (
                    key,
                    "counter" if counter else "gauge",
                    f"Conversation state cache {name}",
                    value,
                )
            )
    return rows
