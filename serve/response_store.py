"""Credential-scoped snapshots sharing immutable JSON item blocks without parent dependencies."""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import hashlib
import json
import threading
import time


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class StoreBusy(ValueError):
    pass


class ResponseStore:
    def __init__(self, max_entries=128, max_bytes=64 * 1024 * 1024, ttl=3600, clock=time.monotonic,
                 max_entry_bytes=None, owner_entries=0, owner_bytes=0, assembly_bytes=None):
        self.max_entries, self.max_bytes, self.ttl, self.clock = max_entries, max_bytes, ttl, clock
        self.max_entry_bytes = max_bytes if max_entry_bytes is None else max_entry_bytes
        self.owner_entries, self.owner_bytes = owner_entries, owner_bytes
        self.assembly_limit = max_bytes if assembly_bytes is None else assembly_bytes
        self._entries, self._blocks = OrderedDict(), {}
        self._bytes, self._inflight, self._inflight_peak = 0, 0, 0
        self._owners, self._lock = {}, threading.RLock()
        self._metrics = Counter({name: 0 for name in ("hits", "misses", "writes", "expired", "evicted", "deleted",
                                                     "rejected", "assembly_rejected", "write_seconds", "read_seconds",
                                                     "lock_wait_seconds", "lock_hold_seconds")})

    @contextmanager
    def _guard(self):
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            self._metrics["lock_wait_seconds"] += acquired - started
            try:
                yield
            finally:
                self._metrics["lock_hold_seconds"] += time.perf_counter() - acquired

    @classmethod
    def from_limits(cls, limits):
        return cls(max_entries=limits["CACHE_ENTRIES"], max_bytes=limits["CACHE_BYTES"], ttl=limits["CACHE_TTL_S"],
                   max_entry_bytes=limits["ENTRY_BYTES"], owner_entries=limits["OWNER_ENTRIES"],
                   owner_bytes=limits["OWNER_BYTES"], assembly_bytes=limits["ASSEMBLY_BYTES"])

    def _drop(self, key, reason=None):
        entry = self._entries.pop(key)
        freed = entry["size"]
        for block in entry["input"] + entry["output"]:
            self._blocks[block][1] -= 1
            if self._blocks[block][1] == 0:
                freed += len(self._blocks.pop(block)[0])
        self._bytes -= freed
        self._owners[key[0]] -= freed
        if self._owners[key[0]] == 0:
            del self._owners[key[0]]
        if reason:
            self._metrics[reason] += 1

    def _expire(self):
        now = self.clock()
        for key, entry in list(self._entries.items()):
            if entry["deadline"] <= now:
                self._drop(key, "expired")

    def put(self, owner, response, items):
        started = time.perf_counter()
        output = response.get("output", [])
        metadata = encoded({key: value for key, value in response.items() if key != "output"})
        blocks, refs = {}, []
        for group in (items, output):
            group_refs = []
            for item in group:
                data = encoded(item)
                block = owner, hashlib.sha256(data).hexdigest()
                blocks[block] = data
                group_refs.append(block)
            refs.append(tuple(group_refs))
        descriptor = len(metadata) + sum(len(encoded([key[1] for key in group])) for group in refs)
        logical_size = len(metadata) + sum(len(blocks[key]) for group in refs for key in group)
        key = owner, response["id"]
        with self._guard():
            self._expire()
            if logical_size > self.max_entry_bytes or self.max_entries < 1:
                self._metrics["rejected"] += 1
                raise ValueError("stored response exceeds the in-memory Responses cache capacity; use store=false")
            # Plan all evictions first so an impossible write leaves existing snapshots intact.
            counts = {block: record[1] for block, record in self._blocks.items()}
            size, owner_size = self._bytes, self._owners.get(owner, 0)
            remaining = list(self._entries)
            removed = []

            def discard(victim):
                nonlocal size, owner_size
                entry = self._entries[victim]
                freed = entry["size"]
                for block in entry["input"] + entry["output"]:
                    counts[block] -= 1
                    if counts[block] == 0:
                        freed += len(self._blocks[block][0])
                size -= freed
                if victim[0] == owner:
                    owner_size -= freed
                remaining.remove(victim)
                removed.append(victim)

            if key in self._entries:
                discard(key)
            added = descriptor
            for group in refs:
                for block in group:
                    if counts.get(block, 0) == 0:
                        added += len(blocks[block])
                    counts[block] = counts.get(block, 0) + 1
            size += added
            owner_size += added
            while True:
                quota = ((self.owner_entries and sum(k[0] == owner for k in remaining) + 1 > self.owner_entries) or
                         (self.owner_bytes and owner_size > self.owner_bytes))
                if not quota and len(remaining) + 1 <= self.max_entries and size <= self.max_bytes:
                    break
                victim = next((k for k in remaining if not quota or k[0] == owner), None)
                if victim is None:
                    self._metrics["rejected"] += 1
                    raise ValueError("stored response exceeds the in-memory Responses cache capacity; use store=false")
                discard(victim)
            for victim in removed:
                self._drop(victim, "evicted" if victim != key else None)
            actual = descriptor
            for group in refs:
                for block in group:
                    if block not in self._blocks:
                        self._blocks[block] = [blocks[block], 0]
                        actual += len(blocks[block])
                    self._blocks[block][1] += 1
            self._entries[key] = {"metadata": metadata, "input": refs[0], "output": refs[1],
                                  "size": descriptor, "deadline": self.clock() + self.ttl}
            self._bytes += actual
            self._owners[owner] = self._owners.get(owner, 0) + actual
            self._metrics["writes"] += 1
            self._metrics["write_seconds"] += time.perf_counter() - started

    def get(self, owner, response_id):
        started = time.perf_counter()
        with self._guard():
            self._expire()
            key = owner, response_id
            if key not in self._entries:
                self._metrics["misses"] += 1
                raise KeyError(response_id)
            entry = self._entries[key]
            groups = [tuple(self._blocks[block][0] for block in entry[group]) for group in ("input", "output")]
            metadata = entry["metadata"]
            assembly = len(metadata) + sum(len(data) for group in groups for data in group)
            if self._inflight + assembly > self.assembly_limit:
                self._metrics["assembly_rejected"] += 1
                raise StoreBusy("Responses snapshot assembly is at capacity; retry later")
            self._inflight += assembly
            self._inflight_peak = max(self._inflight_peak, self._inflight)
            self._entries.move_to_end(key)
            self._metrics["hits"] += 1
        try:
            response = json.loads(metadata)
            response["output"] = [json.loads(data) for data in groups[1]]
            return {"response": response, "input": [json.loads(data) for data in groups[0]]}
        finally:
            with self._guard():
                self._inflight -= assembly
                self._metrics["read_seconds"] += time.perf_counter() - started

    def delete(self, owner, response_id):
        with self._guard():
            self._expire()
            key = owner, response_id
            if key not in self._entries:
                raise KeyError(response_id)
            self._drop(key, "deleted")

    def expand(self, owner, req):
        previous = req.get("previous_response_id")
        if previous is None:
            return req
        if not isinstance(previous, str) or not previous:
            raise ValueError("previous_response_id must be a nonempty string")
        stored = self.get(owner, previous)
        added = req.get("input", [])
        if isinstance(added, str):
            added = [{"role": "user", "content": added}]
        if not isinstance(added, list):
            raise ValueError("input must be a string or array of input items")
        return {**req, "input": stored["input"] + stored["response"]["output"] + added}

    def stats(self):
        with self._guard():
            self._expire()
            return {**dict(self._metrics), "entries": len(self._entries), "blocks": len(self._blocks),
                    "bytes": self._bytes, "inflight_bytes": self._inflight, "peak_inflight_bytes": self._inflight_peak,
                    "max_entries": self.max_entries, "max_bytes": self.max_bytes, "ttl_s": self.ttl}
