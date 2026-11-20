#!/usr/bin/env python3
"""Print opt-in GLM shared-pool settings for an explicitly chosen thread count.

Linux only. Read this process's allowed, online CPUs and their NUMA/SMT
topology. Choose one hardware thread per core before SMT siblings, cycling
through NUMA nodes with capacity. This tool neither benchmarks nor changes
affinity, configuration files, or a running service.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import sys


def cpu_list(text: str) -> set[int]:
    """Read the comma/range notation used by Linux sysfs CPU lists."""
    cpus: set[int] = set()
    if not text.strip():
        return cpus
    for part in text.strip().split(","):
        if re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part) is None:
            raise ValueError(f"invalid sysfs CPU list: {text.strip()!r}")
        bounds = part.split("-")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first:
            raise ValueError(f"reversed CPU range: {part}")
        cpus.update(range(first, last + 1))
    return cpus


def cpu_order(sys_root: Path, allowed: set[int]) -> list[int]:
    """Order allowed online CPUs, core first, then SMT, round-robin by node."""
    base = sys_root / "devices/system"
    eligible = allowed & cpu_list((base / "cpu/online").read_text())
    if not eligible:
        raise ValueError("the process has no allowed online CPUs")
    # The engine's Linux pin_thread uses a fixed cpu_set_t and ignores larger
    # IDs. Do not produce a list that it would only partially apply.
    if max(eligible) >= 1024:
        raise ValueError("CPU IDs >= 1024 are unsupported by the engine's fixed affinity mask")

    locations: dict[int, int] = {}
    for node in base.glob("node/node[0-9]*"):
        if not node.name[4:].isdigit():
            continue
        for cpu in eligible & cpu_list((node / "cpulist").read_text()):
            if cpu in locations:
                raise ValueError(f"CPU {cpu} appears in more than one NUMA node")
            locations[cpu] = int(node.name[4:])
    if set(locations) != eligible:
        missing = sorted(eligible - locations.keys())
        raise ValueError(f"NUMA node information is missing for CPUs {missing}")

    # thread_siblings_list identifies the physical core without assuming that
    # core IDs are unique across packages/dies, or that SMT is two-way/enabled.
    siblings: dict[int, tuple[int, ...]] = {}
    for cpu in sorted(eligible):
        path = base / f"cpu/cpu{cpu}/topology/thread_siblings_list"
        group = cpu_list(path.read_text())
        if cpu not in group:
            raise ValueError(f"CPU {cpu} is absent from its own SMT sibling list")
        siblings[cpu] = tuple(sorted(group))
    cores: dict[int, list[list[int]]] = {}
    for group in sorted(set(siblings.values())):
        visible = sorted(eligible.intersection(group))
        node = locations[visible[0]]
        if any(siblings[cpu] != group or locations[cpu] != node for cpu in visible):
            raise ValueError("inconsistent SMT/NUMA topology; generate the list again on a stable host")
        cores.setdefault(node, []).append(visible)

    nodes = [sorted(cores[node], key=lambda group: group[0]) for node in sorted(cores)]
    order: list[int] = []
    for sibling in range(max(len(group) for groups in nodes for group in groups)):
        per_node = [[group[sibling] for group in groups if sibling < len(group)] for groups in nodes]
        for row in itertools.zip_longest(*per_node):
            order.extend(cpu for cpu in row if cpu is not None)
    return order


def pool_cpus(sys_root: Path, allowed: set[int], threads: int) -> list[int]:
    if threads <= 0:
        raise ValueError("threads must be a positive integer selected for this machine")
    order = cpu_order(sys_root, allowed)
    if threads > len(order):
        raise ValueError(f"requested {threads} threads, but only {len(order)} allowed online CPUs are available")
    return order[:threads]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True,
                        help="explicit CPU-lane count to compare on this machine; no default or automatic tuning")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be a positive integer")
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        parser.error("Linux with sched_getaffinity and readable sysfs topology is required")
    try:
        cpus = pool_cpus(Path("/sys"), set(os.sched_getaffinity(0)), args.threads)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"env": {"MAYA_GLM_CPU_LANE": str(args.threads),
                              "MAYA_GLM_POOL_CPUS": ",".join(map(str, cpus))}}, indent=2))
    print(f"Generated {len(cpus)} entries for {args.threads - 1} background workers; "
          "the last entry is unused, and calling threads are not pinned by this list.\n"
          "Merge the two settings into a separate test config for the global shared pool. "
          "This is a placement proposal, not a recommended thread count or a measured speedup.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
