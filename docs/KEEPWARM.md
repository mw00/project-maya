# Keeping the model files warm in the page cache

`tools/maya_keepwarm.py` (Linux) records which ranges of the model's files the OS page cache holds right now and, after a reboot, a `drop_caches`, or an eviction wave, asks the kernel with `posix_fadvise(WILLNEED)` to read exactly those back — never whole files, nothing written, nothing locked. On a three-GPU box running the DeepSeek-V4.1-Flash `Mix-256G` pack (378.1 GiB over 10 files), one resident scan took **16.9–22 s**, and a boot-time restore of a saved set whose pages were already resident asked the kernel back **107–159 GiB of ranges in about one second** — pages still resident cost no I/O at all, evicted ones re-read at drive speed instead of trickling back through the next hours of prompts.

## Why the tool exists

For the DS4 CPU-lane experts the page cache *is* the RAM tier: SETTINGS.md's page-cache rows ("the OS page cache decides which experts hold"). The engine already remembers **what** to keep warm — it saves per-expert route counts to `<pack>/expert-profile.bin` at the end of every run (serve included), and the next start warms the most-routed experts from that file (see `DS4_WARM_CAP_GB`, `MAYA_DS4_PREREAD`). It does not remember **where**: page residency dies with the reboot, and `tools/refresh_ds4_profile.py` improves the same-restart gap one tier up (expert VRAM placement). So after a restart the first minutes re-read the cold tail of each shard from the SSD while the cache re-fills through ordinary faults — on the reference box the repeated-prompt difference between a cold and a warm cache is roughly **2.7 vs 14.3 tokens/s** in the first response.

## Modes

- `status` - the guards' lead (`MemAvailable`, io PSI), the state file's summary, three files' current residency.
- `snapshot` - `mincore()` over every weight file of the warm set (2 MiB windows mapped 256 MiB at a time; the walk of 378.1 GiB takes ~17–22 s of user CPU). The found ranges are **merged** with the previously saved set (below), then written to `maya-keepwarm.json` beside the repo root (`[start, length, generation]` spans).
- `restore` - `WILLNEED` every saved range (never whole files), after the guards pass.
- `daemon` - restore at boot, then snapshot every `--interval` (default 900 s).

## The merge: what survives, what ages out

Each snapshot's set is the union of "resident now" and what previous snapshots learned, with generations:

- ranges resident right now are generation **0** and are never pruned;
- parts of a prior snapshot's ranges the traffic has since evicted age one generation per snapshot and are dropped after `--gen-life` (default 8);
- `--cap-gb` (default 120) is a backstop over the whole saved set: the resident part always counts into it untouched and the learned budget is what remains, oldest generations first;
- `--cap-gb 0` keeps only what is resident now (`gen-life`, the learned history, stays unused).

The intent: a region the traffic evicted for one quiet hour stays in the restore set, but a working set that moved on is not remembered forever; what restores stays inside the cache budget the box actually had free. Writes are atomic per process (`maya-keepwarm.json.tmp.<pid>`), so the daemon and a manual snapshot cannot corrupt each other.

## The guards

Restores (and only restores) hold off while either guard trips — again checked per file, `--force` overrides once:

- `--floor-gb` (default 24): `MemAvailable` in GB below which nothing is read back;
- `--psi-max` (default 12): `/proc/pressure/io` `some avg10` percent above which nothing is read back.

The guard is not hypothetical: one measured boot restore stopped on its own when the box's io PSI sat at 71% (still in the readahead of the previous restore's ~30 GiB of re-reads) and resumed cleanly on the next cycle. The daemon runs `Nice=10` / `IOSchedulingClass=idle` from the unit below and never `mlock`s — everything it keeps stays ordinary, kernel-reclaimable page cache, so it cannot out-promise the engine's own budgeting (`DS4_RAM_MARGIN_GB`).

## Run it

```console
$ python3 tools/maya_keepwarm.py status                    # the config is found beside the repo root
$ python3 tools/maya_keepwarm.py snapshot maya-ds4-hip.json  # or pass the config / --pack DIR
$ python3 tools/maya_keepwarm.py restore                   # manual, respects the guards
```

The files come from the config the setup runs (a `maya-*.json` beside the repo root whose `args` carry `--pack`); pass the config or `--pack` explicitly when several configs qualify. A systemd unit for the daemon (edit the user, root and config paths):

```ini
[Unit]
Description=maya-keepwarm: keep Project Maya's model files resident in the page cache
After=multi-user.target

[Service]
Type=simple
User=ehco
ExecStart=/usr/bin/python3 /opt/project-maya/tools/maya_keepwarm.py daemon /opt/project-maya/maya-ds4-hip.json
Restart=on-failure
RestartSec=30
Nice=10
IOSchedulingClass=idle
StandardOutput=append:%h/maya-keepwarm.log
StandardError=append:%h/maya-keepwarm.log

[Install]
WantedBy=multi-user.target
```

| Option | Default | Meaning |
| --- | --- | --- |
| `config` (positional) | auto-discovered | the running install's config .json (its `--pack` is kept warm) |
| `--pack` | from the config | the pack directory directly |
| `--cap-gb` | 120 | the saved set's ceiling in GiB; `0` = resident-only |
| `--gen-life` | 8 | snapshots an unseen, evicted range survives |
| `--floor-gb` | 24 | restore holds off below this much `MemAvailable` |
| `--psi-max` | 12 | restore holds off above this io PSI `some avg10` percent |
| `--interval` | 900 | the daemon's snapshot period, seconds |
| `--no-boot-restore` | off | daemon starts snapshotting without restoring first |
| `--force` | off | one restore ignoring the guards |

## What it is not

Not an engine setting and no new `MAYA_*` environment variable; the engine need not be restarted (the daemon and a running Maya coexist; snapshots read via `mmap`). It is read-only about the model: files are only ever opened `O_RDONLY`, and nothing in the snapshot depends on model contents. Windows and other non-Linux platforms exit with a hint (the calls need `/proc`, `mincore`, `posix_fadvise`). Pinned memory is out of its reach by design: on the reference box the Engram carriers' tables live pinned in the heap tier, their file pages mostly leave the cache, and their snapshots honestly record ~0 — the tool restores only what the cache actually held.

## Measurement scope

One box: three 32 GB RDNA3 discrete cards, 128 GB host RAM, the deepseek GGUF `Mix-256G` pack (378.1 GiB) plus the Maya-XS IQ1_S Engram carriers, a v1.1.56 engine. Snapshots were taken at idle and under live server traffic; the restore numbers above are from three restarts (107–159 GiB saved sets). These are working-set residency measurements, not speed claims beyond the 2.7/14.3 tokens/s cold/warm first-response pair already recorded on this box. The state file format is version 3 and grows with distinct ranges, not with traffic.

CPU-only unit tests: `python -m unittest discover -s tools -p "test_maya_keepwarm.py"` (merge, generation-ageing, cap and config-resolution logic; 16 cases).
