# Roadmap

This is where the project is going, not a promise of dates. Priorities change when a new model lands or when a
hardware report changes what matters. Issues and pull requests that move these forward are welcome.

## Now

- **Presentation and documentation**: screenshot-led README, a docs index, a compatibility matrix, an API reference,
  and an FAQ. This repository restructure is the first step.
- **Speed reports as data**: a structured intake form and a generator that keeps the README table and
  `docs/benchmarks/SPEED-REPORTS.md` current.
- **Windows and AMD parity**: the same behaviour and the same settings as the NVIDIA path.

## Next

- **Official container image** (`ghcr.io/mw00/project-maya`): nightly and release tags, models mounted as a volume.
- **Apple silicon validation**: a real Mac running GLM-5.3-Flash with Metal, not just CI unit tests.
- **Troubleshooting guide** built from the issues that come back most often.
- **Docs site** with search and versioned pages, built from the same markdown.
- **Developer ergonomics**: `pyproject.toml`, a `Makefile`, formatting and link checks in CI.

## Later

- More model families in the same engine, as their weights and licences allow.
- Packaging that does not require a compiler for the Python-side tools.
- A test layout that mirrors `src/`, so a contributor can run the subset that matches their change.

## Not planned

- A hosted service or an account system. Maya runs on your machine; that is the point.
- Telemetry.
