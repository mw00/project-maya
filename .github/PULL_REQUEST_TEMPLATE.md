## What this changes

<!-- One paragraph. Link the issue if there is one. -->

## Type

- [ ] Bug fix
- [ ] Performance change
- [ ] New feature or setting
- [ ] Documentation / presentation
- [ ] Build, CI or packaging

## Checks

- [ ] `./setup.sh` builds, or the CPU-only build passes
- [ ] `python -m pytest` (server and tools) passes
- [ ] If this changes a kernel path: NVIDIA still builds, and the HIP mapping is covered
- [ ] If this changes performance: numbers before and after, same harness, in the description
- [ ] If this changes behaviour: the docs say so, and `CHANGELOG.md` has a line
- [ ] No hostnames, personal paths, internal plan identifiers or secrets in the diff
- [ ] Links and anchors still resolve (docs, README, translations)

## Notes for the reviewer

<!-- Anything you did not measure, anything you are unsure about, follow-up work. -->
