# cmake/strata.d

One CMake fragment per glm-dsa feature (for example `20-glm-dsa-attention.cmake`), included in sorted order by
`cmake/strata.cmake` after the base lists. A fragment registers its engine sources with
`target_sources(strata_engine PRIVATE ...)` (guard with `if(TARGET strata_engine)` and, for `.cu` files,
`STRATA_ENABLE_CUDA`) and its tests inside `if(STRATA_BUILD_TESTS)`, labelling GPU tests `LABELS gpu`.
Labels: `gpu` (needs a device; exit 77 without one), `realpack` (needs a real model file; register it only when the
path exists), `fixture` (a FIXTURES_SETUP generator). `90-ci-tests.cmake` runs last and labels Maya's device tests,
so `ctest -LE gpu` is green on a CPU box (docs/strata/engine/TESTS.md). An edit of a Maya-owned file needs an
STRATA-EDITS.txt line and a docs/strata/fork-allowlist.txt entry in the same commit.
