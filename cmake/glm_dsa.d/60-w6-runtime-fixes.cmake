# glm-dsa wave 6, runtime review fixes: the MAYA_GLM_SEQS multi decode through the B2.3 residual hooks, the
# per-socket lane pool kept for the load (B8.3), the per-node free-memory cap of bound pins and per-node THP budgets
# (B8.2 / B8.7), and the node-split lane's cpu_threads knob on both nodes (B8.4).  The engine sources are already
# registered (45-glm-tiers.cmake); the CPU tests are extended there (glm_tier_plan_test, glm_tier_engine_test).
if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU (no build): no fast decode path calls gf::embed_streams / gf::head_prep outside glm_residual.cu.
  add_test(NAME glm_residual_hooks_lint
           COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_dsa/glm_residual_hooks_lint.cmake)
endif()
