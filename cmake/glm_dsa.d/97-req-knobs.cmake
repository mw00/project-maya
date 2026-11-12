# glm-dsa, speed plan code change 2: per-request engine knobs (include/strata/core/glm_req_knobs.hpp).  The knob store
# and the GEN-key parser are header-only (Maya-owned glm_fast_path.cu / glm_window.cu / glm_fast.cu and glm-dsa's
# glm_tier_numa.cu, glm_residual.cu and glm_ngram_draft.cpp read it); glm_req_knobs.cpp applies a request's keys and
# holds Glm5Model::set_pcie_share_head, called from generate.cpp's GEN line, so it joins strata_engine on any GPU
# backend (its lane_row4 call only where the row4 lane exists: STRATA_NATIVE_EXPERTS, strata_kernels_cpu's PUBLIC
# define).  CPU checks: glm_env_test (key parsing, start values, the graphs' request reason) and glm_ngram_draft_test.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_req_knobs.cpp)
endif()
