# glm-dsa wave 4: the memory tiers (B8.2 segmented parallel pinning, B8.3 per-socket CPU-lane pools, B8.4 NUMA
# node-split expert layout behind MAYA_GLM_NUMA_SPLIT=1, B8.7 THP knob and AnonHugePages report) and the B14.8 lane
# parity cases.
#
# glm_tier_plan.cpp is the pure host arithmetic (segments, split layout and offsets, THP plans, pool modes);
# glm_tier_lane.cpp the node-split lane's half-expert math over ggml-cpu; glm_tier_numa.cu the run-time side (pinning,
# split-aware device pulls and copies, the per-node pools).  Maya-owned glm_fast_path.cu calls into them through
# one-line hooks (STRATA-EDITS.txt), so they join strata_engine on any GPU backend (a HIP build compiles src/*.cu as
# HIP); every entry point falls through to Maya's code when its feature is off.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE
    src/core/glm_tier_plan.cpp
    src/core/glm_tier_lane.cpp
    src/core/glm_tier_numa.cu)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: segment plans and the shrink-only-last rule, the split layout's offset arithmetic (split_loc / unsplit_off /
  # copy ops / chunked scatter), THP plans, buddyinfo and smaps parsing, the lane pool modes.  No strata_engine.
  add_executable(glm_tier_plan_test tests/glm_dsa/glm_tier_plan_test.cpp src/core/glm_tier_plan.cpp)
  target_include_directories(glm_tier_plan_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_tier_plan_test COMMAND glm_tier_plan_test)

  if(TARGET ggml-cpu)
    # CPU: the node-split lane (two half experts + one partial add) vs the unsplit lane at GLM-5.3's 6144 x 2048 on
    # IQ2_XXS / IQ2_S / IQ3_XXS / Q2_K experts; every RAM-tier consumer's layout path (device gather walk, promotion
    # and demotion copy ops, chunked disk scatter) against the unified blob.
    add_executable(glm_tier_lane_test tests/glm_dsa/glm_tier_lane_test.cpp src/core/glm_tier_plan.cpp
                                      src/core/glm_tier_lane.cpp)
    target_include_directories(glm_tier_lane_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_link_libraries(glm_tier_lane_test PRIVATE strata_kernels_cpu)
    add_test(NAME glm_tier_lane_test COMMAND glm_tier_lane_test)

    # CPU, B14.8: every lane kernel nt=1 vs nt>1 per row, row ranges that are not multiples of 4 / 8, and the
    # multi-token i-quant kernels vs a double reference at max-magnitude grid entries and +-127 / -128 activations.
    add_executable(glm_lane_parity_test tests/glm_dsa/glm_lane_parity_test.cpp)
    target_link_libraries(glm_lane_parity_test PRIVATE strata_kernels_cpu)
    add_test(NAME glm_lane_parity_test COMMAND glm_lane_parity_test)
  endif()

  if(TARGET strata_engine)
    # CPU (links the engine, needs no device): the pooled node-split lane (glmtier::cpu_experts on two NodeLanes) vs
    # the unsplit lane, and lane_pools' mode choice.
    add_executable(glm_tier_engine_test tests/glm_dsa/glm_tier_engine_test.cpp)
    target_include_directories(glm_tier_engine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glm_tier_engine_test PRIVATE strata_engine)
    add_test(NAME glm_tier_engine_test COMMAND glm_tier_engine_test)
  endif()

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # GPU: the split fetch / prefetch kernels and the h2d / d2h copy ops vs the unified blob (bit-exact); on one expert
    # the node-split CPU lane vs Maya's unsplit lane (max rel < 1e-5), and the lane (Q8_K activations) and the device's
    # moe_gate_up / moe_down (q8_1) each vs a double reference in its own activation format.
    add_executable(glm_tier_gpu_test tests/glm_dsa/glm_tier_gpu_test.cu)
    target_include_directories(glm_tier_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glm_tier_gpu_test PRIVATE strata_engine)
    glm_dsa_hip_links(glm_tier_gpu_test)
    add_test(NAME glm_tier_gpu_test COMMAND glm_tier_gpu_test)
    set_tests_properties(glm_tier_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
