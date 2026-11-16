# glm-dsa wave 3: the glm-dsa router at 256 x 8 (B5.1) and the 3-GPU split on full-indexer boundaries (B6.1-B6.4).
#
# glm_dsa_split.cpp holds the split's pure host-side rules (state bytes, boundary snapping, planner terms, CPU-lane
# groups); Maya-owned glm_model.cu and glm_fast_path.cu call it on every GPU backend, so it joins strata_engine
# whenever that exists.

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_dsa_split.cpp)
endif()

if(STRATA_BUILD_TESTS)
  # B5.1: the router's CPU twin vs the F64 HF oracle (plain C++, no engine: runs on any build).
  add_executable(glm_dsa_router_cpu_test tests/glm_dsa/glm_dsa_router_cpu_test.cpp)
  target_include_directories(glm_dsa_router_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_dsa_router_cpu_test COMMAND glm_dsa_router_cpu_test)

  if(TARGET strata_engine)
    # B6.1-B6.3: the split's rules on a GLM-5.3 geometry (CPU only; links the engine for glm_split_bounds_ok).
    add_executable(glm_dsa_split_test tests/glm_dsa/glm_dsa_split_test.cpp)
    target_include_directories(glm_dsa_split_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_split_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
    add_test(NAME glm_dsa_split_test COMMAND glm_dsa_split_test)
  endif()

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # B5.1: moe_route at 256 x 8 vs the oracle on 10^5 rows, the shared expert's unclamped SwiGLU (needs a GPU).
    add_executable(glm_dsa_router_test tests/glm_dsa/glm_dsa_router_test.cpp)
    target_include_directories(glm_dsa_router_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_router_test PRIVATE strata_engine)
    glm_dsa_hip_links(glm_dsa_router_test)
    add_test(NAME glm_dsa_router_test COMMAND glm_dsa_router_test)
    set_tests_properties(glm_dsa_router_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)

    # B6.4: 1 part vs 3 parts on a synthetic glm-dsa pack (skips until the pack fixture and the forward exist).
    add_executable(glm_dsa_pack_fast_test tests/glm_dsa/glm_dsa_pack_fast_test.cpp)
    target_include_directories(glm_dsa_pack_fast_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_pack_fast_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
    add_test(NAME glm_dsa_pack_fast_test COMMAND glm_dsa_pack_fast_test)
    set_tests_properties(glm_dsa_pack_fast_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
