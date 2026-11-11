# glm-dsa wave 5: MTP on 3 GPUs (B11.2 head grouping and pricing, the glm-dsa rollback, B11.3 acceptance logging)
# and the prefill expert streaming driver (B9.9, with the B14.7 lane-first policy and the node-split host share).
#
#   src/core/glm_dsa_spec.cpp            host: head-group pricing, MAYA_GLM_SPEC_HEAD, acceptance stats, rollback rows
#   src/core/glm_dsa_prefill_stream_plan.cpp  host: the early prestage schedule and registry, the landing ring's node
#   src/core/glm_dsa_prefill_share.cpp   host (ggml-cpu): one node's share of a node-split layer's prompt experts
#   src/core/glm_dsa_prefill_stream.cpp  CUDA runtime: the next parts' early prestage (a Glm5Model member), the
#                                        NUMA-bound landing ring, the node-split host share on the node lanes
# Maya-owned glm_fast_path.cu, glm_model.cu and glm_prefill.cu call them through one-line hooks (STRATA-EDITS.txt),
# so they join strata_engine on any GPU backend.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_dsa_spec.cpp src/core/glm_dsa_prefill_stream_plan.cpp
                                       src/core/glm_dsa_prefill_share.cpp src/core/glm_dsa_prefill_stream.cpp)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: head pricing and choice, the head rule (Flash = Maya's), acceptance accounting, GLM-5.3 rollback rows.
  add_executable(glm_dsa_spec_test tests/glm_dsa/glm_dsa_spec_test.cpp src/core/glm_dsa_spec.cpp
                                   src/core/glm_state_layout.cpp)
  target_include_directories(glm_dsa_spec_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${_strata_gpu_include_directories})
  if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h (via glm_model.hpp) needs the HIP headers
    target_link_libraries(glm_dsa_spec_test PRIVATE ${_strata_gpu_runtime_target})
  endif()
  add_test(NAME glm_dsa_spec_test COMMAND glm_dsa_spec_test)
  # ... and with the head tuner's no-drafting candidate on (S3)
  add_test(NAME glm_dsa_spec_test_plain COMMAND glm_dsa_spec_test)
  set_tests_properties(glm_dsa_spec_test_plain PROPERTIES ENVIRONMENT "MAYA_GLM_SPEC_TUNE_PLAIN=1")

  # CPU: a rejected draft on the mixer twins - only row p written, the redo bit-identical, no stale selection.
  add_executable(glm_dsa_spec_rollback_test tests/glm_dsa/glm_dsa_spec_rollback_test.cpp src/core/glm_dsa_cpu_ref.cpp
                                            src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp)
  target_include_directories(glm_dsa_spec_rollback_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
  add_test(NAME glm_dsa_spec_rollback_test COMMAND glm_dsa_spec_rollback_test)

  if(TARGET ggml-cpu)
    # CPU: the B9.9 switches, early prestage schedule and registry, the landing ring's node, and the node-split host
    # share (two nodes' halves + one add) vs Maya's unsplit multi-row share on GLM-5.3 experts.
    add_executable(glm_dsa_prefill_stream_test tests/glm_dsa/glm_dsa_prefill_stream_test.cpp
                                               src/core/glm_dsa_prefill_stream_plan.cpp src/core/glm_dsa_prefill_share.cpp
                                               src/core/glm_tier_plan.cpp src/core/glm_tier_lane.cpp
                                               src/core/glm_dsa_prompt_plan.cpp src/core/glm_dsa_cpu_fast_twin.cpp
                                               src/core/glm_dsa_cpu_volta_twin.cpp)
    target_include_directories(glm_dsa_prefill_stream_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_link_libraries(glm_dsa_prefill_stream_test PRIVATE strata_kernels_cpu)
    add_test(NAME glm_dsa_prefill_stream_test COMMAND glm_dsa_prefill_stream_test)
  endif()

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # GPU: the rollback checks on the device decode kernels (MID geometry, FP16 / INT8 latents).
    add_executable(glm_dsa_spec_gpu_test tests/glm_dsa/glm_dsa_spec_gpu_test.cu)
    target_include_directories(glm_dsa_spec_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_spec_gpu_test PRIVATE strata_engine)
    glm_dsa_hip_links(glm_dsa_spec_gpu_test)
    add_test(NAME glm_dsa_spec_gpu_test COMMAND glm_dsa_spec_gpu_test)
    set_tests_properties(glm_dsa_spec_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
    # GPU: the B9.9 landing ring (NUMA-bound on glm-dsa, cudaHostAlloc on Flash) is DMA-able and frees cleanly.
    add_executable(glm_dsa_prefill_stream_gpu_test tests/glm_dsa/glm_dsa_prefill_stream_gpu_test.cu)
    target_include_directories(glm_dsa_prefill_stream_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_prefill_stream_gpu_test PRIVATE strata_engine)
    glm_dsa_hip_links(glm_dsa_prefill_stream_gpu_test)
    add_test(NAME glm_dsa_prefill_stream_gpu_test COMMAND glm_dsa_prefill_stream_gpu_test)
    set_tests_properties(glm_dsa_prefill_stream_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
