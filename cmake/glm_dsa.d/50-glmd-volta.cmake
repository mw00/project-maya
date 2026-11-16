# B10.1-B10.8: the Volta glm-dsa decode kernels (fused sparse-gather MLA, WMMA indexer scorer, multi-block top-k,
# native Q8_0 headwise GEMV, FP16 index keys, decode CUDA-graph cache) and their CPU twins.
# The kernels join strata_engine on every GPU backend (glmd_fast.cu, registered by 40-glmd-decode-attn.cmake, calls
# glmd::headwise for Q8_0 weights); the twins are plain C++ and join it on any backend.  HIP builds compile the
# portable kernels (headwise, the CUDA-core scorer, the multi-block top-k, the graph position block) and leave out the
# two nvcuda::wmma kernels (the fused MLA and the WMMA scorer): the decode policy keeps them off there.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_dsa_cpu_volta_twin.cpp)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/kernels/cuda/glmd_volta_mla.cu src/kernels/cuda/glmd_volta_idx.cu
                                         src/kernels/cuda/glmd_graph.cu)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  # glm_dsa_cpu_fast_twin.cpp's mla2 twin runs vtwin::headwise for Q8_0 weights
  if(TARGET glmd_fast_cpu_test)
    target_sources(glmd_fast_cpu_test PRIVATE src/core/glm_dsa_cpu_volta_twin.cpp)
  endif()

  # the twins against the B3.4 CPU reference and sort oracles (plain C++, no strata_engine: runs anywhere)
  add_executable(glmd_volta_cpu_test tests/glm_dsa/glmd_volta_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                                     src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp
                                     src/core/glm_dsa_fast_plan.cpp)
  target_include_directories(glmd_volta_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glmd_volta_cpu_test COMMAND glmd_volta_cpu_test)

  # the kernels against glmd::ref_*, the simple glmd kernels and the twins (needs a GPU, exits 77 without one; HIP
  # skips the WMMA kernels' cases)
  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    add_executable(glmd_volta_parity tests/glm_dsa/glmd_volta_parity.cu)
    target_link_libraries(glmd_volta_parity PRIVATE strata_engine)
    glm_dsa_hip_links(glmd_volta_parity)
    add_test(NAME glmd_volta_parity COMMAND glmd_volta_parity)
    set_tests_properties(glmd_volta_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()

# B10.8: FP16 index keys in the state descriptor, the split planner and the slot fingerprint (CPU only)
if(STRATA_BUILD_TESTS AND TARGET strata_engine)
  add_executable(glm_idx_fp16_test tests/glm_dsa/glm_idx_fp16_test.cpp src/core/glm_state_layout.cpp
                                   src/core/glm_dsa_split.cpp)
  target_include_directories(glm_idx_fp16_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${_strata_gpu_include_directories})
  if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h (via glm_model.hpp) needs the HIP headers
    target_link_libraries(glm_idx_fp16_test PRIVATE ${_strata_gpu_runtime_target})
  endif()
  add_test(NAME glm_idx_fp16_test COMMAND glm_idx_fp16_test)
endif()

# B10.6 / B10.7: the new mv types' dots (glmd_mv_dots.cuh, compiled for the host) on ggml-quantized rows (CPU only)
if(STRATA_BUILD_TESTS AND TARGET ggml-base)
  add_executable(glmd_mv_cpu_test tests/glm_dsa/glmd_mv_cpu_test.cpp)
  target_link_libraries(glmd_mv_cpu_test PRIVATE ggml-base)
  add_test(NAME glmd_mv_cpu_test COMMAND glmd_mv_cpu_test)
endif()

# B10.6 / B10.7 on a GPU: gf::mv's new types vs the host dots, the dominant types vs ggml dequant; run as built and
# with the B10.6 switches (label gpu, skip 77)
if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine AND TARGET ggml-base)
  add_executable(glmd_mv_parity tests/glm_dsa/glmd_mv_parity.cu)
  target_link_libraries(glmd_mv_parity PRIVATE strata_engine ggml-base)
  glm_dsa_hip_links(glmd_mv_parity)
  add_test(NAME glmd_mv_parity COMMAND glmd_mv_parity)
  add_test(NAME glmd_mv_parity_sm70 COMMAND glmd_mv_parity)
  set_tests_properties(glmd_mv_parity glmd_mv_parity_sm70 PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  set_tests_properties(glmd_mv_parity_sm70 PROPERTIES ENVIRONMENT "MAYA_GLMD_MV_NW2=1;MAYA_GLMD_MV_TYPED=1")
endif()

# B10.1 / B14.4: the decode graph key rules (CPU) and one captured part replayed over tokens (label gpu)
if(STRATA_BUILD_TESTS)
  add_executable(glmd_graph_cpu_test tests/glm_dsa/glmd_graph_cpu_test.cpp)
  target_include_directories(glmd_graph_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glmd_graph_cpu_test COMMAND glmd_graph_cpu_test)
  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    add_executable(glmd_graph_test tests/glm_dsa/glmd_graph_test.cu)
    target_link_libraries(glmd_graph_test PRIVATE strata_engine)
    glm_dsa_hip_links(glmd_graph_test)
    add_test(NAME glmd_graph_test COMMAND glmd_graph_test)
    set_tests_properties(glmd_graph_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
