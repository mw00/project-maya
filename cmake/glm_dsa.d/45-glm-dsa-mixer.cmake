# B4.6 / B4.7 / B11.1 / B14.3-B14.5: the glm-dsa decode attention mixer (Glm5Model::fast_dsa_glm), its kernels
# (glmd::idx_score, f32_to_bf16, iota) and the token-by-token prompt with the NextN cache fill.
#
# Maya-owned glm_fast_path.cu (fast_dsa) and glm_prefill.cu (prefill) dispatch into these on every backend, so an
# engine without a GPU backend gets the refusing stub instead of a link error.  CUDA and HIP get the real mixer.
if(TARGET strata_engine)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/core/glm_dsa_mixer.cu src/kernels/cuda/glmd_mixer.cu)
  else()
    target_sources(strata_engine PRIVATE src/core/glm_dsa_mixer_stub.cpp)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: the mixer orchestration (glmdsa::mixer_layer) on the twins vs the B3.4 reference, SMALL and MID geometries,
  # the named wrong readings, the dense switch, the stale-selection refusal.  Plain C++, no strata_engine.
  add_executable(glm_dsa_mixer_cpu_test tests/glm_dsa/glm_dsa_mixer_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                                        src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp)
  target_include_directories(glm_dsa_mixer_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
  add_test(NAME glm_dsa_mixer_cpu_test COMMAND glm_dsa_mixer_cpu_test)

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # B4.7 kernel parity (GPU): the same orchestration on the device kernels vs the twins and the reference, the
    # named wrong readings, idx_score vs Maya's dsa_score, mla2's device count, and the B14.4 graph replay test.
    add_executable(glmd_kernel_parity tests/glm_dsa/glmd_kernel_parity.cu)
    target_include_directories(glmd_kernel_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glmd_kernel_parity PRIVATE strata_engine)
    glm_dsa_hip_links(glmd_kernel_parity)
    add_test(NAME glmd_kernel_parity COMMAND glmd_kernel_parity)
    set_tests_properties(glmd_kernel_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # B4.7 layer parity + B11.1 NextN (GPU): a synthetic PACK through the fast path vs the slow path per layer
    # (full+dense, full+MoE, shared+MoE) and the NextN draft vs ref/glm_dsa.py --mtp-out.
    add_executable(glmd_layer_parity tests/glm_dsa/glmd_layer_parity.cpp)
    target_include_directories(glmd_layer_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glmd_layer_parity PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
    target_compile_definitions(glmd_layer_parity PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME glmd_layer_parity COMMAND glmd_layer_parity)
    set_tests_properties(glmd_layer_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 1800
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  # B6.4: the pack fast test makes its own synthetic pack (tools/glm_dsa_synth_gguf.py + tools/iq_pack.py) and
  # finds the tools through the source tree; glmd_width_gpu_test exits 77 without a device.
  if(TEST glm_dsa_pack_fast_test)
    target_compile_definitions(glm_dsa_pack_fast_test PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(glm_dsa_pack_fast_test PROPERTIES TIMEOUT 3600 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  # B3.6 / B4.7 / B6.4: the end-to-end tests make their fixtures with the repository's Python CLIs
  # (tests/glm_dsa/glm_dsa_fixtures.hpp), which need gguf-py
  foreach(_t glm_dsa_model_test glm_dsa_pack_fast_test glmd_layer_parity)
    if(TEST ${_t} AND DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
      set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
    endif()
  endforeach()
  if(TEST glmd_width_gpu_test)
    set_tests_properties(glmd_width_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
