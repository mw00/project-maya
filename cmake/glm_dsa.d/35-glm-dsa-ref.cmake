# B3.5 / B3.6: the glm-dsa F32 reference forward (Glm5Model::step_layers_dsa, the MAYA_GLM_SLOW path) and its tests.
if(TARGET strata_engine)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/core/glm_dsa_ref.cu src/core/glm_dsa_ref_host.cpp)
  else()
    # glm_model.cu dispatches glm-dsa here on every backend; without the GPU reference kernels it refuses cleanly
    target_sources(strata_engine PRIVATE src/core/glm_dsa_ref_stub.cpp)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # B3.5: the reference forward's layer math as a CPU double-accumulation forward over a tiny random glm-dsa model.
  # Plain C++ (no strata_engine), so it runs on any build.
  add_executable(glm_dsa_ref_cpu_test tests/glm_dsa/glm_dsa_ref_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
    src/core/glm_dsa_ref_host.cpp)
  target_include_directories(glm_dsa_ref_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_dsa_ref_cpu_test COMMAND glm_dsa_ref_cpu_test)

  # B3.6: SMALL / BIG / LONG synthetic glm-dsa GGUFs through the slow path vs the oracle's logits (needs a GPU and
  # the B3.1 / B3.2 tools; skips with code 77 otherwise).
  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    add_executable(glm_dsa_model_test tests/glm_dsa/glm_dsa_model_test.cpp)
    target_include_directories(glm_dsa_model_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_model_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
    target_compile_definitions(glm_dsa_model_test PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME glm_dsa_model_test COMMAND glm_dsa_model_test)
    set_tests_properties(glm_dsa_model_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 1800
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endif()
