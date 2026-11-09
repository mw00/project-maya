# glm-dsa glm-dsa tests (B0.12). Included from cmake/strata.cmake when STRATA_BUILD_TESTS is on.
# GPU tests carry the label "gpu"; tests needing the real GLM-5.3 pack carry "realpack" and skip themselves
# unless STRATA_TEST_PACK is set.
if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  add_executable(glmd_stub_test tests/glm_dsa/glmd_stub_test.cpp)
  target_link_libraries(glmd_stub_test PRIVATE strata_engine)
  glm_dsa_hip_links(glmd_stub_test)
  add_test(NAME glmd_stub_test COMMAND glmd_stub_test)
endif()

# B1.10: glm-dsa loader tests (CPU only: synthetic GGUF headers written by the test, no weights read, no GPU needed).
if(TARGET strata_engine)
  add_executable(glm_dsa_loader_test tests/glm_dsa/glm_dsa_loader_test.cpp)
  target_include_directories(glm_dsa_loader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_dsa_loader_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  add_test(NAME glm_dsa_loader_test COMMAND glm_dsa_loader_test)
endif()

# B3.4: the glm-dsa CPU reference against first principles.  Plain C++ (no strata_engine), so it runs on any build.
add_executable(glmd_ref_cpu_test tests/glm_dsa/glmd_ref_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp)
target_include_directories(glmd_ref_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME glmd_ref_cpu_test COMMAND glmd_ref_cpu_test)

# B3.4: the F32 reference kernels against the CPU reference (needs a GPU).
if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  add_executable(glmd_ref_parity tests/glm_dsa/glmd_ref_parity.cu)
  target_link_libraries(glmd_ref_parity PRIVATE strata_engine)
  glm_dsa_hip_links(glmd_ref_parity)
  add_test(NAME glmd_ref_parity COMMAND glmd_ref_parity)
  set_tests_properties(glmd_ref_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()

# B2.1 / B5.2: the routing ring's width limits and the CPU lane's native experts at n_embd 6144 (CPU only, runs in CI).
if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_kernels_cpu AND STRATA_NATIVE_EXPERTS)
  add_executable(glmd_width_cpu_test tests/glm_dsa/glmd_width_cpu_test.cpp)
  target_include_directories(glmd_width_cpu_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glmd_width_cpu_test PRIVATE strata_kernels_cpu)
  if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h needs the HIP headers
    target_link_libraries(glmd_width_cpu_test PRIVATE ${_strata_gpu_runtime_target})
  endif()
  add_test(NAME glmd_width_cpu_test COMMAND glmd_width_cpu_test)
endif()

# B2.1 / B2.2 / B2.5: norm kernels at 6144 / 8192 vs F64, refused shapes counted, the 6144-wide ring canary.
if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  add_executable(glmd_width_gpu_test tests/glm_dsa/glmd_width_gpu_test.cpp)
  target_include_directories(glmd_width_gpu_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glmd_width_gpu_test PRIVATE strata_engine)
  glm_dsa_hip_links(glmd_width_gpu_test)
  add_test(NAME glmd_width_gpu_test COMMAND glmd_width_gpu_test)
  set_tests_properties(glmd_width_gpu_test PROPERTIES LABELS gpu)
endif()
