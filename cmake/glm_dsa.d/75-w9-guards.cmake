# glm-dsa wave 9 (guards): B15.1 the FP16 activation range guard of the prompt path, B15.2 the MMQ read-past guard of
# the expert banks, B15.5 thread pools from the affinity mask, B15.6 the CUDA-graph generation, B15.7 the
# random-routing bench mode.
#   src/core/glm_guards_host.cpp      plain C++: MAYA_GLM_F16_GUARD, the guard's CPU twins, the bank gap ranges, the
#                                     graph generation (every backend; the CPU tests link it directly)
#   src/kernels/cuda/glmd_guards.cu   the guard kernels and binding table, mmq_malloc / mmq_zero_gaps; joins on any GPU
#                                     backend because Maya-owned glm_prefill.cu / glm_fast_path.cu call it (HIP: the
#                                     plain fallbacks)
#   header-only: glmd_random_route.hpp (the route kernel's hook), glm_cpu_affinity.hpp

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_guards_host.cpp src/kernels/cuda/glmd_guards.cu)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # B15.1 (CPU): amax 1e5 rows through the dense, shared-expert and routed-down paths (twins); bit-identity without
  # outliers; the scale rule.
  add_executable(glm_f16_guard_cpu_test tests/glm_dsa/glm_f16_guard_cpu_test.cpp src/core/glm_guards_host.cpp)
  target_include_directories(glm_f16_guard_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_f16_guard_cpu_test COMMAND glm_f16_guard_cpu_test)

  # B15.2 (CPU): the padding arithmetic and a host model of the staging ring / prestage buffer over NaN memory.
  add_executable(glm_mmq_pad_cpu_test tests/glm_dsa/glm_mmq_pad_cpu_test.cpp src/core/glm_guards_host.cpp)
  target_include_directories(glm_mmq_pad_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_mmq_pad_cpu_test COMMAND glm_mmq_pad_cpu_test)

  # B15.6 (CPU): the generation in the decode-graph key (DsaGraphs' bookkeeping over a mock capture).
  add_executable(glm_graph_gen_cpu_test tests/glm_dsa/glm_graph_gen_cpu_test.cpp src/core/glm_guards_host.cpp)
  target_include_directories(glm_graph_gen_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  find_package(Threads REQUIRED)
  target_link_libraries(glm_graph_gen_cpu_test PRIVATE Threads::Threads)
  add_test(NAME glm_graph_gen_cpu_test COMMAND glm_graph_gen_cpu_test)

  # B15.7 (CPU): the random-routing id generator (distinct, deterministic, uniform, the weights kept).
  add_executable(glm_random_route_cpu_test tests/glm_dsa/glm_random_route_cpu_test.cpp)
  target_include_directories(glm_random_route_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_random_route_cpu_test COMMAND glm_random_route_cpu_test)

  # B15.5 (CPU): mask_cpus() follows a narrowed affinity mask (numactl / taskset), hardware_concurrency() does not.
  add_executable(glm_cpu_affinity_test tests/glm_dsa/glm_cpu_affinity_test.cpp)
  target_include_directories(glm_cpu_affinity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_cpu_affinity_test COMMAND glm_cpu_affinity_test)
  set_tests_properties(glm_cpu_affinity_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  # GPU (label gpu, exit 77 without a device): the guard kernels == the twins, cuBLAS FP16 GEMMs over amax 1e5 rows
  # finite and bit-identical elsewhere, the WMMA prompt MLA with an overflowing Q tile (CUDA: the kernel is Volta's).
  add_executable(glmd_f16_guard_gpu_test tests/glm_dsa/glmd_f16_guard_gpu_test.cu)
  target_include_directories(glmd_f16_guard_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(STRATA_ENABLE_CUDA)
    target_link_libraries(glmd_f16_guard_gpu_test PRIVATE strata_engine CUDA::cublas)
  else()
    target_link_libraries(glmd_f16_guard_gpu_test PRIVATE strata_engine ${_strata_gpu_blas_target})
  endif()
  glm_dsa_hip_links(glmd_f16_guard_gpu_test)
  add_test(NAME glmd_f16_guard_gpu_test COMMAND glmd_f16_guard_gpu_test)
  set_tests_properties(glmd_f16_guard_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)

  # GPU: a decode-graph cache over an arena resized mid-session re-captures (B15.6).
  add_executable(glmd_graph_gen_gpu_test tests/glm_dsa/glmd_graph_gen_gpu_test.cu)
  target_include_directories(glmd_graph_gen_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                             ${_strata_gpu_include_directories})
  target_link_libraries(glmd_graph_gen_gpu_test PRIVATE strata_engine)
  glm_dsa_hip_links(glmd_graph_gen_gpu_test)
  add_test(NAME glmd_graph_gen_gpu_test COMMAND glmd_graph_gen_gpu_test)
  set_tests_properties(glmd_graph_gen_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()

if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_mmq AND TARGET strata_engine)
  # GPU: MMQ over a bank whose bytes past the last expert's 512-B zero tail, and whose slot gaps before mmq_zero_gaps,
  # are NaN: finite and equal to the clean bank (B15.2).
  add_executable(glmd_mmq_pad_gpu_test tests/glm_dsa/glmd_mmq_pad_gpu_test.cu)
  target_link_libraries(glmd_mmq_pad_gpu_test PRIVATE strata_mmq strata_engine ggml-base)
  glm_dsa_hip_links(glmd_mmq_pad_gpu_test)
  add_test(NAME glmd_mmq_pad_gpu_test COMMAND glmd_mmq_pad_gpu_test)
  set_tests_properties(glmd_mmq_pad_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()
