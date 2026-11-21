# cmake/ik2_hip_kernels.cmake - Inkling 2's expert activation (SITU: kActSitu in src/ds4/ds4_iq.cuh, IqCpu::act in
# src/ds4/ds4_gguf_cpu.hpp) on maya-ds4's ggml block-type kernels and the CPU lane: its tests (no model).  Included
# from CMakeLists.txt beside maya-hy4's block (docs/INKLING2_AMD.md), so _strata_gpu_* are in scope.
if(STRATA_BUILD_TESTS)
  set(_ink2_situ_sources tests/ik2_hip_situ_test.cu src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
  # SITU through the token, window and prompt forms (light experts and tiles) and the CPU lane at Inkling 2 Small's
  # shapes and types (ggml-quantized random experts) against a double-precision reference; each run of it a path the
  # environment picks: the defaults (on RDNA 3 / 4 and NVIDIA: the tensor-core tiles), the F32 tiles, no light experts
  # (every set's expert on the tiles), the tensor-core tiles of 256 rows, the CPU lane's ggml rows (not its own)
  add_executable(ik2_hip_situ_test ${_ink2_situ_sources})
  target_include_directories(ik2_hip_situ_test PRIVATE ${_strata_gpu_include_directories} include third_party/ggml)
  target_link_libraries(ik2_hip_situ_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
  target_compile_definitions(ik2_hip_situ_test PRIVATE MAYA_DS4_IQ_ONLY)
  add_test(NAME ik2_hip_situ_test COMMAND ik2_hip_situ_test)
  add_test(NAME ik2_hip_situ_test_f32_tiles COMMAND ik2_hip_situ_test)
  set_tests_properties(ik2_hip_situ_test_f32_tiles PROPERTIES ENVIRONMENT MAYA_DS4_NO_TC=1)
  add_test(NAME ik2_hip_situ_test_no_light COMMAND ik2_hip_situ_test)
  set_tests_properties(ik2_hip_situ_test_no_light PROPERTIES ENVIRONMENT MAYA_DS4_NO_LIGHT=1)
  add_test(NAME ik2_hip_situ_test_tc_rm4 COMMAND ik2_hip_situ_test)
  set_tests_properties(ik2_hip_situ_test_tc_rm4 PROPERTIES ENVIRONMENT DS4_IQ_TC_RM=4)
  add_test(NAME ik2_hip_situ_test_ggml_rows COMMAND ik2_hip_situ_test)
  set_tests_properties(ik2_hip_situ_test_ggml_rows PROPERTIES ENVIRONMENT DS4_IQ_OWN=0)
  set_tests_properties(ik2_hip_situ_test ik2_hip_situ_test_f32_tiles ik2_hip_situ_test_no_light
                       ik2_hip_situ_test_tc_rm4 ik2_hip_situ_test_ggml_rows PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
  # the existing activations (SwiGLU, swigluoai) through every form and type, bit for bit across a change to ds4_iq:
  # the same file with IK2_HIP_SITU_LEGACY_ONLY (ds4_iq's API before kActSitu) - `ik2_hip_situ_dump --dump F` on the
  # sources before the change, `--check F` after (built by name)
  add_executable(ik2_hip_situ_dump EXCLUDE_FROM_ALL ${_ink2_situ_sources})
  target_include_directories(ik2_hip_situ_dump PRIVATE ${_strata_gpu_include_directories} include third_party/ggml)
  target_link_libraries(ik2_hip_situ_dump PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} Threads::Threads)
  target_compile_definitions(ik2_hip_situ_dump PRIVATE MAYA_DS4_IQ_ONLY IK2_HIP_SITU_LEGACY_ONLY)
endif()
