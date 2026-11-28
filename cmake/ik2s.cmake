# cmake/ik2s.cmake - maya-ik2s: Thinking Machines' Inkling 2 Small (unsloth's GGUF set, packed in place by
# tools/ik2s/pack_gguf.py) on 2-4 AMD GPUs, a layer split with a CPU lane for the experts VRAM does not hold
# (docs/INKLING2_AMD.md); its drafter is the model's own DSpark block (docs/INKLING2_DSPARK.md).  Its own engine beside
# maya-ik2 (src/ik2: Inkling-2-Mini) - the two share no code.  It runs on maya-ds4's ggml block-type kernels
# (src/ds4/ds4_iq, with Inkling 2's SITU activation) and maya-hy4's small kernels and lane threads, built as maya-hy4
# builds them.  AMD under Linux only (MAYA_ENABLE_HIP); included from CMakeLists.txt inside its GPU block
# (_maya_gpu_* in scope).
# (src/ik2s/ik2_hip_iq.cu: src/ds4/ds4_iq.cu itself plus the engine's own GEMV on it, mv_lds - in its place)
set(_ik2s_shared src/hy4/hy4_kernels.cu src/hy4/hy4_lane.cpp src/ik2s/ik2_hip_iq.cu src/ds4/ds4_iq_cpu.cpp
                 src/ds4/ds4_gguf_cpu.cpp)
add_executable(maya-ik2s src/ik2s/ik2_hip_main.cpp src/ik2s/ik2_hip_model.cu src/ik2s/ik2_hip_kernels.cu
                         src/ik2s/ik2_hip_lane.cpp src/ik2s/ik2_hip_dspark.cu ${_ik2s_shared})
target_include_directories(maya-ik2s PRIVATE ${_maya_gpu_include_directories} include src/ik2s src/hy4 src/mm3
                                             third_party/ggml)
target_link_libraries(maya-ik2s PRIVATE maya_kernels_cpu maya_spec ${_maya_gpu_runtime_target} ${_maya_gpu_blas_target}
                                        Threads::Threads)
# (ds4_iq.cu as maya-mm3 and maya-hy4 take it; without DS4_IQ_MIN_FORMS, which would change the K-quants' rounding)
target_compile_definitions(maya-ik2s PRIVATE MAYA_DS4_IQ_ONLY IK2_HIP_IQ_SITU=1)
if(MAYA_BUILD_TESTS)
  # Inkling 2's expert activation (SITU: kActSitu in src/ds4/ds4_iq.cuh; the CPU lane's in src/ik2s/ik2_hip_situ_cpu.hpp)
  # through the token, window and prompt forms (light experts and tiles) and the CPU lane at Inkling 2 Small's shapes
  # and types (ggml-quantized random experts) against a double-precision reference; each run a path the environment
  # picks: the defaults (on RDNA 3 / 4: the matrix-core tiles), the F32 tiles, no light experts, the matrix-core tiles
  # of 256 rows, the CPU lane's ggml rows (not its own).  No model.
  set(_ik2s_situ_sources tests/ik2_hip_situ_test.cu src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
  add_executable(ik2s_situ_test ${_ik2s_situ_sources})
  target_include_directories(ik2s_situ_test PRIVATE ${_maya_gpu_include_directories} include third_party/ggml)
  target_link_libraries(ik2s_situ_test PRIVATE maya_kernels_cpu ${_maya_gpu_runtime_target} Threads::Threads)
  target_compile_definitions(ik2s_situ_test PRIVATE MAYA_DS4_IQ_ONLY)
  add_test(NAME ik2s_situ_test COMMAND ik2s_situ_test)
  add_test(NAME ik2s_situ_test_f32_tiles COMMAND ik2s_situ_test)
  set_tests_properties(ik2s_situ_test_f32_tiles PROPERTIES ENVIRONMENT MAYA_DS4_NO_TC=1)
  add_test(NAME ik2s_situ_test_no_light COMMAND ik2s_situ_test)
  set_tests_properties(ik2s_situ_test_no_light PROPERTIES ENVIRONMENT MAYA_DS4_NO_LIGHT=1)
  add_test(NAME ik2s_situ_test_tc_rm4 COMMAND ik2s_situ_test)
  set_tests_properties(ik2s_situ_test_tc_rm4 PROPERTIES ENVIRONMENT DS4_IQ_TC_RM=4)
  add_test(NAME ik2s_situ_test_ggml_rows COMMAND ik2s_situ_test)
  set_tests_properties(ik2s_situ_test_ggml_rows PROPERTIES ENVIRONMENT DS4_IQ_OWN=0)
  set_tests_properties(ik2s_situ_test ik2s_situ_test_f32_tiles ik2s_situ_test_no_light ik2s_situ_test_tc_rm4
                       ik2s_situ_test_ggml_rows PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 LABELS gpu)
  # the activations that were there before (SwiGLU, swigluoai) through every form and type, bit for bit across a
  # change to ds4_iq: the same file with IK2_HIP_SITU_LEGACY_ONLY (ds4_iq's API without kActSitu) -
  # `ik2s_situ_dump --dump F` built from the sources before the change, `--check F` after (built by name)
  add_executable(ik2s_situ_dump EXCLUDE_FROM_ALL ${_ik2s_situ_sources})
  target_include_directories(ik2s_situ_dump PRIVATE ${_maya_gpu_include_directories} include third_party/ggml)
  target_link_libraries(ik2s_situ_dump PRIVATE maya_kernels_cpu ${_maya_gpu_runtime_target} Threads::Threads)
  target_compile_definitions(ik2s_situ_dump PRIVATE MAYA_DS4_IQ_ONLY IK2_HIP_SITU_LEGACY_ONLY)
  # the engine's fused small kernels against the separate launches they replace, bit for bit (no model)
  add_executable(ik2s_fuse_test tests/ik2_hip_fuse_test.cu src/ik2s/ik2_hip_kernels.cu src/hy4/hy4_kernels.cu)
  target_include_directories(ik2s_fuse_test PRIVATE ${_maya_gpu_include_directories} include src/ik2s src/hy4)
  target_link_libraries(ik2s_fuse_test PRIVATE ${_maya_gpu_runtime_target})
  add_test(NAME ik2s_fuse_test COMMAND ik2s_fuse_test)
  set_tests_properties(ik2s_fuse_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 LABELS gpu)
endif()
# the drafter alone against a reference run's drafts (llama.cpp PR #32218: tools/ik2s/ref/inkling-dspark-eval.cpp) from
# its saved taps - it needs the model's files and a GPU, so it is built by name and is not a ctest
add_executable(ik2s_dspark_test EXCLUDE_FROM_ALL tests/ik2_hip_dspark_test.cpp src/ik2s/ik2_hip_dspark.cu
                                src/ik2s/ik2_hip_kernels.cu ${_ik2s_shared})
target_include_directories(ik2s_dspark_test PRIVATE ${_maya_gpu_include_directories} include src/ik2s src/hy4 src/mm3
                                                    third_party/ggml)
target_link_libraries(ik2s_dspark_test PRIVATE maya_kernels_cpu ${_maya_gpu_runtime_target} ${_maya_gpu_blas_target}
                                               Threads::Threads)
target_compile_definitions(ik2s_dspark_test PRIVATE MAYA_DS4_IQ_ONLY)
# kernel benchmarks (no model, built by name): decode attention (chunk / tiles / matrix-core kernels) and the mix
foreach(_b attn mix)
  add_executable(ik2s_${_b}_bench EXCLUDE_FROM_ALL tests/ik2_hip_${_b}_bench.cu src/ik2s/ik2_hip_kernels.cu
                                  src/hy4/hy4_kernels.cu)
  target_include_directories(ik2s_${_b}_bench PRIVATE ${_maya_gpu_include_directories} include src/ik2s src/hy4)
  target_link_libraries(ik2s_${_b}_bench PRIVATE ${_maya_gpu_runtime_target})
endforeach()
