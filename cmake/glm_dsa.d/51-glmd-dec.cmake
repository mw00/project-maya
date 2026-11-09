# L1-4: the glm-dsa decode attention shaped for the GPU (glmd_dec.cu: mla2 calls it; on HIP it builds for every arch
# and runs where glmd_dec.cu's device policy takes it - the archs it was measured faster on, S7b) and the decode
# mixer's bench / gate at GLM-5.3's shapes (label gpu, exits 77 without a GPU; a HIP build runs all of it but the
# fused attention, which is CUDA's nvcuda::wmma).
if(TARGET strata_engine AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
  target_sources(strata_engine PRIVATE src/kernels/cuda/glmd_dec.cu)
endif()

if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine AND TARGET ggml-base)
  add_executable(glmd_dec_bench tests/glm_dsa/glmd_dec_bench.cu)
  target_link_libraries(glmd_dec_bench PRIVATE strata_engine ggml-base)
  if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h and STRATA_USE_HIP
    target_include_directories(glmd_dec_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glmd_dec_bench PRIVATE ${_strata_gpu_runtime_target})
  endif()
  add_test(NAME glmd_dec_bench COMMAND glmd_dec_bench 8)
  set_tests_properties(glmd_dec_bench PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()
