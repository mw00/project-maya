# glm-dsa EXL3 (1/3): the EXL3 kernels at GLM-5.3's shapes on synthetic tensors - random trellis words and FP16
# scales packed by the test, no model, no quant - against the scalar CPU reference (exl3_ref) and the CPU lane's
# kernels (exl3_cpu): tests/glm_dsa/glm_dsa_exl3_test.cu.  NVIDIA (CUDA) and AMD (HIP) builds alike, label gpu (exit
# 77 without a device).  Besides the default run, the sections an environment switch changes run again with it:
#   CUDA  STRATA_GLM_EXL3_DECODE=new / =legacy: mv's two kernel families (before sm_80 v1.1.8's by default, exl3.cu's
#         from sm_80); STRATA_EXL3_WMMA=0: sm_7x's portable prompt-path expert kernels instead of the WMMA ones
#   HIP   DS4_EXL3_MV1=1 / =0: the one-activation GEMV and the generic one; EXL3_WMMA=0: the portable prompt-path
#         expert kernels instead of the matrix-core ones (gfx11's default)
#   both  STRATA_FORCE_AVX2=1: the CPU lane's AVX2 tier on an AVX-512 CPU, for the calls that take the lane's tier
#         (no GPU used)
if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_kernels AND
   TARGET strata_kernels_cpu)
  enable_testing()
  add_executable(glm_dsa_exl3_test tests/glm_dsa/glm_dsa_exl3_test.cu)
  target_link_libraries(glm_dsa_exl3_test PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target}
                        ${_strata_gpu_blas_target})
  add_test(NAME glm_dsa_exl3_test COMMAND glm_dsa_exl3_test)
  set(_glm_dsa_exl3_gpu_tests glm_dsa_exl3_test)
  if(STRATA_ENABLE_HIP)
    add_test(NAME glm_dsa_exl3_mv1
             COMMAND ${CMAKE_COMMAND} -E env DS4_EXL3_MV1=1 $<TARGET_FILE:glm_dsa_exl3_test> --only mv,batch,head)
    add_test(NAME glm_dsa_exl3_mv_generic
             COMMAND ${CMAKE_COMMAND} -E env DS4_EXL3_MV1=0 $<TARGET_FILE:glm_dsa_exl3_test> --only mv,batch,head)
    add_test(NAME glm_dsa_exl3_wmma_off
             COMMAND ${CMAKE_COMMAND} -E env EXL3_WMMA=0 $<TARGET_FILE:glm_dsa_exl3_test> --only set)
    list(APPEND _glm_dsa_exl3_gpu_tests glm_dsa_exl3_mv1 glm_dsa_exl3_mv_generic glm_dsa_exl3_wmma_off)
  else()
    add_test(NAME glm_dsa_exl3_decode_new
             COMMAND ${CMAKE_COMMAND} -E env STRATA_GLM_EXL3_DECODE=new $<TARGET_FILE:glm_dsa_exl3_test>
                     --only mv,batch,head)
    add_test(NAME glm_dsa_exl3_decode_legacy
             COMMAND ${CMAKE_COMMAND} -E env STRATA_GLM_EXL3_DECODE=legacy $<TARGET_FILE:glm_dsa_exl3_test>
                     --only mv,batch,head)
    add_test(NAME glm_dsa_exl3_wmma_off
             COMMAND ${CMAKE_COMMAND} -E env STRATA_EXL3_WMMA=0 $<TARGET_FILE:glm_dsa_exl3_test> --only set)
    list(APPEND _glm_dsa_exl3_gpu_tests glm_dsa_exl3_decode_new glm_dsa_exl3_decode_legacy glm_dsa_exl3_wmma_off)
  endif()
  set_tests_properties(${_glm_dsa_exl3_gpu_tests} PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77)
  add_test(NAME glm_dsa_exl3_cpu_avx2
           COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_AVX2=1 $<TARGET_FILE:glm_dsa_exl3_test> --only cpu)
  set_tests_properties(glm_dsa_exl3_cpu_avx2 PROPERTIES TIMEOUT 300)
endif()
