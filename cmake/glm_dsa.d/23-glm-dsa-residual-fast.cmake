# B2.3 plain residual seams (src/core/glm_residual.cu) and B2.4 glm-dsa fast binder + arena (src/core/glm_dsa_fast.cu,
# host half src/core/glm_dsa_fast_plan.cpp).
#
# Maya-owned src/core/glm_fast_path.cu calls into these (one-line hooks, STRATA-EDITS.txt), and on a HIP build every
# src/*.cu compiles as HIP (cmake/hip_backend.cmake), so they join strata_engine on any GPU backend: they use only
# portable runtime calls (no cuda_fp16, no warp intrinsics) and do nothing on glm5-next.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE
    src/core/glm_residual.cu
    src/core/glm_dsa_fast.cu
    src/core/glm_dsa_fast_plan.cpp)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: weight resolution order, arena arithmetic at GLM-5.3 / 64K ctx, the q8_1 host twins.  No strata_engine.
  add_executable(glmd_fast_plan_test tests/glm_dsa/glmd_fast_plan_test.cpp src/core/glm_dsa_fast_plan.cpp)
  target_include_directories(glmd_fast_plan_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
  add_test(NAME glmd_fast_plan_test COMMAND glmd_fast_plan_test)

  # GPU: res_read / res_close / res_embed / res_head vs a host double reference at n_embd 128, 4096, 6144; the
  # q8_1 fallback dequantizer and glmdsa::mv with BF16 / F32 weights against the native path.
  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    add_executable(glmd_residual_gpu_test tests/glm_dsa/glmd_residual_gpu_test.cu)
    target_include_directories(glmd_residual_gpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                               ${_strata_gpu_include_directories})
    target_link_libraries(glmd_residual_gpu_test PRIVATE strata_engine)
    glm_dsa_hip_links(glmd_residual_gpu_test)
    add_test(NAME glmd_residual_gpu_test COMMAND glmd_residual_gpu_test)
    set_tests_properties(glmd_residual_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
