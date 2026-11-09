# B9: the glm-dsa batched prompt path on V100.
#   src/core/glm_dsa_prompt_plan.cpp    host: tiling (B9.1, B14.6), attention walk crossover (B14.12), upload-or-lane
#                                       (B14.7), per-part upload schedule (B9.9), CPU twins of the batched kernels
#   src/kernels/cuda/glmd_batch.cu      CUDA: batched prep / rope (B9.2), score tiles + multi-row selector (B9.3),
#                                       prompt MLA F32 (B9.4) and WMMA 576 (B9.5), HF router rows, plain residual rows
#   src/core/glm_prefill_dsa.cu         the prompt mixer + NextN cache fill (B9.6), FP16 absorbed GEMMs (B9.7);
#                                       hooked from Maya-owned glm_prefill.cu (STRATA-EDITS.txt).  HIP: the same path
#                                       (hipBLAS GEMMs, the F32 prompt attention: the WMMA 576 kernel is Volta's)
#   src/core/glm_dsa_prefill_stream.cpp B9.9 / B14.7 drivers (host)
#   src/prefill/moe_mmq_strata.cu       in strata_mmq: the q8_1 MMQ block layout for the B14.1 clamp and (CUDA) the
#                                       B14.2 sm_70 instance check (sees mmq.cuh as moe_mmq.cu does)
if(TARGET strata_engine)
  # glm_prefill_dsa.cu joins on any GPU backend: Maya-owned glm_prefill.cu calls it
  target_sources(strata_engine PRIVATE src/core/glm_dsa_prompt_plan.cpp src/core/glm_prefill_dsa.cu)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/kernels/cuda/glmd_batch.cu)
  endif()
endif()
if(TARGET strata_mmq AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
  target_sources(strata_mmq PRIVATE src/prefill/moe_mmq_strata.cu)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU (runs anywhere): tiled multi-row selection == token-path selection at 2,049 / 9,000 tokens, attention walks,
  # T-row prep mapping, tiling sizes at 64K, upload-or-lane threshold, three-link schedule.
  add_executable(glmd_batch_cpu_test tests/glm_dsa/glmd_batch_cpu_test.cpp src/core/glm_dsa_prompt_plan.cpp
                                     src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp)
  target_include_directories(glmd_batch_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glmd_batch_cpu_test COMMAND glmd_batch_cpu_test)
endif()

if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  # GPU (label gpu, exit 77 without a device): batched prep == token prep bytes, tiled selection == glmd::topk on
  # every row (2,049 / 9,000 tokens), prompt MLA F32 / WMMA (CUDA) vs the twin, HF router rows, the q8_1 MMQ clamp.
  add_executable(glmd_batch_parity tests/glm_dsa/glmd_batch_parity.cu)
  target_link_libraries(glmd_batch_parity PRIVATE strata_engine)
  glm_dsa_hip_links(glmd_batch_parity)
  add_test(NAME glmd_batch_parity COMMAND glmd_batch_parity)
  set_tests_properties(glmd_batch_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  # GPU: B9.10 the prompt layer (glmpf::dsa_layer) vs a token-by-token composition of the decode kernels on a
  # synthetic two-layer model: 32 / 2,100 tokens (dense, dense-masked, gather), two chunks; caches and selections.
  add_executable(glm_dsa_prefill_test tests/glm_dsa/glm_dsa_prefill_test.cu)
  target_include_directories(glm_dsa_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core
                             ${_strata_gpu_include_directories})
  target_link_libraries(glm_dsa_prefill_test PRIVATE strata_engine)
  glm_dsa_hip_links(glm_dsa_prefill_test)
  add_test(NAME glm_dsa_prefill_test COMMAND glm_dsa_prefill_test)
  set_tests_properties(glm_dsa_prefill_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()

if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND TARGET strata_mmq)
  # CPU (runs anywhere, no device touched): B14.2 every host-picked MMQ tile width has a real sm_70 instance.
  add_executable(glmd_mmq_sm70_test tests/glm_dsa/glmd_mmq_sm70_test.cpp)
  target_link_libraries(glmd_mmq_sm70_test PRIVATE strata_mmq)
  add_test(NAME glmd_mmq_sm70_test COMMAND glmd_mmq_sm70_test)
endif()
if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_mmq)
  # GPU: grouped MMQ with ids == per-expert launches == host reference, per type (B9.8). Its host side runs on every
  # hardware thread: an estimated 2-4 min on the test PC's 80 threads at 2.0 GHz (on one thread it went past 900 s).
  add_executable(glmd_mmq_ids_test tests/glm_dsa/glmd_mmq_ids_test.cu)
  target_link_libraries(glmd_mmq_ids_test PRIVATE strata_mmq ggml-base Threads::Threads)
  if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h (HIP's strata_mmq links the bare ROCm targets)
    target_link_libraries(glmd_mmq_ids_test PRIVATE ${_strata_gpu_runtime_target})
  endif()
  add_test(NAME glmd_mmq_ids_test COMMAND glmd_mmq_ids_test)
  set_tests_properties(glmd_mmq_ids_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 900)
endif()
