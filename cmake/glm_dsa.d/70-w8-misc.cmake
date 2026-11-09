# glm-dsa wave 8 (misc): B14.11 Walsh-Hadamard index keys, B14.15 the bucket top-k, B14.14 the NVMe prompt cache, B14.13
# n-gram drafts, B14.10 the ik_llama.cpp oracle script.  Every feature is behind a switch that defaults off.
#   B14.11 / B14.15 live in the existing glmd sources (glmd_fast.cu, glmd_batch.cu, glmd_had.cuh, the CPU twins).
#   B14.14 / B14.13: plain C++ stores and matchers (every backend) plus their engine glue.

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_prompt_cache.cpp src/core/glm_ngram_draft.cpp
                                       src/core/glm_prompt_cache_engine.cpp)
  # glm_prompt_cache_engine.cpp (every backend) calls its members; portable runtime calls only (HIP: hip_compat)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/core/glm_prompt_cache_model.cu)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # B14.15 (CPU): the bucket selector's twin == the radix selector == a sort oracle, ties and padding included.
  add_executable(glmd_topk_hist_cpu_test tests/glm_dsa/glmd_topk_hist_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                 src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp)
  target_include_directories(glmd_topk_hist_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glmd_topk_hist_cpu_test COMMAND glmd_topk_hist_cpu_test)

  # B14.13 (CPU): the n-gram matcher and the draft bookkeeping.
  add_executable(glm_ngram_draft_test tests/glm_dsa/glm_ngram_draft_test.cpp src/core/glm_ngram_draft.cpp)
  target_include_directories(glm_ngram_draft_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_ngram_draft_test COMMAND glm_ngram_draft_test)

  # B14.14 (CPU): the prompt-cache store (hashing, LRU, longest prefix, fingerprint refusal).
  add_executable(glm_prompt_cache_test tests/glm_dsa/glm_prompt_cache_test.cpp src/core/glm_prompt_cache.cpp)
  target_include_directories(glm_prompt_cache_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_prompt_cache_test COMMAND glm_prompt_cache_test)

  # B14.10 (CPU, no build of ik_llama.cpp): the oracle script parses, --help works, the dumper compiles against ik's
  # headers when the clone is there (STRATA_IK_SRC, default /home/user/ik_llama.cpp; skipped otherwise).
  set(STRATA_IK_SRC "$ENV{STRATA_IK_SRC}" CACHE PATH "an ik_llama.cpp source tree for the B14.10 oracle checks")
  add_test(NAME strata_ik_oracle_help
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/strata/ik_oracle.sh selftest)
  set_tests_properties(strata_ik_oracle_help PROPERTIES SKIP_RETURN_CODE 77
                       ENVIRONMENT "STRATA_IK_SRC=${STRATA_IK_SRC};CXX=${CMAKE_CXX_COMPILER}")

  if(TARGET strata_engine)
    # B14.11 (CPU): the rotation, score preservation, FP16 outlier keys, the mixer through the twins, the layout.
    add_executable(glm_idx_hadamard_cpu_test tests/glm_dsa/glm_idx_hadamard_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                   src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp
                   src/core/glm_state_layout.cpp src/core/glm_dsa_split.cpp)
    target_include_directories(glm_idx_hadamard_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/core ${_strata_gpu_include_directories})
    if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h (via glm_model.hpp) needs the HIP headers
      target_link_libraries(glm_idx_hadamard_cpu_test PRIVATE ${_strata_gpu_runtime_target})
    endif()
    add_test(NAME glm_idx_hadamard_cpu_test COMMAND glm_idx_hadamard_cpu_test)
  endif()

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # GPU (label gpu, exit 77 without a device): topk_hist and the rotated prep / rope against the twins.
    add_executable(glmd_w8_parity tests/glm_dsa/glmd_w8_parity.cu)
    target_link_libraries(glmd_w8_parity PRIVATE strata_engine)
    glm_dsa_hip_links(glmd_w8_parity)
    add_test(NAME glmd_w8_parity COMMAND glmd_w8_parity)
    set_tests_properties(glmd_w8_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
