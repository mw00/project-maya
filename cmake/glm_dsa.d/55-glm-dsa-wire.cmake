# glm-dsa wave 5: the Volta decode kernels wired into the glm-dsa mixer behind one dispatch policy
# (include/strata/kernels/glmd_decode_policy.hpp), FP16 index keys end to end, native Q8_0 W_kb / W_vb, the per-part
# decode CUDA graphs, and the prompt / decode cache agreement.  Sources live in fragments 45 / 50 (the mixer, the
# prompt path, the kernels); this fragment registers the tests.
if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: the policy, the struct reconciliation, the mixer through the policy twins (FP16 keys, Q8_0 W_kb / W_vb, the
  # fused / WMMA / radix kernels forced), graph extents, prompt vs decode caches and selection.  Plain C++.
  add_executable(glm_dsa_wire_cpu_test tests/glm_dsa/glm_dsa_wire_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                 src/core/glm_dsa_cpu_fast_twin.cpp src/core/glm_dsa_cpu_volta_twin.cpp src/core/glm_dsa_prompt_plan.cpp)
  target_include_directories(glm_dsa_wire_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
  add_test(NAME glm_dsa_wire_cpu_test COMMAND glm_dsa_wire_cpu_test)
endif()

# GPU (label gpu, exit 77 without a device or the fixture tools): the batched prompt and the token path write the same
# caches and decode alike on a synthetic pack, as built, with FP16 index keys, with every Volta switch forced and the
# decode graphs on, and on a pack that keeps W_kb / W_vb native Q8_0 (STRATA_PACK_NATIVE32).  (HIP: the _volta case's
# WMMA switches have no kernel there - the decode policy keeps the fused MLA and the WMMA scorer off.)
if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  add_executable(glm_dsa_prefill_decode_test tests/glm_dsa/glm_dsa_prefill_decode_test.cpp)
  target_include_directories(glm_dsa_prefill_decode_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_dsa_prefill_decode_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  target_compile_definitions(glm_dsa_prefill_decode_test PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME glm_dsa_prefill_decode_test COMMAND glm_dsa_prefill_decode_test)
  add_test(NAME glm_dsa_prefill_decode_test_fp16 COMMAND glm_dsa_prefill_decode_test)
  add_test(NAME glm_dsa_prefill_decode_test_volta COMMAND glm_dsa_prefill_decode_test)
  add_test(NAME glm_dsa_prefill_decode_test_native32 COMMAND glm_dsa_prefill_decode_test --native32)
  # IQ1_M experts (no MMQ instance: the FP16 expert products) on a geometry the batched prompt path covers; _rows
  # again with at most 100 rows a GEMM (an expert's rows in several blocks, the down's in-place order)
  add_test(NAME glm_dsa_prefill_decode_test_iq1m COMMAND glm_dsa_prefill_decode_test --iq1m)
  add_test(NAME glm_dsa_prefill_decode_test_iq1m_rows COMMAND glm_dsa_prefill_decode_test --iq1m)
  set(_pd_env "")
  if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
    set(_pd_env "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
  endif()
  foreach(_t glm_dsa_prefill_decode_test glm_dsa_prefill_decode_test_fp16 glm_dsa_prefill_decode_test_volta
             glm_dsa_prefill_decode_test_native32 glm_dsa_prefill_decode_test_iq1m glm_dsa_prefill_decode_test_iq1m_rows)
    set_tests_properties(${_t} PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 3600
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if(_pd_env)
      set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "${_pd_env}")
    endif()
  endforeach()
  set_property(TEST glm_dsa_prefill_decode_test_fp16 APPEND PROPERTY ENVIRONMENT "STRATA_GLM_IDX_FP16=1")
  set_property(TEST glm_dsa_prefill_decode_test_iq1m_rows APPEND PROPERTY ENVIRONMENT "STRATA_GLM_PREFILL_F16_ROWS=100")
  set_property(TEST glm_dsa_prefill_decode_test_volta APPEND PROPERTY ENVIRONMENT
               "STRATA_GLMD_FUSED_MLA=1;STRATA_GLMD_IDX_WMMA=1;STRATA_GLMD_TOPK_MB=1;STRATA_GLM_DSA_GRAPHS=1;STRATA_GLM_IDX_FP16=1")
endif()
