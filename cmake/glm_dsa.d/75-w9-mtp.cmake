# glm-dsa wave 9 (MTP and draft hygiene): B15.3 the NextN carrier's sequence state, B15.8 heat-decay expert eviction,
# B15.9 the adaptive draft depth.  Every behaviour change is behind a switch that defaults off, except B15.3's state
# hygiene, which only touches glm-dsa with a NextN block (Flash reads FastState::head_x as before).

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_mtp_state.cpp src/core/glm_heat_evict.cpp)
  # Maya-owned glm_fast_path.cu / glm_multi.cu call its members on every GPU backend (HIP: through hip_compat; a
  # no-op off glm-dsa, as on CUDA)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    target_sources(strata_engine PRIVATE src/core/glm_mtp_state_model.cu)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # B15.3 (CPU): the carrier state machine, the MTP1 trailer, the block identity and the draft budget, in a toy twin
  # of the engine's MTP data flow (first draft after a restore / reset == a fresh run; Maya's shared row differs).
  add_executable(glm_mtp_state_test tests/glm_dsa/glm_mtp_state_test.cpp src/core/glm_mtp_state.cpp)
  target_include_directories(glm_mtp_state_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_mtp_state_test COMMAND glm_mtp_state_test)

  # B15.8 (CPU): the heat table, the LRU-window victim, upload coalescing, the swap rule, and the policy simulator
  # (Maya's aged counts vs heat decay on synthetic drifting routing traces; prints the hit rates).
  add_executable(glm_heat_evict_test tests/glm_dsa/glm_heat_evict_test.cpp src/core/glm_heat_evict.cpp)
  target_include_directories(glm_heat_evict_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_heat_evict_test COMMAND glm_heat_evict_test)

  # B15.9 (CPU): the adaptive draft depth controller, the proposal tracker, rounds on code-like / prose-like streams vs
  # fixed depths, and NgramSession under MAYA_GLM_DRAFT_ADAPTIVE.
  add_executable(glm_draft_adaptive_test tests/glm_dsa/glm_draft_adaptive_test.cpp src/core/glm_ngram_draft.cpp)
  target_include_directories(glm_draft_adaptive_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_draft_adaptive_test COMMAND glm_draft_adaptive_test)
endif()

# B15.4 (Python, CPU): tools/strata/extract_mtp_block.py on synthetic PACK / SMALL GGUFs (whole and sharded), and the
# engine's GGUF reader (strata-gguf) on its output.
if(STRATA_BUILD_TESTS)
  if(NOT STRATA_PYTHON3)
    find_program(STRATA_PYTHON3 python3)
  endif()
  if(STRATA_PYTHON3)
    add_test(NAME strata_py_test_extract_mtp_block
             COMMAND ${STRATA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/strata/test_extract_mtp_block.py
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(strata_py_test_extract_mtp_block PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    set(_emb_env "")
    if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
      list(APPEND _emb_env "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
    endif()
    if(TARGET strata-gguf)
      list(APPEND _emb_env "MAYA_GGUF_BIN=$<TARGET_FILE:strata-gguf>")
    endif()
    if(_emb_env)
      set_property(TEST strata_py_test_extract_mtp_block APPEND PROPERTY ENVIRONMENT "${_emb_env}")
    endif()
  endif()
endif()
