# glm-dsa wave 7 (build / tests / docs review fixes):
#   glm_prefill_types_test   CPU: the glm-dsa prompt's extra FP16 types (src/core/glm_prefill_types.hpp: F16, Q4_1,
#                            Q5_1, Q2_K) against ggml's to_float
#   strata_py_test_prefill_types         the prompt's FP16 type lists vs the pack and tools/strata/type_coverage.py
#   strata_py_test_expert_parity_shards  the sharded native_expert_parity_real driver (B5.3 / B13.1)
#   strata_py_test_pack_test_gate        the glm_pack_test_real logits gate (B7.1 / B13.1)
#   strata_docs_consistency              SWITCHES.md vs the getenv calls, TESTS.md vs `ctest -N` of this build
if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

if(TARGET ggml-base)
  add_executable(glm_prefill_types_test tests/glm_dsa/glm_prefill_types_test.cpp)
  target_link_libraries(glm_prefill_types_test PRIVATE ggml-base)
  add_test(NAME glm_prefill_types_test COMMAND glm_prefill_types_test)
endif()

if(NOT STRATA_PYTHON3)
  find_program(STRATA_PYTHON3 python3)
endif()
if(STRATA_PYTHON3)
  foreach(_py IN ITEMS tools/strata/test_prefill_types.py tools/strata/test_expert_parity_shards.py
                       tools/strata/test_pack_test_gate.py)
    get_filename_component(_name "${_py}" NAME_WE)
    add_test(NAME strata_py_${_name} COMMAND ${STRATA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/${_py}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(strata_py_${_name} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
      set_property(TEST strata_py_${_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
    endif()
  endforeach()
endif()
