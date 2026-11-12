# glm-dsa wave 6 (gaps): B3.7 slow-path expert size classes; the B7.4a / B7.5 / B7.6 / B12.7 / B5.3 tools are Python
# (tools/strata/*, tools/glm_dsa_mtp_probe.py) and are discovered by the Python CI job; their unittest suites are also
# registered here as CPU ctests when a python3 with numpy is found.
#
# B3.7: Maya-owned glm_model.cu calls glm_pool_merge_classes on every GPU backend, so the source joins strata_engine.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_pool_classes.cpp)
endif()

if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

# CPU: the class merge (no strata_engine)
add_executable(glm_pool_classes_test tests/glm_dsa/glm_pool_classes_test.cpp src/core/glm_pool_classes.cpp)
target_include_directories(glm_pool_classes_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME glm_pool_classes_test COMMAND glm_pool_classes_test)

# CPU: the GLM_CB_DIR dump policy (GLM_CB_APPEND / GLM_CB_ONLY, B7.6)
add_executable(glm_cb_dump_test tests/glm_dsa/glm_cb_dump_test.cpp)
target_include_directories(glm_cb_dump_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
add_test(NAME glm_cb_dump_test COMMAND glm_cb_dump_test)

# GPU: PACK9 (9 expert size classes) through MAYA_GLM_SLOW=1 with 9 / 8 / 1 classes and the bounce ring
if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  add_executable(glm_dsa_pack9_slow_test tests/glm_dsa/glm_dsa_pack9_slow_test.cpp)
  target_include_directories(glm_dsa_pack9_slow_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_dsa_pack9_slow_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  target_compile_definitions(glm_dsa_pack9_slow_test PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME glm_dsa_pack9_slow_test COMMAND glm_dsa_pack9_slow_test)
  set_tests_properties(glm_dsa_pack9_slow_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 1800
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
    set_property(TEST glm_dsa_pack9_slow_test APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
  endif()
endif()

# The Python suites of this wave as CPU ctests (the strata-tools.yml job runs them too)
if(NOT STRATA_PYTHON3)
  find_program(STRATA_PYTHON3 python3)
endif()
if(STRATA_PYTHON3)
  foreach(_py IN ITEMS tools/strata/test_parity_compare.py tools/strata/test_seam_bisect.py tools/strata/test_parity_run.py tools/strata/test_soak_run.py
                       tools/strata/test_type_coverage.py
                       tools/test_glm_dsa_slice_oracle.py)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_py}")
      get_filename_component(_name "${_py}" NAME_WE)
      add_test(NAME strata_py_${_name} COMMAND ${STRATA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/${_py}
               WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(strata_py_${_name} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
      if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
        set_property(TEST strata_py_${_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
      endif()
    endif()
  endforeach()
endif()
