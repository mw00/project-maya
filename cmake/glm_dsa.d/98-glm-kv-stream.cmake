# KV streaming on glm-dsa (GLM-5.3 744B): one residency map an indexer group, the identity window.
if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU only: the group assignment (glm_kv_leads) and the protocol on a host twin of the device kernels.
  add_executable(glm_kv_groups_test tests/glm_dsa/glm_kv_groups_test.cpp)
  target_include_directories(glm_kv_groups_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_kv_groups_test COMMAND glm_kv_groups_test)

  if(STRATA_ENABLE_CUDA AND TARGET strata_kernels)
    # GPU, synthetic: the group's resolve / copy launches, the identity reset and the debug check (exit 77 without a device)
    add_executable(glm_kv_group_test tests/glm_kv_group_test.cu)
    target_include_directories(glm_kv_group_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_kv_group_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME glm_kv_group_test COMMAND glm_kv_group_test)
    set_tests_properties(glm_kv_group_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()

  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    # GPU, synthetic packs: a streamed model against the resident one, bit for bit, through a whole conversation
    add_executable(glm_dsa_kv_stream_test tests/glm_dsa/glm_dsa_kv_stream_test.cpp)
    target_include_directories(glm_dsa_kv_stream_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dsa_kv_stream_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
    target_compile_definitions(glm_dsa_kv_stream_test PRIVATE STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME glm_dsa_kv_stream_test COMMAND glm_dsa_kv_stream_test)
    set_tests_properties(glm_dsa_kv_stream_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 3600
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESOURCE_LOCK glm_dsa_fixtures)
    if(DEFINED STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/gguf-py")
      set_property(TEST glm_dsa_kv_stream_test APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${STRATA_GGML_DIR}/gguf-py")
    endif()
  endif()
endif()
