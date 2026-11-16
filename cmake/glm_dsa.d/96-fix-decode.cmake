# glm-dsa fix pass (decode path and CUDA kernels): the one switch parser (include/strata/core/glm_env.hpp) and the
# readers that use it.

if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

# CPU: glmenv's parse / switch / int / num rules; parse_switch and decode_policy_env, random_routing_seed, the tier
# plan's thp_mode / numa_split_requested, the state layout's IDX_FP16 / IDX_HADAMARD and PREFILL_STREAM with the
# spellings that used to flip a switch the wrong way.
if(TARGET strata_engine)
  add_executable(glm_env_test tests/glm_dsa/glm_env_test.cpp)
  target_include_directories(glm_env_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
  target_link_libraries(glm_env_test PRIVATE strata_engine)
  add_test(NAME glm_env_test COMMAND glm_env_test)
  set_tests_properties(glm_env_test PROPERTIES TIMEOUT 60)
  # B15.7: the random-routing bench mode's isolation (spec decode off, kept states marked)
  add_test(NAME glm_env_test_random_routing COMMAND glm_env_test --random-routing)
  set_tests_properties(glm_env_test_random_routing PROPERTIES TIMEOUT 60
                       ENVIRONMENT "MAYA_GLM_RANDOM_ROUTING=1;MAYA_GLM_RANDOM_ROUTING_SEED=7")
endif()
