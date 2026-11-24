# Tests that link maya_engine alone: the engine's objects they use now pull the glm-dsa prompt path's object in,
# whose GEMMs are maya_prefill's (on HIP: Gemm::f16) - maya_prefill after maya_engine, as the engines link (#355)
foreach(_t file_expert_source_test expert_cache_lend_test glm_env_test glm_maya_sync_test glm_tier_engine_test
           glm_tier_gpu_test glmd_dec_bench hip_expert_cache_staging hip_glm_expert_helper)
  if(TARGET ${_t} AND TARGET maya_prefill)
    target_link_libraries(${_t} PRIVATE maya_prefill)
  endif()
endforeach()
