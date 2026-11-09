# B4.1 / B4.2: the per-layer state layout descriptor (glm5-next and glm-dsa) and its Glm5Model binding.
# Host-only sources: Maya-owned engine files (load, load_pack, the split search, seq_layout, the slots) call them on
# every GPU backend, so they join strata_engine whenever it exists.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE
    src/core/glm_state_layout.cpp
    src/core/glm_state_bind.cpp
  )
endif()

if(STRATA_BUILD_TESTS)
  # CPU only: the descriptor against Maya's old carver arithmetic (Flash) and the GLM-5.3 formula (glm-dsa).
  if(TARGET strata_engine)
    add_executable(glm_state_layout_test tests/glm_dsa/glm_state_layout_test.cpp src/core/glm_state_layout.cpp)
    target_include_directories(glm_state_layout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${_strata_gpu_include_directories})
    if(STRATA_ENABLE_HIP)   # hip_compat's cuda_runtime.h (via glm_model.hpp) needs the HIP headers
      target_link_libraries(glm_state_layout_test PRIVATE ${_strata_gpu_runtime_target})
    endif()
    add_test(NAME glm_state_layout_test COMMAND glm_state_layout_test)
  endif()
  # GPU: a descriptor-laid arena's slot runs round-trip through the host (skips with 77 without a device).
  if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
    add_executable(glm_state_slot_gpu_test tests/glm_dsa/glm_state_slot_gpu_test.cpp)
    target_include_directories(glm_state_slot_gpu_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_state_slot_gpu_test PRIVATE strata_engine)
    glm_dsa_hip_links(glm_state_slot_gpu_test)
    add_test(NAME glm_state_slot_gpu_test COMMAND glm_state_slot_gpu_test)
    set_tests_properties(glm_state_slot_gpu_test PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
