# B4.3-B4.5: glm-dsa decode attention kernels (dsa_prep, dsa_rope_q, mla2, the top-2048 selector) and their CPU twins.
# The kernels join maya_engine on every GPU backend (HIP compiles them through hip_compat); the twins are plain C++
# and join it on any backend.
if(TARGET maya_engine)
  target_sources(maya_engine PRIVATE src/core/glm_dsa_cpu_fast_twin.cpp)
  if(MAYA_ENABLE_CUDA OR MAYA_ENABLE_HIP)
    target_sources(maya_engine PRIVATE src/kernels/cuda/glmd_fast.cu)
  endif()
endif()

if(MAYA_BUILD_TESTS)
  # The twins against the B3.4 CPU reference and a sort-based top-k oracle (plain C++, no maya_engine: runs anywhere).
  add_executable(glmd_fast_cpu_test tests/glm_dsa/glmd_fast_cpu_test.cpp src/core/glm_dsa_cpu_ref.cpp
                                    src/core/glm_dsa_cpu_fast_twin.cpp)
  target_include_directories(glmd_fast_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glmd_fast_cpu_test COMMAND glmd_fast_cpu_test)

  # The kernels against glmd::ref_* and the twins (needs a GPU, exits 77 without one).
  if((MAYA_ENABLE_CUDA OR MAYA_ENABLE_HIP) AND TARGET maya_engine)
    add_executable(glmd_fast_parity tests/glm_dsa/glmd_fast_parity.cu)
    target_link_libraries(glmd_fast_parity PRIVATE maya_engine)
    glm_dsa_hip_links(glmd_fast_parity)
    add_test(NAME glmd_fast_parity COMMAND glmd_fast_parity)
    set_tests_properties(glmd_fast_parity PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endif()
endif()
