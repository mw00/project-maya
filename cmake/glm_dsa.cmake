# the glm-dsa port build registration (B0.12).
# Every new glm-dsa source (glm-dsa engine code, glmd_* kernels) and every glm-dsa test is registered here, so the
# Maya-owned CMakeLists.txt carries a single include line. Add new files to the lists below.

# A HIP build's glm_prefill.cu (in strata_engine) reaches strata::prefill::Gemm (hipBLASLt, strata_prefill), as
# CMakeLists.txt's _strata_gpu_glm_libs says: a glm-dsa GPU test that links strata_engine alone takes strata_prefill too
# there (a CUDA build: nothing).
function(glm_dsa_hip_links tgt)
  if(STRATA_ENABLE_HIP AND TARGET strata_prefill)
    target_link_libraries(${tgt} PRIVATE strata_prefill)
  endif()
endfunction()

# glm-dsa engine sources compiled into strata_engine on every GPU backend (a HIP build compiles the .cu files as HIP,
# cmake/hip_backend.cmake).
set(GLM_DSA_ENGINE_SOURCES
  src/kernels/cuda/glmd_stub.cu
  src/kernels/cuda/glmd_ref.cu
  src/core/glm_dsa_cpu_ref.cpp
)

if((STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND TARGET strata_engine)
  target_sources(strata_engine PRIVATE ${GLM_DSA_ENGINE_SOURCES})
endif()

if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_dsa/glm_dsa_tests.cmake")
  enable_testing()
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_dsa/glm_dsa_tests.cmake)
endif()

# glm-dsa host-only engine sources (B1.2-B1.4): the glm-dsa geometry reader, the tensor-shape check and the loader
# hooks. Maya-owned engine files call into them on every GPU backend, so they join strata_engine whenever it exists.
set(GLM_DSA_ENGINE_HOST_SOURCES
  src/core/glm_dsa_geometry.cpp
  src/core/glm_geometry_check.cpp
)

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE ${GLM_DSA_ENGINE_HOST_SOURCES})
endif()

# Per-feature registration fragments (one file per feature, so parallel work never edits the same list):
# each cmake/glm_dsa.d/*.cmake may call target_sources(strata_engine ...) and add tests.
file(GLOB _glm_dsa_fragments CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/glm_dsa.d/*.cmake")
list(SORT _glm_dsa_fragments)
foreach(_f IN LISTS _glm_dsa_fragments)
  include(${_f})
endforeach()
