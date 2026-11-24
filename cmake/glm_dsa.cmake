# the glm-dsa port build registration (B0.12).
# Every new glm-dsa source (glm-dsa engine code, glmd_* kernels) and every glm-dsa test is registered here, so the
# Maya-owned CMakeLists.txt carries a single include line. Add new files to the lists below.

# A HIP build's glm_prefill.cu (in maya_engine) reaches maya::prefill::Gemm (hipBLASLt, maya_prefill), as
# CMakeLists.txt's _maya_gpu_glm_libs says: a glm-dsa GPU test that links maya_engine alone takes maya_prefill too
# there (a CUDA build: nothing).
function(glm_dsa_hip_links tgt)
  if(MAYA_ENABLE_HIP AND TARGET maya_prefill)
    target_link_libraries(${tgt} PRIVATE maya_prefill)
  endif()
endfunction()

# glm-dsa engine sources compiled into maya_engine on every GPU backend (a HIP build compiles the .cu files as HIP,
# cmake/hip_backend.cmake).
set(GLM_DSA_ENGINE_SOURCES
  src/kernels/cuda/glmd_stub.cu
  src/kernels/cuda/glmd_ref.cu
  src/core/glm_dsa_cpu_ref.cpp
)

if((MAYA_ENABLE_CUDA OR MAYA_ENABLE_HIP) AND TARGET maya_engine)
  target_sources(maya_engine PRIVATE ${GLM_DSA_ENGINE_SOURCES})
endif()

if(MAYA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_dsa/glm_dsa_tests.cmake")
  enable_testing()
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/glm_dsa/glm_dsa_tests.cmake)
endif()

# glm-dsa host-only engine sources (B1.2-B1.4): the glm-dsa geometry reader, the tensor-shape check and the loader
# hooks. Maya-owned engine files call into them on every GPU backend, so they join maya_engine whenever it exists.
set(GLM_DSA_ENGINE_HOST_SOURCES
  src/core/glm_dsa_geometry.cpp
  src/core/glm_geometry_check.cpp
)

if(TARGET maya_engine)
  target_sources(maya_engine PRIVATE ${GLM_DSA_ENGINE_HOST_SOURCES})
endif()

# Per-feature registration fragments (one file per feature, so parallel work never edits the same list):
# each cmake/glm_dsa.d/*.cmake may call target_sources(maya_engine ...) and add tests.
file(GLOB _glm_dsa_fragments CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/glm_dsa.d/*.cmake")
list(SORT _glm_dsa_fragments)
foreach(_f IN LISTS _glm_dsa_fragments)
  include(${_f})
endforeach()
