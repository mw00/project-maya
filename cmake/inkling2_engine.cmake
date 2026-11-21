# cmake/inkling2_engine.cmake - maya-inkling2: Thinking Machines' Inkling 2 Small (a GGUF set,
# tools/inkling2/pack_gguf.py) on maya-ds4's ggml block-type kernels (src/ds4/ds4_iq), built like maya-hy4
# (docs/INKLING2_AMD.md).  Included from CMakeLists.txt inside its GPU block (_strata_gpu_* in scope).
#   src/hy4/hy4_kernels.cu  the shared small kernels (the int8 rows, the F32 GEMVs, the sigmoid router)
#   src/hy4/hy4_lane.cpp    the CPU lane's thread pool
add_executable(maya-inkling2 src/inkling2/inkling2_main.cpp src/inkling2/inkling2_model.cu
                             src/inkling2/inkling2_kernels.cu src/inkling2/inkling2_lane.cpp
                             src/inkling2/inkling2_dspark.cu
                             src/hy4/hy4_kernels.cu src/hy4/hy4_lane.cpp
                             src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
target_include_directories(maya-inkling2 PRIVATE ${_strata_gpu_include_directories} include src/inkling2 src/hy4
                                                 src/mm3 third_party/ggml)
target_link_libraries(maya-inkling2 PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target}
                                            Threads::Threads)
target_compile_definitions(maya-inkling2 PRIVATE MAYA_DS4_IQ_ONLY)
# (the routed experts' SITU activation in ds4_iq's own kernels, once src/ds4/ds4_iq.cuh has it: until then the engine
#  runs an expert as two GEMVs with its own SITU kernel between them)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/src/ds4/ds4_iq.cuh _ink_situ REGEX "kActSitu")
if(_ink_situ)
  target_compile_definitions(maya-inkling2 PRIVATE INKLING2_IQ_SITU=1)
endif()
