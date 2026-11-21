# cmake/ik2_hip_engine.cmake - maya-ik2's AMD build: Thinking Machines' Inkling 2 Small (a GGUF set's pack,
# tools/ik2/pack_gguf.py) on maya-ds4's ggml block-type kernels (src/ds4/ds4_iq), built like maya-hy4
# (docs/INKLING2_AMD.md).  Included from CMakeLists.txt inside its GPU block (_strata_gpu_* in scope).
#   src/hy4/hy4_kernels.cu  the shared small kernels (the int8 rows, the F32 GEMVs, the sigmoid router)
#   src/hy4/hy4_lane.cpp    the CPU lane's thread pool
add_executable(maya-ik2 src/ik2/hip/ik2_hip_main.cpp src/ik2/hip/ik2_hip_model.cu
                        src/ik2/hip/ik2_hip_kernels.cu src/ik2/hip/ik2_hip_lane.cpp
                        src/ik2/hip/ik2_hip_dspark.cu
                        src/hy4/hy4_kernels.cu src/hy4/hy4_lane.cpp
                        src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
target_include_directories(maya-ik2 PRIVATE ${_strata_gpu_include_directories} include src/ik2/hip src/hy4
                                            src/mm3 third_party/ggml)
target_link_libraries(maya-ik2 PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target}
                                       Threads::Threads)
target_compile_definitions(maya-ik2 PRIVATE MAYA_DS4_IQ_ONLY)
# (the routed experts' SITU activation in ds4_iq's own kernels, once src/ds4/ds4_iq.cuh has it: until then the engine
#  runs an expert as two GEMVs with its own SITU kernel between them)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/src/ds4/ds4_iq.cuh _ink_situ REGEX "kActSitu")
if(_ink_situ)
  target_compile_definitions(maya-ik2 PRIVATE IK2_HIP_IQ_SITU=1)
endif()
