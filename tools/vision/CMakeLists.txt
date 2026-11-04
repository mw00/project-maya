# tools/vision - maya-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build-vision --target maya-vision
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

# Maya's build options by the names its docs give them: -DMAYA_X=... sets STRATA_X, the name this file reads
get_cmake_property(_maya_cache CACHE_VARIABLES)
foreach(_v IN LISTS _maya_cache)
  if(_v MATCHES "^MAYA_(.+)$")
    set(STRATA_${CMAKE_MATCH_1} "${${_v}}")
  endif()
endforeach()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  if(NOT GGML_CPU_ARM_ARCH)
    set(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm+fp16" CACHE STRING "ggml-cpu ARM arch flags" FORCE)
  endif()
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
option(STRATA_VISION_HIP "run the vision encoder on the GPU via HIP/ROCm (builds ggml's HIP backend)" OFF)
set(STRATA_VISION_HIP_TARGETS "" CACHE STRING "AMDGPU targets for the HIP vision build, e.g. gfx1201")
set(GGML_HIP ${STRATA_VISION_HIP} CACHE BOOL "" FORCE)
if(STRATA_VISION_HIP)
  if(STRATA_VISION_CUDA)
    message(FATAL_ERROR "STRATA_VISION_CUDA and STRATA_VISION_HIP are mutually exclusive")
  endif()
  set(AMDGPU_TARGETS "${STRATA_VISION_HIP_TARGETS}" CACHE STRING "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)
# float32 sums for the encoder's matmuls and attention on the GPU (ggml_vision_precision.py): on maya-vision's own
# ggml only - the engines compile the same ggml files without it
foreach(_t ggml-cuda ggml-hip)
  if(TARGET ${_t})
    target_compile_definitions(${_t} PRIVATE MAYA_VISION_F32_SUMS)
  endif()
endforeach()

add_executable(maya-vision strata_vision.cpp)
target_link_libraries(maya-vision PRIVATE mtmd llama)
target_compile_features(maya-vision PRIVATE cxx_std_17)
set_target_properties(maya-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
