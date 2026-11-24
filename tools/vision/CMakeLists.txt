# tools/vision - maya-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DMAYA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build-vision --target maya-vision
cmake_minimum_required(VERSION 3.21)
project(maya_vision C CXX)

# the older -DSTRATA_X spelling of the build options still works (as MAYA_X, with a deprecation note)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/maya_old_names.cmake)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(MAYA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(MAYA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(MAYA_PORTABLE AND MSVC)
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
  if(MAYA_PORTABLE)
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
set(GGML_CUDA ${MAYA_VISION_CUDA} CACHE BOOL "" FORCE)
option(MAYA_VISION_HIP "run the vision encoder on the GPU via HIP/ROCm (builds ggml's HIP backend)" OFF)
set(MAYA_VISION_HIP_TARGETS "" CACHE STRING "AMDGPU targets for the HIP vision build, e.g. gfx1201")
set(GGML_HIP ${MAYA_VISION_HIP} CACHE BOOL "" FORCE)
if(MAYA_VISION_HIP)
  if(MAYA_VISION_CUDA)
    message(FATAL_ERROR "MAYA_VISION_CUDA and MAYA_VISION_HIP are mutually exclusive")
  endif()
  set(AMDGPU_TARGETS "${MAYA_VISION_HIP_TARGETS}" CACHE STRING "" FORCE)
endif()

# Only what Maya's picture reader runs is compiled (MAYA_VISION_FULL=ON: all of llama.cpp's image library, as before).
# Nothing that picks a code path is touched - which kernel a picture's sums take is decided by the same code on the
# same inputs - so a picture's embeddings are the same bits; what is left out is replaced by a function that stops
# with a message (below):
#  - mtmd's projector files other than the towers of the models Maya runs (MAYA_VISION_TOWERS);
#  - ggml's quantized-weight matmul kernels (MMQ, one instance per GGUF quant type): an mmproj's weights are F32/F16
#    (maya-vision refuses, at its start, an mmproj with a quantized tensor);
#  - flash attention for quantized or BF16 keys/values (a text model's KV cache types; the encoder's are F16).
option(MAYA_VISION_FULL "compile every projector and kernel of llama.cpp's image library, not only what Maya's towers use" OFF)
set(MAYA_VISION_TOWERS "glm5next;inkling2v;qwen2vl;qwen3vl" CACHE STRING "tools/mtmd/models/<name>.cpp files that are compiled")
if(MAYA_VISION_FULL)
  unset(GGML_CUDA_FA_QUANTS CACHE)
else()
  set(GGML_CUDA_FA_QUANTS "f16-f16" CACHE STRING "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)
if(NOT MAYA_VISION_FULL)
  include(${CMAKE_CURRENT_LIST_DIR}/lean.cmake)
endif()
if(MSVC AND TARGET ggml-cuda)
  target_compile_options(ggml-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
# float32 sums for the encoder's matmuls and attention on the GPU (ggml_vision_precision.py): on maya-vision's own
# ggml only - the engines compile the same ggml files without it
foreach(_t ggml-cuda ggml-hip)
  if(TARGET ${_t})
    target_compile_definitions(${_t} PRIVATE MAYA_VISION_F32_SUMS)
  endif()
endforeach()

add_executable(maya-vision strata_vision.cpp)
if(NOT MAYA_VISION_FULL)
  target_compile_definitions(maya-vision PRIVATE MAYA_VISION_LEAN)
endif()
if(MSVC)
  target_compile_options(maya-vision PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
target_link_libraries(maya-vision PRIVATE mtmd llama)
target_compile_features(maya-vision PRIVATE cxx_std_17)
set_target_properties(maya-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
