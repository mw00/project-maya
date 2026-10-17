# Opt-in HIP configuration. Strata's CUDA-shaped kernels target wave32 RDNA2 / RDNA3 / RDNA4 (64 KiB LDS per
# workgroup, a signed dot4 instruction). CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it
# is not on PATH.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Maya HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
endif()
# The following architecture classifications describe upstream Strata, not Maya model validation.
# Validated on real cards: gfx1100 (RX 7900 XT / XTX) and gfx1201 (RX 9070 / 9070 XT, Radeon AI PRO R9700) by the
# maintainers; gfx1101 (RX 7800 XT, #254) and gfx1200 (RX 9060 XT, #256) by their owners. gfx1102 (RX 7600) has the
# same LDS limit and dot4 instruction and passed ctest (#192), but no model run has been reported yet. RDNA2 gfx1030 (RX 6800 / 6900) has the
# same LDS limit and wave32 but an older dot4 instruction (v_dot4_i32_i8, hip_compat/intrinsics.hpp); a community
# report ran it (#311), the maintainers have not. gfx1151 (Ryzen AI Max / Strix Halo, Radeon 8060S, RDNA3.5) is the
# same wave32 / 64 KiB LDS / sudot4 family as gfx1100, on a unified-memory APU; experimental.
set(_strata_hip_validated gfx1100 gfx1201 gfx1151)   # gfx1151: Strix Halo (v1.1.10)
set(_strata_hip_community gfx1101 gfx1200)
set(_strata_hip_unvalidated gfx1010 gfx1011 gfx1012 gfx1102 gfx1103 gfx1030 gfx1031 gfx1034 gfx1150)   # gfx1103: Radeon 780M iGPU, opt-in, portable kernels (no WMMA)
# gfx908 (AMD Instinct MI100, CDNA1) is wave64, not wave32: it runs through the hip_compat layer's logical-32-lane
# half-wavefront warp mapping (see hip_compat/intrinsics.hpp). Experimental; validated on a 4x MI100 box.
set(_strata_hip_wave64 gfx908)
# CMake hands HIP a ';' list, but a -DCMAKE_HIP_ARCHITECTURES typed by hand (or ROCm's own Windows tooling) may use
# spaces, which foreach(IN LISTS) would otherwise treat as one element.
string(REPLACE " " ";" _strata_hip_norm "${CMAKE_HIP_ARCHITECTURES}")
list(FILTER _strata_hip_norm EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _strata_hip_norm)
set(STRATA_HIP_ARCH_LIST "")
foreach(_arch IN LISTS _strata_hip_norm)
  if(_arch STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")      # gfx1100:xnack- -> gfx1100
  if(_base IN_LIST _strata_hip_validated)
  elseif(_base IN_LIST _strata_hip_community)
    message(STATUS "Maya HIP: ${_base} was validated by community reports (docs/AMD_MAYA.md)")
  elseif(_base IN_LIST _strata_hip_unvalidated)
    message(WARNING "Maya HIP: ${_base} builds, but it is not validated on a real card yet; please report results")
  elseif(_base IN_LIST _strata_hip_wave64)
    message(STATUS "Maya HIP: ${_base} is a wave64 CDNA card running through the half-wavefront compat layer (experimental)")
  else()
    message(FATAL_ERROR
      "Maya's HIP build supports wave32 gfx1100, gfx1101, gfx1151, gfx1200 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}; "
      "wave64: ${_strata_hip_wave64}); "
      "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND STRATA_HIP_ARCH_LIST "${_base}")
endforeach()
list(REMOVE_DUPLICATES STRATA_HIP_ARCH_LIST)
# wave64 CDNA cards (gfx908) run the CUDA-shaped kernels through the hip_compat layer's logical-32-lane
# half-wavefront mapping; the headers switch to their wave64 form only when this define reaches the build.
set(STRATA_HIP_WAVE64 OFF)
foreach(_a IN LISTS STRATA_HIP_ARCH_LIST)
  if(_a IN_LIST _strata_hip_wave64)
    set(STRATA_HIP_WAVE64 ON)
  endif()
endforeach()
if(NOT STRATA_HIP_ARCH_LIST)
  message(FATAL_ERROR "Maya HIP: CMAKE_HIP_ARCHITECTURES is empty")
endif()
# Normalize the actual compiler list as well as the validation list.
set(CMAKE_HIP_ARCHITECTURES "${_strata_hip_norm}" CACHE STRING "Maya HIP target architecture(s)" FORCE)
# The compiled architectures reach the runtime device check (src/core/device.cu) as "gfx1100,gfx1201": a binary
# carried to a card it has no code for stops at startup with a clear message instead of "invalid device function".
string(REPLACE ";" "," STRATA_HIP_ARCHS "${STRATA_HIP_ARCH_LIST}")
# rocWMMA headers are only needed when the build targets WMMA-capable archs (see glm_batch.cu's f16q prompt
# attention); RDNA 2-only builds skip the include - ROCm 7 no longer ships rocWMMA and a WMMA-less GPU never
# takes that path anyway.
set(_strata_hip_wmma OFF)
foreach(_arch IN LISTS STRATA_HIP_ARCH_LIST)
  if(_arch MATCHES "^(gfx1100|gfx1101|gfx1102|gfx1150|gfx1151|gfx1200|gfx1201)$")
    set(_strata_hip_wmma ON)
    break()
  endif()
endforeach()
if(_strata_hip_wmma)
  set(STRATA_HAS_WMMA_DEF "STRATA_HAS_WMMA=1")
else()
  set(STRATA_HAS_WMMA_DEF "STRATA_HAS_WMMA=0")
endif()

enable_language(HIP)
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "The ROCm hip CMake package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "The ROCm hipblas CMake package did not provide roc::hipblas")
endif()
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Maya: hipBLASLt not found; solution-table dispatch is unavailable")
endif()

# HIP's link step produces a PIE; make Strata and ggml objects PIC for the ROCm linker.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(STRATA_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
add_library(strata_hip_runtime INTERFACE)
target_include_directories(strata_hip_runtime BEFORE INTERFACE
  "${STRATA_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1 "${STRATA_HAS_WMMA_DEF}" "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\""
  $<$<BOOL:${STRATA_HIP_WAVE64}>:STRATA_HIP_WAVE64=1>)
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
foreach(_language IN ITEMS CXX HIP)
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# CMake does not infer HIP from Strata's existing CUDA-shaped .cu suffixes.
file(GLOB_RECURSE _strata_hip_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
if(_strata_hip_sources)
  set_source_files_properties(${_strata_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
foreach(_source IN ITEMS tests/hip/intrinsics.cpp tests/hip/native_qsa_score.cpp)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    set_source_files_properties("${_source}" PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "Strata: HIP enabled, arch ${STRATA_HIP_ARCHS}")
