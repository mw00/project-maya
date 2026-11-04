# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2.  The kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard
# below refuses anything older: silently building for a pre-Ampere arch would produce a binary that runs
# and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.27 LANGUAGES CXX)

# Maya's build options by the names its docs give them: -DMAYA_X=... sets STRATA_X, the name this file reads
get_cmake_property(_maya_cache CACHE_VARIABLES)
foreach(_v IN LISTS _maya_cache)
  if(_v MATCHES "^MAYA_(.+)$")
    set(STRATA_${CMAKE_MATCH_1} "${${_v}}")
  endif()
endforeach()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# x86 gets the AVX-512/AVX2 expert kernels; other architectures (aarch64: DGX Spark / Grace) take the
# portable fallbacks and ggml-cpu's own NEON dot products for the native packs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
  set(STRATA_CPU_X86 ON)
else()
  set(STRATA_CPU_X86 OFF)
endif()
if(NOT STRATA_CPU_X86 AND NOT MSVC)
  # ggml-cpu's reference sums must not be re-associated by the compiler (parity with the x86 kernels' order).
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
option(STRATA_ENABLE_HIP "Build the HIP targets for gfx1100 wave32 hardware" OFF)
option(STRATA_PREFILL_MMQ "Build the GGML MMQ prompt path for HIP (a HIP build needs it)" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
# the GLM prompt path (src/core/glm_prefill.cu, always in strata_engine) multiplies its experts with MMQ, so a HIP
# build without it does not link (found by @boxwrench in #17)
if(STRATA_ENABLE_HIP AND NOT STRATA_PREFILL_MMQ)
  message(FATAL_ERROR "A HIP build needs -DSTRATA_PREFILL_MMQ=ON: the GLM prompt path multiplies its experts with "
                      "MMQ (./maya.sh --backend hip sets it)")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

# Apple Silicon: maya-metal (GLM-5.3-Flash through ggml's Metal backend, cmake/metal_backend.cmake) and nothing else
option(MAYA_ENABLE_METAL "Build maya-metal for Apple Silicon (GLM-5.3-Flash on the Metal GPU)" OFF)
if(MAYA_ENABLE_METAL)   # (the CUDA / AMD options do not apply to this build)
  include(cmake/metal_backend.cmake)
  return()
endif()

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  # "native" builds for every GPU in the PC that CUDA_VISIBLE_DEVICES leaves visible - at configure time and again at
  # build time (nvcc -arch=native), so a display card beside the GPUs (a Quadro P400 or a GTX 1060 next to V100s, sm_61)
  # was compiled for too and could stop the build (#56). It becomes the list of those cards' own architectures here:
  # below sm_60 is left out, and so is Pascal when a Volta or newer card is there (a Pascal-only PC keeps it:
  # experimental); none left is an error that says why. The cache keeps "native", so a later configure looks again.
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(_found)
    foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
      string(REGEX REPLACE "-.*$" "" _base "${_a}")
      if(_base MATCHES "^[0-9]+$")
        list(APPEND _found "${_base}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _found)
    set(_volta OFF)
    foreach(_base IN LISTS _found)
      if(NOT _base LESS 70)
        set(_volta ON)
      endif()
    endforeach()
    set(_kept)
    set(_left)
    foreach(_base IN LISTS _found)
      if(_base LESS 60 OR (_volta AND _base LESS 70))
        list(APPEND _left "sm_${_base}")
      else()
        list(APPEND _kept "${_base}")
      endif()
    endforeach()
    if(NOT _kept)
      message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES=native, but none of this PC's GPUs (${CMAKE_CUDA_ARCHITECTURES_NATIVE}) is one Maya "
        "builds for (compute capability 6.0 or newer): name the architecture, e.g. -DCMAKE_CUDA_ARCHITECTURES=70.")
    endif()
    if(_left)
      list(JOIN _left ", " _left_s)
      message(STATUS "Maya: CMAKE_CUDA_ARCHITECTURES=native left out ${_left_s} (a display card beside the GPUs)")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_kept}")
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Developed on sm_120 (RTX 50) and Volta (sm_70: GLM, maya-ds4 and maya-mm3 run on V100s); GLM also builds for
    # Pascal (sm_6x, experimental - maya.py's MIN_CC).  Below sm_60 is refused.  maya-mm3 is built for sm_70 and
    # newer only (its kernels take more than Pascal's 4 KB of parameters: _mm3_archs below).
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    if(_base LESS 60)
      message(FATAL_ERROR
        "Maya needs an NVIDIA GPU of compute capability 6.0 or newer (Pascal experimental, Volta / V100 and the "
        "RTX 20 / 30 / 40 / 50 series), but CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
    endif()
  endforeach()
  message(STATUS "Maya: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  # Pascal (sm_6x - a "native" build on a Tesla P40, as maya.py's arch_setting() asks for below sm_75): GP102 / GP104
  # cache global loads in L2 only unless the code is compiled to cache them in L1 as well, and the decode's GEMVs - many
  # small loads of weights, scales and the q8_1 activations every warp of a block reads - starve there (Maya-L's Q6_K
  # dense GEMV ran at a quarter of a P40's bandwidth or less).  -dlcm=ca caches them in L1, as Volta and newer do by
  # default: two P40s, decode 9.7 -> 16.9 tok/s with Maya-L and 9.4 -> 28.0 with Maya-S24, the same answers (#212).
  # A build for sm_70 and newer only is unchanged.
  set(_strata_archs "${CMAKE_CUDA_ARCHITECTURES}")
  if(_strata_archs MATCHES "native")
    set(_strata_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
  endif()
  foreach(_a IN LISTS _strata_archs)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")
    if(_base MATCHES "^6[0-9]$")
      string(APPEND CMAKE_CUDA_FLAGS " -Xptxas -dlcm=ca")
      message(STATUS "Maya: Pascal (sm_${_base}): global loads cached in L1 (-Xptxas -dlcm=ca)")
      break()
    endif()
  endforeach()
  # CMake links CUDA code with the static runtime (CMAKE_CUDA_RUNTIME_LIBRARY's default); on Windows the same runtime
  # must be named here too - MSVC's linker refuses cudart_static.lib and cudart.lib (the DLL's) in one program,
  # where Linux's takes the first of the duplicates
  if(WIN32)
    set(_strata_gpu_runtime_target CUDA::cudart_static)
  else()
    set(_strata_gpu_runtime_target CUDA::cudart)
  endif()
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target roc::hipblas)
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  add_executable(exl3_hip_scalar_policy_test tests/exl3_hip_scalar_policy_test.cpp)
  add_test(NAME exl3_hip_scalar_policy_test COMMAND exl3_hip_scalar_policy_test)
  add_executable(ds4_path_policy_test tests/ds4_path_policy_test.cpp)
  add_test(NAME ds4_path_policy_test COMMAND ds4_path_policy_test)
  add_executable(ds4_draft_placement_test tests/ds4_draft_placement_test.cpp)
  add_test(NAME ds4_draft_placement_test COMMAND ds4_draft_placement_test)
  add_executable(ds4_prefill_stats_test tests/ds4_prefill_stats_test.cpp)
  add_test(NAME ds4_prefill_stats_test COMMAND ds4_prefill_stats_test)
  add_executable(ds4_pcie_test tests/ds4_pcie_test.cpp)
  add_test(NAME ds4_pcie_test COMMAND ds4_pcie_test)
  add_executable(ds4_ram_slice_test tests/ds4_ram_slice_test.cpp)
  add_test(NAME ds4_ram_slice_test COMMAND ds4_ram_slice_test)
  # maya-ds4 --serve's watchdog: it fires on a request that stopped moving, never on a long one still moving (#539)
  add_executable(ds4_watchdog_test tests/ds4_watchdog_test.cpp)
  add_test(NAME ds4_watchdog_test COMMAND ds4_watchdog_test)
  # maya-ds4's restore points: the one a prompt goes back to, the ones a read keeps (marks= / shared=), the cases
  add_executable(ds4_marks_test tests/ds4_marks_test.cpp)
  target_link_libraries(ds4_marks_test PRIVATE strata_warnings)
  add_test(NAME ds4_marks_test COMMAND ds4_marks_test)
  # maya-ds4's log-probabilities (GEN logprobs=): the reference, the penalties, ties, the kernel's pieces, the LP line
  add_executable(ds4_logprobs_test tests/ds4_logprobs_test.cpp)
  target_link_libraries(ds4_logprobs_test PRIVATE strata_warnings)
  add_test(NAME ds4_logprobs_test COMMAND ds4_logprobs_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # --ablate's direction loader: what it takes and what it refuses (synthetic GGUFs: no model, no GPU)
  add_executable(ds4_ablate_test tests/ds4_ablate_test.cpp)
  target_link_libraries(ds4_ablate_test PRIVATE strata_artifact)
  add_test(NAME ds4_ablate_test COMMAND ds4_ablate_test)
endif()
# maya-ds4's Engram reader: a batch read ahead (a prompt's next part's rows) waits behind the others, every read the
# file's bytes (src/ds4/ds4_engram_io.hpp; a file in the temp folder, host code only)
if(NOT WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ds4_engram_reader_test.cpp")
  add_executable(ds4_engram_reader_test EXCLUDE_FROM_ALL tests/ds4_engram_reader_test.cpp)
  target_link_libraries(ds4_engram_reader_test PRIVATE strata_warnings Threads::Threads)
  add_test(NAME ds4_engram_reader_test COMMAND ds4_engram_reader_test)
endif()
# maya-ds4 --parallel, DS4_MULTI_ATTN: a multi step's window buffers side by side and the attention's launches over them
# (src/ds4/ds4_multi_attn.hpp; host code only, so a build without the suite builds it by name)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ds4_multi_attn_plan_test.cpp")
  add_executable(ds4_multi_attn_plan_test EXCLUDE_FROM_ALL tests/ds4_multi_attn_plan_test.cpp)
  target_link_libraries(ds4_multi_attn_plan_test PRIVATE strata_warnings)
  add_test(NAME ds4_multi_attn_plan_test COMMAND ds4_multi_attn_plan_test)
endif()
# DeepSeek's prompt-path and memory rules (src/ds4/ds4_prompt_mem.hpp; pure functions, CPU only, so a build without
# the suite builds it by name)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ds4_prompt_mem_test.cpp")
  add_executable(ds4_prompt_mem_test EXCLUDE_FROM_ALL tests/ds4_prompt_mem_test.cpp)
  target_link_libraries(ds4_prompt_mem_test PRIVATE strata_warnings)
  add_test(NAME ds4_prompt_mem_test COMMAND ds4_prompt_mem_test)
endif()
# STRATA_HIPBLASLT_TUNING's table list: ':' on Linux, ';' on Windows with drive letters kept (host code only, so a
# build without HIP builds it by name)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hipblaslt_table_list_test.cpp")
  add_executable(hipblaslt_table_list_test EXCLUDE_FROM_ALL tests/hipblaslt_table_list_test.cpp)
  target_include_directories(hipblaslt_table_list_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  target_link_libraries(hipblaslt_table_list_test PRIVATE strata_warnings)
  add_test(NAME hipblaslt_table_list_test COMMAND hipblaslt_table_list_test)
endif()
# the engine's start-up line of the GLM settings set in the environment (include/strata/platform/env_report.hpp; host
# code only, so a build without the suite builds it by name)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/env_report_test.cpp")
  add_executable(env_report_test EXCLUDE_FROM_ALL tests/env_report_test.cpp)
  target_include_directories(env_report_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(env_report_test PRIVATE strata_warnings)
  add_test(NAME env_report_test COMMAND env_report_test)
endif()

# ---- idle_gate_test --------------------------------------------------------
# The gate an idle engine's service threads sleep on, and a request's line wakes (strata/platform/idle_gate.hpp):
# host code only, like env_report_test above.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/idle_gate_test.cpp")
  add_executable(idle_gate_test EXCLUDE_FROM_ALL tests/idle_gate_test.cpp)
  target_include_directories(idle_gate_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(idle_gate_test PRIVATE strata_warnings)
  find_package(Threads)
  if(Threads_FOUND)
    target_link_libraries(idle_gate_test PRIVATE Threads::Threads)
  endif()
  add_test(NAME idle_gate_test COMMAND idle_gate_test)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "maya: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/router_sigmoid.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/glm_hc.cu
                          src/kernels/cuda/glm_kda.cu
                          src/kernels/cuda/glm_ffn.cu
                          src/kernels/cuda/glm_dsa.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/cuda/glm_fast.cu
                          src/kernels/cuda/glm_batch.cu
                          src/kernels/cuda/exl3.cu
                          src/kernels/cuda/exl3_legacy.cu
                          src/kernels/cuda/glm_kv_stream.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    add_executable(glm_layer_graph_test tests/glm_layer_graph_test.cu)
    target_link_libraries(glm_layer_graph_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME glm_layer_graph_test COMMAND glm_layer_graph_test)
    set_tests_properties(glm_layer_graph_test PROPERTIES TIMEOUT 60)
    # the EXL3 kernels against the scalar CPU reference (random matrices, no model)
    add_executable(exl3_test tests/exl3_test.cu)
    target_link_libraries(exl3_test PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_test(NAME exl3_test COMMAND exl3_test)
    set_tests_properties(exl3_test PROPERTIES TIMEOUT 120)
    if(STRATA_ENABLE_HIP)
      # exl3_test exercises the default. Check explicit enable and rollback too, including mixed-row fallback.
      add_test(NAME exl3_mv1_test COMMAND ${CMAKE_COMMAND} -E env DS4_EXL3_MV1=1 $<TARGET_FILE:exl3_test>)
      add_test(NAME exl3_mv_generic_test COMMAND ${CMAKE_COMMAND} -E env DS4_EXL3_MV1=0 $<TARGET_FILE:exl3_test>)
      set_tests_properties(exl3_mv1_test exl3_mv_generic_test PROPERTIES TIMEOUT 120)
    endif()
    # the same decoders against exllamav3 itself (tools/exl3_exllamav3_fixture.py's tiles): the same bits
    add_test(NAME exl3_exllamav3 COMMAND exl3_test --exllamav3 ${CMAKE_CURRENT_SOURCE_DIR}/data/exl3-exllamav3)
    set_tests_properties(exl3_exllamav3 PROPERTIES TIMEOUT 60)
  endif()
  # the batched rows' MoE pieces (the fused wait / fetch, the rows' CPU-lane waits, a group's experts each once)
  # against the per-row kernels (random Q8_0 experts, no model)
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(glm_moe_rows_test tests/glm_moe_rows_test.cu)
    target_link_libraries(glm_moe_rows_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME glm_moe_rows_test COMMAND glm_moe_rows_test)
    set_tests_properties(glm_moe_rows_test PROPERTIES TIMEOUT 60)
  endif()
  # the batched rows' one-launch kernels (hc_rows, kda_layers over the rows, mv_nt) against the per-row kernels, bitwise
  # (random inputs, no model)
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(glm_rows_kernels_test tests/glm_rows_kernels_test.cu)
    target_link_libraries(glm_rows_kernels_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME glm_rows_kernels_test COMMAND glm_rows_kernels_test)
    set_tests_properties(glm_rows_kernels_test PROPERTIES TIMEOUT 120)
    # the rows' DSA attention in one launch (STRATA_GLM_ROWS_ATTN) against mla() a row, resident and streamed caches
    add_executable(glm_rows_attn_test tests/glm_rows_attn_test.cu)
    target_link_libraries(glm_rows_attn_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME glm_rows_attn_test COMMAND glm_rows_attn_test)
    set_tests_properties(glm_rows_attn_test PROPERTIES TIMEOUT 300)
  endif()
  # the INT8 latent records' FP16 reads against the plain conversion, every code x every scale (GPU, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(glm_lat8_test tests/glm_lat8_test.cu)
    target_link_libraries(glm_lat8_test PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME glm_lat8_test COMMAND glm_lat8_test)
    set_tests_properties(glm_lat8_test PROPERTIES TIMEOUT 60)
  endif()
  # __dp4a as the kernels see it (sm_60: kernels/dp4a.hpp's vmad fallback) against the byte-wise form (GPU, no model)
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(dp4a_sm60_test tests/dp4a_sm60_test.cu)
    target_link_libraries(dp4a_sm60_test PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME dp4a_sm60_test COMMAND dp4a_sm60_test)
    set_tests_properties(dp4a_sm60_test PROPERTIES TIMEOUT 60)
  endif()
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON
  if(STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
  # the Volta grouped-WMMA prompt experts (STRATA_PF_WMMA) against a double reference over the same FP16 weights
  # (GPU, synthetic, no model); skips with 77 below sm_70
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(pf_wmma_parity src/kernels/pf_wmma_parity.cpp)
    target_include_directories(pf_wmma_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pf_wmma_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME pf_wmma_parity COMMAND pf_wmma_parity)
    set_tests_properties(pf_wmma_parity PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    # EXL3: multiply-adds fused where the source writes them, not after inlining - otherwise a kernel's rounding
    # depends on where the compiler inlined its GEMV, and a verify window's bits stop matching the token path's
    set_source_files_properties(src/kernels/cuda/exl3.cu PROPERTIES COMPILE_OPTIONS "-ffp-contract=on")
  endif()

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  # This oracle is a CUDA ggml build; do not pull CUDA libraries into a HIP build.
  if(STRATA_ENABLE_CUDA AND STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
    src/core/glm_model.cu src/core/glm_fast_path.cu src/core/glm_prefill.cu src/core/glm_multi.cu src/core/glm_rows.cu
    src/core/glm_window.cu src/core/glm_dflash.cu src/core/glm_mtp.cu src/core/glm_mtp_rows.cu)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  if(WIN32 AND STRATA_ENABLE_CUDA)
    target_link_libraries(strata_engine PRIVATE dxgi)
  endif()
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    # the DFlash2 drafter against a CPU float forward of the same GGUF (glm_dflash_test <draft.gguf>; needs a GPU)
    add_executable(glm_dflash_test tests/glm_dflash_test.cu)
    target_include_directories(glm_dflash_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_dflash_test PRIVATE strata_engine ${_strata_gpu_runtime_target})
    # (strata_prefill after strata_engine, as the engines link: on HIP glm_prefill.cu uses its Gemm - #355)
    if(STRATA_ENABLE_HIP)
      target_link_libraries(glm_dflash_test PRIVATE strata_prefill)
    endif()
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    # the fast path's multi-column GEMV: every column against the single-column call, and its time (needs a GPU)
    # (AMD builds its own glm_mv_bench from tests/hip/)
    add_executable(glm_mv_bench tests/glm_mv_bench.cu)
    target_include_directories(glm_mv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_mv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_executable(glm_pack_graph_test tests/glm_pack_graph_test.cpp)
    target_include_directories(glm_pack_graph_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_pack_graph_test PRIVATE strata_engine ${_strata_gpu_runtime_target})
    # Full-pack diagnostic: compare batched prefill with the token-at-a-time reference on one GPU.
    add_executable(glm_prefill_probe tests/glm_prefill_probe.cpp)
    target_include_directories(glm_prefill_probe PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_prefill_probe PRIVATE strata_engine ${_strata_gpu_runtime_target})
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
  endif()
  # the expert cache's tail lent to the on-demand vision encoder and taken back (needs a GPU: 77 skips)
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_cache_lend_test.cpp")
    enable_testing()
    add_executable(expert_cache_lend_test tests/core/expert_cache_lend_test.cpp)
    target_include_directories(expert_cache_lend_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(expert_cache_lend_test PRIVATE strata_engine ${_strata_gpu_runtime_target})
    add_test(NAME expert_cache_lend_test COMMAND expert_cache_lend_test)
    set_tests_properties(expert_cache_lend_test PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
  endif()
  # --expert-profile-save's profile (rank_learned_profile, write_expert_profile) and the profile header's version
  # (a version other than 1 is refused by its number); the split decode's groups (split_groups); the EXL3 prompt
  # experts' tensor-core switch by chunk length (exl3_mma_rows): CPU only
  # (a version other than 1 is refused by its number); the split decode's groups (split_groups); where a read cuts to
  # keep its system prompt's state (prefix_cut): CPU only
  # (a version other than 1 is refused by its number); the split decode's groups (split_groups): CPU only
  # (a version other than 1 is refused by its number); the split decode's groups (split_groups); the pieces a prompt is
  # read in between decode steps (prefill_pieces): CPU only
  foreach(_t expert_profile_save_test expert_profile_version_test split_groups_test exl3_mma_rows_test prefix_cut_test mtp_rows_test prefill_pieces_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/${_t}.cpp")
      enable_testing()
      add_executable(${_t} tests/core/${_t}.cpp)
      # (strata_prefill after strata_engine, as the engines link: on HIP glm_prefill.cu uses its Gemm - #355)
      target_link_libraries(${_t} PRIVATE strata_engine strata_prefill)
      add_test(NAME ${_t} COMMAND ${_t})
      set_tests_properties(${_t} PROPERTIES TIMEOUT 60)
    endif()
  endforeach()

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
    target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)

    add_executable(tune_hipblaslt tools/hip/tune_hipblaslt.cpp)
    target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
  endif()

  add_executable(maya src/program/generate.cpp)
  target_include_directories(maya PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(maya PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})

  # DeepSeek-V4.1-Flash (EXL3, tools/ds4/pack.py; or a GGUF, tools/ds4/pack_gguf.py): its own engine over the kernel
  # libraries (src/ds4)
  # (CUDA, or HIP through include/strata/hip_compat as the GLM engine; Linux and Windows)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(maya-ds4 src/ds4/ds4_main.cpp src/ds4/ds4_model.cu src/ds4/ds4_kernels.cu src/ds4/ds4_vision.cu
                              src/ds4/ds4_gguf.cu src/ds4/ds4_gguf_cpu.cpp src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp
                              src/ds4/ds4_lt.cu src/ds4/ds4_host_ar.cpp src/ds4/ds4_packinfo.cpp)
    target_include_directories(maya-ds4 PRIVATE ${_strata_gpu_include_directories} third_party/ggml)
    # the commit and the release `maya-ds4 --version` and --verbose name, written at every build (cmake/build_id.cmake:
    # git's commit, a source archive's BUILD_ID file, or -DMAYA_BUILD_ID=...; the release from VERSION)
    set(MAYA_BUILD_ID "" CACHE STRING "maya-ds4 --version's build id (empty: git's commit, or the BUILD_ID file)")
    add_custom_target(maya_build_id
      COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}
              -DOUT=${CMAKE_CURRENT_BINARY_DIR}/generated/maya_build_id.h -DID=${MAYA_BUILD_ID}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_id.cmake
      BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/maya_build_id.h
      VERBATIM)
    add_dependencies(maya-ds4 maya_build_id)
    target_include_directories(maya-ds4 PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(maya-ds4 PRIVATE strata_kernels strata_kernels_cpu strata_spec ${_strata_gpu_runtime_target}
                          ${_strata_gpu_blas_target})
    if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)   # (the prompt path's dense GEMMs: ds4_lt.cu)
      set_source_files_properties(src/ds4/ds4_lt.cu PROPERTIES COMPILE_DEFINITIONS STRATA_HIPBLASLT_AVAILABLE=1)
      target_link_libraries(maya-ds4 PRIVATE roc::hipblaslt)
    endif()
    if(STRATA_ENABLE_HIP)
      target_sources(maya-ds4 PRIVATE src/ds4/ds4_peer_calibrate.cu)
      target_link_libraries(maya-ds4 PRIVATE strata_mmq)
    endif()
    if(NOT WIN32)   # mmap / mbind (NUMA)
      add_executable(ds4_cpu_bench tests/ds4_cpu_bench.cpp)
      target_link_libraries(ds4_cpu_bench PRIVATE strata_kernels_cpu)
      # the CPU lane's EXL3 experts: flow against steps bit for bit, and their time (no GPU, no model)
      add_executable(ds4_cpu_flow_test tests/ds4_cpu_flow_test.cpp)
      target_link_libraries(ds4_cpu_flow_test PRIVATE strata_kernels_cpu)
    endif()
    # the CPU lane's MXFP4 rows: every tier this CPU runs (and ggml-cpu's dot) at the experts' real size (no GPU)
    add_executable(ds4_mx_bench tests/ds4_mx_bench.cpp src/ds4/ds4_gguf_cpu.cpp)
    target_link_libraries(ds4_mx_bench PRIVATE strata_kernels_cpu)
    # the CPU lane's time an expert by format, as the PCIe lane's calibration times it (no GPU), and the split it plans
    add_executable(ds4_lane_bench tests/ds4_lane_bench.cpp src/ds4/ds4_gguf_cpu.cpp src/ds4/ds4_iq_cpu.cpp)
    target_include_directories(ds4_lane_bench PRIVATE third_party/ggml)
    target_link_libraries(ds4_lane_bench PRIVATE strata_kernels_cpu)
    # the CPU lane's exact AVX-512 rows without VBMI (src/ds4/ds4_iq512_kern.inc: IQ2_XS, IQ3_XXS, IQ1_M) bit for bit
    # against ggml-cpu's dot, one thread, no GPU: forced on (any CPU with AVX-512 F/BW/VL/DQ and BMI2), and this CPU's
    # default choice; --bench times the rows
    enable_testing()
    add_executable(ds4_iq512_rows_test tests/ds4_iq512_rows_test.cpp src/ds4/ds4_iq_cpu.cpp)
    target_include_directories(ds4_iq512_rows_test PRIVATE third_party/ggml)
    target_link_libraries(ds4_iq512_rows_test PRIVATE strata_kernels_cpu)
    add_test(NAME ds4_iq512_rows COMMAND ds4_iq512_rows_test)
    set_tests_properties(ds4_iq512_rows PROPERTIES ENVIRONMENT DS4_IQ512=1 TIMEOUT 120)
    add_test(NAME ds4_iq512_rows_default COMMAND ds4_iq512_rows_test)
    set_tests_properties(ds4_iq512_rows_default PROPERTIES TIMEOUT 120)
    # the i-quant pieces on the host (no GPU): IQ1_M's code table (ds4_iq1_codes.hpp, the persistent device kernels'
    # decode) and IQ1_S's codes against ggml's own values bit for bit, IQ4_NL experts on the CPU lane against ggml-cpu's
    # dot bit for bit
    add_executable(ds4_iq_host_test tests/ds4_iq_host_test.cpp src/ds4/ds4_iq_cpu.cpp)
    target_include_directories(ds4_iq_host_test PRIVATE third_party/ggml)
    target_link_libraries(ds4_iq_host_test PRIVATE strata_kernels_cpu)
    add_test(NAME ds4_iq_host_test COMMAND ds4_iq_host_test)
    # the GGUF kernels (BF16 matrices, MXFP4 experts) against a double-precision reference, and the packed KV caches
    # against the BF16 rows bit for bit (needs a GPU, no model)
    add_executable(ds4_gguf_test tests/ds4_gguf_test.cu src/ds4/ds4_gguf.cu src/ds4/ds4_gguf_cpu.cpp src/ds4/ds4_iq.cu
                                 src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_kernels.cu
                                 src/ds4/ds4_host_ar.cpp)
    target_include_directories(ds4_gguf_test PRIVATE ${_strata_gpu_include_directories} third_party/ggml)
    target_link_libraries(ds4_gguf_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_HIP)
      target_link_libraries(ds4_gguf_test PRIVATE strata_mmq)
    endif()
    # DS4_MULTI_ATTN: a multi step's attention for all its rows in one launch a step against each conversation's run
    # alone (attn_core's launches), bit for bit on random rings and caches - decode rows, verify windows, drafter
    # blocks (needs a GPU, no model; built by name)
    add_executable(ds4_multi_attn_test EXCLUDE_FROM_ALL tests/ds4_multi_attn_test.cu src/ds4/ds4_kernels.cu
                                       src/ds4/ds4_host_ar.cpp)
    target_include_directories(ds4_multi_attn_test PRIVATE ${_strata_gpu_include_directories} include)
    target_link_libraries(ds4_multi_attn_test PRIVATE ${_strata_gpu_runtime_target})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ds4_packinfo_test.cpp")
      # maya-ds4 --pack-info on packs written for it (healthy, dead, short, a type no kernel takes, a drafter its
      # metadata leaves out, a profile of another size): host code only, no GPU
      enable_testing()
      add_executable(ds4_packinfo_test tests/ds4_packinfo_test.cpp src/ds4/ds4_packinfo.cpp)
      add_test(NAME ds4_packinfo_test COMMAND ds4_packinfo_test)
      set_tests_properties(ds4_packinfo_test PROPERTIES TIMEOUT 60)
    endif()
  endif()

  # MiniMax-M3.1-Flash (EXL3, tools/mm3/pack.py): its own engine (src/mm3), with its own copy of the EXL3 device kernels
  # (swigluoai) - none of the other engines' code but maya-ds4's OS helpers (src/ds4/ds4_os.hpp).  CUDA and AMD (HIP),
  # on Linux and Windows.  CUDA: sm_70 and newer only - its EXL3 kernels take more than the 4 KB of parameters before Volta allows
  # (a "native" build on a PC with an older display card beside its GPUs - a Quadro P400 - leaves that card out).
  set(_mm3_archs)
  set(_mm3_filter OFF)
  if(STRATA_ENABLE_CUDA AND _strata_archs)
    set(_mm3_filter ON)
    foreach(_a IN LISTS _strata_archs)
      string(REGEX REPLACE "-.*$" "" _base "${_a}")
      if(NOT _base MATCHES "^[0-9]+$")
        set(_mm3_filter OFF)   # ("all" / "all-major": as CMake gives them)
      elseif(NOT _base LESS 70)
        list(APPEND _mm3_archs "${_a}")
      endif()
    endforeach()
    if(_mm3_filter AND NOT _mm3_archs)
      message(STATUS "Maya: maya-mm3 not built - it needs sm_70 or newer, arch ${_strata_archs}")
    endif()
  endif()
  if((STRATA_ENABLE_CUDA AND (_mm3_archs OR NOT _mm3_filter)) OR STRATA_ENABLE_HIP)
    # (a GGUF's weights - tools/mm3/pack_gguf.py, MiniMax's Maya quants: maya-ds4's ggml block-type kernels, src/ds4/ds4_iq,
    #  on the device and its CPU rows on the CPU lane - mm3_gguf.cu, mm3_cpu_gg.cpp)
    set(_mm3_gguf_sources src/mm3/mm3_gguf.cu src/mm3/mm3_gv.cu src/ds4/ds4_iq.cu src/ds4/ds4_lt.cu)   # (ds4_lt: mm3_gguf.cu's AMD GEMMs)
    set(_mm3_gguf_cpu_sources src/mm3/mm3_cpu_gg.cpp src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
    # (AMD (HIP) builds the same engine: the EXL3 kernels' HIP paths as the shared exl3.cu's, tensor parallel over
    #  peer copies, the DSpark draft and the vision tower through hipBLAS - include/strata/hip_compat)
    set(_mm3_nv_sources src/mm3/mm3_exl3.cu src/mm3/mm3_dspark.cu src/mm3/mm3_vision.cu src/mm3/mm3_tp.cu)
    add_executable(maya-mm3 src/mm3/mm3_main.cpp src/mm3/mm3_model.cu src/mm3/mm3_kernels.cu src/mm3/mm3_cpu.cpp
                            src/mm3/mm3_host.cpp ${_mm3_nv_sources} ${_mm3_gguf_sources} ${_mm3_gguf_cpu_sources})
    target_include_directories(maya-mm3 PRIVATE ${_strata_gpu_include_directories} include src/mm3 third_party/ggml)
    # (the CPU lane: the shared CPU EXL3 kernels, used as they are; ggml-cpu's dots for a GGUF's experts)
    target_link_libraries(maya-mm3 PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})
    # (ds4_iq.cu without the rest of maya-ds4: its dependent launch (src/ds4/ds4_pdl.cuh) and its AMD prompt experts'
    #  MMQ (MAYA_DS4_EXPERT_MMQ) left out - MiniMax's kernels launch as before)
    target_compile_definitions(maya-mm3 PRIVATE MAYA_DS4_IQ_ONLY)
    if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)   # (the prompt path's dense GEMMs on AMD: ds4_lt.cu, measured)
      set_source_files_properties(src/ds4/ds4_lt.cu PROPERTIES COMPILE_DEFINITIONS STRATA_HIPBLASLT_AVAILABLE=1)
      target_link_libraries(maya-mm3 PRIVATE roc::hipblaslt)
    endif()
    if(STRATA_BUILD_TESTS)
      # maya-mm3's EXL3 kernels against the scalar CPU reference (random matrices, no model): exl3_test's cases with
      # swigluoai, the Volta / Turing tensor-core tiles and the FP16 GEMM path before sm_80
      add_executable(mm3_exl3_test tests/mm3_exl3_test.cu src/mm3/mm3_exl3.cu ${_mm3_gguf_sources})
      target_include_directories(mm3_exl3_test PRIVATE ${_strata_gpu_include_directories} include src/mm3 third_party/ggml)
      target_link_libraries(mm3_exl3_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target}
                            ${_strata_gpu_blas_target})
      if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
        target_link_libraries(mm3_exl3_test PRIVATE roc::hipblaslt)
      endif()
      target_compile_definitions(mm3_exl3_test PRIVATE MAYA_DS4_IQ_ONLY)
      add_test(NAME mm3_exl3_test COMMAND mm3_exl3_test)
      set_tests_properties(mm3_exl3_test PROPERTIES TIMEOUT 300)
      # the GGUF weights (no model): the dense types' values, rows and GEMVs against ggml's dequantize_row_*, the GEMMs,
      # the routed experts with swigluoai (gate and up of one type and of two) on the token / window / prompt paths
      # against a double reference and each other, and the CPU lane's GGUF experts against the device's
      add_executable(mm3_gguf_test tests/mm3_gguf_test.cu src/mm3/mm3_exl3.cu src/mm3/mm3_cpu.cpp
                                   ${_mm3_gguf_sources} ${_mm3_gguf_cpu_sources})
      target_include_directories(mm3_gguf_test PRIVATE ${_strata_gpu_include_directories} include src/mm3 third_party/ggml)
      target_link_libraries(mm3_gguf_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target}
                            ${_strata_gpu_blas_target})
      if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
        target_link_libraries(mm3_gguf_test PRIVATE roc::hipblaslt)
      endif()
      target_compile_definitions(mm3_gguf_test PRIVATE MAYA_DS4_IQ_ONLY)
      add_test(NAME mm3_gguf_test COMMAND mm3_gguf_test)
      set_tests_properties(mm3_gguf_test PROPERTIES TIMEOUT 600)
      # iq_mmvq's multi-column path against single-column calls: every supported IQ format and width, compared
      # in raw bits (no model).  The block index is outermost there, so a batch reads each weight block once.
      add_executable(iq_mmvq_parity_test tests/iq_mmvq_parity_test.cu)
      target_include_directories(iq_mmvq_parity_test PRIVATE ${_strata_gpu_include_directories} include)
      target_link_libraries(iq_mmvq_parity_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
      add_test(NAME iq_mmvq_parity_test COMMAND iq_mmvq_parity_test)
      set_tests_properties(iq_mmvq_parity_test PROPERTIES TIMEOUT 300)

      # the K / V caches in every format (BF16, INT8, FP8, FP4, mixed; rows rotated or not) and the sparse attention
      # over them: exact over the decoded rows, KV streaming the same bits, each format's error (no model)
      add_executable(mm3_kv_test tests/mm3_kv_test.cu src/mm3/mm3_kernels.cu)
      target_include_directories(mm3_kv_test PRIVATE ${_strata_gpu_include_directories} include src/mm3)
      target_link_libraries(mm3_kv_test PRIVATE ${_strata_gpu_runtime_target})
      add_test(NAME mm3_kv_test COMMAND mm3_kv_test)
      set_tests_properties(mm3_kv_test PROPERTIES TIMEOUT 600)
      if(NOT WIN32 OR STRATA_ENABLE_HIP)
        # maya-mm3's kernels against F64 host references: the candidates' top-k, the INT8 / rotated K / V attention and
        # its split, the sampled draft step, the pipeline's waits (no model)
        add_executable(mm3_kernels_test tests/mm3_kernels_test.cu src/mm3/mm3_kernels.cu)
        target_include_directories(mm3_kernels_test PRIVATE ${_strata_gpu_include_directories} include src/mm3)
        target_link_libraries(mm3_kernels_test PRIVATE ${_strata_gpu_runtime_target})
        add_test(NAME mm3_kernels_test COMMAND mm3_kernels_test)
        set_tests_properties(mm3_kernels_test PROPERTIES TIMEOUT 600)
      endif()
      if(NOT WIN32)
        # the CPU lane's one-dispatch schedule against its four phases on random EXL3 sub-experts: the same bits
        # (its own stubs of expert_layout.cpp / pool.cpp: the shared CPU EXL3 kernels' source alone, not the library)
        add_executable(mm3_lane_test tests/mm3_lane_test.cpp src/mm3/mm3_cpu.cpp src/kernels/cpu/exl3_cpu.cpp)
        target_include_directories(mm3_lane_test PRIVATE include src/mm3)
        target_compile_options(mm3_lane_test PRIVATE -mavx2 -mfma -mf16c)
        target_link_libraries(mm3_lane_test PRIVATE Threads::Threads)
        add_test(NAME mm3_lane_test COMMAND mm3_lane_test)
        set_tests_properties(mm3_lane_test PROPERTIES TIMEOUT 600)
        # the sampler (mm3_sample.hpp): candidates' distributions, penalties, speculative sampling's exactness
        add_executable(mm3_sampler_test tests/mm3_sampler_test.cpp)
        target_include_directories(mm3_sampler_test PRIVATE include src/mm3)
        add_test(NAME mm3_sampler_test COMMAND mm3_sampler_test)
        set_tests_properties(mm3_sampler_test PROPERTIES TIMEOUT 300)
        # the CPU lane's arithmetic (mm3_lane_plan.hpp): default workers with / without SMT, decode's re-timing
        add_executable(mm3_lane_plan_test tests/mm3_lane_plan_test.cpp)
        target_include_directories(mm3_lane_plan_test PRIVATE src/mm3)
        add_test(NAME mm3_lane_plan_test COMMAND mm3_lane_plan_test)
        # --parallel's choice of conversation (mm3_slots.hpp): a short shared start never takes a long one's cache
        add_executable(mm3_slots_test tests/mm3_slots_test.cpp)
        target_include_directories(mm3_slots_test PRIVATE src/mm3)
        add_test(NAME mm3_slots_test COMMAND mm3_slots_test)
      endif()
    endif()
    if(_mm3_filter)
      foreach(_t maya-mm3 mm3_exl3_test mm3_gguf_test mm3_kv_test mm3_kernels_test)
        if(TARGET ${_t})
          set_target_properties(${_t} PROPERTIES CUDA_ARCHITECTURES "${_mm3_archs}")
        endif()
      endforeach()
      message(STATUS "Maya: maya-mm3 for ${_mm3_archs}")
    endif()
  endif()

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    # Optional real-weight fixture test; see docs/EXL3-HELPER.md for the fixture format and invocation.
    add_executable(hip_ds4_exl3_helper EXCLUDE_FROM_ALL tests/hip/ds4_exl3_helper.cpp)
    target_link_libraries(hip_ds4_exl3_helper PRIVATE strata_kernels strata_hip_runtime)
    add_executable(ds4_peer_allreduce_test EXCLUDE_FROM_ALL tests/ds4_peer_allreduce_test.cu
                       src/ds4/ds4_kernels.cu src/ds4/ds4_host_ar.cpp src/ds4/ds4_peer_calibrate.cu)
    target_include_directories(ds4_peer_allreduce_test PRIVATE ${_strata_gpu_include_directories} third_party/ggml)
    target_link_libraries(ds4_peer_allreduce_test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_executable(hip_expert_kernel_bench tests/hip/expert_kernel_bench.cpp)
    target_include_directories(hip_expert_kernel_bench PRIVATE third_party/ggml)
    target_link_libraries(hip_expert_kernel_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_expert_kernel_parity COMMAND hip_expert_kernel_bench --parity-only)
    set_tests_properties(hip_expert_kernel_parity PROPERTIES TIMEOUT 120)
    # the dense GEMV (mv) at the decode shapes: the RDNA3 kernel against the originals, bit for bit
    add_executable(glm_mv_bench tests/hip/glm_mv_bench.cu)
    target_include_directories(glm_mv_bench PRIVATE third_party/ggml)
    target_link_libraries(glm_mv_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME glm_mv_parity COMMAND glm_mv_bench --parity-only)
    set_tests_properties(glm_mv_parity PROPERTIES TIMEOUT 120)
    # the decode attention's indexer scores and MLA split attention: the take-two kernels against the originals, bit for bit
    add_executable(glm_attn_bench tests/hip/glm_attn_bench.cu)
    target_link_libraries(glm_attn_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME glm_attn_parity COMMAND glm_attn_bench --parity-only)
    set_tests_properties(glm_attn_parity PROPERTIES TIMEOUT 120)
    add_executable(ds4_attention_layout_bench EXCLUDE_FROM_ALL tests/ds4_attention_layout_bench.cu src/ds4/ds4_kernels.cu)
    target_include_directories(ds4_attention_layout_bench PRIVATE ${_strata_gpu_include_directories} third_party/ggml)
    target_link_libraries(ds4_attention_layout_bench PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    add_executable(hip_glm_expert_helper tests/hip/glm_expert_helper.cpp)
    target_include_directories(hip_glm_expert_helper PRIVATE third_party/ggml)
    target_link_libraries(hip_glm_expert_helper PRIVATE strata_engine)
    add_test(NAME hip_glm_expert_helper COMMAND hip_glm_expert_helper)
    set_tests_properties(hip_glm_expert_helper PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
      add_executable(hip_glm_handoff tests/hip/glm_handoff.cpp)
      target_link_libraries(hip_glm_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_glm_handoff COMMAND hip_glm_handoff)
      list(APPEND _strata_hip_tests hip_glm_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/glm_prefill_attention.cpp")
      add_executable(hip_glm_prefill_attention tests/hip/glm_prefill_attention.cpp)
      target_link_libraries(hip_glm_prefill_attention PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_glm_prefill_attention COMMAND hip_glm_prefill_attention)
      list(APPEND _strata_hip_tests hip_glm_prefill_attention)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(iq1_s_parity src/kernels/iq1_s_parity.cpp)
  target_include_directories(iq1_s_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq1_s_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  add_test(NAME iq1_s_parity COMMAND iq1_s_parity)
  # not a ctest: needs 2 GPUs + peer access; run by hand when a driver wedges on P2P (see NEXT-STEPS.md)
  add_executable(peer_probe src/kernels/peer_probe.cu)
  target_include_directories(peer_probe PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(peer_probe PRIVATE ${_strata_gpu_runtime_target})
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(glm_hc_parity src/kernels/glm_hc_parity.cpp)
  target_include_directories(glm_hc_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_hc_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_hc_parity COMMAND glm_hc_parity --selftest)

  add_executable(router_sigmoid_parity src/kernels/router_sigmoid_parity.cpp)
  target_include_directories(router_sigmoid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_sigmoid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_sigmoid_parity COMMAND router_sigmoid_parity --selftest)

  add_executable(glm_kda_parity src/kernels/glm_kda_parity.cpp)
  target_include_directories(glm_kda_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_kda_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_kda_parity COMMAND glm_kda_parity --selftest)

  add_executable(glm_ffn_parity src/kernels/glm_ffn_parity.cpp)
  target_include_directories(glm_ffn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_ffn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_ffn_parity COMMAND glm_ffn_parity --selftest)

  add_executable(glm_dsa_parity src/kernels/glm_dsa_parity.cpp)
  target_include_directories(glm_dsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_dsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_dsa_parity COMMAND glm_dsa_parity --selftest)

  # KV streaming of the DSA latent cache (glm_kv_stream.hpp): streamed vs resident attention, bitwise (GPU, synthetic)
  add_executable(glm_kv_stream_parity src/kernels/glm_kv_stream_parity.cpp)
  target_include_directories(glm_kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_kv_stream_parity COMMAND glm_kv_stream_parity --selftest)

  add_executable(glm_layer_parity src/kernels/glm_layer_parity.cpp)
  target_include_directories(glm_layer_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_layer_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME glm_layer_parity COMMAND glm_layer_parity --selftest)
  set_tests_properties(glm_layer_parity PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # On HIP the GLM prompt GEMMs can go through strata::prefill::Gemm (hipBLASLt), so the GLM binaries link
  # strata_prefill after strata_engine, as `strata` does.
  if(STRATA_ENABLE_HIP)
    set(_strata_gpu_glm_libs strata_engine strata_prefill)
  else()
    set(_strata_gpu_glm_libs strata_engine)
  endif()
  add_executable(glm_model_test src/core/glm_model_test.cpp)
  target_include_directories(glm_model_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_model_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  add_test(NAME glm_model_test COMMAND glm_model_test --selftest)
  set_tests_properties(glm_model_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # the 45-layer big fixture: its 1.2 GB GGUF is generated into the build tree by a ctest
  # fixture (far too large to commit); the f64 expectation IS committed (data/glm5-big-dumps)
  add_executable(glm_model_big_test src/core/glm_model_big_test.cpp)
  if(NOT WIN32)   # pread / unistd.h
    add_executable(glm_cpu_bench src/core/glm_cpu_bench.cpp)
    target_link_libraries(glm_cpu_bench PRIVATE strata_kernels_cpu)
  endif()
  # the unified-memory pool/RAM-tier sizing policy (host only)
  add_executable(glm_memory_test tests/glm_memory_test.cpp)
  add_test(NAME glm_memory_test COMMAND glm_memory_test)
  # the devices of a layer split beside the expert helper (host only)
  add_executable(glm_split_devices_test tests/glm_split_devices_test.cpp)
  target_include_directories(glm_split_devices_test PRIVATE include)
  add_test(NAME glm_split_devices_test COMMAND glm_split_devices_test)
  add_executable(glm_pack_test src/core/glm_pack_test.cpp)
  target_include_directories(glm_pack_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_pack_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  target_include_directories(glm_model_big_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(glm_model_big_test PRIVATE ${_strata_gpu_glm_libs} ${_strata_gpu_runtime_target})
  find_program(STRATA_PYTHON3 python3)
  if(STRATA_PYTHON3)
    add_test(NAME glm_model_big_gen
             COMMAND ${STRATA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_big_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(glm_model_big_gen PROPERTIES FIXTURES_SETUP glm5_big_gguf)
    add_test(NAME glm_model_big_test
             COMMAND glm_model_big_test ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(glm_model_big_test PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                         FIXTURES_REQUIRED glm5_big_gguf)
  endif()

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # PR #120: the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
  add_executable(kv_hybrid_parity src/kernels/kv_hybrid_parity.cpp)
  target_include_directories(kv_hybrid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp")
  add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
  target_include_directories(native_mmvq_multi PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_mmvq_multi PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test runs ON THE TARGET PC. `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  if(STRATA_ENABLE_CUDA)
    add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)
  else()
    add_test(NAME hip_device_selftest COMMAND strata-device --selftest)
  endif()

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu")
    add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
    target_include_directories(graph_registry_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(graph_registry_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME graph_registry_test COMMAND graph_registry_test)
  endif()

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/pinned_capture.cu")
    add_executable(pinned_capture bench/micro/pinned_capture.cu)
    target_include_directories(pinned_capture PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pinned_capture PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  if(STRATA_ENABLE_CUDA)
    set(STRATA_CUDA_TESTS ON)
  endif()
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  # the cost-aware window (maya-ds4's DS4_DRAFT_COST=1): synthetic round times and acceptance, the lookup unchanged
  add_executable(draft_cost_test src/spec/draft_cost_test.cpp)
  target_link_libraries(draft_cost_test PRIVATE strata_spec)
  add_test(NAME draft_cost_test COMMAND draft_cost_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # the allocation failures the recovery recognizes: a prompt-path failure ends the request, not the process
  # (header-only, no library to link)
  add_executable(prefill_failure_test src/prefill/failure_test.cpp)
  target_include_directories(prefill_failure_test PRIVATE include)
  add_test(NAME prefill_failure_test COMMAND prefill_failure_test)
endif()

set(_strata_cpu_sources src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
                        src/kernels/cpu/exl3_ref.cpp)
if(STRATA_CPU_X86)
  list(APPEND _strata_cpu_sources src/kernels/cpu/expert.cpp src/kernels/cpu/q2_avx2.cpp
       src/kernels/cpu/exl3_cpu.cpp)
else()
  list(APPEND _strata_cpu_sources src/kernels/cpu/portable.cpp)
endif()
add_library(strata_kernels_cpu STATIC ${_strata_cpu_sources})
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(STRATA_CPU_X86 AND MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(STRATA_CPU_X86)
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  # A toolchain whose assembler has no VNNI (#535: Rocky Linux 9's GCC 11 + binutils 2.35, "unsupported instruction
  # `vpdpbusd'") builds the kernel with AVX512-BW's u8 x s8 dot products instead - the same sums (expert.cpp) - and
  # MAYA_CPU_VNNI_EMULATE=ON chooses that on any toolchain (tests)
  option(MAYA_CPU_VNNI_EMULATE "CPU expert kernel: AVX512-BW in place of VNNI's dot products" OFF)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma")
  check_cxx_source_compiles([=[
#include <immintrin.h>
__m256i f(__m256i a, __m256i b, __m256i c) { return _mm256_dpbusd_epi32(a, b, c); }
__m512i g(__m512i a, __m512i b, __m512i c) { return _mm512_dpbusd_epi32(a, b, c); }
int main() { return 0; }
]=] MAYA_CPU_VNNI_ASM)
  unset(CMAKE_REQUIRED_FLAGS)
  if(MAYA_CPU_VNNI_ASM AND NOT MAYA_CPU_VNNI_EMULATE)
    set_source_files_properties(src/kernels/cpu/expert.cpp
                                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  else()
    if(NOT MAYA_CPU_VNNI_EMULATE)
      message(STATUS "Maya: the compiler or its assembler has no AVX512-VNNI - the CPU expert kernel uses AVX512-BW's "
                     "dot products (the same sums)")
    endif()
    set_source_files_properties(src/kernels/cpu/expert.cpp
                                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mf16c;-mfma"
                                           COMPILE_DEFINITIONS "STRATA_CPU_DPBUSD_BW=1")
  endif()
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(NOT STRATA_CPU_X86)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    if(NOT GGML_CPU_ARM_ARCH)
      set(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm+fp16" CACHE STRING "ggml-cpu ARM arch flags" FORCE)
    endif()
  elseif(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  if(STRATA_CPU_X86)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                             src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx512.cpp
                                             src/kernels/cpu/iq_avx512_one.cpp)
  else()
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp)
  endif()
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(STRATA_CPU_X86 AND MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    # Match the non-MSVC no-contraction mode: IQ1_M's subnormal-scale parity checks need it too.
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:strict")
    set_source_files_properties(src/kernels/cpu/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  elseif(STRATA_CPU_X86)
    # (no VNNI / VBMI needed - cpu_avx512bw_ok() is enough to run them; no FP contraction: their sums must take
    # ggml-cpu's float steps exactly, see iq_avx512.cpp)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma;-ffp-contract=off")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
    # the one-token IQ2_XXS rows: VBMI's multishift and BMI2's pdep (run only where cpu_avx512_ok() and BMI2 say so)
    set_source_files_properties(src/kernels/cpu/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mbmi2;-mf16c;-mfma;-ffp-contract=off")
    # (AVX-512F/BW only: Skylake-SP Xeons have no VNNI, and this kernel does not need it)
    set_source_files_properties(src/kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  # DeepSeek's DSpark markov head as Q8_0 blocks (DS4_MARKOV_Q8, src/ds4/ds4_markov_q8.hpp) byte for byte against
  # ggml's quantize_row_q8_0_ref, a head group member's rows against the whole head's (no GPU)
  add_executable(ds4_markov_q8_test tests/ds4_markov_q8_test.cpp)
  target_include_directories(ds4_markov_q8_test PRIVATE include)
  target_link_libraries(ds4_markov_q8_test PRIVATE ggml-base)
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds4_markov_q8_test COMMAND ds4_markov_q8_test)
    # GLM's split-base expert rows (the CPU lane on a GGUF's own gate/up/down tensors) vs the contiguous blob path
    add_executable(native_split_test src/kernels/cpu/native_split_test.cpp)
    target_link_libraries(native_split_test PRIVATE strata_kernels_cpu)
    add_test(NAME native_split_test COMMAND native_split_test)
    # the multi-token expert rows (iq_avx512 / iq_avx2) bit for bit against ggml-cpu's dot, and GLM's split rows for a
    # window of tokens against the same tokens one at a time
    if(STRATA_CPU_X86)   # (#471: x86-only kernels - native rows, the IQ1_S rows without pdep)
      add_executable(native_rows_exact_test src/kernels/cpu/native_rows_exact_test.cpp)
      target_link_libraries(native_rows_exact_test PRIVATE strata_kernels_cpu)
      add_test(NAME native_rows_exact_test COMMAND native_rows_exact_test)
      # DeepSeek's CPU-lane IQ1_S rows (src/ds4/ds4_iq_cpu.cpp) bit for bit against ggml-cpu's dot: the form without pdep
      # (AMD Zen / Zen+ / Zen 2's default) forced on any CPU, and the measured choice
      add_executable(ds4_iq1_rows_test tests/ds4_iq1_rows_test.cpp src/ds4/ds4_iq_cpu.cpp)
      target_include_directories(ds4_iq1_rows_test PRIVATE third_party/ggml)
      target_link_libraries(ds4_iq1_rows_test PRIVATE strata_kernels_cpu)
      add_test(NAME ds4_iq1_rows_nopdep COMMAND ds4_iq1_rows_test)
      set_tests_properties(ds4_iq1_rows_nopdep PROPERTIES ENVIRONMENT DS4_IQ1_PDEP=0)
      add_test(NAME ds4_iq1_rows_measured COMMAND ds4_iq1_rows_test)
      set_tests_properties(ds4_iq1_rows_measured PROPERTIES ENVIRONMENT DS4_IQ1_PDEP=1)
    endif()
  endif()
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    # the dense GEMV for several conversations at once (mv_nt) against the one-conversation GEMV, bit for bit; the
    # second run with the generic kernel alone (synthetic weights from ggml's quantizers)
    add_executable(glm_mv_nt_test tests/glm_mv_nt_test.cu)
    target_include_directories(glm_mv_nt_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_mv_nt_test PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
    add_test(NAME glm_mv_nt_test COMMAND glm_mv_nt_test)
    add_test(NAME glm_mv_nt_generic COMMAND glm_mv_nt_test)
    set_tests_properties(glm_mv_nt_generic PROPERTIES ENVIRONMENT STRATA_GLM_MV_GENERIC=1)
    set_tests_properties(glm_mv_nt_test glm_mv_nt_generic PROPERTIES TIMEOUT 300)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    # The build-tree copy of ggml-cuda gets third_party/ggml/mmq-fused-swiglu.patch: fused expert SwiGLU and
    # explicit IQ1_S/IQ2_S/IQ3_S register-byte extraction. NVCC 13.2 on Windows can miscompile local byte-pointer
    # aliases into out-of-bounds codebook reads (covered by the independent CPU-dot regression).
    # Padded destination IDs in partial expert tiles are also guarded before loading shared memory.
    # The checkout stays untouched. The copy is redone when the checkout, its mmq.cuh or the patch changes. A checkout the patch
    # does not apply to builds from its own sources, and the prompt path keeps the separate SwiGLU kernel.
    set(_strata_mmq_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/mmq-fused-swiglu.patch)
    set(_strata_mmq_copy ${CMAKE_BINARY_DIR}/strata-ggml-cuda)
    file(SHA256 ${_strata_mmq_patch} _strata_mmq_patch_sha)
    file(SHA256 ${_strata_gg}/src/ggml-cuda/mmq.cuh _strata_mmq_src_sha)
    set(_strata_mmq_stamp "${_strata_gg}|${_strata_mmq_src_sha}|${_strata_mmq_patch_sha}")
    set(_strata_mmq_done "")
    if(EXISTS ${_strata_mmq_copy}/.strata-stamp)
      file(READ ${_strata_mmq_copy}/.strata-stamp _strata_mmq_done)
    endif()
    set(_strata_mmq_glu ON)
    if(NOT _strata_mmq_done STREQUAL _strata_mmq_stamp)
      find_package(Git)
    endif()
    if(NOT _strata_mmq_done STREQUAL _strata_mmq_stamp AND NOT GIT_FOUND)
      # (a source zip without git - docs/WINDOWS.md offers one: the checkout's own MMQ sources, as for a patch that
      # does not apply)
      message(WARNING "git was not found to apply third_party/ggml/mmq-fused-swiglu.patch: the pinned MMQ sources are "
                      "built as they are, and the prompt path keeps the separate SwiGLU kernel")
      set(_strata_mmq_glu OFF)
    elseif(NOT _strata_mmq_done STREQUAL _strata_mmq_stamp)
      file(REMOVE_RECURSE ${_strata_mmq_copy})
      file(COPY ${_strata_gg}/src/ggml-cuda/ DESTINATION ${_strata_mmq_copy})
      # (GIT_CEILING_DIRECTORIES: git applies the patch as a plain patch tool, not inside the repository holding the
      # build tree)
      execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                              ${GIT_EXECUTABLE} apply --ignore-whitespace ${_strata_mmq_patch}
                      WORKING_DIRECTORY ${_strata_mmq_copy} RESULT_VARIABLE _strata_mmq_rc ERROR_VARIABLE _strata_mmq_err)
      if(_strata_mmq_rc EQUAL 0)
        file(WRITE ${_strata_mmq_copy}/.strata-stamp "${_strata_mmq_stamp}")
      else()
        message(WARNING "third_party/ggml/mmq-fused-swiglu.patch (made for the pinned ggml, third_party/ggml/VERSION.txt) "
                        "does not apply to ${_strata_gg}/src/ggml-cuda - its own MMQ sources are built and "
                        "the prompt path keeps the separate SwiGLU kernel:\n${_strata_mmq_err}")
        file(REMOVE_RECURSE ${_strata_mmq_copy})
        set(_strata_mmq_glu OFF)
      endif()
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_strata_mmq_patch} ${_strata_gg}/src/ggml-cuda/mmq.cuh)
    if(_strata_mmq_glu)
      set(_strata_mmq ${_strata_mmq_copy})
    else()
      set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    endif()
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu
                ${_strata_mmq}/convert.cu)
    if(_strata_mmq_glu)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_GLU=1)
    endif()
    foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 iq1_s q2_k q3_k q4_k q5_k q6_k)   # q8_0: the drafter's matrices (E-9); iq1_s, k-quants: GLM experts (docs/QUANT-PLAN.md)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)
    # the GLM fast path's prompt path (src/core/glm_prefill.cu) multiplies its experts with MMQ too
    target_link_libraries(strata_engine PUBLIC strata_mmq ${_strata_gpu_blas_target})
    if(STRATA_BUILD_TESTS)
      # the fused gate/up + SwiGLU product against the plain product + the SwiGLU kernel, bit for bit (synthetic
      # weights from ggml's quantizers)
      add_executable(glm_mmq_glu_test tests/glm_mmq_glu_test.cu)
      target_include_directories(glm_mmq_glu_test PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(glm_mmq_glu_test PRIVATE strata_mmq strata_kernels ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME glm_mmq_glu_test COMMAND glm_mmq_glu_test)
      set_tests_properties(glm_mmq_glu_test PROPERTIES TIMEOUT 300)
    endif()
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    if(STRATA_GGML_DIR)
      set(_strata_gg "${STRATA_GGML_DIR}/ggml")
    else()
      set(_strata_gg "${strata_llamacpp_SOURCE_DIR}/ggml")
    endif()
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/moe_fused_rdna4.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_mmq}/quantize.cu"
      "${_strata_mmq}/convert.cu")
    target_compile_definitions(strata_engine PRIVATE STRATA_HIP_PREFILL_FUSED=1)
    foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 iq1_s q2_k q3_k q4_k q5_k q6_k)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)

    # Maya's GLM batched prompt path calls MMQ and BLAS directly, as on CUDA.
    target_link_libraries(strata_engine PUBLIC strata_mmq ${_strata_gpu_blas_target})
    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
    add_executable(hip_prefill_fused_parity tests/hip/prefill_fused_parity.cpp)
    target_include_directories(hip_prefill_fused_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_parity PRIVATE strata_prefill strata_kernels)
    add_test(NAME hip_prefill_fused_parity COMMAND hip_prefill_fused_parity)
    set_tests_properties(hip_prefill_fused_parity PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prefill_fused_parity)
    # the fused experts' speed at a layer's shape (not a test: hip_prefill_fused_bench [tokens] [reps] [gu] [down] [ds4])
    add_executable(hip_prefill_fused_bench EXCLUDE_FROM_ALL tests/hip/prefill_fused_bench.cpp)
    target_include_directories(hip_prefill_fused_bench PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_bench PRIVATE strata_prefill strata_kernels)
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)   # #93: it needs the device dequantizers and a GPU runtime
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)
  # the EXL3 CPU lane's tiers on random experts: a band's compute speed, and an expert's time on T threads from DRAM
  # in the engine's jobs (what the start-up timing compares with a copy over PCIe) - a benchmark, not a test
  if(STRATA_CPU_X86)
    add_executable(exl3_cpu_bench src/kernels/cpu/exl3_cpu_bench.cpp)
    target_link_libraries(exl3_cpu_bench PRIVATE strata_kernels_cpu)
  endif()

  if(WIN32)
    add_executable(ds4_copy_cache_windows_test tests/ds4_copy_cache_windows_test.cpp)
    target_include_directories(ds4_copy_cache_windows_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME ds4_copy_cache_windows_test COMMAND ds4_copy_cache_windows_test)
  endif()

  if(UNIX AND NOT APPLE)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp src/platform/direct_file.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
    add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
  # maya-ds4's RAM expert store (--ssd-cache, src/ds4/ds4_store.hpp) over files of fake experts written in the build
  # directory (~1 MB, removed): reads, where they land, eviction, the read queue, giving RAM back, preload, the miss
  # split's order, put - host code only, no GPU
  add_executable(ds4_store_test tests/ds4_store_test.cpp)
  target_include_directories(ds4_store_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ds4)
  target_link_libraries(ds4_store_test PRIVATE strata_warnings Threads::Threads)
  add_test(NAME ds4_store_test COMMAND ds4_store_test ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(ds4_store_test PROPERTIES TIMEOUT 120)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()
