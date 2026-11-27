# Windows (MSVC, x86): ggml-cpu's SIMD level must not depend on CMake test programs being allowed to run.
#
# With GGML_NATIVE, ggml's FindSIMD.cmake decides AVX / AVX2 + FMA / AVX-512 by RUNNING small freshly built
# programs (check_c_source_runs).  Windows Smart App Control (or another allow-list tool) can block those exes; a
# blocked check reads as "this CPU lacks it", the result is cached, and ggml-cpu is built for /arch:AVX only - no
# error, but the CPU expert lane is much slower.  Here the answer comes from the CPU itself (CPUID + XGETBV, read by
# cmake/win_cpu_features.py, nothing built or run): what the CPU has is written into FindSIMD's cached results (the
# same values a successful check stores), so its test programs for those are not run at all; what the CPU lacks is
# still checked by FindSIMD as before.  MAYA_WIN_CPU_FEATURES (avx,avx2,fma,f16c,avx512) gives the answer instead
# (maya.py passes the one it read); empty: this CPU is asked.  Every line printed starts with "Maya CPU guard:".
set(MAYA_WIN_CPU_FEATURES "" CACHE STRING "Windows: this CPU's SIMD for ggml-cpu (avx,avx2,fma,f16c,avx512); empty = ask the CPU")

function(maya_win_cpu_guard_seed)
  set(_feat "${MAYA_WIN_CPU_FEATURES}")
  set(_from "MAYA_WIN_CPU_FEATURES")
  if(_feat STREQUAL "")
    set(_from "this CPU (CPUID)")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(COMMAND "${Python3_EXECUTABLE}" -I "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/win_cpu_features.py"
                      OUTPUT_VARIABLE _feat RESULT_VARIABLE _rc ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                      TIMEOUT 60)
      if(NOT _rc EQUAL 0)
        set(_feat "")
      endif()
    endif()
  endif()
  string(TOLOWER "${_feat}" _feat)
  string(REPLACE "," ";" _feat "${_feat}")
  if(NOT _feat)
    set(MAYA_WIN_CPU_KNOWN OFF PARENT_SCOPE)
    message(STATUS "Maya CPU guard: the CPU's SIMD could not be read (no Python?) - ggml's own run checks decide")
    return()
  endif()
  set(MAYA_WIN_CPU_KNOWN ON PARENT_SCOPE)
  # FindSIMD's types: AVX2 is taken only together with FMA (ggml's MSVC AVX2 build is AVX2 + FMA + F16C)
  set(_types "")
  if("avx" IN_LIST _feat)
    list(APPEND _types AVX)
  endif()
  if("avx2" IN_LIST _feat AND "fma" IN_LIST _feat)
    list(APPEND _types AVX2 FMA)
  endif()
  if("avx512" IN_LIST _feat)
    list(APPEND _types AVX512)
  endif()
  set(_fixed "")
  foreach(_t IN LISTS _types)
    if(DEFINED CACHE{${_t}_FOUND} AND NOT "$CACHE{${_t}_FOUND}")
      list(APPEND _fixed ${_t})          # an earlier configure's check said no (its program was blocked)
    endif()
    set(${_t}_FOUND TRUE CACHE BOOL "${_t} support" FORCE)
  endforeach()
  string(REPLACE ";" " " _shown "${_types}")
  message(STATUS "Maya CPU guard: ggml-cpu SIMD from ${_from}: ${_shown}")
  if(_fixed)
    string(REPLACE ";" " " _fixed "${_fixed}")
    message(STATUS "Maya CPU guard: corrected ${_fixed}: an earlier configure's check program did not run "
                   "(Windows Smart App Control can block them); the CPU has it, so ggml-cpu is built with it")
  endif()
endfunction()

# After ggml: without the CPU's own answer, say so when a check program could not run (never a silent AVX build).
function(maya_win_cpu_guard_check)
  if(MAYA_WIN_CPU_KNOWN)
    return()
  endif()
  set(_blocked "")
  foreach(_v HAS_AVX_1 HAS_AVX_2 HAS_AVX2_1 HAS_AVX2_2 HAS_FMA_1 HAS_FMA_2 HAS_AVX512_1 HAS_AVX512_2)
    if("$CACHE{${_v}_EXITCODE}" MATCHES "FAILED_TO_RUN")
      list(APPEND _blocked ${_v})
    endif()
  endforeach()
  if(_blocked AND (NOT "$CACHE{AVX2_FOUND}" OR NOT "$CACHE{FMA_FOUND}"))
    string(REPLACE ";" " " _blocked "${_blocked}")
    message(WARNING "Maya CPU guard: CMake check programs could not run (${_blocked}; Windows Smart App Control "
                    "can block them), so ggml-cpu may be built without AVX2 and run slower.  If this CPU has AVX2, "
                    "configure again with -DMAYA_WIN_CPU_FEATURES=avx,avx2,fma,f16c (add avx512 if it has AVX-512).")
  endif()
endfunction()
