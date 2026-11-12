# tests/exl3_dump_compare.cmake - v1.1.8's EXL3 kernels with the FP32 weight decode against the FP16 one, bit for bit:
# exl3_test --dump-legacy under MAYA_GLM_EXL3_DECODE=legacy and =legacy-f32, the two files the same bytes, each run's
# start-up line naming its decode.  Unset (the default) never takes the FP32 decode; where it takes v1.1.8's kernels
# (sm_6x / sm_7x: it must) it writes the legacy run's bytes too.
#
# The FP32 decode is built for sm_6x only: on compute capability 7 and later legacy-f32 is accepted and runs the FP16
# kernels - the start-up line "exl3: CUDA<n> decodes with v1.1.8's kernels (legacy-f32 is built for sm_6x only; the
# same bits)" and legacy_f32() false; the same bytes.  MAYA_GLM_EXL3_DECODE<n> (device n's own setting) the same,
# over the global setting; one that names no decode falls back to the global one.
#
# Each invocation writes its dumps to a directory of its own under OUT (two runs in one build directory at once, e.g.
# ctest on two GPUs, would otherwise overwrite each other's files) and removes it when every check passes.
#
#   cmake -DEXE=<exl3_test> -DOUT=<dir> -P tests/exl3_dump_compare.cmake
foreach(_v EXE OUT)
  if(NOT DEFINED ${_v})
    message(FATAL_ERROR "exl3_dump_compare: -D${_v}=... missing")
  endif()
endforeach()

# this run's directory: a timestamp and random characters (string(RANDOM) seeds from the system's random source, so
# runs started in the same second differ), made fresh
string(TIMESTAMP _stamp "%Y%m%d-%H%M%S")
foreach(_try RANGE 8)
  string(RANDOM LENGTH 12 ALPHABET "0123456789abcdefghijklmnopqrstuvwxyz" _rnd)
  set(DIR "${OUT}/run-${_stamp}-${_rnd}")
  if(NOT EXISTS "${DIR}")
    break()
  endif()
endforeach()
if(EXISTS "${DIR}")
  message(FATAL_ERROR "exl3_dump_compare: no fresh directory under ${OUT}")
endif()
file(MAKE_DIRECTORY "${DIR}")

set(_line_legacy "exl3: CUDA[0-9]+ decodes with v1\\.1\\.8's kernels\n")
set(_line_f32 "exl3: CUDA[0-9]+ decodes with v1\\.1\\.8's kernels, FP32 decode\n")
set(_line_f32_off "exl3: CUDA[0-9]+ decodes with v1\\.1\\.8's kernels \\(legacy-f32 is built for sm_6x only; the same bits\\)\n")
# the environment every run starts from: no decode setting of the caller's (exl3_test runs on device 0: its own
# setting is MAYA_GLM_EXL3_DECODE0)
set(_clean_env --unset=MAYA_GLM_EXL3_DECODE --unset=STRATA_GLM_EXL3_DECODE
               --unset=MAYA_GLM_EXL3_DECODE0 --unset=STRATA_GLM_EXL3_DECODE0)

# one run with the settings in ARGN (NAME=value): its log (stdout and stderr) in <var>; a non-zero exit fails the test
function(_dump tag var)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env ${_clean_env} ${ARGN}
                          ${EXE} --dump-legacy ${DIR}/exl3_dump_${tag}.bin
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _so ERROR_VARIABLE _se)
  message("---- ${tag}: ${ARGN}\n${_so}${_se}")
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "exl3_test --dump-legacy (${tag}: ${ARGN}) exited ${_rc}; its files are in ${DIR}")
  endif()
  set(${var} "${_so}${_se}" PARENT_SCOPE)
endfunction()

# the per-output hash lines of two logs side by side: the first outputs that differ
function(_diff_lines a b)
  string(REGEX MATCHALL "  [^\n]* bits [0-9a-f]+" _la "${a}")
  string(REGEX MATCHALL "  [^\n]* bits [0-9a-f]+" _lb "${b}")
  list(LENGTH _la _n)
  set(_shown 0)
  if(_n GREATER 0)
    math(EXPR _last "${_n} - 1")
    foreach(_i RANGE ${_last})
      list(GET _la ${_i} _x)
      list(LENGTH _lb _nb)
      if(_i LESS _nb)
        list(GET _lb ${_i} _y)
      else()
        set(_y "(missing)")
      endif()
      if(NOT _x STREQUAL _y AND _shown LESS 5)
        message("  differs:\n   ${_x}\n   ${_y}")
        math(EXPR _shown "${_shown} + 1")
      endif()
    endforeach()
  endif()
endfunction()

# what one run's log must show; mode: plain (v1.1.8's FP16 kernels), f32 (legacy-f32: the FP32 decode on sm_6x, the
# FP16 kernels with their own line on sm_70+), default (unset: never the FP32 decode, v1.1.8's kernels on sm_6x / 7x)
function(_expect tag log mode)
  set(_bad "${_bad_all}")
  if(NOT log MATCHES "dump-legacy: sm_([0-9]+)")
    list(APPEND _bad "${tag}: no \"dump-legacy: sm_<cc>\" line")
    set(_bad_all "${_bad}" PARENT_SCOPE)
    return()
  endif()
  set(_cc "${CMAKE_MATCH_1}")
  math(EXPR _major "${_cc} / 10")   # (the minor version is one digit)
  if(mode STREQUAL "f32" AND _major LESS 7)
    if(NOT log MATCHES "${_line_f32}")
      list(APPEND _bad "${tag}: no \"exl3: CUDA<n> decodes with v1.1.8's kernels, FP32 decode\" line")
    endif()
    if(NOT log MATCHES "legacy_f32\\(\\) true")
      list(APPEND _bad "${tag}: legacy_f32() is not true on sm_6x")
    endif()
    if(log MATCHES "legacy-f32 is built for sm_6x only")
      list(APPEND _bad "${tag}: sm_6x calls legacy-f32 unbuilt")
    endif()
  elseif(mode STREQUAL "f32")
    if(NOT log MATCHES "${_line_f32_off}")
      list(APPEND _bad "${tag}: no \"exl3: CUDA<n> decodes with v1.1.8's kernels (legacy-f32 is built for sm_6x only, the same bits)\" line on sm_${_cc}")
    endif()
    if(log MATCHES "FP32 decode")
      list(APPEND _bad "${tag}: sm_70+ names the FP32 decode")
    endif()
    if(NOT log MATCHES "legacy_f32\\(\\) false")
      list(APPEND _bad "${tag}: legacy_f32() is not false on sm_70+")
    endif()
  else()
    if(log MATCHES "FP32 decode")
      list(APPEND _bad "${tag}: its log names the FP32 decode")
    endif()
    if(log MATCHES "legacy-f32 is built for sm_6x only")
      list(APPEND _bad "${tag}: its log names legacy-f32")
    endif()
    if(NOT log MATCHES "legacy_f32\\(\\) false")
      list(APPEND _bad "${tag}: legacy_f32() is not false")
    endif()
    if(mode STREQUAL "plain" AND NOT log MATCHES "${_line_legacy}")
      list(APPEND _bad "${tag}: no \"exl3: CUDA<n> decodes with v1.1.8's kernels\" line")
    endif()
    if(mode STREQUAL "default" AND (_major EQUAL 6 OR _major EQUAL 7) AND NOT log MATCHES "${_line_legacy}")
      list(APPEND _bad "${tag} on sm_${_cc}: no \"decodes with v1.1.8's kernels\" line (the default must stay legacy)")
    endif()
  endif()
  set(_bad_all "${_bad}" PARENT_SCOPE)
endfunction()

# run <tag>'s bytes against the legacy run's
function(_same tag log)
  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${DIR}/exl3_dump_legacy.bin ${DIR}/exl3_dump_${tag}.bin
                  RESULT_VARIABLE _rc)
  set(_bad "${_bad_all}")
  if(_rc EQUAL 0)
    message("exl3_dump_compare: legacy vs ${tag}: the same bytes")
  else()
    _diff_lines("${log_legacy}" "${log}")
    list(APPEND _bad "legacy vs ${tag}: the dumps differ")
  endif()
  set(_bad_all "${_bad}" PARENT_SCOPE)
endfunction()

set(_bad_all "")
_dump(legacy log_legacy MAYA_GLM_EXL3_DECODE=legacy)
_dump(legacy-f32 log_f32 MAYA_GLM_EXL3_DECODE=legacy-f32)
_dump(unset log_unset)
# the device's own setting (MAYA_GLM_EXL3_DECODE<n>) over the global one, both ways; one that names no decode falls
# back to the global one
_dump(dev-f32 log_dev_f32 MAYA_GLM_EXL3_DECODE=legacy MAYA_GLM_EXL3_DECODE0=legacy-f32)
_dump(dev-legacy log_dev_legacy MAYA_GLM_EXL3_DECODE=legacy-f32 MAYA_GLM_EXL3_DECODE0=legacy)
_dump(dev-invalid log_dev_invalid MAYA_GLM_EXL3_DECODE=legacy-f32 MAYA_GLM_EXL3_DECODE0=bogus)
_expect(legacy "${log_legacy}" plain)
_expect(legacy-f32 "${log_f32}" f32)
_expect(unset "${log_unset}" default)
_expect(dev-f32 "${log_dev_f32}" f32)
_expect(dev-legacy "${log_dev_legacy}" plain)
_expect(dev-invalid "${log_dev_invalid}" f32)
_same(dev-f32 "${log_dev_f32}")
_same(dev-legacy "${log_dev_legacy}")
_same(dev-invalid "${log_dev_invalid}")
# legacy-f32's bytes are legacy's (on sm_70+ too, where it runs the FP16 kernels)
_same(legacy-f32 "${log_f32}")
# the default: where it takes v1.1.8's kernels, their bytes
if(log_unset MATCHES "${_line_legacy}")
  _same(unset "${log_unset}")
else()
  message("exl3_dump_compare: unset takes another path here: not compared")
endif()

if(_bad_all)
  string(REPLACE ";" "\n  " _msg "${_bad_all}")
  message(FATAL_ERROR "exl3_dump_compare: FAIL (the dumps are kept in ${DIR})\n  ${_msg}")
endif()
file(REMOVE_RECURSE "${DIR}")
message("exl3_dump_compare: ok")
