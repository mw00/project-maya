# cmake/k31_pack_roundtrip.cmake - the k31_pack_roundtrip test: tools/k31/pack_gguf.py packs the tiny synthetic K3.1
# fixture (written by tools/k31/geo_ref.py --write-tiny, as CI does), and k31_pack_test reads the pack with
# k31::read_pack and checks it against the GGUF and against k31::read's geometry and expert ids.  CPU only.
#   cmake -DPYTHON=<python3 with numpy> -DSRC=<source dir> -DPACK_TEST=<k31_pack_test> -DWORK=<scratch dir>
#         -P cmake/k31_pack_roundtrip.cmake
cmake_minimum_required(VERSION 3.24)
foreach(_v PYTHON SRC PACK_TEST WORK)
  if(NOT ${_v})
    message(FATAL_ERROR "k31_pack_roundtrip: -D${_v}=... is required")
  endif()
endforeach()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
function(run what)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "k31_pack_roundtrip: ${what} failed (exit ${_rc})\n${_out}${_err}")
  endif()
  message("${what}: ${_out}")
endfunction()
run("tiny fixture" "${PYTHON}" "${SRC}/tools/k31/geo_ref.py" "${WORK}/tiny.gguf" --write-tiny)
run("pack" "${PYTHON}" "${SRC}/tools/k31/pack_gguf.py" --model "${WORK}/tiny.gguf" --out "${WORK}/pack")
run("read back" "${PACK_TEST}" "${WORK}/pack/k31.txt")
