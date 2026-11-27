# tools/vision/lean.cmake - maya-vision compiles only what Maya's picture reader runs (included by CMakeLists.txt
# after llama.cpp's targets exist, unless MAYA_VISION_FULL=ON).  The pinned llama.cpp checkout stays untouched: the
# targets' source lists are shortened here, and what the remaining code still names gets a function that stops with
# a message (written into the build folder).  No code that chooses a kernel or a path is left out or replaced.

function(_maya_write_if_changed path text)
  set(_old "")
  if(EXISTS ${path})
    file(READ ${path} _old)
  endif()
  if(NOT _old STREQUAL text)
    file(WRITE ${path} "${text}")
  endif()
endfunction()

# ---- 1. mtmd: the projector files of the towers Maya runs; every other clip_graph_<x>::build() stops
set(_mtmd_dir ${LLAMA_DIR}/tools/mtmd)
get_target_property(_src mtmd SOURCES)
set(_keep "")
set(_stubs "")
set(_seen "")
set(_n_out 0)
foreach(_f IN LISTS _src)
  if(NOT _f MATCHES "^models/([A-Za-z0-9_.-]+)\\.cpp$")
    list(APPEND _keep ${_f})
    continue()
  endif()
  if(CMAKE_MATCH_1 IN_LIST MAYA_VISION_TOWERS)
    list(APPEND _keep ${_f})
    continue()
  endif()
  math(EXPR _n_out "${_n_out} + 1")
  # (only definition lines are read - the start of a line, up to the "(" - so no line with a ";" or "[" enters a list)
  file(STRINGS ${_mtmd_dir}/${_f} _defs REGEX "^(ggml_cgraph \\* clip_graph_[A-Za-z0-9_]+::build\\(\\)|ggml_tensor \\* clip_graph_[A-Za-z0-9_]+::build_mm\\(ggml_tensor \\* w, ggml_tensor \\* x\\) const|std::vector<c2w_state_slot> list_[A-Za-z0-9_]+\\(const clip_hparams & hparams, const clip_model & model\\))")
  foreach(_d IN LISTS _defs)
    string(REGEX MATCH "^[^{]*\\)( const)?" _sig "${_d}")
    if(_sig AND NOT _sig IN_LIST _seen)
      list(APPEND _seen "${_sig}")
      string(APPEND _stubs "${_sig} { maya_vision_left_out(\"${_f}\"); }\n")
    endif()
  endforeach()
endforeach()
list(LENGTH MAYA_VISION_TOWERS _n_in)
if(_n_out GREATER 0)
  set(_stub ${CMAKE_BINARY_DIR}/maya-lean/mtmd_left_out.cpp)
  _maya_write_if_changed(${_stub} "// written by tools/vision/lean.cmake: the projectors this build leaves out stop here
#include \"${_mtmd_dir}/models/models.h\"
#include <cstdio>
#include <cstdlib>
[[noreturn]] static void maya_vision_left_out(const char * what) {
    std::fprintf(stderr, \"maya-vision: this picture model's projector (%s) is not part of this build - it has the \"
                         \"towers of the models Maya runs; -DMAYA_VISION_FULL=ON builds every one\\n\", what);
    std::abort();
}
${_stubs}")
  set_property(TARGET mtmd PROPERTY SOURCES ${_keep} ${_stub})
  message(STATUS "maya-vision: ${_n_in} projector files (${MAYA_VISION_TOWERS}), ${_n_out} of llama.cpp's left out")
endif()

# ---- 2. ggml's CUDA / HIP backend: no quantized-weight matmul kernels (one template instance per GGUF quant type).
# mmq.cu - which decides when they run, and calls them - stays; an instance left out is one that stops.
foreach(_t ggml-cuda ggml-hip)
  if(NOT TARGET ${_t})
    continue()
  endif()
  get_target_property(_src ${_t} SOURCES)
  get_target_property(_tdir ${_t} SOURCE_DIR)
  set(_keep "")
  set(_inst "")
  set(_like "")
  foreach(_f IN LISTS _src)
    if(_f MATCHES "template-instances/mmq-instance-[a-z0-9_]+\\.cu$")
      get_filename_component(_abs ${_f} ABSOLUTE BASE_DIR ${_tdir})
      file(STRINGS ${_abs} _decl REGEX "^DECL_MMQ_CASE\\(GGML_TYPE_[A-Z0-9_]+\\)")
      string(REGEX MATCH "GGML_TYPE_[A-Z0-9_]+" _ty "${_decl}")
      if(NOT _ty)
        list(APPEND _keep ${_f})                       # (not the file this expects: compiled as it is)
        continue()
      endif()
      string(APPEND _inst "template void mul_mat_q_case<${_ty}>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);\n")
    else()
      list(APPEND _keep ${_f})
      if(_f MATCHES "mmq\\.cu$")
        set(_like ${_f})
      endif()
    endif()
  endforeach()
  if(_inst AND _like)
    get_filename_component(_cuda_dir ${_like} ABSOLUTE BASE_DIR ${_tdir})
    get_filename_component(_cuda_dir ${_cuda_dir} DIRECTORY)
    set(_stub ${CMAKE_BINARY_DIR}/maya-lean/mmq_left_out.cu)
    _maya_write_if_changed(${_stub} "// written by tools/vision/lean.cmake: the quantized-weight matmul kernels this build leaves out stop here
// (the same declaration as mmq.cuh's mul_mat_q_case, whose instances mmq.cu calls)
#include \"${_cuda_dir}/common.cuh\"
#include <cstdio>
#include <cstdlib>
struct mmq_args;
template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    (void) ctx; (void) args; (void) stream;
    std::fprintf(stderr, \"maya-vision: this build has no kernels for quantized weights (ggml type %d): an mmproj's \"
                         \"are F32/F16; -DMAYA_VISION_FULL=ON builds them\\n\", (int) type);
    std::abort();
}
${_inst}")
    # the language the backend's own files are compiled as (CUDA; HIP, or C++ with ROCm's clang on Windows)
    get_source_file_property(_lang ${_like} TARGET_DIRECTORY ${_t} LANGUAGE)
    if(_lang)
      set_source_files_properties(${_stub} TARGET_DIRECTORY ${_t} PROPERTIES LANGUAGE ${_lang})
    endif()
    set_property(TARGET ${_t} PROPERTY SOURCES ${_keep} ${_stub})
    message(STATUS "maya-vision: ${_t} without the quantized-weight matmul instances")
  endif()
endforeach()
