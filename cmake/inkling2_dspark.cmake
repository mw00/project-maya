# cmake/inkling2_dspark.cmake - maya-inkling2's drafter (Inkling 2 Small's DSpark block: src/inkling2/inkling2_dspark,
# docs/INKLING2_DSPARK.md) and its drafter-only test.  Included from CMakeLists.txt beside the engine's file
# (_strata_gpu_* in scope).  The engine links the module by adding src/inkling2/inkling2_dspark.cu to its sources.
#   inkling2-dspark-test  the drafts of a reference run (llama.cpp PR #32218: tools/inkling2/ref/inkling-dspark-eval.cpp)
#                         from its saved taps - needs the model's files and a GPU, so it is not a ctest
add_executable(inkling2-dspark-test tests/inkling2_dspark_test.cpp src/inkling2/inkling2_dspark.cu
                                    src/inkling2/inkling2_kernels.cu src/hy4/hy4_kernels.cu
                                    src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp src/ds4/ds4_gguf_cpu.cpp)
target_include_directories(inkling2-dspark-test PRIVATE ${_strata_gpu_include_directories} include src/inkling2 src/hy4
                                                        src/mm3 third_party/ggml)
# (the attention's GEMM form: the BLAS the engine links)
target_link_libraries(inkling2-dspark-test PRIVATE strata_kernels_cpu ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target}
                                                   Threads::Threads)
target_compile_definitions(inkling2-dspark-test PRIVATE MAYA_DS4_IQ_ONLY)
