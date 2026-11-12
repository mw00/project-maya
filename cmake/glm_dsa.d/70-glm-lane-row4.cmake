# glm-dsa wave 8, B14.9: the lockstep 4-row CPU decode GEMV for the CPU expert lane (MAYA_GLM_LANE_ROW4=1).
#
# glm_lane_row4.cpp joins strata_kernels_cpu next to Maya's native_expert.cpp (whose nt = 1 split-row functions call
# its two hooks, STRATA-EDITS.txt) on every native-experts build, so it exists exactly when the ggml-cpu path does.
# It holds the switch, the CPU check and the hooks and is compiled like native_expert.cpp (no AVX flags: the hooks run
# on every one-token step, lane on or off, on any CPU).  The kernels are x86-64 only: glm_lane_row4_avx2.cpp, AVX2 +
# FMA + F16C (no VNNI / VBMI: the Xeon Gold 6138 runs them; -ffp-contract=off keeps their float ops exactly ggml's),
# with STRATA_GLM_LANE_ROW4_KERNELS defined on both files.  Every other build - aarch64 (DGX Spark / Grace, which
# CMakeLists.txt gives the portable CPU code), ppc64le (IBM AC922), or x86-64 with -DSTRATA_GLM_LANE_ROW4_KERNELS=OFF,
# which builds exactly what those machines build - gets glm_lane_row4.cpp's no-lane branch: the hooks always return
# false and the nt = 1 loops keep ggml's vec_dot, as before the port.
option(STRATA_GLM_LANE_ROW4_KERNELS
       "glm-dsa CPU lane: build the x86-64 AVX2 4-row GEMV (OFF: its hooks always decline, as on other CPUs)" ON)
if(STRATA_NATIVE_EXPERTS AND TARGET strata_kernels_cpu AND TARGET ggml-cpu)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/glm_lane_row4.cpp)
  if(NOT MSVC)
    set_source_files_properties(src/kernels/cpu/glm_lane_row4.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()

  if(STRATA_CPU_X86 AND STRATA_GLM_LANE_ROW4_KERNELS)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/glm_lane_row4_avx2.cpp)
    set_source_files_properties(src/kernels/cpu/glm_lane_row4.cpp src/kernels/cpu/glm_lane_row4_avx2.cpp
                                PROPERTIES COMPILE_DEFINITIONS STRATA_GLM_LANE_ROW4_KERNELS=1)
    if(MSVC)
      set_source_files_properties(src/kernels/cpu/glm_lane_row4_avx2.cpp PROPERTIES COMPILE_OPTIONS
                                  "/arch:AVX2;/fp:precise")
    else()
      set_source_files_properties(src/kernels/cpu/glm_lane_row4_avx2.cpp PROPERTIES COMPILE_OPTIONS
                                  "-mavx2;-mfma;-mf16c;-ffp-contract=off")
    endif()

    if(STRATA_BUILD_TESTS)
      enable_testing()
      # CPU, B14.9: row4 vs ggml-cpu's vec_dot bit for bit on random blocks of IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS /
      # Q2_K / Q3_K / IQ1_S / IQ4_XS, row counts 1..9 at several offsets, +-127 and -128 / +127 activations, all-0xFF /
      # all-0x00 weights and qh / scale fields, the IQ1_S delta-term probe, the lane hooks (native_gu_rows_split /
      # native_down_rows_split at nt = 1, switch on vs off) and lane_row4_set.  `glm_lane_row4_test --bench` adds the
      # per-thread GB/s microbenchmark (ggml vec_dot vs row4); it is not part of ctest.
      add_executable(glm_lane_row4_test tests/glm_dsa/glm_lane_row4_test.cpp)
      target_link_libraries(glm_lane_row4_test PRIVATE strata_kernels_cpu)
      add_test(NAME glm_lane_row4_test COMMAND glm_lane_row4_test)
    endif()
  endif()
endif()
