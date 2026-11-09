# glm-dsa B8.1 (warm): the load-time expert warm-up in file order (STRATA_GLM_WARM_SEQ, on for glm-dsa) and the
# digest of what it loaded (STRATA_GLM_WARM_SUM).
#
# glm_warm_seq.cpp is plain C++ (the switches, the file order, the run count, the counters); glm_warm_seq_model.cu the
# engine side (the parts' turn on the disk, the digests).  Maya-owned glm_fast_path.cu calls both from fast_warm
# (STRATA-EDITS.txt), so they join strata_engine on any GPU backend (a HIP build compiles src/*.cu as HIP).
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_warm_seq.cpp src/core/glm_warm_seq_model.cu)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # CPU: the switches, file_order / read_order / runs, fast_warm's job list in slot order vs file order on mock packs
  # (3 runs a batch; printed at GLM-5.3's shape), the digest hash, the counters.  No engine.
  add_executable(glm_warm_seq_test tests/glm_dsa/glm_warm_seq_test.cpp src/core/glm_warm_seq.cpp)
  target_include_directories(glm_warm_seq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME glm_warm_seq_test COMMAND glm_warm_seq_test)
  set_tests_properties(glm_warm_seq_test PROPERTIES TIMEOUT 120)
endif()
