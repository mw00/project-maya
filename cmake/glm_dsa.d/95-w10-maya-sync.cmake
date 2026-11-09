# glm-dsa wave 10: the Maya v1.1.3 merge (docs/strata/engine/MAYA-SYNC.md).  glm_maya_sync.cpp holds the arch gates
# of Maya's new features (EXL3, DFlash2, --parallel's batched rows) and the expert profile's glm-dsa hint; its tests
# check them and the profile at the glm-dsa layout (79 x 256).  Sorted after 90-ci-tests.cmake, so it also labels the
# device-only tests v1.1.3 brought (they fail, not skip, without a GPU).

if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/glm_maya_sync.cpp)
endif()

if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

# W10 (CPU): the gates on both arches, a 79 x 256 profile ranked / written / read back, the Flash layout refused.
if(TARGET strata_engine)
  add_executable(glm_maya_sync_test tests/glm_dsa/glm_maya_sync_test.cpp)
  target_include_directories(glm_maya_sync_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(glm_maya_sync_test PRIVATE strata_engine)
  add_test(NAME glm_maya_sync_test COMMAND glm_maya_sync_test)
  set_tests_properties(glm_maya_sync_test PROPERTIES TIMEOUT 60)
endif()

# W10 (Python, CPU): tools/strata/maya_sync.py, tools/strata/glm_dsa_profile.py and maya.py's profile / draft hooks;
# the engine's reader reads the tool's profile through glm_maya_sync_test.
if(NOT STRATA_PYTHON3)
  find_program(STRATA_PYTHON3 python3)
endif()
if(STRATA_PYTHON3)
  add_test(NAME strata_py_test_glm_dsa_profile
           COMMAND ${STRATA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/strata/test_glm_dsa_profile.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(strata_py_test_glm_dsa_profile PROPERTIES TIMEOUT 120)
  if(TARGET glm_maya_sync_test)
    set_property(TEST strata_py_test_glm_dsa_profile APPEND PROPERTY ENVIRONMENT
                 "STRATA_MAYA_SYNC_BIN=$<TARGET_FILE:glm_maya_sync_test>")
  endif()
endif()

# Maya v1.1.3's device-only ctests: LABELS gpu (exl3_test needs a device; glm_mv_nt_test / _generic run the GEMVs)
foreach(_t IN ITEMS exl3_test glm_mv_nt_test glm_mv_nt_generic glm_dflash_test)
  if(TEST ${_t})
    set_property(TEST ${_t} APPEND PROPERTY LABELS gpu)
  endif()
endforeach()
