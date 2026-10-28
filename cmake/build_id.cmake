# cmake/build_id.cmake - the commit and the release maya-ds4 is built from, for `maya-ds4 --version` and the
# "ds4: build <commit>" line --verbose prints.  Run at every build (the maya_build_id target), so a commit made since
# the last build is seen without configuring again; the header is rewritten only when it changes, and only then is
# ds4_main.cpp compiled again.
#   cmake -DSRC=<source dir> -DOUT=<header> [-DID=<id>] -P cmake/build_id.cmake
# The id: ID when given (-DMAYA_BUILD_ID=... at configure), else a BUILD_ID file's first line (a source archive has no
# .git: whoever packs it writes the commit there), else git's commit (12 hex digits, "-dirty" with changes to tracked
# files not committed), else "unknown".  The release: the VERSION file's (what the server's About page shows).
set(id "${ID}")
if(id STREQUAL "" AND EXISTS "${SRC}/BUILD_ID")
  file(STRINGS "${SRC}/BUILD_ID" id LIMIT_COUNT 1)
  string(STRIP "${id}" id)
endif()
if(id STREQUAL "")
  find_program(_git git)
  if(_git)
    execute_process(COMMAND "${_git}" -C "${SRC}" rev-parse --short=12 HEAD OUTPUT_VARIABLE id RESULT_VARIABLE rc
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT rc EQUAL 0)
      set(id "")
    else()
      execute_process(COMMAND "${_git}" -C "${SRC}" diff --quiet HEAD -- RESULT_VARIABLE dirty ERROR_QUIET)
      if(dirty EQUAL 1)
        string(APPEND id "-dirty")
      endif()
    endif()
  endif()
endif()
if(id STREQUAL "")
  set(id "unknown")
endif()
set(release "")
if(EXISTS "${SRC}/VERSION")
  file(STRINGS "${SRC}/VERSION" release LIMIT_COUNT 1)
  string(STRIP "${release}" release)
endif()
if(release STREQUAL "")
  set(release "?")
endif()
string(REPLACE "\"" "" id "${id}")
string(REPLACE "\"" "" release "${release}")
file(WRITE "${OUT}.tmp" "// written by cmake/build_id.cmake at every build\n#define MAYA_BUILD_ID \"${id}\"\n"
                        "#define MAYA_RELEASE \"${release}\"\n")
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
