set(TARGET llama-inkling-dump)
add_executable(${TARGET} inkling-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
