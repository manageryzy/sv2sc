set(UTILS_SOURCES
    logger.cpp
    error_reporter.cpp
    performance_profiler.cpp
)

add_library(sv2sc_utils STATIC ${UTILS_SOURCES})

target_link_libraries(sv2sc_utils PUBLIC
    spdlog::spdlog
    fmt::fmt
)

target_include_directories(sv2sc_utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Ensure C++20 compatibility and Clang support
target_compile_features(sv2sc_utils PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sv2sc_utils PRIVATE
        -Wno-unused-parameter
        -Wno-deprecated-declarations
    )
endif()