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