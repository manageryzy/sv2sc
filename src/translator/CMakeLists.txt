set(TRANSLATOR_SOURCES
    vcs_args_parser.cpp
)

add_library(sv2sc_translator STATIC ${TRANSLATOR_SOURCES})

target_link_libraries(sv2sc_translator PUBLIC
    fmt::fmt
)

target_include_directories(sv2sc_translator PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)