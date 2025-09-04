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

# Ensure C++20 compatibility and Clang support
target_compile_features(sv2sc_translator PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sv2sc_translator PRIVATE
        -Wno-unused-parameter
        -Wno-deprecated-declarations
    )
endif()