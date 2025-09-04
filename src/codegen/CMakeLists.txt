set(CODEGEN_SOURCES
    systemc_generator.cpp
    template_engine.cpp
)

add_library(sv2sc_codegen STATIC ${CODEGEN_SOURCES})

target_link_libraries(sv2sc_codegen PUBLIC
    fmt::fmt
    sv2sc_utils
    spdlog::spdlog
)

target_include_directories(sv2sc_codegen PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Ensure C++20 compatibility and Clang support
target_compile_features(sv2sc_codegen PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sv2sc_codegen PRIVATE
        -Wno-unused-parameter
        -Wno-deprecated-declarations
        -Wno-unused-variable
    )
endif()