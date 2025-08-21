set(CODEGEN_SOURCES
    systemc_generator.cpp
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