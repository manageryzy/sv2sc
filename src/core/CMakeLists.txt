set(CORE_SOURCES
    ast_visitor.cpp
)

add_library(sv2sc_core STATIC ${CORE_SOURCES})

target_link_libraries(sv2sc_core PUBLIC
    slang::slang
    sv2sc_codegen
)

target_include_directories(sv2sc_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)