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

# Ensure C++20 compatibility and Clang support
target_compile_features(sv2sc_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sv2sc_core PRIVATE
        -Wno-unused-parameter
        -Wno-deprecated-declarations
    )
endif()