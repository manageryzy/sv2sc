add_subdirectory(core)
add_subdirectory(translator)
add_subdirectory(codegen)
add_subdirectory(utils)

set(SV2SC_SOURCES
    main.cpp
    sv2sc.cpp
)

add_executable(sv2sc ${SV2SC_SOURCES})

target_link_libraries(sv2sc PRIVATE
    sv2sc_core
    sv2sc_translator
    sv2sc_codegen
    sv2sc_utils
    slang::slang
    fmt::fmt
    spdlog::spdlog
)

target_include_directories(sv2sc PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)