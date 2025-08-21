# Unit Tests CMakeLists.txt

# Create unit tests
set(UNIT_TEST_SOURCES
    test_vcs_args_parser.cpp
    test_systemc_generator.cpp
)

add_executable(unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(unit_tests PRIVATE
    sv2sc_translator
    sv2sc_codegen
    sv2sc_utils
    fmt::fmt
    Catch2::Catch2WithMain
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Register tests with CTest
catch_discover_tests(unit_tests TEST_PREFIX "unit.")