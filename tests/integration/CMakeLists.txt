# Integration Tests CMakeLists.txt

# Create integration tests
set(INTEGRATION_TEST_SOURCES
    test_translation_flow.cpp
)

add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})

target_link_libraries(integration_tests PRIVATE
    sv2sc_core
    sv2sc_translator
    sv2sc_codegen
    sv2sc_utils
    slang::slang
    fmt::fmt
    Catch2::Catch2WithMain
)

target_include_directories(integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Register tests with CTest
catch_discover_tests(integration_tests TEST_PREFIX "integration.")