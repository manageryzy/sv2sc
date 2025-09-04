# Integration Tests CMakeLists.txt
# End-to-end integration tests for sv2sc pipeline

# MLIR integration tests
add_executable(test_mlir_integration test_mlir_integration.cpp)
target_link_libraries(test_mlir_integration sv2sc_core sv2sc_mlir Catch2::Catch2WithMain)

# Translation flow tests
add_executable(test_translation_flow test_translation_flow.cpp)
target_link_libraries(test_translation_flow sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

# Separate compilation tests
add_executable(test_separate_compilation test_separate_compilation.cpp)
target_link_libraries(test_separate_compilation sv2sc_core sv2sc_codegen Catch2::Catch2WithMain)

# SystemC emitter integration tests (conditional on MLIR support)
if(SV2SC_ENABLE_MLIR)
    add_executable(test_systemc_emitter_integration test_systemc_emitter_integration.cpp)
    target_link_libraries(test_systemc_emitter_integration sv2sc_mlir Catch2::Catch2WithMain)
    target_include_directories(test_systemc_emitter_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# NBA splitting tests (guard if source exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_nba_splitting.cpp)
    add_executable(test_nba_splitting test_nba_splitting.cpp)
    target_link_libraries(test_nba_splitting sv2sc_core Catch2::Catch2WithMain)
    add_test(NAME test_nba_splitting COMMAND test_nba_splitting)
else()
    message(STATUS "Skipping test_nba_splitting (source file not found)")
endif()

# Sensitivity extraction tests (guard if source exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_sensitivity_extraction.cpp)
    add_executable(test_sensitivity_extraction test_sensitivity_extraction.cpp)
    target_link_libraries(test_sensitivity_extraction sv2sc_core Catch2::Catch2WithMain)
    add_test(NAME test_sensitivity_extraction COMMAND test_sensitivity_extraction)
else()
    message(STATUS "Skipping test_sensitivity_extraction (source file not found)")
endif()

# Register all integration tests
add_test(NAME test_mlir_integration COMMAND test_mlir_integration)
add_test(NAME test_translation_flow COMMAND test_translation_flow)
add_test(NAME test_separate_compilation COMMAND test_separate_compilation)

# Register SystemC emitter integration tests (conditional)
if(SV2SC_ENABLE_MLIR)
    add_test(NAME test_systemc_emitter_integration COMMAND test_systemc_emitter_integration)
endif()

# Shell script tests
add_test(NAME e2e_simple_test COMMAND ${CMAKE_SOURCE_DIR}/tests/integration_tests/e2e_simple_test.sh)
add_test(NAME final_syntax_test COMMAND ${CMAKE_SOURCE_DIR}/tests/integration_tests/final_syntax_test.sh)