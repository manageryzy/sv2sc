# Unit Tests CMakeLists.txt
# Consolidated unit tests for sv2sc components



# Unit tests for core components
add_executable(test_systemc_generator test_systemc_generator.cpp)
target_link_libraries(test_systemc_generator sv2sc_core Catch2::Catch2WithMain)

add_executable(test_vcs_args_parser test_vcs_args_parser.cpp)
target_link_libraries(test_vcs_args_parser sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

# Phase-specific parser tests
add_executable(test_vcs_args_parser_phase1 test_vcs_args_parser_phase1.cpp)
target_link_libraries(test_vcs_args_parser_phase1 sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

add_executable(test_vcs_args_parser_phase2 test_vcs_args_parser_phase2.cpp)
target_link_libraries(test_vcs_args_parser_phase2 sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

add_executable(test_vcs_args_parser_phase3 test_vcs_args_parser_phase3.cpp)
target_link_libraries(test_vcs_args_parser_phase3 sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

add_executable(test_vcs_args_parser_phase4 test_vcs_args_parser_phase4.cpp)
target_link_libraries(test_vcs_args_parser_phase4 sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

# Automation framework tests
add_executable(test_automation_framework test_automation_framework.cpp)
target_link_libraries(test_automation_framework sv2sc_core Catch2::Catch2WithMain)
target_include_directories(test_automation_framework PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Build verification tests
add_executable(test_mlir_build_verification test_mlir_build_verification.cpp test_data_manager.cpp)
target_link_libraries(test_mlir_build_verification sv2sc_core Catch2::Catch2WithMain)
target_include_directories(test_mlir_build_verification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Utility tests
add_executable(test_data_manager test_data_manager.cpp)
target_link_libraries(test_data_manager sv2sc_core Catch2::Catch2WithMain)
target_include_directories(test_data_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CIRCT fallback emitter tests (conditional on MLIR support)
if(SV2SC_ENABLE_MLIR)
    add_executable(test_circt_compatible_emitter test_circt_compatible_emitter.cpp)
    target_link_libraries(test_circt_compatible_emitter sv2sc_mlir Catch2::Catch2WithMain)
    target_include_directories(test_circt_compatible_emitter PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/mlir/systemc
    )
endif()

# Register all unit tests with proper working directory
add_test(NAME test_systemc_generator COMMAND test_systemc_generator)
set_tests_properties(test_systemc_generator PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vcs_args_parser COMMAND test_vcs_args_parser)
set_tests_properties(test_vcs_args_parser PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vcs_args_parser_phase1 COMMAND test_vcs_args_parser_phase1)
set_tests_properties(test_vcs_args_parser_phase1 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vcs_args_parser_phase2 COMMAND test_vcs_args_parser_phase2)
set_tests_properties(test_vcs_args_parser_phase2 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vcs_args_parser_phase3 COMMAND test_vcs_args_parser_phase3)
set_tests_properties(test_vcs_args_parser_phase3 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_vcs_args_parser_phase4 COMMAND test_vcs_args_parser_phase4)
set_tests_properties(test_vcs_args_parser_phase4 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_automation_framework COMMAND test_automation_framework)
set_tests_properties(test_automation_framework PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_mlir_build_verification COMMAND test_mlir_build_verification)
set_tests_properties(test_mlir_build_verification PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_data_manager COMMAND test_data_manager)
set_tests_properties(test_data_manager PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Register CIRCT fallback tests (conditional)
if(SV2SC_ENABLE_MLIR)
    add_test(NAME test_circt_compatible_emitter COMMAND test_circt_compatible_emitter)
    set_tests_properties(test_circt_compatible_emitter PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()