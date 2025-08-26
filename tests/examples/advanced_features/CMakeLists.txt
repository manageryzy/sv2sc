# Advanced SystemVerilog Features Test Suite

# Test for performance profiling
add_complete_sv2sc_test_suite(
    TEST_NAME performance_test_sv2sc
    TOP_MODULE simple_counter
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/performance_test.sv
    EXPECTED_PORTS 4
)

# Test for error reporting
add_complete_sv2sc_test_suite(
    TEST_NAME error_reporting_sv2sc
    TOP_MODULE error_module
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/error_test.sv
    EXPECTED_PORTS 3
)

# Test for advanced SystemVerilog constructs
add_complete_sv2sc_test_suite(
    TEST_NAME advanced_constructs_sv2sc
    TOP_MODULE advanced_module
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/advanced_constructs.sv
    EXPECTED_PORTS 6
)

# Test for template engine features
add_complete_sv2sc_test_suite(
    TEST_NAME template_features_sv2sc
    TOP_MODULE template_module
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/template_test.sv
    EXPECTED_PORTS 4
)
