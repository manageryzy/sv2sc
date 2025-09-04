# CIRCT Fallback Generator Test Suite
# Uses the standard sv2sc testing framework from cmake/SystemCTestUtils.cmake

# Include the SystemC testing utilities
include(${CMAKE_SOURCE_DIR}/cmake/SystemCTestUtils.cmake)

# Test 1: Simple module translation with CIRCT fallback
add_sv2sc_test_suite(
    TEST_NAME circt_fallback_simple
    TOP_MODULE simple_module
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/simple_module.sv
    EXPECTED_PORTS 5
    EXPECTED_QUALITY GOOD
    TRANSLATOR_ARGS "--use-mlir"
    GENERATE_TESTBENCH
)

# Test 2: Test standard mode vs MLIR mode comparison
add_sv2sc_test_suite(
    TEST_NAME circt_fallback_standard_mode
    TOP_MODULE simple_module
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/simple_module.sv
    EXPECTED_PORTS 5
    EXPECTED_QUALITY GOOD
    # No --use-mlir flag for standard mode
)

# Test 3: Complex module with CIRCT fallback
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/complex_module.sv)
    add_sv2sc_test_suite(
        TEST_NAME circt_fallback_complex
        TOP_MODULE complex_module
        SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/complex_module.sv
        EXPECTED_QUALITY FAIR
        TRANSLATOR_ARGS "--use-mlir"
        GENERATE_TESTBENCH
    )
endif()

# Test 4: Memory array module with CIRCT fallback
add_sv2sc_test_suite(
    TEST_NAME circt_fallback_memory
    TOP_MODULE memory
    SV_SOURCE ${CMAKE_SOURCE_DIR}/tests/examples/memory_array/memory.sv
    EXPECTED_PORTS 8
    EXPECTED_QUALITY GOOD
    TRANSLATOR_ARGS "--use-mlir"
    GENERATE_TESTBENCH
)

# Test 5: Generate example with CIRCT fallback
add_sv2sc_test_suite(
    TEST_NAME circt_fallback_generate
    TOP_MODULE generate_adder
    SV_SOURCE ${CMAKE_SOURCE_DIR}/tests/examples/generate_example/generate_adder.sv
    EXPECTED_PORTS 3
    EXPECTED_QUALITY GOOD
    TRANSLATOR_ARGS "--use-mlir"
    GENERATE_TESTBENCH
)

message(STATUS "Configured CIRCT fallback test suite with 5 test cases")

