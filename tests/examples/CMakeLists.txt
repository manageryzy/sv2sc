# Test for basic counter DUT module
add_sv2sc_test_suite(
    TEST_NAME counter_dut
    TOP_MODULE counter
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/basic_counter/counter.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/basic_counter/counter_tb.sv
    EXPECTED_PORTS 7
    EXPECTED_QUALITY EXCELLENT
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# Test for memory DUT module
add_sv2sc_test_suite(
    TEST_NAME memory_dut
    TOP_MODULE memory
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/memory_array/memory.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/memory_array/memory_tb.sv
    EXPECTED_PORTS 15
    EXPECTED_QUALITY GOOD
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# Test for generate adder DUT module
add_sv2sc_test_suite(
    TEST_NAME generate_adder_dut
    TOP_MODULE generate_adder
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/generate_example/generate_adder.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/generate_example/generate_adder_tb.sv
    EXPECTED_PORTS 5
    EXPECTED_QUALITY GOOD
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# CIRCT fallback test suite (conditional on MLIR support)
if(SV2SC_ENABLE_MLIR)
    add_subdirectory(circt_fallback_test)
endif()

# Keep original subdirectories for reference
add_subdirectory(basic_counter)
add_subdirectory(memory_array)
add_subdirectory(generate_example)
add_subdirectory(picorv32)
add_subdirectory(advanced_features)
