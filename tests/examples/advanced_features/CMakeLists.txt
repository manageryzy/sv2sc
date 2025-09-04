# Advanced SystemVerilog Features Test Suite

# Test for ALU DUT module
add_sv2sc_test_suite(
    TEST_NAME alu_dut
    TOP_MODULE alu_dut
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/alu_dut.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/alu_tb.sv
    EXPECTED_PORTS 13
    EXPECTED_QUALITY EXCELLENT
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# Test for FSM DUT module
add_sv2sc_test_suite(
    TEST_NAME fsm_dut
    TOP_MODULE fsm_dut
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/fsm_dut.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/fsm_tb.sv
    EXPECTED_PORTS 11
    EXPECTED_QUALITY EXCELLENT
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# Test for Interface DUT module
add_sv2sc_test_suite(
    TEST_NAME interface_dut
    TOP_MODULE interface_dut
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/interface_dut.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/interface_tb.sv
    EXPECTED_PORTS 8
    EXPECTED_QUALITY GOOD
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)

# Test for Pipeline DUT module
add_sv2sc_test_suite(
    TEST_NAME pipeline_dut
    TOP_MODULE pipeline_dut
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_dut.sv
    SV_TESTBENCH ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_tb.sv
    EXPECTED_PORTS 10
    EXPECTED_QUALITY GOOD
    GENERATE_TESTBENCH
    VERILATOR_COMPARISON
    TRANSLATOR_ARGS "--use-mlir"
)
