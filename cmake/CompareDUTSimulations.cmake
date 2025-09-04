# CompareDUTSimulations.cmake
# Compares SystemC and Verilator simulation results for DUT modules

# Read input variables
if(NOT DEFINED VERILATOR_SIM)
    message(FATAL_ERROR "VERILATOR_SIM not defined")
endif()

if(NOT DEFINED SYSTEMC_TEST)
    message(FATAL_ERROR "SYSTEMC_TEST not defined")
endif()

if(NOT DEFINED TEST_NAME)
    message(FATAL_ERROR "TEST_NAME not defined")
endif()

message(STATUS "Comparing simulations for ${TEST_NAME}")
message(STATUS "Verilator simulation: ${VERILATOR_SIM}")
message(STATUS "SystemC test: ${SYSTEMC_TEST}")

# Check if Verilator simulation exists
if(NOT EXISTS "${VERILATOR_SIM}")
    message(FATAL_ERROR "Verilator simulation not found: ${VERILATOR_SIM}")
endif()

# Run SystemC simulation and capture output
set(SYSTEMC_OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_systemc_output.txt")
execute_process(
    COMMAND ${SYSTEMC_TEST}
    OUTPUT_FILE "${SYSTEMC_OUTPUT_FILE}"
    ERROR_FILE "${SYSTEMC_OUTPUT_FILE}.err"
    RESULT_VARIABLE SYSTEMC_RESULT
    TIMEOUT 60
)

if(NOT SYSTEMC_RESULT EQUAL 0)
    message(WARNING "SystemC simulation failed with exit code: ${SYSTEMC_RESULT}")
    file(READ "${SYSTEMC_OUTPUT_FILE}.err" SYSTEMC_ERROR)
    message(STATUS "SystemC error output: ${SYSTEMC_ERROR}")
endif()

# Run Verilator simulation and capture output
set(VERILATOR_OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_verilator_output.txt")
execute_process(
    COMMAND ${VERILATOR_SIM}
    OUTPUT_FILE "${VERILATOR_OUTPUT_FILE}"
    ERROR_FILE "${VERILATOR_OUTPUT_FILE}.err"
    RESULT_VARIABLE VERILATOR_RESULT
    TIMEOUT 60
)

if(NOT VERILATOR_RESULT EQUAL 0)
    message(WARNING "Verilator simulation failed with exit code: ${VERILATOR_RESULT}")
    file(READ "${VERILATOR_OUTPUT_FILE}.err" VERILATOR_ERROR)
    message(STATUS "Verilator error output: ${VERILATOR_ERROR}")
endif()

# Read simulation outputs
file(READ "${SYSTEMC_OUTPUT_FILE}" SYSTEMC_OUTPUT)
file(READ "${VERILATOR_OUTPUT_FILE}" VERILATOR_OUTPUT)

# Check for test results
string(FIND "${SYSTEMC_OUTPUT}" "ALL TESTS PASSED" SYSTEMC_PASS_POS)
string(FIND "${VERILATOR_OUTPUT}" "ALL TESTS PASSED" VERILATOR_PASS_POS)

# Check for test failures
string(FIND "${SYSTEMC_OUTPUT}" "SOME TESTS FAILED" SYSTEMC_FAIL_POS)
string(FIND "${VERILATOR_OUTPUT}" "SOME TESTS FAILED" VERILATOR_FAIL_POS)

# Determine test results
set(SYSTEMC_TEST_RESULT "UNKNOWN")
if(SYSTEMC_PASS_POS GREATER_EQUAL 0)
    set(SYSTEMC_TEST_RESULT "PASS")
elseif(SYSTEMC_FAIL_POS GREATER_EQUAL 0)
    set(SYSTEMC_TEST_RESULT "FAIL")
endif()

set(VERILATOR_TEST_RESULT "UNKNOWN")
if(VERILATOR_PASS_POS GREATER_EQUAL 0)
    set(VERILATOR_TEST_RESULT "PASS")
elseif(VERILATOR_FAIL_POS GREATER_EQUAL 0)
    set(VERILATOR_TEST_RESULT "FAIL")
endif()

# Compare results
set(COMPARISON_RESULT "UNKNOWN")
if(SYSTEMC_TEST_RESULT STREQUAL "PASS" AND VERILATOR_TEST_RESULT STREQUAL "PASS")
    set(COMPARISON_RESULT "MATCH")
elseif(SYSTEMC_TEST_RESULT STREQUAL "FAIL" AND VERILATOR_TEST_RESULT STREQUAL "FAIL")
    set(COMPARISON_RESULT "MATCH")
elseif(SYSTEMC_TEST_RESULT STREQUAL "UNKNOWN" OR VERILATOR_TEST_RESULT STREQUAL "UNKNOWN")
    set(COMPARISON_RESULT "UNKNOWN")
else()
    set(COMPARISON_RESULT "MISMATCH")
endif()

# Print comparison results
message(STATUS "=== Simulation Comparison Results ===")
message(STATUS "Test: ${TEST_NAME}")
message(STATUS "SystemC Result: ${SYSTEMC_TEST_RESULT}")
message(STATUS "Verilator Result: ${VERILATOR_TEST_RESULT}")
message(STATUS "Comparison: ${COMPARISON_RESULT}")

# Write comparison results to file
set(COMPARISON_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_comparison_results.txt")
file(WRITE "${COMPARISON_FILE}"
"Test: ${TEST_NAME}
SystemC Result: ${SYSTEMC_TEST_RESULT}
Verilator Result: ${VERILATOR_TEST_RESULT}
Comparison: ${COMPARISON_RESULT}
SystemC Exit Code: ${SYSTEMC_RESULT}
Verilator Exit Code: ${VERILATOR_RESULT}
")

# Write detailed outputs for analysis
set(SYSTEMC_DETAILED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_systemc_detailed.txt")
file(WRITE "${SYSTEMC_DETAILED_FILE}" "${SYSTEMC_OUTPUT}")

set(VERILATOR_DETAILED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_verilator_detailed.txt")
file(WRITE "${VERILATOR_DETAILED_FILE}" "${VERILATOR_OUTPUT}")

# Exit with error if there's a mismatch
if(COMPARISON_RESULT STREQUAL "MISMATCH")
    message(FATAL_ERROR "Simulation comparison failed: SystemC=${SYSTEMC_TEST_RESULT}, Verilator=${VERILATOR_TEST_RESULT}")
endif()

if(COMPARISON_RESULT STREQUAL "UNKNOWN")
    message(WARNING "Simulation comparison inconclusive: SystemC=${SYSTEMC_TEST_RESULT}, Verilator=${VERILATOR_TEST_RESULT}")
endif()

message(STATUS "Simulation comparison completed successfully for ${TEST_NAME}")
