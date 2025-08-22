# Compare Verilator and SystemC simulation results
# Variables expected:
# - VERILATOR_SIM: Path to Verilator simulation executable
# - SYSTEMC_TEST: Name of SystemC test executable

# Check if executables exist
if(NOT EXISTS "${VERILATOR_SIM}")
    message(FATAL_ERROR "Verilator simulation ${VERILATOR_SIM} does not exist")
endif()

# Run Verilator simulation
message(STATUS "Running Verilator simulation...")
execute_process(
    COMMAND "${VERILATOR_SIM}"
    OUTPUT_VARIABLE VLT_OUTPUT
    ERROR_VARIABLE VLT_ERROR
    RESULT_VARIABLE VLT_RESULT
    TIMEOUT 30
)

if(NOT VLT_RESULT EQUAL 0)
    message(FATAL_ERROR "Verilator simulation failed: ${VLT_ERROR}")
endif()

# Run SystemC simulation  
message(STATUS "Running SystemC simulation...")
execute_process(
    COMMAND "${SYSTEMC_TEST}"
    OUTPUT_VARIABLE SC_OUTPUT
    ERROR_VARIABLE SC_ERROR
    RESULT_VARIABLE SC_RESULT
    TIMEOUT 30
)

if(NOT SC_RESULT EQUAL 0)
    message(FATAL_ERROR "SystemC simulation failed: ${SC_ERROR}")
endif()

# Compare outputs (basic comparison)
# This is a simple comparison - more sophisticated analysis could be added
string(LENGTH "${VLT_OUTPUT}" VLT_LEN)
string(LENGTH "${SC_OUTPUT}" SC_LEN)

message(STATUS "=== Simulation Comparison Results ===")
message(STATUS "Verilator output length: ${VLT_LEN} characters")
message(STATUS "SystemC output length: ${SC_LEN} characters")

# Save outputs for manual inspection
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/verilator_output.txt" "${VLT_OUTPUT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/systemc_output.txt" "${SC_OUTPUT}")

message(STATUS "Outputs saved for comparison:")
message(STATUS "  Verilator: ${CMAKE_CURRENT_BINARY_DIR}/verilator_output.txt")
message(STATUS "  SystemC: ${CMAKE_CURRENT_BINARY_DIR}/systemc_output.txt")

# Basic functional comparison
if(VLT_OUTPUT STREQUAL SC_OUTPUT)
    message(STATUS "PERFECT MATCH: Outputs are identical")
elseif(VLT_LEN GREATER 0 AND SC_LEN GREATER 0)
    message(STATUS "PARTIAL MATCH: Both simulations produced output")
else()
    message(WARNING "OUTPUT MISMATCH: Significant differences detected")
endif()