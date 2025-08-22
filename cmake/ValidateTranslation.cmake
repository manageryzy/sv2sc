# Validate SystemC translation quality
# Variables expected:
# - TEST_OUTPUT_DIR: Directory containing translated files
# - TOP_MODULE: Top module name
# - EXPECTED_PORTS: Expected number of ports (optional)

# Check if header file exists
set(HEADER_FILE "${TEST_OUTPUT_DIR}/${TOP_MODULE}.h")
set(CPP_FILE "${TEST_OUTPUT_DIR}/${TOP_MODULE}.cpp")

if(NOT EXISTS "${HEADER_FILE}")
    message(FATAL_ERROR "Header file ${HEADER_FILE} does not exist")
endif()

if(NOT EXISTS "${CPP_FILE}")
    message(FATAL_ERROR "Implementation file ${CPP_FILE} does not exist")
endif()

# Read and analyze header file
file(READ "${HEADER_FILE}" HEADER_CONTENT)

# Count ports if expected ports specified
if(EXPECTED_PORTS)
    string(REGEX MATCHALL "sc_in<|sc_out<|sc_inout<" PORT_MATCHES "${HEADER_CONTENT}")
    list(LENGTH PORT_MATCHES ACTUAL_PORTS)
    
    if(NOT ACTUAL_PORTS EQUAL EXPECTED_PORTS)
        message(WARNING "Port count mismatch: expected ${EXPECTED_PORTS}, found ${ACTUAL_PORTS}")
    else()
        message(STATUS "Port count validation passed: ${ACTUAL_PORTS} ports")
    endif()
endif()

# Check for known issues
string(REGEX MATCHALL "unknown_expr" UNKNOWN_EXPRS "${HEADER_CONTENT}")
list(LENGTH UNKNOWN_EXPRS NUM_UNKNOWN)

string(REGEX MATCHALL "Skipping assignment" SKIPPED_ASSIGNS "${HEADER_CONTENT}")
list(LENGTH SKIPPED_ASSIGNS NUM_SKIPPED)

# Calculate translation quality score
math(EXPR TOTAL_ISSUES "${NUM_UNKNOWN} + ${NUM_SKIPPED}")

if(TOTAL_ISSUES EQUAL 0)
    set(QUALITY_SCORE "100%")
    set(QUALITY_STATUS "EXCELLENT")
elseif(TOTAL_ISSUES LESS_EQUAL 2)
    set(QUALITY_SCORE "80%")
    set(QUALITY_STATUS "GOOD")
elseif(TOTAL_ISSUES LESS_EQUAL 5)
    set(QUALITY_SCORE "60%")
    set(QUALITY_STATUS "FAIR")
else()
    set(QUALITY_SCORE "40%")
    set(QUALITY_STATUS "NEEDS_WORK")
endif()

# Print validation results
message(STATUS "=== Translation Validation Results ===")
message(STATUS "Module: ${TOP_MODULE}")
message(STATUS "Unknown expressions: ${NUM_UNKNOWN}")
message(STATUS "Skipped assignments: ${NUM_SKIPPED}")
message(STATUS "Quality Score: ${QUALITY_SCORE} (${QUALITY_STATUS})")
message(STATUS "Header: ${HEADER_FILE}")
message(STATUS "Implementation: ${CPP_FILE}")

# Return with appropriate status
if(QUALITY_STATUS STREQUAL "NEEDS_WORK")
    message(WARNING "Translation quality is poor - manual review recommended")
endif()