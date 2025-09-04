# ValidateDUTTranslation.cmake
# Validates DUT translations by checking port counts, quality metrics, and generated files

# Read input variables
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR not defined")
endif()

if(NOT DEFINED TOP_MODULE)
    message(FATAL_ERROR "TOP_MODULE not defined")
endif()

# Optional parameters
if(NOT DEFINED EXPECTED_PORTS)
    set(EXPECTED_PORTS "0")
endif()

if(NOT DEFINED EXPECTED_QUALITY)
    set(EXPECTED_QUALITY "GOOD")
endif()

# Check if generated files exist
set(HEADER_FILE "${TEST_OUTPUT_DIR}/${TOP_MODULE}.h")
set(CPP_FILE "${TEST_OUTPUT_DIR}/${TOP_MODULE}.cpp")

if(NOT EXISTS "${HEADER_FILE}")
    message(FATAL_ERROR "Generated header file not found: ${HEADER_FILE}")
endif()

if(NOT EXISTS "${CPP_FILE}")
    message(FATAL_ERROR "Generated cpp file not found: ${CPP_FILE}")
endif()

message(STATUS "Validating DUT translation for ${TOP_MODULE}")
message(STATUS "Header file: ${HEADER_FILE}")
message(STATUS "CPP file: ${CPP_FILE}")

# Read header file content
file(READ "${HEADER_FILE}" HEADER_CONTENT)

# Count ports
string(REGEX MATCHALL "sc_in<|sc_out<|sc_inout<" PORT_MATCHES "${HEADER_CONTENT}")
list(LENGTH PORT_MATCHES ACTUAL_PORTS)

# Count unknown expressions
string(REGEX MATCHALL "unknown_expr" UNKNOWN_EXPR_MATCHES "${HEADER_CONTENT}")
list(LENGTH UNKNOWN_EXPR_MATCHES UNKNOWN_EXPR_COUNT)

# Count skipped assignments
string(REGEX MATCHALL "skipped_assignment" SKIPPED_ASSIGN_MATCHES "${HEADER_CONTENT}")
list(LENGTH SKIPPED_ASSIGN_MATCHES SKIPPED_ASSIGN_COUNT)

# Determine quality level
set(QUALITY_LEVEL "NEEDS_WORK")
if(UNKNOWN_EXPR_COUNT EQUAL 0 AND SKIPPED_ASSIGN_COUNT EQUAL 0)
    set(QUALITY_LEVEL "EXCELLENT")
elseif(UNKNOWN_EXPR_COUNT LESS 5 AND SKIPPED_ASSIGN_COUNT LESS 5)
    set(QUALITY_LEVEL "GOOD")
elseif(UNKNOWN_EXPR_COUNT LESS 10 AND SKIPPED_ASSIGN_COUNT LESS 10)
    set(QUALITY_LEVEL "FAIR")
endif()

# Validate port count if expected
set(PORT_VALIDATION "PASS")
if(NOT EXPECTED_PORTS EQUAL "0")
    if(NOT ACTUAL_PORTS EQUAL EXPECTED_PORTS)
        set(PORT_VALIDATION "FAIL")
        message(WARNING "Port count mismatch: expected ${EXPECTED_PORTS}, got ${ACTUAL_PORTS}")
    endif()
endif()

# Validate quality if expected
set(QUALITY_VALIDATION "PASS")
if(NOT EXPECTED_QUALITY STREQUAL "ANY")
    if(QUALITY_LEVEL STREQUAL "NEEDS_WORK" AND NOT EXPECTED_QUALITY STREQUAL "NEEDS_WORK")
        set(QUALITY_VALIDATION "FAIL")
        message(WARNING "Quality level below expected: got ${QUALITY_LEVEL}, expected ${EXPECTED_QUALITY}")
    endif()
endif()

# Print validation results
message(STATUS "=== DUT Translation Validation Results ===")
message(STATUS "Module: ${TOP_MODULE}")
message(STATUS "Port Count: ${ACTUAL_PORTS} (expected: ${EXPECTED_PORTS}) - ${PORT_VALIDATION}")
message(STATUS "Unknown Expressions: ${UNKNOWN_EXPR_COUNT}")
message(STATUS "Skipped Assignments: ${SKIPPED_ASSIGN_COUNT}")
message(STATUS "Quality Level: ${QUALITY_LEVEL} (expected: ${EXPECTED_QUALITY}) - ${QUALITY_VALIDATION}")
message(STATUS "Overall Validation: ${PORT_VALIDATION}")

# Write results to file for test framework
set(RESULTS_FILE "${TEST_OUTPUT_DIR}/validation_results.txt")
file(WRITE "${RESULTS_FILE}"
"Module: ${TOP_MODULE}
Port Count: ${ACTUAL_PORTS}
Expected Ports: ${EXPECTED_PORTS}
Port Validation: ${PORT_VALIDATION}
Unknown Expressions: ${UNKNOWN_EXPR_COUNT}
Skipped Assignments: ${SKIPPED_ASSIGN_COUNT}
Quality Level: ${QUALITY_LEVEL}
Expected Quality: ${EXPECTED_QUALITY}
Quality Validation: ${QUALITY_VALIDATION}
Overall Result: ${PORT_VALIDATION}
")

# Exit with error if validation failed
if(PORT_VALIDATION STREQUAL "FAIL" OR QUALITY_VALIDATION STREQUAL "FAIL")
    message(FATAL_ERROR "DUT translation validation failed for ${TOP_MODULE}")
endif()

message(STATUS "DUT translation validation PASSED for ${TOP_MODULE}")
