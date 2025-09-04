# Main Tests CMakeLists.txt
# Consolidated test structure for sv2sc project

# Include all test categories
add_subdirectory(basic_tests)
add_subdirectory(unit_tests)

# Optional heavy C++ test suites (need Catch2, SystemC runtime, Verilator etc.)
option(SV2SC_ENABLE_HEAVY_TESTS "Build/run heavy C++ test executables" OFF)
if(SV2SC_ENABLE_HEAVY_TESTS)
    add_subdirectory(integration_tests)
    add_subdirectory(functional_tests)
    add_subdirectory(performance_tests)
    add_subdirectory(security_tests)
    add_subdirectory(verification_tests)
else()
    message(STATUS "SV2SC_ENABLE_HEAVY_TESTS=OFF: Skipping heavy test suites")
endif()

add_subdirectory(examples)