# Use new SystemC test utilities for all examples

# Counter example
add_complete_sv2sc_test_suite(
    TEST_NAME counter_sv2sc
    TOP_MODULE counter
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/basic_counter/counter.sv
    EXPECTED_PORTS 4
)

# Memory example  
add_complete_sv2sc_test_suite(
    TEST_NAME memory_sv2sc
    TOP_MODULE memory
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/memory_array/memory.sv
    EXPECTED_PORTS 7
)

# Generate adder example
add_complete_sv2sc_test_suite(
    TEST_NAME generate_adder_sv2sc
    TOP_MODULE generate_adder  
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/generate_example/generate_adder.sv
    EXPECTED_PORTS 5
)

# Keep original subdirectories for reference
add_subdirectory(basic_counter)
add_subdirectory(memory_array)
add_subdirectory(generate_example)