# Verification Tests CMakeLists.txt
# Verification tests for sv2sc translation accuracy

# PicoRV32 verification tests
add_executable(picorv32_systemc_test picorv32/testbench/systemc_testbench.cpp)
target_link_libraries(picorv32_systemc_test sv2sc_core SystemC::systemc)

find_program(VERILATOR_EXECUTABLE verilator)
if(VERILATOR_EXECUTABLE)
    add_executable(picorv32_verilator_test picorv32/testbench/verilator_testbench.cpp)
    target_link_libraries(picorv32_verilator_test sv2sc_core)
    add_test(NAME picorv32_verilator_test COMMAND picorv32_verilator_test)
else()
    message(WARNING "Verilator not found; skipping picorv32_verilator_test build")
endif()

# Register verification tests
add_test(NAME picorv32_systemc_test COMMAND picorv32_systemc_test)

# Shell script tests
add_test(NAME picorv32_e2e_test COMMAND ${CMAKE_SOURCE_DIR}/tests/verification_tests/run_e2e_test.sh)
add_test(NAME picorv32_comparison COMMAND ${CMAKE_SOURCE_DIR}/tests/verification_tests/scripts/run_comparison.sh)
