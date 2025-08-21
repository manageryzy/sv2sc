add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/counter.h ${CMAKE_CURRENT_BINARY_DIR}/counter.cpp
    COMMAND sv2sc -top counter -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/counter.sv
    DEPENDS sv2sc ${CMAKE_CURRENT_SOURCE_DIR}/counter.sv
    COMMENT "Translating counter.sv to SystemC"
)

# Create a simple SystemC test program
set(SYSTEMC_TEST_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/test_counter_systemc.cpp)
file(WRITE ${SYSTEMC_TEST_SOURCE}
"#include \"counter.h\"
#include <systemc.h>
#include <iostream>

int sc_main(int argc, char* argv[]) {
    (void)argc; // Suppress unused parameter warning
    (void)argv;
    
    // Create counter instance  
    counter dut(\"counter_dut\");
    
    // Create signals
    sc_signal<sc_logic> clk_sig;
    sc_signal<sc_logic> reset_sig;
    sc_signal<sc_logic> enable_sig;
    sc_signal<sc_lv<8>> count_sig;
    
    // Connect signals to ports
    dut.clk(clk_sig);
    dut.reset(reset_sig); 
    dut.enable(enable_sig);
    dut.count(count_sig);
    
    std::cout << \"SystemC counter instantiation successful!\" << std::endl;
    std::cout << \"Generated SystemC code compiles and links correctly.\" << std::endl;
    
    return 0;
}
")

add_executable(counter_systemc_test
    ${CMAKE_CURRENT_BINARY_DIR}/counter.h
    ${CMAKE_CURRENT_BINARY_DIR}/counter.cpp
    ${SYSTEMC_TEST_SOURCE}
)

# Force C++14 for SystemC compatibility
set_target_properties(counter_systemc_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Ensure SystemC compilation uses C++14 flags
target_compile_features(counter_systemc_test PRIVATE cxx_std_14)

target_link_libraries(counter_systemc_test PRIVATE SystemC::systemc)
target_include_directories(counter_systemc_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(counter_example 
    DEPENDS counter_systemc_test
    COMMENT "Generated and tested SystemC counter example"
)