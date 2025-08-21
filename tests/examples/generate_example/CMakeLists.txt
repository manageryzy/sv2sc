add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generate_adder.h ${CMAKE_CURRENT_BINARY_DIR}/generate_adder.cpp
    COMMAND sv2sc -top generate_adder -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/generate_adder.sv
    DEPENDS sv2sc ${CMAKE_CURRENT_SOURCE_DIR}/generate_adder.sv
    COMMENT "Translating generate_adder.sv to SystemC"
)

add_custom_target(generate_example 
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generate_adder.h ${CMAKE_CURRENT_BINARY_DIR}/generate_adder.cpp
)