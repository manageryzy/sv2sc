add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/memory.h ${CMAKE_CURRENT_BINARY_DIR}/memory.cpp
    COMMAND sv2sc -top memory -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/memory.sv
    DEPENDS sv2sc ${CMAKE_CURRENT_SOURCE_DIR}/memory.sv
    COMMENT "Translating memory.sv to SystemC"
)

add_custom_target(memory_example 
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/memory.h ${CMAKE_CURRENT_BINARY_DIR}/memory.cpp
)