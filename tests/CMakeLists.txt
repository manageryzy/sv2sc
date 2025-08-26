# Tests CMakeLists.txt

# Add subdirectories for different test types
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit)
    add_subdirectory(unit)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/integration AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/integration)
    add_subdirectory(integration)
endif()

# Only add examples if BUILD_EXAMPLES is not already handling it
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples AND NOT BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add PicoRV32 verification suite
option(BUILD_PICORV32_VERIFICATION "Build PicoRV32 verification suite" OFF)
if(BUILD_PICORV32_VERIFICATION)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/verification/picorv32)
        add_subdirectory(verification/picorv32)
    else()
        message(WARNING "PicoRV32 verification directory not found")
    endif()
endif()