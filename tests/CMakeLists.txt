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