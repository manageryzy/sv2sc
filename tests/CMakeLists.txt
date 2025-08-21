# Tests CMakeLists.txt

# Add subdirectories for different test types
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit)
    add_subdirectory(unit)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/integration AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/integration)
    add_subdirectory(integration)
endif()