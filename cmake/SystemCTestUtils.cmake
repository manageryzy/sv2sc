# SystemC Testing Utilities
# Functions for testing SystemVerilog to SystemC translation workflow

# Function to create a complete test workflow for SV->SC translation and verification
function(add_sv2sc_test_suite)
    set(options GENERATE_TESTBENCH VERILATOR_COMPARISON)
    set(oneValueArgs 
        TEST_NAME           # Name of the test
        TOP_MODULE          # Top module name for translation
        SV_SOURCE          # SystemVerilog source file
        SV_TESTBENCH       # SystemVerilog testbench file (optional)
        EXPECTED_PORTS     # Expected number of ports (optional)
        EXPECTED_QUALITY   # Expected quality level (EXCELLENT/GOOD/FAIR/NEEDS_WORK)
        TRANSLATOR_ARGS    # Additional arguments to pass to sv2sc translator
    )
    set(multiValueArgs
        SV_INCLUDES        # Include directories for SV compilation
        SV_DEFINES         # Preprocessor defines
        DEPENDENCIES       # Additional dependencies
    )
    
    cmake_parse_arguments(SV2SC_TEST 
        "${options}" 
        "${oneValueArgs}" 
        "${multiValueArgs}" 
        ${ARGN}
    )
    
    if(NOT SV2SC_TEST_TEST_NAME)
        message(FATAL_ERROR "add_sv2sc_test_suite requires TEST_NAME")
    endif()
    
    if(NOT SV2SC_TEST_TOP_MODULE)
        message(FATAL_ERROR "add_sv2sc_test_suite requires TOP_MODULE")
    endif()
    
    if(NOT SV2SC_TEST_SV_SOURCE)
        message(FATAL_ERROR "add_sv2sc_test_suite requires SV_SOURCE")
    endif()
    
    # Create test-specific output directory
    set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tests/${SV2SC_TEST_TEST_NAME})
    
    # Build include arguments
    set(INCLUDE_ARGS "")
    foreach(inc_dir ${SV2SC_TEST_SV_INCLUDES})
        list(APPEND INCLUDE_ARGS "-I" "${inc_dir}")
    endforeach()
    
    # Build define arguments  
    set(DEFINE_ARGS "")
    foreach(define ${SV2SC_TEST_SV_DEFINES})
        list(APPEND DEFINE_ARGS "-D" "${define}")
    endforeach()
    
    # Parse additional translator arguments
    set(ADDITIONAL_ARGS "")
    if(SV2SC_TEST_TRANSLATOR_ARGS)
        string(REPLACE " " ";" ADDITIONAL_ARGS_LIST "${SV2SC_TEST_TRANSLATOR_ARGS}")
        set(ADDITIONAL_ARGS ${ADDITIONAL_ARGS_LIST})
    endif()
    
    # Step 1: SystemVerilog to SystemC translation
    add_custom_command(
        OUTPUT 
            ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h 
            ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "PATH=$ENV{PATH}:$<TARGET_FILE_DIR:sv2sc>" sv2sc 
            -top ${SV2SC_TEST_TOP_MODULE}
            -o ${TEST_OUTPUT_DIR}
            ${INCLUDE_ARGS}
            ${DEFINE_ARGS}
            ${ADDITIONAL_ARGS}
            ${SV2SC_TEST_SV_SOURCE}
        # Copy generated files from common output locations if needed
        # 1) Build-tree default output
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h || true
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.cpp || true
        # 2) Source-tree output (if translator writes there)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h || true
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.cpp || true
        # 3) Local output under current binary dir
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h || true
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.cpp || true
        DEPENDS sv2sc ${SV2SC_TEST_SV_SOURCE} ${SV2SC_TEST_DEPENDENCIES}
        COMMENT "Translating ${SV2SC_TEST_SV_SOURCE} to SystemC"
        VERBATIM
    )

    # Ensure generation runs during normal builds
    add_custom_target(${SV2SC_TEST_TEST_NAME}_gen ALL
        DEPENDS
            ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
            ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
    )
    
    # Step 2: Create SystemC testbench if requested
    if(SV2SC_TEST_GENERATE_TESTBENCH)
        set(TESTBENCH_FILE ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TEST_NAME}_tb.cpp)
        
        # Generate a comprehensive SystemC testbench
        add_custom_command(
            OUTPUT ${TESTBENCH_FILE}
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_NAME=${SV2SC_TEST_TEST_NAME}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DOUTPUT_FILE=${TESTBENCH_FILE}
                -DHEADER_FILE=${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
                -DSV_TESTBENCH=${SV2SC_TEST_SV_TESTBENCH}
                -P ${CMAKE_SOURCE_DIR}/cmake/GenerateComprehensiveTestbench.cmake
            DEPENDS ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.h
            COMMENT "Generating SystemC testbench for ${SV2SC_TEST_TEST_NAME}"
        )
        
        # Step 3: Build SystemC executable
        add_executable(${SV2SC_TEST_TEST_NAME}_systemc_test
            ${TESTBENCH_FILE}
        )
        
        # Create placeholder files for CMake configuration
        set(PLACEHOLDER_CPP ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp)
        if(NOT EXISTS ${PLACEHOLDER_CPP})
            file(WRITE ${PLACEHOLDER_CPP} "// Placeholder - will be replaced by translation\n")
        endif()
        
        # Add the generated SystemC source as a dependency
        target_sources(${SV2SC_TEST_TEST_NAME}_systemc_test PRIVATE 
            ${TEST_OUTPUT_DIR}/output/${SV2SC_TEST_TOP_MODULE}.cpp
        )
        
        # Ensure the generated files exist before compilation
        add_dependencies(${SV2SC_TEST_TEST_NAME}_systemc_test ${SV2SC_TEST_TEST_NAME}_gen)
        
        # Configure SystemC compilation with C++14 for compatibility
        set_target_properties(${SV2SC_TEST_TEST_NAME}_systemc_test PROPERTIES
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        
        target_link_libraries(${SV2SC_TEST_TEST_NAME}_systemc_test PRIVATE SystemC::systemc)
        target_include_directories(${SV2SC_TEST_TEST_NAME}_systemc_test PRIVATE 
            ${TEST_OUTPUT_DIR}
            ${TEST_OUTPUT_DIR}/output
        )
        
        # Step 4: Add translation validation test
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_translation_validation
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DEXPECTED_PORTS=${SV2SC_TEST_EXPECTED_PORTS}
                -DEXPECTED_QUALITY=${SV2SC_TEST_EXPECTED_QUALITY}
                -P ${CMAKE_SOURCE_DIR}/cmake/ValidateDUTTranslation.cmake
        )
        
        # Step 5: Add SystemC simulation test
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_systemc_simulation
            COMMAND ${SV2SC_TEST_TEST_NAME}_systemc_test
        )
        
        # Step 6: Add Verilator comparison if requested and testbench provided
        if(SV2SC_TEST_VERILATOR_COMPARISON AND SV2SC_TEST_SV_TESTBENCH)
            find_program(VERILATOR_EXECUTABLE verilator)
            
            if(VERILATOR_EXECUTABLE)
                set(VLT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/verilator/${SV2SC_TEST_TEST_NAME})
                
                # Build Verilator simulation
                add_custom_target(${SV2SC_TEST_TEST_NAME}_verilator
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${VLT_OUTPUT_DIR}
                    COMMAND ${VERILATOR_EXECUTABLE}
                        --cc --exe --build --timing -Wno-TIMESCALEMOD
                        -o ${SV2SC_TEST_TEST_NAME}_sim
                        --top-module ${SV2SC_TEST_TEST_NAME}_tb
                        ${SV2SC_TEST_SV_SOURCE} ${SV2SC_TEST_SV_TESTBENCH}
                    WORKING_DIRECTORY ${VLT_OUTPUT_DIR}
                    COMMENT "Building Verilator simulation for ${SV2SC_TEST_TEST_NAME}"
                )
                
                # Add Verilator simulation test
                add_test(
                    NAME ${SV2SC_TEST_TEST_NAME}_verilator_simulation
                    COMMAND ${VLT_OUTPUT_DIR}/${SV2SC_TEST_TEST_NAME}_sim
                )
                
                # Add comparison test
                add_test(
                    NAME ${SV2SC_TEST_TEST_NAME}_verilator_vs_systemc
                    COMMAND ${CMAKE_COMMAND}
                        -DVERILATOR_SIM=${VLT_OUTPUT_DIR}/${SV2SC_TEST_TEST_NAME}_sim
                        -DSYSTEMC_TEST=${SV2SC_TEST_TEST_NAME}_systemc_test
                        -DTEST_NAME=${SV2SC_TEST_TEST_NAME}
                        -P ${CMAKE_SOURCE_DIR}/cmake/CompareDUTSimulations.cmake
                )
            else()
                message(WARNING "Verilator not found - skipping Verilator comparison for ${SV2SC_TEST_TEST_NAME}")
            endif()
        endif()
        
    else()
        # Just create a translation validation test
        add_custom_target(${SV2SC_TEST_TEST_NAME}_translation
            DEPENDS ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h
        )
        
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_translation_validation
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DEXPECTED_PORTS=${SV2SC_TEST_EXPECTED_PORTS}
                -DEXPECTED_QUALITY=${SV2SC_TEST_EXPECTED_QUALITY}
                -P ${CMAKE_SOURCE_DIR}/cmake/ValidateDUTTranslation.cmake
        )
    endif()
    
    message(STATUS "Created test suite: ${SV2SC_TEST_TEST_NAME}")
    
endfunction()

# Legacy function for backward compatibility
function(add_comprehensive_dut_test_suite)
    # Convert to new function call
    add_sv2sc_test_suite(${ARGN} GENERATE_TESTBENCH VERILATOR_COMPARISON)
endfunction()

# Legacy function for backward compatibility  
function(add_sv2sc_test)
    # Convert to new function call
    add_sv2sc_test_suite(${ARGN} GENERATE_TESTBENCH)
endfunction()