# SystemC Testing Utilities
# Functions for testing SystemVerilog to SystemC translation workflow

# Function to create a complete test workflow for SV->SC translation and verification
function(add_sv2sc_test)
    set(options GENERATE_TESTBENCH)
    set(oneValueArgs 
        TEST_NAME           # Name of the test
        TOP_MODULE          # Top module name for translation
        SV_SOURCE          # SystemVerilog source file
        EXPECTED_PORTS     # Expected number of ports (optional)
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
        message(FATAL_ERROR "add_sv2sc_test requires TEST_NAME")
    endif()
    
    if(NOT SV2SC_TEST_TOP_MODULE)
        message(FATAL_ERROR "add_sv2sc_test requires TOP_MODULE")
    endif()
    
    if(NOT SV2SC_TEST_SV_SOURCE)
        message(FATAL_ERROR "add_sv2sc_test requires SV_SOURCE")
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
    
    # Step 1: SystemVerilog to SystemC translation
    add_custom_command(
        OUTPUT 
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h 
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.cpp
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}
        COMMAND sv2sc 
            -top ${SV2SC_TEST_TOP_MODULE}
            -o ${TEST_OUTPUT_DIR}
            ${INCLUDE_ARGS}
            ${DEFINE_ARGS}
            ${SV2SC_TEST_SV_SOURCE}
        DEPENDS sv2sc ${SV2SC_TEST_SV_SOURCE} ${SV2SC_TEST_DEPENDENCIES}
        COMMENT "Translating ${SV2SC_TEST_SV_SOURCE} to SystemC"
        VERBATIM
    )
    
    # Step 2: Create SystemC testbench if requested
    if(SV2SC_TEST_GENERATE_TESTBENCH)
        set(TESTBENCH_FILE ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TEST_NAME}_tb.cpp)
        
        # Generate a basic SystemC testbench template
        add_custom_command(
            OUTPUT ${TESTBENCH_FILE}
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_NAME=${SV2SC_TEST_TEST_NAME}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DOUTPUT_FILE=${TESTBENCH_FILE}
                -DHEADER_FILE=${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h
                -P ${CMAKE_SOURCE_DIR}/cmake/GenerateSystemCTestbench.cmake
            DEPENDS ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h
            COMMENT "Generating SystemC testbench for ${SV2SC_TEST_TEST_NAME}"
        )
        
        # Step 3: Build SystemC executable
        add_executable(${SV2SC_TEST_TEST_NAME}_systemc_test
            ${TESTBENCH_FILE}
            ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.cpp
        )
        
        # Configure SystemC compilation with C++14 for compatibility
        set_target_properties(${SV2SC_TEST_TEST_NAME}_systemc_test PROPERTIES
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        
        target_link_libraries(${SV2SC_TEST_TEST_NAME}_systemc_test PRIVATE SystemC::systemc)
        target_include_directories(${SV2SC_TEST_TEST_NAME}_systemc_test PRIVATE 
            ${TEST_OUTPUT_DIR}
        )
        
        # Step 4: Add CTest integration
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_translation_test
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DEXPECTED_PORTS=${SV2SC_TEST_EXPECTED_PORTS}
                -P ${CMAKE_SOURCE_DIR}/cmake/ValidateTranslation.cmake
        )
        
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_systemc_simulation
            COMMAND ${SV2SC_TEST_TEST_NAME}_systemc_test
        )
        
        # Set test dependencies
        set_tests_properties(${SV2SC_TEST_TEST_NAME}_systemc_simulation 
            PROPERTIES DEPENDS ${SV2SC_TEST_TEST_NAME}_translation_test
        )
        
    else()
        # Just create a translation validation test
        add_custom_target(${SV2SC_TEST_TEST_NAME}_translation
            DEPENDS ${TEST_OUTPUT_DIR}/${SV2SC_TEST_TOP_MODULE}.h
        )
        
        add_test(
            NAME ${SV2SC_TEST_TEST_NAME}_translation_test
            COMMAND ${CMAKE_COMMAND} 
                -DTEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}
                -DTOP_MODULE=${SV2SC_TEST_TOP_MODULE}
                -DEXPECTED_PORTS=${SV2SC_TEST_EXPECTED_PORTS}
                -P ${CMAKE_SOURCE_DIR}/cmake/ValidateTranslation.cmake
        )
    endif()
    
endfunction()

# Function to add Verilator comparison test
function(add_verilator_comparison_test)
    set(options "")
    set(oneValueArgs
        TEST_NAME
        TOP_MODULE  
        SV_SOURCE
        SV_TESTBENCH
    )
    set(multiValueArgs "")
    
    cmake_parse_arguments(VLT_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT VLT_TEST_TEST_NAME OR NOT VLT_TEST_TOP_MODULE OR 
       NOT VLT_TEST_SV_SOURCE OR NOT VLT_TEST_SV_TESTBENCH)
        message(FATAL_ERROR "add_verilator_comparison_test requires TEST_NAME, TOP_MODULE, SV_SOURCE, and SV_TESTBENCH")
    endif()
    
    find_program(VERILATOR_EXECUTABLE verilator)
    
    if(VERILATOR_EXECUTABLE)
        set(VLT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/verilator/${VLT_TEST_TEST_NAME})
        
        # Build Verilator simulation
        add_custom_target(${VLT_TEST_TEST_NAME}_verilator
            COMMAND ${CMAKE_COMMAND} -E make_directory ${VLT_OUTPUT_DIR}
            COMMAND ${VERILATOR_EXECUTABLE}
                --cc --exe --build --timing -Wno-TIMESCALEMOD
                -o ${VLT_TEST_TEST_NAME}_sim
                ${VLT_TEST_SV_SOURCE} ${VLT_TEST_SV_TESTBENCH}
                --top-module ${VLT_TEST_TEST_NAME}_tb
            WORKING_DIRECTORY ${VLT_OUTPUT_DIR}
            COMMENT "Building Verilator simulation for ${VLT_TEST_TEST_NAME}"
        )
        
        # Add comparison test
        add_test(
            NAME ${VLT_TEST_TEST_NAME}_verilator_vs_systemc
            COMMAND ${CMAKE_COMMAND}
                -DVERILATOR_SIM=${VLT_OUTPUT_DIR}/${VLT_TEST_TEST_NAME}_sim
                -DSYSTEMC_TEST=${VLT_TEST_TEST_NAME}_systemc_test
                -P ${CMAKE_SOURCE_DIR}/cmake/CompareSimulations.cmake
        )
    else()
        message(WARNING "Verilator not found - skipping Verilator comparison for ${VLT_TEST_TEST_NAME}")
    endif()
    
endfunction()

# Function to create a test suite combining SV2SC translation and Verilator comparison
function(add_complete_sv2sc_test_suite)
    set(options "")
    set(oneValueArgs
        TEST_NAME
        TOP_MODULE
        SV_SOURCE  
        SV_TESTBENCH
        EXPECTED_PORTS
    )
    set(multiValueArgs
        SV_INCLUDES
        SV_DEFINES
    )
    
    cmake_parse_arguments(SUITE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Add SV2SC translation test with SystemC testbench
    add_sv2sc_test(
        TEST_NAME ${SUITE_TEST_NAME}
        TOP_MODULE ${SUITE_TOP_MODULE}
        SV_SOURCE ${SUITE_SV_SOURCE}
        EXPECTED_PORTS ${SUITE_EXPECTED_PORTS}
        SV_INCLUDES ${SUITE_SV_INCLUDES}
        SV_DEFINES ${SUITE_SV_DEFINES}
        GENERATE_TESTBENCH
    )
    
    # Add Verilator comparison if testbench is provided
    if(SUITE_SV_TESTBENCH)
        add_verilator_comparison_test(
            TEST_NAME ${SUITE_TEST_NAME}
            TOP_MODULE ${SUITE_TOP_MODULE}
            SV_SOURCE ${SUITE_SV_SOURCE}
            SV_TESTBENCH ${SUITE_SV_TESTBENCH}
        )
    endif()
    
endfunction()