# PicoRV32 Verification Suite
# Compares sv2sc translation with Verilator simulation

cmake_minimum_required(VERSION 3.20)
project(picorv32_verification)

# Options for verification
option(BUILD_VERILATOR_REF "Build Verilator reference implementation" ON)
option(BUILD_SV2SC_IMPL "Build sv2sc translated implementation" ON)
option(RUN_COMPARISON "Run comparison tests" ON)
option(GENERATE_WAVEFORMS "Generate VCD waveforms" OFF)

# Find required tools
find_program(VERILATOR verilator)
if(NOT VERILATOR AND BUILD_VERILATOR_REF)
    message(WARNING "Verilator not found, disabling reference build")
    set(BUILD_VERILATOR_REF OFF)
endif()

# Set paths
set(PICORV32_SOURCE "${CMAKE_SOURCE_DIR}/third-party/picorv32/picorv32.v")
set(SV2SC_EXECUTABLE "${CMAKE_BINARY_DIR}/src/sv2sc")
set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/output")
set(VERILATOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/verilator")
set(SYSTEMC_DIR "${CMAKE_CURRENT_BINARY_DIR}/systemc")

# Create output directories
file(MAKE_DIRECTORY ${OUTPUT_DIR})
file(MAKE_DIRECTORY ${VERILATOR_DIR})
file(MAKE_DIRECTORY ${SYSTEMC_DIR})

# ============================================================================
# sv2sc Translation
# ============================================================================
if(BUILD_SV2SC_IMPL)
    message(STATUS "Configuring sv2sc translation for PicoRV32")
    
    # Custom command to translate PicoRV32 with sv2sc
    add_custom_command(
        OUTPUT ${SYSTEMC_DIR}/picorv32.h ${SYSTEMC_DIR}/picorv32.cpp
        COMMAND ${SV2SC_EXECUTABLE}
            -top picorv32
            -o ${SYSTEMC_DIR}
            ${PICORV32_SOURCE}
        DEPENDS ${SV2SC_EXECUTABLE} ${PICORV32_SOURCE}
        COMMENT "Translating PicoRV32 to SystemC using sv2sc"
        VERBATIM
    )
    
    # Create SystemC executable
    add_executable(picorv32_systemc
        ${SYSTEMC_DIR}/picorv32.h
        ${SYSTEMC_DIR}/picorv32.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testbench/systemc_testbench.cpp
    )
    
    target_include_directories(picorv32_systemc PRIVATE
        ${SYSTEMC_DIR}
        ${SystemC_INCLUDE_DIRS}
    )
    
    target_link_libraries(picorv32_systemc PRIVATE
        ${SystemC_LIBRARIES}
        pthread
    )
    
    # Set C++ standard for SystemC
    set_target_properties(picorv32_systemc PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
    )
endif()

# ============================================================================
# Verilator Reference Build
# ============================================================================
if(BUILD_VERILATOR_REF)
    message(STATUS "Configuring Verilator build for PicoRV32")
    
    # Verilator flags
    set(VERILATOR_FLAGS
        --cc                          # Generate C++ output
        --exe                         # Generate executable
        --build                       # Build immediately
        -Wall                         # Enable all warnings
        --trace                       # Enable waveform tracing
        --assert                      # Enable assertions
        --coverage                    # Enable coverage
        -O3                          # Optimization level
        --x-assign fast              # Handle X assignments
        --x-initial fast             # Handle X initial values
        --Mdir ${VERILATOR_DIR}     # Output directory
    )
    
    # Custom command to build with Verilator
    add_custom_command(
        OUTPUT ${VERILATOR_DIR}/Vpicorv32
        COMMAND ${VERILATOR}
            ${VERILATOR_FLAGS}
            ${CMAKE_CURRENT_SOURCE_DIR}/testbench/verilator_testbench.cpp
            ${PICORV32_SOURCE}
        WORKING_DIRECTORY ${VERILATOR_DIR}
        DEPENDS ${PICORV32_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/testbench/verilator_testbench.cpp
        COMMENT "Building PicoRV32 with Verilator"
        VERBATIM
    )
    
    # Create custom target for Verilator build
    add_custom_target(picorv32_verilator ALL
        DEPENDS ${VERILATOR_DIR}/Vpicorv32
    )
endif()

# ============================================================================
# Comparison Tests
# ============================================================================
if(RUN_COMPARISON)
    # Test programs for PicoRV32
    set(TEST_PROGRAMS
        simple_test
        memory_test
        arithmetic_test
        branch_test
        interrupt_test
    )
    
    # Generate test programs
    foreach(test_prog ${TEST_PROGRAMS})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/test_programs/${test_prog}.S
            ${OUTPUT_DIR}/${test_prog}.S
            COPYONLY
        )
        
        # Assemble test program (requires riscv toolchain)
        find_program(RISCV_AS riscv32-unknown-elf-as riscv64-unknown-elf-as)
        find_program(RISCV_LD riscv32-unknown-elf-ld riscv64-unknown-elf-ld)
        find_program(RISCV_OBJCOPY riscv32-unknown-elf-objcopy riscv64-unknown-elf-objcopy)
        
        if(RISCV_AS AND RISCV_LD AND RISCV_OBJCOPY)
            add_custom_command(
                OUTPUT ${OUTPUT_DIR}/${test_prog}.hex
                COMMAND ${RISCV_AS} -march=rv32i -o ${OUTPUT_DIR}/${test_prog}.o ${OUTPUT_DIR}/${test_prog}.S
                COMMAND ${RISCV_LD} -Ttext=0x00000000 -o ${OUTPUT_DIR}/${test_prog}.elf ${OUTPUT_DIR}/${test_prog}.o
                COMMAND ${RISCV_OBJCOPY} -O verilog ${OUTPUT_DIR}/${test_prog}.elf ${OUTPUT_DIR}/${test_prog}.hex
                DEPENDS ${OUTPUT_DIR}/${test_prog}.S
                COMMENT "Building test program: ${test_prog}"
            )
            
            # Create test target
            add_test(
                NAME picorv32_compare_${test_prog}
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_outputs.py
                    --systemc $<TARGET_FILE:picorv32_systemc>
                    --verilator ${VERILATOR_DIR}/Vpicorv32
                    --program ${OUTPUT_DIR}/${test_prog}.hex
                    --timeout 10000
                    --output ${OUTPUT_DIR}/${test_prog}_comparison.log
            )
        else()
            message(WARNING "RISC-V toolchain not found, skipping test program generation")
        endif()
    endforeach()
    
    # Add comparison test
    if(BUILD_SV2SC_IMPL AND BUILD_VERILATOR_REF)
        add_test(
            NAME picorv32_basic_comparison
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_comparison.sh
                ${SYSTEMC_DIR}
                ${VERILATOR_DIR}
                ${OUTPUT_DIR}
        )
    endif()
endif()

# ============================================================================
# Performance Benchmarking
# ============================================================================
add_custom_target(benchmark
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark.py
        --systemc $<TARGET_FILE:picorv32_systemc>
        --verilator ${VERILATOR_DIR}/Vpicorv32
        --iterations 1000
        --output ${OUTPUT_DIR}/benchmark_results.json
    DEPENDS picorv32_systemc picorv32_verilator
    COMMENT "Running performance benchmark"
)

# ============================================================================
# Waveform Generation
# ============================================================================
if(GENERATE_WAVEFORMS)
    add_custom_target(generate_waveforms
        COMMAND $<TARGET_FILE:picorv32_systemc> --vcd ${OUTPUT_DIR}/systemc.vcd
        COMMAND ${VERILATOR_DIR}/Vpicorv32 +trace
        DEPENDS picorv32_systemc picorv32_verilator
        COMMENT "Generating waveforms for comparison"
    )
endif()

# ============================================================================
# Report Generation
# ============================================================================
add_custom_target(verification_report
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_report.py
        --systemc-dir ${SYSTEMC_DIR}
        --verilator-dir ${VERILATOR_DIR}
        --output-dir ${OUTPUT_DIR}
        --report ${OUTPUT_DIR}/verification_report.html
    COMMENT "Generating verification report"
)

# Summary
message(STATUS "=== PicoRV32 Verification Configuration ===")
message(STATUS "  sv2sc translation: ${BUILD_SV2SC_IMPL}")
message(STATUS "  Verilator build: ${BUILD_VERILATOR_REF}")
message(STATUS "  Comparison tests: ${RUN_COMPARISON}")
message(STATUS "  Waveform generation: ${GENERATE_WAVEFORMS}")
message(STATUS "  Output directory: ${OUTPUT_DIR}")
