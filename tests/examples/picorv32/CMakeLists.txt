# PicoRV32 RISC-V Processor Core Translation Example
# This is a comprehensive test of the sv2sc translator on a real-world processor

# The PicoRV32 processor has a complex interface with many ports and parameters
# We'll focus on the basic configuration without RISC-V formal verification
add_complete_sv2sc_test_suite(
    TEST_NAME picorv32_sv2sc
    TOP_MODULE picorv32
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/picorv32/picorv32.v
    EXPECTED_PORTS 20  # Base ports without RISCV_FORMAL interface
)

# Additional test with simplified configuration
# This creates a simpler test that disables most advanced features
add_sv2sc_test(
    TEST_NAME picorv32_simple_sv2sc
    TOP_MODULE picorv32
    SV_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/picorv32/picorv32.v
    TRANSLATOR_ARGS -D ENABLE_COUNTERS=0 -D ENABLE_REGS_16_31=0 -D ENABLE_PCPI=0 -D ENABLE_IRQ=0 -D ENABLE_TRACE=0
    EXPECTED_PORTS 8  # Minimal configuration ports
)

# Note: PicoRV32 is a complex processor core with:
# - 25+ configurable parameters 
# - Memory interface with look-ahead signals
# - Optional PCPI (Pico Co-Processor Interface)
# - Optional IRQ handling
# - Optional RISC-V formal verification interface
# - Optional trace interface
#
# This translation test will stress-test the sv2sc translator's ability to:
# - Handle large modules (3000+ lines)
# - Process multiple conditional compilation blocks
# - Translate complex state machines
# - Handle parametric module definitions
# - Convert various SystemVerilog constructs to SystemC