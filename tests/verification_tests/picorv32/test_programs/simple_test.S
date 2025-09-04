# Simple test program for PicoRV32
# Tests basic arithmetic and memory operations

.section .text
.global _start

_start:
    # Initialize registers
    li x1, 0            # Zero out x1
    li x2, 0x100        # Stack pointer
    li x3, 0            # Zero out x3
    
    # Test basic arithmetic
    li x4, 10           # Load 10 into x4
    li x5, 20           # Load 20 into x5
    add x6, x4, x5      # x6 = 10 + 20 = 30
    sub x7, x5, x4      # x7 = 20 - 10 = 10
    
    # Test memory operations
    li x8, 0x1000       # Memory address
    sw x6, 0(x8)        # Store 30 at address 0x1000
    lw x9, 0(x8)        # Load from address 0x1000 into x9
    
    # Test branch
    beq x6, x9, pass    # Branch if x6 == x9 (should be equal)
    
fail:
    li x10, 0xDEAD      # Load failure code
    j done
    
pass:
    li x10, 0xGOOD      # Load success code
    
done:
    # Write to UART (at address 0x10000000)
    li x11, 0x10000000  # UART address
    li x12, 'P'         # Character 'P' for Pass
    sb x12, 0(x11)      # Write to UART
    li x12, 'A'
    sb x12, 0(x11)
    li x12, 'S'
    sb x12, 0(x11)
    li x12, 'S'
    sb x12, 0(x11)
    li x12, '\n'
    sb x12, 0(x11)
    
    # Infinite loop (trap)
    ebreak              # Trigger trap
    
.section .data
    .word 0x12345678
    .word 0xABCDEF00
