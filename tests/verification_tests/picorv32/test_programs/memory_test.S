# Memory test program for PicoRV32
# Tests various memory access patterns

.section .text
.global _start

_start:
    # Initialize stack pointer
    li sp, 0x2000
    
    # Test byte operations
    li t0, 0x2000
    li t1, 0xAB
    sb t1, 0(t0)       # Store byte
    lb t2, 0(t0)       # Load byte
    
    # Test halfword operations
    li t1, 0x1234
    sh t1, 2(t0)       # Store halfword
    lh t2, 2(t0)       # Load halfword
    
    # Test word operations
    li t1, 0xDEADBEEF
    sw t1, 4(t0)       # Store word
    lw t2, 4(t0)       # Load word
    
    # Test memory pattern
    li t0, 0x1000      # Base address
    li t3, 16          # Loop counter
    li t4, 0           # Pattern
    
loop:
    sw t4, 0(t0)       # Store pattern
    addi t4, t4, 1     # Increment pattern
    addi t0, t0, 4     # Next word
    addi t3, t3, -1    # Decrement counter
    bnez t3, loop      # Continue if not zero
    
    # Verify pattern
    li t0, 0x1000      # Reset base
    li t3, 16          # Reset counter
    li t4, 0           # Expected pattern
    
verify:
    lw t5, 0(t0)       # Load value
    bne t4, t5, fail   # Check match
    addi t4, t4, 1     # Next expected
    addi t0, t0, 4     # Next address
    addi t3, t3, -1    # Decrement
    bnez t3, verify    # Continue
    
pass:
    # Success - write to UART
    li t0, 0x10000000
    li t1, 'M'
    sb t1, 0(t0)
    li t1, 'E'
    sb t1, 0(t0)
    li t1, 'M'
    sb t1, 0(t0)
    li t1, 'O'
    sb t1, 0(t0)
    li t1, 'K'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    j done
    
fail:
    # Failure - write error
    li t0, 0x10000000
    li t1, 'F'
    sb t1, 0(t0)
    li t1, 'A'
    sb t1, 0(t0)
    li t1, 'I'
    sb t1, 0(t0)
    li t1, 'L'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    
done:
    ebreak
