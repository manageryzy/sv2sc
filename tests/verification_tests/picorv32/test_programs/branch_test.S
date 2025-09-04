# Branch test program for PicoRV32
# Tests control flow instructions

.section .text
.global _start

_start:
    # Test BEQ (branch if equal)
    li t0, 10
    li t1, 10
    beq t0, t1, beq_pass
    j fail
beq_pass:
    
    # Test BNE (branch if not equal)
    li t0, 10
    li t1, 20
    bne t0, t1, bne_pass
    j fail
bne_pass:
    
    # Test BLT (branch if less than)
    li t0, -5
    li t1, 10
    blt t0, t1, blt_pass
    j fail
blt_pass:
    
    # Test BGE (branch if greater or equal)
    li t0, 10
    li t1, 10
    bge t0, t1, bge_pass
    j fail
bge_pass:
    
    # Test BLTU (branch if less than unsigned)
    li t0, 0x80000000
    li t1, 0x7FFFFFFF
    bltu t1, t0, bltu_pass
    j fail
bltu_pass:
    
    # Test BGEU (branch if greater or equal unsigned)
    li t0, 0xFFFFFFFF
    li t1, 0x00000000
    bgeu t0, t1, bgeu_pass
    j fail
bgeu_pass:
    
    # Test JAL (jump and link)
    jal ra, function_test
    li t0, 42
    bne t0, a0, fail
    
    # Test loop
    li t0, 5        # Loop counter
    li t1, 0        # Sum
loop:
    add t1, t1, t0  # Add counter to sum
    addi t0, t0, -1 # Decrement counter
    bnez t0, loop   # Continue if not zero
    
    # Check sum (5+4+3+2+1 = 15)
    li t2, 15
    bne t1, t2, fail
    
pass:
    # Success
    li t0, 0x10000000
    li t1, 'B'
    sb t1, 0(t0)
    li t1, 'R'
    sb t1, 0(t0)
    li t1, 'O'
    sb t1, 0(t0)
    li t1, 'K'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    j done
    
fail:
    # Failure
    li t0, 0x10000000
    li t1, 'B'
    sb t1, 0(t0)
    li t1, 'A'
    sb t1, 0(t0)
    li t1, 'D'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    
done:
    ebreak

# Test function
function_test:
    li a0, 42       # Return value
    jalr zero, ra, 0 # Return
