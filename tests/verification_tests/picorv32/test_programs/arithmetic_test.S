# Arithmetic test program for PicoRV32
# Tests ALU operations

.section .text
.global _start

_start:
    # Test ADD
    li t0, 100
    li t1, 200
    add t2, t0, t1     # t2 = 300
    li t3, 300
    bne t2, t3, fail
    
    # Test SUB
    sub t2, t1, t0     # t2 = 100
    li t3, 100
    bne t2, t3, fail
    
    # Test AND
    li t0, 0xFF00
    li t1, 0x0FF0
    and t2, t0, t1     # t2 = 0x0F00
    li t3, 0x0F00
    bne t2, t3, fail
    
    # Test OR
    or t2, t0, t1      # t2 = 0xFFF0
    li t3, 0xFFF0
    bne t2, t3, fail
    
    # Test XOR
    li t0, 0xAAAA
    li t1, 0x5555
    xor t2, t0, t1     # t2 = 0xFFFF
    li t3, 0xFFFF
    bne t2, t3, fail
    
    # Test shifts
    li t0, 0x1
    slli t1, t0, 4     # t1 = 0x10
    li t3, 0x10
    bne t1, t3, fail
    
    srli t2, t1, 2     # t2 = 0x4
    li t3, 0x4
    bne t2, t3, fail
    
    # Test SLT
    li t0, -10
    li t1, 10
    slt t2, t0, t1     # t2 = 1 (true)
    li t3, 1
    bne t2, t3, fail
    
pass:
    # Success
    li t0, 0x10000000
    li t1, 'A'
    sb t1, 0(t0)
    li t1, 'L'
    sb t1, 0(t0)
    li t1, 'U'
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
    li t1, 'E'
    sb t1, 0(t0)
    li t1, 'R'
    sb t1, 0(t0)
    li t1, 'R'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    
done:
    ebreak
