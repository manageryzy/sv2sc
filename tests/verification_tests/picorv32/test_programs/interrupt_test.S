# Interrupt test program for PicoRV32
# Tests interrupt handling

.section .text
.global _start

_start:
    # Setup interrupt vector
    la t0, irq_handler
    csrw mtvec, t0
    
    # Enable interrupts
    li t0, 0x8      # MIE bit
    csrs mstatus, t0
    li t0, 0x888    # Enable all interrupts
    csrw mie, t0
    
    # Initialize counter
    li s0, 0        # Interrupt count
    li s1, 0        # Main loop count
    
main_loop:
    # Do some work
    addi s1, s1, 1
    
    # Check if we've had interrupts
    li t0, 3
    bge s0, t0, done_test
    
    # Continue looping
    li t0, 1000
    blt s1, t0, main_loop
    
    # If we get here without interrupts, fail
    j fail

irq_handler:
    # Save context
    addi sp, sp, -16
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw ra, 12(sp)
    
    # Increment interrupt counter
    addi s0, s0, 1
    
    # Write to UART
    li t0, 0x10000000
    li t1, 'I'
    sb t1, 0(t0)
    
    # Clear interrupt
    csrr t0, mcause
    li t1, 0x80000000
    and t0, t0, t1
    beqz t0, not_interrupt
    
    # Restore context
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    
    mret

not_interrupt:
    # Not an interrupt, handle as exception
    j fail

done_test:
    # Check if we got expected interrupts
    li t0, 3
    bne s0, t0, fail
    
pass:
    # Success
    li t0, 0x10000000
    li t1, 'I'
    sb t1, 0(t0)
    li t1, 'R'
    sb t1, 0(t0)
    li t1, 'Q'
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
    li t1, 'I'
    sb t1, 0(t0)
    li t1, 'R'
    sb t1, 0(t0)
    li t1, 'Q'
    sb t1, 0(t0)
    li t1, 'F'
    sb t1, 0(t0)
    li t1, '\n'
    sb t1, 0(t0)
    
done:
    # Disable interrupts
    li t0, 0x8
    csrc mstatus, t0
    ebreak
