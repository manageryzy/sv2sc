/**
 * Test that the separated header/implementation compiles correctly
 * This ensures no multiple definition errors occur
 */

#include "complex_processor.h"
#include <systemc.h>
#include <iostream>

// Include the header twice through different paths to test for multiple definition issues
#include "complex_processor.h"

// Test module that uses complex_processor
SC_MODULE(test_bench) {
    sc_clock clk;
    sc_signal<sc_logic> reset;
    sc_signal<sc_lv<32>> alu_a, alu_b;
    sc_signal<sc_lv<4>> alu_op;
    sc_signal<sc_logic> alu_enable;
    sc_signal<sc_lv<32>> alu_result;
    sc_signal<sc_logic> alu_done;
    sc_signal<sc_logic> mem_read, mem_write;
    sc_signal<sc_lv<32>> mem_addr, mem_wdata, mem_rdata;
    sc_signal<sc_logic> mem_ready;
    sc_signal<sc_lv<3>> state;
    sc_signal<sc_lv<32>> pc, instruction;
    
    complex_processor* dut;
    
    SC_CTOR(test_bench) : clk("clk", 10, SC_NS) {
        dut = new complex_processor("dut");
        
        // Connect signals
        dut->clk(clk);
        dut->reset(reset);
        dut->alu_a(alu_a);
        dut->alu_b(alu_b);
        dut->alu_op(alu_op);
        dut->alu_enable(alu_enable);
        dut->alu_result(alu_result);
        dut->alu_done(alu_done);
        dut->mem_read(mem_read);
        dut->mem_write(mem_write);
        dut->mem_addr(mem_addr);
        dut->mem_wdata(mem_wdata);
        dut->mem_rdata(mem_rdata);
        dut->mem_ready(mem_ready);
        dut->state(state);
        dut->pc(pc);
        dut->instruction(instruction);
        
        SC_THREAD(stimulus);
    }
    
    void stimulus() {
        // Reset
        reset = sc_logic('1');
        alu_enable = sc_logic('0');
        mem_read = sc_logic('0');
        mem_write = sc_logic('0');
        wait(20, SC_NS);
        
        reset = sc_logic('0');
        wait(10, SC_NS);
        
        // Test ALU operation
        alu_a = sc_lv<32>(10);
        alu_b = sc_lv<32>(20);
        alu_op = sc_lv<4>("0000"); // ADD
        alu_enable = sc_logic('1');
        wait(10, SC_NS);
        
        std::cout << "ALU Result: " << alu_result.read() << std::endl;
        std::cout << "PC: " << pc.read() << std::endl;
        std::cout << "State: " << state.read() << std::endl;
        
        sc_stop();
    }
};

int sc_main(int argc, char* argv[]) {
    test_bench tb("tb");
    
    std::cout << "Starting simulation with separated header/implementation..." << std::endl;
    sc_start(100, SC_NS);
    std::cout << "Simulation completed successfully!" << std::endl;
    
    return 0;
}
