/**
 * SystemC Testbench for PicoRV32
 * This testbench provides a complete environment for testing the sv2sc-translated PicoRV32
 */

#include <systemc.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include "picorv32.h"

SC_MODULE(picorv32_testbench) {
    // Clock and reset
    sc_clock clk;
    sc_signal<bool> resetn;
    
    // Memory interface signals
    sc_signal<bool> mem_valid;
    sc_signal<bool> mem_instr;
    sc_signal<bool> mem_ready;
    sc_signal<sc_uint<32>> mem_addr;
    sc_signal<sc_uint<32>> mem_wdata;
    sc_signal<sc_uint<4>> mem_wstrb;
    sc_signal<sc_uint<32>> mem_rdata;
    
    // Look-Ahead Interface (optional)
    sc_signal<bool> mem_la_read;
    sc_signal<bool> mem_la_write;
    sc_signal<sc_uint<32>> mem_la_addr;
    sc_signal<sc_uint<32>> mem_la_wdata;
    sc_signal<sc_uint<4>> mem_la_wstrb;
    
    // Pico Co-Processor Interface (optional)
    sc_signal<bool> pcpi_valid;
    sc_signal<sc_uint<32>> pcpi_insn;
    sc_signal<sc_uint<32>> pcpi_rs1;
    sc_signal<sc_uint<32>> pcpi_rs2;
    sc_signal<bool> pcpi_wr;
    sc_signal<sc_uint<32>> pcpi_rd;
    sc_signal<bool> pcpi_wait;
    sc_signal<bool> pcpi_ready;
    
    // IRQ Interface
    sc_signal<sc_uint<32>> irq;
    sc_signal<sc_uint<32>> eoi;
    
    // Trace Interface (optional)
    sc_signal<bool> trace_valid;
    sc_signal<sc_uint<36>> trace_data;
    
    // Additional control signals
    sc_signal<bool> trap;
    
    // Memory model (simplified)
    std::vector<uint32_t> memory;
    static const uint32_t MEM_SIZE = 0x100000; // 1MB memory
    
    // DUT instance
    picorv32* dut;
    
    // Constructor
    SC_CTOR(picorv32_testbench) : 
        clk("clk", 10, SC_NS),
        memory(MEM_SIZE / 4, 0)
    {
        // Instantiate DUT
        dut = new picorv32("picorv32");
        
        // Connect clock and reset
        dut->clk(clk);
        dut->resetn(resetn);
        
        // Connect memory interface
        dut->mem_valid(mem_valid);
        dut->mem_instr(mem_instr);
        dut->mem_ready(mem_ready);
        dut->mem_addr(mem_addr);
        dut->mem_wdata(mem_wdata);
        dut->mem_wstrb(mem_wstrb);
        dut->mem_rdata(mem_rdata);
        
        // Connect look-ahead interface
        dut->mem_la_read(mem_la_read);
        dut->mem_la_write(mem_la_write);
        dut->mem_la_addr(mem_la_addr);
        dut->mem_la_wdata(mem_la_wdata);
        dut->mem_la_wstrb(mem_la_wstrb);
        
        // Connect co-processor interface
        dut->pcpi_valid(pcpi_valid);
        dut->pcpi_insn(pcpi_insn);
        dut->pcpi_rs1(pcpi_rs1);
        dut->pcpi_rs2(pcpi_rs2);
        dut->pcpi_wr(pcpi_wr);
        dut->pcpi_rd(pcpi_rd);
        dut->pcpi_wait(pcpi_wait);
        dut->pcpi_ready(pcpi_ready);
        
        // Connect IRQ interface
        dut->irq(irq);
        dut->eoi(eoi);
        
        // Connect trace interface
        dut->trace_valid(trace_valid);
        dut->trace_data(trace_data);
        
        // Connect trap signal
        dut->trap(trap);
        
        // Register processes
        SC_THREAD(stimulus_process);
        SC_METHOD(memory_process);
        sensitive << clk.pos();
        SC_METHOD(monitor_process);
        sensitive << clk.pos();
        SC_METHOD(trace_process);
        sensitive << trace_valid;
    }
    
    // Load hex file into memory
    bool load_hex(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open hex file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line[0] == '@') {
                // Address line
                uint32_t addr = std::stoul(line.substr(1), nullptr, 16);
                continue;
            } else {
                // Data line
                std::istringstream iss(line);
                std::string hex_value;
                uint32_t addr = 0; // Should track from @ lines
                while (iss >> hex_value) {
                    uint32_t value = std::stoul(hex_value, nullptr, 16);
                    if (addr < MEM_SIZE) {
                        memory[addr / 4] = value;
                    }
                    addr += 4;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded hex file: " << filename << std::endl;
        return true;
    }
    
    // Stimulus process
    void stimulus_process() {
        // Initialize signals
        resetn.write(false);
        irq.write(0);
        pcpi_wr.write(0);
        pcpi_rd.write(0);
        pcpi_wait.write(0);
        pcpi_ready.write(0);
        
        // Reset sequence
        wait(100, SC_NS);
        resetn.write(true);
        std::cout << "Reset released at " << sc_time_stamp() << std::endl;
        
        // Run for specified time or until trap
        for (int cycle = 0; cycle < 10000; cycle++) {
            wait(clk.period());
            
            if (trap.read()) {
                std::cout << "CPU trap at " << sc_time_stamp() << std::endl;
                break;
            }
            
            // Simulate interrupts occasionally
            if (cycle == 5000) {
                irq.write(0x1);
                wait(clk.period());
                irq.write(0x0);
            }
        }
        
        // Final statistics
        print_statistics();
        sc_stop();
    }
    
    // Memory process - handles memory read/write requests
    void memory_process() {
        if (mem_valid.read() && !mem_ready.read()) {
            uint32_t addr = mem_addr.read();
            
            if (mem_wstrb.read() != 0) {
                // Write operation
                uint32_t wdata = mem_wdata.read();
                uint32_t wstrb = mem_wstrb.read();
                uint32_t mem_idx = (addr & 0xFFFFF) / 4;
                
                if (mem_idx < memory.size()) {
                    uint32_t old_val = memory[mem_idx];
                    uint32_t new_val = old_val;
                    
                    if (wstrb & 0x1) new_val = (new_val & 0xFFFFFF00) | (wdata & 0x000000FF);
                    if (wstrb & 0x2) new_val = (new_val & 0xFFFF00FF) | (wdata & 0x0000FF00);
                    if (wstrb & 0x4) new_val = (new_val & 0xFF00FFFF) | (wdata & 0x00FF0000);
                    if (wstrb & 0x8) new_val = (new_val & 0x00FFFFFF) | (wdata & 0xFF000000);
                    
                    memory[mem_idx] = new_val;
                    
                    // Special addresses for I/O
                    if (addr == 0x10000000) {
                        // UART output
                        std::cout << "UART: " << static_cast<char>(wdata & 0xFF);
                    }
                }
            } else {
                // Read operation
                uint32_t mem_idx = (addr & 0xFFFFF) / 4;
                if (mem_idx < memory.size()) {
                    mem_rdata.write(memory[mem_idx]);
                } else {
                    mem_rdata.write(0);
                }
            }
            
            // Memory is always ready next cycle (simplified)
            mem_ready.write(true);
        } else {
            mem_ready.write(false);
        }
    }
    
    // Monitor process - tracks CPU state
    void monitor_process() {
        static int instruction_count = 0;
        static int cycle_count = 0;
        
        if (resetn.read()) {
            cycle_count++;
            
            if (mem_valid.read() && mem_ready.read() && mem_instr.read()) {
                instruction_count++;
                
                if (instruction_count % 1000 == 0) {
                    std::cout << "Instructions executed: " << instruction_count 
                             << " at " << sc_time_stamp() << std::endl;
                }
            }
        }
    }
    
    // Trace process - handles instruction trace output
    void trace_process() {
        if (trace_valid.read()) {
            uint64_t trace = trace_data.read();
            std::cout << "TRACE: 0x" << std::hex << trace << std::dec 
                     << " at " << sc_time_stamp() << std::endl;
        }
    }
    
    // Print statistics
    void print_statistics() {
        std::cout << "\n=== Simulation Statistics ===" << std::endl;
        std::cout << "Simulation time: " << sc_time_stamp() << std::endl;
        // Additional statistics can be added here
    }
    
    // Destructor
    ~picorv32_testbench() {
        delete dut;
    }
};

// Main function
int sc_main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string hex_file = "firmware.hex";
    std::string vcd_file = "";
    int timeout_cycles = 100000;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hex" && i + 1 < argc) {
            hex_file = argv[++i];
        } else if (arg == "--vcd" && i + 1 < argc) {
            vcd_file = argv[++i];
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeout_cycles = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --hex <file>     Load hex file (default: firmware.hex)" << std::endl;
            std::cout << "  --vcd <file>     Generate VCD file" << std::endl;
            std::cout << "  --timeout <n>    Timeout in cycles (default: 100000)" << std::endl;
            return 0;
        }
    }
    
    // Create testbench
    picorv32_testbench tb("testbench");
    
    // Load program
    if (!hex_file.empty()) {
        if (!tb.load_hex(hex_file)) {
            std::cerr << "Warning: Could not load hex file, using empty memory" << std::endl;
        }
    }
    
    // Setup VCD tracing if requested
    sc_trace_file* vcd = nullptr;
    if (!vcd_file.empty()) {
        vcd = sc_create_vcd_trace_file(vcd_file.c_str());
        sc_trace(vcd, tb.clk, "clk");
        sc_trace(vcd, tb.resetn, "resetn");
        sc_trace(vcd, tb.mem_valid, "mem_valid");
        sc_trace(vcd, tb.mem_ready, "mem_ready");
        sc_trace(vcd, tb.mem_addr, "mem_addr");
        sc_trace(vcd, tb.mem_wdata, "mem_wdata");
        sc_trace(vcd, tb.mem_rdata, "mem_rdata");
        sc_trace(vcd, tb.trap, "trap");
    }
    
    // Set timeout
    sc_set_time_resolution(1, SC_NS);
    sc_start(timeout_cycles * 10, SC_NS);
    
    // Cleanup
    if (vcd) {
        sc_close_vcd_trace_file(vcd);
    }
    
    return 0;
}
