/**
 * Verilator Testbench for PicoRV32
 * Reference implementation for comparison with sv2sc translation
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vpicorv32.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iomanip>

class PicoRV32_TB {
public:
    Vpicorv32* dut;
    VerilatedVcdC* tfp;
    vluint64_t sim_time;
    std::vector<uint32_t> memory;
    static const uint32_t MEM_SIZE = 0x100000; // 1MB
    
    PicoRV32_TB() : sim_time(0), memory(MEM_SIZE / 4, 0) {
        // Create DUT
        dut = new Vpicorv32;
        
        // Setup tracing
        Verilated::traceEverOn(true);
        tfp = nullptr;
        
        // Initialize signals
        dut->clk = 0;
        dut->resetn = 0;
        dut->mem_ready = 0;
        dut->mem_rdata = 0;
        dut->pcpi_wr = 0;
        dut->pcpi_rd = 0;
        dut->pcpi_wait = 0;
        dut->pcpi_ready = 0;
        dut->irq = 0;
    }
    
    ~PicoRV32_TB() {
        if (tfp) {
            tfp->close();
            delete tfp;
        }
        delete dut;
    }
    
    void open_trace(const std::string& filename) {
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open(filename.c_str());
    }
    
    void close_trace() {
        if (tfp) {
            tfp->close();
            delete tfp;
            tfp = nullptr;
        }
    }
    
    bool load_hex(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open hex file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        uint32_t current_addr = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '/' || line[0] == '#') {
                continue; // Skip comments and empty lines
            }
            
            if (line[0] == '@') {
                // Address specification
                current_addr = std::stoul(line.substr(1), nullptr, 16);
            } else {
                // Data values
                std::istringstream iss(line);
                std::string hex_value;
                while (iss >> hex_value) {
                    uint32_t value = std::stoul(hex_value, nullptr, 16);
                    uint32_t mem_idx = current_addr / 4;
                    if (mem_idx < memory.size()) {
                        memory[mem_idx] = value;
                    }
                    current_addr += 4;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded hex file: " << filename << std::endl;
        return true;
    }
    
    void tick() {
        // Toggle clock
        dut->clk = 0;
        dut->eval();
        if (tfp) tfp->dump(sim_time++);
        
        dut->clk = 1;
        dut->eval();
        if (tfp) tfp->dump(sim_time++);
        
        // Handle memory interface
        handle_memory();
    }
    
    void handle_memory() {
        static bool mem_valid_last = false;
        static uint32_t mem_addr_last = 0;
        static uint32_t mem_wstrb_last = 0;
        static uint32_t mem_wdata_last = 0;
        
        if (dut->mem_valid && !mem_valid_last) {
            // New memory request
            mem_addr_last = dut->mem_addr;
            mem_wstrb_last = dut->mem_wstrb;
            mem_wdata_last = dut->mem_wdata;
        }
        
        if (dut->mem_valid && !dut->mem_ready) {
            uint32_t addr = mem_addr_last;
            uint32_t mem_idx = (addr & (MEM_SIZE - 1)) / 4;
            
            if (mem_wstrb_last != 0) {
                // Write operation
                uint32_t wdata = mem_wdata_last;
                uint32_t wstrb = mem_wstrb_last;
                
                if (mem_idx < memory.size()) {
                    uint32_t old_val = memory[mem_idx];
                    uint32_t new_val = old_val;
                    
                    if (wstrb & 0x1) new_val = (new_val & 0xFFFFFF00) | (wdata & 0x000000FF);
                    if (wstrb & 0x2) new_val = (new_val & 0xFFFF00FF) | (wdata & 0x0000FF00);
                    if (wstrb & 0x4) new_val = (new_val & 0xFF00FFFF) | (wdata & 0x00FF0000);
                    if (wstrb & 0x8) new_val = (new_val & 0x00FFFFFF) | (wdata & 0xFF000000);
                    
                    memory[mem_idx] = new_val;
                    
                    // Special I/O addresses
                    if (addr == 0x10000000) {
                        // UART output
                        std::cout << static_cast<char>(wdata & 0xFF);
                        std::cout.flush();
                    }
                }
            } else {
                // Read operation
                if (mem_idx < memory.size()) {
                    dut->mem_rdata = memory[mem_idx];
                } else {
                    dut->mem_rdata = 0;
                }
            }
            
            dut->mem_ready = 1;
        } else {
            dut->mem_ready = 0;
        }
        
        mem_valid_last = dut->mem_valid;
    }
    
    void reset(int cycles = 10) {
        dut->resetn = 0;
        for (int i = 0; i < cycles; i++) {
            tick();
        }
        dut->resetn = 1;
        std::cout << "Reset released at cycle " << sim_time/2 << std::endl;
    }
    
    bool run(int max_cycles = 100000) {
        int cycle = 0;
        int instruction_count = 0;
        
        while (cycle < max_cycles) {
            tick();
            cycle++;
            
            // Check for trap
            if (dut->trap) {
                std::cout << "CPU trap at cycle " << cycle << std::endl;
                return true;
            }
            
            // Count instructions
            if (dut->mem_valid && dut->mem_ready && dut->mem_instr) {
                instruction_count++;
                if (instruction_count % 1000 == 0) {
                    std::cout << "Instructions: " << instruction_count 
                             << " at cycle " << cycle << std::endl;
                }
            }
            
            // Simulate interrupt at specific cycle
            if (cycle == 5000) {
                dut->irq = 0x1;
            } else if (cycle == 5001) {
                dut->irq = 0x0;
            }
        }
        
        std::cout << "Simulation timeout at cycle " << cycle << std::endl;
        return false;
    }
    
    void print_statistics() {
        std::cout << "\n=== Verilator Simulation Statistics ===" << std::endl;
        std::cout << "Total cycles: " << sim_time/2 << std::endl;
    }
};

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Parse arguments
    std::string hex_file = "firmware.hex";
    std::string vcd_file = "";
    int timeout = 100000;
    bool enable_trace = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hex" && i + 1 < argc) {
            hex_file = argv[++i];
        } else if (arg == "--vcd" && i + 1 < argc) {
            vcd_file = argv[++i];
            enable_trace = true;
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeout = std::atoi(argv[++i]);
        } else if (arg == "+trace") {
            enable_trace = true;
            if (vcd_file.empty()) {
                vcd_file = "verilator.vcd";
            }
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --hex <file>     Load hex file (default: firmware.hex)" << std::endl;
            std::cout << "  --vcd <file>     Generate VCD file" << std::endl;
            std::cout << "  +trace           Enable tracing (default: verilator.vcd)" << std::endl;
            std::cout << "  --timeout <n>    Timeout in cycles (default: 100000)" << std::endl;
            return 0;
        }
    }
    
    // Create testbench
    PicoRV32_TB* tb = new PicoRV32_TB();
    
    // Load program
    if (!hex_file.empty()) {
        if (!tb->load_hex(hex_file)) {
            std::cerr << "Warning: Could not load hex file" << std::endl;
        }
    }
    
    // Enable tracing if requested
    if (enable_trace && !vcd_file.empty()) {
        tb->open_trace(vcd_file);
    }
    
    // Reset and run
    tb->reset();
    bool success = tb->run(timeout);
    
    // Print statistics
    tb->print_statistics();
    
    // Cleanup
    if (enable_trace) {
        tb->close_trace();
    }
    delete tb;
    
    return success ? 0 : 1;
}
