# SystemVerilog to SystemC Translator - User Guide

## Quick Start

### Installation
```bash
git clone <repository>
cd sv2sc
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### First Translation
```bash
# Create a simple SystemVerilog module
echo 'module test(input logic a, output logic b); assign b = a; endmodule' > test.sv

# Translate to SystemC
./sv2sc -top test test.sv

# Check generated files
ls -la output/
cat output/test.h
cat output/test.cpp
```

## Command Line Reference

### Basic Syntax
```
sv2sc [options] <input_files...>
```

### VCS-Compatible Options

#### File and Module Options
- `-top <module>`: Specify top-level module name
- `-o <directory>`: Set output directory (default: ./output)

#### Include and Library Paths
- `-I <directory>`: Add include directory
- `+incdir+<directory>`: Add include directory (VCS format)
- `-y <directory>`: Add library directory

#### Preprocessor Options
- `-D <name>[=<value>]`: Define preprocessor macro
- `+define+<name>=<value>`: Define preprocessor macro (VCS format)
- `-U <name>`: Undefine preprocessor macro
- `-timescale <scale>`: Set time scale (e.g., 1ns/1ps)

#### SystemC-Specific Options
- `--clock <signal>`: Specify main clock signal name (default: clk)
- `--reset <signal>`: Specify reset signal name (default: reset)
- `--testbench`: Generate SystemC testbench wrapper
- `--synthesis`: Enable synthesis-oriented translation

#### Output Control
- `--debug`: Enable debug output and logging
- `-v, --verbose`: Enable verbose status messages
- `-V, --version`: Show version information
- `-h, --help`: Display help message

## Translation Examples

### Simple Module
**Input (counter.sv):**
```systemverilog
module counter(
    input logic clk,
    input logic reset,
    output logic [7:0] count
);
    logic [7:0] count_reg;
    
    always_ff @(posedge clk) begin
        if (reset)
            count_reg <= 8'h00;
        else
            count_reg <= count_reg + 1;
    end
    
    assign count = count_reg;
endmodule
```

**Command:**
```bash
sv2sc -top counter counter.sv
```

**Output (counter.h):**
```cpp
#ifndef COUNTER_H
#define COUNTER_H

#include <systemc.h>

SC_MODULE(counter) {
    sc_in<sc_logic> clk;
    sc_in<sc_logic> reset;
    sc_out<sc_lv<8>> count;
    
    sc_signal<sc_lv<8>> count_reg;
    
    SC_CTOR(counter) {
        SC_METHOD(comb_proc);
        sensitive << clk.pos();
        sensitive << reset;
    }

private:
    void comb_proc() {
        // Implementation details
    }
};

#endif
```

### Parameterized Memory
**Input (memory.sv):**
```systemverilog
module memory #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
)(
    input logic clk,
    input logic we,
    input logic [ADDR_WIDTH-1:0] addr,
    input logic [DATA_WIDTH-1:0] din,
    output logic [DATA_WIDTH-1:0] dout
);
    logic [DATA_WIDTH-1:0] mem [2**ADDR_WIDTH-1:0];
    
    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= din;
        dout <= mem[addr];
    end
endmodule
```

**Command:**
```bash
sv2sc -D ADDR_WIDTH=10 -D DATA_WIDTH=16 -top memory memory.sv
```

### Complex Project
**Command:**
```bash
sv2sc -I ./include -I ./rtl \
      +incdir+./interfaces \
      -D SYNTHESIS=1 \
      -D CLOCK_FREQ=100000000 \
      -y ./lib \
      -top cpu_top \
      -o ./systemc_output \
      --clock clk_i \
      --reset rst_n \
      --verbose \
      rtl/cpu_top.sv rtl/alu.sv rtl/regfile.sv
```

## Supported SystemVerilog Features

### ✅ Fully Supported
- **Module declarations** with parameters
- **Port declarations** (input, output, inout)  
- **Data types**: logic, bit, reg, wire
- **Vector types** with bit ranges
- **Always blocks**: always_ff, always_comb, always@
- **Assignments**: blocking (=) and non-blocking (<=)
- **Generate blocks** with genvar loops
- **Arrays**: 1D and multi-dimensional
- **Delays**: #delay statements
- **Continuous assignments**: assign statements

### ⚠️ Partial Support
- **Functions and tasks**: Simple functions supported
- **Interfaces**: Basic interface translation
- **Packages**: Import statements processed
- **Assertions**: Converted to comments

### ❌ Not Yet Supported
- **Classes and objects**: OOP constructs
- **SystemVerilog unions**: Union types
- **Randomization**: rand/randc
- **Coverage**: Functional coverage
- **DPI-C**: Foreign function interface

## Type Mapping Reference

| SystemVerilog Type | SystemC Type | Notes |
|-------------------|--------------|-------|
| `logic` | `sc_logic` | 4-state logic |
| `bit` | `sc_bit` | 2-state logic |
| `logic [N:0]` | `sc_lv<N+1>` | N+1 bit vector |
| `bit [N:0]` | `sc_bv<N+1>` | N+1 bit vector |
| `int` | `sc_int<32>` | 32-bit signed |
| `integer` | `sc_int<32>` | 32-bit signed |
| `reg [N:0]` | `sc_lv<N+1>` | Legacy type |
| `wire [N:0]` | `sc_signal<sc_lv<N+1>>` | Wire type |

### Port Mapping
| SystemVerilog | SystemC | Purpose |
|--------------|---------|---------|
| `input` | `sc_in<>` | Input port |
| `output` | `sc_out<>` | Output port |
| `inout` | `sc_inout<>` | Bidirectional port |

### Array Mapping
```systemverilog
// SystemVerilog
logic [7:0] mem [255:0];    // 256 x 8-bit array
logic [3:0] data [1:0][7:0]; // 2D array

// SystemC
sc_vector<sc_signal<sc_lv<8>>> mem;  // Vector of signals
// 2D arrays use nested sc_vector
```

## Error Handling and Debugging

### Common Errors

**1. File Not Found**
```
Error: Input file does not exist: design.sv
```
Solution: Check file path and spelling

**2. Parse Errors**
```
Warning: design.sv:15 - Parse diagnostic
```
Solution: Fix SystemVerilog syntax errors

**3. Unsupported Constructs**
```
Warning: Unsupported construct at line 42: class definition
```
Solution: Use supported SystemVerilog subset

### Debug Options

**Enable Debug Logging:**
```bash
sv2sc --debug -top test test.sv
```

**Verbose Output:**
```bash
sv2sc --verbose -top test test.sv
```

**Check Log File:**
```bash
cat output/sv2sc.log
```

## Integration with SystemC Projects

### CMake Integration
```cmake
# CMakeLists.txt
find_package(SystemCLanguage CONFIG REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/design.h
           ${CMAKE_CURRENT_BINARY_DIR}/design.cpp
    COMMAND sv2sc -top design -o ${CMAKE_CURRENT_BINARY_DIR} 
            ${CMAKE_CURRENT_SOURCE_DIR}/design.sv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/design.sv
)

add_executable(testbench 
    testbench.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/design.cpp
)

target_link_libraries(testbench SystemC::systemc)
```

### Makefile Integration
```makefile
DESIGN_SV = design.sv
DESIGN_SC = design.cpp design.h

$(DESIGN_SC): $(DESIGN_SV)
	sv2sc -top design $(DESIGN_SV)

testbench: testbench.cpp $(DESIGN_SC)
	g++ -o testbench testbench.cpp design.cpp -lsystemc
```

### Simulation Testbench
```cpp
// testbench.cpp
#include "design.h"

int sc_main(int argc, char* argv[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<sc_logic> reset;
    sc_signal<sc_lv<8>> output;
    
    design dut("dut");
    dut.clk(clk);
    dut.reset(reset);
    dut.output(output);
    
    // Simulation
    sc_start(100, SC_NS);
    return 0;
}
```

## Best Practices

### SystemVerilog Code Preparation
1. **Use explicit types**: Prefer `logic` over implicit types
2. **Clear port declarations**: Specify direction and width clearly
3. **Avoid mixed assignments**: Don't mix blocking/non-blocking in same block
4. **Simple generate**: Keep generate blocks straightforward

### Translation Workflow
1. **Start small**: Test with simple modules first
2. **Incremental approach**: Translate one module at a time
3. **Verify output**: Check generated SystemC syntax
4. **Test compilation**: Ensure SystemC code compiles

### Performance Optimization
1. **Use appropriate types**: Choose sc_bit vs sc_logic wisely
2. **Minimize signal updates**: Reduce unnecessary signal writes
3. **Optimize sensitivity**: Include only necessary signals
4. **Consider sc_vector**: Use for large arrays

## Troubleshooting

### Build Issues
**CMake fails to find slang:**
- Ensure internet connection for FetchContent
- Check CMake version (3.20+ required)
- Clear build directory and retry

**Compilation errors:**
- Verify C++20 compiler support
- Check all dependencies are available
- Review compiler error messages

### Translation Issues
**Missing module ports:**
- Check SystemVerilog syntax
- Ensure all ports are declared properly
- Verify module hierarchy

**Type conversion errors:**
- Review type mapping reference
- Check bit widths and signedness
- Verify array dimensions

### Simulation Issues
**SystemC compilation fails:**
- Ensure SystemC installation
- Check include paths and libraries
- Verify generated code syntax

**Runtime errors:**
- Check signal connections
- Verify clock and reset handling
- Review process sensitivity lists