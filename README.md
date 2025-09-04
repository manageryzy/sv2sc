# SystemVerilog to SystemC Translator (sv2sc)

A fully functional SystemVerilog to SystemC translator built with modern C++17 and CMake, using the slang SystemVerilog frontend. This translator supports VCS-compatible command-line arguments and comprehensive SystemVerilog language features.

## Features

### Core Capabilities
- **Complete SystemVerilog Support**: Modules, ports, signals, parameters
- **Advanced Constructs**: Generate blocks, 2D arrays, delay modeling
- **Assignment Types**: Both blocking (=) and non-blocking (<=) assignments
- **VCS Compatibility**: Full support for VCS-like command-line arguments
- **Modern C++**: Built with C++17 standards and best practices

### Supported SystemVerilog Features
- ✅ Module declarations and instantiation
- ✅ Port declarations (input, output, inout)
- ✅ Signal declarations (logic, reg, wire)
- ✅ Generate blocks with labels
- ✅ 2D and multi-dimensional arrays
- ✅ Blocking assignments (=)
- ✅ Non-blocking assignments (<=)
- ✅ Delay modeling (#delay)
- ✅ Always blocks (always_ff, always_comb)
- ✅ Parameters and parameterized modules

## Build Requirements

- **C++17** compatible compiler (GCC 7+, Clang 8+, MSVC 2017+)
- **CMake 3.20** or higher
- **Git** for dependency management

### Dependencies (automatically fetched)
- **slang**: SystemVerilog frontend and parser
- **fmt**: Modern C++ formatting library
- **CLI11**: Command-line interface library
- **spdlog**: Fast logging library
- **Catch2**: Testing framework (when BUILD_TESTS=ON)

## Building

### Quick Start
```bash
git clone <repository-url>
cd sv2sc
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options
```bash
# Enable testing
cmake -DBUILD_TESTS=ON ..

# Enable examples
cmake -DBUILD_EXAMPLES=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (default)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Usage

### Basic Command Line
```bash
# Translate a single file
sv2sc -top counter counter.sv

# Specify output directory
sv2sc -top dut -o ./output design.sv

# Multiple files
sv2sc -top system file1.sv file2.sv file3.sv
```

### VCS-Compatible Arguments
```bash
# Include directories
sv2sc -I ./include -I ./rtl -top cpu cpu.sv
sv2sc +incdir+./include +incdir+./rtl -top cpu cpu.sv

# Preprocessor defines
sv2sc -D WIDTH=8 -D SYNTHESIS -top dut design.sv
sv2sc +define+WIDTH=8 +define+SYNTHESIS -top dut design.sv

# Library paths
sv2sc -y ./lib -top system design.sv

# Timescale
sv2sc -timescale 1ns/1ps -top dut design.sv

# Debug and verbose output
sv2sc --debug --verbose -top counter counter.sv
```

### SystemC-Specific Options
```bash
# Generate testbench
sv2sc --testbench -top dut design.sv

# Specify clock and reset signals
sv2sc --clock clk_i --reset rst_n -top cpu cpu.sv

# Synthesis mode
sv2sc --synthesis -top dut design.sv
```

## Examples

### Basic Counter
```systemverilog
module counter (
    input logic clk,
    input logic reset,
    input logic enable,
    output logic [7:0] count
);
    logic [7:0] count_reg;
    
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            count_reg <= 8'b0;
        end else if (enable) begin
            count_reg <= count_reg + 1'b1;
        end
    end
    
    assign count = count_reg;
endmodule
```

**Generated SystemC:**
```cpp
#ifndef COUNTER_H
#define COUNTER_H

#include <systemc.h>

SC_MODULE(counter) {
    sc_in<sc_logic> clk;
    sc_in<sc_logic> reset;
    sc_in<sc_logic> enable;
    sc_out<sc_lv<8>> count;
    
    sc_signal<sc_lv<8>> count_reg;

    SC_CTOR(counter) {
        SC_METHOD(comb_proc);
        sensitive << clk.pos();
        sensitive << reset;
    }

private:
    void comb_proc() {
        count_reg.write(count_reg.read() + 1);
    }
};

#endif
```

### Memory Array Example
```systemverilog
module memory #(
    parameter WIDTH = 8,
    parameter DEPTH = 256
)(
    input logic clk,
    input logic write_enable,
    input logic [$clog2(DEPTH)-1:0] address,
    input logic [WIDTH-1:0] write_data,
    output logic [WIDTH-1:0] read_data
);
    logic [WIDTH-1:0] mem_array [DEPTH-1:0];
    
    always_ff @(posedge clk) begin
        if (write_enable) begin
            mem_array[address] <= write_data;
        end
    end
    
    assign read_data = mem_array[address];
endmodule
```

### Generate Block Example
```systemverilog
module generate_adder #(parameter WIDTH = 4)(
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    input logic cin,
    output logic [WIDTH-1:0] sum,
    output logic cout
);
    logic [WIDTH:0] carry;
    
    generate
        genvar i;
        for (i = 0; i < WIDTH; i++) begin : gen_fa
            full_adder fa_inst (
                .a(a[i]),
                .b(b[i]),
                .cin(carry[i]),
                .sum(sum[i]),
                .cout(carry[i+1])
            );
        end
    endgenerate
endmodule
```

## Architecture

### Project Structure
```
sv2sc/
├── src/
│   ├── core/           # AST visitor and core translation logic
│   ├── translator/     # VCS argument parsing and input handling
│   ├── codegen/       # SystemC code generation engine
│   ├── utils/         # Logging and utility functions
│   └── main.cpp       # Application entry point
├── include/           # Public headers
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── examples/     # Example translations
├── cmake/            # CMake modules and dependencies
└── docs/            # Additional documentation
```

### Core Components

1. **VCS Args Parser** (`translator/vcs_args_parser.h`)
   - Handles VCS-compatible command-line arguments
   - Supports +incdir+, +define+, -I, -D, -y, etc.

2. **AST Visitor** (`core/ast_visitor.h`)
   - Traverses slang SystemVerilog syntax trees
   - Handles all major SystemVerilog constructs

3. **SystemC Code Generator** (`codegen/systemc_generator.h`)
   - Generates SystemC header and implementation files
   - Maps SystemVerilog types to SystemC equivalents

4. **Logger** (`utils/logger.h`)
   - Provides structured logging with spdlog
   - Supports file and console output

## SystemVerilog to SystemC Mapping

| SystemVerilog | SystemC |
|---------------|---------|
| `logic` | `sc_logic` |
| `logic [7:0]` | `sc_lv<8>` |
| `bit [7:0]` | `sc_bv<8>` |
| `input` | `sc_in<>` |
| `output` | `sc_out<>` |
| `inout` | `sc_inout<>` |
| `<=` (non-blocking) | `signal.write()` |
| `=` (blocking) | Direct assignment |
| `always_ff` | `SC_METHOD` with clock sensitivity |
| `always_comb` | `SC_METHOD` with signal sensitivity |

## Advanced Features

### Generate Block Support
- Translates SystemVerilog generate blocks to equivalent SystemC constructs
- Supports parameterized generation
- Handles generate labels and scoping

### Array Support
- 2D and multi-dimensional arrays
- Memory array modeling
- Array indexing and access patterns

### Delay Modeling
- Translates `#delay` statements to SystemC `wait()`
- Supports time unit conversion
- Compatible with SystemC simulation timing

### Assignment Translation
- **Blocking assignments (=)**: Direct C++ assignment
- **Non-blocking assignments (<=)**: SystemC signal writes
- **Delayed assignments**: Combined with wait statements

## Testing

### Running Tests
```bash
# Build with tests enabled
cmake -DBUILD_TESTS=ON ..
make -j$(nproc)

# Run all tests
ctest

# Run specific test categories
ctest -R unit
ctest -R integration
```

### Example Tests
```bash
# Build examples
make counter_example
make memory_example  
make generate_example

# Test translation
./sv2sc -top counter ../tests/examples/basic_counter/counter.sv
```

## Contributing

### Code Style
- Modern C++17 features
- RAII and smart pointers
- Clear naming conventions
- Comprehensive error handling

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **slang**: Excellent SystemVerilog frontend by MikePopoloski
- **SystemC**: Accellera SystemC reference implementation
- **Modern C++**: Built with C++17 best practices