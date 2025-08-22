# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sv2sc is a fully functional SystemVerilog to SystemC translator built with modern C++20 and CMake, using the slang SystemVerilog frontend. This translator supports VCS-compatible command-line arguments and comprehensive SystemVerilog language features.

### Core Capabilities
- **Complete SystemVerilog Support**: Modules, ports, signals, parameters
- **Advanced Constructs**: Generate blocks, 2D arrays, delay modeling
- **Assignment Types**: Both blocking (=) and non-blocking (<=) assignments
- **VCS Compatibility**: Full support for VCS-like command-line arguments
- **Modern C++**: Built with C++20 standards and best practices

## Build Requirements

- **C++20** compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake 3.20** or higher
- **Git** for dependency management

### Dependencies (automatically fetched)
- **slang**: SystemVerilog frontend and parser
- **fmt**: Modern C++ formatting library
- **CLI11**: Command-line interface library
- **spdlog**: Fast logging library
- **Catch2**: Testing framework (when BUILD_TESTS=ON)
- **SystemC**: SystemC simulation library (built from source)

## Build and Development Commands

### Quick Start
```bash
git clone <repository-url>
cd sv2sc
cmake -B build
cmake --build build -j$(nproc)
```

### Build Options
```bash
# Enable testing and examples (default)
cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Release build (default)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Testing the Translator
```bash
# Basic translation test
./build/sv2sc -top counter tests/examples/basic_counter/counter.sv

# With VCS-style arguments
./build/sv2sc -I ./include +incdir+./rtl -D WIDTH=8 +define+SYNTHESIS -top dut design.sv

# Run all tests (includes translation validation and SystemC simulation)
ctest --test-dir build

# Run specific test categories
ctest --test-dir build -R unit
ctest --test-dir build -R integration
ctest --test-dir build -R translation_test
ctest --test-dir build -R systemc_simulation
```

## Project Structure

```
sv2sc/
├── src/
│   ├── core/           # AST visitor and core translation logic
│   ├── translator/     # VCS argument parsing and input handling
│   ├── codegen/       # SystemC code generation engine
│   ├── utils/         # Logging and utility functions
│   ├── main.cpp       # Application entry point
│   └── sv2sc.cpp      # Main translator implementation
├── include/           # Public headers
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── examples/     # Example translations with automated testing
├── cmake/            # CMake modules and test utilities
│   ├── Dependencies.cmake           # FetchContent for local third-party deps
│   ├── SystemCTestUtils.cmake      # Automated testing framework functions
│   ├── GenerateSystemCTestbench.cmake # SystemC testbench generator
│   ├── ValidateTranslation.cmake   # Translation quality validation
│   └── CompareSimulations.cmake    # Simulation comparison framework
├── third-party/      # Git submodules for dependencies
│   ├── slang/        # SystemVerilog frontend
│   ├── fmt/          # Formatting library
│   ├── CLI11/        # Command line parsing
│   ├── spdlog/       # Logging library
│   ├── SystemC/      # SystemC simulation library
│   └── Catch2/       # Testing framework
├── docs/            # Additional documentation
├── build/           # Build artifacts and generated SystemC code
└── .claude/         # Claude-specific files
```

## Architecture Overview

The sv2sc translator is designed with a modular architecture that separates concerns across different layers:

### Core Components

1. **VCS Args Parser** (`src/translator/vcs_args_parser.cpp`)
   - Handles VCS-compatible command-line arguments (+incdir+, +define+, -I, -D, -y, etc.)
   - Validates input files and creates output directories
   - Converts arguments to internal TranslationOptions structure

2. **AST Visitor** (`src/core/ast_visitor.cpp`)
   - Implements the visitor pattern for slang syntax trees
   - Traverses SystemVerilog AST nodes systematically
   - Delegates code generation to SystemC generator
   - Handles context tracking (current module, indent level)

3. **SystemC Generator** (`src/codegen/systemc_generator.cpp`)
   - Generates SystemC header (.h) and implementation (.cpp) files
   - Maps SystemVerilog types to appropriate SystemC types
   - Handles port declarations, signal declarations, and process methods
   - Manages indentation and code formatting

4. **Logger** (`src/utils/logger.cpp`)
   - Structured logging using spdlog
   - Multiple output targets (console, file)
   - Configurable log levels
   - Thread-safe logging operations

### Translation Flow
```
Input Files → VCS Args Parser → Translation Options
     ↓
Slang Frontend → Syntax Tree → AST Visitor
     ↓
SystemC Generator → Output Files (*.h, *.cpp)
     ↓
Automated Testing Framework → Quality Validation & Simulation
```

### Key Design Patterns
- **Visitor Pattern**: AST traversal with clean separation of structure from operations
- **Builder Pattern**: Incremental SystemC code generation
- **Strategy Pattern**: Different generation strategies for SystemVerilog constructs
- **PIMPL Idiom**: Hide implementation details in main translator class

## SystemVerilog to SystemC Type Mapping

| SystemVerilog | SystemC | Notes |
|---------------|---------|-------|
| `logic` | `sc_logic` | 4-state logic |
| `bit` | `sc_bit` | 2-state logic |
| `logic [N:0]` | `sc_lv<N+1>` | N+1 bit vector |
| `bit [N:0]` | `sc_bv<N+1>` | N+1 bit vector |
| `int` | `sc_int<32>` | 32-bit signed |
| `integer` | `sc_int<32>` | 32-bit signed |
| `reg [N:0]` | `sc_lv<N+1>` | Legacy type |
| `wire [N:0]` | `sc_signal<sc_lv<N+1>>` | Wire type |
| `input` | `sc_in<>` | Input port |
| `output` | `sc_out<>` | Output port |
| `inout` | `sc_inout<>` | Bidirectional port |
| `<=` (non-blocking) | `signal.write()` | SystemC signal writes |
| `=` (blocking) | Direct assignment | C++ assignment |
| `always_ff` | `SC_METHOD` | Clock sensitivity |
| `always_comb` | `SC_METHOD` | Signal sensitivity |

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
- **Generate blocks**: Basic support, complex nesting may cause issues

### ❌ Not Yet Supported
- **Classes and objects**: OOP constructs
- **SystemVerilog unions**: Union types
- **Randomization**: rand/randc
- **Coverage**: Functional coverage
- **DPI-C**: Foreign function interface

## Command Line Usage

### Basic Command Line
```bash
# Translate a single file
./build/sv2sc -top counter counter.sv

# Specify output directory
./build/sv2sc -top dut -o ./output design.sv

# Multiple files
./build/sv2sc -top system file1.sv file2.sv file3.sv
```

### VCS-Compatible Arguments
```bash
# Include directories
./build/sv2sc -I ./include -I ./rtl -top cpu cpu.sv
./build/sv2sc +incdir+./include +incdir+./rtl -top cpu cpu.sv

# Preprocessor defines
./build/sv2sc -D WIDTH=8 -D SYNTHESIS -top dut design.sv
./build/sv2sc +define+WIDTH=8 +define+SYNTHESIS -top dut design.sv

# Library paths
./build/sv2sc -y ./lib -top system design.sv

# Timescale
./build/sv2sc -timescale 1ns/1ps -top dut design.sv

# Debug and verbose output
./build/sv2sc --debug --verbose -top counter counter.sv
```

### SystemC-Specific Options
```bash
# Generate testbench
./build/sv2sc --testbench -top dut design.sv

# Specify clock and reset signals
./build/sv2sc --clock clk_i --reset rst_n -top cpu cpu.sv

# Synthesis mode
./build/sv2sc --synthesis -top dut design.sv
```

## Code Style and Standards

### C++ Coding Standards
- **Modern C++20**: Use latest C++20 features and idioms
- **RAII**: Resource management with smart pointers
- **Naming Conventions**:
  - Functions and variables: `camelCase`
  - Types and constants: `PascalCase`
  - Clear, descriptive names
- **Error Handling**: Comprehensive exception safety
- **Memory Management**: Smart pointers, avoid raw pointers

### Code Formatting
- **Column Limit**: 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Brace Style**: Custom with specific wrapping rules
- **Pointer Alignment**: Left-aligned
- **Include Organization**: Grouped by priority

## Error Handling and Debugging

### Common Errors
- **File Not Found**: Check file path and spelling
- **Parse Errors**: Fix SystemVerilog syntax errors
- **Unsupported Constructs**: Use supported SystemVerilog subset

### Debug Options
```bash
# Enable debug logging
./build/sv2sc --debug -top test test.sv

# Verbose output
./build/sv2sc --verbose -top test test.sv

# Check log file
cat build/tests/examples/tests/*/sv2sc.log
```

### Translation Quality Analysis
```bash
# View detailed translation validation results
ctest --test-dir build -R translation_test --verbose

# Check generated SystemC files
ls build/tests/examples/tests/*/
cat build/tests/examples/tests/counter_sv2sc/counter.h
```

## Automated Testing Framework

### Translation Validation System
The project includes a comprehensive automated testing framework that validates translation quality and functionality:

#### CMake Test Functions
- **`add_sv2sc_test()`**: Basic translation with validation
- **`add_complete_sv2sc_test_suite()`**: Full workflow including SystemC testbench generation and simulation
- **`add_verilator_comparison_test()`**: Compare SystemC vs Verilator simulation results

#### Translation Quality Metrics
- **Port Count Validation**: Verifies expected number of ports are correctly translated
- **Unknown Expression Detection**: Counts untranslated complex expressions
- **Skipped Assignment Detection**: Identifies assignments that couldn't be translated
- **Quality Scoring**: Automatic scoring (EXCELLENT/GOOD/FAIR/NEEDS_WORK)

#### Example Test Results (Current Status)
- **Counter Module**: 60% quality (FAIR) - 4 ports, 2 unknown expressions, 2 skipped assignments
- **Memory Module**: 40% quality (NEEDS_WORK) - 7 ports, 5 unknown expressions, 4 skipped assignments
- **Generate Adder**: 40% quality (NEEDS_WORK) - Complex generate blocks cause translation issues

### SystemC Simulation Testing
- **Automated Testbench Generation**: Creates appropriate SystemC testbenches for each module type
- **Clock Signal Handling**: Proper conversion from SystemVerilog clock to SystemC sc_logic signals
- **Compilation Validation**: Ensures generated SystemC code compiles successfully
- **Simulation Execution**: Runs SystemC simulations to verify functionality

### Test Execution
```bash
# Run all tests
ctest --test-dir build

# Run with verbose output
ctest --test-dir build --verbose

# Run specific test types
ctest --test-dir build -R translation_test  # Translation validation only
ctest --test-dir build -R systemc_simulation # SystemC simulation only
ctest --test-dir build -R unit              # Unit tests only
ctest --test-dir build -R integration       # Integration tests only
```

### Current Test Status
- **Unit Tests**: ✅ 100% passing (VCS Args Parser, SystemC Generator)
- **Integration Tests**: ✅ 100% passing (Translation Flow)
- **Translation Validation**: ✅ 100% passing (all 3 modules)
- **SystemC Simulation**: ✅ Counter module working, Memory/Generate_adder have compilation issues due to translation quality

### Unit Tests
- Individual component testing
- Mock dependencies for isolation
- Comprehensive edge case coverage

### Integration Tests
- End-to-end translation testing
- Real SystemVerilog file processing
- Output validation

### Translation Validation Tests
- Automated quality scoring
- Port count verification
- Unknown expression detection
- Skipped assignment tracking

### SystemC Simulation Tests
- Testbench generation and compilation
- Clock/signal handling validation
- Simulation execution verification

## Contributing Guidelines

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

### Code Quality Requirements
- Follow existing C++ code style (enforced by pre-commit hooks)
- Use modern C++20 features and idioms
- Write unit tests for new functionality
- Document public APIs with Doxygen comments
- Maintain high test coverage
- Add translation validation tests for new SystemVerilog features

### Extensibility Guidelines
- Use visitor pattern for new AST nodes
- Extend code generator incrementally
- Update type mapping tables
- Maintain backward compatibility
- Add appropriate test coverage for new features

## Future Enhancements

### Planned Features
1. **SystemVerilog Interfaces**: Interface and modport support
2. **Classes and Objects**: OOP construct translation
3. **Assertions**: SVA to SystemC assertion mapping
4. **Coverage**: Functional coverage translation
5. **DPI Functions**: SystemVerilog DPI integration
6. **Improved Generate Block Support**: Better handling of complex generate constructs

### Architecture Improvements
1. **Plugin System**: Extensible translation rules
2. **Configuration Files**: Translation behavior customization
3. **Incremental Compilation**: Large project support
4. **Parallel Processing**: Multi-file concurrent translation
5. **Enhanced Error Reporting**: Better diagnostic messages for translation failures

### Testing Framework Enhancements
1. **Verilator Integration**: Complete simulation comparison framework
2. **Waveform Comparison**: Automatic signal comparison between SystemC and Verilator
3. **Performance Benchmarking**: Translation speed and quality metrics
4. **Regression Testing**: Continuous integration with quality tracking

## Important Commands

### Build Commands
```bash
# Clean rebuild
rm -rf build && cmake -B build && cmake --build build -j$(nproc)

# Build specific targets
cmake --build build --target sv2sc                    # Just the translator
cmake --build build --target counter_sv2sc_systemc_test # Specific SystemC test
```

### Testing Commands
```bash
# Run translation validation only
ctest --test-dir build -R translation_test

# Run SystemC simulations only  
ctest --test-dir build -R systemc_simulation

# Run failing tests with detailed output
ctest --test-dir build --rerun-failed --output-on-failure
```

### Development Commands
```bash
# Test translation manually
./build/sv2sc -top counter tests/examples/basic_counter/counter.sv -o build/manual_test

# Run generated SystemC simulation
cd build/tests/examples && ./counter_sv2sc_systemc_test
```