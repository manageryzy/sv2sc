# PicoRV32 Project Structure Analysis

## Overview
PicoRV32 is a well-organized RISC-V CPU implementation with comprehensive build infrastructure, test suites, and example implementations. This document analyzes the project structure and key components.

## Top-Level Files

### Core Design Files
- **`picorv32.v`** (Main CPU core)
  - Contains all CPU variants: `picorv32`, `picorv32_axi`, `picorv32_wb`
  - Includes co-processor cores: `picorv32_pcpi_mul`, `picorv32_pcpi_div`
  - ~3000+ lines of well-documented Verilog
  
- **`picorv32.core`** (FuseSoC configuration)
  - Core metadata for FuseSoC tool integration
  - Defines file sets, targets, and parameters
  - Supports multiple simulation and synthesis flows

### Build System
- **`Makefile`** (Primary build system)
  - Comprehensive build targets for testing and synthesis
  - Toolchain management and automation
  - Multi-platform support (Linux, macOS, Windows)

### Testbenches
- **`testbench.v`** (Main testbench)
  - Full-featured testbench with memory model
  - Supports firmware loading and execution
  - Configurable with trace output and VCD generation
  
- **`testbench_ez.v`** (Simple testbench)
  - Self-contained test without external firmware
  - Good for quick functionality verification
  - No RISC-V toolchain required
  
- **`testbench_wb.v`** (Wishbone testbench)
  - Tests Wishbone interface variant
  - Memory controller with Wishbone protocol

- **`testbench.cc`** (Verilator C++ testbench)
  - High-performance simulation using Verilator
  - C++ test environment with better debugging

## Directory Structure Analysis

### `/firmware/` - Test Firmware Suite
**Purpose**: Comprehensive software tests for CPU validation

**Key Files**:
- `start.S` - Assembly boot code and interrupt handlers
- `firmware.c` - Main test coordinator  
- `irq.c` - Interrupt handler implementation
- `multest.c` - Multiply/divide instruction testing
- `print.c` - Console I/O functions
- `sieve.c` - Sieve of Eratosthenes algorithm test
- `hello.c` - Simple "hello world" test
- `stats.c` - Performance counter reporting
- `custom_ops.S` - PicoRV32 custom instruction macros

**Build System**:
- `Makefile` - Firmware build automation
- `makehex.py` - Binary to hex conversion
- `riscv.ld` - Linker script for memory layout
- `sections.lds` - Alternative linker configuration

### `/dhrystone/` - Performance Benchmark
**Purpose**: Industry-standard performance benchmarking

**Key Files**:
- `dhry_1.c`, `dhry_2.c` - Dhrystone benchmark source
- `dhry.h` - Benchmark header definitions
- `testbench.v` - Dedicated benchmark testbench
- `start.S` - Bootstrap code for benchmark
- `stdlib.c` - Minimal C library implementation
- `syscalls.c` - System call stubs

**Unique Features**:
- RISC-V-specific performance optimizations
- Cycle-accurate timing measurements
- Multiple test configurations (with/without look-ahead)

### `/tests/` - RISC-V ISA Compliance Tests
**Purpose**: Formal RISC-V instruction set compliance

**Contents**: Standard riscv-tests suite including:
- Individual instruction tests (rv32ui-*.S)
- Arithmetic and logic verification
- Memory access pattern testing
- Control flow instruction verification

### `/picosoc/` - Example SoC Implementation
**Purpose**: Complete System-on-Chip demonstration

**Key Components**:
- `picosoc.v` - Main SoC wrapper with memory controller
- `spimemio.v` - SPI flash memory interface  
- `simpleuart.v` - UART communication controller
- `hx8kdemo.v` - iCE40 HX8K development board implementation
- `icebreaker.v` - iCEBreaker board implementation
- `firmware.c` - Interactive SoC firmware with shell

**FPGA Support**:
- Multiple Lattice iCE40 board configurations
- Automated build flows for hardware generation
- Real-time hardware testing capability

### `/scripts/` - Tool-Specific Build Scripts
**Purpose**: Integration with commercial and open-source EDA tools

#### `/scripts/vivado/` - Xilinx Vivado Support
- `synth_*.tcl` - Synthesis scripts for different configurations  
- `*.xdc` - Timing constraints for Xilinx FPGAs
- `system.v` - Complete system wrapper for FPGA
- `table.sh` - Automated timing analysis across device families

#### `/scripts/quartus/` - Intel Quartus Support  
- `*.qsf` - Quartus project settings for different configurations
- `*.sdc` - Synopsys Design Constraints timing files
- `synth_system.tcl` - System-level synthesis automation

#### `/scripts/icestorm/` - Open Source FPGA Flow
- `example.v` - Complete iCE40 FPGA example
- `firmware.c` - Embedded software for FPGA demo
- Automated build flow using Yosys + nextpnr + icepack

#### `/scripts/yosys/` - Open Source Synthesis
- `synth_*.ys` - Yosys synthesis scripts
- Support for various technology libraries
- ASIC and FPGA synthesis flows

#### `/scripts/smtbmc/` - Formal Verification
- `*.sh` - SMT-based formal verification scripts  
- `*.v` - Property checking modules
- Protocol compliance verification (AXI, Wishbone)

#### Tool Integration Scripts
- **`/csmith/`** - Random C code generation testing
- **`/torture/`** - Random RISC-V instruction testing
- **`/cxxdemo/`** - C++ support demonstration
- **`/presyn/`** - Pre-synthesis optimization
- **`/romload/`** - ROM-based code loading

## Build Infrastructure Analysis

### Multi-Level Build System
1. **Top-level Makefile**: Coordinates overall build process
2. **Directory-specific Makefiles**: Handle local build requirements
3. **Tool-specific scripts**: Integrate with various EDA tools
4. **Python utilities**: Handle format conversions and automation

### Toolchain Support
**RISC-V Toolchains**:
- Automated download and build of RISC-V GCC
- Support for RV32I, RV32IM, RV32IC, RV32IMC variants
- Custom linker scripts for different memory layouts

**Simulation Tools**:
- Icarus Verilog (open source)
- Verilator (high performance)
- ModelSim/QuestaSim (commercial)
- VCS (commercial)

**Synthesis Tools**:  
- Yosys (open source)
- Vivado (Xilinx)
- Quartus (Intel)
- Design Compiler (Synopsys)

### Quality Assurance Infrastructure
**Testing Framework**:
- Unit tests for individual instructions
- Integration tests for complete programs
- Performance benchmarking
- Hardware-in-the-loop testing

**Verification Methods**:
- Directed testing with known-good programs
- Random testing with torture tests
- Formal verification with SMT solvers
- Cross-platform validation

**Documentation**:
- Comprehensive README with examples
- Inline code documentation
- Build instructions for all supported tools
- Performance characterization data

## Key Design Patterns

### Parameterized Architecture
The CPU core uses extensive parameterization for:
- Feature enables (multiplication, division, compressed ISA)
- Performance tuning (barrel shifter, dual-port registers)
- Memory interface configuration
- Interrupt system options

### Interface Abstraction
Multiple interface variants sharing common core:
- Native memory interface (simple valid/ready)
- AXI4-Lite interface (AMBA standard)
- Wishbone interface (OpenCores standard)
- Adapter patterns for interface conversion

### Modular Testing
Separate test environments for different aspects:
- Functional testing (instruction execution)
- Performance testing (cycle counts, benchmarks)
- Interface testing (protocol compliance)
- System testing (complete SoC operation)

### Tool Independence
Build system supports multiple tool flows:
- Open source tools (Yosys, Icarus, Verilator)  
- Commercial tools (Vivado, Quartus, VCS)
- Mixed flows (open synthesis + commercial P&R)
- Cross-platform compatibility (Linux, macOS, Windows)

## Development Workflow

### Typical Development Process
1. **Code changes** in `picorv32.v`
2. **Quick verification** with `make test_ez`
3. **Full testing** with `make test`  
4. **Performance check** with `make dhrystone`
5. **Synthesis verification** with tool-specific scripts
6. **Hardware validation** with PicoSoC examples

### Continuous Integration Ready
The project structure supports automated CI/CD:
- Comprehensive test suites with clear pass/fail criteria
- Automated tool installation and setup
- Performance regression detection
- Multi-platform validation

This well-structured project serves as an excellent reference for RISC-V CPU implementation and verification methodologies.
