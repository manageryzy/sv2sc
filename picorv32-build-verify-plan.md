# PicoRV32 Build and Verification Plan

## Overview

PicoRV32 is a size-optimized RISC-V CPU core that implements the RV32IMC instruction set. This document provides a comprehensive build and verification plan covering synthesis, simulation, testing, and deployment strategies.

## Project Structure

```
picorv32/
├── picorv32.v              # Main CPU core (Verilog)
├── picorv32.core           # FuseSoC core description
├── Makefile                # Main build system
├── testbench.v             # Primary testbench
├── testbench_ez.v          # Simple testbench (no firmware)
├── testbench_wb.v          # Wishbone testbench
├── firmware/               # Test firmware and examples
├── dhrystone/              # Dhrystone benchmark
├── tests/                  # RISC-V compliance tests
├── picosoc/                # Example SoC implementation
└── scripts/                # Tool-specific build scripts
```

## Build Requirements

### Core Dependencies
- **RISC-V Toolchain**: `riscv32-unknown-elf-gcc` or compatible
- **Simulation**: Icarus Verilog, Verilator, or ModelSim
- **Synthesis**: Yosys, Vivado, Quartus, or other synthesis tools
- **Python**: Version 3.x for build scripts
- **Make**: GNU Make for build automation

### Optional Tools
- **FuseSoC**: For core management and automation
- **GTKWave**: For waveform viewing
- **OpenOCD**: For hardware debugging

## Build Targets and Configurations

### 1. Core Variants

#### Standard Configurations
```makefile
# Small configuration (minimal features)
ENABLE_COUNTERS = 0
ENABLE_REGS_16_31 = 0
TWO_STAGE_SHIFT = 0
BARREL_SHIFTER = 0

# Regular configuration (default)
ENABLE_COUNTERS = 1
ENABLE_REGS_DUALPORT = 1
TWO_STAGE_SHIFT = 1

# Large configuration (all features)
ENABLE_PCPI = 1
ENABLE_MUL = 1
ENABLE_DIV = 1
ENABLE_IRQ = 1
COMPRESSED_ISA = 1
BARREL_SHIFTER = 1
```

#### Interface Variants
- **`picorv32`**: Native memory interface
- **`picorv32_axi`**: AXI4-Lite interface  
- **`picorv32_wb`**: Wishbone interface
- **`picorv32_axi_adapter`**: Native to AXI4 adapter

### 2. FPGA Targets

#### Xilinx (Vivado)
```bash
# Area optimization
cd scripts/vivado
make synth_area_small    # Minimal LUT usage
make synth_area_regular  # Standard configuration
make synth_area_large    # Full-featured

# Timing optimization
make synth_speed         # Maximum frequency
make synth_system        # Complete system
```

#### Intel (Quartus)
```bash
cd scripts/quartus
make synth_area_small
make synth_area_regular
make synth_area_large
make synth_speed
```

#### Lattice iCE40 (Yosys + nextpnr)
```bash
cd scripts/icestorm
make firmware.hex
make example.bin        # Bitstream generation
make example_tb.vcd     # Testbench simulation
```

### 3. ASIC Targets

#### Open Source Flow (Yosys)
```bash
cd scripts/yosys
yosys -s synth_sim.ys   # Simulation netlist
yosys -s synth_gates.ys # Gate-level netlist
```

#### Commercial Tools
- **Design Compiler**: Via Yosys front-end
- **Genus**: Standard synthesis flow
- **Custom PDKs**: Configurable through scripts

## Verification Strategy

### 1. Functional Verification

#### Unit Tests (RISC-V ISA Tests)
```bash
# Build and run instruction-level tests
make test_rvtst

# Individual instruction tests
make test_lui test_auipc test_jal test_jalr
make test_beq test_bne test_blt test_bge
make test_lb test_lh test_lw test_sb test_sh test_sw
make test_addi test_slti test_xori test_ori test_andi
make test_add test_sub test_sll test_slt test_xor
```

#### Integration Tests
```bash
# Standard testbench with firmware
make test

# Simple testbench (no external firmware)
make test_ez

# Wishbone interface test
make test_wb

# Compressed instruction test
make COMPRESSED_ISA=1 test
```

#### Benchmark Tests
```bash
# Dhrystone performance benchmark
cd dhrystone
make dhry.hex
make sim                # Run simulation
make sim_nola           # Without look-ahead interface

# Custom benchmarks
cd firmware
make test               # Comprehensive test suite
```

### 2. Performance Verification

#### Timing Analysis
```bash
# Generate timing reports
make timing_xilinx      # Xilinx timing analysis
make timing_intel       # Intel timing analysis
make timing_lattice     # Lattice timing analysis

# Performance benchmarking
make benchmark          # Run Dhrystone
make profile           # Instruction profiling
```

#### Resource Utilization
```bash
# Area reports for different configurations
make area_small         # Minimal configuration
make area_regular       # Standard configuration  
make area_large         # Full-featured configuration

# Compare resource usage
make area_compare
```

### 3. Formal Verification

#### Property Checking
```bash
cd scripts/smtbmc
./axicheck.sh          # AXI protocol compliance
./tracecmp.sh          # Trace comparison
./notrap_validop.sh    # Valid operation verification
```

#### Equivalence Checking
```bash
# Compare different implementations
make equiv_check_axi    # AXI vs Native interface
make equiv_check_wb     # Wishbone vs Native interface
```

## Build Process

### 1. Environment Setup

#### Toolchain Installation
```bash
# Install RISC-V toolchain
make download-tools
make -j$(nproc) build-riscv32i-tools
make -j$(nproc) build-riscv32ic-tools
make -j$(nproc) build-riscv32im-tools
make -j$(nproc) build-riscv32imc-tools

# Alternative: Use distribution packages
sudo apt-get install gcc-riscv64-unknown-elf
export TOOLCHAIN_PREFIX=riscv64-unknown-elf-
```

#### Simulation Setup
```bash
# Install Icarus Verilog
sudo apt-get install iverilog gtkwave

# Or build from source for latest features
git clone https://github.com/steveicarus/iverilog.git
cd iverilog && make && sudo make install
```

### 2. Core Build Steps

#### Basic Simulation
```bash
# 1. Build test firmware
cd firmware
make firmware.hex

# 2. Run basic simulation
cd ..
make test

# 3. View results
gtkwave testbench.vcd
```

#### Custom Configuration
```bash
# Create custom build configuration
cat > config.mk << EOF
ENABLE_COUNTERS = 1
ENABLE_MUL = 1
ENABLE_DIV = 1
ENABLE_COMPRESSED_ISA = 1
PROGADDR_RESET = 0x00100000
STACKADDR = 0x00200000
EOF

# Build with custom configuration
make -f config.mk test
```

#### Multi-Target Build
```bash
# Build for multiple FPGA targets
make xilinx_build
make intel_build
make lattice_build

# Generate reports
make reports
```

### 3. SoC Integration

#### PicoSoC Example
```bash
cd picosoc
make hx8kdemo.bin       # iCE40 HX8K demo
make icebreaker.bin     # iCEBreaker board
make firmware.hex       # SoC firmware
```

#### Custom SoC
```verilog
// Instantiate PicoRV32 in custom design
picorv32 #(
    .ENABLE_COUNTERS(1),
    .ENABLE_MUL(1),
    .PROGADDR_RESET(32'h10000000),
    .STACKADDR(32'h20000000)
) cpu (
    .clk(clk),
    .resetn(resetn),
    .mem_valid(mem_valid),
    .mem_ready(mem_ready),
    .mem_addr(mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_rdata(mem_rdata)
);
```

## Test Execution Plan

### 1. Regression Test Suite

#### Daily Tests
```bash
#!/bin/bash
# daily_regression.sh

# Basic functionality
make test_ez
make test

# ISA compliance
make test_rvtst

# Interface tests
make test_wb
make ENABLE_PCPI=1 test

# Performance tests
cd dhrystone && make sim
cd ../firmware && make test
```

#### Weekly Tests
```bash
#!/bin/bash
# weekly_regression.sh

# Extended configuration matrix
for config in small regular large; do
    make clean
    make CONFIG=$config test
done

# Multi-tool verification
make xilinx_verify
make intel_verify
make yosys_verify

# Formal verification
cd scripts/smtbmc && ./axicheck.sh
```

### 2. Hardware Validation

#### FPGA Validation
```bash
# Build FPGA bitstream
cd picosoc
make icebreaker.bin

# Program and test hardware
iceprog icebreaker.bin
minicom -D /dev/ttyUSB1

# Run hardware tests
echo "sieve" > /dev/ttyUSB1
echo "multest" > /dev/ttyUSB1
```

#### Silicon Validation
```bash
# Generate ASIC test patterns
make asic_test_vectors

# Post-silicon validation
make silicon_test_suite
```

## Quality Assurance

### 1. Code Quality

#### Lint Checking
```bash
# Verilator lint
make lint

# Custom lint rules
verilator --lint-only -Wall picorv32.v
```

#### Coverage Analysis
```bash
# Functional coverage
make coverage_functional

# Code coverage
make coverage_code

# Generate coverage report
make coverage_report
```

### 2. Documentation

#### Design Documentation
- **Architecture specification**
- **Interface documentation** 
- **Timing constraints**
- **Power analysis**

#### Verification Documentation
- **Test plan**
- **Coverage reports**
- **Performance benchmarks**
- **Silicon validation results**

## Continuous Integration

### 1. Automated Testing

#### GitHub Actions / CI Pipeline
```yaml
# .github/workflows/ci.yml
name: PicoRV32 CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install iverilog
    - name: Run tests
      run: make test
    - name: Run regression
      run: make regression
```

#### Nightly Builds
```bash
# nightly_build.sh
#!/bin/bash

# Full regression suite
make clean_all
make regression_full

# Performance tracking
make benchmark_tracking

# Report generation
make nightly_report
```

### 2. Release Process

#### Version Control
- **Semantic versioning**: vX.Y.Z
- **Release branches**: release/vX.Y
- **Tag management**: Automated tagging

#### Release Validation
```bash
# Pre-release validation
make release_test_suite
make performance_validation
make documentation_check

# Release packaging
make release_package
make release_notes
```

## Troubleshooting Guide

### 1. Common Build Issues

#### Toolchain Problems
```bash
# Check toolchain installation
which riscv32-unknown-elf-gcc
riscv32-unknown-elf-gcc --version

# Fix PATH issues
export PATH=/opt/riscv32i/bin:$PATH
```

#### Simulation Issues
```bash
# Debug simulation failures
make test DEBUG=1
gtkwave testbench.vcd

# Check firmware loading
hexdump -C firmware/firmware.hex | head
```

### 2. Performance Issues

#### Timing Closure
- **Pipeline balancing**: Enable TWO_CYCLE_ALU
- **Register optimization**: Use ENABLE_REGS_DUALPORT
- **Memory interface**: Optimize mem_ready timing

#### Resource Usage
- **LUT reduction**: Disable unused features
- **RAM optimization**: Use LATCHED_MEM_RDATA
- **Multiplier sharing**: Configure ENABLE_FAST_MUL

## Best Practices

### 1. Development Workflow
1. **Feature branches**: Isolate changes
2. **Regression testing**: Run before commits
3. **Performance monitoring**: Track metrics
4. **Documentation updates**: Keep current

### 2. Verification Methodology
1. **Directed testing**: Target specific features
2. **Random testing**: Use torture tests  
3. **Formal methods**: Property verification
4. **Hardware validation**: FPGA prototyping

### 3. Configuration Management
1. **Parameter consistency**: Validate configurations
2. **Interface compatibility**: Check connections
3. **Timing constraints**: Verify all paths
4. **Resource budgets**: Monitor utilization

This build and verification plan provides comprehensive coverage for PicoRV32 development, testing, and deployment across multiple platforms and use cases.