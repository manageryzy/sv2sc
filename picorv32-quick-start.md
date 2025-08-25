# PicoRV32 Quick Start Guide

## Immediate Testing (No Toolchain Required)

### 1. Simple Testbench
```bash
# Run basic functionality test without firmware
make test_ez

# This runs testbench_ez.v which has built-in test patterns
# No RISC-V toolchain required
```

### 2. View Simulation Results
```bash
# Generate waveforms
make test_ez VCD=1

# View with GTKWave
gtkwave testbench.vcd
```

## Full Testing (Requires RISC-V Toolchain)

### 1. Install RISC-V Toolchain
```bash
# Option 1: Build from source (recommended)
make download-tools
make -j$(nproc) build-riscv32i-tools

# Option 2: Use distribution packages  
sudo apt-get install gcc-riscv64-unknown-elf
export TOOLCHAIN_PREFIX=riscv64-unknown-elf-
```

### 2. Build and Test Firmware
```bash
# Build test firmware
cd firmware
make

# Run comprehensive tests
cd ..
make test
```

### 3. Run Benchmarks
```bash
# Dhrystone benchmark
cd dhrystone
make
make sim

# View performance results
# Output shows DMIPS/MHz performance
```

## Key Build Targets

### Core Testing
```bash
make test          # Full test with firmware
make test_ez       # Simple test (no firmware)
make test_wb       # Wishbone interface test
make test_sp       # Single-port register file
make test_synth    # Synthesis simulation
```

### Firmware Builds
```bash
cd firmware
make firmware.hex    # Main test firmware
make firmware.bin    # Binary format
make stats          # Build statistics
```

### Dhrystone Benchmark
```bash
cd dhrystone  
make dhry.hex       # Build benchmark
make sim            # Run simulation
make sim_nola       # Without look-ahead interface
make timing         # Timing analysis
```

### Tool-Specific Builds

#### Xilinx Vivado
```bash
cd scripts/vivado
make synth_area     # Area optimization
make synth_speed    # Speed optimization
make table.txt      # Timing table generation
```

#### Intel Quartus
```bash
cd scripts/quartus
make synth_area
make synth_speed
make synth_system
```

#### Lattice iCE40
```bash
cd scripts/icestorm
make example.bin    # Generate bitstream
make prog           # Program FPGA
```

## Configuration Options

### Core Parameters (Verilog)
```verilog
// Size optimized
picorv32 #(
    .ENABLE_COUNTERS(0),
    .ENABLE_REGS_16_31(0), 
    .TWO_STAGE_SHIFT(0),
    .BARREL_SHIFTER(0)
) cpu_small (...);

// Performance optimized
picorv32 #(
    .ENABLE_COUNTERS(1),
    .ENABLE_REGS_DUALPORT(1),
    .TWO_STAGE_SHIFT(1),
    .BARREL_SHIFTER(1),
    .ENABLE_FAST_MUL(1),
    .ENABLE_DIV(1),
    .TWO_CYCLE_ALU(1)
) cpu_fast (...);
```

### Make Variables
```bash
# Core configuration
make ENABLE_MUL=1 ENABLE_DIV=1 test
make COMPRESSED_ISA=1 test
make ENABLE_IRQ=1 test

# Toolchain configuration
make TOOLCHAIN_PREFIX=riscv32-unknown-elf- test

# Simulation options
make TRACE=1 test        # Enable trace output
make VCD=1 test          # Generate VCD file
make VERBOSE=1 test      # Verbose output
```

## Hardware Testing

### PicoSoC Examples
```bash
cd picosoc

# iCE40 HX8K demo
make hx8kdemo.bin
iceprog hx8kdemo.bin

# iCEBreaker board
make icebreaker.bin
iceprog icebreaker.bin

# Connect terminal
minicom -D /dev/ttyUSB1
# Then type commands like: sieve, multest, benchmark
```

## Common Issues and Solutions

### Build Failures
```bash
# Missing toolchain
export PATH=/opt/riscv32i/bin:$PATH
which riscv32-unknown-elf-gcc

# Icarus version too old
# Upgrade to latest master branch
git clone https://github.com/steveicarus/iverilog.git
```

### Simulation Issues
```bash
# Increase timeout for slow simulations
make TIMEOUT=10000 test

# Debug mode
make DEBUG=1 test
```

### Performance Issues
```bash
# Use compressed ISA
make COMPRESSED_ISA=1 test

# Enable optimizations
make ENABLE_FAST_MUL=1 BARREL_SHIFTER=1 test
```

## File Structure Overview

```
picorv32/
├── picorv32.v              # Main CPU core
├── Makefile                # Primary build file
├── testbench.v             # Main testbench
├── testbench_ez.v          # Simple testbench
│
├── firmware/               # Test programs
│   ├── Makefile           # Firmware build
│   ├── firmware.c         # Main test program
│   ├── start.S            # Boot code
│   └── *.c                # Test modules
│
├── dhrystone/             # Performance benchmark  
│   ├── Makefile          # Benchmark build
│   └── dhry_*.c          # Dhrystone source
│
├── tests/                 # RISC-V compliance tests
├── picosoc/              # Example SoC
│   └── *.v               # SoC components
│
└── scripts/              # Tool-specific builds
    ├── vivado/          # Xilinx tools
    ├── quartus/         # Intel tools  
    └── icestorm/        # Open source tools
```

## Next Steps

1. **Start Simple**: Run `make test_ez` first
2. **Install Toolchain**: Build RISC-V tools
3. **Run Full Tests**: Execute `make test`
4. **Try Benchmarks**: Run Dhrystone performance test
5. **Hardware Demo**: Build and run PicoSoC
6. **Customize**: Modify parameters for your needs

## Resources

- **Main Repository**: https://github.com/YosysHQ/picorv32
- **RISC-V Spec**: https://riscv.org/specifications/
- **PicoRV32 Paper**: See repository documentation
- **Community**: RISC-V forums and mailing lists