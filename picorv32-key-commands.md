# PicoRV32 Key Build and Verification Commands

## Essential Commands Reference

### Quick Test Commands
```bash
# Immediate testing (no toolchain required)
make test_ez                    # Simple testbench with built-in patterns
make test_ez VCD=1             # Generate VCD waveform file

# Full testing (requires RISC-V toolchain)  
make test                      # Comprehensive firmware test
make test VCD=1                # With waveform generation
make test TRACE=1              # With instruction trace
```

### Toolchain Setup
```bash
# Build RISC-V toolchain from source
make download-tools            # Download toolchain sources
make -j$(nproc) build-tools    # Build all variants (RV32I/IM/IC/IMC)

# Individual toolchain builds
make build-riscv32i-tools      # RV32I only
make build-riscv32ic-tools     # RV32I + Compressed
make build-riscv32im-tools     # RV32I + Multiply/Divide
make build-riscv32imc-tools    # RV32I + Multiply/Divide + Compressed
```

### Core Build Variants
```bash
# Different testbenches
make test                      # Standard testbench (testbench.v)
make test_ez                   # Simple testbench (testbench_ez.v)  
make test_wb                   # Wishbone testbench (testbench_wb.v)
make test_sp                   # Single-port registers
make test_axi                  # AXI interface test
make test_synth                # Post-synthesis simulation

# Configuration variants
make ENABLE_MUL=1 test         # With hardware multiplier
make ENABLE_DIV=1 test         # With hardware divider
make COMPRESSED_ISA=1 test     # With compressed instructions
make ENABLE_IRQ=1 test         # With interrupt support
make BARREL_SHIFTER=1 test     # With barrel shifter
```

### Firmware Commands
```bash
cd firmware

# Build main test firmware
make                           # Build firmware.hex
make firmware.hex              # Explicit hex format
make firmware.bin              # Binary format
make firmware.elf              # ELF executable

# Individual test modules
make hello.hex                 # Hello world
make sieve.hex                 # Sieve of Eratosthenes  
make multest.hex               # Multiply/divide test
make stats.hex                 # Performance statistics
```

### Benchmark Commands
```bash
cd dhrystone

# Build and run Dhrystone
make                           # Build dhry.hex
make sim                       # Run simulation
make sim_nola                  # Without look-ahead interface
make timing                    # Timing analysis with performance counters

# View results
make sim | grep DMIPS          # Extract performance numbers
```

### Hardware Synthesis

#### Xilinx Vivado
```bash
cd scripts/vivado

# Synthesis targets
make synth_area                # Area optimization
make synth_speed               # Speed optimization  
make synth_system              # Complete system

# Specific configurations
make synth_area_small          # Minimal features
make synth_area_regular        # Standard configuration
make synth_area_large          # All features enabled

# Timing analysis
make table.txt                 # Generate timing table
```

#### Intel Quartus
```bash
cd scripts/quartus

# Similar targets as Vivado
make synth_area_small
make synth_area_regular  
make synth_area_large
make synth_speed
make synth_system
```

#### Lattice iCE40 (Open Source)
```bash
cd scripts/icestorm

# Build for iCE40 FPGA
make example.bin               # Generate bitstream
make example.asc               # ASCII bitstream
make example.blif              # Logic synthesis
make prog                      # Program FPGA
```

#### Yosys (Generic Synthesis)
```bash
cd scripts/yosys
yosys -s synth_sim.ys          # Simulation netlist
yosys -s synth_gates.ys        # Gate-level netlist
make synth_osu018              # OSU 0.18um PDK
```

### SoC Examples
```bash
cd picosoc

# iCE40 HX8K development board
make hx8kdemo.bin              # Build bitstream
make hx8kdemo.rpt              # Timing report
iceprog hx8kdemo.bin           # Program FPGA

# iCEBreaker board
make icebreaker.bin
iceprog icebreaker.bin

# Terminal connection
minicom -D /dev/ttyUSB1 -b 115200

# SoC firmware
make firmware.hex              # SoC test firmware
make firmware.dis              # Disassembly
```

### Formal Verification
```bash
cd scripts/smtbmc

# Property verification
./axicheck.sh                 # AXI protocol compliance
./tracecmp.sh                 # Trace comparison
./mulcmp.sh                   # Multiply instruction verification
./notrap_validop.sh           # Valid operation checking
```

### Advanced Testing
```bash
# Torture testing (requires riscv-torture)
cd scripts/torture
./test.sh                     # Random instruction sequences

# Compliance testing
make test_rvtst               # RISC-V ISA compliance tests

# Coverage analysis
make coverage                 # Functional coverage
make coverage_report          # Generate coverage report
```

### Debug and Analysis
```bash
# Waveform debugging
make test VCD=1               # Generate testbench.vcd
gtkwave testbench.vcd         # View waveforms

# Trace analysis  
make test TRACE=1             # Generate instruction trace
python3 showtrace.py testbench.trace firmware/firmware.elf

# Performance analysis
make test | grep "Cycle count"     # Extract cycle counts
make test | grep "CPI"             # Cycles per instruction
```

### Cleanup Commands
```bash
# Clean build artifacts
make clean                    # Remove simulation files
make clean_all               # Remove all generated files
make distclean               # Complete cleanup including tools

# Per-directory cleanup
cd firmware && make clean
cd dhrystone && make clean  
cd picosoc && make clean
```

### Utility Commands
```bash
# Lint checking
make lint                     # Verilator lint check
verilator --lint-only picorv32.v

# File format conversion
python3 firmware/makehex.py firmware.bin 1024 > firmware.hex
python3 scripts/romload/hex8tohex32.py < input.hex > output.hex

# Statistics and analysis
make area                     # Resource utilization
make timing                   # Timing summary
make power                    # Power analysis (if supported)
```

### Configuration Examples
```bash
# Minimal configuration
make ENABLE_COUNTERS=0 ENABLE_REGS_16_31=0 TWO_STAGE_SHIFT=0 test

# High-performance configuration  
make ENABLE_FAST_MUL=1 BARREL_SHIFTER=1 TWO_CYCLE_ALU=1 test

# Interrupt testing
make ENABLE_IRQ=1 ENABLE_IRQ_QREGS=1 test

# Compressed ISA testing
make COMPRESSED_ISA=1 test

# Custom memory layout
make PROGADDR_RESET=0x10000 STACKADDR=0x20000 test
```

### Environment Variables
```bash
# Toolchain configuration
export TOOLCHAIN_PREFIX=riscv32-unknown-elf-
export PATH=/opt/riscv32i/bin:$PATH

# Simulation configuration  
export TIMEOUT=10000          # Simulation timeout
export VCD=1                  # Always generate VCD
export TRACE=1               # Always generate trace

# Tool paths
export YOSYS_PATH=/usr/local/bin/yosys
export VIVADO_PATH=/opt/Xilinx/Vivado/2021.1
```

### Common Make Targets Summary
```bash
# Testing
test test_ez test_wb test_sp test_axi test_synth test_rvtst

# Building  
build-tools download-tools build-riscv32i-tools

# Synthesis
synth_area synth_speed synth_system  

# Analysis
lint coverage timing area power

# Cleanup
clean clean_all distclean

# Utilities
help                          # Show available targets
```

This reference provides the essential commands for building, testing, and verifying PicoRV32 across different tools and configurations.