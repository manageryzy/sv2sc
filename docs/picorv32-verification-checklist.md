# PicoRV32 Verification Checklist

## Pre-Verification Setup ✓

### Environment Setup
- [ ] Install RISC-V toolchain (`make build-tools`)
- [ ] Verify toolchain installation (`which riscv32-unknown-elf-gcc`)
- [ ] Install simulation tools (Icarus Verilog, Verilator)
- [ ] Install waveform viewer (GTKWave)
- [ ] Clone repository and check file integrity

### Initial Sanity Check
```bash
# Quick functionality test
make test_ez                    # Should pass without errors
make test_ez VCD=1             # Generates testbench.vcd
gtkwave testbench.vcd          # Verify basic waveforms
```

## Core Functional Verification ✓

### 1. Basic Instruction Set (RV32I)
```bash
# Integer arithmetic and logic
make test_rvtst                # RISC-V compliance tests
```

**Verify these instruction categories:**
- [ ] **LUI/AUIPC**: Upper immediate instructions
- [ ] **JAL/JALR**: Jump and link instructions  
- [ ] **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
- [ ] **Loads**: LB, LH, LW, LBU, LHU
- [ ] **Stores**: SB, SH, SW
- [ ] **Arithmetic**: ADDI, SLTI, SLTIU, XORI, ORI, ANDI
- [ ] **Shifts**: SLLI, SRLI, SRAI
- [ ] **Register ops**: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND

### 2. Extended Instructions (RV32IM)
```bash
# Multiply/Divide extension
make ENABLE_MUL=1 ENABLE_DIV=1 test
cd firmware && make multest.hex && cd .. && make test
```

**Verify these instructions:**
- [ ] **MUL**: 32-bit multiply
- [ ] **MULH**: High multiply (signed × signed)
- [ ] **MULHSU**: High multiply (signed × unsigned)
- [ ] **MULHU**: High multiply (unsigned × unsigned)
- [ ] **DIV**: Division (signed)
- [ ] **DIVU**: Division (unsigned)
- [ ] **REM**: Remainder (signed)
- [ ] **REMU**: Remainder (unsigned)

### 3. Compressed Instructions (RV32IC)
```bash
# 16-bit compressed instruction set
make COMPRESSED_ISA=1 test
```

**Verify compressed instruction mapping:**
- [ ] **C.ADDI4SPN** → ADDI (stack pointer)
- [ ] **C.LW/C.SW** → LW/SW (register-relative)
- [ ] **C.ADDI/C.NOP** → ADDI/NOP
- [ ] **C.JAL** → JAL (20-bit immediate)
- [ ] **C.LI** → ADDI (load immediate)
- [ ] **C.LUI** → LUI (load upper immediate)
- [ ] **C.SRLI/C.SRAI** → SRLI/SRAI (immediate shifts)
- [ ] **C.ANDI** → ANDI (bitwise and immediate)
- [ ] **C.SUB/C.XOR/C.OR/C.AND** → Register operations
- [ ] **C.J** → JAL (11-bit immediate)
- [ ] **C.BEQZ/C.BNEZ** → BEQ/BNE (zero comparison)

## Interface Verification ✓

### 1. Native Memory Interface
```bash
make test                      # Standard memory interface
```

**Verify interface signals:**
- [ ] **mem_valid**: Request valid signal
- [ ] **mem_ready**: Memory ready response
- [ ] **mem_addr**: Address bus (32-bit aligned)
- [ ] **mem_wdata**: Write data (32-bit)
- [ ] **mem_wstrb**: Write strobes (byte enables)
- [ ] **mem_rdata**: Read data (32-bit)
- [ ] **mem_instr**: Instruction fetch indicator

**Verify memory access patterns:**
- [ ] **Word access**: 32-bit aligned reads/writes
- [ ] **Halfword access**: 16-bit aligned with proper strobes
- [ ] **Byte access**: 8-bit with single strobe
- [ ] **Instruction fetches**: mem_instr assertion
- [ ] **Data accesses**: mem_instr deassertion

### 2. Look-Ahead Interface
```bash
make test                      # Uses look-ahead by default
make test_nola                 # Without look-ahead (dhrystone)
```

**Verify look-ahead signals:**
- [ ] **mem_la_read**: Next cycle read prediction
- [ ] **mem_la_write**: Next cycle write prediction
- [ ] **mem_la_addr**: Next cycle address prediction
- [ ] **mem_la_wdata**: Next cycle write data prediction
- [ ] **mem_la_wstrb**: Next cycle write strobe prediction

### 3. AXI4-Lite Interface
```bash
make test_axi                  # AXI4-Lite wrapper test
```

**Verify AXI signals:**
- [ ] **Read Address Channel**: ARVALID, ARREADY, ARADDR
- [ ] **Read Data Channel**: RVALID, RREADY, RDATA, RRESP
- [ ] **Write Address Channel**: AWVALID, AWREADY, AWADDR
- [ ] **Write Data Channel**: WVALID, WREADY, WDATA, WSTRB
- [ ] **Write Response Channel**: BVALID, BREADY, BRESP

### 4. Wishbone Interface
```bash
make test_wb                   # Wishbone master interface
```

**Verify Wishbone signals:**
- [ ] **CYC**: Cycle indicator
- [ ] **STB**: Strobe signal
- [ ] **WE**: Write enable
- [ ] **ADR**: Address bus
- [ ] **DAT_O**: Data output (write)
- [ ] **DAT_I**: Data input (read)
- [ ] **SEL**: Byte select
- [ ] **ACK**: Acknowledge

## Performance Verification ✓

### 1. Cycle Count Verification
```bash
cd dhrystone
make sim                       # Run Dhrystone benchmark
```

**Expected performance metrics:**
- [ ] **CPI**: ~4.0 cycles per instruction (dual-port config)
- [ ] **CPI**: ~4.5-5.0 cycles per instruction (single-port config)
- [ ] **DMIPS/MHz**: ~0.3-0.5 (configuration dependent)

### 2. Instruction Timing
**Verify cycle counts match specification:**
- [ ] **Direct jump (JAL)**: 3 cycles
- [ ] **ALU reg + immediate**: 3 cycles
- [ ] **ALU reg + reg**: 3-4 cycles (port dependent)
- [ ] **Branch not taken**: 3-4 cycles
- [ ] **Memory load**: 5 cycles
- [ ] **Memory store**: 5-6 cycles
- [ ] **Branch taken**: 5-6 cycles
- [ ] **Indirect jump (JALR)**: 6 cycles
- [ ] **Shift operations**: 4-14 cycles (config dependent)

### 3. Configuration Impact
```bash
# Test different configurations
make ENABLE_REGS_DUALPORT=0 test    # Single-port registers
make BARREL_SHIFTER=1 test           # Barrel shifter
make TWO_CYCLE_ALU=1 test           # Two-cycle ALU
make TWO_CYCLE_COMPARE=1 test       # Two-cycle compare
```

## Advanced Feature Verification ✓

### 1. Interrupt Support
```bash
make ENABLE_IRQ=1 test
cd firmware && make && cd .. && make test
```

**Verify interrupt functionality:**
- [ ] **IRQ assertion**: External interrupt handling
- [ ] **IRQ masking**: MASKIRQ instruction
- [ ] **IRQ return**: RETIRQ instruction  
- [ ] **Q registers**: GETQ/SETQ instructions
- [ ] **Timer interrupt**: Built-in timer
- [ ] **EBREAK handling**: Debug trap
- [ ] **Illegal instruction**: Trap generation

### 2. Co-Processor Interface (PCPI)
```bash
make ENABLE_PCPI=1 test
```

**Verify PCPI signals:**
- [ ] **pcpi_valid**: Instruction dispatch
- [ ] **pcpi_insn**: Instruction word
- [ ] **pcpi_rs1**: Source register 1
- [ ] **pcpi_rs2**: Source register 2
- [ ] **pcpi_wr**: Write result
- [ ] **pcpi_rd**: Result data
- [ ] **pcpi_wait**: Extension busy
- [ ] **pcpi_ready**: Extension complete

### 3. Trace Interface
```bash
make test TRACE=1
python3 showtrace.py testbench.trace firmware/firmware.elf
```

**Verify trace output:**
- [ ] **trace_valid**: Trace data valid
- [ ] **trace_data**: Instruction trace data (36-bit)
- [ ] **PC tracking**: Program counter values
- [ ] **Instruction correlation**: Trace matches execution

## Stress Testing ✓

### 1. Extended Runtime Tests
```bash
# Long-running tests
cd dhrystone && make DHRYSTONE_RUNS=10000 sim
cd firmware && make stress_test
```

### 2. Random Instruction Testing
```bash
cd scripts/torture
./test.sh                     # Random RISC-V instruction sequences
```

### 3. Corner Case Testing
```bash
# Edge conditions
make test ENABLE_CATCH_MISALIGN=1    # Misaligned access detection
make test ENABLE_CATCH_ILLINSN=1     # Illegal instruction detection
```

## Synthesis Verification ✓

### 1. Logic Synthesis
```bash
cd scripts/yosys
make synth_sim                 # Post-synthesis simulation
make lint_check               # Lint verification
```

### 2. FPGA Implementation
```bash
# Xilinx
cd scripts/vivado && make synth_area
# Intel
cd scripts/quartus && make synth_area  
# Lattice
cd scripts/icestorm && make example.bin
```

**Verify synthesis reports:**
- [ ] **LUT count**: Within expected range (750-2000)
- [ ] **Register count**: Within expected range (450-1100)
- [ ] **Memory blocks**: Appropriate usage
- [ ] **Timing closure**: Meets frequency targets
- [ ] **No synthesis warnings**: Clean synthesis

### 3. Hardware Validation
```bash
cd picosoc
make icebreaker.bin           # Build for real hardware
iceprog icebreaker.bin        # Program FPGA
minicom -D /dev/ttyUSB1       # Connect terminal
```

**Hardware test commands:**
```
# Type in terminal:
sieve                         # Sieve of Eratosthenes
multest                       # Multiply/divide test  
benchmark                     # Performance benchmark
echo hello world             # Echo test
```

## Documentation Verification ✓

### 1. Code Comments
- [ ] **Module interfaces**: Well documented
- [ ] **Parameter descriptions**: Complete and accurate
- [ ] **Signal descriptions**: Clear and consistent
- [ ] **Configuration options**: Properly explained

### 2. User Documentation
- [ ] **README accuracy**: Instructions work as written
- [ ] **Build instructions**: Complete and current
- [ ] **Examples**: All examples build and run
- [ ] **Performance data**: Matches actual measurements

## Final Verification Report ✓

### Summary Checklist
- [ ] **All basic instructions**: RV32I compliance verified
- [ ] **Extended instructions**: M-extension working (if enabled)
- [ ] **Compressed instructions**: C-extension working (if enabled)
- [ ] **Memory interface**: All access types working
- [ ] **Performance**: Meets specification benchmarks
- [ ] **Synthesis**: Clean synthesis on target tools
- [ ] **Hardware**: Works on real FPGA hardware

### Performance Summary
```
Configuration: _____________
CPI: _______ cycles/instruction
DMIPS/MHz: _______ 
Max Frequency: _______ MHz
LUT Usage: _______ LUTs
Register Usage: _______ FFs
```

### Sign-off
- [ ] **Functional verification**: Complete ✓
- [ ] **Performance verification**: Complete ✓  
- [ ] **Synthesis verification**: Complete ✓
- [ ] **Hardware verification**: Complete ✓
- [ ] **Ready for production use**: ✓

**Verification Engineer**: _________________ **Date**: _________
