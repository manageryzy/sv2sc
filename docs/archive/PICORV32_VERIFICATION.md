# PicoRV32 Verification Suite

## Overview

This verification suite provides a comprehensive framework for comparing the sv2sc-translated PicoRV32 implementation against the Verilator reference implementation. The goal is to validate the correctness and performance of our SystemVerilog to SystemC translation.

## Architecture

```
tests/verification/picorv32/
├── CMakeLists.txt           # Build configuration
├── testbench/
│   ├── systemc_testbench.cpp    # SystemC testbench for sv2sc output
│   └── verilator_testbench.cpp  # Verilator reference testbench
├── test_programs/
│   ├── simple_test.S        # Basic arithmetic test
│   ├── memory_test.S        # Memory operations test
│   ├── arithmetic_test.S    # Complex arithmetic test
│   ├── branch_test.S        # Branch and jump test
│   └── interrupt_test.S     # Interrupt handling test
├── scripts/
│   ├── compare_outputs.py   # Output comparison tool
│   ├── run_comparison.sh    # Main comparison script
│   ├── benchmark.py         # Performance benchmarking
│   └── generate_report.py   # Report generation
└── output/                  # Generated outputs and reports
```

## Building the Verification Suite

### Prerequisites

1. **sv2sc tool** - Built and available in the build directory
2. **Verilator** (optional but recommended) - For reference implementation
3. **RISC-V toolchain** (optional) - For compiling test programs
4. **SystemC** - Installed and configured

### Build Commands

```bash
# Configure with verification enabled
cmake -B build \
    -DBUILD_PICORV32_VERIFICATION=ON \
    -DBUILD_VERILATOR_REF=ON \
    -DBUILD_SV2SC_IMPL=ON

# Build everything
cmake --build build -j$(nproc)

# Run verification tests
cd build
ctest -R picorv32
```

### Configuration Options

- `BUILD_PICORV32_VERIFICATION` - Enable the verification suite
- `BUILD_VERILATOR_REF` - Build Verilator reference (requires Verilator)
- `BUILD_SV2SC_IMPL` - Build sv2sc translated implementation
- `RUN_COMPARISON` - Enable comparison tests
- `GENERATE_WAVEFORMS` - Generate VCD waveforms for debugging

## Running Tests

### Basic Comparison

```bash
# Run the comparison script
./tests/verification/picorv32/scripts/run_comparison.sh \
    build/tests/verification/picorv32/systemc \
    build/tests/verification/picorv32/verilator \
    build/tests/verification/picorv32/output
```

### Individual Simulations

```bash
# Run SystemC simulation
./build/tests/verification/picorv32/picorv32_systemc \
    --hex firmware.hex \
    --vcd systemc.vcd \
    --timeout 10000

# Run Verilator simulation
./build/tests/verification/picorv32/verilator/Vpicorv32 \
    --hex firmware.hex \
    --vcd verilator.vcd \
    --timeout 10000
```

### Automated Comparison

```bash
# Compare specific test program
python3 tests/verification/picorv32/scripts/compare_outputs.py \
    --systemc build/tests/verification/picorv32/picorv32_systemc \
    --verilator build/tests/verification/picorv32/verilator/Vpicorv32 \
    --program tests/verification/picorv32/test_programs/simple_test.hex \
    --output comparison_results.json
```

## Test Programs

### simple_test.S
Basic test covering:
- Register initialization
- Arithmetic operations (ADD, SUB)
- Memory operations (LW, SW)
- Branch instructions
- UART output

### memory_test.S
Memory subsystem test:
- Sequential memory access
- Random access patterns
- Byte/halfword/word operations
- Memory-mapped I/O

### arithmetic_test.S
Arithmetic operations:
- All ALU operations
- Multiplication/division
- Shift operations
- Immediate operations

### branch_test.S
Control flow:
- Conditional branches
- Unconditional jumps
- Function calls (JAL/JALR)
- Loop constructs

### interrupt_test.S
Interrupt handling:
- External interrupts
- Timer interrupts
- Interrupt priorities
- Nested interrupts

## Comparison Metrics

The verification suite compares:

1. **Functional Correctness**
   - Instruction execution count
   - Memory state after execution
   - Register values at checkpoints
   - UART/console output

2. **Timing Accuracy**
   - Cycle count
   - Instruction per cycle (IPC)
   - Memory access patterns

3. **Exception Handling**
   - Trap detection
   - Exception cause
   - Exception PC

4. **Performance Metrics**
   - Simulation speed
   - Memory usage
   - Compilation time

## Results Interpretation

### Comparison Output Format

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "program": "simple_test.hex",
  "systemc": {
    "success": true,
    "metrics": {
      "instructions": 1000,
      "cycles": 1500,
      "uart_output": "PASS\n",
      "trap": true
    }
  },
  "verilator": {
    "success": true,
    "metrics": {
      "instructions": 1000,
      "cycles": 1500,
      "uart_output": "PASS\n",
      "trap": true
    }
  },
  "comparison": {
    "status": "PASS",
    "instruction_match": true,
    "uart_match": true,
    "trap_match": true,
    "match_percentage": 100.0
  }
}
```

### Success Criteria

A test passes if:
- Both simulations complete without errors
- Instruction counts match (±1% tolerance)
- UART outputs are identical
- Trap status matches
- No undefined behavior detected

## Debugging Failed Tests

### Enable Verbose Output

```bash
# Run with debug output
./picorv32_systemc --hex test.hex --debug --vcd debug.vcd

# Compare with verbose mode
python3 compare_outputs.py --verbose ...
```

### Waveform Analysis

```bash
# Generate waveforms
make generate_waveforms

# View with GTKWave
gtkwave systemc.vcd verilator.vcd
```

### Common Issues

1. **Type Mismatches**
   - Check signal width conversions
   - Verify 4-state vs 2-state logic handling

2. **Timing Differences**
   - Review clock edge sensitivity
   - Check reset polarity and timing

3. **Memory Model Differences**
   - Verify memory initialization
   - Check byte-enable handling

## Performance Benchmarking

### Run Benchmarks

```bash
# Run performance comparison
make benchmark

# Results in output/benchmark_results.json
```

### Metrics Collected

- **Translation Time**: Time to convert .v to SystemC
- **Compilation Time**: Time to compile generated code
- **Simulation Speed**: Instructions/second
- **Memory Usage**: Peak memory consumption

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: PicoRV32 Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y verilator systemc
      - name: Build sv2sc
        run: |
          cmake -B build -DBUILD_PICORV32_VERIFICATION=ON
          cmake --build build
      - name: Run verification
        run: |
          cd build
          ctest -R picorv32 --output-on-failure
```

## Extending the Verification Suite

### Adding New Test Programs

1. Create assembly file in `test_programs/`
2. Add to `TEST_PROGRAMS` list in CMakeLists.txt
3. Create expected output reference

### Adding New Comparison Metrics

1. Extend parsing in `compare_outputs.py`
2. Add metric extraction to testbenches
3. Update comparison logic

### Supporting Other Cores

The framework can be extended to verify other RISC-V cores:
1. Copy verification structure
2. Adapt testbenches for core-specific signals
3. Update memory maps and I/O addresses

## Known Limitations

1. **Partial SystemVerilog Support**
   - Some advanced SystemVerilog features may not translate perfectly
   - Workarounds documented in translation notes

2. **Performance Overhead**
   - SystemC simulation typically slower than Verilator
   - Optimization opportunities identified in profiling

3. **Memory Model Simplifications**
   - Simplified memory model for faster simulation
   - May not catch all timing-related bugs

## Future Enhancements

1. **Formal Verification**
   - Add equivalence checking using SymbiYosys
   - Property-based testing

2. **Coverage Analysis**
   - Code coverage metrics
   - Functional coverage points

3. **Regression Testing**
   - Automated nightly runs
   - Performance regression detection

4. **Multi-core Support**
   - Verify multi-core configurations
   - Cache coherence testing

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review generated logs in `output/`
3. File an issue with comparison results attached
