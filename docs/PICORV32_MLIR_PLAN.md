# PicoRV32 MLIR Pipeline Implementation Plan

## Current Status (as of build time)

### ✅ Completed Steps

1. **CIRCT Integration in CMake**
   - Configured to build LLVM+MLIR+CIRCT in-tree
   - Using CIRCT from `third-party/circt` submodule
   - Dependencies.cmake updated for in-tree build

2. **Build Configuration**
   ```bash
   cmake -B build -DSV2SC_ENABLE_MLIR=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target sv2sc -j16
   ```
   - Currently building: ~38% complete
   - Estimated completion: 30-45 minutes

3. **Test Infrastructure**
   - `scripts/picorv32_mlir_test.sh`: Complete test suite
   - `scripts/monitor_build.sh`: Build progress monitor
   - Tests both basic and MLIR translation pipelines

### 🚧 In Progress

- **LLVM+MLIR+CIRCT Build**: Building as part of main CMake build
- **SVToHWBuilder**: Has placeholder CIRCT operations, ready for real ops

### 📋 Remaining Tasks

1. **Complete Build** (automated)
2. **Test MLIR Pipeline** 
3. **Translate PicoRV32**
4. **Build and Run SystemC Simulation**

## Implementation Architecture

### Build System
```
sv2sc/
├── third-party/
│   └── circt/          # CIRCT submodule with LLVM
├── build/
│   ├── llvm/           # Built LLVM+MLIR
│   └── circt/          # Built CIRCT
└── src/mlir/           # MLIR translation code
```

### Translation Pipeline
```
SystemVerilog (.v) 
    ↓ [slang parser]
AST 
    ↓ [SVToHWBuilder]
HW Dialect (MLIR)
    ↓ [Lowering passes]
SystemC Dialect
    ↓ [Code emission]
SystemC (.h/.cpp)
```

## Quick Start Commands

### Monitor Build Progress
```bash
# Watch build progress
./scripts/monitor_build.sh

# Or check manually
tail -f build_sv2sc.log
```

### Test PicoRV32 Translation (after build completes)
```bash
# Run complete test suite
./scripts/picorv32_mlir_test.sh

# Or test individual components:

# 1. Basic translation (no MLIR)
./build/src/sv2sc -top picorv32 -o output/basic third-party/picorv32/picorv32.v

# 2. MLIR translation
./build/src/sv2sc --use-mlir -top picorv32 -o output/mlir third-party/picorv32/picorv32.v

# 3. Dump MLIR IR
./build/src/sv2sc --use-mlir --dump-mlir -top picorv32 third-party/picorv32/picorv32.v > picorv32.mlir
```

## Expected Results

### With MLIR Pipeline
- Translation should use CIRCT HW dialect
- Generated SystemC should be semantically equivalent
- Performance may differ from string-based translation
- MLIR IR can be inspected for debugging

### Test Metrics
- Port count validation
- Translation quality score
- SystemC compilation success
- Simulation execution (if compilation succeeds)

## Troubleshooting

### Build Issues
```bash
# Check build errors
grep -i error build_sv2sc.log

# Clean and rebuild if needed
rm -rf build
cmake -B build -DSV2SC_ENABLE_MLIR=ON
cmake --build build --target sv2sc
```

### MLIR Not Available
```bash
# Verify MLIR support
./build/src/sv2sc --help | grep "MLIR Pipeline"

# If not found, check CMake configuration
grep "MLIR" build/CMakeCache.txt
```

### Translation Failures
```bash
# Enable debug output
./build/src/sv2sc --use-mlir --mlir-diagnostics --debug -top picorv32 picorv32.v

# Check generated files
ls -la output/mlir/
```

## Technical Details

### CIRCT Operations Used
- `hw.module`: Module definitions
- `hw.wire`: Wire/signal declarations
- `comb.add/sub/mul`: Arithmetic operations
- `comb.icmp`: Comparison operations
- `comb.and/or/xor`: Bitwise operations
- `comb.mux`: Conditional selection
- `seq.compreg`: Sequential logic

### Type Mapping
| SystemVerilog | HW Dialect | SystemC |
|--------------|------------|---------|
| logic [N:0] | hw.int<N+1> | sc_lv<N+1> |
| bit [N:0] | hw.int<N+1> | sc_bv<N+1> |
| wire | hw.wire | sc_signal |
| reg | seq.compreg | sc_signal |

## Performance Expectations

### Build Time
- LLVM+MLIR+CIRCT: 45-90 minutes (first time)
- Subsequent builds: 5-10 minutes (incremental)

### Translation Time
- Basic (string-based): ~1.3 seconds for PicoRV32
- MLIR pipeline: ~2-5 seconds (includes optimization passes)

### Simulation Performance
- Depends on SystemC compilation success
- NBA splitting may improve performance 2-5x

## Next Development Steps

1. **Complete Real CIRCT Operations** (after build)
   - Replace placeholder operations in SVToHWBuilder
   - Test with simple examples first

2. **Optimize Pass Pipeline**
   - Add custom optimization passes
   - Configure pass pipeline for performance

3. **Fix SystemC Generation Issues**
   - Address type mismatches
   - Handle complex expressions
   - Fix signal connectivity

4. **Full Verification Suite**
   - Compare with Verilator reference
   - Run RISC-V test programs
   - Performance benchmarking

## References

- [CIRCT Documentation](https://circt.llvm.org/)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
- [PicoRV32 Repository](https://github.com/YosysHQ/picorv32)
- [SystemC Documentation](https://www.accellera.org/downloads/standards/systemc)

---

*Last Updated: Build in progress*  
*Status: Building LLVM+MLIR+CIRCT (38% complete)*
