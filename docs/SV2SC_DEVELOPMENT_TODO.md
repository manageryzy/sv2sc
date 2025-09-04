# SV2SC Development TODO List

## Project Status
**Last Updated**: September 2, 2025  
**Current State**: CIRCT fallback generator successfully implemented as default SystemC emitter

## ✅ Recently Completed
- [x] **CIRCT Fallback Implementation**: Created comprehensive CIRCT-compatible SystemC generator
- [x] **Default Integration**: Made fallback generator the primary SystemC emitter
- [x] **Build System**: Successfully integrated with CMake and MLIR infrastructure
- [x] **Basic Functionality**: Simple modules translate correctly (3-4ms performance)
- [x] **Field Consistency**: Fixed EmitResult field naming issues (error vs errorMessage)
- [x] **Clean Architecture**: Removed dependency on external CIRCT ExportSystemC

## 🔥 Critical Priority (Fix Immediately)

### 1. Fix Unit Test Build Issues
**Status**: BROKEN - Tests not building  
**Impact**: Cannot maintain code quality without working tests  
**Location**: `tests/unit_tests/`, `CMakeLists.txt`  
**Issue**: Unit test executables missing from build output
```bash
# Current error:
Could not find executable /home/mana/workspace/sv2sc/build/tests/unit_tests/test_systemc_generator
```
**Action Required**:
- Fix CMake configuration for test targets
- Ensure Catch2 linking is correct
- Verify test source compilation

### 2. Enhance SystemC Emission Patterns
**Status**: INCOMPLETE - Missing core patterns  
**Impact**: Generated SystemC code has "Unsupported operation" comments  
**Location**: `src/mlir/systemc/CIRCTCompatibleEmitter.cpp`, `SystemCEmissionPatterns.cpp`  
**Missing Patterns**:
- `emitc.include` - SystemC header includes
- `arith.constant` - Constant value emission
- `systemc.method` - SC_METHOD generation
- `systemc.sensitive` - Sensitivity list generation
- `systemc.ctor` - SC_CTOR generation
- `systemc.func` - Function/method bodies
- `systemc.convert` - Type conversions

**Current Error Example**:
```
Emission error: No emission pattern found for operation: emitc.include
Emission error: No emission pattern found for operation: arith.constant
```

## 🚨 High Priority (Next 1-2 weeks)

### 3. Fix MLIR Pipeline for Complex Modules
**Status**: FAILING for complex designs  
**Impact**: MLIR mode only works for simple modules  
**Location**: `src/mlir/pipeline/SV2SCPassPipeline.cpp`, `passes/HWToSystemCLoweringPass.cpp`  
**Issue**: Type checking errors in SystemC dialect
```
error: 'systemc.signal.read' op operand #0 must be a SystemC sc_in<T> type 
but got '!systemc.out<!systemc.uint<8>>'
```
**Root Cause**: Attempting to read from output ports during lowering

### 4. Implement Procedural Block Support
**Status**: TODO - Critical for always blocks  
**Impact**: Cannot properly handle SystemVerilog always blocks  
**Location**: `src/mlir/SVToHWBuilder.cpp:205`  
**Current State**: Placeholder comments, no actual implementation
```cpp
// TODO: Implement proper seq::CompRegOp with regions
// TODO: Implement proper comb operations with regions
// TODO: Implement proper always block with regions
```

## 📋 Medium Priority (Next 1-2 months)

### 5. Statement and Expression Conversion
**Status**: INCOMPLETE - Many TODOs  
**Location**: `src/mlir/SVToHWBuilder.cpp` (lines 1083-1131)  
**Missing Features**:
- Assignment statements (blocking/non-blocking)
- If-else statement conversion
- For/while loop conversion
- Case statement conversion
- Function/task calls
- Timing control statements (@posedge, @negedge)

### 6. Type System Enhancement
**Status**: BASIC - Only primitive types  
**Location**: `src/mlir/SVToHWBuilder.cpp:1219-1239`  
**Missing Types**:
- Struct type mapping
- Enum type mapping  
- String type mapping
- Array types (partially implemented)

### 7. Expression Handling Improvements
**Status**: INCOMPLETE - Slang API issues  
**Location**: `src/mlir/SVToHWBuilder.cpp` (lines 843-867, 876, 977)  
**Issues**:
- Fix slang API usage for expressions
- Implement missing unary operators (logical negation, increment/decrement)
- Fix conditional expressions
- Fix concatenation expressions

## 🔧 Long-term Goals (Next 3-6 months)

### 8. Comprehensive Testing Framework
**Status**: NEEDS EXPANSION  
**Current**: Basic example tests only  
**Required**:
- Unit tests for all components
- Integration tests for SystemVerilog features
- Performance regression testing
- Real-world design validation
- Automated quality metrics

### 9. Performance Optimization
**Status**: GOOD baseline, needs profiling  
**Current Performance**:
- Simple modules: 3-4ms
- Complex modules: 35ms (standard mode)
- MLIR mode: 1.5-45x speedup potential
**Optimization Areas**:
- MLIR pass pipeline efficiency
- Memory usage optimization
- Parallel processing for multi-file designs

### 10. Advanced SystemVerilog Features
**Status**: NOT STARTED  
**Features Needed**:
- SystemVerilog interfaces and modports
- Classes and OOP constructs
- Assertions and coverage (SVA)
- DPI-C integration
- Randomization (rand/randc)

## 📊 Technical Debt

### Code Quality Issues
- **Location**: `src/mlir/SVToHWBuilder.cpp`
- **Issue**: Many TODO comments and placeholder implementations
- **Count**: 40+ TODO items identified
- **Impact**: Incomplete functionality, maintenance burden

### Architecture Improvements Needed
- **Multi-file Design Processing**: Currently basic (line 99 in MLIRTranslator.cpp)
- **Error Handling**: Needs comprehensive error recovery
- **Memory Management**: Review for large design handling
- **API Consistency**: Standardize interfaces between components

## 🎯 Success Metrics

### Immediate (1-2 weeks)
- [ ] All unit tests building and passing
- [ ] SystemC emission patterns cover 90% of common operations
- [ ] MLIR mode works for medium complexity modules

### Medium-term (1-2 months)  
- [ ] Complete SystemVerilog statement support
- [ ] Comprehensive type system
- [ ] Performance benchmarks established
- [ ] Test coverage >80%

### Long-term (3-6 months)
- [ ] Full CPU designs (PicoRV32) translate correctly
- [ ] Advanced SystemVerilog features supported
- [ ] Production-ready performance and reliability
- [ ] Complete documentation and user guides

## 🔍 Investigation Needed

### Build System Issues
- Why are unit tests not being built?
- Are there missing dependencies in CMake configuration?
- Is the test discovery mechanism working correctly?

### MLIR Pipeline Debugging
- What's causing the type checking failures?
- Are the SystemC dialect operations correctly defined?
- Is the lowering pass sequence optimal?

### Performance Analysis
- Where are the bottlenecks in translation?
- Can we parallelize any operations?
- What's the memory usage pattern for large designs?

## 📝 Notes for Future Development

### Architecture Decisions Made
1. **Fallback as Default**: Our CIRCT-compatible emitter is now the primary SystemC generator
2. **Template Engine**: Using state-of-the-art template library for code generation
3. **Modern CMake**: CMake 3.20+ with FetchContent for dependencies
4. **C++20**: Full modern C++ feature usage
5. **MLIR Integration**: Optional but preferred for complex transformations

### Key Files to Know
- `src/mlir/systemc/CIRCTCompatibleEmitter.cpp` - Main fallback generator (648 lines)
- `src/mlir/systemc/SystemCEmissionPatterns.cpp` - Emission pattern definitions
- `src/mlir/SVToHWBuilder.cpp` - SystemVerilog to HW dialect conversion
- `src/mlir/MLIRTranslator.cpp` - Main MLIR translation orchestrator
- `cmake/SystemCTestUtils.cmake` - Test framework utilities

### Development Environment
- **OS**: Linux (WSL2 compatible)
- **Compiler**: Clang 18.1.8 with C++20
- **Build**: CMake with Ninja generator
- **Dependencies**: LLVM/MLIR/CIRCT (auto-fetched)
- **Testing**: Catch2 framework (when working)

---

**For questions or clarification on any TODO item, refer to the source code locations provided or check the git history for recent changes.**
