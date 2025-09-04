# Post-Build Validation Procedures

## Quick Validation Sequence (2 minutes)

Once the build completes, execute these commands to validate the enhanced CIRCT integration system:

### **Step 1: Basic Functionality Check**
```bash
# Verify sv2sc executable was created
ls -la build/src/sv2sc
echo "Expected: Executable file with recent timestamp"

# Check MLIR options are available
./build/src/sv2sc --help | grep -A 5 "MLIR Pipeline Options:"
echo "Expected: MLIR options displayed (--use-mlir, --mlir-diagnostics, --dump-mlir)"
```

### **Step 2: Integration Mode Verification**
```bash
# Check current integration mode
cmake -B build -DSV2SC_ENABLE_MLIR=ON 2>&1 | grep "Integration mode:"
echo "Expected: 'Integration mode: MOCK' or similar"

# Verify mock headers were created
ls -la build/mock-mlir/include/mlir/IR/
echo "Expected: MLIRContext.h, Builders.h, Types.h, Value.h, Module.h"
```

### **Step 3: MLIR Pipeline Functional Test**
```bash
# Test basic MLIR pipeline (if test files exist)
if [ -f tests/examples/basic_counter/counter.sv ]; then
  ./build/src/sv2sc --use-mlir -top counter tests/examples/basic_counter/counter.sv
  echo "Expected: Translation completes without fatal errors"
fi

# Test MLIR diagnostics option
./build/src/sv2sc --use-mlir --mlir-diagnostics --help >/dev/null 2>&1
echo $? # Expected: 0 (success)
```

## Comprehensive Validation Sequence (10 minutes)

### **Phase 1: Build Verification**
```bash
# Check all MLIR components were built
ls -la build/src/mlir/CMakeFiles/sv2sc_mlir.dir/
echo "Expected: .o files for MLIRContextManager, SVToHWBuilder, etc."

# Verify library linking
ldd build/src/sv2sc | grep -i mlir || echo "Mock mode: no real MLIR libraries linked"
echo "Expected: Either MLIR libraries or 'Mock mode' message"
```

### **Phase 2: Integration Mode Testing**
```bash
# Test explicit mode setting
cmake -B build-test -DSV2SC_ENABLE_MLIR=ON -DCIRCT_INTEGRATION_MODE=MOCK
echo "Expected: Configuration succeeds with mock mode"

# Clean up test build
rm -rf build-test
```

### **Phase 3: Mock Header Validation**
```bash
# Check essential mock headers exist and are valid
for header in MLIRContext.h Builders.h Types.h Value.h Module.h; do
  if [ -f "build/mock-mlir/include/mlir/IR/$header" ]; then
    echo "✅ $header exists"
    head -3 "build/mock-mlir/include/mlir/IR/$header"
  else
    echo "❌ $header missing"
  fi
done
```

### **Phase 4: Error Handling Validation**
```bash
# Test behavior with invalid MLIR options
./build/src/sv2sc --use-mlir --invalid-option 2>&1 | head -3
echo "Expected: Proper error message, not crash"

# Test behavior with missing files
./build/src/sv2sc --use-mlir -top nonexistent nonexistent.sv 2>&1 | head -3  
echo "Expected: File not found error, not crash"
```

## Performance and Integration Tests

### **Build Performance Metrics**
```bash
# Check build time components
echo "Build component analysis:"
echo "- slang library: largest component (80%+ of build time)"
echo "- sv2sc_mlir: MLIR infrastructure (compiles quickly with mocks)"
echo "- sv2sc core: translator components"

# Verify incremental build performance
touch src/mlir/SVToHWBuilder.cpp
time cmake --build build --target sv2sc
echo "Expected: Fast incremental build (< 30 seconds)"
```

### **Memory Usage Validation**
```bash
# Test memory usage with MLIR enabled
./build/src/sv2sc --use-mlir --help >/dev/null &
PID=$!
sleep 1
ps -p $PID -o pid,vsz,rss,comm 2>/dev/null || echo "Process completed quickly"
echo "Expected: Reasonable memory usage, process runs successfully"
```

## Integration Pathway Testing

### **Mock to System Package Simulation**
```bash
# Simulate what would happen with system packages
echo "Simulating system package detection..."
echo "Would run: find_package(LLVM REQUIRED CONFIG)"
echo "Would run: find_package(MLIR REQUIRED CONFIG)" 
echo "Would run: find_package(CIRCT REQUIRED CONFIG)"
echo "Result: If packages found, would use CIRCTSystemIntegration.cmake"
```

### **Mock to Source Build Simulation**
```bash
# Check if source build infrastructure is ready
if [ -d "third-party/circt" ]; then
  echo "✅ CIRCT source available"
  echo "Build command ready: cd third-party/circt && cmake -S llvm/llvm -B llvm-build"
else
  echo "❌ CIRCT source not available"
fi
```

## Documentation Validation

### **Verify Documentation Files**
```bash
echo "Checking documentation completeness:"
for doc in CIRCT_INTEGRATION_GUIDE.md ENHANCED_CIRCT_INTEGRATION_STATUS.md PHASE4_READINESS_STATUS.md; do
  if [ -f "docs/$doc" ]; then
    echo "✅ docs/$doc ($(wc -l < docs/$doc) lines)"
  else
    echo "❌ docs/$doc missing"
  fi
done
```

### **Validate CMake Module Files**
```bash
echo "Checking CMake integration modules:"
for module in CIRCTSystemIntegration.cmake CIRCTSourceIntegration.cmake CIRCTMockIntegration.cmake; do
  if [ -f "cmake/$module" ]; then
    echo "✅ cmake/$module"
  else
    echo "❌ cmake/$module missing"
  fi
done
```

## Success Criteria Checklist

After running all validations, verify these success criteria:

### **✅ Basic Functionality**
- [ ] sv2sc executable created and runs
- [ ] MLIR command-line options available (--use-mlir, --mlir-diagnostics, --dump-mlir)
- [ ] Help message displays correctly
- [ ] No crashes on basic operations

### **✅ Mock Integration System**  
- [ ] Mock headers generated automatically
- [ ] All essential MLIR/CIRCT interfaces available
- [ ] Compilation succeeds with mock dependencies
- [ ] Integration mode correctly identified as MOCK

### **✅ Architecture Validation**
- [ ] MLIR infrastructure compiles successfully
- [ ] SVToHWBuilder operations execute (with placeholders)
- [ ] Pass pipeline infrastructure functional
- [ ] Error handling works correctly

### **✅ Enhancement Features**
- [ ] Modular integration system operational
- [ ] Automatic detection logic functional
- [ ] Multiple integration pathway support
- [ ] Comprehensive documentation available

### **✅ Production Readiness**
- [ ] Clear upgrade pathways documented
- [ ] System package integration prepared  
- [ ] Source build integration prepared
- [ ] Deployment strategies documented

## Troubleshooting Common Issues

### **Build Issues**
```bash
# If build fails with missing headers:
echo "Check: build/mock-mlir/include directory created?"
echo "Solution: Re-run cmake configuration"

# If linking fails:
echo "Check: MLIR::Dependencies target created?"
echo "Solution: Verify CIRCTMockIntegration.cmake included"
```

### **Runtime Issues**
```bash
# If MLIR options not recognized:
echo "Check: SV2SC_HAS_MLIR compilation flag set?"
echo "Solution: Verify mock integration included SV2SC_HAS_MLIR"

# If mock operations fail:
echo "Check: Mock headers have required interfaces?"
echo "Solution: Update mock header generation in CIRCTMockIntegration.cmake"
```

This validation suite ensures that our enhanced CIRCT integration system works correctly and provides a solid foundation for Phase 4 real CIRCT integration when ready.