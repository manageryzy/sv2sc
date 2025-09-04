# CIRCT Integration Guide - Phase 4 Implementation

## Overview

This guide provides comprehensive instructions for integrating real CIRCT (Circuit IR Compilers and Tools) support into sv2sc, transitioning from the current mock MLIR setup to a production-ready LLVM-based hardware compiler infrastructure.

## Current Status

### ✅ What's Ready (Phases 1-3)
- **Complete MLIR Infrastructure**: Context management, pass pipelines, builder framework
- **Realistic HW Dialect Operations**: 50+ operations with CIRCT operation mapping comments
- **Comprehensive Expression Support**: All major SystemVerilog expression types
- **Advanced Module Processing**: Variables, procedural blocks, continuous assignments
- **Mock Testing Environment**: Validates architecture without full CIRCT build

### 🔄 What Needs CIRCT Integration (Phase 4)
- **Real HW Dialect Operations**: Replace placeholder constants with actual CIRCT operations
- **CIRCT Type System**: Proper type conversion using CIRCT type infrastructure
- **SystemC Dialect Integration**: Connect to CIRCT's SystemC generation capabilities
- **LLVM Build Integration**: Full LLVM+MLIR+CIRCT compilation environment

## Phase 4 Implementation Options

### Option A: System Package Installation (Fastest)

**For Ubuntu/Debian systems:**
```bash
# Install LLVM/MLIR packages (if available)
sudo apt update
sudo apt install llvm-dev mlir-tools libmlir-dev

# Check available CIRCT packages
apt search circt

# If CIRCT packages exist, install them
sudo apt install circt-dev circt-tools
```

**Update Dependencies.cmake:**
```cmake
# Replace mock setup with:
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

# Use system-provided libraries
set_property(TARGET MLIR::Dependencies PROPERTY
    INTERFACE_LINK_LIBRARIES 
    ${MLIR_LIBRARIES} ${CIRCT_LIBRARIES} ${LLVM_LIBRARIES})
```

### Option B: Pre-built CIRCT Installation

**Download CIRCT Release:**
```bash
# Download latest CIRCT release
wget https://github.com/llvm/circt/releases/latest/download/circt-linux.tar.gz
tar xzf circt-linux.tar.gz -C /opt/
export CIRCT_DIR=/opt/circt
```

**Update Dependencies.cmake:**
```cmake
# Set CIRCT installation path
set(CIRCT_ROOT /opt/circt)
find_package(CIRCT REQUIRED PATHS ${CIRCT_ROOT})
```

### Option C: Build from Source (Most Control)

**This is the most comprehensive but time-intensive approach (2-4 hours build time).**

#### Step 1: Prepare Build Environment
```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential cmake ninja-build python3-dev

# Optional: Install ccache for faster rebuilds
sudo apt install ccache
export CC="ccache gcc"
export CXX="ccache g++"
```

#### Step 2: Build LLVM+MLIR+CIRCT
```bash
# Clone and prepare CIRCT (already done in our case)
cd third-party/circt
git submodule update --init --recursive

# Configure LLVM+MLIR build
cmake -S llvm/llvm -B llvm-build -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DLLVM_ENABLE_PROJECTS="mlir" \\
    -DLLVM_TARGETS_TO_BUILD="host" \\
    -DLLVM_ENABLE_ASSERTIONS=ON \\
    -DLLVM_ENABLE_RTTI=ON \\
    -DLLVM_ENABLE_EH=ON \\
    -DLLVM_BUILD_EXAMPLES=OFF \\
    -DLLVM_BUILD_TESTS=OFF

# Build LLVM+MLIR (2-3 hours)
cmake --build llvm-build --target all -j$(nproc)

# Configure CIRCT build
cmake -S . -B circt-build -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DMLIR_DIR=$PWD/llvm-build/lib/cmake/mlir \\
    -DLLVM_DIR=$PWD/llvm-build/lib/cmake/llvm \\
    -DCIRCT_BUILD_TOOLS=ON \\
    -DCIRCT_ENABLE_BINDINGS_PYTHON=OFF

# Build CIRCT (30-60 minutes)
cmake --build circt-build --target all -j$(nproc)
```

#### Step 3: Update sv2sc Dependencies.cmake
```cmake
# Replace mock setup with real CIRCT paths
if(SV2SC_ENABLE_MLIR)
    set(CIRCT_DIR "${CMAKE_SOURCE_DIR}/third-party/circt/circt-build")
    set(MLIR_DIR "${CMAKE_SOURCE_DIR}/third-party/circt/llvm-build/lib/cmake/mlir")
    set(LLVM_DIR "${CMAKE_SOURCE_DIR}/third-party/circt/llvm-build/lib/cmake/llvm")
    
    find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR})
    find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
    find_package(CIRCT REQUIRED CONFIG PATHS ${CIRCT_DIR})
    
    # Include real CIRCT directories
    include_directories(SYSTEM ${CIRCT_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
endif()
```

## Code Migration Steps

### Step 1: Replace Placeholder Operations

**Current placeholder pattern:**
```cpp
// src/mlir/SVToHWBuilder.cpp - BEFORE
case slang::ast::BinaryOperator::Add:
    LOG_DEBUG("Binary add operation: creating comb.add");
    // Note: In real CIRCT integration, this would use comb.add
    return builder_.create<circt::hw::ConstantOp>(loc, leftValue.getType(), 
        mlir::IntegerAttr::get(leftValue.getType(), 42)); // Placeholder result
```

**Real CIRCT integration:**
```cpp
// src/mlir/SVToHWBuilder.cpp - AFTER
case slang::ast::BinaryOperator::Add:
    LOG_DEBUG("Binary add operation: creating comb.add");
    // Real CIRCT operation
    return builder_.create<circt::comb::AddOp>(loc, leftValue, rightValue);
```

### Step 2: Update Include Headers

**Replace mock includes:**
```cpp
// BEFORE (mock)
#include "mlir/IR/MLIRContext.h"      // Mock header
#include "circt/Dialect/HW/HWDialect.h"  // Mock header

// AFTER (real CIRCT)
#include "mlir/IR/MLIRContext.h"      // Real MLIR header
#include "circt/Dialect/HW/HWDialect.h"  // Real CIRCT header
#include "circt/Dialect/Comb/CombOps.h"  // Real combinational operations
```

### Step 3: Enhance Type Conversion System

**Current placeholder:**
```cpp
mlir::Type SVToHWBuilder::convertSystemVerilogTypeToHWType(const slang::ast::Type& svType) {
    // TODO: Implement real type conversion
    return builder_.getI32Type(); // Placeholder
}
```

**Real CIRCT type conversion:**
```cpp
mlir::Type SVToHWBuilder::convertSystemVerilogTypeToHWType(const slang::ast::Type& svType) {
    if (svType.isIntegral()) {
        auto& integralType = svType.as<slang::ast::IntegralType>();
        if (integralType.getBitWidth() == 1) {
            return circt::hw::InOutType::get(builder_.getContext(), 
                builder_.getI1Type());
        } else {
            return circt::hw::ArrayType::get(
                builder_.getI1Type(), integralType.getBitWidth());
        }
    }
    
    // Handle other SystemVerilog types...
    return mlir::Type{};
}
```

### Step 4: Implement Real Operations

**All placeholder operations in SVToHWBuilder.cpp need updating:**

```cpp
// Arithmetic Operations
case slang::ast::BinaryOperator::Add:
    return builder_.create<circt::comb::AddOp>(loc, leftValue, rightValue);

case slang::ast::BinaryOperator::Subtract:
    return builder_.create<circt::comb::SubOp>(loc, leftValue, rightValue);

case slang::ast::BinaryOperator::Multiply:
    return builder_.create<circt::comb::MulOp>(loc, leftValue, rightValue);

// Comparison Operations  
case slang::ast::BinaryOperator::Equality:
    return builder_.create<circt::comb::ICmpOp>(loc, 
        circt::comb::ICmpPredicate::eq, leftValue, rightValue);

// Bitwise Operations
case slang::ast::BinaryOperator::BinaryAnd:
    return builder_.create<circt::comb::AndOp>(loc, leftValue, rightValue);

// Conditional Operations
case slang::ast::ConditionalExpression:
    return builder_.create<circt::comb::MuxOp>(loc, conditionValue, 
        trueValue, falseValue);
```

## Testing and Validation

### Step 1: Build Verification
```bash
# Clean build with real CIRCT
rm -rf build
cmake -B build -DSV2SC_ENABLE_MLIR=ON
cmake --build build --target sv2sc -j$(nproc)
```

### Step 2: Functionality Testing
```bash
# Test MLIR pipeline
./build/sv2sc --use-mlir -top counter tests/examples/basic_counter/counter.sv

# Test MLIR diagnostics
./build/sv2sc --use-mlir --mlir-diagnostics -top counter counter.sv

# Test IR dumping
./build/sv2sc --use-mlir --dump-mlir -top counter counter.sv
```

### Step 3: Integration Testing
```bash
# Run existing test suite
ctest --test-dir build

# Run MLIR-specific tests
ctest --test-dir build -R mlir
```

## Performance Optimization

### Build Performance
- **Use Ninja**: Faster build system than Make
- **Enable ccache**: Caches compiled objects for faster rebuilds
- **Parallel builds**: Use `-j$(nproc)` for maximum parallelization
- **Link-time optimization**: Enable LTO for release builds

### Runtime Performance
- **Release builds**: Always use `CMAKE_BUILD_TYPE=Release` for production
- **LLVM optimizations**: Enable LLVM optimization passes in pass pipeline
- **Memory management**: Use MLIR's memory pool allocation patterns

## Troubleshooting

### Common Build Issues

**1. Missing CIRCT libraries:**
```bash
# Error: Could not find CIRCT libraries
# Solution: Verify CIRCT_DIR points to installation with lib/cmake/circt/
export CIRCT_DIR=/path/to/circt/installation
```

**2. Version mismatch:**
```bash
# Error: MLIR version mismatch with CIRCT
# Solution: Ensure LLVM, MLIR, and CIRCT are from compatible versions
# Recommended: Use same commit/release for all three
```

**3. Linking errors:**
```bash
# Error: Undefined symbols from CIRCT libraries
# Solution: Add missing CIRCT libraries to Dependencies.cmake
set(CIRCT_LIBRARIES
    CIRCTCombDialect      # Add this if missing
    CIRCTSeqDialect       # Add this if missing
    # ... other required dialects
)
```

### Runtime Issues

**1. Operation not registered:**
```bash
# Error: 'comb.add' op isn't registered
# Solution: Ensure dialect is loaded in MLIRContextManager
context->loadDialect<circt::comb::CombDialect>();
```

**2. Type conversion failures:**
```bash
# Error: Cannot convert SystemVerilog type to HW type
# Solution: Extend convertSystemVerilogTypeToHWType() method
# Check SystemVerilog type kind and map to appropriate CIRCT type
```

## Advanced Features (Future Enhancements)

### 1. Advanced Analysis Passes
```cpp
// Custom analysis passes for sv2sc
class SystemCOptimizationPass : public mlir::OperationPass<mlir::ModuleOp> {
    // Analyze and optimize SystemC-specific patterns
};
```

### 2. Formal Verification Integration
```cpp
// Integration with CIRCT's formal verification capabilities
class FormalVerificationPass : public mlir::OperationPass<mlir::ModuleOp> {
    // Generate assertions and verification conditions
};
```

### 3. High-Level Synthesis (HLS) Support
```cpp
// Transform high-level SystemVerilog to synthesizable RTL
class HLSLoweringPass : public mlir::OperationPass<mlir::ModuleOp> {
    // Lower high-level constructs to hardware primitives
};
```

## Conclusion

Phase 4 CIRCT integration transforms sv2sc from a proof-of-concept MLIR infrastructure into a production-ready hardware compiler leveraging the full power of the LLVM ecosystem. The careful architectural design in Phases 1-3 ensures that this transition requires minimal code changes while unlocking sophisticated hardware compilation capabilities.

**Key Benefits after CIRCT Integration:**
- **Industry-standard IR**: MLIR-based intermediate representation
- **Optimization infrastructure**: LLVM-style pass-based optimization
- **Extensible framework**: Easy addition of custom dialects and passes
- **Formal verification**: Integration with modern verification tools
- **Community ecosystem**: Access to LLVM/MLIR development community

The implementation provides a solid foundation for advanced hardware design automation while maintaining full compatibility with existing SystemVerilog workflows.