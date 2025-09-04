# MLIR/CIRCT Integration Plan for sv2sc

## Overview

This document outlines the comprehensive plan to transform sv2sc from a direct string-based translator into a modern multi-level compiler using MLIR/CIRCT infrastructure. This architectural upgrade will enable advanced optimizations, formal verification integration, and access to the broader hardware compilation ecosystem.

## Motivation

The current sv2sc architecture has several limitations:
- **Tight Coupling**: Direct AST-to-SystemC string generation limits optimization opportunities
- **Limited Analysis**: No intermediate representation for sophisticated transformations
- **String-Based Generation**: Error-prone and difficult to maintain code generation
- **Pattern Recognition**: Heuristic-based conditional logic detection
- **Scalability Issues**: Direct translation doesn't scale to complex optimization passes

### MLIR/CIRCT Advantages

MLIR/CIRCT provides:
- **Battle-tested Infrastructure**: Proven compiler framework with robust pass management
- **IEEE 1666-2011 Compliance**: Mature SystemC dialect following standards
- **Ecosystem Integration**: Access to formal verification, cross-compilation, and research tools
- **Scalable Architecture**: Multi-level IR enabling sophisticated optimizations
- **Industry Alignment**: Built on mainstream LLVM adoption

## Architecture Overview

### Current Pipeline
```
SystemVerilog → slang AST → SVToSCVisitor → SystemCCodeGenerator → SystemC Code
```

### New MLIR Pipeline
```
SystemVerilog → slang AST → SVToHWBuilder → HW Dialect → HWToSystemC Pass → SystemC Dialect → SystemC Emission → SystemC Code
                              ↑               ↑                    ↑                     ↑
                          MLIR Level 4    MLIR Level 3         MLIR Level 2         MLIR Level 1
```

**Level 4**: Hardware-agnostic structural representation
**Level 3**: Hardware-specific but target-agnostic operations  
**Level 2**: SystemC-specific constructs and semantics
**Level 1**: Final code emission to C++ SystemC

## Implementation Plan

### Phase 1: Foundation Setup (Weeks 1-3)

#### 1.1 CIRCT/MLIR Dependencies Integration

**CMake Configuration**
```cmake
# Add MLIR/CIRCT dependency option
option(SV2SC_ENABLE_MLIR "Enable MLIR-based translation pipeline" OFF)

if(SV2SC_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)
    find_package(CIRCT REQUIRED CONFIG)
    
    # Link MLIR libraries
    target_link_libraries(sv2sc PRIVATE
        MLIRIR
        MLIRPass
        MLIRAnalysis
        MLIRTransforms
        MLIRSCF
        CIRCTHWDialect
        CIRCTSystemCDialect
        CIRCTEmitSystemC
    )
    
    target_compile_definitions(sv2sc PRIVATE SV2SC_HAS_MLIR)
endif()
```

**Deliverables:**
- Updated CMakeLists.txt with MLIR/CIRCT dependencies
- Feature flag system for gradual migration
- Build verification on development systems

#### 1.2 Create MLIR Infrastructure Layer

**Directory Structure:**
```
src/mlir/
├── MLIRContext.h/cpp           # MLIR context wrapper
├── passes/                     # Pass implementations
├── patterns/                   # Conversion patterns
├── pipeline/                   # Pass pipeline management
└── utils/                      # MLIR utility functions
```

**Core Classes:**
```cpp
class MLIRContextManager {
public:
    MLIRContextManager();
    mlir::MLIRContext& getContext();
    void loadRequiredDialects();
    
private:
    std::unique_ptr<mlir::MLIRContext> context_;
};

class SV2SCPassPipeline {
public:
    void buildPipeline(mlir::OpPassManager& pm);
    bool runPipeline(mlir::ModuleOp module);
};
```

**Deliverables:**
- MLIR context initialization and dialect loading
- Basic pass pipeline infrastructure
- Integration points with existing translator

#### 1.3 Implement SVToHW Builder Foundation

**Core Builder Class:**
```cpp
class SVToHWBuilder {
public:
    SVToHWBuilder(mlir::MLIRContext* context);
    mlir::ModuleOp buildFromAST(const slang::ast::InstanceBodySymbol& moduleAST);
    
private:
    mlir::hw::ModuleOp buildModule(const slang::ast::InstanceBodySymbol& moduleAST);
    mlir::Value buildExpression(const slang::ast::Expression& expr);
    void buildStatement(const slang::ast::Statement& stmt);
    
    mlir::MLIRContext* context_;
    mlir::OpBuilder builder_;
    mlir::hw::ModuleOp currentModule_;
};
```

**Deliverables:**
- Basic SystemVerilog AST to HW dialect conversion
- Module and port declaration handling
- Simple expression translation

### Phase 2: Core Translation (Weeks 4-6)

#### 2.1 Complete SystemVerilog to HW Dialect Translation

**Expression Handling:**
- Binary operations (+, -, &, |, ==, !=, <, >, etc.)
- Unary operations (~, !, +, -)
- Conditional expressions (ternary operator)
- Array indexing and bit selection
- Function calls and method invocations

**Statement Handling:**
- Always blocks (always_ff, always_comb, always)
- Assignment statements (blocking and non-blocking)
- Conditional statements (if-else)
- Generate blocks and loops
- Module instantiation

**Deliverables:**
- Complete expression conversion system
- All statement types converted to HW dialect
- Support for complex SystemVerilog constructs

#### 2.2 HW to SystemC Dialect Lowering

**Conversion Patterns:**
```cpp
class ModuleOpLowering : public mlir::ConversionPattern {
    mlir::LogicalResult matchAndRewrite(
        mlir::hw::HWModuleOp op, ArrayRef<mlir::Value> operands,
        mlir::ConversionPatternRewriter& rewriter) const override;
};

class RegisterToSignalPattern : public mlir::ConversionPattern {
    // Convert HW register operations to SystemC signals with processes
};

class AlwaysBlockPattern : public mlir::ConversionPattern {
    // Convert procedural blocks to SystemC methods
};
```

**Pass Implementation:**
```cpp
class HWToSystemCLoweringPass : public mlir::ConversionPass<mlir::hw::HWModuleOp> {
    void runOnOperation() override;
    void configureConversionTarget(mlir::ConversionTarget& target);
    void populateRewritePatterns(mlir::RewritePatternSet& patterns);
};
```

**Deliverables:**
- Complete HW to SystemC conversion patterns
- Process generation for sequential and combinational logic
- Port and signal type mapping

#### 2.3 SystemC Code Emission

**Emission Pass:**
```cpp
class SystemCEmissionPass : public mlir::Pass {
public:
    void runOnOperation() override;
    
private:
    void emitModule(mlir::systemc::SCModuleOp module);
    void writeOutputFiles();
    
    std::string headerBuffer_;
    std::string implBuffer_;
};
```

**Deliverables:**
- Integration with CIRCT's SystemC emission
- Output file generation matching existing format
- Verification against current test suite

### Phase 3: Analysis Infrastructure (Weeks 7-9)

#### 3.1 Type Analysis System

**Type Analyzer:**
```cpp
class SystemCTypeAnalysisPass : public mlir::Pass {
public:
    void runOnOperation() override;
    
private:
    void analyzeSignalUsage(mlir::hw::HWModuleOp module);
    void annotateWithSystemCTypes(mlir::hw::HWModuleOp module);
    SystemCDataType inferOptimalType(mlir::Value value);
};
```

**Analysis Results:**
- Arithmetic vs logic usage patterns
- Optimal SystemC type selection (sc_uint vs sc_lv)
- Parametric width optimization
- Clock and reset signal detection

**Deliverables:**
- Complete type analysis framework
- Type annotations for optimization
- Improved SystemC type selection

#### 3.2 Dependency and Sensitivity Analysis

**Dependency Analysis:**
```cpp
class SystemCDependencyAnalysisPass : public mlir::Pass {
public:
    void runOnOperation() override;
    
private:
    void buildDependencyGraph(mlir::hw::HWModuleOp module);
    void detectCombinationalLoops();
    void computeSensitivityLists();
};
```

**Sensitivity Optimization:**
```cpp
class SystemCSensitivityOptimizationPass : public mlir::Pass {
    // Generate optimal sensitivity lists
    // Remove redundant sensitivity entries
    // Optimize for simulation performance
};
```

**Deliverables:**
- Signal dependency graph construction
- Optimal sensitivity list generation
- Combinational loop detection

#### 3.3 Process Structure Optimization

**Process Analysis:**
```cpp
class SystemCProcessAnalysisPass : public mlir::Pass {
public:
    void runOnOperation() override;
    
private:
    void analyzeProcessComplexity();
    void planProcessSplitting();
    void identifyMergeOpportunities();
};
```

**Optimization Strategies:**
- Intelligent process splitting based on complexity
- Process merging for efficiency
- Clock domain separation
- Resource usage optimization

**Deliverables:**
- Process structure analysis
- Smart splitting algorithms
- Process optimization recommendations

### Phase 4: Advanced Optimizations (Weeks 10-11)

#### 4.1 Standard MLIR Optimizations

**Enabled Passes:**
- Constant folding and propagation
- Dead code elimination
- Common subexpression elimination
- Function inlining for small processes
- Canonicalization and simplification

**Integration:**
```cpp
void SV2SCPassPipeline::addStandardOptimizations(mlir::OpPassManager& pm) {
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(createConstantFoldingPass());
}
```

**Deliverables:**
- Integration of standard MLIR optimization passes
- SystemC-aware optimization configuration
- Performance improvements measurement

#### 4.2 SystemC-Specific Optimizations

**Custom Optimization Pass:**
```cpp
class SystemCOptimizationPass : public mlir::Pass {
public:
    void runOnOperation() override;
    
private:
    void optimizeSensitivityLists();
    void mergeCompatibleProcesses();
    void eliminateRedundantSignals();
    void optimizeClockDomains();
};
```

**Advanced Features:**
- Clock domain analysis and optimization
- Resource estimation for synthesis guidance
- Timing-aware optimizations
- Power optimization opportunities

**Deliverables:**
- SystemC-specific optimization suite
- Resource usage analysis
- Timing and power optimization

#### 4.3 Formal Verification Integration

**Property Insertion:**
```cpp
class FormalPropertyInsertionPass : public mlir::Pass {
    // Insert SystemVerilog assertions as MLIR operations
    // Enable formal verification tool integration
    // Support bounded model checking
};
```

**Verification Features:**
- SVA assertion conversion
- Property specification support
- Formal tool integration hooks
- Coverage point generation

**Deliverables:**
- Formal verification integration framework
- Assertion conversion system
- Verification tool compatibility

### Phase 5: Production Deployment (Week 12)

#### 5.1 Testing and Validation

**Test Requirements:**
- All existing regression tests pass with MLIR pipeline
- Performance benchmarking against legacy pipeline
- Output equivalence validation
- Large design scalability testing (PicoRV32)

**Quality Metrics:**
- ≤20% compilation time overhead
- Improved SystemC code quality
- No functionality regressions
- Successful complex design handling

**Deliverables:**
- Complete test suite validation
- Performance benchmarking results
- Quality improvement measurements

#### 5.2 Migration and Documentation

**Migration Strategy:**
- Feature flag for gradual adoption
- Legacy pipeline preservation for compatibility
- User migration guide and documentation
- Training materials for advanced features

**Documentation Updates:**
- User guide updates for MLIR features
- Developer documentation for MLIR infrastructure
- Optimization guide for users
- Troubleshooting and debugging guide

**Deliverables:**
- Updated user documentation
- Migration guide and tools
- Developer documentation
- Training materials

## Implementation Details

### Directory Structure

```
sv2sc/
├── src/
│   ├── mlir/
│   │   ├── MLIRContext.h/cpp
│   │   ├── SVToHWBuilder.h/cpp
│   │   ├── passes/
│   │   │   ├── HWToSystemCLoweringPass.h/cpp
│   │   │   ├── SystemCTypeAnalysisPass.h/cpp
│   │   │   ├── SystemCDependencyAnalysisPass.h/cpp
│   │   │   ├── SystemCProcessAnalysisPass.h/cpp
│   │   │   ├── SystemCOptimizationPass.h/cpp
│   │   │   └── SystemCEmissionPass.h/cpp
│   │   ├── patterns/
│   │   │   ├── RegisterToSignalPattern.h/cpp
│   │   │   ├── AlwaysBlockPattern.h/cpp
│   │   │   ├── ModulePattern.h/cpp
│   │   │   └── ExpressionPatterns.h/cpp
│   │   ├── pipeline/
│   │   │   └── SV2SCPassPipeline.h/cpp
│   │   └── utils/
│   │       ├── MLIRUtilities.h/cpp
│   │       └── TypeConversion.h/cpp
│   ├── core/
│   │   └── MLIRTranslator.h/cpp
│   └── main.cpp                    # Updated with MLIR option
├── tests/
│   ├── mlir/                      # MLIR-specific tests
│   │   ├── unit/                  # Unit tests for passes
│   │   ├── integration/           # End-to-end MLIR tests
│   │   └── regression/            # Regression tests
│   └── examples/                  # Updated with MLIR validation
└── docs/
    ├── MLIR_CIRCT_INTEGRATION.md  # This document
    ├── MLIR_USER_GUIDE.md         # User guide for MLIR features
    └── MLIR_DEVELOPER_GUIDE.md    # Developer documentation
```

### Integration Points

**Command Line Interface:**
```bash
# Enable MLIR pipeline
./sv2sc --use-mlir -top counter counter.sv

# Legacy pipeline (default during transition)
./sv2sc -top counter counter.sv

# MLIR with specific optimizations
./sv2sc --use-mlir --optimize-level 2 -top counter counter.sv

# Debug MLIR passes
./sv2sc --use-mlir --debug-pass-pipeline -top counter counter.sv
```

**Translation Options Extension:**
```cpp
struct TranslationOptions {
    // Existing options...
    
    // MLIR-specific options
    bool useMLIRPipeline = false;
    int optimizationLevel = 1;
    bool enableMLIRDiagnostics = false;
    std::string mlirPassPipeline;
    bool dumpMLIR = false;
};
```

## Risk Mitigation

### Technical Risks

1. **MLIR Learning Curve**
   - Mitigation: Gradual team training, external consultation
   - Timeline: Include learning buffer in early phases

2. **Performance Overhead**
   - Mitigation: Continuous benchmarking, optimization focus
   - Target: ≤20% compilation time increase

3. **Integration Complexity**  
   - Mitigation: Incremental integration, feature flags
   - Fallback: Preserve legacy pipeline

4. **CIRCT API Changes**
   - Mitigation: Pin to stable CIRCT version, track updates
   - Plan: Regular CIRCT version updates

### Project Risks

1. **Development Timeline**
   - Mitigation: Parallel development tracks, early validation
   - Buffer: Include 2-week buffer for each phase

2. **Resource Requirements**
   - Mitigation: Early MLIR expertise development
   - Support: External MLIR/CIRCT consulting if needed

3. **Compatibility Issues**
   - Mitigation: Comprehensive regression testing
   - Validation: Output equivalence verification

## Success Metrics

### Functional Metrics
- ✅ All existing tests pass with MLIR pipeline
- ✅ Complex designs (PicoRV32) translate successfully  
- ✅ Output SystemC code is functionally equivalent
- ✅ No regressions in supported SystemVerilog features

### Performance Metrics
- 📈 ≤20% compilation time overhead vs legacy pipeline
- 📈 Improved SystemC simulation performance
- 📈 Better resource utilization in generated code
- 📈 Reduced memory usage during translation

### Quality Metrics
- 🎯 Improved type selection (sc_uint vs sc_lv optimization)
- 🎯 Optimized sensitivity lists reduce simulation overhead
- 🎯 Better process structure for maintainability
- 🎯 Formal verification integration capabilities

### Strategic Metrics
- 🚀 Access to CIRCT ecosystem tools and optimizations
- 🚀 Foundation for future hardware compilation features
- 🚀 Industry alignment with LLVM/MLIR adoption
- 🚀 Research collaboration opportunities

## Next Steps

1. **Documentation Review**: Stakeholder review of this plan
2. **Environment Setup**: Install CIRCT/MLIR development dependencies
3. **Team Training**: MLIR fundamentals and CIRCT ecosystem overview
4. **Phase 1 Kickoff**: Begin Foundation Setup implementation
5. **Progress Tracking**: Establish weekly progress reviews and milestone tracking

## References

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [CIRCT Project](https://circt.llvm.org/)
- [SystemC Dialect Documentation](https://circt.llvm.org/docs/Dialects/SystemC/)
- [SystemC Dialect Rationale](https://circt.llvm.org/docs/Dialects/SystemC/RationaleSystemC/)
- [IEEE 1666-2011 SystemC Standard](https://standards.ieee.org/standard/1666-2011.html)

---

*This document serves as the master plan for MLIR/CIRCT integration into sv2sc. It will be updated as implementation progresses and new insights are gained.*