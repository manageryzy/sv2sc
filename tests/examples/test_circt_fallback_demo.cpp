//===- test_circt_fallback_demo.cpp - CIRCT Fallback Demo Test -----------===//
//
// This file demonstrates the CIRCT-compatible fallback generator in action
// with real SystemVerilog to SystemC translation examples.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <filesystem>
#include <fstream>

#ifdef SV2SC_HAS_MLIR
#include "mlir/systemc/CIRCTCompatibleEmitter.h"
#include "mlir/systemc/SystemCEmissionPatterns.h"
#include "mlir/systemc/SystemCEmitter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#endif

using namespace sv2sc::mlir_support;

void demonstrateCIRCTFallback() {
#ifdef SV2SC_HAS_MLIR
    std::cout << "=== CIRCT-Compatible Fallback Generator Demo ===" << std::endl;
    
    // Create MLIR context and basic module
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    module->setAttr("sym_name", builder.getStringAttr("DemoCounter"));
    
    // Add some basic operations to simulate SystemC content
    auto funcType = builder.getFunctionType({}, {});
    auto clockFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "clock_process", funcType);
    auto resetFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "reset_process", funcType);
    
    module.push_back(clockFunc);
    module.push_back(resetFunc);
    
    // Test 1: Direct CIRCT-Compatible Emitter
    std::cout << "\n1. Testing Direct CIRCT-Compatible Emitter:" << std::endl;
    
    CIRCTCompatibleEmitter directEmitter;
    registerAllSystemCEmitters(directEmitter);
    
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "sv2sc_demo_direct";
    std::filesystem::create_directories(tempDir);
    
    auto directResult = directEmitter.emitSplit(module, tempDir.string());
    
    if (directResult.success) {
        std::cout << "✅ Direct emitter succeeded!" << std::endl;
        std::cout << "   Header: " << directResult.headerPath << std::endl;
        std::cout << "   Implementation: " << directResult.implPath << std::endl;
        
        // Show generated header content
        std::ifstream headerFile(directResult.headerPath);
        std::string headerContent((std::istreambuf_iterator<char>(headerFile)),
                                 std::istreambuf_iterator<char>());
        
        std::cout << "\n   Generated Header Content (first 500 chars):" << std::endl;
        std::cout << "   " << std::string(50, '-') << std::endl;
        std::cout << headerContent.substr(0, 500) << std::endl;
        if (headerContent.size() > 500) std::cout << "   ... (truncated)" << std::endl;
        std::cout << "   " << std::string(50, '-') << std::endl;
    } else {
        std::cout << "❌ Direct emitter failed: " << directResult.error << std::endl;
    }
    
    // Test 2: Integrated SystemCEmitter (with fallback)
    std::cout << "\n2. Testing Integrated SystemCEmitter (with fallback):" << std::endl;
    
    SystemCEmitter integratedEmitter;
    
    std::filesystem::path tempDir2 = std::filesystem::temp_directory_path() / "sv2sc_demo_integrated";
    std::filesystem::create_directories(tempDir2);
    
    auto integratedResult = integratedEmitter.emitSplit(module, tempDir2.string());
    
    if (integratedResult.success) {
        std::cout << "✅ Integrated emitter succeeded!" << std::endl;
        std::cout << "   Header: " << integratedResult.headerPath << std::endl;
        std::cout << "   Implementation: " << integratedResult.implPath << std::endl;
        
        // Show generated implementation content
        std::ifstream implFile(integratedResult.implPath);
        std::string implContent((std::istreambuf_iterator<char>(implFile)),
                               std::istreambuf_iterator<char>());
        
        std::cout << "\n   Generated Implementation Content:" << std::endl;
        std::cout << "   " << std::string(50, '-') << std::endl;
        std::cout << implContent << std::endl;
        std::cout << "   " << std::string(50, '-') << std::endl;
    } else {
        std::cout << "❌ Integrated emitter failed: " << integratedResult.error << std::endl;
    }
    
    // Test 3: Pattern System Demonstration
    std::cout << "\n3. Testing Pattern System:" << std::endl;
    
    CIRCTCompatibleEmitter patternEmitter;
    
    // Test individual patterns
    std::cout << "   Testing operation patterns:" << std::endl;
    patternEmitter.emitOp("systemc.module");
    patternEmitter.emitOp("systemc.signal.write");
    patternEmitter.emitOp("systemc.method");
    
    std::cout << "   Testing type patterns:" << std::endl;
    patternEmitter.emitType("systemc.in<bool>");
    patternEmitter.emitType("systemc.uint<32>");
    patternEmitter.emitType("systemc.signal<sc_logic>");
    
    std::cout << "   Testing attribute patterns:" << std::endl;
    patternEmitter.emitAttr("42");
    patternEmitter.emitAttr("\"test_string\"");
    
    if (patternEmitter.hasErrors()) {
        std::cout << "   ⚠️  Some patterns generated errors:" << std::endl;
        for (const auto& error : patternEmitter.getErrors()) {
            std::cout << "      - " << error << std::endl;
        }
    } else {
        std::cout << "   ✅ All patterns executed without errors!" << std::endl;
    }
    
    // Test 4: Unified File Generation
    std::cout << "\n4. Testing Unified File Generation:" << std::endl;
    
    std::filesystem::path unifiedFile = std::filesystem::temp_directory_path() / "sv2sc_demo_unified.h";
    
    auto unifiedResult = directEmitter.emitUnified(module, unifiedFile.string());
    
    if (unifiedResult.success) {
        std::cout << "✅ Unified file generation succeeded!" << std::endl;
        std::cout << "   File: " << unifiedResult.headerPath << std::endl;
        
        // Show file size
        auto fileSize = std::filesystem::file_size(unifiedFile);
        std::cout << "   Size: " << fileSize << " bytes" << std::endl;
    } else {
        std::cout << "❌ Unified file generation failed: " << unifiedResult.error << std::endl;
    }
    
    // Test 5: Error Handling
    std::cout << "\n5. Testing Error Handling:" << std::endl;
    
    CIRCTCompatibleEmitter errorEmitter;
    
    // Test invalid operation
    errorEmitter.emitOp("invalid.operation");
    errorEmitter.emitType("invalid.type");
    
    if (errorEmitter.hasErrors()) {
        std::cout << "✅ Error handling working correctly!" << std::endl;
        std::cout << "   Captured " << errorEmitter.getErrors().size() << " errors:" << std::endl;
        for (const auto& error : errorEmitter.getErrors()) {
            std::cout << "      - " << error << std::endl;
        }
    } else {
        std::cout << "❌ Error handling not working as expected" << std::endl;
    }
    
    // Cleanup
    std::cout << "\n6. Cleaning up test files..." << std::endl;
    try {
        std::filesystem::remove_all(tempDir);
        std::filesystem::remove_all(tempDir2);
        std::filesystem::remove(unifiedFile);
        std::cout << "✅ Cleanup completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠️  Cleanup warning: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== Demo Completed Successfully! ===" << std::endl;
    
#else
    std::cout << "MLIR support not enabled - demo skipped" << std::endl;
#endif
}

int main() {
    try {
        demonstrateCIRCTFallback();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Demo failed with unknown exception" << std::endl;
        return 1;
    }
}

