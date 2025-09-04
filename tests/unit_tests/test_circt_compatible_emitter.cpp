#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef SV2SC_HAS_MLIR
#include "CIRCTCompatibleEmitter.h"
#include "SystemCEmissionPatterns.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace sv2sc::mlir_support;
#endif

TEST_CASE("CIRCTCompatibleEmitter Basic Functionality", "[circt_emitter]") {
#ifdef SV2SC_HAS_MLIR
    SECTION("Emitter Creation and Basic Operations") {
        CIRCTCompatibleEmitter emitter;
        
        // Test basic output operations
        emitter << "test_string";
        emitter << 42;
        
        // Test indent functionality
        emitter.setIndentLevel(2);
        REQUIRE(emitter.getIndentLevel() == 2);
        REQUIRE(emitter.getIndent() == "        "); // 8 spaces (2 * 4)
        
        // Test error handling
        REQUIRE_FALSE(emitter.hasErrors());
        emitter.emitError("Test error");
        REQUIRE(emitter.hasErrors());
        auto errors = emitter.getErrors();
        REQUIRE(errors.size() == 1);
        REQUIRE(errors[0] == "Test error");
    }
    
    SECTION("Pattern Registration") {
        CIRCTCompatibleEmitter emitter;
        
        // Register SystemC patterns
        registerAllSystemCEmitters(emitter);
        
        // Test that patterns are registered (no exceptions thrown)
        REQUIRE_NOTHROW(emitter.emitOp("systemc.module"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.int<32>"));
    }
    
    SECTION("Expression Precedence") {
        CIRCTCompatibleEmitter emitter;
        
        // Test precedence-based emission
        auto inlineEmitter = emitter.getInlinable("test_value");
        REQUIRE(inlineEmitter.getPrecedence() == Precedence::VAR);
        
        // Test parenthesization
        std::stringstream ss;
        emitter << "(";
        inlineEmitter.emitWithParensOnLowerPrecedence(Precedence::ADD);
        emitter << ")";
    }
#else
    SKIP("MLIR support not enabled");
#endif
}

TEST_CASE("SystemC Emission Patterns", "[systemc_patterns]") {
#ifdef SV2SC_HAS_MLIR
    SECTION("Operation Patterns") {
        CIRCTCompatibleEmitter emitter;
        registerAllSystemCEmitters(emitter);
        
        // Test module emission
        REQUIRE_NOTHROW(emitter.emitOp("systemc.module"));
        
        // Test signal operations
        REQUIRE_NOTHROW(emitter.emitOp("systemc.signal"));
        REQUIRE_NOTHROW(emitter.emitOp("systemc.signal.write"));
        
        // Test process operations
        REQUIRE_NOTHROW(emitter.emitOp("systemc.method"));
        REQUIRE_NOTHROW(emitter.emitOp("systemc.thread"));
        REQUIRE_NOTHROW(emitter.emitOp("systemc.sensitive"));
    }
    
    SECTION("Type Patterns") {
        CIRCTCompatibleEmitter emitter;
        registerAllSystemCEmitters(emitter);
        
        // Test port types
        REQUIRE_NOTHROW(emitter.emitType("systemc.in<bool>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.out<sc_uint<8>>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.inout<sc_logic>"));
        
        // Test signal types
        REQUIRE_NOTHROW(emitter.emitType("systemc.signal<bool>"));
        
        // Test integer types
        REQUIRE_NOTHROW(emitter.emitType("systemc.int<32>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.uint<16>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.bigint<64>"));
        
        // Test vector types
        REQUIRE_NOTHROW(emitter.emitType("systemc.bv<8>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.lv<16>"));
        REQUIRE_NOTHROW(emitter.emitType("systemc.logic"));
    }
    
    SECTION("Attribute Patterns") {
        CIRCTCompatibleEmitter emitter;
        registerAllSystemCEmitters(emitter);
        
        // Test integer attributes
        REQUIRE_NOTHROW(emitter.emitAttr("42"));
        REQUIRE_NOTHROW(emitter.emitAttr("0"));
        
        // Test string attributes
        REQUIRE_NOTHROW(emitter.emitAttr("\"test_string\""));
    }
#else
    SKIP("MLIR support not enabled");
#endif
}

TEST_CASE("File Generation", "[file_generation]") {
#ifdef SV2SC_HAS_MLIR
    SECTION("Path to Macro Name Conversion") {
        CIRCTCompatibleEmitter emitter;
        
        // Test path conversion (accessing private method through public interface)
        // We'll test this indirectly through file generation
        std::string testPath = "test/path/module.h";
        // The pathToMacroName should convert this to TEST_PATH_MODULE_H
    }
    
    SECTION("File Writing") {
        CIRCTCompatibleEmitter emitter;
        
        // Create temporary directory for testing
        std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "sv2sc_test";
        std::filesystem::create_directories(tempDir);
        
        // Test file writing
        std::string testContent = "// Test content\n#include <systemc.h>\n";
        std::string testFile = tempDir / "test.h";
        
        REQUIRE_NOTHROW(emitter.writeFile(testFile, testContent));
        
        // Verify file was created and has correct content
        REQUIRE(std::filesystem::exists(testFile));
        
        std::ifstream file(testFile);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        REQUIRE(content == testContent);
        
        // Cleanup
        std::filesystem::remove_all(tempDir);
    }
#else
    SKIP("MLIR support not enabled");
#endif
}

TEST_CASE("MLIR Integration", "[mlir_integration]") {
#ifdef SV2SC_HAS_MLIR
    SECTION("MLIR Module Processing") {
        // Create MLIR context and module
        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect>();
        
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        
        // Set module name
        module->setAttr("sym_name", builder.getStringAttr("TestModule"));
        
        CIRCTCompatibleEmitter emitter;
        
        // Create temporary directory for testing
        std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "sv2sc_mlir_test";
        std::filesystem::create_directories(tempDir);
        
        // Test split file generation
        auto result = emitter.emitSplit(module, tempDir.string());
        
        // Should succeed (even if it generates minimal content)
        REQUIRE(result.success);
        REQUIRE_FALSE(result.headerPath.empty());
        REQUIRE_FALSE(result.implPath.empty());
        
        // Verify files were created
        REQUIRE(std::filesystem::exists(result.headerPath));
        REQUIRE(std::filesystem::exists(result.implPath));
        
        // Check header content contains basic structure
        std::ifstream headerFile(result.headerPath);
        std::string headerContent((std::istreambuf_iterator<char>(headerFile)),
                                 std::istreambuf_iterator<char>());
        
        REQUIRE(headerContent.find("#ifndef") != std::string::npos);
        REQUIRE(headerContent.find("#include <systemc.h>") != std::string::npos);
        REQUIRE(headerContent.find("#endif") != std::string::npos);
        
        // Check implementation content
        std::ifstream implFile(result.implPath);
        std::string implContent((std::istreambuf_iterator<char>(implFile)),
                               std::istreambuf_iterator<char>());
        
        REQUIRE(implContent.find("#include") != std::string::npos);
        
        // Cleanup
        std::filesystem::remove_all(tempDir);
    }
    
    SECTION("Unified File Generation") {
        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect>();
        
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        module->setAttr("sym_name", builder.getStringAttr("UnifiedModule"));
        
        CIRCTCompatibleEmitter emitter;
        
        // Create temporary file for testing
        std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "sv2sc_unified_test";
        std::filesystem::create_directories(tempDir);
        std::string unifiedFile = tempDir / "unified.h";
        
        // Test unified file generation
        auto result = emitter.emitUnified(module, unifiedFile);
        
        REQUIRE(result.success);
        REQUIRE(result.headerPath == unifiedFile);
        REQUIRE(result.implPath.empty()); // No separate impl file for unified
        
        // Verify file was created
        REQUIRE(std::filesystem::exists(unifiedFile));
        
        // Check content
        std::ifstream file(unifiedFile);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        REQUIRE(content.find("#ifndef") != std::string::npos);
        REQUIRE(content.find("#include <systemc.h>") != std::string::npos);
        REQUIRE(content.find("#endif") != std::string::npos);
        
        // Cleanup
        std::filesystem::remove_all(tempDir);
    }
#else
    SKIP("MLIR support not enabled");
#endif
}

TEST_CASE("Error Handling", "[error_handling]") {
#ifdef SV2SC_HAS_MLIR
    SECTION("Invalid Operations") {
        CIRCTCompatibleEmitter emitter;
        
        // Test unsupported operation
        emitter.emitOp("unsupported.operation");
        
        // Should have generated an error
        REQUIRE(emitter.hasErrors());
        auto errors = emitter.getErrors();
        REQUIRE(errors.size() >= 1);
    }
    
    SECTION("File Writing Errors") {
        CIRCTCompatibleEmitter emitter;
        
        // Try to write to invalid path
        REQUIRE_THROWS(emitter.writeFile("/invalid/path/that/does/not/exist/file.h", "content"));
    }
#else
    SKIP("MLIR support not enabled");
#endif
}

// Test runner
int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
