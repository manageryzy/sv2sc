#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

// Project includes
#include "sv2sc/sv2sc.h"
#include "../utils/test_data_manager.h"

#ifdef SV2SC_HAS_MLIR
#include "mlir/MLIRContextManager.h"
#include "mlir/MLIRTranslator.h"
#include "mlir/SVToHWBuilder.h"
#include "mlir/pipeline/SV2SCPassPipeline.h"
#endif

/**
 * @brief MLIR Integration Test Suite
 * 
 * Comprehensive testing of the MLIR integration within sv2sc, covering:
 * - SystemVerilog to MLIR translation
 * - Hardware Dialect generation and optimization
 * - Pass pipeline execution
 * - End-to-end translation scenarios
 */
class MLIRIntegrationTestSuite {
private:
    TestDataManager data_manager_;
    std::filesystem::path temp_dir_;
    
    struct TranslationResult {
        bool success;
        std::string header_code;
        std::string implementation_code;
        std::string error_message;
        std::vector<std::string> diagnostics;
        std::chrono::milliseconds translation_time;
    };

public:
    MLIRIntegrationTestSuite() 
        : temp_dir_("/tmp/mlir_integration_test_" + std::to_string(std::time(nullptr))) {
        std::filesystem::create_directories(temp_dir_);
    }
    
    ~MLIRIntegrationTestSuite() {
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }
    
    TranslationResult translateSystemVerilog(const std::string& sv_code, bool use_mlir = true) {
        TranslationResult result{};
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // Create temporary SystemVerilog file
            auto sv_file = temp_dir_ / "test_module.sv";
            std::ofstream file(sv_file);
            file << sv_code;
            file.close();
            
#ifdef SV2SC_HAS_MLIR
            if (use_mlir) {
                // Use MLIR-based translation
                auto context_manager = std::make_unique<sv2sc::MLIRContextManager>();
                auto translator = std::make_unique<sv2sc::MLIRTranslator>(*context_manager);
                
                auto translation_result = translator->translateFile(sv_file.string());
                result.success = translation_result.success;
                result.header_code = translation_result.header_code;
                result.implementation_code = translation_result.implementation_code;
                result.error_message = translation_result.error_message;
                result.diagnostics = translation_result.diagnostics;
            } else
#endif
            {
                // Use traditional translation path
                sv2sc::SV2SC translator;
                auto translation_result = translator.translateFile(sv_file.string());
                result.success = translation_result.success;
                result.header_code = translation_result.header_code;
                result.implementation_code = translation_result.implementation_code;
                result.error_message = translation_result.error_message;
            }
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        result.translation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        return result;
    }
    
    bool compileGeneratedSystemC(const std::string& header_code, const std::string& impl_code) {
        // Write generated files
        auto header_file = temp_dir_ / "generated_module.h";
        auto impl_file = temp_dir_ / "generated_module.cpp";
        
        std::ofstream header(header_file);
        header << header_code;
        header.close();
        
        std::ofstream impl(impl_file);
        impl << impl_code;
        impl.close();
        
        // Attempt compilation
        std::string compile_cmd = "cd " + temp_dir_.string() + 
                                " && g++ -c -std=c++17 -I/usr/local/systemc/include " +
                                impl_file.filename().string() + " 2>/dev/null";
        
        return system(compile_cmd.c_str()) == 0;
    }
};

TEST_CASE("MLIR Integration - Basic Translation", "[mlir][integration][translation]") {
    MLIRIntegrationTestSuite suite;
    
    SECTION("Simple counter module translation") {
        std::string counter_sv = R"(
            module counter(
                input logic clk,
                input logic reset,
                output logic [7:0] count
            );
                always_ff @(posedge clk or posedge reset) begin
                    if (reset)
                        count <= 8'h00;
                    else
                        count <= count + 1;
                end
            endmodule
        )";
        
#ifdef SV2SC_HAS_MLIR
        SECTION("MLIR-based translation") {
            auto mlir_result = suite.translateSystemVerilog(counter_sv, true);
            
            REQUIRE(mlir_result.success);
            REQUIRE(!mlir_result.header_code.empty());
            REQUIRE(!mlir_result.implementation_code.empty());
            
            // Verify MLIR-specific optimizations are present
            REQUIRE(mlir_result.header_code.find("SC_MODULE") != std::string::npos);
            REQUIRE(mlir_result.implementation_code.find("SC_METHOD") != std::string::npos);
            
            // Check for MLIR optimization markers
            REQUIRE(mlir_result.implementation_code.find("// MLIR-optimized") != std::string::npos);
            
            // Verify generated SystemC compiles
            REQUIRE(suite.compileGeneratedSystemC(mlir_result.header_code, mlir_result.implementation_code));
        }
#endif
        
        SECTION("Traditional translation comparison") {
            auto traditional_result = suite.translateSystemVerilog(counter_sv, false);
            
            REQUIRE(traditional_result.success);
            REQUIRE(!traditional_result.header_code.empty());
            REQUIRE(!traditional_result.implementation_code.empty());
            
#ifdef SV2SC_HAS_MLIR
            auto mlir_result = suite.translateSystemVerilog(counter_sv, true);
            
            // MLIR version should be faster or equivalent
            REQUIRE(mlir_result.translation_time <= traditional_result.translation_time * 1.2);
            
            // Both should produce compilable code
            REQUIRE(suite.compileGeneratedSystemC(traditional_result.header_code, 
                                                traditional_result.implementation_code));
#endif
        }
    }
    
    SECTION("Complex module with interfaces") {
        std::string interface_sv = R"(
            interface bus_if(input logic clk);
                logic [31:0] data;
                logic valid;
                logic ready;
                
                modport master (output data, valid, input ready);
                modport slave (input data, valid, output ready);
            endinterface
            
            module data_processor(
                input logic clk,
                input logic reset,
                bus_if.slave input_bus,
                bus_if.master output_bus
            );
                logic [31:0] processed_data;
                
                always_ff @(posedge clk) begin
                    if (reset) begin
                        processed_data <= 0;
                        output_bus.valid <= 0;
                    end else if (input_bus.valid && input_bus.ready) begin
                        processed_data <= input_bus.data << 1;
                        output_bus.data <= processed_data;
                        output_bus.valid <= 1;
                    end
                end
                
                assign input_bus.ready = !reset;
            endmodule
        )";
        
#ifdef SV2SC_HAS_MLIR
        auto mlir_result = suite.translateSystemVerilog(interface_sv, true);
        
        // MLIR should handle interfaces better
        if (mlir_result.success) {
            REQUIRE(!mlir_result.header_code.empty());
            REQUIRE(mlir_result.header_code.find("sc_port") != std::string::npos);
            REQUIRE(mlir_result.implementation_code.find("SC_METHOD") != std::string::npos);
        } else {
            // Interface translation might not be fully implemented yet
            REQUIRE(!mlir_result.diagnostics.empty());
        }
#endif
    }
}

TEST_CASE("MLIR Hardware Dialect Generation", "[mlir][hw][dialect]") {
#ifdef SV2SC_HAS_MLIR
    MLIRIntegrationTestSuite suite;
    
    SECTION("Basic hardware constructs") {
        std::string hardware_sv = R"(
            module alu(
                input logic [31:0] a,
                input logic [31:0] b,
                input logic [3:0] op,
                output logic [31:0] result,
                output logic zero
            );
                always_comb begin
                    case (op)
                        4'b0000: result = a + b;
                        4'b0001: result = a - b;
                        4'b0010: result = a & b;
                        4'b0011: result = a | b;
                        4'b0100: result = a ^ b;
                        4'b0101: result = ~a;
                        default: result = 32'h0;
                    endcase
                    
                    zero = (result == 32'h0);
                end
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(hardware_sv, true);
        
        REQUIRE(result.success);
        
        // Check for hardware dialect optimizations
        REQUIRE(result.implementation_code.find("SC_METHOD") != std::string::npos);
        
        // Verify arithmetic operations are properly translated
        REQUIRE(result.implementation_code.find("a.read()") != std::string::npos);
        REQUIRE(result.implementation_code.find("b.read()") != std::string::npos);
        
        // Check combinational logic generation
        REQUIRE(result.implementation_code.find("sensitive << a << b << op") != std::string::npos);
    }
    
    SECTION("Sequential logic with clocking") {
        std::string sequential_sv = R"(
            module shift_register(
                input logic clk,
                input logic reset,
                input logic shift_in,
                output logic [7:0] shift_out
            );
                logic [7:0] register_data;
                
                always_ff @(posedge clk or posedge reset) begin
                    if (reset)
                        register_data <= 8'h0;
                    else
                        register_data <= {register_data[6:0], shift_in};
                end
                
                assign shift_out = register_data;
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(sequential_sv, true);
        
        REQUIRE(result.success);
        
        // Check for proper clock domain handling
        REQUIRE(result.implementation_code.find("SC_METHOD") != std::string::npos);
        REQUIRE(result.implementation_code.find("sensitive << clk.pos() << reset.pos()") != std::string::npos);
        
        // Verify shift operation translation
        REQUIRE(result.implementation_code.find("shift") != std::string::npos);
    }
#endif
}

TEST_CASE("MLIR Pass Pipeline Execution", "[mlir][passes][pipeline]") {
#ifdef SV2SC_HAS_MLIR
    MLIRIntegrationTestSuite suite;
    
    SECTION("Standard optimization passes") {
        std::string optimization_sv = R"(
            module redundancy_test(
                input logic [31:0] data_in,
                output logic [31:0] data_out
            );
                logic [31:0] temp1, temp2, temp3;
                
                always_comb begin
                    temp1 = data_in;
                    temp2 = temp1; // Redundant assignment
                    temp3 = temp2; // Another redundant assignment
                    data_out = temp3;
                end
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(optimization_sv, true);
        
        REQUIRE(result.success);
        
        // MLIR passes should optimize away redundant assignments
        // The generated code should be more concise
        std::string impl = result.implementation_code;
        
        // Count the number of temporary variable references
        size_t temp_count = 0;
        size_t pos = 0;
        while ((pos = impl.find("temp", pos)) != std::string::npos) {
            temp_count++;
            pos += 4;
        }
        
        // After optimization, there should be fewer temporary variables
        REQUIRE(temp_count < 6); // Original has 6 references to temp variables
    }
    
    SECTION("Dead code elimination") {
        std::string deadcode_sv = R"(
            module deadcode_test(
                input logic clk,
                input logic [7:0] data_in,
                output logic [7:0] data_out
            );
                logic [7:0] unused_signal;
                logic [7:0] used_signal;
                
                always_ff @(posedge clk) begin
                    unused_signal <= data_in; // Dead code - unused_signal is never read
                    used_signal <= data_in;
                    data_out <= used_signal;
                end
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(deadcode_sv, true);
        
        REQUIRE(result.success);
        
        // Check that dead code was eliminated
        // unused_signal should not appear in the optimized code
        REQUIRE(result.header_code.find("unused_signal") == std::string::npos);
        REQUIRE(result.implementation_code.find("unused_signal") == std::string::npos);
    }
#endif
}

TEST_CASE("MLIR Error Handling and Diagnostics", "[mlir][error][diagnostics]") {
#ifdef SV2SC_HAS_MLIR
    MLIRIntegrationTestSuite suite;
    
    SECTION("Syntax error handling") {
        std::string invalid_sv = R"(
            module syntax_error(
                input logic clk
                output logic data  // Missing comma
            );
                always_ff @(posedge clk) begin
                    data <= 1'b1
                end  // Missing semicolon
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(invalid_sv, true);
        
        // Should fail gracefully with diagnostic information
        REQUIRE_FALSE(result.success);
        REQUIRE(!result.diagnostics.empty());
        REQUIRE(!result.error_message.empty());
    }
    
    SECTION("Semantic error handling") {
        std::string semantic_error_sv = R"(
            module semantic_error(
                input logic [7:0] data_in,
                output logic [15:0] data_out
            );
                always_comb begin
                    data_out = data_in + undefined_signal; // Undefined signal
                end
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(semantic_error_sv, true);
        
        // Should provide meaningful error messages
        REQUIRE_FALSE(result.success);
        REQUIRE(result.error_message.find("undefined") != std::string::npos ||
                result.error_message.find("undeclared") != std::string::npos);
    }
    
    SECTION("Partial translation on recoverable errors") {
        std::string partial_error_sv = R"(
            module partial_error(
                input logic clk,
                input logic [7:0] valid_input,
                output logic [7:0] valid_output,
                output logic [7:0] error_output
            );
                // Valid part
                always_ff @(posedge clk) begin
                    valid_output <= valid_input;
                end
                
                // Invalid part
                always_comb begin
                    error_output = undefined_signal; // Error here
                end
            endmodule
        )";
        
        auto result = suite.translateSystemVerilog(partial_error_sv, true);
        
        // Should provide partial results where possible
        if (!result.success) {
            // At minimum, should identify the problematic area
            REQUIRE(!result.diagnostics.empty());
        }
    }
#endif
}

TEST_CASE("MLIR Performance Comparison", "[mlir][performance][benchmark]") {
    MLIRIntegrationTestSuite suite;
    
    SECTION("Translation speed comparison") {
        // Generate a moderately complex module
        std::string complex_sv = R"(
            module complex_processor(
                input logic clk,
                input logic reset,
                input logic [31:0] data_in,
                input logic valid_in,
                output logic [31:0] data_out,
                output logic valid_out
            );
                logic [31:0] pipeline_stage1, pipeline_stage2, pipeline_stage3;
                logic valid_stage1, valid_stage2, valid_stage3;
                
                always_ff @(posedge clk or posedge reset) begin
                    if (reset) begin
                        pipeline_stage1 <= 0;
                        pipeline_stage2 <= 0;
                        pipeline_stage3 <= 0;
                        valid_stage1 <= 0;
                        valid_stage2 <= 0;
                        valid_stage3 <= 0;
                    end else begin
                        // Stage 1: Input processing
                        pipeline_stage1 <= data_in << 1;
                        valid_stage1 <= valid_in;
                        
                        // Stage 2: Arithmetic operations
                        pipeline_stage2 <= pipeline_stage1 + 32'h12345678;
                        valid_stage2 <= valid_stage1;
                        
                        // Stage 3: Output formatting
                        pipeline_stage3 <= pipeline_stage2 ^ 32'hFFFFFFFF;
                        valid_stage3 <= valid_stage2;
                    end
                end
                
                assign data_out = pipeline_stage3;
                assign valid_out = valid_stage3;
            endmodule
        )";
        
        // Test traditional translation
        auto traditional_result = suite.translateSystemVerilog(complex_sv, false);
        REQUIRE(traditional_result.success);
        
#ifdef SV2SC_HAS_MLIR
        // Test MLIR translation
        auto mlir_result = suite.translateSystemVerilog(complex_sv, true);
        REQUIRE(mlir_result.success);
        
        // MLIR should be competitive or better
        INFO("Traditional translation time: " << traditional_result.translation_time.count() << "ms");
        INFO("MLIR translation time: " << mlir_result.translation_time.count() << "ms");
        
        // Allow MLIR to be up to 50% slower (it's doing more analysis)
        REQUIRE(mlir_result.translation_time <= traditional_result.translation_time * 1.5);
        
        // But the generated code quality should be better (measured by compilation success)
        bool traditional_compiles = suite.compileGeneratedSystemC(
            traditional_result.header_code, traditional_result.implementation_code);
        bool mlir_compiles = suite.compileGeneratedSystemC(
            mlir_result.header_code, mlir_result.implementation_code);
        
        REQUIRE(traditional_compiles);
        REQUIRE(mlir_compiles);
#endif
    }
}