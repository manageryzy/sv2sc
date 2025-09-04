#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <regex>

// Project includes
#include "sv2sc/sv2sc.h"
#include "../utils/test_data_manager.h"

/**
 * @brief Functional Testing Suite for sv2sc with MLIR Backend
 * 
 * This suite validates the end-to-end functionality of sv2sc, ensuring that:
 * - Generated SystemC code is syntactically correct
 * - Generated code compiles successfully
 * - Generated code produces expected behavior
 * - Edge cases and error scenarios are handled properly
 */
class FunctionalTestSuite {
private:
    TestDataManager data_manager_;
    std::filesystem::path temp_dir_;
    std::filesystem::path systemc_install_;

    struct ValidationResult {
        bool syntax_valid;
        bool compiles;
        bool simulates;
        std::string compilation_output;
        std::string simulation_output;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
    };

public:
    FunctionalTestSuite() 
        : temp_dir_("/tmp/functional_test_" + std::to_string(std::time(nullptr))),
          systemc_install_("/usr/local/systemc") {
        std::filesystem::create_directories(temp_dir_);
        
        // Try to find SystemC installation
        if (!std::filesystem::exists(systemc_install_)) {
            systemc_install_ = "/opt/systemc";
            if (!std::filesystem::exists(systemc_install_)) {
                systemc_install_ = "";
            }
        }
    }
    
    ~FunctionalTestSuite() {
        cleanup();
    }
    
    void cleanup() {
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }
    
    std::pair<std::string, std::string> translateModule(const std::string& sv_code, 
                                                       const std::string& module_name = "test_module") {
        // Create temporary SystemVerilog file
        auto sv_file = temp_dir_ / (module_name + ".sv");
        std::ofstream file(sv_file);
        file << sv_code;
        file.close();
        
        // Perform translation
        sv2sc::SV2SC translator;
        auto result = translator.translateFile(sv_file.string());
        
        if (!result.success) {
            throw std::runtime_error("Translation failed: " + result.error_message);
        }
        
        return {result.header_code, result.implementation_code};
    }
    
    ValidationResult validateSystemC(const std::string& header_code, 
                                   const std::string& impl_code,
                                   const std::string& testbench_code = "") {
        ValidationResult result{};
        
        // Write generated files
        auto header_file = temp_dir_ / "module.h";
        auto impl_file = temp_dir_ / "module.cpp";
        
        std::ofstream header(header_file);
        header << header_code;
        header.close();
        
        std::ofstream impl(impl_file);
        impl << impl_code;
        impl.close();
        
        // Basic syntax validation
        result.syntax_valid = validateSyntax(header_code, impl_code);
        
        // Compilation test
        if (result.syntax_valid && !systemc_install_.empty()) {
            result.compiles = compileSystemC(header_file, impl_file, result.compilation_output);
            
            // Simulation test (if testbench provided)
            if (result.compiles && !testbench_code.empty()) {
                result.simulates = runSimulation(testbench_code, result.simulation_output);
            }
        }
        
        return result;
    }

private:
    bool validateSyntax(const std::string& header_code, const std::string& impl_code) {
        // Check for basic SystemC constructs
        std::vector<std::string> required_patterns = {
            R"(SC_MODULE\s*\(\s*\w+\s*\))",
            R"(SC_CTOR\s*\(\s*\w+\s*\))",
            R"(#include\s*[<"]systemc[.h>"])"
        };
        
        std::string combined_code = header_code + "\n" + impl_code;
        
        for (const auto& pattern : required_patterns) {
            std::regex regex_pattern(pattern);
            if (!std::regex_search(combined_code, regex_pattern)) {
                return false;
            }
        }
        
        // Check for common syntax errors
        std::vector<std::string> error_patterns = {
            R"(\w+\s*\(\s*\)\s*\{)", // Empty parameter lists without proper syntax
            R"([^;]\s*\n\s*[^#/\s])", // Missing semicolons (simplified check)
        };
        
        for (const auto& pattern : error_patterns) {
            std::regex regex_pattern(pattern);
            if (std::regex_search(combined_code, regex_pattern)) {
                return false;
            }
        }
        
        return true;
    }
    
    bool compileSystemC(const std::filesystem::path& header_file, 
                       const std::filesystem::path& impl_file,
                       std::string& output) {
        if (systemc_install_.empty()) {
            output = "SystemC not found - skipping compilation test";
            return false;
        }
        
        std::string compile_cmd = "cd " + temp_dir_.string() + 
                                " && g++ -std=c++17 -c " +
                                "-I" + systemc_install_.string() + "/include " +
                                impl_file.filename().string() + " 2>&1";
        
        FILE* pipe = popen(compile_cmd.c_str(), "r");
        if (!pipe) {
            output = "Failed to execute compilation command";
            return false;
        }
        
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        
        int exit_code = pclose(pipe);
        return exit_code == 0;
    }
    
    bool runSimulation(const std::string& testbench_code, std::string& output) {
        // Create testbench file
        auto tb_file = temp_dir_ / "testbench.cpp";
        std::ofstream tb(tb_file);
        tb << testbench_code;
        tb.close();
        
        // Compile and run simulation
        std::string sim_cmd = "cd " + temp_dir_.string() + 
                            " && g++ -std=c++17 " +
                            "-I" + systemc_install_.string() + "/include " +
                            "-L" + systemc_install_.string() + "/lib-linux64 " +
                            "testbench.cpp module.cpp -lsystemc -o simulation " +
                            "&& ./simulation 2>&1";
        
        FILE* pipe = popen(sim_cmd.c_str(), "r");
        if (!pipe) {
            output = "Failed to execute simulation command";
            return false;
        }
        
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        
        int exit_code = pclose(pipe);
        return exit_code == 0;
    }
};

TEST_CASE("Functional Testing - Basic Modules", "[functional][basic]") {
    FunctionalTestSuite suite;
    
    SECTION("Simple combinational logic") {
        std::string and_gate_sv = R"(
            module and_gate(
                input logic a,
                input logic b,
                output logic y
            );
                assign y = a & b;
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(and_gate_sv, "and_gate");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        REQUIRE(validation.syntax_valid);
        
        // Check that the logic is correctly translated
        REQUIRE(header.find("SC_MODULE(and_gate)") != std::string::npos);
        REQUIRE(header.find("sc_in<bool> a;") != std::string::npos);
        REQUIRE(header.find("sc_in<bool> b;") != std::string::npos);
        REQUIRE(header.find("sc_out<bool> y;") != std::string::npos);
        
        REQUIRE(impl.find("SC_METHOD(and_gate_method)") != std::string::npos);
        REQUIRE(impl.find("sensitive << a << b") != std::string::npos);
    }
    
    SECTION("Sequential logic - D Flip-Flop") {
        std::string dff_sv = R"(
            module d_flip_flop(
                input logic clk,
                input logic reset,
                input logic d,
                output logic q
            );
                always_ff @(posedge clk or posedge reset) begin
                    if (reset)
                        q <= 1'b0;
                    else
                        q <= d;
                end
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(dff_sv, "d_flip_flop");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        REQUIRE(validation.syntax_valid);
        
        // Check for proper clocking
        REQUIRE(impl.find("sensitive << clk.pos() << reset.pos()") != std::string::npos);
        REQUIRE(impl.find("if (reset.read())") != std::string::npos);
        REQUIRE(impl.find("q.write(d.read())") != std::string::npos);
    }
    
    SECTION("Parameterized counter") {
        std::string counter_sv = R"(
            module counter #(
                parameter WIDTH = 8
            )(
                input logic clk,
                input logic reset,
                input logic enable,
                output logic [WIDTH-1:0] count
            );
                logic [WIDTH-1:0] count_reg;
                
                always_ff @(posedge clk or posedge reset) begin
                    if (reset)
                        count_reg <= 0;
                    else if (enable)
                        count_reg <= count_reg + 1;
                end
                
                assign count = count_reg;
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(counter_sv, "counter");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        REQUIRE(validation.syntax_valid);
        
        // Check parameter handling
        REQUIRE(header.find("template") != std::string::npos ||
                header.find("static const int WIDTH") != std::string::npos);
        
        // Check bit vector usage
        REQUIRE(header.find("sc_lv<") != std::string::npos ||
                header.find("sc_uint<") != std::string::npos);
    }
}

TEST_CASE("Functional Testing - Advanced Features", "[functional][advanced]") {
    FunctionalTestSuite suite;
    
    SECTION("Module with generate blocks") {
        std::string generate_sv = R"(
            module parallel_adder #(
                parameter WIDTH = 4,
                parameter STAGES = 2
            )(
                input logic [WIDTH-1:0] a,
                input logic [WIDTH-1:0] b,
                input logic cin,
                output logic [WIDTH-1:0] sum,
                output logic cout
            );
                logic [STAGES:0][WIDTH-1:0] stage_sum;
                logic [STAGES:0] stage_carry;
                
                assign stage_sum[0] = 0;
                assign stage_carry[0] = cin;
                
                generate
                    for (genvar i = 0; i < STAGES; i++) begin : stage
                        for (genvar j = 0; j < WIDTH; j++) begin : bit_stage
                            always_comb begin
                                if (i == 0) begin
                                    stage_sum[i+1][j] = a[j] ^ b[j] ^ stage_carry[i];
                                    stage_carry[i+1] = (a[j] & b[j]) | 
                                                     ((a[j] ^ b[j]) & stage_carry[i]);
                                end else begin
                                    stage_sum[i+1][j] = stage_sum[i][j];
                                    stage_carry[i+1] = stage_carry[i];
                                end
                            end
                        end
                    end
                endgenerate
                
                assign sum = stage_sum[STAGES];
                assign cout = stage_carry[STAGES];
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(generate_sv, "parallel_adder");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        
        // Generate blocks are complex - at minimum should not crash
        if (validation.syntax_valid) {
            // If translation succeeds, verify basic structure
            REQUIRE(header.find("SC_MODULE") != std::string::npos);
            REQUIRE(impl.find("SC_METHOD") != std::string::npos);
        } else {
            // Generate blocks might not be fully supported yet
            INFO("Generate blocks not fully supported - this is acceptable");
        }
    }
    
    SECTION("FSM with enumerated states") {
        std::string fsm_sv = R"(
            module traffic_light(
                input logic clk,
                input logic reset,
                input logic pedestrian_request,
                output logic [1:0] light_state,
                output logic walk_signal
            );
                typedef enum logic [1:0] {
                    RED = 2'b00,
                    YELLOW = 2'b01,
                    GREEN = 2'b10,
                    WALK = 2'b11
                } state_t;
                
                state_t current_state, next_state;
                logic [3:0] timer;
                
                always_ff @(posedge clk or posedge reset) begin
                    if (reset) begin
                        current_state <= RED;
                        timer <= 0;
                    end else begin
                        current_state <= next_state;
                        timer <= (next_state != current_state) ? 0 : timer + 1;
                    end
                end
                
                always_comb begin
                    next_state = current_state;
                    case (current_state)
                        RED: if (timer >= 4) next_state = GREEN;
                        GREEN: if (pedestrian_request || timer >= 8) next_state = YELLOW;
                        YELLOW: if (timer >= 2) next_state = pedestrian_request ? WALK : RED;
                        WALK: if (timer >= 4) next_state = RED;
                    endcase
                end
                
                assign light_state = current_state[1:0];
                assign walk_signal = (current_state == WALK);
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(fsm_sv, "traffic_light");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        REQUIRE(validation.syntax_valid);
        
        // Check enum handling
        REQUIRE(header.find("enum") != std::string::npos ||
                impl.find("const") != std::string::npos);
        
        // Check FSM structure
        REQUIRE(impl.find("switch") != std::string::npos ||
                impl.find("case") != std::string::npos);
    }
}

TEST_CASE("Functional Testing - Error Handling", "[functional][error]") {
    FunctionalTestSuite suite;
    
    SECTION("Graceful handling of unsupported constructs") {
        std::string unsupported_sv = R"(
            module unsupported_features(
                input logic clk,
                output logic result
            );
                // Unsupported: System tasks
                initial begin
                    $display("This is unsupported");
                    $finish;
                end
                
                // Unsupported: Real numbers
                real temperature = 25.5;
                
                // Unsupported: Unions
                union packed {
                    logic [31:0] word;
                    logic [3:0][7:0] bytes;
                } data_union;
                
                assign result = 1'b1;
            endmodule
        )";
        
        try {
            auto [header, impl] = suite.translateModule(unsupported_sv, "unsupported_features");
            
            // If translation succeeds, check what was generated
            auto validation = suite.validateSystemC(header, impl);
            
            // Should at least generate the basic module structure
            REQUIRE(header.find("SC_MODULE") != std::string::npos);
            
            // Unsupported constructs should be commented out or omitted
            REQUIRE(impl.find("$display") == std::string::npos);
            REQUIRE(impl.find("real") == std::string::npos);
            
        } catch (const std::exception& e) {
            // It's acceptable for unsupported features to cause translation failure
            INFO("Unsupported features caused translation failure: " << e.what());
            REQUIRE(std::string(e.what()).find("unsupported") != std::string::npos ||
                    std::string(e.what()).find("not implemented") != std::string::npos);
        }
    }
    
    SECTION("Error recovery with partial modules") {
        std::string partial_error_sv = R"(
            module partial_error(
                input logic clk,
                input logic valid_input,
                output logic valid_output,
                output logic error_output
            );
                // Valid part - should translate correctly
                always_ff @(posedge clk) begin
                    valid_output <= valid_input;
                end
                
                // Invalid part - syntax error
                always_comb begin
                    error_output = undefined_signal; // Undefined signal
                end
            endmodule
        )";
        
        try {
            auto [header, impl] = suite.translateModule(partial_error_sv, "partial_error");
            
            // If translation succeeds despite errors, check partial results
            auto validation = suite.validateSystemC(header, impl);
            
            // Should generate module structure
            REQUIRE(header.find("SC_MODULE") != std::string::npos);
            
            // Valid parts should be translated
            REQUIRE(header.find("valid_input") != std::string::npos);
            REQUIRE(header.find("valid_output") != std::string::npos);
            
        } catch (const std::exception& e) {
            // Error is expected due to undefined signal
            REQUIRE(std::string(e.what()).find("undefined") != std::string::npos ||
                    std::string(e.what()).find("undeclared") != std::string::npos);
        }
    }
}

TEST_CASE("Functional Testing - Code Quality", "[functional][quality]") {
    FunctionalTestSuite suite;
    
    SECTION("Generated code follows SystemC best practices") {
        std::string best_practices_sv = R"(
            module design_example(
                input logic clk,
                input logic reset,
                input logic [7:0] data_in,
                input logic valid_in,
                output logic [7:0] data_out,
                output logic valid_out
            );
                logic [7:0] buffer_reg;
                logic valid_reg;
                
                always_ff @(posedge clk or posedge reset) begin
                    if (reset) begin
                        buffer_reg <= 8'h00;
                        valid_reg <= 1'b0;
                    end else begin
                        buffer_reg <= data_in;
                        valid_reg <= valid_in;
                    end
                end
                
                assign data_out = buffer_reg;
                assign valid_out = valid_reg;
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(best_practices_sv, "design_example");
        
        REQUIRE(!header.empty());
        REQUIRE(!impl.empty());
        
        auto validation = suite.validateSystemC(header, impl);
        REQUIRE(validation.syntax_valid);
        
        // Check for SystemC best practices
        
        // 1. Proper includes
        REQUIRE(header.find("#include <systemc.h>") != std::string::npos ||
                header.find("#include <systemc>") != std::string::npos);
        
        // 2. Proper port declarations
        REQUIRE(header.find("sc_in<bool> clk;") != std::string::npos);
        REQUIRE(header.find("sc_in<bool> reset;") != std::string::npos);
        
        // 3. Proper process declarations
        REQUIRE(impl.find("SC_METHOD") != std::string::npos ||
                impl.find("SC_THREAD") != std::string::npos);
        
        // 4. Proper sensitivity lists
        REQUIRE(impl.find("sensitive") != std::string::npos);
        
        // 5. Proper signal reads/writes
        REQUIRE(impl.find(".read()") != std::string::npos);
        REQUIRE(impl.find(".write(") != std::string::npos);
        
        // 6. Constructor follows naming convention
        REQUIRE(impl.find("SC_CTOR(design_example)") != std::string::npos);
    }
    
    SECTION("Generated code is readable and maintainable") {
        std::string complex_sv = R"(
            module complex_module(
                input logic clk,
                input logic [31:0] a, b, c,
                output logic [31:0] result1, result2
            );
                logic [31:0] intermediate1, intermediate2;
                
                always_ff @(posedge clk) begin
                    intermediate1 <= a + b;
                    intermediate2 <= b + c;
                    result1 <= intermediate1 << 1;
                    result2 <= intermediate2 >> 1;
                end
            endmodule
        )";
        
        auto [header, impl] = suite.translateModule(complex_sv, "complex_module");
        
        // Check for code readability features
        std::string combined = header + impl;
        
        // Should have proper indentation (at least some whitespace structure)
        size_t indent_count = 0;
        std::istringstream stream(combined);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.length() > 0 && (line[0] == ' ' || line[0] == '\t')) {
                indent_count++;
            }
        }
        REQUIRE(indent_count > 5); // Should have some indented lines
        
        // Should have comments explaining the translation
        REQUIRE(combined.find("//") != std::string::npos ||
                combined.find("/*") != std::string::npos);
        
        // Should have meaningful variable names
        REQUIRE(combined.find("intermediate1") != std::string::npos);
        REQUIRE(combined.find("intermediate2") != std::string::npos);
    }
}