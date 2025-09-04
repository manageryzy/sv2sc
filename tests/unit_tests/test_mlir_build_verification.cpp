#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <system_error>

// Test data manager for build verification
#include "test_data_manager.h"

using sv2sc::testing::TestDataManager;

/**
 * @brief Build Verification Test Suite for MLIR-integrated sv2sc
 * 
 * This test suite verifies that the MLIR-integrated sv2sc project builds correctly
 * across different configurations and environments. It tests both clean builds and
 * incremental builds, verifying dependency resolution and linking.
 */
class MLIRBuildVerificationSuite {
private:
    std::filesystem::path build_dir_;
    std::filesystem::path source_dir_;
    TestDataManager data_manager_;

public:
    struct BuildConfiguration {
        std::string name;
        std::vector<std::string> cmake_flags;
        std::vector<std::string> expected_targets;
        bool should_succeed;
    };

    struct BuildResult {
        int exit_code;
        std::string stdout_output;
        std::string stderr_output;
        std::chrono::milliseconds build_time;
        std::vector<std::string> generated_files;
        bool mlir_enabled;
    };

public:
    MLIRBuildVerificationSuite() 
        : build_dir_("/tmp/sv2sc_build_test_" + std::to_string(std::time(nullptr))),
          source_dir_(std::filesystem::current_path()) {
        std::filesystem::create_directories(build_dir_);
    }
    
    ~MLIRBuildVerificationSuite() {
        cleanup();
    }
    
    void cleanup() {
        if (std::filesystem::exists(build_dir_)) {
            std::filesystem::remove_all(build_dir_);
        }
    }
    
    BuildResult configureBuild(const BuildConfiguration& config) {
        BuildResult result{};
        auto start_time = std::chrono::steady_clock::now();
        
        // Prepare CMake command
        std::string cmake_cmd = "cd " + build_dir_.string() + " && cmake ";
        for (const auto& flag : config.cmake_flags) {
            cmake_cmd += flag + " ";
        }
        cmake_cmd += source_dir_.string() + " 2>&1";
        
        // Execute CMake configuration
        FILE* pipe = popen(cmake_cmd.c_str(), "r");
        if (!pipe) {
            result.exit_code = -1;
            return result;
        }
        
        char buffer[128];
        std::string cmake_output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            cmake_output += buffer;
        }
        
        result.exit_code = pclose(pipe);
        result.stdout_output = cmake_output;
        
        auto end_time = std::chrono::steady_clock::now();
        result.build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
            
        // Check if MLIR was enabled in configuration
        result.mlir_enabled = cmake_output.find("MLIR translation infrastructure configured") 
                            != std::string::npos;
        
        return result;
    }
    
    BuildResult executeBuild(const std::vector<std::string>& targets = {}) {
        BuildResult result{};
        auto start_time = std::chrono::steady_clock::now();
        
        // Prepare build command
        std::string build_cmd = "cd " + build_dir_.string() + " && make -j$(nproc) ";
        for (const auto& target : targets) {
            build_cmd += target + " ";
        }
        build_cmd += "2>&1";
        
        // Execute build
        FILE* pipe = popen(build_cmd.c_str(), "r");
        if (!pipe) {
            result.exit_code = -1;
            return result;
        }
        
        char buffer[128];
        std::string build_output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            build_output += buffer;
        }
        
        result.exit_code = pclose(pipe);
        result.stdout_output = build_output;
        
        auto end_time = std::chrono::steady_clock::now();
        result.build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        // Collect generated files
        collectGeneratedFiles(result);
        
        return result;
    }
    
private:
    void collectGeneratedFiles(BuildResult& result) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(build_dir_)) {
            if (entry.is_regular_file()) {
                auto path = entry.path();
                auto extension = path.extension().string();
                
                if (extension == ".a" || extension == ".so" || 
                    path.filename() == "sv2sc" || extension == ".o") {
                    result.generated_files.push_back(path.string());
                }
            }
        }
    }
};

TEST_CASE("MLIR Build Verification", "[build][mlir][verification]") {
    MLIRBuildVerificationSuite suite;
    
    SECTION("Clean build with MLIR enabled") {
        MLIRBuildVerificationSuite::BuildConfiguration config{
            "mlir_enabled",
            {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_BUILD_TYPE=Debug", "-DBUILD_TESTS=ON"},
            {"sv2sc", "sv2sc_mlir", "sv2sc_core", "sv2sc_translator", "sv2sc_codegen"},
            true
        };
        
        // Configure build
        auto configure_result = suite.configureBuild(config);
        INFO("CMake output: " << configure_result.stdout_output);
        REQUIRE(configure_result.exit_code == 0);
        REQUIRE(configure_result.mlir_enabled);
        
        // Execute build
        auto build_result = suite.executeBuild();
        INFO("Build output: " << build_result.stdout_output);
        REQUIRE(build_result.exit_code == 0);
        REQUIRE(build_result.build_time < std::chrono::minutes(10)); // Reasonable build time
        
        // Verify expected targets were built
        bool found_sv2sc = false;
        bool found_mlir_lib = false;
        
        for (const auto& file : build_result.generated_files) {
            if (file.find("sv2sc") != std::string::npos && file.find(".a") == std::string::npos) {
                found_sv2sc = true;
            }
            if (file.find("sv2sc_mlir") != std::string::npos) {
                found_mlir_lib = true;
            }
        }
        
        REQUIRE(found_sv2sc);
        REQUIRE(found_mlir_lib);
    }
    
    SECTION("Build without MLIR (fallback mode)") {
        MLIRBuildVerificationSuite::BuildConfiguration config{
            "mlir_disabled",
            {"-DSV2SC_ENABLE_MLIR=OFF", "-DCMAKE_BUILD_TYPE=Release"},
            {"sv2sc", "sv2sc_core", "sv2sc_translator", "sv2sc_codegen"},
            true
        };
        
        // Configure build
        auto configure_result = suite.configureBuild(config);
        REQUIRE(configure_result.exit_code == 0);
        REQUIRE_FALSE(configure_result.mlir_enabled);
        
        // Execute build
        auto build_result = suite.executeBuild();
        REQUIRE(build_result.exit_code == 0);
        
        // Verify MLIR components are not built
        for (const auto& file : build_result.generated_files) {
            REQUIRE(file.find("sv2sc_mlir") == std::string::npos);
        }
    }
    
    SECTION("Dependency resolution verification") {
        MLIRBuildVerificationSuite::BuildConfiguration config{
            "dependency_test",
            {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_BUILD_TYPE=Debug"},
            {},
            true
        };
        
        auto configure_result = suite.configureBuild(config);
        REQUIRE(configure_result.exit_code == 0);
        
        // Check for required dependencies in CMake output
        REQUIRE(configure_result.stdout_output.find("Found LLVM") != std::string::npos);
        REQUIRE(configure_result.stdout_output.find("Found MLIR") != std::string::npos);
        REQUIRE(configure_result.stdout_output.find("Found CIRCT") != std::string::npos);
    }
    
    SECTION("Incremental build verification") {
        MLIRBuildVerificationSuite::BuildConfiguration config{
            "incremental",
            {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_BUILD_TYPE=Debug"},
            {},
            true
        };
        
        // Initial build
        auto configure_result = suite.configureBuild(config);
        REQUIRE(configure_result.exit_code == 0);
        
        auto initial_build = suite.executeBuild();
        REQUIRE(initial_build.exit_code == 0);
        auto initial_time = initial_build.build_time;
        
        // Incremental build (should be faster)
        auto incremental_build = suite.executeBuild();
        REQUIRE(incremental_build.exit_code == 0);
        
        // Incremental build should be significantly faster
        REQUIRE(incremental_build.build_time < initial_time / 2);
    }
    
    SECTION("Build failure scenarios") {
        // Test handling of missing dependencies
        MLIRBuildVerificationSuite::BuildConfiguration bad_config{
            "missing_deps",
            {"-DSV2SC_ENABLE_MLIR=ON", "-DLLVM_DIR=/nonexistent"},
            {},
            false
        };
        
        auto configure_result = suite.configureBuild(bad_config);
        // Should fail gracefully with clear error message
        REQUIRE(configure_result.exit_code != 0);
        REQUIRE(configure_result.stdout_output.find("Could not find") != std::string::npos);
    }
}

TEST_CASE("Build Performance Benchmarks", "[build][performance][benchmark]") {
    MLIRBuildVerificationSuite suite;
    
    SECTION("Build time benchmarks") {
        std::vector<std::pair<std::string, MLIRBuildVerificationSuite::BuildConfiguration>> configs = {
            {"minimal", {"minimal", {"-DSV2SC_ENABLE_MLIR=OFF"}, {}, true}},
            {"full_debug", {"full_debug", {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_BUILD_TYPE=Debug", "-DBUILD_TESTS=ON"}, {}, true}},
            {"release_optimized", {"release_optimized", {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_BUILD_TYPE=Release"}, {}, true}}
        };
        
        for (const auto& [name, config] : configs) {
            INFO("Testing configuration: " << name);
            
            auto configure_result = suite.configureBuild(config);
            REQUIRE(configure_result.exit_code == 0);
            
            auto build_result = suite.executeBuild();
            REQUIRE(build_result.exit_code == 0);
            
            // Performance assertions based on configuration
            if (name == "minimal") {
                REQUIRE(build_result.build_time < std::chrono::minutes(2));
            } else if (name == "release_optimized") {
                REQUIRE(build_result.build_time < std::chrono::minutes(8));
            }
            
            INFO("Build time for " << name << ": " << build_result.build_time.count() << "ms");
        }
    }
}

TEST_CASE("Cross-Platform Build Verification", "[build][platform][portability]") {
    MLIRBuildVerificationSuite suite;
    
    SECTION("Compiler compatibility") {
        // Test different compiler configurations
        std::vector<std::pair<std::string, std::string>> compilers = {
            {"GCC", "g++"},
            {"Clang", "clang++"}
        };
        
        for (const auto& [name, compiler] : compilers) {
            if (system(("which " + compiler + " > /dev/null 2>&1").c_str()) == 0) {
                INFO("Testing with compiler: " << name);
                
                MLIRBuildVerificationSuite::BuildConfiguration config{
                    name + "_build",
                    {"-DSV2SC_ENABLE_MLIR=ON", "-DCMAKE_CXX_COMPILER=" + compiler},
                    {},
                    true
                };
                
                auto configure_result = suite.configureBuild(config);
                REQUIRE(configure_result.exit_code == 0);
                
                auto build_result = suite.executeBuild();
                REQUIRE(build_result.exit_code == 0);
            }
        }
    }
}