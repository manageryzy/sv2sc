#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace sv2sc::testing {

/**
 * @brief Manages test data creation, cleanup, and lifecycle
 * 
 * Provides RAII-based management of test files, directories, and resources
 * with automatic cleanup to prevent test pollution.
 */
class TestDataManager {
public:
    enum class ErrorType {
        SYNTAX_ERROR,
        MISSING_SEMICOLON,
        INVALID_IDENTIFIER,
        TYPE_MISMATCH,
        UNMATCHED_BLOCKS,
        CIRCULAR_DEPENDENCY
    };
    
    struct TestModule {
        std::string name;
        std::vector<std::string> ports;
        std::vector<std::string> processes;
        std::size_t complexity_score;
    };

private:
    std::vector<std::filesystem::path> temp_files_;
    std::vector<std::filesystem::path> temp_dirs_;
    std::mt19937 rng_{std::random_device{}()};
    std::unordered_set<std::string> used_names_;
    
public:
    TestDataManager() = default;
    ~TestDataManager() { cleanup(); }
    
    // Disable copy/move to ensure proper cleanup
    TestDataManager(const TestDataManager&) = delete;
    TestDataManager& operator=(const TestDataManager&) = delete;
    
    // === Basic Module Generation ===
    
    /**
     * @brief Generate basic SystemVerilog module
     */
    std::string generateBasicModule(const std::string& name);
    
    /**
     * @brief Generate counter module with configurable width
     */
    std::string generateCounter(const std::string& name, int width = 8);
    
    /**
     * @brief Generate FSM module with configurable states
     */
    std::string generateFSM(const std::string& name, int num_states = 4);
    
    /**
     * @brief Generate memory module with configurable size
     */
    std::string generateMemory(const std::string& name, int depth = 256, int width = 32);
    
    // === Complex Module Generation ===
    
    /**
     * @brief Generate complex module with specified parameters
     */
    std::string generateComplexModule(const std::string& name, 
                                    int num_ports = 8, 
                                    int num_processes = 4,
                                    bool include_interfaces = false,
                                    bool include_assertions = false);
    
    /**
     * @brief Generate CPU-like module (similar to PicoRV32 complexity)
     */
    std::string generateCPUModule(const std::string& name);
    
    /**
     * @brief Generate hierarchical design with submodules
     */
    std::string generateHierarchicalDesign(const std::string& top_name, int num_levels = 3);
    
    // === Error Case Generation ===
    
    /**
     * @brief Generate malformed SystemVerilog for error testing
     */
    std::string generateMalformedSV(ErrorType type, const std::string& base_module = "");
    
    /**
     * @brief Generate edge case constructs
     */
    std::string generateEdgeCase(const std::string& case_type);
    
    /**
     * @brief Generate massive module for stress testing
     */
    std::string generateStressTestModule(const std::string& name, std::size_t target_lines = 10000);
    
    // === File Management ===
    
    /**
     * @brief Create temporary file with content
     */
    std::filesystem::path createTempFile(const std::string& content, 
                                       const std::string& extension = ".sv");
    
    /**
     * @brief Create temporary directory
     */
    std::filesystem::path createTempDir(const std::string& name = "");
    
    /**
     * @brief Create test project structure
     */
    std::filesystem::path createTestProject(const std::string& project_name,
                                          const std::vector<TestModule>& modules);
    
    // === Golden Reference Management ===
    
    /**
     * @brief Load expected SystemC output for comparison
     */
    std::string loadGoldenReference(const std::string& test_name);
    
    /**
     * @brief Save new golden reference (for test maintenance)
     */
    void saveGoldenReference(const std::string& test_name, const std::string& content);
    
    /**
     * @brief Validate golden reference integrity
     */
    bool validateGoldenReference(const std::string& test_name);
    
    // === Test Configuration ===
    
    /**
     * @brief Generate test configuration file
     */
    std::string generateTestConfig(const std::unordered_map<std::string, std::string>& options);
    
    /**
     * @brief Get test data directory path
     */
    std::filesystem::path getTestDataDir() const;
    
    /**
     * @brief Get temporary directory for current test
     */
    std::filesystem::path getTempDir() const;
    
    // === Cleanup and Maintenance ===
    
    /**
     * @brief Manual cleanup of all created resources
     */
    void cleanup();
    
    /**
     * @brief Get list of created temporary files
     */
    const std::vector<std::filesystem::path>& getTempFiles() const { return temp_files_; }
    
    /**
     * @brief Get list of created temporary directories
     */
    const std::vector<std::filesystem::path>& getTempDirs() const { return temp_dirs_; }

private:
    std::string generateUniqueModuleName(const std::string& prefix = "test");
    std::string generateRandomIdentifier(std::size_t length = 8);
    std::string generatePortList(int num_ports, bool include_complex_types = false);
    std::string generateProcessBlock(const std::string& type, int complexity = 1);
    std::string generateAssignmentBlock(int num_assignments);
    std::string addRandomComments(const std::string& code);
    
    void registerTempFile(const std::filesystem::path& path);
    void registerTempDir(const std::filesystem::path& path);
};

/**
 * @brief RAII helper for scoped test resources
 */
class ScopedTestResource {
private:
    std::function<void()> cleanup_func_;
    
public:
    template<typename CleanupFunc>
    explicit ScopedTestResource(CleanupFunc&& func) 
        : cleanup_func_(std::forward<CleanupFunc>(func)) {}
    
    ~ScopedTestResource() {
        if (cleanup_func_) {
            cleanup_func_();
        }
    }
    
    ScopedTestResource(const ScopedTestResource&) = delete;
    ScopedTestResource& operator=(const ScopedTestResource&) = delete;
    
    ScopedTestResource(ScopedTestResource&& other) noexcept
        : cleanup_func_(std::move(other.cleanup_func_)) {
        other.cleanup_func_ = nullptr;
    }
    
    ScopedTestResource& operator=(ScopedTestResource&& other) noexcept {
        if (this != &other) {
            cleanup_func_ = std::move(other.cleanup_func_);
            other.cleanup_func_ = nullptr;
        }
        return *this;
    }
};

/**
 * @brief Test fixture base class with data management
 */
class TestFixtureBase {
protected:
    std::unique_ptr<TestDataManager> data_manager_;
    
public:
    TestFixtureBase() : data_manager_(std::make_unique<TestDataManager>()) {}
    virtual ~TestFixtureBase() = default;
    
    TestDataManager& getDataManager() { return *data_manager_; }
    const TestDataManager& getDataManager() const { return *data_manager_; }
};

} // namespace sv2sc::testing