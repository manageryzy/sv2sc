#include "test_data_manager.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <chrono>

namespace sv2sc::testing {

std::string TestDataManager::generateBasicModule(const std::string& name) {
    std::ostringstream oss;
    
    oss << "// Generated test module: " << name << "\n"
        << "module " << name << " (\n"
        << "    input logic clk,\n"
        << "    input logic reset,\n"
        << "    output logic [7:0] data_out\n"
        << ");\n\n"
        << "    always_ff @(posedge clk) begin\n"
        << "        if (reset) begin\n"
        << "            data_out <= 8'b0;\n"
        << "        end else begin\n"
        << "            data_out <= data_out + 1;\n"
        << "        end\n"
        << "    end\n\n"
        << "endmodule\n";
        
    return oss.str();
}

std::string TestDataManager::generateCounter(const std::string& name, int width) {
    std::ostringstream oss;
    
    oss << "// Generated counter module: " << name << "\n"
        << "module " << name << " (\n"
        << "    input logic clk,\n"
        << "    input logic reset,\n"
        << "    input logic enable,\n"
        << "    output logic [" << (width-1) << ":0] count\n"
        << ");\n\n"
        << "    always_ff @(posedge clk) begin\n"
        << "        if (reset) begin\n"
        << "            count <= " << width << "'b0;\n"
        << "        end else if (enable) begin\n"
        << "            count <= count + 1;\n"
        << "        end\n"
        << "    end\n\n"
        << "endmodule\n";
        
    return oss.str();
}

std::string TestDataManager::generateFSM(const std::string& name, int num_states) {
    std::ostringstream oss;
    
    // Calculate state width
    int state_width = 1;
    while ((1 << state_width) < num_states) {
        state_width++;
    }
    
    oss << "// Generated FSM module: " << name << "\n"
        << "module " << name << " (\n"
        << "    input logic clk,\n"
        << "    input logic reset,\n"
        << "    input logic start,\n"
        << "    output logic done,\n"
        << "    output logic [" << (state_width-1) << ":0] state_out\n"
        << ");\n\n"
        << "    typedef enum logic [" << (state_width-1) << ":0] {\n";
    
    // Generate state enum
    for (int i = 0; i < num_states; ++i) {
        oss << "        STATE_" << i;
        if (i < num_states - 1) oss << ",";
        oss << "\n";
    }
    
    oss << "    } state_t;\n\n"
        << "    state_t current_state, next_state;\n\n"
        << "    // State register\n"
        << "    always_ff @(posedge clk) begin\n"
        << "        if (reset) begin\n"
        << "            current_state <= STATE_0;\n"
        << "        end else begin\n"
        << "            current_state <= next_state;\n"
        << "        end\n"
        << "    end\n\n"
        << "    // Next state logic\n"
        << "    always_comb begin\n"
        << "        next_state = current_state;\n"
        << "        case (current_state)\n";
    
    // Generate state transitions
    for (int i = 0; i < num_states; ++i) {
        oss << "            STATE_" << i << ": begin\n";
        if (i == 0) {
            oss << "                if (start) next_state = STATE_1;\n";
        } else if (i == num_states - 1) {
            oss << "                next_state = STATE_0;\n";
        } else {
            oss << "                next_state = STATE_" << (i + 1) << ";\n";
        }
        oss << "            end\n";
    }
    
    oss << "        endcase\n"
        << "    end\n\n"
        << "    // Output logic\n"
        << "    assign done = (current_state == STATE_" << (num_states-1) << ");\n"
        << "    assign state_out = state_t'(current_state);\n\n"
        << "endmodule\n";
        
    return oss.str();
}

std::string TestDataManager::generateComplexModule(const std::string& name, 
                                                  int num_ports, 
                                                  int num_processes,
                                                  bool include_interfaces,
                                                  bool include_assertions) {
    std::ostringstream oss;
    
    oss << "// Generated complex module: " << name << "\n";
    
    if (include_interfaces) {
        oss << "interface axi_if;\n"
            << "    logic awvalid, awready;\n"
            << "    logic [31:0] awaddr;\n"
            << "    logic wvalid, wready;\n"
            << "    logic [31:0] wdata;\n"
            << "    logic bvalid, bready;\n"
            << "    logic [1:0] bresp;\n"
            << "endinterface\n\n";
    }
    
    oss << "module " << name << " (\n"
        << "    input logic clk,\n"
        << "    input logic reset,\n";
    
    // Generate additional ports
    for (int i = 0; i < num_ports; ++i) {
        std::string port_type = (i % 3 == 0) ? "input" : "output"; 
        int width = 1 + (i % 32);
        oss << "    " << port_type << " logic [" << (width-1) << ":0] port_" << i;
        if (i < num_ports - 1) oss << ",";
        oss << "\n";
    }
    
    oss << ");\n\n";
    
    // Generate internal signals
    oss << "    // Internal signals\n";
    for (int i = 0; i < num_processes * 2; ++i) {
        int width = 8 + (i % 24);
        oss << "    logic [" << (width-1) << ":0] internal_sig_" << i << ";\n";
    }
    oss << "\n";
    
    // Generate processes
    for (int i = 0; i < num_processes; ++i) {
        if (i % 2 == 0) {
            // Clocked process
            oss << "    // Process " << i << " (clocked)\n"
                << "    always_ff @(posedge clk) begin\n"
                << "        if (reset) begin\n";
            for (int j = 0; j < 3; ++j) {
                int sig_idx = i * 2 + j;
                oss << "            internal_sig_" << sig_idx << " <= 0;\n";
            }
            oss << "        end else begin\n";
            for (int j = 0; j < 3; ++j) {
                int sig_idx = i * 2 + j;
                int src_idx = (sig_idx + 1) % (num_processes * 2);
                oss << "            internal_sig_" << sig_idx << " <= internal_sig_" << src_idx << " + " << (j+1) << ";\n";
            }
            oss << "        end\n"
                << "    end\n\n";
        } else {
            // Combinational process  
            oss << "    // Process " << i << " (combinational)\n"
                << "    always_comb begin\n";
            for (int j = 0; j < 2; ++j) {
                int sig_idx = i * 2 + j;
                int src1_idx = (sig_idx + 2) % (num_processes * 2);
                int src2_idx = (sig_idx + 3) % (num_processes * 2);
                oss << "        internal_sig_" << sig_idx << " = internal_sig_" << src1_idx 
                    << " ^ internal_sig_" << src2_idx << ";\n";
            }
            oss << "    end\n\n";
        }
    }
    
    // Generate output assignments
    oss << "    // Output assignments\n";
    for (int i = 0; i < num_ports; ++i) {
        if ((i % 3) != 0) { // Only for output ports
            int src_sig = i % (num_processes * 2);
            oss << "    assign port_" << i << " = internal_sig_" << src_sig << ";\n";
        }
    }
    
    if (include_assertions) {
        oss << "\n    // Assertions\n";
        oss << "    assert property (@(posedge clk) disable iff (reset) internal_sig_0 >= 0);\n";
        oss << "    assert property (@(posedge clk) disable iff (reset) port_1 <= 32'hFFFFFFFF);\n";
    }
    
    oss << "\nendmodule\n";
    
    return oss.str();
}

std::string TestDataManager::generateMalformedSV(ErrorType type, const std::string& base_module) {
    std::string base = base_module.empty() ? generateBasicModule("error_test") : base_module;
    
    switch (type) {
        case ErrorType::SYNTAX_ERROR:
            // Remove semicolon from assignment
            return std::regex_replace(base, std::regex("data_out <= 8'b0;"), "data_out <= 8'b0");
            
        case ErrorType::MISSING_SEMICOLON:
            // Remove semicolon from end statement
            return std::regex_replace(base, std::regex("end;"), "end");
            
        case ErrorType::INVALID_IDENTIFIER:
            // Use reserved keyword as identifier
            return std::regex_replace(base, std::regex("data_out"), "module");
            
        case ErrorType::TYPE_MISMATCH:
            // Assign wrong width
            return std::regex_replace(base, std::regex("8'b0"), "16'b0");
            
        case ErrorType::UNMATCHED_BLOCKS:
            // Remove an 'end' statement
            return std::regex_replace(base, std::regex("    end\nendmodule"), "endmodule");
            
        case ErrorType::CIRCULAR_DEPENDENCY:
            return base + "\n// Circular dependency\nmodule dep1; dep2 inst(); endmodule\n"
                         "module dep2; dep1 inst(); endmodule\n";
            
        default:
            return base;
    }
}

std::filesystem::path TestDataManager::createTempFile(const std::string& content, 
                                                     const std::string& extension) {
    // Generate unique filename
    auto now = std::chrono::steady_clock::now();
    auto timestamp = now.time_since_epoch().count();
    
    std::ostringstream filename;
    filename << "test_" << timestamp << "_" << generateRandomIdentifier(6) << extension;
    
    auto temp_dir = std::filesystem::temp_directory_path();
    auto filepath = temp_dir / filename.str();
    
    // Write content to file
    std::ofstream file(filepath);
    if (!file) {
        throw std::runtime_error("Failed to create temporary file: " + filepath.string());
    }
    
    file << content;
    file.close();
    
    registerTempFile(filepath);
    return filepath;
}

std::filesystem::path TestDataManager::createTempDir(const std::string& name) {
    std::string dir_name = name;
    if (dir_name.empty()) {
        auto now = std::chrono::steady_clock::now();
        auto timestamp = now.time_since_epoch().count();
        dir_name = "testdir_" + std::to_string(timestamp) + "_" + generateRandomIdentifier(6);
    }
    
    auto temp_dir = std::filesystem::temp_directory_path();
    auto dirpath = temp_dir / dir_name;
    
    std::filesystem::create_directories(dirpath);
    
    registerTempDir(dirpath);
    return dirpath;
}

std::string TestDataManager::loadGoldenReference(const std::string& test_name) {
    auto test_data_dir = getTestDataDir();
    auto golden_path = test_data_dir / "golden" / (test_name + ".expected");
    
    if (!std::filesystem::exists(golden_path)) {
        throw std::runtime_error("Golden reference not found: " + golden_path.string());
    }
    
    std::ifstream file(golden_path);
    std::ostringstream content;
    content << file.rdbuf();
    
    return content.str();
}

void TestDataManager::saveGoldenReference(const std::string& test_name, const std::string& content) {
    auto test_data_dir = getTestDataDir();
    auto golden_dir = test_data_dir / "golden";
    
    std::filesystem::create_directories(golden_dir);
    
    auto golden_path = golden_dir / (test_name + ".expected");
    
    std::ofstream file(golden_path);
    file << content;
}

std::filesystem::path TestDataManager::getTestDataDir() const {
    // Check environment variable first
    if (const char* env_dir = std::getenv("SV2SC_TEST_DATA_DIR")) {
        return std::filesystem::path(env_dir);
    }
    
    // Default to project test data directory
    return std::filesystem::current_path() / "tests" / "data";
}

void TestDataManager::cleanup() {
    // Remove temporary files
    for (const auto& file : temp_files_) {
        try {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        } catch (const std::exception&) {
            // Continue cleanup even if one file fails
        }
    }
    temp_files_.clear();
    
    // Remove temporary directories
    for (const auto& dir : temp_dirs_) {
        try {
            if (std::filesystem::exists(dir)) {
                std::filesystem::remove_all(dir);
            }
        } catch (const std::exception&) {
            // Continue cleanup even if one directory fails
        }
    }
    temp_dirs_.clear();
    
    used_names_.clear();
}

std::string TestDataManager::generateUniqueModuleName(const std::string& prefix) {
    std::string name;
    int counter = 0;
    
    do {
        name = prefix + "_" + std::to_string(counter++);
    } while (used_names_.find(name) != used_names_.end());
    
    used_names_.insert(name);
    return name;
}

std::string TestDataManager::generateRandomIdentifier(std::size_t length) {
    const std::string chars = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::uniform_int_distribution<> dis(0, chars.size() - 1);
    
    std::string result;
    result.reserve(length);
    
    // First character must be a letter
    result += chars[dis(rng_) % 26]; 
    
    for (std::size_t i = 1; i < length; ++i) {
        result += chars[dis(rng_)];
    }
    
    return result;
}

void TestDataManager::registerTempFile(const std::filesystem::path& path) {
    temp_files_.push_back(path);
}

void TestDataManager::registerTempDir(const std::filesystem::path& path) {
    temp_dirs_.push_back(path);
}

} // namespace sv2sc::testing