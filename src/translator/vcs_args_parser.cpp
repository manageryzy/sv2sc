#include "translator/vcs_args_parser.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace sv2sc::translator {

VCSArgsParser::VCSArgsParser() {
    setupParser();
}

bool VCSArgsParser::parse(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "-top" && i + 1 < argc) {
            args_.topModule = argv[++i];
        }
        else if (arg == "-o" && i + 1 < argc) {
            args_.outputDir = argv[++i];
        }
        else if (arg == "-timescale" && i + 1 < argc) {
            args_.timescale = argv[++i];
        }
        else if (arg == "-debug") {
            args_.enableDebug = true;
        }
        else if (arg == "-v" || arg == "-verbose") {
            args_.enableVerbose = true;
        }
        else if (arg == "-I" && i + 1 < argc) {
            args_.includePaths.push_back(argv[++i]);
        }
        else if (arg.substr(0, 8) == "+incdir+") {
            args_.includePaths.push_back(arg.substr(8));
        }
        else if (arg == "-y" && i + 1 < argc) {
            args_.libraryPaths.push_back(argv[++i]);
        }
        else if (arg == "-D" && i + 1 < argc) {
            args_.defines.push_back(argv[++i]);
        }
        else if (arg.substr(0, 8) == "+define+") {
            args_.defines.push_back(arg.substr(8));
        }
        else if (arg == "-U" && i + 1 < argc) {
            args_.undefines.push_back(argv[++i]);
        }
        else if (arg == "--clock" && i + 1 < argc) {
            args_.clockSignal = argv[++i];
        }
        else if (arg == "--reset" && i + 1 < argc) {
            args_.resetSignal = argv[++i];
        }
        else if (arg == "--synthesis") {
            args_.enableSynthesis = true;
        }
        else if (arg == "--testbench") {
            args_.generateTestbench = true;
        }
        else if (arg == "--no-elab") {
            args_.enableElaboration = false;
        }
        else if (arg == "-V" || arg == "--version") {
            printVersion();
            return false;
        }
        else if (arg == "-h" || arg == "--help") {
            printHelp();
            return false;
        }
        else if (!arg.empty() && arg[0] != '-') {
            // Input files
            args_.inputFiles.push_back(arg);
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printHelp();
            return false;
        }
    }
    
    return validateArguments();
}

void VCSArgsParser::setupParser() {
    // Initialize default values
    args_.outputDir = "./output";
    args_.clockSignal = "clk";
    args_.resetSignal = "reset";
    args_.enableElaboration = true;
}

bool VCSArgsParser::validateArguments() {
    if (args_.inputFiles.empty()) {
        std::cerr << "Error: No input files specified" << std::endl;
        return false;
    }
    
    // Validate that all input files exist
    for (const auto& file : args_.inputFiles) {
        if (!std::filesystem::exists(file)) {
            std::cerr << "Error: Input file does not exist: " << file << std::endl;
            return false;
        }
    }
    
    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(args_.outputDir)) {
        try {
            std::filesystem::create_directories(args_.outputDir);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error: Cannot create output directory: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Expand relative paths
    for (auto& path : args_.includePaths) {
        path = expandPath(path);
    }
    
    for (auto& path : args_.libraryPaths) {
        path = expandPath(path);
    }
    
    // Process defines
    processDefines(args_.defines);
    
    return true;
}

void VCSArgsParser::processDefines(const std::vector<std::string>& defines) {
    // Process defines in the format NAME=VALUE or just NAME
    for (auto& define : args_.defines) {
        auto pos = define.find('=');
        if (pos == std::string::npos) {
            // Just NAME, add =1
            define += "=1";
        }
    }
}

std::string VCSArgsParser::expandPath(const std::string& path) {
    return std::filesystem::absolute(path);
}

void VCSArgsParser::printHelp() const {
    std::cout << R"(
SystemVerilog to SystemC Translator (sv2sc)

Usage: sv2sc [options] <input_files...>

VCS-Compatible Options:
  -top <module>         Specify top module name
  -o <dir>              Output directory (default: ./output)
  -I <dir>              Add include directory
  +incdir+<dir>         Add include directory (VCS format)
  -y <dir>              Add library directory
  -D <name>[=<value>]   Define preprocessor macro
  +define+<name>=<val>  Define preprocessor macro (VCS format)
  -U <name>             Undefine preprocessor macro
  -timescale <spec>     Set timescale

SystemC Options:
  --clock <signal>      Clock signal name (default: clk)
  --reset <signal>      Reset signal name (default: reset)
  --testbench           Generate SystemC testbench
  --synthesis           Enable synthesis mode
  --no-elab             Disable elaboration

General Options:
  -v, --verbose         Enable verbose output
  --debug               Enable debug output
  -V, --version         Show version information
  -h, --help            Show this help message

Examples:
  sv2sc -top counter counter.sv
  sv2sc -I include -D WIDTH=8 -top dut design.sv
  sv2sc +incdir+./rtl +define+SYNTHESIS=1 -top cpu cpu.sv
)";
}

void VCSArgsParser::printVersion() const {
    std::cout << "sv2sc (SystemVerilog to SystemC Translator) version 1.0.0" << std::endl;
}

} // namespace sv2sc::translator