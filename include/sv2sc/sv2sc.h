#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sv2sc {

struct TranslationOptions {
    // Core file processing
    std::vector<std::string> inputFiles;
    std::string outputDir = "./output";
    std::string topModule;
    
    // SystemC generation options
    bool generateTestbench = false;
    std::string clockSignal = "clk";
    std::string resetSignal = "reset";
    bool enableSynthesis = false;
    
    // Language and compilation options
    bool enableSystemVerilog = true;
    std::string timescale = "1ns/1ps";
    bool enable64Bit = false;
    
    // Preprocessor and include handling
    std::vector<std::string> includePaths;
    std::unordered_map<std::string, std::string> defineMap;
    std::vector<std::string> undefines;
    
    // Library and dependency management
    std::vector<std::string> libraryPaths;
    std::vector<std::string> libraryFiles;
    std::vector<std::string> fileExtensions;
    
    // Debug and output control
    bool enableDebug = false;
    bool enableVerbose = false;
    std::string outputName = "simv";
    
    // MLIR pipeline options
    bool useMLIRPipeline = false;
    int optimizationLevel = 1;
    bool enableMLIRDiagnostics = false;
    bool dumpMLIR = false;
    
    // Legacy defines (for backward compatibility)
    std::vector<std::string> defines;
};

class SystemVerilogToSystemCTranslator {
public:
    explicit SystemVerilogToSystemCTranslator(const TranslationOptions& options);
    ~SystemVerilogToSystemCTranslator();

    bool translate();
    const std::vector<std::string>& getErrors() const;
    const std::vector<std::string>& getWarnings() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace sv2sc