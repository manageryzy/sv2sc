#include "sv2sc/sv2sc.h"
#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <slang/syntax/SyntaxTree.h>
#include <slang/text/SourceManager.h>
#include <slang/ast/Compilation.h>
#include <slang/util/Bag.h>
#include <slang/parsing/Preprocessor.h>
#include <filesystem>
#include <fstream>

namespace sv2sc {

class SystemVerilogToSystemCTranslator::Impl {
public:
    explicit Impl(const TranslationOptions& options) : options_(options) {
        setupLogger();
    }
    
    bool translate() {
        try {
            LOG_INFO("Starting SystemVerilog to SystemC translation");
            
            if (options_.inputFiles.size() == 1) {
                // Single file processing
                return processFile(options_.inputFiles[0]);
            } else {
                // Multi-file design-level compilation
                return processDesign();
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Translation failed with exception: {}", e.what());
            errors_.push_back(fmt::format("Exception: {}", e.what()));
            return false;
        }
    }
    
    bool processDesign() {
        LOG_INFO("Processing design with {} files", options_.inputFiles.size());
        
        try {
            // Create compilation with VCS options
            slang::ast::Compilation compilation = createCompilationWithOptions();
            
            // Add all source files to compilation
            for (const auto& inputFile : options_.inputFiles) {
                LOG_INFO("Loading file: {}", inputFile);
                
                std::ifstream file(inputFile);
                if (!file.is_open()) {
                    std::string error = fmt::format("Cannot open input file: {}", inputFile);
                    LOG_ERROR(error);
                    errors_.push_back(error);
                    return false;
                }
                
                std::string content((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
                file.close();
                
                // Create syntax tree with proper source name  
                auto syntaxTree = slang::syntax::SyntaxTree::fromText(
                    std::string_view(content), std::string_view(inputFile), std::string_view(inputFile));
                
                // Check for parsing errors
                if (syntaxTree->diagnostics().size() > 0) {
                    LOG_WARN("Parsing diagnostics found in {}", inputFile);
                    for (const auto& diag : syntaxTree->diagnostics()) {
                        std::string diagMsg = fmt::format("{}:{} - Parse diagnostic", 
                            inputFile, diag.location.offset());
                        LOG_WARN(diagMsg);
                        warnings_.push_back(diagMsg);
                    }
                }
                
                compilation.addSyntaxTree(syntaxTree);
            }
            
            // Elaborate the design
            if (!options_.topModule.empty()) {
                LOG_INFO("Elaborating top module: {}", options_.topModule);
            }
            
            // Process the compilation
            return processCompilation(compilation);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Design processing failed: {}", e.what());
            errors_.push_back(fmt::format("Design processing error: {}", e.what()));
            return false;
        }
    }
    
    bool processCompilation(slang::ast::Compilation& compilation) {
        LOG_INFO("Processing compilation with SystemC generation");
        
        try {
            // Create and configure SystemC code generator
            codegen::SystemCCodeGenerator generator;
            configureSystemCGenerator(generator);
            
            // Create visitor for AST traversal
            core::SVToSCVisitor visitor(generator);
            
            // Get compilation root and process
            auto& root = compilation.getRoot();
            
            // If top module is specified, find and process only that hierarchy
            if (!options_.topModule.empty()) {
                bool foundTopModule = false;
                
                // Search for the top module in the design
                for (auto& instance : root.topInstances) {
                    LOG_DEBUG("Found top-level instance: {}", instance->name);
                    
                    if (instance->name == options_.topModule) {
                        LOG_INFO("Processing top module: {}", options_.topModule);
                        visitor.visit(*instance);
                        foundTopModule = true;
                        
                        // Generate output files for top module
                        std::string headerPath = options_.outputDir + "/" + options_.topModule + ".h";
                        std::string implPath = options_.outputDir + "/" + options_.topModule + ".cpp";
                        
                        if (!generator.writeToFile(headerPath, implPath)) {
                            std::string error = fmt::format("Failed to write output files for top module: {}", options_.topModule);
                            LOG_ERROR(error);
                            errors_.push_back(error);
                            return false;
                        }
                        
                        break;
                    }
                }
                
                if (!foundTopModule) {
                    std::string error = fmt::format("Top module '{}' not found in design", options_.topModule);
                    LOG_ERROR(error);
                    errors_.push_back(error);
                    return false;
                }
            } else {
                // Process all top-level modules
                LOG_INFO("Processing all top-level modules");
                
                for (auto& instance : root.topInstances) {
                    LOG_INFO("Processing module: {}", instance->name);
                    visitor.visit(*instance);
                    
                    // Generate output files for each module
                    std::string headerPath = options_.outputDir + "/" + std::string(instance->name) + ".h";
                    std::string implPath = options_.outputDir + "/" + std::string(instance->name) + ".cpp";
                    
                    if (!generator.writeToFile(headerPath, implPath)) {
                        std::string error = fmt::format("Failed to write output files for module: {}", instance->name);
                        LOG_ERROR(error);
                        errors_.push_back(error);
                        return false;
                    }
                }
            }
            
            // Generate testbench if requested
            if (options_.generateTestbench) {
                generateTestbench(generator);
            }
            
            LOG_INFO("Compilation processing completed successfully");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Compilation processing failed: {}", e.what());
            errors_.push_back(fmt::format("Compilation error: {}", e.what()));
            return false;
        }
    }
    
    const std::vector<std::string>& getErrors() const { return errors_; }
    const std::vector<std::string>& getWarnings() const { return warnings_; }

private:
    TranslationOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;
    
    void setupLogger() {
        auto& logger = utils::Logger::getInstance();
        
        if (options_.enableDebug) {
            logger.setLevel(spdlog::level::debug);
        } else if (options_.enableVerbose) {
            logger.setLevel(spdlog::level::info);
        } else {
            logger.setLevel(spdlog::level::warn);
        }
        
        // Enable file logging in output directory
        std::filesystem::create_directories(options_.outputDir);
        std::string logFile = options_.outputDir + "/sv2sc.log";
        logger.enableFileLogging(logFile);
    }
    
    slang::ast::Compilation createCompilationWithOptions() {
        // Create compilation with proper slang configuration
        slang::Bag options;
        
        // Create preprocessor options
        slang::parsing::PreprocessorOptions preprocessorOpts;
        
        // Configure preprocessor defines
        for (const auto& [name, value] : options_.defineMap) {
            LOG_DEBUG("Adding preprocessor define: {}={}", name, value);
            // Use slang's preprocessor define mechanism
            std::string defineStr = name + "=" + value;
            preprocessorOpts.predefines.emplace_back(defineStr);
        }
        
        // Configure include directories
        for (const auto& includePath : options_.includePaths) {
            LOG_DEBUG("Configuring include path: {}", includePath);
            preprocessorOpts.additionalIncludePaths.emplace_back(includePath);
        }
        
        // Set the preprocessor options into the bag
        options.set(preprocessorOpts);
        
        // Configure timescale if specified
        if (!options_.timescale.empty()) {
            LOG_DEBUG("Setting timescale: {}", options_.timescale);
            // Parse timescale and configure compilation
            // Example: "1ns/1ps" -> timeUnit=1ns, timePrecision=1ps
        }
        
        return slang::ast::Compilation(options);
    }
    
    void configureSystemCGenerator(codegen::SystemCCodeGenerator& generator) {
        // Configure SystemC generation based on VCS options
        LOG_DEBUG("Configuring SystemC generator with VCS options");
        
        // Configure clock and reset signals
        if (!options_.clockSignal.empty()) {
            LOG_DEBUG("Setting clock signal: {}", options_.clockSignal);
            // generator.setClockSignal(options_.clockSignal); // Would need API
        }
        
        if (!options_.resetSignal.empty()) {
            LOG_DEBUG("Setting reset signal: {}", options_.resetSignal);
            // generator.setResetSignal(options_.resetSignal); // Would need API
        }
        
        // Configure synthesis mode
        if (options_.enableSynthesis) {
            LOG_DEBUG("Enabling synthesis mode");
            // generator.setSynthesisMode(true); // Would need API
        }
        
        // Configure testbench generation
        if (options_.generateTestbench) {
            LOG_DEBUG("Enabling testbench generation");
            // generator.setGenerateTestbench(true); // Would need API
        }
        
        // Configure timescale
        if (!options_.timescale.empty()) {
            LOG_DEBUG("Setting timescale: {}", options_.timescale);
            // generator.setTimescale(options_.timescale); // Would need API
        }
        
        // Configure 64-bit mode
        if (options_.enable64Bit) {
            LOG_DEBUG("Enabling 64-bit mode");
            // generator.set64BitMode(true); // Would need API
        }
    }
    
    void generateTestbench(codegen::SystemCCodeGenerator& generator) {
        LOG_INFO("Generating SystemC testbench");
        
        if (options_.topModule.empty()) {
            LOG_WARN("No top module specified, skipping testbench generation");
            return;
        }
        
        try {
            std::string tbName = options_.topModule + "_tb";
            std::string tbHeaderPath = options_.outputDir + "/" + tbName + ".h";
            std::string tbImplPath = options_.outputDir + "/" + tbName + ".cpp";
            
            LOG_DEBUG("Creating testbench files: {} and {}", tbHeaderPath, tbImplPath);
            
            // Generate testbench header
            std::ofstream tbHeader(tbHeaderPath);
            if (!tbHeader.is_open()) {
                LOG_ERROR("Failed to create testbench header file: {}", tbHeaderPath);
                return;
            }
            
            tbHeader << "#ifndef " << tbName << "_H\n";
            tbHeader << "#define " << tbName << "_H\n\n";
            tbHeader << "#include <systemc.h>\n";
            tbHeader << "#include \"" << options_.topModule << ".h\"\n\n";
            tbHeader << "SC_MODULE(" << tbName << ") {\n";
            tbHeader << "    // Clock and reset signals\n";
            tbHeader << "    sc_clock " << options_.clockSignal << ";\n";
            tbHeader << "    sc_signal<sc_logic> " << options_.resetSignal << ";\n\n";
            tbHeader << "    // Test signals\n";
            tbHeader << "    sc_signal<sc_lv<32>> data_in;\n";
            tbHeader << "    sc_signal<sc_lv<32>> data_out;\n";
            tbHeader << "    sc_signal<sc_logic> valid;\n\n";
            tbHeader << "    // DUT instance\n";
            tbHeader << "    " << options_.topModule << " dut;\n\n";
            tbHeader << "    SC_CTOR(" << tbName << ") : \n";
            tbHeader << "        " << options_.clockSignal << "(\"" << options_.clockSignal << "\", 10, SC_NS),\n";
            tbHeader << "        dut(\"dut\")\n";
            tbHeader << "    {\n";
            tbHeader << "        // Connect DUT\n";
            tbHeader << "        dut." << options_.clockSignal << "(" << options_.clockSignal << ");\n";
            tbHeader << "        dut." << options_.resetSignal << "(" << options_.resetSignal << ");\n";
            tbHeader << "        dut.data_in(data_in);\n";
            tbHeader << "        dut.data_out(data_out);\n";
            tbHeader << "        dut.valid(valid);\n\n";
            tbHeader << "        // Test process\n";
            tbHeader << "        SC_THREAD(stimulus);\n";
            tbHeader << "        SC_THREAD(monitor);\n";
            tbHeader << "    }\n\n";
            tbHeader << "private:\n";
            tbHeader << "    void stimulus();\n";
            tbHeader << "    void monitor();\n";
            tbHeader << "};\n\n";
            tbHeader << "#endif\n";
            tbHeader.close();
            
            // Generate testbench implementation
            std::ofstream tbImpl(tbImplPath);
            if (!tbImpl.is_open()) {
                LOG_ERROR("Failed to create testbench implementation file: {}", tbImplPath);
                return;
            }
            
            tbImpl << "#include \"" << tbName << ".h\"\n\n";
            tbImpl << "void " << tbName << "::stimulus() {\n";
            tbImpl << "    // Reset sequence\n";
            tbImpl << "    " << options_.resetSignal << ".write(SC_LOGIC_1);\n";
            tbImpl << "    data_in.write(0);\n";
            tbImpl << "    wait(50, SC_NS);\n";
            tbImpl << "    " << options_.resetSignal << ".write(SC_LOGIC_0);\n";
            tbImpl << "    wait(10, SC_NS);\n\n";
            tbImpl << "    // Test stimulus\n";
            tbImpl << "    for (int i = 0; i < 10; i++) {\n";
            tbImpl << "        data_in.write(i);\n";
            tbImpl << "        wait(10, SC_NS);\n";
            tbImpl << "    }\n\n";
            tbImpl << "    // Finish simulation\n";
            tbImpl << "    wait(100, SC_NS);\n";
            tbImpl << "    sc_stop();\n";
            tbImpl << "}\n\n";
            tbImpl << "void " << tbName << "::monitor() {\n";
            tbImpl << "    while (true) {\n";
            tbImpl << "        wait(1, SC_NS);\n";
            tbImpl << "        if (valid.read() == SC_LOGIC_1) {\n";
            tbImpl << "            std::cout << \"Time: \" << sc_time_stamp()\n";
            tbImpl << "                      << \" Data Out: \" << data_out.read()\n";
            tbImpl << "                      << \" Valid: \" << valid.read() << std::endl;\n";
            tbImpl << "        }\n";
            tbImpl << "    }\n";
            tbImpl << "}\n\n";
            tbImpl << "// Main function for standalone testbench\n";
            tbImpl << "#ifndef DISABLE_TB_MAIN\n";
            tbImpl << "int sc_main(int argc, char* argv[]) {\n";
            tbImpl << "    " << tbName << " tb(\"testbench\");\n\n";
            tbImpl << "    // Optional VCD tracing\n";
            tbImpl << "    sc_trace_file* tf = sc_create_vcd_trace_file(\"" << options_.topModule << "_trace\");\n";
            tbImpl << "    if (tf) {\n";
            tbImpl << "        sc_trace(tf, tb." << options_.clockSignal << ", \"" << options_.clockSignal << "\");\n";
            tbImpl << "        sc_trace(tf, tb." << options_.resetSignal << ", \"" << options_.resetSignal << "\");\n";
            tbImpl << "        sc_trace(tf, tb.data_in, \"data_in\");\n";
            tbImpl << "        sc_trace(tf, tb.data_out, \"data_out\");\n";
            tbImpl << "        sc_trace(tf, tb.valid, \"valid\");\n";
            tbImpl << "    }\n\n";
            tbImpl << "    // Start simulation\n";
            tbImpl << "    sc_start();\n\n";
            tbImpl << "    if (tf) {\n";
            tbImpl << "        sc_close_vcd_trace_file(tf);\n";
            tbImpl << "    }\n\n";
            tbImpl << "    return 0;\n";
            tbImpl << "}\n";
            tbImpl << "#endif\n";
            tbImpl.close();
            
            LOG_INFO("Successfully generated testbench files: {} and {}", tbHeaderPath, tbImplPath);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Testbench generation failed: {}", e.what());
        }
    }
    
    bool processFile(const std::string& inputFile) {
        LOG_INFO("Processing file: {}", inputFile);
        
        try {
            // Create source manager with include paths
            slang::SourceManager sourceManager;
            for (const auto& includePath : options_.includePaths) {
                LOG_DEBUG("Adding include directory: {}", includePath);
                // Note: slang SourceManager doesn't have addIncludeDirectory method
                // This would need to be handled at compilation level
            }
            
            // Load the file
            std::ifstream file(inputFile);
            if (!file.is_open()) {
                std::string error = fmt::format("Cannot open input file: {}", inputFile);
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            file.close();
            
            // Create syntax tree with enhanced options
            auto syntaxTree = slang::syntax::SyntaxTree::fromText(
                std::string_view(content), std::string_view(inputFile), std::string_view(inputFile));
            
            // Check for parsing errors
            if (syntaxTree->diagnostics().size() > 0) {
                LOG_WARN("Parsing diagnostics found:");
                for (const auto& diag : syntaxTree->diagnostics()) {
                    std::string diagMsg = fmt::format("{}:{} - {}", 
                        inputFile, diag.location.offset(), "Parse diagnostic");
                    LOG_WARN(diagMsg);
                    warnings_.push_back(diagMsg);
                }
            }
            
            // Create compilation with VCS options
            slang::ast::Compilation compilation = createCompilationWithOptions();
            compilation.addSyntaxTree(syntaxTree);
            
            // Create and configure code generator
            codegen::SystemCCodeGenerator generator;
            configureSystemCGenerator(generator);
            
            // Create visitor and process the compilation units
            core::SVToSCVisitor visitor(generator);
            
            // Visit the root of the compilation
            auto& root = compilation.getRoot();
            visitor.visit(root);
            
            // Generate output files
            std::string baseName = std::filesystem::path(inputFile).stem();
            std::string headerPath = options_.outputDir + "/" + baseName + ".h";
            std::string implPath = options_.outputDir + "/" + baseName + ".cpp";
            
            if (!generator.writeToFile(headerPath, implPath)) {
                std::string error = fmt::format("Failed to write output files for: {}", inputFile);
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            // Generate testbench if requested
            if (options_.generateTestbench) {
                generateTestbench(generator);
            }
            
            LOG_INFO("Successfully translated {} to SystemC", inputFile);
            return true;
            
        } catch (const std::exception& e) {
            std::string error = fmt::format("Error processing {}: {}", inputFile, e.what());
            LOG_ERROR(error);
            errors_.push_back(error);
            return false;
        }
    }
};

SystemVerilogToSystemCTranslator::SystemVerilogToSystemCTranslator(const TranslationOptions& options)
    : pImpl(std::make_unique<Impl>(options)) {
}

SystemVerilogToSystemCTranslator::~SystemVerilogToSystemCTranslator() = default;

bool SystemVerilogToSystemCTranslator::translate() {
    return pImpl->translate();
}

const std::vector<std::string>& SystemVerilogToSystemCTranslator::getErrors() const {
    return pImpl->getErrors();
}

const std::vector<std::string>& SystemVerilogToSystemCTranslator::getWarnings() const {
    return pImpl->getWarnings();
}

} // namespace sv2sc