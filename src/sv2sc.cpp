#include "sv2sc/sv2sc.h"
#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <slang/syntax/SyntaxTree.h>
#include <slang/text/SourceManager.h>
#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/util/Bag.h>
#include <slang/parsing/Preprocessor.h>
#include <filesystem>
#include <functional>
#include <fstream>
#include <set>

namespace sv2sc {

class SystemVerilogToSystemCTranslator::Impl {
public:
    explicit Impl(const TranslationOptions& options) : options_(options) {
        setupLogger();
    }
    
    bool translate() {
        try {
            LOG_INFO("Starting SystemVerilog to SystemC translation");
            
            // Always use multi-module design-level compilation for better module handling
            return processDesign();
            
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
            
            // Elaborate the design - force elaboration of all definitions
            LOG_INFO("Elaborating all definitions in the design");
            
            // Get all definitions and try to elaborate each one
            auto allDefs = compilation.getDefinitions();
            for (const auto* symbol : allDefs) {
                if (symbol->kind == slang::ast::SymbolKind::Definition) {
                    auto& def = symbol->as<slang::ast::DefinitionSymbol>();
                    if (def.definitionKind == slang::ast::DefinitionKind::Module) {
                        LOG_DEBUG("Attempting to elaborate definition: {}", def.name);
                        // This should create instances that we can find later
                    }
                }
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
            
            // Get compilation root
            auto& root = compilation.getRoot();
            
            // Find all module definitions in the compilation using getDefinitions()
            std::vector<const slang::ast::InstanceBodySymbol*> moduleDefinitions;
            std::set<std::string> moduleNames; // Track unique module names to avoid duplicates
            
            // First approach: Find instantiated modules (these are properly elaborated)
            LOG_INFO("Searching for instantiated modules");
            std::function<void(const slang::ast::Symbol&)> findInstancedModules = [&](const slang::ast::Symbol& symbol) {
                if (symbol.kind == slang::ast::SymbolKind::Instance) {
                    auto& inst = symbol.as<slang::ast::InstanceSymbol>();
                    std::string moduleName = std::string(inst.body.getDefinition().name);
                    
                    // Skip if we've already processed this module
                    if (moduleNames.find(moduleName) == moduleNames.end()) {
                        LOG_DEBUG("Found module definition from instance: {}", moduleName);
                        moduleNames.insert(moduleName);
                        moduleDefinitions.push_back(&inst.body);
                    }
                }
                
                // Recursively search child symbols if this symbol is a scope
                if (symbol.isScope()) {
                    auto& scope = symbol.as<slang::ast::Scope>();
                    for (auto& child : scope.members()) {
                        findInstancedModules(child);
                    }
                }
            };
            findInstancedModules(root);
            
            // Second approach: Create individual elaborations for missing modules
            LOG_INFO("Getting all module definitions to check for missing modules");
            auto allDefinitions = compilation.getDefinitions();
            std::vector<std::string> missingModules;
            
            for (const auto* symbol : allDefinitions) {
                if (symbol->kind == slang::ast::SymbolKind::Definition) {
                    auto& def = symbol->as<slang::ast::DefinitionSymbol>();
                    if (def.definitionKind == slang::ast::DefinitionKind::Module) {
                        std::string moduleName = std::string(def.name);
                        
                        // Check if this module is missing from our instances
                        if (moduleNames.find(moduleName) == moduleNames.end()) {
                            missingModules.push_back(moduleName);
                            LOG_DEBUG("Module {} needs individual elaboration", moduleName);
                        }
                    }
                }
            }
            
            // For each missing module, create a separate compilation with it as top module
            LOG_INFO("Creating individual elaborations for {} missing modules", missingModules.size());
            std::vector<std::unique_ptr<slang::ast::Compilation>> tempCompilations; // Keep alive
            
            for (const auto& moduleName : missingModules) {
                LOG_DEBUG("Creating individual elaboration for: {}", moduleName);
                
                try {
                    // Create compilation options with this module as top
                    slang::ast::CompilationOptions compOpts;
                    compOpts.topModules.insert(moduleName);
                    
                    // Create option bag with compilation options
                    slang::Bag optionBag;
                    optionBag.set(compOpts);
                    
                    // Add preprocessor options to maintain consistency
                    slang::parsing::PreprocessorOptions preprocessorOpts;
                    for (const auto& [name, value] : options_.defineMap) {
                        std::string defineStr = name + "=" + value;
                        preprocessorOpts.predefines.emplace_back(defineStr);
                    }
                    for (const auto& includePath : options_.includePaths) {
                        preprocessorOpts.additionalIncludePaths.emplace_back(includePath);
                    }
                    optionBag.set(preprocessorOpts);
                    
                    // Create temporary compilation with its own SourceManager
                    auto tempComp = std::make_unique<slang::ast::Compilation>(optionBag);
                    
                    // Add all source files to the temporary compilation using fromFile directly
                    for (const auto& inputFile : options_.inputFiles) {
                        // Use fromFile to let slang manage file paths properly 
                        auto syntaxTreeResult = slang::syntax::SyntaxTree::fromFile(std::string_view(inputFile));
                        if (syntaxTreeResult.has_value()) {
                            auto syntaxTree = syntaxTreeResult.value();
                            if (syntaxTree->diagnostics().empty()) {
                                tempComp->addSyntaxTree(syntaxTree);
                            } else {
                                LOG_WARN("Parse errors in {} for temp compilation {}, but continuing", inputFile, moduleName);
                                tempComp->addSyntaxTree(syntaxTree);  // Add anyway, may still work
                            }
                        } else {
                            LOG_WARN("Failed to load {} for temp compilation {}", inputFile, moduleName);
                        }
                    }
                    
                    // Force elaboration by calling getRoot()
                    auto& tempRoot = tempComp->getRoot();
                    
                    // Search for the elaborated module in the temporary compilation
                    std::function<void(const slang::ast::Symbol&)> findElaboratedModule = [&](const slang::ast::Symbol& symbol) {
                        if (symbol.kind == slang::ast::SymbolKind::Instance) {
                            auto& inst = symbol.as<slang::ast::InstanceSymbol>();
                            std::string instModuleName = std::string(inst.body.getDefinition().name);
                            
                            if (instModuleName == moduleName && 
                                moduleNames.find(instModuleName) == moduleNames.end()) {
                                LOG_DEBUG("Found elaborated module: {}", instModuleName);
                                moduleNames.insert(instModuleName);
                                moduleDefinitions.push_back(&inst.body);
                            }
                        }
                        
                        if (symbol.isScope()) {
                            auto& scope = symbol.as<slang::ast::Scope>();
                            for (auto& child : scope.members()) {
                                findElaboratedModule(child);
                            }
                        }
                    };
                    findElaboratedModule(tempRoot);
                    
                    // Keep the compilation alive to prevent pointer invalidation
                    tempCompilations.push_back(std::move(tempComp));
                    
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to elaborate module {}: {}", moduleName, e.what());
                    // Continue with other modules
                }
            }
            
            if (moduleDefinitions.empty()) {
                std::string error = "No module definitions found in design";
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            LOG_INFO("Found {} module definitions", moduleDefinitions.size());
            
            // Check if specified top module exists
            bool foundTopModule = options_.topModule.empty();
            if (!options_.topModule.empty()) {
                for (auto* moduleDef : moduleDefinitions) {
                    if (std::string(moduleDef->getDefinition().name) == options_.topModule) {
                        foundTopModule = true;
                        break;
                    }
                }
                
                if (!foundTopModule) {
                    std::string error = fmt::format("Top module '{}' not found in design", options_.topModule);
                    LOG_ERROR(error);
                    errors_.push_back(error);
                    return false;
                }
            }
            
            // Process each module definition separately
            core::SVToSCVisitor visitor(generator);
            int modulesProcessed = 0;
            
            for (auto* moduleDef : moduleDefinitions) {
                std::string moduleName = std::string(moduleDef->getDefinition().name);
                LOG_INFO("Processing module: {}", moduleName);
                
                // Visit the module definition directly instead of the entire root
                visitor.visit(*moduleDef);
                
                modulesProcessed++;
            }
            
            LOG_INFO("Processed {} module definitions", modulesProcessed);
            
            // Write all generated modules to separate files
            if (!generator.writeAllModuleFiles(options_.outputDir)) {
                std::string error = "Failed to write module files";
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            // Generate main header file including all modules
            if (!generator.generateMainHeader(options_.outputDir)) {
                LOG_WARN("Failed to generate main header file, but continuing...");
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