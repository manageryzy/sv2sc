#include "mlir/MLIRTranslator.h"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// CIRCT includes  
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Target/ExportSystemC.h"

// sv2sc SystemC emitter
#include "mlir/systemc/SystemCEmitter.h"


// Slang includes
#include <slang/syntax/SyntaxTree.h>
#include <slang/text/SourceManager.h>
#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>

// Logging
#include "utils/logger.h"

// Standard library
#include <fstream>
#include <filesystem>

namespace sv2sc::mlir_support {

MLIRSystemVerilogToSystemCTranslator::MLIRSystemVerilogToSystemCTranslator(const TranslationOptions& options)
    : options_(options) {
    LOG_INFO("Initializing MLIR-based SystemVerilog to SystemC translator");
}

bool MLIRSystemVerilogToSystemCTranslator::translate() {
    LOG_INFO("Starting MLIR-based translation pipeline");
    
    try {
        // Initialize MLIR infrastructure
        if (!initializeMLIRInfrastructure()) {
            addError("Failed to initialize MLIR infrastructure");
            return false;
        }
        
        // Process the design using MLIR pipeline
        if (!processDesignWithMLIR()) {
            addError("MLIR design processing failed");
            return false;
        }
        
        LOG_INFO("MLIR-based translation completed successfully");
        return true;
        
    } catch (const std::exception& e) {
        addError(fmt::format("MLIR translation exception: {}", e.what()));
        return false;
    }
}

bool MLIRSystemVerilogToSystemCTranslator::initializeMLIRInfrastructure() {
    LOG_DEBUG("Initializing MLIR infrastructure");
    
    // Create MLIR context manager
    contextManager_ = std::make_unique<MLIRContextManager>();
    
    if (!contextManager_->isInitialized()) {
        addError("Failed to initialize MLIR context and dialects");
        return false;
    }
    
    // Create SVToHW builder
    hwBuilder_ = std::make_unique<SVToHWBuilder>(&contextManager_->getContext());
    
    // Create pass pipeline
    passPipeline_ = std::make_unique<SV2SCPassPipeline>();
    
    // Configure pass pipeline based on options
    passPipeline_->enableTiming(options_.enableVerbose);
    passPipeline_->enableDiagnostics(options_.enableMLIRDiagnostics);
    passPipeline_->enableIRDumping(options_.dumpMLIR);
    
    LOG_INFO("MLIR infrastructure initialized successfully");
    return true;
}

bool MLIRSystemVerilogToSystemCTranslator::processDesignWithMLIR() {
    LOG_INFO("Processing design with MLIR pipeline (simplified implementation)");
    
    if (options_.inputFiles.empty()) {
        addError("No input files specified");
        return false;
    }
    
    // For now, process each file individually
    // TODO: Implement proper multi-file design-level processing
    for (const auto& inputFile : options_.inputFiles) {
        if (!processFileWithMLIR(inputFile)) {
            addError(fmt::format("Failed to process file: {}", inputFile));
            return false;
        }
    }
    
    return true;
}

bool MLIRSystemVerilogToSystemCTranslator::processFileWithMLIR(const std::string& inputFile) {
    LOG_INFO("Processing file with MLIR: {}", inputFile);
    
    try {
        // Load SystemVerilog file
        std::ifstream file(inputFile);
        if (!file.is_open()) {
            addError(fmt::format("Cannot open input file: {}", inputFile));
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse with slang
        auto syntaxTree = slang::syntax::SyntaxTree::fromText(
            std::string_view(content), std::string_view(inputFile), std::string_view(inputFile));
        
        if (syntaxTree->diagnostics().size() > 0) {
            addWarning(fmt::format("Parse diagnostics found in {}", inputFile));
            for (const auto& diag : syntaxTree->diagnostics()) {
                addWarning(fmt::format("{}:{} - Parse diagnostic", inputFile, diag.location.offset()));
            }
        }
        
        // Create compilation
        slang::ast::Compilation compilation;
        compilation.addSyntaxTree(syntaxTree);
        
        // Get compilation root and find modules
        auto& root = compilation.getRoot();
        
        // Find module definitions
        std::vector<const slang::ast::InstanceBodySymbol*> moduleDefinitions;
        std::function<void(const slang::ast::Symbol&)> findModules = [&](const slang::ast::Symbol& symbol) {
            if (symbol.kind == slang::ast::SymbolKind::Instance) {
                auto& inst = symbol.as<slang::ast::InstanceSymbol>();
                moduleDefinitions.push_back(&inst.body);
            }
            if (symbol.isScope()) {
                auto& scope = symbol.as<slang::ast::Scope>();
                for (auto& child : scope.members()) {
                    findModules(child);
                }
            }
        };
        findModules(root);
        
        if (moduleDefinitions.empty()) {
            // Try alternative approach: look for module definitions directly
            auto allDefinitions = compilation.getDefinitions();
            for (const auto* symbol : allDefinitions) {
                if (symbol->kind == slang::ast::SymbolKind::Definition) {
                    auto& def = symbol->as<slang::ast::DefinitionSymbol>();
                    if (def.definitionKind == slang::ast::DefinitionKind::Module) {
                        // Create a temporary compilation to elaborate this module
                        try {
                            slang::ast::CompilationOptions compOpts;
                            compOpts.topModules.insert(std::string(def.name));
                            slang::Bag optionBag;
                            optionBag.set(compOpts);
                            
                            auto tempComp = std::make_unique<slang::ast::Compilation>(optionBag);
                            tempComp->addSyntaxTree(syntaxTree);
                            auto& tempRoot = tempComp->getRoot();
                            
                            // Find the elaborated module
                            std::function<void(const slang::ast::Symbol&)> findElaborated = [&](const slang::ast::Symbol& symbol) {
                                if (symbol.kind == slang::ast::SymbolKind::Instance) {
                                    auto& inst = symbol.as<slang::ast::InstanceSymbol>();
                                    if (std::string(inst.body.getDefinition().name) == std::string(def.name)) {
                                        moduleDefinitions.push_back(&inst.body);
                                    }
                                }
                                if (symbol.isScope()) {
                                    auto& scope = symbol.as<slang::ast::Scope>();
                                    for (auto& child : scope.members()) {
                                        findElaborated(child);
                                    }
                                }
                            };
                            findElaborated(tempRoot);
                            
                        } catch (const std::exception& e) {
                            addWarning(fmt::format("Failed to elaborate module {}: {}", def.name, e.what()));
                        }
                    }
                }
            }
        }
        
        if (moduleDefinitions.empty()) {
            addWarning(fmt::format("No module definitions found in {}", inputFile));
            return true; // Not necessarily an error, file might have other content
        }
        
        // Process each module with MLIR
        // NOTE: Each module needs its own MLIR module to avoid conflicts
        for (auto* moduleDef : moduleDefinitions) {
            LOG_INFO("Converting module to HW dialect: {}", moduleDef->getDefinition().name);
            
            try {
                // Convert SystemVerilog AST to HW dialect
                auto mlirModule = hwBuilder_->buildFromAST(*moduleDef);
                
                // Verify the module is valid before running passes
                if (!mlirModule) {
                    addError(fmt::format("Failed to build MLIR module: {}", moduleDef->getDefinition().name));
                    continue; // Skip this module but continue with others
                }
                
                // Run MLIR passes
                if (!runMLIRPasses(mlirModule)) {
                    addError(fmt::format("MLIR passes failed for module: {}", moduleDef->getDefinition().name));
                    // Continue processing other modules instead of failing completely
                    continue;
                }
                
                // Emit SystemC code
                if (!emitSystemCFromMLIR(mlirModule)) {
                    addError(fmt::format("SystemC emission failed for module: {}", moduleDef->getDefinition().name));
                    continue;
                }
                
                LOG_INFO("Successfully processed module: {}", moduleDef->getDefinition().name);
                
            } catch (const std::exception& e) {
                addError(fmt::format("Exception processing module {}: {}", 
                                   moduleDef->getDefinition().name, e.what()));
                continue; // Continue with next module
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        addError(fmt::format("Exception processing file {}: {}", inputFile, e.what()));
        return false;
    }
}

bool MLIRSystemVerilogToSystemCTranslator::runMLIRPasses(mlir::ModuleOp module) {
    LOG_INFO("Running MLIR pass pipeline");
    
    if (!module) {
        addError("Invalid MLIR module");
        return false;
    }
    
    // Debug: Print module before passes
    std::string moduleStrBefore;
    llvm::raw_string_ostream moduleStreamBefore(moduleStrBefore);
    module.print(moduleStreamBefore);
    moduleStreamBefore.flush();
    LOG_ERROR("DEBUG: Module before passes:\\n{}", moduleStrBefore);
    
    if (!passPipeline_) {
        addError("Pass pipeline not initialized");
        return false;
    }
    
    // Run the complete pass pipeline
    bool success = passPipeline_->runPipeline(module, options_.optimizationLevel);
    
    // Debug: Print module after passes
    std::string moduleStrAfter;
    llvm::raw_string_ostream moduleStreamAfter(moduleStrAfter);
    module.print(moduleStreamAfter);
    moduleStreamAfter.flush();
    LOG_ERROR("DEBUG: Module after passes:\\n{}", moduleStrAfter);
    
    if (success) {
        LOG_INFO("MLIR pass pipeline completed successfully");
    } else {
        addError("MLIR pass pipeline failed");
    }
    
    return success;
}

bool MLIRSystemVerilogToSystemCTranslator::emitSystemCFromMLIR(mlir::ModuleOp module) {
    LOG_INFO("Emitting SystemC from MLIR module using sv2sc CIRCT-compatible emitter");

    if (!module) {
        addError("Invalid MLIR module for SystemC emission");
        return false;
    }

    std::filesystem::create_directories(options_.outputDir);
    
    // Use our sv2sc CIRCT-compatible emitter as the default implementation
    LOG_INFO("Using sv2sc CIRCT-compatible emitter (default implementation)");
    return generateSystemCFromModule(module);
}

bool MLIRSystemVerilogToSystemCTranslator::generateSystemCFromModule(mlir::ModuleOp module) {
    LOG_INFO("Generating SystemC from MLIR SystemC dialect using sv2sc CIRCT-compatible emitter");
    
    // Use our CIRCT-compatible emitter for SystemC generation
    try {
        sv2sc::mlir_support::SystemCEmitter emitter;
        auto result = emitter.emitSplit(module, options_.outputDir);
        
        if (result.success) {
            LOG_INFO("sv2sc CIRCT-compatible emitter successfully generated SystemC files");
            return true;
        } else {
            LOG_WARN("sv2sc CIRCT-compatible emitter failed: {}, falling back to basic generation", result.errorMessage);
        }
    } catch (const std::exception& e) {
        LOG_WARN("sv2sc CIRCT-compatible emitter exception: {}, falling back to basic generation", e.what());
    }

    // Fallback to basic generation if emitter fails
    LOG_INFO("Using basic SystemC generation as fallback");
    
    // Extract module name
    std::string moduleName = "mlir_generated";
    
    // Find the systemc.module operation
    mlir::Operation* systemcModuleOp = nullptr;
    int operationCount = 0;
    module.walk([&](mlir::Operation* op) {
        operationCount++;
        std::string opName = op->getName().getStringRef().str();
        if (opName == "systemc.module") {
            systemcModuleOp = op;
            if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
                moduleName = nameAttr.str();
            }
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    
    LOG_INFO("Walked {} operations, found systemc.module: {}", operationCount, systemcModuleOp != nullptr);
    
    if (!systemcModuleOp) {
        addError("No systemc.module found in MLIR module");
        return false;
    }
    
    // Generate header file
    std::string headerPath = options_.outputDir + "/" + moduleName + ".h";
    std::ofstream headerFile(headerPath);
    if (!headerFile.is_open()) {
        addError(fmt::format("Cannot create SystemC header file: {}", headerPath));
        return false;
    }
    
    // Generate implementation file
    std::string implPath = options_.outputDir + "/" + moduleName + ".cpp";
    std::ofstream implFile(implPath);
    if (!implFile.is_open()) {
        addError(fmt::format("Cannot create SystemC implementation file: {}", implPath));
        return false;
    }
    
    // Write header
    headerFile << "// Generated SystemC header from MLIR SystemC dialect\n";
    headerFile << "#ifndef " << moduleName << "_H\n";
    headerFile << "#define " << moduleName << "_H\n\n";
    headerFile << "#include <systemc.h>\n\n";
    
    headerFile << "SC_MODULE(" << moduleName << ") {\n";
    
    // Extract ports from systemc.module operation
    if (auto funcType = systemcModuleOp->getAttrOfType<mlir::TypeAttr>("function_type")) {
        if (auto functionType = llvm::dyn_cast<mlir::FunctionType>(funcType.getValue())) {
            for (unsigned i = 0; i < functionType.getNumInputs(); ++i) {
                auto inputType = functionType.getInput(i);
                std::string portName = "port_" + std::to_string(i);
                headerFile << "    sc_in<sc_logic> " << portName << ";\n";
            }
            for (unsigned i = 0; i < functionType.getNumResults(); ++i) {
                auto resultType = functionType.getResult(i);
                std::string portName = "out_" + std::to_string(i);
                headerFile << "    sc_out<sc_logic> " << portName << ";\n";
            }
        }
    }
    
    headerFile << "\n    SC_CTOR(" << moduleName << ") {\n";
    headerFile << "        SC_METHOD(process);\n";
    headerFile << "        // Add sensitivity list here\n";
    headerFile << "    }\n";
    headerFile << "\nprivate:\n";
    headerFile << "    void process();\n";
    headerFile << "};\n\n";
    headerFile << "#endif // " << moduleName << "_H\n";
    headerFile.close();
    
    // Write implementation
    implFile << "// Generated SystemC implementation from MLIR SystemC dialect\n";
    implFile << "#include \"" << moduleName << ".h\"\n\n";
    implFile << "void " << moduleName << "::process() {\n";
    implFile << "    // Process implementation from MLIR SystemC dialect\n";
    implFile << "    // TODO: Convert systemc.func operations to C++ code\n";
    implFile << "}\n";
    implFile.close();
    
    LOG_INFO("Generated SystemC files using fallback approach: {} and {}", headerPath, implPath);
    return true;
}

void MLIRSystemVerilogToSystemCTranslator::addError(const std::string& error) {
    errors_.push_back(error);
    LOG_ERROR("MLIR Translator Error: {}", error);
}

void MLIRSystemVerilogToSystemCTranslator::addWarning(const std::string& warning) {
    warnings_.push_back(warning);
    LOG_WARN("MLIR Translator Warning: {}", warning);
}



// Utility functions
bool isMLIRSupportAvailable() {
    LOG_DEBUG("Checking MLIR support availability");
    
    try {
        MLIRContextManager context;
        return context.isInitialized();
    } catch (const std::exception& e) {
        LOG_ERROR("MLIR support check failed: {}", e.what());
        return false;
    }
}

std::string getMLIRVersionInfo() {
    return "MLIR/CIRCT support enabled (sv2sc integration)";
}

} // namespace sv2sc::mlir_support