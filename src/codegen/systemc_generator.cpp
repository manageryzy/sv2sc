#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <fmt/format.h>
#include <fstream>
#include <algorithm>

namespace sv2sc::codegen {

SystemCCodeGenerator::SystemCCodeGenerator() {
}

void SystemCCodeGenerator::beginModule(const std::string& moduleName) {
    currentModule_ = moduleName;
    
    // Check if this module already exists to prevent duplicates
    if (modules_.find(moduleName) != modules_.end()) {
        LOG_DEBUG("Skipping duplicate module generation: {}", moduleName);
        return;
    }
    
    // Create new module data
    auto moduleData = std::make_unique<ModuleData>();
    moduleData->name = moduleName;
    modules_[moduleName] = std::move(moduleData);
    
    LOG_INFO("Starting SystemC code generation for module: {}", moduleName);
    
    // Get current module data and initialize header structure
    auto* module = getCurrentModuleData();
    if (module) {
        // Note: We'll generate includes and forward declarations when the module is complete
        // For now, just start the basic structure
        module->headerCode << fmt::format("#ifndef {}_H\n", moduleName);
        module->headerCode << fmt::format("#define {}_H\n\n", moduleName);
        module->headerCode << "#include <systemc.h>\n\n";
        
        // Forward declarations and includes will be inserted here in endModule()
        
        module->headerCode << fmt::format("SC_MODULE({}) {{\npublic:\n", moduleName);
        
        // Implementation constructor start
        module->implCode << fmt::format("#include \"{}.h\"\n\n", moduleName);
        module->implCode << fmt::format("{}::{}", moduleName, moduleName);
    }
}

void SystemCCodeGenerator::endModule() {
    auto* module = getCurrentModuleData();
    if (!module) {
        LOG_DEBUG("No current module data found for: {} (likely a duplicate)", currentModule_);
        return;
    }
    
    if (module->isComplete) {
        LOG_DEBUG("Module {} is already complete, skipping", currentModule_);
        return;
    }
    
    // Complete header with proper separation
    module->headerCode << "\n";
    module->headerCode << getIndent() << "SC_HAS_PROCESS(" << currentModule_ << ");\n";
    module->headerCode << getIndent() << currentModule_ << "(sc_module_name name);\n";
    module->headerCode << generateProcessMethodsForModule(*module);
    module->headerCode << "};\n\n";
    module->headerCode << "#endif\n";
    
    // Mark module as complete
    module->isComplete = true;
    
    // Implementation is now generated separately in generateModuleImplementation()
    // No need to populate implCode here as it's generated on demand
    
    LOG_INFO("Completed SystemC code generation for module: {}", currentModule_);
}

void SystemCCodeGenerator::addPort(const Port& port) {
    if (isSkippingModule()) return;
    
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->ports.push_back(port);
    
    std::string portDecl = generatePortDeclaration(port);
    module->headerCode << getIndent() << portDecl << ";\n";
    
    LOG_DEBUG("Added port declaration: {}", portDecl);
}

void SystemCCodeGenerator::addSignal(const Signal& signal) {
    if (isSkippingModule()) return;
    
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // If this is the first signal and we have parameters, add them before signals
    if (module->signals.empty() && !module->parameters.empty()) {
        module->headerCode << "\n" << getIndent() << "// Parameters\n";
        for (const auto& param : module->parameters) {
            module->headerCode << getIndent() << "static const int " << param.name << " = " << param.value << ";\n";
            LOG_DEBUG("Added parameter declaration: {} = {}", param.name, param.value);
        }
    }
    
    module->signals.push_back(signal);
    
    std::string signalDecl = generateSignalDeclaration(signal);
    module->headerCode << getIndent() << signalDecl << ";\n";
    
    LOG_DEBUG("Added signal declaration: {}", signalDecl);
}

std::string SystemCCodeGenerator::addModuleInstance(const std::string& instanceName, const std::string& moduleName) {
    auto* module = getCurrentModuleData();
    if (!module) return "";
    
    // Make instance names unique by appending index if duplicates exist
    std::string uniqueInstanceName = instanceName;
    int index = 0;
    
    // Check if this instance name already exists
    bool nameExists = false;
    do {
        nameExists = false;
        for (const auto& existingInstance : module->instances) {
            if (existingInstance.instanceName == uniqueInstanceName) {
                nameExists = true;
                uniqueInstanceName = fmt::format("{}_{}", instanceName, index);
                index++;
                break;
            }
        }
    } while (nameExists);
    
    ModuleInstance instance;
    instance.instanceName = uniqueInstanceName;
    instance.moduleName = moduleName;
    
    module->instances.push_back(instance);
    module->dependencies.insert(moduleName);
    
    // Add instance declaration to header
    module->headerCode << getIndent() << moduleName << "* " << uniqueInstanceName << ";\n";
    
    LOG_DEBUG("Added module instance: {} of type {} (unique name: {})", instanceName, moduleName, uniqueInstanceName);
    
    return uniqueInstanceName;
}

void SystemCCodeGenerator::addModuleDependency(const std::string& dependencyModule) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->dependencies.insert(dependencyModule);
    LOG_DEBUG("Added module dependency: {}", dependencyModule);
}

void SystemCCodeGenerator::addPortConnection(const std::string& instanceName, const std::string& portName, const std::string& signalName) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Find the instance and add the connection
    for (auto& instance : module->instances) {
        if (instance.instanceName == instanceName) {
            instance.portConnections.emplace_back(portName, signalName);
            LOG_DEBUG("Connected port {}.{} to signal {}", instanceName, portName, signalName);
            return;
        }
    }
    
    LOG_WARN("Instance {} not found for port connection", instanceName);
}

void SystemCCodeGenerator::updateSignalType(const std::string& signalName, bool preferArithmetic) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Find the signal and update its type preference
    for (auto& signal : module->signals) {
        if (signal.name == signalName) {
            bool wasArithmetic = signal.preferArithmetic;
            signal.preferArithmetic = preferArithmetic;
            
            if (wasArithmetic != preferArithmetic) {
                LOG_DEBUG("Updated signal '{}' type preference to arithmetic: {}", signalName, preferArithmetic);
                
                // Note: Full regeneration is complex, for now just log the change
                // TODO: Implement full header regeneration
            }
            return;
        }
    }
}

void SystemCCodeGenerator::addHeaderComment(const std::string& comment) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->headerCode << comment << "\n";
    LOG_DEBUG("Added header comment: {}", comment);
}

void SystemCCodeGenerator::addBlockingAssignment(const std::string& lhs, const std::string& rhs) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Sanitize the RHS expression to handle don't-care values
    std::string sanitizedRhs = sanitizeExpression(rhs);
    
    // For SystemC, we need to check if lhs is a signal and use write() method
    // For now, assume signals need .write() and ports use direct assignment
    if (lhs != "unknown_expr" && sanitizedRhs != "unknown_expr") {
        module->processCode << fmt::format("{}        {}.write({});\n", getIndent(), lhs, sanitizedRhs);
    } else {
        module->processCode << fmt::format("{}        // Skipping assignment: {} = {}\n", getIndent(), lhs, sanitizedRhs);
    }
    LOG_DEBUG("Added blocking assignment: {} = {}", lhs, sanitizedRhs);
}

void SystemCCodeGenerator::addNonBlockingAssignment(const std::string& lhs, const std::string& rhs) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Sanitize the RHS expression to handle don't-care values
    std::string sanitizedRhs = sanitizeExpression(rhs);
    
    module->processCode << fmt::format("{}        {}.write({});\n", getIndent(), lhs, sanitizedRhs);
    LOG_DEBUG("Added non-blocking assignment: {} <= {}", lhs, sanitizedRhs);
}

void SystemCCodeGenerator::addDelayedAssignment(const std::string& lhs, const std::string& rhs, const std::string& delay) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->processCode << fmt::format("{}        wait({});\n", getIndent(), delay);
    module->processCode << fmt::format("{}        {}.write({});\n", getIndent(), lhs, rhs);
    LOG_DEBUG("Added delayed assignment: {} <= {} after {}", lhs, rhs, delay);
}

void SystemCCodeGenerator::addCombinationalAssignment(const std::string& lhs, const std::string& rhs) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Sanitize the RHS expression to handle don't-care values
    std::string sanitizedRhs = sanitizeExpression(rhs);
    
    if (lhs != "unknown_expr" && sanitizedRhs != "unknown_expr") {
        std::string convertedRhs = applyTypeConversion(lhs, sanitizedRhs);
        module->combProcessCode << fmt::format("{}        {}.write({});\n", getIndent(), lhs, convertedRhs);
        module->hasCombProcess = true;
    } else {
        module->combProcessCode << fmt::format("{}        // Skipping combinational assignment: {} = {}\n", getIndent(), lhs, sanitizedRhs);
    }
    LOG_DEBUG("Added combinational assignment: {} = {}", lhs, sanitizedRhs);
}

void SystemCCodeGenerator::addSequentialAssignment(const std::string& lhs, const std::string& rhs) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Sanitize the RHS expression to handle don't-care values
    std::string sanitizedRhs = sanitizeExpression(rhs);
    
    if (lhs != "unknown_expr" && sanitizedRhs != "unknown_expr") {
        std::string convertedRhs = applyTypeConversion(lhs, sanitizedRhs);
        module->seqProcessCode << fmt::format("{}        {}.write({});\n", getIndent(), lhs, convertedRhs);
        module->hasSeqProcess = true;
    } else {
        module->seqProcessCode << fmt::format("{}        // Skipping sequential assignment: {} <= {}\n", getIndent(), lhs, sanitizedRhs);
    }
    LOG_DEBUG("Added sequential assignment: {} <= {}", lhs, sanitizedRhs);
}

void SystemCCodeGenerator::beginGenerateBlock(const std::string& label) {
    addComment(fmt::format("Generate block: {}", label.empty() ? "unnamed" : label));
}

void SystemCCodeGenerator::endGenerateBlock() {
    addComment("End generate block");
}

void SystemCCodeGenerator::addComment(const std::string& comment) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->headerCode << getIndent() << "// " << comment << "\n";
}

void SystemCCodeGenerator::addRawCode(const std::string& code) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->headerCode << code;
}

void SystemCCodeGenerator::beginConditional(const std::string& condition) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    module->processCode << fmt::format("{}        if ({}) {{\n", getIndent(), condition);
    indentLevel_++;
}

void SystemCCodeGenerator::addElse() {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    indentLevel_--;
    module->processCode << fmt::format("{}        }} else {{\n", getIndent());
    indentLevel_++;
}

void SystemCCodeGenerator::endConditional() {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    indentLevel_--;
    module->processCode << fmt::format("{}        }}\n", getIndent());
}

std::string SystemCCodeGenerator::generateHeader() const {
    // For backward compatibility, return the first completed module or empty
    for (const auto& pair : modules_) {
        if (pair.second->isComplete) {
            return pair.second->headerCode.str();
        }
    }
    return "";
}

std::string SystemCCodeGenerator::generateImplementation() const {
    // For backward compatibility, return the first completed module or empty
    for (const auto& pair : modules_) {
        if (pair.second->isComplete) {
            std::string impl = pair.second->implCode.str();
            if (impl.empty()) {
                return fmt::format("// {}.cpp - Header-only SystemC implementation\n// All implementation is contained in {}.h\n", pair.first, pair.first);
            }
            return impl;
        }
    }
    return "";
}

bool SystemCCodeGenerator::writeToFile(const std::string& headerPath, const std::string& implPath) const {
    // Write header file
    std::ofstream headerFile(headerPath);
    if (!headerFile.is_open()) {
        LOG_ERROR("Failed to open header file for writing: {}", headerPath);
        return false;
    }
    headerFile << generateHeader();
    headerFile.close();
    
    // Write implementation file
    std::ofstream implFile(implPath);
    if (!implFile.is_open()) {
        LOG_ERROR("Failed to open implementation file for writing: {}", implPath);
        return false;
    }
    implFile << generateImplementation();
    implFile.close();
    
    LOG_INFO("Successfully wrote files: {} and {}", headerPath, implPath);
    return true;
}

// New helper methods for multi-module support
ModuleData* SystemCCodeGenerator::getCurrentModuleData() {
    if (currentModule_.empty()) return nullptr;
    auto it = modules_.find(currentModule_);
    return (it != modules_.end()) ? it->second.get() : nullptr;
}

const ModuleData* SystemCCodeGenerator::getCurrentModuleData() const {
    if (currentModule_.empty()) return nullptr;
    auto it = modules_.find(currentModule_);
    return (it != modules_.end()) ? it->second.get() : nullptr;
}

std::string SystemCCodeGenerator::generateModuleHeader(const std::string& moduleName) const {
    auto it = modules_.find(moduleName);
    if (it == modules_.end() || !it->second->isComplete) {
        return "";
    }
    
    const auto& module = *it->second;
    std::stringstream header;
    
    // Header guard
    header << fmt::format("#ifndef {}_H\n", moduleName);
    header << fmt::format("#define {}_H\n\n", moduleName);
    
    // System includes
    header << "#include <systemc.h>\n\n";
    
    // Forward declarations
    header << generateForwardDeclarations(module);
    
    // User includes (if needed)
    header << generateIncludes(module);
    
    // Module declaration
    header << fmt::format("SC_MODULE({}) {{\npublic:\n", moduleName);
    
    // Ports
    for (const auto& port : module.ports) {
        header << "    " << generatePortDeclaration(port) << ";\n";
    }
    
    // Parameters
    LOG_DEBUG("Header generation: module has {} parameters", module.parameters.size());
    if (!module.parameters.empty()) {
        header << "\n    // Parameters\n";
        for (const auto& param : module.parameters) {
            header << "    static const int " << param.name << " = " << param.value << ";\n";
            LOG_DEBUG("Generated parameter: {} = {}", param.name, param.value);
        }
    } else {
        LOG_DEBUG("No parameters to generate in header");
    }
    
    // Signals
    if (!module.signals.empty()) {
        header << "\n    // Internal signals\n";
        for (const auto& signal : module.signals) {
            header << "    " << generateSignalDeclaration(signal) << ";\n";
        }
    }
    
    // Module instances
    header << generateModuleInstances(module);
    
    // Constructor - use SC_HAS_PROCESS for proper separation
    header << "\n    SC_HAS_PROCESS(" << moduleName << ");\n";
    header << "    " << moduleName << "(sc_module_name name);\n";
    
    // Process methods
    header << generateProcessMethodsForModule(module);
    
    // Close module
    header << "};\n\n";
    header << "#endif\n";
    
    return header.str();
}

std::string SystemCCodeGenerator::generateModuleImplementation(const std::string& moduleName) const {
    auto it = modules_.find(moduleName);
    if (it == modules_.end() || !it->second->isComplete) {
        return "";
    }
    
    const auto& module = *it->second;
    std::stringstream impl;
    
    // Include the header
    impl << "#include \"" << moduleName << ".h\"\n\n";
    
    // Generate constructor implementation
    impl << moduleName << "::" << moduleName << "(sc_module_name name) : sc_module(name) {\n";
    impl << generateConstructorForModule(module);
    impl << "}\n\n";
    
    // Generate process method implementations
    impl << generateProcessMethodImplementations(module, moduleName);
    
    // Add any additional implementation from implCode if present
    std::string additionalImpl = module.implCode.str();
    if (!additionalImpl.empty()) {
        impl << "\n// Additional implementation\n";
        impl << additionalImpl;
    }
    
    return impl.str();
}

std::vector<std::string> SystemCCodeGenerator::getGeneratedModuleNames() const {
    std::vector<std::string> names;
    for (const auto& pair : modules_) {
        if (pair.second->isComplete) {
            names.push_back(pair.first);
        }
    }
    return names;
}

bool SystemCCodeGenerator::writeModuleFiles(const std::string& moduleName, const std::string& outputDir) const {
    auto it = modules_.find(moduleName);
    if (it == modules_.end() || !it->second->isComplete) {
        LOG_ERROR("Module {} not found or not complete", moduleName);
        return false;
    }

    std::string headerPath = outputDir + "/" + moduleName + ".h";
    std::string implPath = outputDir + "/" + moduleName + ".cpp";

    // Write header file
    std::ofstream headerFile(headerPath);
    if (!headerFile.is_open()) {
        LOG_ERROR("Failed to open header file for writing: {}", headerPath);
        return false;
    }
    headerFile << it->second->headerCode.str();
    headerFile.close();

    // Write implementation file
    std::ofstream implFile(implPath);
    if (!implFile.is_open()) {
        LOG_ERROR("Failed to open implementation file for writing: {}", implPath);
        return false;
    }
    std::string impl = it->second->implCode.str();
    if (impl.empty()) {
        impl = fmt::format("// {}.cpp - Header-only SystemC implementation\n// All implementation is contained in {}.h\n", moduleName, moduleName);
    }
    implFile << impl;
    implFile.close();

    LOG_INFO("Successfully wrote module files: {} and {}", headerPath, implPath);
    return true;
}

bool SystemCCodeGenerator::writeAllModuleFiles(const std::string& outputDir) const {
    LOG_INFO("Writing all module files to directory: {}", outputDir);
    
    bool success = true;
    int filesWritten = 0;
    
    for (const auto& pair : modules_) {
        const auto& moduleName = pair.first;
        const auto& module = pair.second;
        
        if (!module->isComplete) {
            LOG_WARN("Skipping incomplete module: {}", moduleName);
            continue;
        }
        
        // Write module files using the new header generation approach
        std::string headerPath = outputDir + "/" + moduleName + ".h";
        std::string implPath = outputDir + "/" + moduleName + ".cpp";
        
        // Write header file
        std::ofstream headerFile(headerPath);
        if (!headerFile.is_open()) {
            LOG_ERROR("Failed to open header file for writing: {}", headerPath);
            success = false;
            continue;
        }
        // Use the complete header code built by endModule() instead of generateModuleHeader()
        headerFile << module->headerCode.str();
        headerFile.close();
        
        // Write implementation file
        std::ofstream implFile(implPath);
        if (!implFile.is_open()) {
            LOG_ERROR("Failed to open implementation file for writing: {}", implPath);
            success = false;
            continue;
        }
        
        std::string impl = generateModuleImplementation(moduleName);
        if (impl.empty()) {
            impl = fmt::format("// {}.cpp - Header-only SystemC implementation\n// All implementation is contained in {}.h\n", moduleName, moduleName);
        }
        implFile << impl;
        implFile.close();
        
        LOG_DEBUG("Successfully wrote module files: {} and {}", headerPath, implPath);
        filesWritten++;
    }
    
    LOG_INFO("Successfully wrote {} module files to {}", filesWritten, outputDir);
    return success;
}

bool SystemCCodeGenerator::generateMainHeader(const std::string& outputDir, const std::string& mainHeaderName) const {
    std::string headerPath = outputDir + "/" + mainHeaderName;
    
    std::ofstream headerFile(headerPath);
    if (!headerFile.is_open()) {
        LOG_ERROR("Failed to open main header file for writing: {}", headerPath);
        return false;
    }
    
    // Generate main header that includes all modules
    headerFile << "#ifndef ALL_MODULES_H\n";
    headerFile << "#define ALL_MODULES_H\n\n";
    headerFile << "// Main header including all translated SystemC modules\n";
    headerFile << "// Generated by sv2sc translator\n\n";
    
    headerFile << "#include <systemc.h>\n\n";
    
    // Include all module headers
    for (const auto& pair : modules_) {
        if (pair.second->isComplete) {
            headerFile << fmt::format("#include \"{}.h\"\n", pair.first);
        }
    }
    
    headerFile << "\n#endif\n";
    headerFile.close();
    
    LOG_INFO("Successfully wrote main header: {}", headerPath);
    return true;
}

// Helper method for generating constructor specific to a module
std::string SystemCCodeGenerator::generateConstructorForModule(const ModuleData& module) const {
    std::stringstream constructor;
    
    // Check if we have new-style process blocks
    bool hasProcessBlocks = !module.processBlocks.empty();
    
    // Register processes (new style - multiple process blocks)
    if (hasProcessBlocks) {
        constructor << getIndent() << "    // Register process blocks\n";
        
        for (const auto& block : module.processBlocks) {
            constructor << getIndent() << "    SC_METHOD(" << block.name << ");\n";
            
            // Add sensitivity
            if (!block.sensitivity.empty()) {
                constructor << getIndent() << "    sensitive";
                bool first = true;
                for (const auto& signal : block.sensitivity) {
                    if (!first) constructor << " << ";
                    else constructor << " << ";
                    
                    // Handle clock edges
                    if (signal == block.clockSignal && block.isSequential) {
                        constructor << signal << ".pos()";
                    } else {
                        constructor << signal;
                    }
                    first = false;
                }
                constructor << ";\n";
            }
            constructor << "\n";
        }
    }
    
    // Legacy process registration (if no new-style blocks)
    bool hasAnyProcesses = !hasProcessBlocks && 
                          (!module.processCode.str().empty() || 
                           module.hasCombProcess || 
                           module.hasSeqProcess);
    
    // Generate sensitivity list for processes - only if we have processes
    if (hasAnyProcesses) {
        constructor << getIndent() << "    // Process sensitivity (legacy)\n";
        
        // Add clock and reset detection
        bool hasClk = std::any_of(module.ports.begin(), module.ports.end(), 
            [](const Port& p) { return p.name == "clk" || p.name == "clock"; });
        bool hasReset = std::any_of(module.ports.begin(), module.ports.end(),
            [](const Port& p) { return p.name == "reset" || p.name == "rst" || p.name == "resetn"; });
        
        // Register combinational process
        if (module.hasCombProcess && !module.combProcessCode.str().empty()) {
            constructor << getIndent() << "    SC_METHOD(comb_proc);\n";
            
            // Use tracked sensitive signals if available, otherwise fall back to all inputs
            if (!module.combSensitiveSignals.empty()) {
                constructor << getIndent() << "    sensitive";
                bool firstSensitive = true;
                for (const auto& signalName : module.combSensitiveSignals) {
                    if (firstSensitive) {
                        constructor << " << " << signalName;
                        firstSensitive = false;
                    } else {
                        constructor << " << " << signalName;
                    }
                }
                constructor << ";\n\n";
            } else {
                // Fallback: Add sensitivity to all input ports for combinational logic
                constructor << getIndent() << "    sensitive";
                bool firstSensitive = true;
                for (const auto& port : module.ports) {
                    if (port.direction == PortDirection::INPUT) {
                        if (firstSensitive) {
                            constructor << " << " << port.name;
                            firstSensitive = false;
                        } else {
                            constructor << " << " << port.name;
                        }
                    }
                }
                constructor << ";\n\n";
            }
        }
        
        // Register sequential process  
        if (module.hasSeqProcess && !module.seqProcessCode.str().empty()) {
            constructor << getIndent() << "    SC_METHOD(seq_proc);\n";
            
            if (hasClk) {
                constructor << getIndent() << "    sensitive << clk.pos();\n";
            }
            
            if (hasReset) {
                // Determine if reset is active low or high
                bool isResetN = std::any_of(module.ports.begin(), module.ports.end(),
                    [](const Port& p) { return p.name == "resetn" || p.name == "rst_n"; });
                if (isResetN) {
                    // Active low reset - level sensitive
                    constructor << getIndent() << "    sensitive << resetn;\n";
                } else {
                    // Active high reset - level sensitive for async reset
                    constructor << getIndent() << "    sensitive << reset;\n";
                }
            }
            constructor << getIndent() << "\n";
        }
        
        // Legacy: handle old single process code for backward compatibility
        if (!module.processCode.str().empty() && !module.hasCombProcess && !module.hasSeqProcess) {
            constructor << getIndent() << "    SC_METHOD(comb_proc);\n";
            
            if (hasClk) {
                constructor << getIndent() << "    sensitive << clk.pos();\n";
            }
            
            if (hasReset) {
                constructor << getIndent() << "    sensitive << reset;\n";
            }
            
            constructor << getIndent() << "\n";
        }
    }
    
    // Initialize and connect module instances
    if (!module.instances.empty()) {
        constructor << getIndent() << "    // Initialize module instances\n";
        for (const auto& instance : module.instances) {
            constructor << getIndent() << "    " << instance.instanceName << " = new " 
                       << instance.moduleName << "(\"" << instance.instanceName << "\");\n";
            
            // Connect ports
            for (const auto& connection : instance.portConnections) {
                constructor << getIndent() << "    " << instance.instanceName 
                           << "->" << connection.first << "(" << connection.second << ");\n";
            }
        }
        constructor << getIndent() << "\n";
    }
    
    return constructor.str();
}

// Helper method for generating process method DECLARATIONS for header
std::string SystemCCodeGenerator::generateProcessMethodsForModule(const ModuleData& module) const {
    std::stringstream methods;
    
    // Check for new-style process blocks first
    bool hasProcessBlocks = !module.processBlocks.empty();
    
    if (hasProcessBlocks) {
        methods << "\nprivate:\n";
        
        // Only declare methods in header
        for (const auto& block : module.processBlocks) {
            methods << "    void " << block.name << "();\n";
        }
        
        return methods.str();
    }
    
    // Legacy process generation
    bool hasAnyProcesses = !module.processCode.str().empty() || 
                          module.hasCombProcess || 
                          module.hasSeqProcess;
    
    if (hasAnyProcesses) {
        methods << "\nprivate:\n";
        
        // Only declare methods in header
        if (module.hasCombProcess && !module.combProcessCode.str().empty()) {
            methods << "    void comb_proc();\n";
        }
        
        if (module.hasSeqProcess && !module.seqProcessCode.str().empty()) {
            methods << "    void seq_proc();\n";
        }
        
        if (!module.processCode.str().empty() && !module.hasCombProcess && !module.hasSeqProcess) {
            methods << "    void comb_proc();\n";
        }
    }
    
    return methods.str();
}

// New helper method for generating process method IMPLEMENTATIONS for .cpp file
std::string SystemCCodeGenerator::generateProcessMethodImplementations(const ModuleData& module, 
                                                                      const std::string& moduleName) const {
    std::stringstream implementations;
    
    // Check for new-style process blocks first
    bool hasProcessBlocks = !module.processBlocks.empty();
    
    if (hasProcessBlocks) {
        // Generate method implementations
        for (const auto& block : module.processBlocks) {
            implementations << "void " << moduleName << "::" << block.name << "() {\n";
            implementations << block.code.str();
            implementations << "}\n\n";
        }
        
        return implementations.str();
    }
    
    // Legacy process generation
    bool hasAnyProcesses = !module.processCode.str().empty() || 
                          module.hasCombProcess || 
                          module.hasSeqProcess;
    
    if (hasAnyProcesses) {
        // Generate method implementations
        if (module.hasCombProcess && !module.combProcessCode.str().empty()) {
            implementations << "void " << moduleName << "::comb_proc() {\n";
            implementations << module.combProcessCode.str();
            implementations << "}\n\n";
        }
        
        if (module.hasSeqProcess && !module.seqProcessCode.str().empty()) {
            implementations << "void " << moduleName << "::seq_proc() {\n";
            implementations << module.seqProcessCode.str();
            implementations << "}\n\n";
        }
        
        if (!module.processCode.str().empty() && !module.hasCombProcess && !module.hasSeqProcess) {
            implementations << "void " << moduleName << "::comb_proc() {\n";
            implementations << module.processCode.str();
            implementations << "}\n\n";
        }
    }
    
    return implementations.str();
}

// Helper methods for generating forward declarations and includes
std::string SystemCCodeGenerator::generateForwardDeclarations(const ModuleData& module) const {
    std::stringstream forward;
    
    if (!module.dependencies.empty()) {
        forward << "// Forward declarations\n";
        for (const auto& dep : module.dependencies) {
            forward << fmt::format("class {};\n", dep);
        }
        forward << "\n";
    }
    
    return forward.str();
}

std::string SystemCCodeGenerator::generateIncludes(const ModuleData& module) const {
    std::stringstream includes;
    
    // For now, we use forward declarations instead of includes to avoid circular dependencies
    // In a more sophisticated implementation, we could analyze the dependency graph
    // and include headers only when necessary
    
    return includes.str();
}

std::string SystemCCodeGenerator::generateModuleInstances(const ModuleData& module) const {
    std::stringstream instances;
    
    if (!module.instances.empty()) {
        instances << "\n    // Module instances\n";
        for (const auto& instance : module.instances) {
            instances << fmt::format("    {}* {};\n", instance.moduleName, instance.instanceName);
        }
    }
    
    return instances.str();
}

std::string SystemCCodeGenerator::getIndent() const {
    return std::string(indentLevel_ * 4, ' ');
}

std::string SystemCCodeGenerator::sanitizeExpression(const std::string& expr) const {
    std::string result = expr;
    
    // Replace SystemVerilog don't-care literals with SystemC equivalents
    // Pattern: N'dx, N'bx, N'hx where N is the width
    std::regex dontCarePattern(R"((\d+)'[dbho]x)");
    std::smatch match;
    
    while (std::regex_search(result, match, dontCarePattern)) {
        std::string fullMatch = match[0].str();
        std::string widthStr = match[1].str();
        
        try {
            int width = std::stoi(widthStr);
            std::string replacement = fmt::format("sc_lv<{}>(\"{}\")", width, std::string(width, 'X'));
            result.replace(match.position(), match.length(), replacement);
        } catch (...) {
            // If we can't parse the width, use a default
            result.replace(match.position(), match.length(), "sc_lv<32>(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\")");
        }
    }
    
    // Also handle high-impedance values (N'dz, N'bz, N'hz)
    std::regex hiZPattern(R"((\d+)'[dbho]z)");
    while (std::regex_search(result, match, hiZPattern)) {
        std::string fullMatch = match[0].str();
        std::string widthStr = match[1].str();
        
        try {
            int width = std::stoi(widthStr);
            std::string replacement = fmt::format("sc_lv<{}>(\"{}\")", width, std::string(width, 'Z'));
            result.replace(match.position(), match.length(), replacement);
        } catch (...) {
            result.replace(match.position(), match.length(), "sc_lv<32>(\"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\")");
        }
    }
    
    // Handle numeric literals (N'd0, N'b0, N'h0, etc.)
    std::regex numericPattern(R"((\d+)'([dbho])(\d+))");
    while (std::regex_search(result, match, numericPattern)) {
        std::string widthStr = match[1].str();
        std::string base = match[2].str();
        std::string valueStr = match[3].str();
        
        try {
            int width = std::stoi(widthStr);
            int value = 0;
            
            // Parse value based on base
            if (base == "d") {
                value = std::stoi(valueStr, nullptr, 10);
            } else if (base == "b") {
                value = std::stoi(valueStr, nullptr, 2);
            } else if (base == "h" || base == "o") {
                value = std::stoi(valueStr, nullptr, base == "h" ? 16 : 8);
            }
            
            // Generate appropriate SystemC literal
            if (value == 0) {
                std::string replacement = fmt::format("sc_lv<{}>(0)", width);
                result.replace(match.position(), match.length(), replacement);
            } else {
                std::string replacement = fmt::format("sc_lv<{}>({})", width, value);
                result.replace(match.position(), match.length(), replacement);
            }
        } catch (...) {
            // If we can't parse, just convert to a simple format
            std::string replacement = fmt::format("sc_lv<{}>({})", widthStr, valueStr);
            result.replace(match.position(), match.length(), replacement);
        }
    }
    
    return result;
}

std::string SystemCCodeGenerator::applyTypeConversion(const std::string& lhs, const std::string& rhs) const {
    // Handle integer to sc_logic conversions
    if (rhs == "0" || rhs == "1") {
        std::string lhsType = getSignalType(lhs);
        if (lhsType.find("sc_logic") != std::string::npos) {
            return convertIntegerToScLogic(rhs);
        }
    }
    
    // Handle signal to port conversions with type mismatches
    if (rhs.find(".read()") == std::string::npos) {
        // Check for common type conversion cases
        std::string lhsType = getSignalType(lhs);
        std::string rhsType = getSignalType(rhs);
        
        // Debug logging
        LOG_DEBUG("Type conversion: lhs='{}' ({}), rhs='{}' ({})", lhs, lhsType, rhs, rhsType);
        
        // Only proceed if we have type information
        if (!lhsType.empty() && !rhsType.empty()) {
            // sc_uint signal to sc_lv port conversion
            if (lhsType.find("sc_lv") != std::string::npos && 
                (rhsType.find("sc_uint") != std::string::npos || rhsType.find("sc_signal") != std::string::npos)) {
                // Extract bit width from lhs type (e.g., sc_lv<8>)
                size_t start = lhsType.find('<');
                size_t end = lhsType.find('>', start);
                if (start != std::string::npos && end != std::string::npos) {
                    std::string width = lhsType.substr(start + 1, end - start - 1);
                    std::string rhsWithRead = (rhs.find(".read()") == std::string::npos) ? rhs + ".read()" : rhs;
                    std::string result = fmt::format("sc_lv<{}>({})", width, rhsWithRead);
                    LOG_DEBUG("Applied sc_uint->sc_lv conversion: {} -> {}", rhs, result);
                    return result;
                }
            }
            
            // sc_lv signal to sc_uint port conversion
            if (lhsType.find("sc_uint") != std::string::npos && 
                rhsType.find("sc_lv") != std::string::npos) {
                // Extract bit width from lhs type
                size_t start = lhsType.find('<');
                size_t end = lhsType.find('>', start);
                if (start != std::string::npos && end != std::string::npos) {
                    std::string width = lhsType.substr(start + 1, end - start - 1);
                    std::string result = fmt::format("sc_uint<{}>({})", width, rhs + ".read().to_uint()");
                    LOG_DEBUG("Applied sc_lv->sc_uint conversion: {} -> {}", rhs, result);
                    return result;
                }
            }
        }
        
        // sc_signal reference to direct value (needs .read())
        if (rhsType.find("sc_signal") != std::string::npos) {
            return rhs + ".read()";
        }
        
        // If we have a signal name that's not a literal, add .read()
        if (!rhsType.empty() && rhsType.find("sc_") != std::string::npos && 
            rhs != "0" && rhs != "1" && rhs.find("(") == std::string::npos) {
            return rhs + ".read()";
        }
    }
    
    return rhs;
}

std::string SystemCCodeGenerator::convertIntegerToScLogic(const std::string& value) const {
    if (value == "0") {
        return "sc_logic('0')";
    } else if (value == "1") {
        return "sc_logic('1')";
    }
    return value;
}

std::string SystemCCodeGenerator::getSignalType(const std::string& signalName) const {
    const auto* module = getCurrentModuleData();
    if (!module) return "";
    
    // Check ports
    for (const auto& port : module->ports) {
        if (port.name == signalName) {
            return mapDataType(port.dataType, port.width);
        }
    }
    
    // Check signals
    for (const auto& signal : module->signals) {
        if (signal.name == signalName) {
            return mapDataType(signal.dataType, signal.width);
        }
    }
    
    // Default guess based on naming conventions
    if (signalName.find("mem_do_") != std::string::npos) {
        return "sc_logic";
    }
    
    return "";
}

std::string SystemCCodeGenerator::mapDataType(SystemCDataType type, int width) const {
    switch (type) {
        case SystemCDataType::SC_BIT:
            return width > 1 ? fmt::format("sc_bv<{}>", width) : "sc_bit";
        case SystemCDataType::SC_LOGIC:
            return width > 1 ? fmt::format("sc_lv<{}>", width) : "sc_logic";
        case SystemCDataType::SC_INT:
            return "sc_int<32>";
        case SystemCDataType::SC_UINT:
            return "sc_uint<32>";
        case SystemCDataType::SC_BIGINT:
            return fmt::format("sc_bigint<{}>", width);
        case SystemCDataType::SC_BIGUINT:
            return fmt::format("sc_biguint<{}>", width);
        case SystemCDataType::SC_BV:
            return fmt::format("sc_bv<{}>", width);
        case SystemCDataType::SC_LV:
            return fmt::format("sc_lv<{}>", width);
        default:
            return "sc_logic";
    }
}

std::string SystemCCodeGenerator::mapDataType(SystemCDataType type, int width, const std::string& widthExpression) const {
    // If we have a parameter expression, use it instead of the numeric width
    if (!widthExpression.empty()) {
        switch (type) {
            case SystemCDataType::SC_BIT:
                return fmt::format("sc_bv<{}>", widthExpression);
            case SystemCDataType::SC_LOGIC:
                return fmt::format("sc_lv<{}>", widthExpression);
            case SystemCDataType::SC_BV:
                return fmt::format("sc_bv<{}>", widthExpression);
            case SystemCDataType::SC_LV:
                return fmt::format("sc_lv<{}>", widthExpression);
            case SystemCDataType::SC_INT:
                return "sc_int<32>";
            case SystemCDataType::SC_UINT:
                return "sc_uint<32>";
            case SystemCDataType::SC_BIGINT:
                return "sc_bigint<64>";
            case SystemCDataType::SC_BIGUINT:
                return "sc_biguint<64>";
            default:
                return "sc_logic";
        }
    }
    
    // Fallback to numeric width
    return mapDataType(type, width);
}

std::string SystemCCodeGenerator::generatePortDeclaration(const Port& port) const {
    // Use bool for ALL single-bit ports for consistency (compatible with testbench and sc_clock)
    bool isSingleBit = (port.width == 1) && (port.direction != PortDirection::INOUT);
    
    std::string direction;
    switch (port.direction) {
        case PortDirection::INPUT:
            direction = "sc_in";
            break;
        case PortDirection::OUTPUT:
            direction = "sc_out";
            break;
        case PortDirection::INOUT:
            direction = "sc_inout";
            break;
    }
    
    std::string dataType;
    if (isSingleBit) {
        // Use bool for all single-bit ports for type consistency
        dataType = "bool";
    } else {
        dataType = mapDataType(port.dataType, port.width, port.widthExpression);
    }
    
    std::string decl = fmt::format("{}<{}> {}", direction, dataType, port.name);
    
    // Handle arrays
    if (port.isArray) {
        for (int dim : port.arrayDimensions) {
            decl = fmt::format("sc_vector<{}> {} /* [{}] */", decl, port.name, dim);
            break; // For now, handle only first dimension
        }
    }
    
    return decl;
}

std::string SystemCCodeGenerator::generateSignalDeclaration(const Signal& signal) const {
    std::string dataType;
    
    // Choose appropriate SystemC type based on usage
    if (signal.preferArithmetic && signal.width > 1) {
        // Use sc_uint for multi-bit arithmetic signals
        if (!signal.widthExpression.empty()) {
            dataType = fmt::format("sc_uint<{}>", signal.widthExpression);
        } else {
            dataType = fmt::format("sc_uint<{}>", signal.width);
        }
    } else if (signal.preferArithmetic && signal.width == 1) {
        // Use unsigned int for single-bit arithmetic
        dataType = "unsigned int";
    } else {
        // Use consistent bool type for single-bit signals (same as ports)
        bool isSingleBit = (signal.width == 1);
        if (isSingleBit) {
            dataType = "bool";
        } else {
            // Use standard logic type mapping for multi-bit non-arithmetic signals
            dataType = mapDataType(signal.dataType, signal.width, signal.widthExpression);
        }
    }
    
    // Handle arrays
    if (signal.isArray && !signal.arrayDimensions.empty()) {
        // For memory arrays, use simple C array syntax 
        // sc_signal<sc_uint<8>> mem_array[256];
        int arraySize = signal.arrayDimensions[0]; // Handle first dimension for now
        return fmt::format("sc_signal<{}> {}[{}]", dataType, signal.name, arraySize);
    }
    
    // Regular signal declaration
    return fmt::format("sc_signal<{}> {}", dataType, signal.name);
}

std::string SystemCCodeGenerator::generateConstructor() const {
    const auto* module = getCurrentModuleData();
    if (!module) return "";
    
    return generateConstructorForModule(*module);
}

std::string SystemCCodeGenerator::generateProcessMethods() const {
    const auto* module = getCurrentModuleData();
    if (!module) return "";
    
    return generateProcessMethodsForModule(*module);
}

bool SystemCCodeGenerator::isSkippingModule() const {
    // We're skipping if current module is empty or doesn't exist in our modules map
    if (currentModule_.empty()) return true;
    
    auto* module = getCurrentModuleData();
    return (module == nullptr);
}

void SystemCCodeGenerator::addConditionalStart(const std::string& condition, bool isSequential) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    std::string ifStatement = fmt::format("{}        if ({}) {{\n", getIndent(), condition);
    
    // Check if we're using process blocks
    if (currentProcessBlock_) {
        currentProcessBlock_->code << ifStatement;
        LOG_DEBUG("Added conditional start to process block {}: if ({})", 
                 currentProcessBlock_->name, condition);
    } else {
        // Legacy behavior
        if (isSequential) {
            module->hasSeqProcess = true;
            module->seqProcessCode << ifStatement;
        } else {
            module->hasCombProcess = true;
            module->combProcessCode << ifStatement;
        }
        LOG_DEBUG("Added conditional start: if ({})", condition);
    }
}

void SystemCCodeGenerator::addElseClause(bool isSequential) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    std::string elseStatement = fmt::format("{}        }} else {{\n", getIndent());
    
    // Check if we're using process blocks
    if (currentProcessBlock_) {
        currentProcessBlock_->code << elseStatement;
        LOG_DEBUG("Added else clause to process block {}", currentProcessBlock_->name);
    } else {
        // Legacy behavior
        if (isSequential) {
            module->seqProcessCode << elseStatement;
        } else {
            module->combProcessCode << elseStatement;
        }
        LOG_DEBUG("Added else clause");
    }
}

void SystemCCodeGenerator::addConditionalEnd(bool isSequential) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    std::string endStatement = fmt::format("{}        }}\n", getIndent());
    
    // Check if we're using process blocks
    if (currentProcessBlock_) {
        currentProcessBlock_->code << endStatement;
        LOG_DEBUG("Added conditional end to process block {}", currentProcessBlock_->name);
    } else {
        // Legacy behavior
        if (isSequential) {
            module->seqProcessCode << endStatement;
        } else {
            module->combProcessCode << endStatement;
        }
        LOG_DEBUG("Added conditional end");
    }
}

void SystemCCodeGenerator::addCombSensitiveSignal(const std::string& signalName) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Skip clock, reset signals, and parameters for combinational sensitivity
    bool isParameter = std::any_of(module->parameters.begin(), module->parameters.end(),
        [&signalName](const Parameter& p) { return p.name == signalName; });
    
    if (signalName != "clk" && signalName != "clock" && 
        signalName != "reset" && signalName != "resetn" && signalName != "rst_n" &&
        !isParameter) {
        module->combSensitiveSignals.insert(signalName);
        LOG_DEBUG("Added combinational sensitive signal: {}", signalName);
    } else if (isParameter) {
        LOG_DEBUG("Skipped parameter in sensitivity list: {}", signalName);
    }
}

void SystemCCodeGenerator::addSeqSensitiveSignal(const std::string& signalName) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Only add non-clock/reset signals to sequential sensitivity
    if (signalName != "clk" && signalName != "clock" && 
        signalName != "reset" && signalName != "resetn" && signalName != "rst_n") {
        module->seqSensitiveSignals.insert(signalName);
        LOG_DEBUG("Added sequential sensitive signal: {}", signalName);
    }
}

void SystemCCodeGenerator::addParameter(const std::string& name, const std::string& value) {
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    Parameter param;
    param.name = name;
    param.value = value;
    
    module->parameters.push_back(param);
    LOG_DEBUG("Added parameter: {} = {}", name, value);
}

void SystemCCodeGenerator::enableTemplateMode(bool enable) {
    templateMode_ = enable;
    LOG_DEBUG("Template mode {}", enable ? "enabled" : "disabled");
}

void SystemCCodeGenerator::setTemplateDirectory(const std::string& templateDir) {
    templateDirectory_ = templateDir;
    LOG_DEBUG("Template directory set to: {}", templateDir);
}

std::string SystemCCodeGenerator::generateModuleUsingTemplate(const std::string& moduleName) {
    auto it = modules_.find(moduleName);
    if (it == modules_.end()) {
        LOG_ERROR("Module not found for template generation: {}", moduleName);
        return "";
    }
    
    const auto& module = *it->second;
    
    // Prepare template variables
    TemplateEngine::VariableMap variables;
    variables["module_name"] = moduleName;
    
    // Generate ports section
    std::stringstream portsStream;
    for (const auto& port : module.ports) {
        portsStream << "    " << generatePortDeclaration(port) << ";\n";
    }
    variables["ports"] = portsStream.str();
    
    // Generate parameters section
    std::stringstream paramsStream;
    for (const auto& param : module.parameters) {
        paramsStream << "    static const int " << param.name << " = " << param.value << ";\n";
    }
    variables["parameters"] = paramsStream.str();
    
    // Generate signals section
    std::stringstream signalsStream;
    for (const auto& signal : module.signals) {
        signalsStream << "    " << generateSignalDeclaration(signal) << ";\n";
    }
    variables["signals"] = signalsStream.str();
    
    // Generate instances section
    std::stringstream instancesStream;
    for (const auto& instance : module.instances) {
        instancesStream << "    " << instance.moduleName << "* " << instance.instanceName << ";\n";
    }
    variables["instances"] = instancesStream.str();
    
    // Generate instance connections section
    std::stringstream connectionsStream;
    for (const auto& instance : module.instances) {
        connectionsStream << "        " << instance.instanceName << " = new " << instance.moduleName << "(\"" << instance.instanceName << "\");\n";
    }
    variables["instance_connections"] = connectionsStream.str();
    
    // Prepare conditionals
    TemplateEngine::ConditionalMap conditionals;
    conditionals["has_ports"] = !module.ports.empty();
    conditionals["has_parameters"] = !module.parameters.empty();
    conditionals["has_signals"] = !module.signals.empty();
    conditionals["has_processes"] = module.hasCombProcess || module.hasSeqProcess;
    conditionals["has_instances"] = !module.instances.empty();
    conditionals["has_instance_connections"] = !module.instances.empty();
    conditionals["has_process_methods"] = module.hasCombProcess || module.hasSeqProcess;
    
    // Load and render template
    std::string templatePath = templateDirectory_ + "/systemc_module.h.template";
    std::string templateContent = templateEngine_.loadTemplate(templatePath);
    
    if (templateContent.empty()) {
        LOG_ERROR("Failed to load template: {}", templatePath);
        return generateModuleHeader(moduleName); // Fallback to original method
    }
    
    return templateEngine_.render(templateContent, variables, conditionals);
}

// Multiple process block support implementation

void SystemCCodeGenerator::beginProcessBlock(const std::string& blockName, bool isSequential) {
    auto* module = getCurrentModuleData();
    if (!module) {
        LOG_ERROR("Cannot begin process block without active module");
        return;
    }
    
    // Save any current process block
    if (currentProcessBlock_) {
        LOG_WARN("Starting new process block {} while {} is still active", 
                   blockName, currentProcessBlock_->name);
        endProcessBlock();
    }
    
    // Create new process block
    module->processBlocks.emplace_back();
    currentProcessBlock_ = &module->processBlocks.back();
    currentProcessBlock_->name = blockName;
    currentProcessBlock_->isSequential = isSequential;
    
    LOG_DEBUG("Started process block: {} (sequential: {})", blockName, isSequential);
}

void SystemCCodeGenerator::endProcessBlock() {
    if (!currentProcessBlock_) {
        LOG_WARN("endProcessBlock called without active process block");
        return;
    }
    
    // Count lines in the process
    std::string code = currentProcessBlock_->code.str();
    currentProcessBlock_->lineCount = std::count(code.begin(), code.end(), '\n');
    
    // Check if we should split this process
    auto* module = getCurrentModuleData();
    if (module && module->enableProcessSplitting && 
        currentProcessBlock_->lineCount > module->maxProcessLines) {
        splitLargeProcess(*currentProcessBlock_);
    }
    
    LOG_DEBUG("Ended process block: {} ({} lines, {} outputs, {} inputs)", 
             currentProcessBlock_->name, 
             currentProcessBlock_->lineCount,
             currentProcessBlock_->outputs.size(),
             currentProcessBlock_->inputs.size());
    
    currentProcessBlock_ = nullptr;
}

void SystemCCodeGenerator::addAssignmentToCurrentBlock(const std::string& lhs, const std::string& rhs) {
    if (!currentProcessBlock_) {
        LOG_WARN("addAssignmentToCurrentBlock called without active process block");
        // Fall back to legacy behavior
        addNonBlockingAssignment(lhs, rhs);
        return;
    }
    
    std::string sanitizedRhs = sanitizeExpression(rhs);
    
    // Track signal usage
    currentProcessBlock_->outputs.insert(lhs);
    // TODO: Parse rhs to extract input signals
    
    // Add the assignment
    currentProcessBlock_->code << fmt::format("        {}.write({});\n", lhs, sanitizedRhs);
    
    LOG_DEBUG("Added assignment to block {}: {} = {}", 
             currentProcessBlock_->name, lhs, sanitizedRhs);
}

void SystemCCodeGenerator::setCurrentBlockSensitivity(const std::set<std::string>& signals) {
    if (!currentProcessBlock_) {
        LOG_WARN("setCurrentBlockSensitivity called without active process block");
        return;
    }
    
    currentProcessBlock_->sensitivity = signals;
    LOG_DEBUG("Set sensitivity for block {}: {} signals", 
             currentProcessBlock_->name, signals.size());
}

void SystemCCodeGenerator::setCurrentBlockClock(const std::string& clockSignal) {
    if (!currentProcessBlock_) {
        LOG_WARN("setCurrentBlockClock called without active process block");
        return;
    }
    
    currentProcessBlock_->clockSignal = clockSignal;
    currentProcessBlock_->sensitivity.insert(clockSignal);
    LOG_DEBUG("Set clock for block {}: {}", currentProcessBlock_->name, clockSignal);
}

void SystemCCodeGenerator::setCurrentBlockReset(const std::string& resetSignal) {
    if (!currentProcessBlock_) {
        LOG_WARN("setCurrentBlockReset called without active process block");
        return;
    }
    
    currentProcessBlock_->resetSignal = resetSignal;
    currentProcessBlock_->sensitivity.insert(resetSignal);
    LOG_DEBUG("Set reset for block {}: {}", currentProcessBlock_->name, resetSignal);
}

void SystemCCodeGenerator::enableProcessSplitting(bool enable) {
    auto* module = getCurrentModuleData();
    if (module) {
        module->enableProcessSplitting = enable;
        LOG_INFO("Process splitting {} for module {}", 
                enable ? "enabled" : "disabled", module->name);
    }
}

void SystemCCodeGenerator::setMaxProcessLines(int maxLines) {
    auto* module = getCurrentModuleData();
    if (module) {
        module->maxProcessLines = maxLines;
        LOG_INFO("Set max process lines to {} for module {}", maxLines, module->name);
    }
}

void SystemCCodeGenerator::splitLargeProcess(ProcessBlock& process) {
    LOG_INFO("Splitting large process {} with {} lines", process.name, process.lineCount);
    
    // Parse the process code to identify independent assignment groups
    std::string code = process.code.str();
    auto splitProcesses = analyzeAndSplitProcess(code, process.outputs);
    
    if (splitProcesses.size() <= 1) {
        LOG_INFO("Process {} cannot be split further", process.name);
        return;
    }
    
    // Replace the original process with split versions
    auto* module = getCurrentModuleData();
    if (!module) return;
    
    // Remove the original large process
    auto it = std::find_if(module->processBlocks.begin(), module->processBlocks.end(),
                          [&process](const ProcessBlock& pb) { 
                              return pb.name == process.name; 
                          });
    if (it != module->processBlocks.end()) {
        module->processBlocks.erase(it);
    }
    
    // Add the split processes
    for (auto& splitProc : splitProcesses) {
        // Inherit properties from parent process
        splitProc.isSequential = process.isSequential;
        splitProc.clockSignal = process.clockSignal;
        splitProc.resetSignal = process.resetSignal;
        splitProc.sourceBlock = process.sourceBlock;
        
        module->processBlocks.push_back(std::move(splitProc));
    }
    
    LOG_INFO("Split process {} into {} smaller processes", 
            process.name, splitProcesses.size());
}

std::vector<ProcessBlock> SystemCCodeGenerator::analyzeAndSplitProcess(
    const std::string& code, 
    const std::set<std::string>& signals) {
    
    std::vector<ProcessBlock> result;
    
    // Simple heuristic: split by contiguous assignment blocks
    // TODO: Implement more sophisticated dependency analysis
    
    std::istringstream stream(code);
    std::string line;
    ProcessBlock currentBlock;
    int blockIndex = 0;
    int lineCount = 0;
    const int targetLinesPerBlock = 20; // Target size for each split block
    
    while (std::getline(stream, line)) {
        // Check if we should start a new block
        if (lineCount >= targetLinesPerBlock && !currentBlock.code.str().empty()) {
            currentBlock.name = fmt::format("split_proc_{}", blockIndex++);
            currentBlock.lineCount = lineCount;
            result.push_back(std::move(currentBlock));
            currentBlock = ProcessBlock();
            lineCount = 0;
        }
        
        currentBlock.code << line << "\n";
        lineCount++;
        
        // Try to extract signal names from assignments
        // Simple pattern: signal.write(...)
        size_t writePos = line.find(".write(");
        if (writePos != std::string::npos) {
            // Extract signal name before .write
            size_t start = line.find_last_of(" \t", writePos);
            if (start == std::string::npos) start = 0;
            else start++;
            
            std::string signalName = line.substr(start, writePos - start);
            currentBlock.outputs.insert(signalName);
        }
    }
    
    // Add the last block
    if (!currentBlock.code.str().empty()) {
        currentBlock.name = fmt::format("split_proc_{}", blockIndex);
        currentBlock.lineCount = lineCount;
        result.push_back(std::move(currentBlock));
    }
    
    // If we couldn't split effectively, return the original as a single block
    if (result.empty()) {
        ProcessBlock single;
        single.name = "proc_0";
        single.code << code;
        single.outputs = signals;
        single.lineCount = std::count(code.begin(), code.end(), '\n');
        result.push_back(std::move(single));
    }
    
    return result;
}

} // namespace sv2sc::codegen