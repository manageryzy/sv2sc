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
    ports_.clear();
    signals_.clear();
    
    LOG_INFO("Starting SystemC code generation for module: {}", moduleName);
    
    // Generate header class declaration
    headerCode_ << fmt::format("#ifndef {}_H\n", moduleName);
    headerCode_ << fmt::format("#define {}_H\n\n", moduleName);
    headerCode_ << "#include <systemc.h>\n\n";
    headerCode_ << fmt::format("SC_MODULE({}) {{\n", moduleName);
    
    // Implementation constructor start
    implCode_ << fmt::format("#include \"{}.h\"\n\n", moduleName);
    implCode_ << fmt::format("{}::{}", moduleName, moduleName);
}

void SystemCCodeGenerator::endModule() {
    // Complete header
    headerCode_ << "\n";
    headerCode_ << getIndent() << "SC_CTOR(" << currentModule_ << ") {\n";
    headerCode_ << generateConstructor();
    headerCode_ << getIndent() << "}\n";
    headerCode_ << generateProcessMethods();
    headerCode_ << "};\n\n";
    headerCode_ << "#endif\n";
    
    // Complete implementation - constructor body is now in header only
    implCode_.str(""); // Clear implementation since we're using header-only approach
    
    LOG_INFO("Completed SystemC code generation for module: {}", currentModule_);
}

void SystemCCodeGenerator::addPort(const Port& port) {
    ports_.push_back(port);
    
    std::string portDecl = generatePortDeclaration(port);
    headerCode_ << getIndent() << portDecl << ";\n";
    
    LOG_DEBUG("Added port declaration: {}", portDecl);
}

void SystemCCodeGenerator::addSignal(const Signal& signal) {
    signals_.push_back(signal);
    
    std::string signalDecl = generateSignalDeclaration(signal);
    headerCode_ << getIndent() << signalDecl << ";\n";
    
    LOG_DEBUG("Added signal declaration: {}", signalDecl);
}

void SystemCCodeGenerator::addBlockingAssignment(const std::string& lhs, const std::string& rhs) {
    // For SystemC, we need to check if lhs is a signal and use write() method
    // For now, assume signals need .write() and ports use direct assignment
    if (lhs != "unknown_expr" && rhs != "unknown_expr") {
        processCode_ << fmt::format("{}        {}.write({});\n", getIndent(), lhs, rhs);
    } else {
        processCode_ << fmt::format("{}        // Skipping assignment: {} = {}\n", getIndent(), lhs, rhs);
    }
    LOG_DEBUG("Added blocking assignment: {} = {}", lhs, rhs);
}

void SystemCCodeGenerator::addNonBlockingAssignment(const std::string& lhs, const std::string& rhs) {
    processCode_ << fmt::format("{}        {}.write({});\n", getIndent(), lhs, rhs);
    LOG_DEBUG("Added non-blocking assignment: {} <= {}", lhs, rhs);
}

void SystemCCodeGenerator::addDelayedAssignment(const std::string& lhs, const std::string& rhs, const std::string& delay) {
    processCode_ << fmt::format("{}        wait({});\n", getIndent(), delay);
    processCode_ << fmt::format("{}        {}.write({});\n", getIndent(), lhs, rhs);
    LOG_DEBUG("Added delayed assignment: {} <= {} after {}", lhs, rhs, delay);
}

void SystemCCodeGenerator::beginGenerateBlock(const std::string& label) {
    addComment(fmt::format("Generate block: {}", label.empty() ? "unnamed" : label));
}

void SystemCCodeGenerator::endGenerateBlock() {
    addComment("End generate block");
}

void SystemCCodeGenerator::addComment(const std::string& comment) {
    headerCode_ << getIndent() << "// " << comment << "\n";
}

void SystemCCodeGenerator::addRawCode(const std::string& code) {
    headerCode_ << code;
}

std::string SystemCCodeGenerator::generateHeader() const {
    return headerCode_.str();
}

std::string SystemCCodeGenerator::generateImplementation() const {
    std::string impl = implCode_.str();
    if (impl.empty()) {
        // Generate a minimal implementation comment
        return fmt::format("// {}.cpp - Header-only SystemC implementation\n// All implementation is contained in {}.h\n", currentModule_, currentModule_);
    }
    return impl;
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

std::string SystemCCodeGenerator::getIndent() const {
    return std::string(indentLevel_ * 4, ' ');
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

std::string SystemCCodeGenerator::generatePortDeclaration(const Port& port) const {
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
    
    std::string dataType = mapDataType(port.dataType, port.width);
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
    std::string dataType = mapDataType(signal.dataType, signal.width);
    std::string decl = fmt::format("sc_signal<{}> {}", dataType, signal.name);
    
    // Handle arrays
    if (signal.isArray) {
        for (int dim : signal.arrayDimensions) {
            decl = fmt::format("sc_vector<sc_signal<{}>> {} /* [{}] */", dataType, signal.name, dim);
            break; // For now, handle only first dimension
        }
    }
    
    return decl;
}

std::string SystemCCodeGenerator::generateConstructor() const {
    std::stringstream constructor;
    
    // Generate sensitivity list for processes
    constructor << "    // Process sensitivity\n";
    constructor << "    SC_METHOD(comb_proc);\n";
    
    // Add clock and reset sensitivity
    bool hasClk = std::any_of(ports_.begin(), ports_.end(), 
        [](const Port& p) { return p.name == "clk" || p.name == "clock"; });
    bool hasReset = std::any_of(ports_.begin(), ports_.end(),
        [](const Port& p) { return p.name == "reset" || p.name == "rst"; });
    
    if (hasClk) {
        constructor << "    sensitive << clk.pos();\n";
    }
    
    if (hasReset) {
        constructor << "    sensitive << reset;\n";
    }
    
    constructor << "\n";
    return constructor.str();
}

std::string SystemCCodeGenerator::generateProcessMethods() const {
    std::stringstream methods;
    
    methods << "\nprivate:\n";
    methods << "    void comb_proc() {\n";
    methods << processCode_.str();
    methods << "    }\n";
    
    return methods.str();
}

} // namespace sv2sc::codegen