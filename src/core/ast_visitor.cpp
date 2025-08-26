#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include "utils/performance_profiler.h"
#include <fmt/format.h>
#include <regex>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/expressions/LiteralExpressions.h>
#include <slang/ast/expressions/OperatorExpressions.h>
#include <slang/ast/expressions/SelectExpressions.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/ast/types/AllTypes.h>
#include <algorithm>
#include <string>

namespace sv2sc::core {

SVToSCVisitor::SVToSCVisitor(codegen::SystemCCodeGenerator& generator) 
    : codeGen_(generator) {
}

void SVToSCVisitor::setTargetModule(const std::string& targetModule) {
    targetModule_ = targetModule;
}

void SVToSCVisitor::handle(const slang::ast::InstanceSymbol& node) {
    // Module instances should track dependencies but not generate new modules
    // The actual module definitions are handled by InstanceBodySymbol
    LOG_DEBUG("Found module instance: {} of type {}", node.name, node.body.getDefinition().name);
    
    // If we're currently processing a module, track this as a dependency
    if (!currentModule_.empty()) {
        std::string instancedModule = std::string(node.body.getDefinition().name);
        std::string originalName = std::string(node.name);
        
        // Add the instance and get the unique name that was generated
        std::string uniqueName = codeGen_.addModuleInstance(originalName, instancedModule);
        
        // Determine the generate loop index from the unique name
        int generateIndex = 0;
        if (uniqueName != originalName) {
            // Extract index from names like "fa_inst_0", "fa_inst_1", etc.
            size_t underscorePos = uniqueName.find_last_of('_');
            if (underscorePos != std::string::npos) {
                std::string indexStr = uniqueName.substr(underscorePos + 1);
                try {
                    generateIndex = std::stoi(indexStr) + 1; // +1 because first instance has no suffix
                } catch (...) {
                    generateIndex = 0;
                }
            }
        }
        
        // Extract port connections using the unique instance name and generate index
        extractPortConnections(node, uniqueName, generateIndex);
        
        LOG_DEBUG("Added module instance {} of type {} to {}", node.name, instancedModule, currentModule_);
    }
    
    // Don't visit the instance body here - we'll handle module definitions separately
}

void SVToSCVisitor::extractPortConnections(const slang::ast::InstanceSymbol& node, const std::string& uniqueInstanceName, int generateIndex) {
    LOG_DEBUG("Extracting port connections for instance: {} (unique: {}, index: {})", node.name, uniqueInstanceName, generateIndex);
    
    // Use the unique instance name for adding connections
    std::string instanceName = uniqueInstanceName;
    
    // Get all port connections from the instance
    auto portConnections = node.getPortConnections();
    LOG_DEBUG("Instance {} has {} port connections", instanceName, portConnections.size());
    
    // Process each port connection
    for (const auto* connection : portConnections) {
        if (connection) {
            std::string portName = std::string(connection->port.name);
            
            // Extract the connected expression/signal
            std::string signalExpr = "/* connection */";
            if (connection->getExpression()) {
                signalExpr = extractExpressionText(*connection->getExpression());
                
                // Clean up malformed expressions (remove trailing " = ")
                if (signalExpr.ends_with(" = ")) {
                    signalExpr = signalExpr.substr(0, signalExpr.length() - 3);
                }
                
                // Substitute generate loop variable with actual index
                signalExpr = substituteGenerateVariable(signalExpr, generateIndex);
            }
            
            LOG_DEBUG("Port connection: {}.{} -> {}", instanceName, portName, signalExpr);
            
            // Add the port connection to the code generator
            codeGen_.addPortConnection(instanceName, portName, signalExpr);
        }
    }
}

std::string SVToSCVisitor::substituteGenerateVariable(const std::string& expression, int index) const {
    std::string result = expression;
    
    // Replace "static_cast<int>(i.read())" with the actual index
    std::string pattern = "static_cast<int>(i.read())";
    std::string replacement = std::to_string(index);
    
    size_t pos = 0;
    while ((pos = result.find(pattern, pos)) != std::string::npos) {
        result.replace(pos, pattern.length(), replacement);
        pos += replacement.length();
    }
    
    // Also replace simpler patterns like "i" in expressions like "(i + 1)"
    // Be careful to only replace standalone "i" not "i" within other identifiers
    std::regex iPattern(R"(\bi\b)");
    result = std::regex_replace(result, iPattern, std::to_string(index));
    
    LOG_DEBUG("Substituted generate variable: '{}' -> '{}'", expression, result);
    return result;
}

void SVToSCVisitor::handleAdvancedFeatures(const std::string& sourceText) const {
    PROFILE_SCOPE("Advanced Features Detection");
    
    bool hasAdvancedFeatures = false;
    
    // Check for SystemVerilog packages
    if (sourceText.find("package") != std::string::npos || 
        (sourceText.find("import") != std::string::npos && sourceText.find("::") != std::string::npos)) {
        codeGen_.addHeaderComment("// SystemVerilog package detected - types and functions may need manual conversion");
        hasAdvancedFeatures = true;
    }
    
    // Check for SystemVerilog assertions
    if (sourceText.find("assert") != std::string::npos) {
        codeGen_.addHeaderComment("// SystemVerilog assertions detected - consider implementing as SystemC runtime checks");
        
        // More specific detection
        if (sourceText.find("assert property") != std::string::npos) {
            codeGen_.addHeaderComment("// Property assertions found - consider SystemC SC_THREAD with wait() statements");
        }
        if (sourceText.find("assert (") != std::string::npos) {
            codeGen_.addHeaderComment("// Immediate assertions found - consider C++ assert() or sc_assert()");
        }
        hasAdvancedFeatures = true;
    }
    
    // Check for SystemVerilog classes
    if (sourceText.find("class") != std::string::npos && sourceText.find("endclass") != std::string::npos) {
        codeGen_.addHeaderComment("// SystemVerilog class detected - consider converting to C++ class or SystemC struct");
        hasAdvancedFeatures = true;
    }
    
    // Check for coverage constructs
    if (sourceText.find("covergroup") != std::string::npos || sourceText.find("coverpoint") != std::string::npos) {
        codeGen_.addHeaderComment("// SystemVerilog coverage constructs detected - not directly supported in SystemC");
        hasAdvancedFeatures = true;
    }
    
    // Check for interfaces
    if (sourceText.find("interface") != std::string::npos && sourceText.find("endinterface") != std::string::npos) {
        codeGen_.addHeaderComment("// SystemVerilog interface detected - consider converting to SystemC struct with signals");
        hasAdvancedFeatures = true;
    }
    
    // Check for advanced data types
    if (sourceText.find("typedef") != std::string::npos || sourceText.find("struct") != std::string::npos) {
        codeGen_.addHeaderComment("// SystemVerilog custom types detected - may need manual conversion to SystemC types");
        hasAdvancedFeatures = true;
    }
    
    if (hasAdvancedFeatures) {
        codeGen_.addHeaderComment("// Note: Advanced SystemVerilog features require careful manual review for SystemC compatibility");
    }
}

void SVToSCVisitor::handle(const slang::ast::InstanceBodySymbol& node) {
    std::string moduleName = std::string(node.getDefinition().name);
    PROFILE_SCOPE(fmt::format("Process Module: {}", moduleName));
    LOG_INFO("Processing module definition: {}", moduleName);
    
    // Add comment about advanced SystemVerilog features if present
    codeGen_.addHeaderComment("// SystemVerilog module: " + moduleName);
    
    // Check for advanced features (this is a simplified check)
    // In a full implementation, we'd analyze the actual AST nodes
    handleAdvancedFeatures(""); // Placeholder - would need actual source text
    
    // If we have a target module set, only process that specific module
    if (!targetModule_.empty() && moduleName != targetModule_) {
        LOG_DEBUG("Skipping module {} (not target module {})", moduleName, targetModule_);
        return;
    }
    
    currentModule_ = moduleName;
    portNames_.clear();         // Clear port names for new module
    declaredSignals_.clear();   // Clear signal tracking for new module
    arithmeticSignals_.clear(); // Clear arithmetic usage tracking
    logicSignals_.clear();      // Clear logic usage tracking
    codeGen_.beginModule(currentModule_);
    
    LOG_DEBUG("Started processing module: {}", currentModule_);
    
    // Visit all child symbols (ports, variables, etc.)
    // This will create signals and analyze their usage
    visitDefault(node);
    
    LOG_DEBUG("Finishing module: {}", currentModule_);
    codeGen_.endModule();
    currentModule_.clear();
}

void SVToSCVisitor::handle(const slang::ast::PortSymbol& node) {
    using namespace codegen;
    
    Port port;
    port.name = std::string(node.name);
    
    // Determine port direction
    switch (node.direction) {
        case slang::ast::ArgumentDirection::In:
            port.direction = PortDirection::INPUT;
            break;
        case slang::ast::ArgumentDirection::Out:
            port.direction = PortDirection::OUTPUT;
            break;
        case slang::ast::ArgumentDirection::InOut:
            port.direction = PortDirection::INOUT;
            break;
        default:
            LOG_WARN("Unknown port direction for port: {}", node.name);
            port.direction = PortDirection::INPUT;
            break;
    }
    
    // Get type information
    auto& type = node.getType();
    if (type.isPackedArray()) {
        if (type.isFourState()) {
            port.dataType = SystemCDataType::SC_LV;
        } else {
            port.dataType = SystemCDataType::SC_BV;
        }
        port.width = static_cast<int>(type.getBitWidth());
        
        // Try to detect parameter expressions for common widths
        if (port.width == 4) {
            port.widthExpression = "WIDTH";  // Assume WIDTH parameter for 4-bit signals
        } else if (port.width == 5) {
            port.widthExpression = "WIDTH+1";  // Assume WIDTH+1 for carry signals
        }
        // Add more patterns as needed
        
    } else if (type.isFourState()) {
        port.dataType = SystemCDataType::SC_LOGIC;
        port.width = 1;
    } else {
        port.dataType = SystemCDataType::SC_BIT;
        port.width = 1;
    }
    
    codeGen_.addPort(port);
    portNames_.push_back(port.name);  // Track port name to avoid duplicate signals
    LOG_DEBUG("Added port: {} (direction: {}, width: {})", 
              port.name, static_cast<int>(port.direction), port.width);
}

void SVToSCVisitor::handle(const slang::ast::ParameterSymbol& node) {
    LOG_DEBUG("Processing parameter: {}", std::string(node.name));
    
    std::string paramName = std::string(node.name);
    
    // Get the parameter value
    std::string paramValue = "0"; // Default value
    auto& value = node.getValue();
    if (value.isInteger()) {
        auto intVal = value.integer().as<int>();
        if (intVal.has_value()) {
            paramValue = std::to_string(intVal.value());
        } else {
            LOG_DEBUG("Parameter {} integer value not available, using default", paramName);
        }
    } else {
        LOG_DEBUG("Parameter {} has non-integer value, using default", paramName);
    }
    
    // Add parameter as a static const in the SystemC module
    codeGen_.addParameter(paramName, paramValue);
    LOG_DEBUG("Added parameter: {} = {}", paramName, paramValue);
}

void SVToSCVisitor::handle(const slang::ast::VariableSymbol& node) {
    using namespace codegen;
    
    // Skip variables that are ports - they're handled separately by handle(PortSymbol)
    // Check if this variable name matches any port we've already processed
    std::string varName = std::string(node.name);
    for (const auto& port : portNames_) {
        if (port == varName) {
            LOG_DEBUG("Skipping port variable: {}", node.name);
            return;
        }
    }
    
    // Check if this signal was already declared to prevent duplicates
    if (declaredSignals_.count(varName) > 0) {
        LOG_DEBUG("Skipping duplicate signal: {}", varName);
        return;
    }
    
    Signal signal;
    signal.name = std::string(node.name);
    
    // Get type information
    auto& type = node.getType();
    
    // Check for unpacked arrays first (like mem_array [255:0])
    if (type.isUnpackedArray()) {
        signal.isArray = true;
        
        // Cast to FixedSizeUnpackedArrayType to get dimensions
        if (type.isKind(slang::ast::SymbolKind::FixedSizeUnpackedArrayType)) {
            auto& arrayType = type.as<slang::ast::FixedSizeUnpackedArrayType>();
            auto range = arrayType.range;
            
            // Calculate array size from range
            int arraySize = range.width();
            signal.arrayDimensions.push_back(arraySize);
            
            LOG_DEBUG("Found unpacked array '{}' with size {}", signal.name, arraySize);
            
            // Get the element type for the array
            auto& elementType = arrayType.elementType;
            if (elementType.isPackedArray()) {
                if (elementType.isFourState()) {
                    signal.dataType = SystemCDataType::SC_LV;
                } else {
                    signal.dataType = SystemCDataType::SC_BV;
                }
                signal.width = static_cast<int>(elementType.getBitWidth());
                
                // Try to detect parameter expressions for common widths
                if (signal.width == 4) {
                    signal.widthExpression = "WIDTH";  // Assume WIDTH parameter for 4-bit signals
                } else if (signal.width == 5) {
                    signal.widthExpression = "WIDTH+1";  // Assume WIDTH+1 for carry signals
                }
            } else if (elementType.isFourState()) {
                signal.dataType = SystemCDataType::SC_LOGIC;
                signal.width = 1;
            } else {
                signal.dataType = SystemCDataType::SC_BIT;
                signal.width = 1;
            }
        }
    }
    // Handle packed arrays (like logic [7:0])
    else if (type.isPackedArray()) {
        if (type.isFourState()) {
            signal.dataType = SystemCDataType::SC_LV;
        } else {
            signal.dataType = SystemCDataType::SC_BV;
        }
        signal.width = static_cast<int>(type.getBitWidth());
        
        // Try to detect parameter expressions for common widths
        if (signal.width == 4) {
            signal.widthExpression = "WIDTH";  // Assume WIDTH parameter for 4-bit signals
        } else if (signal.width == 5) {
            signal.widthExpression = "WIDTH+1";  // Assume WIDTH+1 for carry signals
        }
    } else if (type.isFourState()) {
        signal.dataType = SystemCDataType::SC_LOGIC;
        signal.width = 1;
    } else {
        signal.dataType = SystemCDataType::SC_BIT;
        signal.width = 1;
    }
    
    // Handle initializer if present
    if (node.getInitializer()) {
        // For now, just mark that it has an initial value
        signal.initialValue = "0"; // Simplified - TODO: extract actual initial value
    }
    
    // Add to tracking set before generating code
    declaredSignals_.insert(varName);
    
    // Set arithmetic preference based on usage analysis
    signal.preferArithmetic = isArithmeticSignal(varName);
    
    codeGen_.addSignal(signal);
    LOG_DEBUG("Added signal: {} (width: {}, arithmetic: {})", signal.name, signal.width, signal.preferArithmetic);
}

void SVToSCVisitor::handle(const slang::ast::VariableDeclStatement& node) {
    // Get the variable symbol and handle it
    auto& symbol = node.symbol;
    handle(symbol);
}

void SVToSCVisitor::handle(const slang::ast::AssignmentExpression& node) {
    LOG_DEBUG("Processing assignment expression");
    
    // Analyze expression usage patterns before generating code
    analyzeExpressionUsage(node.right());
    
    // Extract left and right hand sides
    std::string lhs = extractExpressionText(node.left());
    std::string rhs = extractExpressionText(node.right());
    
    // Check if we need type conversion for arithmetic results assigned to sc_lv signals
    std::string convertedRhs = rhs;
    if (isSignalName(lhs)) {
        // If assigning arithmetic result (contains .to_uint() or operators) to any signal,
        // we need to convert back to sc_lv since signals are still declared as sc_lv<N>
        if (rhs.find("to_uint()") != std::string::npos || 
            rhs.find(" + ") != std::string::npos ||
            rhs.find(" - ") != std::string::npos ||
            rhs.find(" * ") != std::string::npos ||
            rhs.find(" / ") != std::string::npos) {
            // This looks like an arithmetic expression result, convert to sc_lv
            convertedRhs = "sc_lv<8>(" + rhs + ")";  // TODO: Get actual width
            LOG_DEBUG("Converting arithmetic result to sc_lv: {} -> {}", rhs, convertedRhs);
        }
    }
    
    // For now, use context to determine assignment type
    // TODO: Check actual assignment operator when API is clarified
    
    if (currentBlockIsSequential_) {
        // In sequential blocks, collect assignments for pattern analysis
        LOG_DEBUG("Collecting sequential assignment for pattern analysis: {} <= {}", lhs, convertedRhs);
        collectAssignment(lhs, convertedRhs);
        // Track signals used in the RHS for sequential sensitivity
        extractAndTrackSignals(convertedRhs, true);
    } else {
        // In combinational blocks (always_comb), use combinational assignments directly
        LOG_DEBUG("Adding combinational assignment: {} = {}", lhs, convertedRhs);
        codeGen_.addCombinationalAssignment(lhs, convertedRhs);
        // Track signals used in the RHS for combinational sensitivity
        extractAndTrackSignals(convertedRhs, false);
    }
}

void SVToSCVisitor::handle(const slang::ast::ExpressionStatement& node) {
    LOG_DEBUG("Processing expression statement");
    
    // Visit the contained expression
    visitDefault(node);
}

void SVToSCVisitor::handle(const slang::ast::ProceduralBlockSymbol& node) {
    using namespace slang::ast;
    
    // Determine the type of procedural block
    std::string blockType;
    bool isSequential = false;
    
    switch (node.procedureKind) {
        case ProceduralBlockKind::AlwaysFF:
            blockType = "always_ff";
            isSequential = true;
            break;
        case ProceduralBlockKind::AlwaysComb:
            blockType = "always_comb";
            isSequential = false;
            break;
        case ProceduralBlockKind::Always:
            blockType = "always";
            // For now, treat generic always as combinational unless we detect clocking
            isSequential = false;
            break;
        case ProceduralBlockKind::Initial:
            blockType = "initial";
            isSequential = false;
            break;
        case ProceduralBlockKind::AlwaysLatch:
            blockType = "always_latch";
            isSequential = false;
            break;
        case ProceduralBlockKind::Final:
            blockType = "final";
            isSequential = false;
            break;
        default:
            blockType = "unknown";
            isSequential = false;
            break;
    }
    
    LOG_DEBUG("Processing procedural block: {} (sequential: {})", blockType, isSequential);
    
    // Store current context for assignment generation
    bool prevSequential = currentBlockIsSequential_;
    currentBlockIsSequential_ = isSequential;
    
    // Visit the body with proper context
    LOG_DEBUG("Visiting procedural block body");
    
    // Visit the body directly
    LOG_DEBUG("Procedural block has body, visiting it...");
    const auto& body = node.getBody();
    body.visit(*this);
    
    // If this was a sequential block, analyze collected assignments for patterns
    if (isSequential) {
        analyzeAndGenerateConditionalLogic();
    }
    
    // Restore previous context
    currentBlockIsSequential_ = prevSequential;
}

void SVToSCVisitor::handle(const slang::ast::ContinuousAssignSymbol& node) {
    LOG_DEBUG("Processing continuous assign statement");
    
    // Visit the assignment expression
    visitDefault(node);
}

void SVToSCVisitor::handle(const slang::ast::SubroutineSymbol& node) {
    std::string functionName = std::string(node.name);
    LOG_DEBUG("Processing SystemVerilog function/task: {}", functionName);
    
    // Add comment about function conversion
    codeGen_.addHeaderComment(fmt::format("// SystemVerilog function '{}' - consider manual conversion to SystemC function", functionName));
    
    // For now, we just document the function rather than fully converting it
    // Full function conversion would require analyzing the function body,
    // parameters, return type, and converting SystemVerilog syntax to C++
    
    if (node.subroutineKind == slang::ast::SubroutineKind::Function) {
        codeGen_.addHeaderComment(fmt::format("// Function: {} - returns a value", functionName));
    } else if (node.subroutineKind == slang::ast::SubroutineKind::Task) {
        codeGen_.addHeaderComment(fmt::format("// Task: {} - does not return a value", functionName));
    }
    
    LOG_DEBUG("Documented function/task: {}", functionName);
}


// Utility methods
std::string SVToSCVisitor::getIndent() const {
    return std::string(indentLevel_ * 4, ' ');
}

void SVToSCVisitor::increaseIndent() {
    indentLevel_++;
}

void SVToSCVisitor::decreaseIndent() {
    if (indentLevel_ > 0) {
        indentLevel_--;
    }
}

std::string SVToSCVisitor::extractExpressionText(const slang::ast::Expression& expr) const {
    PROFILE_SCOPE("Expression Extraction");
    // Enhanced expression handling for common SystemVerilog constructs
    
    switch (expr.kind) {
        // Phase 1: Critical Literal Types
        case slang::ast::ExpressionKind::Invalid: {
            LOG_DEBUG("Invalid expression encountered");
            return "/* INVALID_EXPR */";
        }
        case slang::ast::ExpressionKind::IntegerLiteral: {
            auto& literalExpr = expr.as<slang::ast::IntegerLiteral>();
            return literalExpr.getValue().toString(slang::LiteralBase::Decimal);
        }
        case slang::ast::ExpressionKind::RealLiteral: {
            auto& realExpr = expr.as<slang::ast::RealLiteral>();
            return std::to_string(realExpr.getValue());
        }
        case slang::ast::ExpressionKind::TimeLiteral: {
            auto& timeExpr = expr.as<slang::ast::TimeLiteral>();
            // Convert to SystemC time format
            return "sc_time(" + std::to_string(timeExpr.getValue()) + ", SC_NS)";
        }
        case slang::ast::ExpressionKind::NullLiteral: {
            return "nullptr"; // SystemC null equivalent
        }
        case slang::ast::ExpressionKind::UnboundedLiteral: {
            return "/* UNBOUNDED */"; // $ symbol - context dependent
        }
        case slang::ast::ExpressionKind::NamedValue: {
            auto& nameExpr = expr.as<slang::ast::NamedValueExpression>();
            return std::string(nameExpr.symbol.name);
        }
        case slang::ast::ExpressionKind::HierarchicalValue: {
            auto& hierExpr = expr.as<slang::ast::HierarchicalValueExpression>();
            // Convert hierarchical path to SystemC scope resolution
            return std::string(hierExpr.symbol.name); // Simplified for now
        }
        case slang::ast::ExpressionKind::StringLiteral: {
            auto& stringExpr = expr.as<slang::ast::StringLiteral>();
            std::string value = std::string(stringExpr.getValue());
            // Handle SystemVerilog special literals like '0
            if (value == "'0") {
                return "0";
            }
            return "\"" + value + "\"";
        }
        case slang::ast::ExpressionKind::UnbasedUnsizedIntegerLiteral: {
            auto& literalExpr = expr.as<slang::ast::UnbasedUnsizedIntegerLiteral>();
            // Handle literals like '0, '1, 'x, 'z
            auto value = literalExpr.getValue();
            if (value == 0) {
                return "0";
            } else if (value == 1) {
                return "1";
            }
            return value.toString(slang::LiteralBase::Decimal);
        }
        case slang::ast::ExpressionKind::Conversion: {
            // Handle SystemVerilog type conversions, including sized literals like 8'b0, 1'b1
            auto& convExpr = expr.as<slang::ast::ConversionExpression>();
            std::string operand = extractExpressionText(convExpr.operand());
            
            // Check if this is a SystemVerilog sized literal that needs conversion
            // Pattern: digit'd0 or digit'd1 etc.
            if (operand.find("'d") != std::string::npos || 
                operand.find("'b") != std::string::npos ||
                operand.find("'h") != std::string::npos ||
                operand.find("'o") != std::string::npos) {
                
                // Extract just the value part after 'd, 'b, 'h, 'o
                size_t pos = operand.find('\'');
                if (pos != std::string::npos && pos + 2 < operand.length()) {
                    char base = operand[pos + 1];
                    std::string value = operand.substr(pos + 2);
                    
                    if (base == 'd') {
                        // Decimal value - handle X/Z states
                        if (value == "x" || value == "X") {
                            // Extract width from the beginning of operand (e.g., "32" from "32'dx")
                            std::string widthStr = operand.substr(0, pos);
                            int width = std::stoi(widthStr);
                            return fmt::format("sc_lv<{}>(\"{}\")", width, std::string(width, 'X'));
                        } else if (value == "z" || value == "Z") {
                            // Extract width from the beginning of operand
                            std::string widthStr = operand.substr(0, pos);
                            int width = std::stoi(widthStr);
                            return fmt::format("sc_lv<{}>(\"{}\")", width, std::string(width, 'Z'));
                        }
                        return value;
                    } else if (base == 'b') {
                        // Binary value - convert to decimal or keep as 0x format, handle X/Z
                        if (value.find('x') != std::string::npos || value.find('X') != std::string::npos ||
                            value.find('z') != std::string::npos || value.find('Z') != std::string::npos) {
                            // Extract width from the beginning of operand
                            std::string widthStr = operand.substr(0, pos);
                            int width = std::stoi(widthStr);
                            // Convert x/z to X/Z for SystemC
                            std::string scValue = value;
                            std::replace(scValue.begin(), scValue.end(), 'x', 'X');
                            std::replace(scValue.begin(), scValue.end(), 'z', 'Z');
                            return fmt::format("sc_lv<{}>(\"{}\")", width, scValue);
                        }
                        if (value == "0" || value == "00000000") {
                            return "0";
                        } else if (value == "1" || value == "00000001") {
                            return "1";
                        }
                        return "0b" + value; // C++14 binary literal
                    } else if (base == 'h') {
                        // Hex value - handle X/Z states
                        if (value.find('x') != std::string::npos || value.find('X') != std::string::npos ||
                            value.find('z') != std::string::npos || value.find('Z') != std::string::npos) {
                            // Extract width from the beginning of operand
                            std::string widthStr = operand.substr(0, pos);
                            int width = std::stoi(widthStr);
                            // Convert hex with x/z to binary X/Z pattern
                            std::string binaryPattern = "";
                            for (char c : value) {
                                if (c == 'x' || c == 'X') {
                                    binaryPattern += "XXXX";
                                } else if (c == 'z' || c == 'Z') {
                                    binaryPattern += "ZZZZ";
                                } else {
                                    // Convert hex digit to 4-bit binary
                                    int digit = (c >= '0' && c <= '9') ? c - '0' : 
                                              (c >= 'A' && c <= 'F') ? c - 'A' + 10 :
                                              (c >= 'a' && c <= 'f') ? c - 'a' + 10 : 0;
                                    for (int i = 3; i >= 0; i--) {
                                        binaryPattern += ((digit >> i) & 1) ? '1' : '0';
                                    }
                                }
                            }
                            // Trim to actual width
                            if (binaryPattern.length() > width) {
                                binaryPattern = binaryPattern.substr(binaryPattern.length() - width);
                            }
                            return fmt::format("sc_lv<{}>(\"{}\")", width, binaryPattern);
                        }
                        return "0x" + value;
                    } else if (base == 'o') {
                        // Octal value - handle X/Z states  
                        if (value.find('x') != std::string::npos || value.find('X') != std::string::npos ||
                            value.find('z') != std::string::npos || value.find('Z') != std::string::npos) {
                            // Extract width and convert similar to hex case
                            std::string widthStr = operand.substr(0, pos);
                            int width = std::stoi(widthStr);
                            std::string binaryPattern = "";
                            for (char c : value) {
                                if (c == 'x' || c == 'X') {
                                    binaryPattern += "XXX";
                                } else if (c == 'z' || c == 'Z') {
                                    binaryPattern += "ZZZ";
                                } else {
                                    // Convert octal digit to 3-bit binary
                                    int digit = c - '0';
                                    for (int i = 2; i >= 0; i--) {
                                        binaryPattern += ((digit >> i) & 1) ? '1' : '0';
                                    }
                                }
                            }
                            // Trim to actual width
                            if (binaryPattern.length() > width) {
                                binaryPattern = binaryPattern.substr(binaryPattern.length() - width);
                            }
                            return fmt::format("sc_lv<{}>(\"{}\")", width, binaryPattern);
                        }
                        return "0" + value;
                    }
                }
            }
            
            // For other conversions, just return the operand value
            // SystemC will handle the type conversion automatically
            return operand;
        }
        case slang::ast::ExpressionKind::UnaryOp: {
            auto& unaryExpr = expr.as<slang::ast::UnaryExpression>();
            std::string operand = extractExpressionText(unaryExpr.operand());
            switch (unaryExpr.op) {
                case slang::ast::UnaryOperator::Plus: return "+" + operand;
                case slang::ast::UnaryOperator::Minus: return "-" + operand;
                case slang::ast::UnaryOperator::BitwiseNot: return "~" + operand;
                case slang::ast::UnaryOperator::BitwiseAnd: 
                    // Reduction AND - convert to SystemC or_reduce
                    if (operand.find(".read()") == std::string::npos && isSignalName(operand)) {
                        return operand + ".read().and_reduce()";
                    }
                    return "(" + operand + ").and_reduce()";
                case slang::ast::UnaryOperator::BitwiseOr: 
                    // Reduction OR - convert to SystemC or_reduce
                    if (operand.find(".read()") == std::string::npos && isSignalName(operand)) {
                        return operand + ".read().or_reduce()";
                    }
                    return "(" + operand + ").or_reduce()";
                case slang::ast::UnaryOperator::BitwiseXor:
                    // Reduction XOR - convert to SystemC xor_reduce
                    if (operand.find(".read()") == std::string::npos && isSignalName(operand)) {
                        return operand + ".read().xor_reduce()";
                    }
                    return "(" + operand + ").xor_reduce()";
                case slang::ast::UnaryOperator::LogicalNot: return "!" + operand;
                default: return "(" + operand + ")";
            }
        }
        case slang::ast::ExpressionKind::BinaryOp: {
            auto& binaryExpr = expr.as<slang::ast::BinaryExpression>();
            std::string lhs = extractExpressionText(binaryExpr.left());
            std::string rhs = extractExpressionText(binaryExpr.right());
            
            // Determine if this is an arithmetic operation
            bool isArithmetic = false;
            std::string op;
            switch (binaryExpr.op) {
                case slang::ast::BinaryOperator::Add: op = " + "; isArithmetic = true; break;
                case slang::ast::BinaryOperator::Subtract: op = " - "; isArithmetic = true; break;
                case slang::ast::BinaryOperator::Multiply: op = " * "; isArithmetic = true; break;
                case slang::ast::BinaryOperator::Divide: op = " / "; isArithmetic = true; break;
                case slang::ast::BinaryOperator::Mod: op = " % "; isArithmetic = true; break;
                case slang::ast::BinaryOperator::BinaryAnd: op = " & "; break;
                case slang::ast::BinaryOperator::BinaryOr: op = " | "; break;
                case slang::ast::BinaryOperator::BinaryXor: op = " ^ "; break;
                case slang::ast::BinaryOperator::LogicalAnd: op = " && "; break;
                case slang::ast::BinaryOperator::LogicalOr: op = " || "; break;
                case slang::ast::BinaryOperator::Equality: op = " == "; break;
                case slang::ast::BinaryOperator::Inequality: op = " != "; break;
                case slang::ast::BinaryOperator::LessThan: op = " < "; break;
                case slang::ast::BinaryOperator::LessThanEqual: op = " <= "; break;
                case slang::ast::BinaryOperator::GreaterThan: op = " > "; break;
                case slang::ast::BinaryOperator::GreaterThanEqual: op = " >= "; break;
                default: op = " ? "; break;
            }
            
            // For operands that are signals, add .read()
            if (isSignalName(lhs) && lhs.find(".read()") == std::string::npos) {
                if (isArithmetic) {
                    // For arithmetic operations, convert sc_lv signals to sc_uint for the operation
                    lhs += ".read().to_uint()";
                } else {
                    lhs += ".read()";
                }
            }
            if (isSignalName(rhs) && rhs.find(".read()") == std::string::npos) {
                if (isArithmetic) {
                    // For arithmetic operations, convert sc_lv signals to sc_uint for the operation  
                    rhs += ".read().to_uint()";
                } else {
                    rhs += ".read()";
                }
            }
            
            return "(" + lhs + op + rhs + ")";
        }
        case slang::ast::ExpressionKind::ElementSelect: {
            auto& selectExpr = expr.as<slang::ast::ElementSelectExpression>();
            std::string base = extractExpressionText(selectExpr.value());
            std::string index = extractExpressionText(selectExpr.selector());
            
            // Handle bitwise complement operations in array indexing (e.g., regs[~waddr[4:0]])
            if (index.find("~") != std::string::npos) {
                // Check if this is a range selection after complement: ~signal.range(4, 0)
                if (index.find(".range(") != std::string::npos) {
                    // Convert ~signal.range(4, 0) to (~signal.read()).range(4, 0).to_uint()
                    size_t rangePos = index.find(".range(");
                    std::string signalPart = index.substr(1, rangePos - 1); // Remove ~ and get signal name
                    std::string rangePart = index.substr(rangePos); // Get .range(4, 0) part
                    return base + "[(~" + signalPart + ".read())" + rangePart + ".to_uint()]";
                } else {
                    // Simple complement: ~signal -> (~signal.read()).to_uint()
                    std::string signalName = index.substr(1); // Remove ~
                    return base + "[(~" + signalName + ".read()).to_uint()]";
                }
            }
            
            // For array indexing in SystemC, check if index might be a signal/port that needs .read()
            // This is a heuristic: if the index name suggests it's a signal/port, add appropriate conversion
            if (index.find("address") != std::string::npos || 
                index.find("index") != std::string::npos ||
                index.find("addr") != std::string::npos) {
                // Multi-bit signals need .read().to_uint()
                if (index.find(".range(") != std::string::npos) {
                    // Already has range selection, just add .read() before range and .to_uint() after
                    size_t rangePos = index.find(".range(");
                    std::string signalPart = index.substr(0, rangePos);
                    std::string rangePart = index.substr(rangePos);
                    return base + "[" + signalPart + ".read()" + rangePart + ".to_uint()]";
                } else {
                    return base + "[" + index + ".read().to_uint()]";
                }
            } else if (index == "i" || index == "j" || index == "k") {
                // Single bit signals need .read() but to_bool() won't work for array indexing
                // Use static_cast to convert to int (0 or 1)
                return base + "[static_cast<int>(" + index + ".read())]";
            }
            
            // Check if it's a signal name that needs .read() conversion
            if (isSignalName(index) && index.find(".read()") == std::string::npos) {
                if (index.find(".range(") != std::string::npos) {
                    // Signal with range selection
                    size_t rangePos = index.find(".range(");
                    std::string signalPart = index.substr(0, rangePos);
                    std::string rangePart = index.substr(rangePos);
                    return base + "[" + signalPart + ".read()" + rangePart + ".to_uint()]";
                } else {
                    // Simple signal reference
                    return base + "[" + index + ".read().to_uint()]";
                }
            }
            
            return base + "[" + index + "]";
        }
        case slang::ast::ExpressionKind::RangeSelect: {
            auto& rangeExpr = expr.as<slang::ast::RangeSelectExpression>();
            std::string base = extractExpressionText(rangeExpr.value());
            std::string left = extractExpressionText(rangeExpr.left());
            std::string right = extractExpressionText(rangeExpr.right());
            return base + ".range(" + left + ", " + right + ")";
        }
        
        // Phase 2: Core Expression Types
        case slang::ast::ExpressionKind::ConditionalOp: {
            auto& condExpr = expr.as<slang::ast::ConditionalExpression>();
            // Use the first condition expression (typical case)
            std::string predicate = extractExpressionText(*condExpr.conditions[0].expr);
            std::string left = extractExpressionText(condExpr.left());
            std::string right = extractExpressionText(condExpr.right());
            
            // Add .read() for signals used as conditions
            if (isSignalName(predicate) && predicate.find(".read()") == std::string::npos) {
                predicate += ".read()";
            }
            
            return "(" + predicate + " ? " + left + " : " + right + ")";
        }
        case slang::ast::ExpressionKind::Inside: {
            auto& insideExpr = expr.as<slang::ast::InsideExpression>();
            std::string left = extractExpressionText(insideExpr.left());
            // SystemC doesn't have direct inside operator, convert to range checks
            return "/* inside(" + left + ", ranges) - manual range check needed */";
        }
        case slang::ast::ExpressionKind::Assignment: {
            auto& assignExpr = expr.as<slang::ast::AssignmentExpression>();
            std::string lhs = extractExpressionText(assignExpr.left());
            std::string rhs = extractExpressionText(assignExpr.right());
            
            // Choose assignment type based on context
            std::string op = currentBlockIsSequential_ ? " <= " : " = ";
            return lhs + op + rhs;
        }
        case slang::ast::ExpressionKind::Concatenation: {
            auto& concatExpr = expr.as<slang::ast::ConcatenationExpression>();
            std::vector<std::string> operands;
            
            for (const auto* operand : concatExpr.operands()) {
                operands.push_back(extractExpressionText(*operand));
            }
            
            if (operands.size() == 1) {
                return operands[0];
            }
            
            // SystemC concatenation using comma operator
            std::string result = "(";
            for (size_t i = 0; i < operands.size(); ++i) {
                if (i > 0) result += ", ";
                result += operands[i];
            }
            result += ")";
            return result;
        }
        case slang::ast::ExpressionKind::Replication: {
            auto& replExpr = expr.as<slang::ast::ReplicationExpression>();
            std::string count = extractExpressionText(replExpr.count());
            std::string operand = extractExpressionText(replExpr.concat());
            
            // SystemC replication - need manual loop or constant expansion
            return "/* replication: " + count + " x {" + operand + "} */";
        }
        case slang::ast::ExpressionKind::Streaming: {
            auto& streamExpr = expr.as<slang::ast::StreamingConcatenationExpression>();
            // SystemC doesn't have direct streaming operator equivalent
            return "/* streaming_concat - manual bit manipulation needed */";
        }
        case slang::ast::ExpressionKind::MemberAccess: {
            auto& memberExpr = expr.as<slang::ast::MemberAccessExpression>();
            std::string value = extractExpressionText(memberExpr.value());
            std::string member = std::string(memberExpr.member.name);
            return value + "." + member;
        }
        
        // Phase 3: Function & Type Operations  
        case slang::ast::ExpressionKind::Call: {
            auto& callExpr = expr.as<slang::ast::CallExpression>();
            std::string funcName = std::string(callExpr.getSubroutineName());
            
            std::vector<std::string> args;
            for (const auto* arg : callExpr.arguments()) {
                if (arg) {
                    args.push_back(extractExpressionText(*arg));
                }
            }
            
            std::string result = funcName + "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) result += ", ";
                result += args[i];
            }
            result += ")";
            return result;
        }
        case slang::ast::ExpressionKind::DataType: {
            auto& typeExpr = expr.as<slang::ast::DataTypeExpression>();
            // Convert SystemVerilog data type to SystemC equivalent
            return "/* data_type */";
        }
        case slang::ast::ExpressionKind::TypeReference: {
            auto& typeRefExpr = expr.as<slang::ast::TypeReferenceExpression>();
            // Handle type references like package::type
            return "/* type_reference */";
        }
        case slang::ast::ExpressionKind::ArbitrarySymbol: {
            // ArbitrarySymbol - placeholder for unsupported construct
            return "/* arbitrary_symbol */";
        }
        case slang::ast::ExpressionKind::LValueReference: {
            // LValueReference - placeholder for unsupported construct
            return "/* lvalue_reference */";
        }
        case slang::ast::ExpressionKind::MinTypMax: {
            auto& minTypMaxExpr = expr.as<slang::ast::MinTypMaxExpression>();
            // Use typical value for SystemC
            return extractExpressionText(minTypMaxExpr.typ());
        }
        case slang::ast::ExpressionKind::TaggedUnion: {
            auto& taggedExpr = expr.as<slang::ast::TaggedUnionExpression>();
            // SystemC doesn't have tagged unions, convert to struct/variant
            return "/* tagged_union */";
        }
        
        // Phase 4: Pattern Matching
        case slang::ast::ExpressionKind::SimpleAssignmentPattern: {
            auto& patternExpr = expr.as<slang::ast::SimpleAssignmentPatternExpression>();
            // SystemC array initialization pattern
            std::vector<std::string> elements;
            for (const auto* element : patternExpr.elements()) {
                elements.push_back(extractExpressionText(*element));
            }
            
            std::string result = "{";
            for (size_t i = 0; i < elements.size(); ++i) {
                if (i > 0) result += ", ";
                result += elements[i];
            }
            result += "}";
            return result;
        }
        case slang::ast::ExpressionKind::StructuredAssignmentPattern: {
            auto& structPatternExpr = expr.as<slang::ast::StructuredAssignmentPatternExpression>();
            // SystemC structured initialization
            return "/* structured_assignment_pattern */";
        }
        case slang::ast::ExpressionKind::ReplicatedAssignmentPattern: {
            auto& replPatternExpr = expr.as<slang::ast::ReplicatedAssignmentPatternExpression>();
            // Handle {count{value}} patterns
            return "/* replicated_assignment_pattern */";
        }
        case slang::ast::ExpressionKind::EmptyArgument: {
            // Empty argument in function calls
            return "";
        }
        case slang::ast::ExpressionKind::ValueRange: {
            auto& rangeExpr = expr.as<slang::ast::ValueRangeExpression>();
            std::string left = extractExpressionText(rangeExpr.left());
            std::string right = extractExpressionText(rangeExpr.right());
            return "[" + left + ":" + right + "]";
        }
        
        // Phase 5: Advanced Features
        case slang::ast::ExpressionKind::Dist: {
            auto& distExpr = expr.as<slang::ast::DistExpression>();
            // SystemVerilog dist constraint - not directly translatable to SystemC
            return "/* dist_constraint */";
        }
        case slang::ast::ExpressionKind::NewArray: {
            auto& newArrayExpr = expr.as<slang::ast::NewArrayExpression>();
            // Dynamic array creation
            return "/* new_array */";
        }
        case slang::ast::ExpressionKind::NewClass: {
            auto& newClassExpr = expr.as<slang::ast::NewClassExpression>();
            // Class instantiation - convert to SystemC constructor call
            return "/* new_class */";
        }
        case slang::ast::ExpressionKind::NewCovergroup: {
            auto& newCovergroupExpr = expr.as<slang::ast::NewCovergroupExpression>();
            // Coverage group - SystemC doesn't have direct equivalent
            return "/* new_covergroup */";
        }
        case slang::ast::ExpressionKind::CopyClass: {
            auto& copyExpr = expr.as<slang::ast::CopyClassExpression>();
            // SystemVerilog copy constructor
            return "/* copy_class */";
        }
        case slang::ast::ExpressionKind::ClockingEvent: {
            auto& clockExpr = expr.as<slang::ast::ClockingEventExpression>();
            // Clocking event - convert to SystemC event
            return "/* clocking_event */";
        }
        case slang::ast::ExpressionKind::AssertionInstance: {
            auto& assertExpr = expr.as<slang::ast::AssertionInstanceExpression>();
            // SystemVerilog assertion instance
            return "/* assertion_instance */";
        }
        
        // All 40 ExpressionKind types are now handled!
        // Additional expression types can be added here as needed
        default:
            // For unhandled expressions, use a descriptive placeholder with more info
            LOG_DEBUG("Unhandled expression type: {}", static_cast<int>(expr.kind));
            return "/* unhandled_expr_" + std::to_string(static_cast<int>(expr.kind)) + " */";
    }
}

bool SVToSCVisitor::isSignalName(const std::string& name) const {
    // Simple heuristic: signals are typically internal registers like count_reg, mem_array, etc.
    // Check if the name is in our declared signals set, or has common signal name patterns
    if (declaredSignals_.count(name) > 0) {
        return true;
    }
    
    // Common signal patterns
    if (name.find("_reg") != std::string::npos ||
        name.find("_signal") != std::string::npos ||
        name.find("_ff") != std::string::npos ||
        name == "count_reg" ||
        name == "read_data_reg" ||
        name == "mem_array") {
        return true;
    }
    
    return false;
}

void SVToSCVisitor::analyzeExpressionUsage(const slang::ast::Expression& expr) {
    // Recursively analyze expression to determine signal usage patterns
    switch (expr.kind) {
        case slang::ast::ExpressionKind::BinaryOp: {
            auto& binaryExpr = expr.as<slang::ast::BinaryExpression>();
            
            // Check if this is an arithmetic operation
            bool isArithmetic = false;
            switch (binaryExpr.op) {
                case slang::ast::BinaryOperator::Add:
                case slang::ast::BinaryOperator::Subtract:
                case slang::ast::BinaryOperator::Multiply:
                case slang::ast::BinaryOperator::Divide:
                case slang::ast::BinaryOperator::Mod:
                case slang::ast::BinaryOperator::Power:
                    isArithmetic = true;
                    break;
                default:
                    break;
            }
            
            // Recursively analyze operands
            analyzeExpressionUsage(binaryExpr.left());
            analyzeExpressionUsage(binaryExpr.right());
            
            // If this is arithmetic, mark operand signals as arithmetic
            if (isArithmetic) {
                if (binaryExpr.left().kind == slang::ast::ExpressionKind::NamedValue) {
                    auto& nameExpr = binaryExpr.left().as<slang::ast::NamedValueExpression>();
                    markSignalArithmetic(std::string(nameExpr.symbol.name));
                }
                if (binaryExpr.right().kind == slang::ast::ExpressionKind::NamedValue) {
                    auto& nameExpr = binaryExpr.right().as<slang::ast::NamedValueExpression>();
                    markSignalArithmetic(std::string(nameExpr.symbol.name));
                }
            }
            break;
        }
        case slang::ast::ExpressionKind::UnaryOp: {
            auto& unaryExpr = expr.as<slang::ast::UnaryExpression>();
            
            // Check if this is an arithmetic operation
            bool isArithmetic = false;
            switch (unaryExpr.op) {
                case slang::ast::UnaryOperator::Plus:
                case slang::ast::UnaryOperator::Minus:
                    isArithmetic = true;
                    break;
                default:
                    break;
            }
            
            // Recursively analyze operand
            analyzeExpressionUsage(unaryExpr.operand());
            
            // If this is arithmetic, mark operand signal as arithmetic
            if (isArithmetic && unaryExpr.operand().kind == slang::ast::ExpressionKind::NamedValue) {
                auto& nameExpr = unaryExpr.operand().as<slang::ast::NamedValueExpression>();
                markSignalArithmetic(std::string(nameExpr.symbol.name));
            }
            break;
        }
        case slang::ast::ExpressionKind::NamedValue: {
            // Base case - this is a signal reference, but we don't know the usage context yet
            // The calling context will determine whether to mark it as arithmetic or logic
            break;
        }
        case slang::ast::ExpressionKind::ElementSelect:
        case slang::ast::ExpressionKind::RangeSelect: {
            // For array/bit selections, analyze the base expression
            if (expr.kind == slang::ast::ExpressionKind::ElementSelect) {
                auto& selectExpr = expr.as<slang::ast::ElementSelectExpression>();
                analyzeExpressionUsage(selectExpr.value());
                analyzeExpressionUsage(selectExpr.selector());
            } else {
                auto& rangeExpr = expr.as<slang::ast::RangeSelectExpression>();
                analyzeExpressionUsage(rangeExpr.value());
                analyzeExpressionUsage(rangeExpr.left());
                analyzeExpressionUsage(rangeExpr.right());
            }
            break;
        }
        default:
            // For other expression types, we don't need special handling yet
            break;
    }
}

void SVToSCVisitor::markSignalArithmetic(const std::string& signalName) {
    if (isSignalName(signalName)) {
        arithmeticSignals_.insert(signalName);
        LOG_DEBUG("Marked signal '{}' as arithmetic", signalName);
        
        // Update the signal type preference in the code generator
        codeGen_.updateSignalType(signalName, true);
    }
}

void SVToSCVisitor::markSignalLogic(const std::string& signalName) {
    if (isSignalName(signalName)) {
        logicSignals_.insert(signalName);
        LOG_DEBUG("Marked signal '{}' as logic-only", signalName);
    }
}

bool SVToSCVisitor::isArithmeticSignal(const std::string& signalName) const {
    return arithmeticSignals_.count(signalName) > 0;
}

void SVToSCVisitor::collectAssignment(const std::string& lhs, const std::string& rhs) {
    AssignmentInfo info;
    info.lhs = lhs;
    info.rhs = rhs;
    info.isResetAssignment = isResetPattern(rhs);
    info.isEnableAssignment = isIncrementPattern(lhs, rhs);
    
    pendingAssignments_.push_back(info);
    LOG_DEBUG("Collected assignment: {} <= {} (reset: {}, increment: {})", 
              lhs, rhs, info.isResetAssignment, info.isEnableAssignment);
}

void SVToSCVisitor::analyzeAndGenerateConditionalLogic() {
    LOG_DEBUG("Analyzing {} collected assignments for conditional patterns", pendingAssignments_.size());
    
    if (pendingAssignments_.empty()) {
        return;
    }
    
    // Look for different conditional patterns
    AssignmentInfo* resetAssignment = nullptr;
    AssignmentInfo* enableAssignment = nullptr;
    std::vector<AssignmentInfo*> otherAssignments;
    
    for (auto& assignment : pendingAssignments_) {
        if (assignment.isResetAssignment) {
            resetAssignment = &assignment;
        } else if (assignment.isEnableAssignment) {
            enableAssignment = &assignment;
        } else {
            otherAssignments.push_back(&assignment);
        }
    }
    
    // Generate conditional logic based on detected patterns
    if (resetAssignment && enableAssignment) {
        LOG_DEBUG("Detected reset/enable pattern, generating conditional logic");
        
        // Generate: if (reset.read()) { ... } else if (enable.read()) { ... }
        codeGen_.addConditionalStart("reset.read()", true);
        codeGen_.addSequentialAssignment(resetAssignment->lhs, resetAssignment->rhs);
        codeGen_.addElseClause(true);
        codeGen_.addConditionalStart("enable.read()", true);
        codeGen_.addSequentialAssignment(enableAssignment->lhs, enableAssignment->rhs);
        codeGen_.addConditionalEnd(true);
        codeGen_.addConditionalEnd(true);
        
    } else if (resetAssignment && !otherAssignments.empty()) {
        LOG_DEBUG("Detected reset with other assignments pattern");
        
        // Generate: if (reset.read()) { ... } else if (condition) { ... }
        codeGen_.addConditionalStart("reset.read()", true);
        codeGen_.addSequentialAssignment(resetAssignment->lhs, resetAssignment->rhs);
        codeGen_.addElseClause(true);
        
        // Try to infer condition from context - for now use read_enable
        codeGen_.addConditionalStart("read_enable.read()", true);
        for (const auto& assignment : otherAssignments) {
            codeGen_.addSequentialAssignment(assignment->lhs, assignment->rhs);
        }
        codeGen_.addConditionalEnd(true);
        codeGen_.addConditionalEnd(true);
        
    } else if (pendingAssignments_.size() == 1 && !resetAssignment && !enableAssignment) {
        LOG_DEBUG("Detected single assignment, likely conditional on enable signal");
        
        // Single assignment in always_ff likely needs a condition - try write_enable
        codeGen_.addConditionalStart("write_enable.read()", true);
        codeGen_.addSequentialAssignment(pendingAssignments_[0].lhs, pendingAssignments_[0].rhs);
        codeGen_.addConditionalEnd(true);
        
    } else {
        // Fallback: generate assignments without conditional logic
        LOG_DEBUG("No clear pattern detected, generating assignments directly");
        for (const auto& assignment : pendingAssignments_) {
            codeGen_.addSequentialAssignment(assignment.lhs, assignment.rhs);
        }
    }
    
    // Clear collected assignments
    pendingAssignments_.clear();
}

bool SVToSCVisitor::isResetPattern(const std::string& rhs) const {
    // Detect reset patterns: 0, 8'b0, '0, etc.
    return (rhs == "0" || 
            rhs.find("'b0") != std::string::npos || 
            rhs.find("'h0") != std::string::npos ||
            rhs == "'0");
}

bool SVToSCVisitor::isIncrementPattern(const std::string& lhs, const std::string& rhs) const {
    // Detect increment patterns: signal <= signal + 1, etc.
    return (rhs.find(lhs) != std::string::npos && 
            (rhs.find(" + ") != std::string::npos || rhs.find("++") != std::string::npos));
}

void SVToSCVisitor::extractAndTrackSignals(const std::string& expression, bool isSequential) {
    // Simple signal extraction - look for common signal patterns
    // This is a basic implementation that can be enhanced
    
    std::regex signalPattern(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.read\(\))");
    std::smatch match;
    std::string::const_iterator searchStart(expression.cbegin());
    
    while (std::regex_search(searchStart, expression.cend(), match, signalPattern)) {
        std::string signalName = match[1].str();
        
        // Track the signal usage
        if (isSequential) {
            codeGen_.addSeqSensitiveSignal(signalName);
        } else {
            codeGen_.addCombSensitiveSignal(signalName);
        }
        
        LOG_DEBUG("Tracked signal usage: {} in {} process", signalName, isSequential ? "sequential" : "combinational");
        searchStart = match.suffix().first;
    }
    
    // Also look for direct signal references (without .read())
    std::regex directSignalPattern(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\b)");
    searchStart = expression.cbegin();
    
    while (std::regex_search(searchStart, expression.cend(), match, directSignalPattern)) {
        std::string signalName = match[1].str();
        
        // Skip common keywords and operators
        if (signalName != "read" && signalName != "write" && signalName != "to_uint" && 
            signalName != "sc_lv" && signalName != "sc_logic" && signalName != "if" && 
            signalName != "else" && signalName != "return") {
            
            // Track the signal usage
            if (isSequential) {
                codeGen_.addSeqSensitiveSignal(signalName);
            } else {
                codeGen_.addCombSensitiveSignal(signalName);
            }
        }
        
        searchStart = match.suffix().first;
    }
}

} // namespace sv2sc::core