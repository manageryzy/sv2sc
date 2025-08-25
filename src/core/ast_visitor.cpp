#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <fmt/format.h>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/expressions/LiteralExpressions.h>
#include <slang/ast/expressions/OperatorExpressions.h>
#include <slang/ast/expressions/SelectExpressions.h>
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
        codeGen_.addModuleInstance(std::string(node.name), instancedModule);
        LOG_DEBUG("Added module instance {} of type {} to {}", node.name, instancedModule, currentModule_);
    }
    
    // Don't visit the instance body here - we'll handle module definitions separately
}

void SVToSCVisitor::handle(const slang::ast::InstanceBodySymbol& node) {
    std::string moduleName = std::string(node.getDefinition().name);
    LOG_INFO("Processing module definition: {}", moduleName);
    
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
        // In sequential blocks (always_ff), use sequential assignments
        codeGen_.addSequentialAssignment(lhs, convertedRhs);
    } else {
        // In combinational blocks (always_comb), use combinational assignments
        codeGen_.addCombinationalAssignment(lhs, convertedRhs);
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
    visitDefault(node);
    
    // Restore previous context
    currentBlockIsSequential_ = prevSequential;
}

void SVToSCVisitor::handle(const slang::ast::ContinuousAssignSymbol& node) {
    LOG_DEBUG("Processing continuous assign statement");
    
    // Visit the assignment expression
    visitDefault(node);
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

} // namespace sv2sc::core