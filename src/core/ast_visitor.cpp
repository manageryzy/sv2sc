#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <fmt/format.h>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/expressions/LiteralExpressions.h>

namespace sv2sc::core {

SVToSCVisitor::SVToSCVisitor(codegen::SystemCCodeGenerator& generator) 
    : codeGen_(generator) {
}

void SVToSCVisitor::handle(const slang::ast::InstanceSymbol& node) {
    LOG_INFO("Processing module instance: {}", node.name);
    
    currentModule_ = std::string(node.name);
    portNames_.clear();  // Clear port names for new module
    codeGen_.beginModule(currentModule_);
    
    // Visit all child symbols (ports, variables, etc.)
    visitDefault(node);
    
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
    
    Signal signal;
    signal.name = std::string(node.name);
    
    // Get type information
    auto& type = node.getType();
    if (type.isPackedArray()) {
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
    
    codeGen_.addSignal(signal);
    LOG_DEBUG("Added signal: {} (width: {})", signal.name, signal.width);
}

void SVToSCVisitor::handle(const slang::ast::VariableDeclStatement& node) {
    // Get the variable symbol and handle it
    auto& symbol = node.symbol;
    handle(symbol);
}

void SVToSCVisitor::handle(const slang::ast::AssignmentExpression& node) {
    LOG_DEBUG("Processing assignment expression");
    
    // Extract left and right hand sides
    std::string lhs = extractExpressionText(node.left());
    std::string rhs = extractExpressionText(node.right());
    
    // For now, treat all assignments as blocking - we'll enhance this later
    // TODO: Determine blocking vs non-blocking based on context (always_ff vs always_comb)
    codeGen_.addBlockingAssignment(lhs, rhs);
}

void SVToSCVisitor::handle(const slang::ast::ExpressionStatement& node) {
    LOG_DEBUG("Processing expression statement");
    
    // Visit the contained expression
    visitDefault(node);
}

void SVToSCVisitor::handle(const slang::ast::ProceduralBlockSymbol& node) {
    LOG_DEBUG("Processing procedural block: {}", node.procedureKind == slang::ast::ProceduralBlockKind::Always ? "always" : "initial");
    
    // For now, just visit the body - we'll enhance this later to handle sensitivity lists
    visitDefault(node);
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
    // This is a simplified implementation - for now just try to get variable names
    
    // Check expression kind and handle accordingly
    switch (expr.kind) {
        case slang::ast::ExpressionKind::NamedValue: {
            auto& nameExpr = expr.as<slang::ast::NamedValueExpression>();
            return std::string(nameExpr.symbol.name);
        }
        case slang::ast::ExpressionKind::IntegerLiteral: {
            auto& literalExpr = expr.as<slang::ast::IntegerLiteral>();
            return literalExpr.getValue().toString(slang::LiteralBase::Decimal);
        }
        default:
            // For other expressions, return a generic placeholder
            return "unknown_expr";
    }
}

} // namespace sv2sc::core