#pragma once

#include "slang/ast/ASTVisitor.h"
#include "utils/logger.h"
#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/PortSymbols.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/ast/symbols/BlockSymbols.h>
#include <slang/ast/statements/MiscStatements.h>
#include <slang/ast/expressions/AssignmentExpressions.h>
#include <slang/ast/Expression.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace sv2sc {
    namespace codegen {
        class SystemCCodeGenerator;
    }
}

namespace sv2sc::core {

using SystemCCodeGenerator = codegen::SystemCCodeGenerator;

class SVToSCVisitor : public slang::ast::ASTVisitor<SVToSCVisitor, true, true> {
public:
    explicit SVToSCVisitor(SystemCCodeGenerator& generator);
    
    // Set target module to focus translation on specific module
    void setTargetModule(const std::string& targetModule);

    void handle(const slang::ast::InstanceSymbol& node);
    void handle(const slang::ast::InstanceBodySymbol& node);  // Module definitions
    void handle(const slang::ast::PortSymbol& node);
    void handle(const slang::ast::VariableSymbol& node);
    void handle(const slang::ast::ParameterSymbol& node);
    void handle(const slang::ast::VariableDeclStatement& node);
    void handle(const slang::ast::AssignmentExpression& node);
    void handle(const slang::ast::ExpressionStatement& node);
    void handle(const slang::ast::ProceduralBlockSymbol& node);
    void handle(const slang::ast::ContinuousAssignSymbol& node);
    void handle(const slang::ast::SubroutineSymbol& node);
    
    // Generic handler to catch unhandled statement types
    template<typename T>
    void visitDefault(const T& node) {
        sv2sc::utils::Logger::getInstance().debug("Visiting unhandled AST node type: {}", typeid(T).name());
        slang::ast::ASTVisitor<SVToSCVisitor, true, true>::visitDefault(node);
    }

private:
    SystemCCodeGenerator& codeGen_;
    std::string currentModule_;
    std::string targetModule_;  // Target module to translate (if set, only translate this module)
    int indentLevel_ = 0;
    std::vector<std::string> portNames_;  // Track port names to avoid duplicate signals
    std::set<std::string> declaredSignals_;  // Track all declared signals to prevent duplicates
    bool currentBlockIsSequential_ = false;  // Track if current procedural block is sequential
    
    // Pattern-based conditional logic detection
    struct AssignmentInfo {
        std::string lhs;
        std::string rhs;
        std::string condition;  // Inferred condition for this assignment
        bool isResetAssignment = false;
        bool isEnableAssignment = false;
    };
    std::vector<AssignmentInfo> pendingAssignments_;  // Collect assignments to analyze patterns
    
    // Signal usage analysis for proper type mapping
    std::set<std::string> arithmeticSignals_;  // Signals used in arithmetic operations
    std::set<std::string> logicSignals_;       // Signals used in logic-only operations

    std::string getIndent() const;
    void increaseIndent();
    void decreaseIndent();
    std::string extractExpressionText(const slang::ast::Expression& expr) const;
    bool isSignalName(const std::string& name) const;
    
    // Signal usage analysis methods
    void analyzeExpressionUsage(const slang::ast::Expression& expr);
    void markSignalArithmetic(const std::string& signalName);
    void markSignalLogic(const std::string& signalName);
    bool isArithmeticSignal(const std::string& signalName) const;
    
    // Pattern-based conditional logic methods
    void collectAssignment(const std::string& lhs, const std::string& rhs);
    void analyzeAndGenerateConditionalLogic();
    bool isResetPattern(const std::string& rhs) const;
    bool isIncrementPattern(const std::string& lhs, const std::string& rhs) const;
    
    // Signal usage tracking helpers
    void extractAndTrackSignals(const std::string& expression, bool isSequential);
    void extractPortConnections(const slang::ast::InstanceSymbol& node, const std::string& uniqueInstanceName, int generateIndex);
    std::string substituteGenerateVariable(const std::string& expression, int index) const;
    void handleAdvancedFeatures(const std::string& sourceText) const;
};

} // namespace sv2sc::core