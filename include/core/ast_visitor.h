#pragma once

#include "slang/ast/ASTVisitor.h"
#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/PortSymbols.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/ast/symbols/BlockSymbols.h>
#include <slang/ast/statements/MiscStatements.h>
#include <slang/ast/expressions/AssignmentExpressions.h>
#include <slang/ast/Expression.h>
#include <memory>
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

    void handle(const slang::ast::InstanceSymbol& node);
    void handle(const slang::ast::PortSymbol& node);
    void handle(const slang::ast::VariableSymbol& node);
    void handle(const slang::ast::VariableDeclStatement& node);
    void handle(const slang::ast::AssignmentExpression& node);
    void handle(const slang::ast::ExpressionStatement& node);
    void handle(const slang::ast::ProceduralBlockSymbol& node);
    void handle(const slang::ast::ContinuousAssignSymbol& node);

private:
    SystemCCodeGenerator& codeGen_;
    std::string currentModule_;
    int indentLevel_ = 0;
    std::vector<std::string> portNames_;  // Track port names to avoid duplicate signals

    std::string getIndent() const;
    void increaseIndent();
    void decreaseIndent();
    std::string extractExpressionText(const slang::ast::Expression& expr) const;
};

} // namespace sv2sc::core