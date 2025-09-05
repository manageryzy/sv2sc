#pragma once

#include <memory>
#include <string>
#include <unordered_map>

// MLIR includes - we always have real CIRCT
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

// CIRCT includes
#include "circt/Dialect/HW/HWOps.h"

// Slang includes
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/PortSymbols.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/ast/Expression.h>
#include <slang/ast/Statement.h>

namespace sv2sc::mlir_support {

/**
 * @brief Builder class for converting SystemVerilog AST to HW dialect
 * 
 * This class implements the first stage of the MLIR-based translation pipeline,
 * converting from slang's SystemVerilog AST representation to CIRCT's HW dialect.
 */
class SVToHWBuilder {
public:
    explicit SVToHWBuilder(mlir::MLIRContext* context);
    ~SVToHWBuilder() = default;

    /**
     * @brief Build HW dialect module from SystemVerilog AST
     * @param moduleAST The SystemVerilog module AST from slang
     * @return MLIR ModuleOp containing the HW dialect representation
     */
    mlir::ModuleOp buildFromAST(const slang::ast::InstanceBodySymbol& moduleAST);

private:
    mlir::MLIRContext* context_;
    mlir::OpBuilder builder_;
    mlir::ModuleOp currentMLIRModule_;
    circt::hw::HWModuleOp currentHWModule_;
    
    // Value mapping for SSA form
    std::unordered_map<std::string, mlir::Value> valueMap_;
    std::unordered_map<std::string, mlir::Type> typeMap_;
    
    // Parameter tracking
    std::unordered_map<std::string, int64_t> parameterMap_;  // Track parameter values
    
    // Sequential logic tracking
    std::unordered_map<std::string, mlir::Value> registerMap_;  // Track register signals
    std::unordered_map<std::string, mlir::Value> resetValueMap_; // Track reset values for registers
    mlir::Value clockSignal_;  // Current clock signal
    mlir::Value resetSignal_; // Current reset signal
    bool inAlwaysFF_ = false;  // Track if we're in an always_ff block
    bool inResetCondition_ = false;  // Track if we're in a reset condition (for reset value detection)
    int statementDepth_ = 0;  // Track recursion depth to prevent infinite loops
    
    // Module building methods
    circt::hw::HWModuleOp buildModule(const slang::ast::InstanceBodySymbol& moduleAST);
    void buildPortList(const slang::ast::InstanceBodySymbol& moduleAST,
                       std::vector<circt::hw::PortInfo>& ports);
    
    // Expression building methods
    mlir::Value buildExpression(const slang::ast::Expression& expr);
    mlir::Value buildLiteralExpression(const slang::ast::Expression& expr);
    mlir::Value buildNamedValueExpression(const slang::ast::Expression& expr);
    mlir::Value buildBinaryExpression(const slang::ast::Expression& expr);
    mlir::Value buildUnaryExpression(const slang::ast::Expression& expr);
    mlir::Value buildConditionalExpression(const slang::ast::Expression& expr);
    mlir::Value buildSelectExpression(const slang::ast::Expression& expr);
    mlir::Value buildMemberAccessExpression(const slang::ast::Expression& expr);
    mlir::Value buildConcatenationExpression(const slang::ast::Expression& expr);
    mlir::Value buildReplicationExpression(const slang::ast::Expression& expr);
    mlir::Value buildAssignmentExpression(const slang::ast::Expression& expr);
    mlir::Value buildCallExpression(const slang::ast::Expression& expr);
    mlir::Value buildConversionExpression(const slang::ast::Expression& expr);
    
    // Statement building methods
    void buildStatement(const slang::ast::Statement& stmt);
    void buildAssignmentStatement(const slang::ast::Statement& stmt);
    void buildConditionalStatement(const slang::ast::Statement& stmt);
    void buildProceduralBlock(const slang::ast::Statement& stmt);
    void buildBlockStatement(const slang::ast::Statement& stmt);
    void buildVariableDeclStatement(const slang::ast::Statement& stmt);
    void buildExpressionStatement(const slang::ast::Statement& stmt);
    void buildTimingControlStatement(const slang::ast::Statement& stmt);
    void buildForLoopStatement(const slang::ast::Statement& stmt);
    void buildWhileLoopStatement(const slang::ast::Statement& stmt);
    void buildCaseStatement(const slang::ast::Statement& stmt);
    
    // Sequential logic generation
    void generateSequentialLogic();
    
    // Type conversion utilities
    mlir::Type convertSVTypeToHW(const slang::ast::Type& svType);
    mlir::Type getIntegerType(int width);
    mlir::Type getBitVectorType(int width);
    
    // Utility methods
    mlir::Location getLocation(const slang::ast::Symbol& symbol);
    mlir::Location getUnknownLocation();
    std::string sanitizeName(const std::string& name);
    std::pair<mlir::Value, mlir::Value> ensureMatchingTypes(mlir::Value val1, mlir::Value val2);
    mlir::Value createTypeSafeMux(mlir::Location loc, mlir::Value condition, mlir::Value trueVal, mlir::Value falseVal);
    
    // Value management
    void setValueForSignal(const std::string& name, mlir::Value value);
    mlir::Value getValueForSignal(const std::string& name);
    bool hasValueForSignal(const std::string& name) const;
    
    // Parameter management
    void setParameter(const std::string& name, int64_t value);
    int64_t getParameter(const std::string& name) const;
    bool hasParameter(const std::string& name) const;
};

} // namespace sv2sc::mlir_support

