# Slang Expression APIs

This document describes the Slang APIs for handling SystemVerilog expressions, operators, and literals.

## Overview

Slang provides a comprehensive expression system that represents all SystemVerilog expressions as AST nodes. The sv2sc project processes these expressions to generate equivalent SystemC code.

## Core Expression Classes

### `slang::ast::Expression`

Base class for all expressions.

```cpp
#include <slang/ast/Expression.h>

// Basic expression processing
std::string extractExpressionText(const slang::ast::Expression& expr) {
    switch (expr.kind) {
        case slang::ast::ExpressionKind::IntegerLiteral:
            // Handle integer literal
            break;
        case slang::ast::ExpressionKind::BinaryOp:
            // Handle binary operation
            break;
        // ... more cases
    }
}
```

**Key Properties:**
- `kind` - Expression kind enumeration
- Type-specific casting with `expr.as<SpecificType>()`

### Expression Kind Enumeration

```cpp
// Major expression categories
enum class ExpressionKind {
    // Literals
    Invalid,
    IntegerLiteral,
    RealLiteral,
    TimeLiteral,
    StringLiteral,
    NullLiteral,
    UnboundedLiteral,
    UnbasedUnsizedIntegerLiteral,
    
    // Values and References
    NamedValue,
    HierarchicalValue,
    
    // Operators
    UnaryOp,
    BinaryOp,
    ConditionalOp,
    
    // Selections and Access
    ElementSelect,
    RangeSelect,
    MemberAccess,
    
    // Complex Expressions
    Concatenation,
    Replication,
    Streaming,
    Call,
    Conversion,
    Assignment,
    
    // Type Expressions
    DataType,
    TypeReference,
    
    // Special
    ArbitrarySymbol,
    LValueReference,
    MinTypMax,
    Inside
};
```

## Literal Expressions

### `slang::ast::IntegerLiteral`

Represents integer literal values.

```cpp
case slang::ast::ExpressionKind::IntegerLiteral: {
    auto& literalExpr = expr.as<slang::ast::IntegerLiteral>();
    
    // Get the value and convert to decimal string
    auto value = literalExpr.getValue();
    return value.toString(slang::LiteralBase::Decimal);
}
```

**Key Methods:**
- `getValue()` - Returns the integer value
- `toString(base)` - Convert to string with specified base

### `slang::ast::RealLiteral`

Represents real (floating-point) literal values.

```cpp
case slang::ast::ExpressionKind::RealLiteral: {
    auto& realExpr = expr.as<slang::ast::RealLiteral>();
    double value = realExpr.getValue();
    return std::to_string(value);
}
```

### `slang::ast::TimeLiteral`

Represents time literal values.

```cpp
case slang::ast::ExpressionKind::TimeLiteral: {
    auto& timeExpr = expr.as<slang::ast::TimeLiteral>();
    
    // Convert to SystemC time format
    return "sc_time(" + std::to_string(timeExpr.getValue()) + ", SC_NS)";
}
```

### `slang::ast::StringLiteral`

Represents string literal values.

```cpp
case slang::ast::ExpressionKind::StringLiteral: {
    auto& stringExpr = expr.as<slang::ast::StringLiteral>();
    std::string value = std::string(stringExpr.getValue());
    
    // Handle SystemVerilog special literals like '0
    if (value == "'0") {
        return "0";
    }
    return "\"" + value + "\"";
}
```

### `slang::ast::UnbasedUnsizedIntegerLiteral`

Represents unbased unsized literals ('0, '1, 'x, 'z).

```cpp
case slang::ast::ExpressionKind::UnbasedUnsizedIntegerLiteral: {
    auto& literalExpr = expr.as<slang::ast::UnbasedUnsizedIntegerLiteral>();
    auto value = literalExpr.getValue();
    
    if (value == 0) return "0";
    if (value == 1) return "1";
    
    return value.toString(slang::LiteralBase::Decimal);
}
```

## Value and Reference Expressions

### `slang::ast::NamedValueExpression`

Represents named values (variables, signals).

```cpp
case slang::ast::ExpressionKind::NamedValue: {
    auto& nameExpr = expr.as<slang::ast::NamedValueExpression>();
    return std::string(nameExpr.symbol.name);
}
```

### `slang::ast::HierarchicalValueExpression`

Represents hierarchical references.

```cpp
case slang::ast::ExpressionKind::HierarchicalValue: {
    auto& hierExpr = expr.as<slang::ast::HierarchicalValueExpression>();
    // Convert hierarchical path to SystemC scope resolution
    return std::string(hierExpr.symbol.name);
}
```

## Operator Expressions

### `slang::ast::UnaryExpression`

Represents unary operations.

```cpp
case slang::ast::ExpressionKind::UnaryOp: {
    auto& unaryExpr = expr.as<slang::ast::UnaryExpression>();
    std::string operand = extractExpressionText(unaryExpr.operand());
    
    switch (unaryExpr.op) {
        case slang::ast::UnaryOperator::Plus: 
            return "+" + operand;
        case slang::ast::UnaryOperator::Minus: 
            return "-" + operand;
        case slang::ast::UnaryOperator::BitwiseNot: 
            return "~" + operand;
        case slang::ast::UnaryOperator::BitwiseAnd: 
            return "&" + operand;
        case slang::ast::UnaryOperator::BitwiseOr: 
            return "|" + operand;
        case slang::ast::UnaryOperator::BitwiseXor: 
            return "^" + operand;
        case slang::ast::UnaryOperator::LogicalNot: 
            return "!" + operand;
        default: 
            return "(" + operand + ")";
    }
}
```

**Unary Operators:**
- `Plus`, `Minus` - Arithmetic unary
- `BitwiseNot`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor` - Bitwise reduction
- `LogicalNot` - Logical negation

### `slang::ast::BinaryExpression`

Represents binary operations.

```cpp
case slang::ast::ExpressionKind::BinaryOp: {
    auto& binaryExpr = expr.as<slang::ast::BinaryExpression>();
    std::string lhs = extractExpressionText(binaryExpr.left());
    std::string rhs = extractExpressionText(binaryExpr.right());
    
    bool isArithmetic = false;
    std::string op;
    
    switch (binaryExpr.op) {
        case slang::ast::BinaryOperator::Add: 
            op = " + "; isArithmetic = true; break;
        case slang::ast::BinaryOperator::Subtract: 
            op = " - "; isArithmetic = true; break;
        case slang::ast::BinaryOperator::Multiply: 
            op = " * "; isArithmetic = true; break;
        case slang::ast::BinaryOperator::Divide: 
            op = " / "; isArithmetic = true; break;
        case slang::ast::BinaryOperator::Mod: 
            op = " % "; isArithmetic = true; break;
        case slang::ast::BinaryOperator::BinaryAnd: 
            op = " & "; break;
        case slang::ast::BinaryOperator::BinaryOr: 
            op = " | "; break;
        case slang::ast::BinaryOperator::BinaryXor: 
            op = " ^ "; break;
        case slang::ast::BinaryOperator::LogicalAnd: 
            op = " && "; break;
        case slang::ast::BinaryOperator::LogicalOr: 
            op = " || "; break;
        case slang::ast::BinaryOperator::Equality: 
            op = " == "; break;
        case slang::ast::BinaryOperator::Inequality: 
            op = " != "; break;
        case slang::ast::BinaryOperator::LessThan: 
            op = " < "; break;
        case slang::ast::BinaryOperator::LessThanEqual: 
            op = " <= "; break;
        case slang::ast::BinaryOperator::GreaterThan: 
            op = " > "; break;
        case slang::ast::BinaryOperator::GreaterThanEqual: 
            op = " >= "; break;
        default: 
            op = " ? "; break;
    }
    
    return "(" + lhs + op + rhs + ")";
}
```

## Selection and Access Expressions

### `slang::ast::ElementSelectExpression`

Represents array element selection.

```cpp
case slang::ast::ExpressionKind::ElementSelect: {
    auto& selectExpr = expr.as<slang::ast::ElementSelectExpression>();
    std::string base = extractExpressionText(selectExpr.value());
    std::string index = extractExpressionText(selectExpr.selector());
    
    return base + "[" + index + "]";
}
```

### `slang::ast::RangeSelectExpression`

Represents bit range selection.

```cpp
case slang::ast::ExpressionKind::RangeSelect: {
    auto& rangeExpr = expr.as<slang::ast::RangeSelectExpression>();
    std::string base = extractExpressionText(rangeExpr.value());
    std::string left = extractExpressionText(rangeExpr.left());
    std::string right = extractExpressionText(rangeExpr.right());
    
    return base + ".range(" + left + ", " + right + ")";
}
```

### `slang::ast::MemberAccessExpression`

Represents struct/union member access.

```cpp
case slang::ast::ExpressionKind::MemberAccess: {
    auto& memberExpr = expr.as<slang::ast::MemberAccessExpression>();
    std::string base = extractExpressionText(memberExpr.value());
    std::string member = std::string(memberExpr.member.name);
    
    return base + "." + member;
}
```

## Complex Expressions

### `slang::ast::ConditionalExpression`

Represents conditional (ternary) expressions.

```cpp
case slang::ast::ExpressionKind::ConditionalOp: {
    auto& condExpr = expr.as<slang::ast::ConditionalExpression>();
    
    // Use the first condition expression
    std::string predicate = extractExpressionText(*condExpr.conditions[0].expr);
    std::string left = extractExpressionText(condExpr.left());
    std::string right = extractExpressionText(condExpr.right());
    
    return "(" + predicate + " ? " + left + " : " + right + ")";
}
```

### `slang::ast::ConcatenationExpression`

Represents bit concatenation.

```cpp
case slang::ast::ExpressionKind::Concatenation: {
    auto& concatExpr = expr.as<slang::ast::ConcatenationExpression>();
    
    std::string result = "(";
    bool first = true;
    
    for (auto operand : concatExpr.operands()) {
        if (!first) result += ", ";
        result += extractExpressionText(*operand);
        first = false;
    }
    
    result += ")"; // SystemC concatenation format
    return result;
}
```

### `slang::ast::ReplicationExpression`

Represents bit replication.

```cpp
case slang::ast::ExpressionKind::Replication: {
    auto& replExpr = expr.as<slang::ast::ReplicationExpression>();
    std::string count = extractExpressionText(replExpr.count());
    std::string value = extractExpressionText(replExpr.concat());
    
    return "/* replication: " + count + " x " + value + " */";
}
```

### `slang::ast::CallExpression`

Represents function/task calls.

```cpp
case slang::ast::ExpressionKind::Call: {
    auto& callExpr = expr.as<slang::ast::CallExpression>();
    
    // Get function name
    std::string funcName;
    if (callExpr.isSystemCall()) {
        funcName = std::string(callExpr.getSubroutineName());
    }
    
    // Build argument list
    std::string args = "";
    // Process arguments...
    
    return funcName + "(" + args + ")";
}
```

## Type Conversion and Assignment

### `slang::ast::ConversionExpression`

Represents type conversions.

```cpp
case slang::ast::ExpressionKind::Conversion: {
    auto& convExpr = expr.as<slang::ast::ConversionExpression>();
    std::string operand = extractExpressionText(convExpr.operand());
    
    // Handle SystemVerilog sized literals like 8'b0, 1'b1
    if (operand.find("'d") != std::string::npos || 
        operand.find("'b") != std::string::npos ||
        operand.find("'h") != std::string::npos) {
        
        // Extract and convert literal format
        // Return appropriate SystemC equivalent
    }
    
    return operand;  // Let SystemC handle automatic conversion
}
```

### `slang::ast::AssignmentExpression`

Represents assignments within expressions.

```cpp
case slang::ast::ExpressionKind::Assignment: {
    auto& assignExpr = expr.as<slang::ast::AssignmentExpression>();
    std::string lhs = extractExpressionText(assignExpr.left());
    std::string rhs = extractExpressionText(assignExpr.right());
    
    // Handle blocking vs non-blocking assignments
    return lhs + " = " + rhs;  // Simplified
}
```

## Usage Patterns

### Expression Analysis

```cpp
void analyzeExpressionUsage(const slang::ast::Expression& expr) {
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
                    isArithmetic = true;
                    break;
            }
            
            if (isArithmetic) {
                // Mark signals as arithmetic for proper SystemC type selection
                markSignalsArithmetic(binaryExpr.left());
                markSignalsArithmetic(binaryExpr.right());
            }
            break;
        }
    }
}
```

### SystemC Signal Handling

```cpp
std::string processSignalExpression(const slang::ast::Expression& expr) {
    std::string result = extractExpressionText(expr);
    
    // Add .read() for SystemC signal access
    if (isSignalName(result) && result.find(".read()") == std::string::npos) {
        result += ".read()";
    }
    
    return result;
}
```

## Integration Notes

- Expression processing is central to SystemVerilog-to-SystemC conversion
- Proper handling of SystemVerilog literals and operators
- Type conversion management between SystemVerilog and SystemC
- Signal access pattern generation for SystemC compatibility
- Support for complex expressions including concatenation and replication