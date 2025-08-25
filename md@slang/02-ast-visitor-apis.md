# Slang AST Visitor APIs

This document describes the Slang AST visitor APIs used for traversing and analyzing SystemVerilog syntax trees.

## Overview

Slang provides a powerful visitor pattern implementation for traversing Abstract Syntax Trees (ASTs). The sv2sc project uses this to systematically process SystemVerilog constructs and convert them to SystemC.

## Core Visitor Classes

### `slang::ast::ASTVisitor<T, VisitStatements, VisitExpressions>`

Base template class for AST visitors.

```cpp
#include "slang/ast/ASTVisitor.h"

// Define a custom visitor
class SVToSCVisitor : public slang::ast::ASTVisitor<SVToSCVisitor, true, true> {
public:
    // Constructor
    explicit SVToSCVisitor(SystemCCodeGenerator& generator);
    
    // Handler methods for specific AST node types
    void handle(const slang::ast::InstanceSymbol& node);
    void handle(const slang::ast::PortSymbol& node);
    void handle(const slang::ast::VariableSymbol& node);
    // ... more handlers
};
```

**Template Parameters:**
- `T` - Derived visitor class (CRTP pattern)
- `VisitStatements` - Whether to visit statement nodes
- `VisitExpressions` - Whether to visit expression nodes

## AST Symbol Types

### `slang::ast::InstanceSymbol`

Represents a module instance in the design.

```cpp
void handle(const slang::ast::InstanceSymbol& node) {
    // Get instance name
    std::string instanceName = std::string(node.name);
    
    // Visit all child symbols (ports, variables, etc.)
    visitDefault(node);
}
```

**Key Properties:**
- `name` - Instance name
- Child symbols accessible through iteration

### `slang::ast::PortSymbol`

Represents a module port.

```cpp
void handle(const slang::ast::PortSymbol& node) {
    // Get port name
    std::string portName = std::string(node.name);
    
    // Get port direction
    switch (node.direction) {
        case slang::ast::ArgumentDirection::In:
            // Handle input port
            break;
        case slang::ast::ArgumentDirection::Out:
            // Handle output port
            break;
        case slang::ast::ArgumentDirection::InOut:
            // Handle inout port
            break;
    }
    
    // Get port type information
    auto& type = node.getType();
    if (type.isPackedArray()) {
        int width = static_cast<int>(type.getBitWidth());
        bool isFourState = type.isFourState();
    }
}
```

**Key Properties:**
- `name` - Port name
- `direction` - Port direction (In/Out/InOut)
- `getType()` - Returns type information

### `slang::ast::VariableSymbol`

Represents a variable declaration.

```cpp
void handle(const slang::ast::VariableSymbol& node) {
    // Get variable name
    std::string varName = std::string(node.name);
    
    // Get type information
    auto& type = node.getType();
    
    // Check if it's an array type
    if (type.isKind(slang::ast::SymbolKind::FixedSizeUnpackedArrayType)) {
        auto& arrayType = type.as<slang::ast::FixedSizeUnpackedArrayType>();
        // Handle array dimensions
    }
    
    // Check for initial value
    if (node.getInitializer()) {
        // Handle initialization expression
    }
}
```

**Key Properties:**
- `name` - Variable name
- `getType()` - Returns type information
- `getInitializer()` - Returns initialization expression if present

### `slang::ast::ProceduralBlockSymbol`

Represents procedural blocks (always, initial).

```cpp
void handle(const slang::ast::ProceduralBlockSymbol& node) {
    using namespace slang::ast;
    
    // Determine block type
    switch (node.procedureKind) {
        case ProceduralBlockKind::Always:
            // Handle always block
            break;
        case ProceduralBlockKind::AlwaysComb:
            // Handle always_comb block
            break;
        case ProceduralBlockKind::AlwaysFF:
            // Handle always_ff block
            break;
        case ProceduralBlockKind::Initial:
            // Handle initial block
            break;
    }
    
    // Visit statements in the block
    visitDefault(node);
}
```

**Key Properties:**
- `procedureKind` - Type of procedural block
- Contains statements accessible through iteration

### `slang::ast::ContinuousAssignSymbol`

Represents continuous assignment statements.

```cpp
void handle(const slang::ast::ContinuousAssignSymbol& node) {
    // Get the assignment expression
    auto& assignment = node.getAssignment();
    
    // Extract left and right hand sides
    // Process the assignment for SystemC conversion
}
```

**Key Methods:**
- `getAssignment()` - Returns the assignment expression

## Statement Types

### `slang::ast::VariableDeclStatement`

Variable declaration statements.

```cpp
void handle(const slang::ast::VariableDeclStatement& node) {
    // Process variable declaration
    // This is typically handled through VariableSymbol
}
```

### `slang::ast::ExpressionStatement`

Expression statements.

```cpp
void handle(const slang::ast::ExpressionStatement& node) {
    // Get the expression
    auto& expr = node.expr;
    
    // Process the expression
    std::string exprText = extractExpressionText(expr);
}
```

**Key Properties:**
- `expr` - The expression being evaluated

### `slang::ast::AssignmentExpression`

Assignment expressions.

```cpp
void handle(const slang::ast::AssignmentExpression& node) {
    // Extract left and right operands
    std::string lhs = extractExpressionText(node.left());
    std::string rhs = extractExpressionText(node.right());
    
    // Generate SystemC assignment
    // handle blocking vs non-blocking assignments
}
```

**Key Methods:**
- `left()` - Returns left-hand side expression
- `right()` - Returns right-hand side expression

## Visitor Usage Patterns

### Basic Visitor Implementation

```cpp
class MyVisitor : public slang::ast::ASTVisitor<MyVisitor, true, true> {
public:
    explicit MyVisitor(/* parameters */) { }
    
    // Handle specific node types
    void handle(const slang::ast::InstanceSymbol& node) {
        // Custom processing
        
        // Continue traversal
        visitDefault(node);
    }
    
    void handle(const slang::ast::PortSymbol& node) {
        // Process port
    }
    
    // Default handler for unhandled nodes
    template<typename T>
    void handle(const T& node) {
        // Default processing or logging
        visitDefault(node);
    }
};
```

### Visitor Invocation

```cpp
// Create visitor instance
MyVisitor visitor(/* parameters */);

// Visit a specific node
visitor.visit(*instanceSymbol);

// Visit entire design
auto& root = compilation.getRoot();
for (auto& instance : root.topInstances) {
    visitor.visit(*instance);
}
```

## Advanced Features

### Context Tracking

```cpp
class ContextAwareVisitor : public slang::ast::ASTVisitor<ContextAwareVisitor, true, true> {
private:
    std::string currentModule_;
    int indentLevel_ = 0;
    std::vector<std::string> contextStack_;

public:
    void handle(const slang::ast::InstanceSymbol& node) {
        // Save previous context
        std::string prevModule = currentModule_;
        
        // Update context
        currentModule_ = std::string(node.name);
        contextStack_.push_back(currentModule_);
        
        // Process with new context
        visitDefault(node);
        
        // Restore context
        contextStack_.pop_back();
        currentModule_ = prevModule;
    }
};
```

### Selective Traversal

```cpp
class SelectiveVisitor : public slang::ast::ASTVisitor<SelectiveVisitor, false, false> {
public:
    // Only handle specific node types
    void handle(const slang::ast::PortSymbol& node) {
        // Process ports only
    }
    
    // Skip other node types by not implementing handlers
};
```

## Integration with sv2sc

The sv2sc project uses the visitor pattern to:

1. **Parse Design Structure**: Identify modules, ports, and signals
2. **Analyze Signal Usage**: Track arithmetic vs logic operations
3. **Generate SystemC Code**: Convert SystemVerilog constructs
4. **Maintain Context**: Track current module and scope

```cpp
// sv2sc visitor usage
core::SVToSCVisitor visitor(systemcGenerator);
auto& root = compilation.getRoot();

for (auto& instance : root.topInstances) {
    if (instance->name == topModuleName) {
        visitor.visit(*instance);
        break;
    }
}
```