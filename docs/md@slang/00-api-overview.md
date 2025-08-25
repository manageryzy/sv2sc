# Slang API Overview and Usage Guide

This document provides a comprehensive overview of the Slang SystemVerilog frontend APIs used in the sv2sc project for converting SystemVerilog designs to SystemC.

## What is Slang?

Slang is a modern SystemVerilog frontend written in C++ that provides:
- Complete SystemVerilog parser with full language support
- Comprehensive AST (Abstract Syntax Tree) representation
- Powerful visitor pattern for AST traversal
- Advanced type system modeling
- Robust preprocessing with macro support
- Excellent error reporting and diagnostics

**Repository**: https://github.com/MikePopoloski/slang

## API Documentation Structure

This documentation is organized into focused modules:

1. **[Core Compilation APIs](01-core-compilation-apis.md)** - Basic compilation and parsing
2. **[AST Visitor APIs](02-ast-visitor-apis.md)** - AST traversal and visitor patterns  
3. **[Expression APIs](03-expression-apis.md)** - Expression handling and operator processing
4. **[Type System APIs](04-type-system-apis.md)** - SystemVerilog type analysis and mapping
5. **[Preprocessing APIs](05-preprocessing-apis.md)** - Preprocessor directives and configuration

## Quick Start

### Basic Compilation

```cpp
#include <slang/ast/Compilation.h>
#include <slang/syntax/SyntaxTree.h>

// 1. Create compilation with options
slang::Bag options;
slang::ast::Compilation compilation(options);

// 2. Parse source files
auto syntaxTree = slang::syntax::SyntaxTree::fromText(content, filename, filename);
compilation.addSyntaxTree(syntaxTree);

// 3. Get design root
auto& root = compilation.getRoot();

// 4. Process top-level instances
for (auto& instance : root.topInstances) {
    processModule(*instance);
}
```

### AST Visitor Pattern

```cpp
#include "slang/ast/ASTVisitor.h"

class MyVisitor : public slang::ast::ASTVisitor<MyVisitor, true, true> {
public:
    void handle(const slang::ast::InstanceSymbol& node) {
        // Process module instance
        std::string moduleName = std::string(node.name);
        visitDefault(node);  // Continue traversal
    }
    
    void handle(const slang::ast::PortSymbol& node) {
        // Process module ports
        auto& type = node.getType();
        // Analyze port type and direction
    }
};

// Usage
MyVisitor visitor;
visitor.visit(*instanceSymbol);
```

## Key Concepts

### 1. Compilation Units

- **`slang::ast::Compilation`** - Main compilation context
- **`slang::syntax::SyntaxTree`** - Parsed source file representation
- **`slang::Bag`** - Configuration container for compilation options

### 2. AST Hierarchy

```
Compilation
├── Root Scope
    ├── Instance Symbols (modules)
        ├── Port Symbols
        ├── Variable Symbols  
        ├── Procedural Block Symbols
        ├── Continuous Assign Symbols
        └── Other Symbols
```

### 3. Visitor Pattern

The visitor pattern enables systematic AST traversal:

```cpp
// Define handlers for specific node types
void handle(const slang::ast::SpecificSymbol& node);

// Use visitDefault() to continue traversal
visitDefault(node);

// Template parameters control what gets visited
ASTVisitor<Derived, VisitStatements, VisitExpressions>
```

### 4. Expression Processing

Expressions are hierarchically structured:

```cpp
switch (expr.kind) {
    case slang::ast::ExpressionKind::IntegerLiteral:
        // Handle literals
        break;
    case slang::ast::ExpressionKind::BinaryOp:
        // Handle operators
        break;
    case slang::ast::ExpressionKind::NamedValue:
        // Handle variable references
        break;
}
```

### 5. Type System

SystemVerilog types map to SystemC equivalents:

```cpp
// Analyze type characteristics
bool isPacked = type.isPackedArray();
bool isFourState = type.isFourState();
size_t width = type.getBitWidth();

// Map to SystemC types
if (isPacked && isFourState) {
    // Use sc_lv<width>
} else if (isPacked) {
    // Use sc_bv<width>
} else if (isFourState) {
    // Use sc_logic
} else {
    // Use sc_bit
}
```

## Common Usage Patterns

### 1. VCS-Compatible Preprocessing

```cpp
slang::parsing::PreprocessorOptions preprocessorOpts;

// Handle +define+ options
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");
preprocessorOpts.predefines.emplace_back("WIDTH=32");

// Handle +incdir+ options  
preprocessorOpts.additionalIncludePaths.emplace_back("./includes");

// Apply to compilation
slang::Bag options;
options.set(preprocessorOpts);
slang::ast::Compilation compilation(options);
```

### 2. Error Handling

```cpp
// Check parsing errors
if (syntaxTree->diagnostics().size() > 0) {
    for (const auto& diag : syntaxTree->diagnostics()) {
        LOG_WARN("Parse diagnostic: {}", diag.formattedMessage);
    }
}

// Check compilation errors
for (const auto& diag : compilation.getAllDiagnostics()) {
    if (diag.isError()) {
        LOG_ERROR("Compilation error: {}", diag.formattedMessage);
    }
}
```

### 3. Context-Aware Processing

```cpp
class ContextVisitor : public slang::ast::ASTVisitor<ContextVisitor, true, true> {
private:
    std::string currentModule_;
    std::vector<std::string> scopeStack_;

public:
    void handle(const slang::ast::InstanceSymbol& node) {
        // Save and update context
        scopeStack_.push_back(currentModule_);
        currentModule_ = std::string(node.name);
        
        // Process with context
        visitDefault(node);
        
        // Restore context
        currentModule_ = scopeStack_.back();
        scopeStack_.pop_back();
    }
};
```

### 4. Type Analysis and Mapping

```cpp
SystemCDataType mapSystemVerilogToSystemC(const slang::ast::Type& type) {
    if (type.isPackedArray()) {
        size_t width = type.getBitWidth();
        
        if (type.isFourState()) {
            return {SystemCType::SC_LV, width};
        } else {
            return {SystemCType::SC_BV, width};
        }
    } else {
        if (type.isFourState()) {
            return {SystemCType::SC_LOGIC, 1};
        } else {
            return {SystemCType::SC_BIT, 1};
        }
    }
}
```

## Advanced Features

### 1. Custom Diagnostics

```cpp
// Add custom diagnostic for unsupported features
class CustomDiagnosticEngine {
public:
    void reportUnsupportedFeature(const slang::ast::Symbol& symbol) {
        LOG_WARN("Unsupported feature in {}: {}", 
                getCurrentModule(), symbol.name);
    }
};
```

### 2. Multi-Pass Analysis

```cpp
class TwoPassProcessor {
public:
    void firstPass(const slang::ast::Compilation& compilation) {
        // Collect information (signals, types, etc.)
        FirstPassVisitor visitor(*this);
        for (auto& instance : compilation.getRoot().topInstances) {
            visitor.visit(*instance);
        }
    }
    
    void secondPass(const slang::ast::Compilation& compilation) {
        // Generate code using collected information
        SecondPassVisitor visitor(*this);
        for (auto& instance : compilation.getRoot().topInstances) {
            visitor.visit(*instance);
        }
    }
};
```

### 3. Performance Optimization

```cpp
// Optimize for large designs
class OptimizedVisitor : public slang::ast::ASTVisitor<OptimizedVisitor, false, false> {
public:
    // Only visit specific node types to improve performance
    void handle(const slang::ast::InstanceSymbol& node) { /* ... */ }
    void handle(const slang::ast::PortSymbol& node) { /* ... */ }
    
    // Skip other node types for faster traversal
};
```

## Integration with sv2sc

The sv2sc project uses Slang APIs in the following workflow:

1. **Parse Command Line**: Extract VCS-compatible options
2. **Configure Preprocessing**: Set defines and include paths
3. **Create Compilation**: Build Slang compilation unit
4. **Parse Source Files**: Add SystemVerilog files to compilation
5. **Elaborate Design**: Let Slang elaborate the design hierarchy
6. **Visit AST**: Use custom visitor to traverse and analyze
7. **Generate SystemC**: Convert SystemVerilog constructs to SystemC
8. **Handle Errors**: Process diagnostics and report issues

```cpp
// sv2sc main workflow
bool SV2SC::processDesign() {
    // Configure preprocessing
    auto preprocessorOpts = createPreprocessorOptions();
    
    // Create compilation
    slang::Bag options;
    options.set(preprocessorOpts);
    slang::ast::Compilation compilation(options);
    
    // Add source files
    for (const auto& file : inputFiles) {
        auto syntaxTree = slang::syntax::SyntaxTree::fromText(
            readFile(file), file, file);
        compilation.addSyntaxTree(syntaxTree);
    }
    
    // Process with visitor
    SVToSCVisitor visitor(codeGenerator);
    auto& root = compilation.getRoot();
    
    for (auto& instance : root.topInstances) {
        if (instance->name == topModule) {
            visitor.visit(*instance);
            break;
        }
    }
    
    return true;
}
```

## Best Practices

1. **Error Handling**: Always check diagnostics after parsing and compilation
2. **Memory Management**: Slang manages AST memory automatically  
3. **Performance**: Use specific visitor templates for better performance
4. **Context Tracking**: Maintain context during AST traversal
5. **Type Analysis**: Analyze signal usage for optimal SystemC type selection
6. **Preprocessing**: Configure proper include paths and defines
7. **Diagnostics**: Provide meaningful error messages to users

## Resources

- **Slang Documentation**: https://sv-lang.com/
- **GitHub Repository**: https://github.com/MikePopoloski/slang
- **SystemVerilog Standard**: IEEE 1800-2017
- **SystemC Standard**: IEEE 1666-2011

## Related Documentation

- [SystemC Code Generation](../docs/CODEGEN.md)
- [VCS Compatibility](../docs/VCS_COMPATIBILITY.md)
- [Architecture Overview](../docs/ARCHITECTURE.md)
