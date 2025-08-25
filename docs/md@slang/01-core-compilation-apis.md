# Slang Core Compilation APIs

This document describes the core Slang APIs used for SystemVerilog compilation and parsing in the sv2sc project.

## Overview

Slang is a SystemVerilog frontend that provides comprehensive parsing, syntax analysis, and AST generation capabilities. The sv2sc project uses Slang to parse SystemVerilog designs and convert them to SystemC.

## Core Classes and APIs

### `slang::ast::Compilation`

The main class for managing SystemVerilog compilation units.

```cpp
#include <slang/ast/Compilation.h>

// Basic construction
slang::Bag options;
slang::ast::Compilation compilation(options);

// Add syntax trees
auto syntaxTree = slang::syntax::SyntaxTree::fromText(content, filename, filename);
compilation.addSyntaxTree(syntaxTree);

// Get the compilation root
auto& root = compilation.getRoot();

// Access top-level instances
for (auto& instance : root.topInstances) {
    // Process each top-level module instance
    std::string moduleName = instance->name;
}
```

**Key Methods:**
- `addSyntaxTree(syntaxTree)` - Add a parsed syntax tree to the compilation
- `getRoot()` - Get the root scope containing all top-level instances
- `elaborateDesign()` - Elaborate the design hierarchy

### `slang::syntax::SyntaxTree`

Represents a parsed SystemVerilog source file.

```cpp
#include <slang/syntax/SyntaxTree.h>

// Parse from string content
auto syntaxTree = slang::syntax::SyntaxTree::fromText(
    std::string_view(content),      // Source content
    std::string_view(filename),     // Source name for diagnostics
    std::string_view(filename)      // Buffer name
);

// Check for parsing errors
if (syntaxTree->diagnostics().size() > 0) {
    for (const auto& diag : syntaxTree->diagnostics()) {
        // Handle diagnostic messages
        auto location = diag.location.offset();
        // Process diagnostic
    }
}
```

**Key Methods:**
- `fromText()` - Static method to parse text content into syntax tree
- `diagnostics()` - Get parsing diagnostics and errors

### `slang::Bag`

Configuration container for compilation options.

```cpp
#include <slang/util/Bag.h>

slang::Bag options;

// Configure preprocessor options
slang::parsing::PreprocessorOptions preprocessorOpts;

// Add defines
preprocessorOpts.predefines.emplace_back("DEFINE_NAME=VALUE");

// Add include paths
preprocessorOpts.additionalIncludePaths.emplace_back("/path/to/includes");

// Set options in bag
options.set(preprocessorOpts);
```

**Key Methods:**
- `set(option)` - Set configuration option in the bag

## Preprocessor Configuration

### `slang::parsing::PreprocessorOptions`

Configuration for SystemVerilog preprocessing.

```cpp
#include <slang/parsing/ParserOptions.h>

slang::parsing::PreprocessorOptions preprocessorOpts;

// Define macros
for (const auto& [name, value] : defineMap) {
    std::string defineStr = name + "=" + value;
    preprocessorOpts.predefines.emplace_back(defineStr);
}

// Include directories
for (const auto& includePath : includePaths) {
    preprocessorOpts.additionalIncludePaths.emplace_back(includePath);
}
```

**Key Properties:**
- `predefines` - Vector of preprocessor defines
- `additionalIncludePaths` - Vector of include search paths

## Usage Example

```cpp
#include <slang/ast/Compilation.h>
#include <slang/syntax/SyntaxTree.h>
#include <slang/parsing/ParserOptions.h>

// Create compilation with options
slang::Bag options;
slang::parsing::PreprocessorOptions preprocessorOpts;

// Configure defines and includes
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");
preprocessorOpts.additionalIncludePaths.emplace_back("./includes");

options.set(preprocessorOpts);
slang::ast::Compilation compilation(options);

// Parse and add source files
std::string content = readFile("design.sv");
auto syntaxTree = slang::syntax::SyntaxTree::fromText(content, "design.sv", "design.sv");

compilation.addSyntaxTree(syntaxTree);

// Get root and process design
auto& root = compilation.getRoot();
for (auto& instance : root.topInstances) {
    // Process top-level modules
    processModule(*instance);
}
```

## Integration Notes

- Slang provides comprehensive SystemVerilog parsing with full language support
- Error handling through diagnostics system
- Configurable preprocessing with defines and include paths  
- AST-based design representation for analysis and transformation
- Thread-safe compilation units
