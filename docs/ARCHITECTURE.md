# SystemVerilog to SystemC Translator - Architecture Guide

## Overview

The sv2sc translator is designed with a modular architecture that separates concerns across different layers:

1. **Input Processing**: VCS-compatible argument parsing and file handling
2. **Frontend**: SystemVerilog parsing using slang
3. **Translation Engine**: AST visitor pattern for SystemVerilog constructs
4. **Code Generation**: SystemC code generation with proper type mapping
5. **Output**: File generation and project organization

## Component Details

### 1. Translator Layer (`src/translator/`)

**VCS Args Parser** (`vcs_args_parser.cpp`)
- Handles command-line argument parsing compatible with VCS
- Supports both VCS-style (+incdir+, +define+) and standard (-I, -D) arguments
- Validates input files and creates output directories
- Converts arguments to internal TranslationOptions structure

Key Features:
- File existence validation
- Path expansion and normalization
- Preprocessor define processing
- Library path management

### 2. Core Translation (`src/core/`)

**AST Visitor** (`ast_visitor.cpp`)
- Implements the visitor pattern for slang syntax trees
- Traverses SystemVerilog AST nodes systematically
- Delegates code generation to SystemC generator
- Handles context tracking (current module, indent level)

Supported Node Types:
- `ModuleDeclarationSyntax`: Module definitions
- `PortDeclarationSyntax`: Input/output/inout ports
- `VariableDeclarationSyntax`: Internal signals and variables
- `BlockingAssignmentStatementSyntax`: Blocking assignments (=)
- `NonblockingAssignmentStatementSyntax`: Non-blocking assignments (<=)
- `GenerateBlockSyntax`: Generate constructs
- `DelayExpressionSyntax`: Timing delays
- `ArrayTypeSyntax`: Array declarations

### 3. Code Generation (`src/codegen/`)

**SystemC Generator** (`systemc_generator.cpp`)
- Generates SystemC header (.h) and implementation (.cpp) files
- Maps SystemVerilog types to appropriate SystemC types
- Handles port declarations, signal declarations, and process methods
- Manages indentation and code formatting

Type Mapping:
```cpp
SystemVerilog -> SystemC
logic         -> sc_logic
logic [N:0]   -> sc_lv<N+1>
bit [N:0]     -> sc_bv<N+1>
input         -> sc_in<>
output        -> sc_out<>
inout         -> sc_inout<>
```

### 4. Utilities (`src/utils/`)

**Logger** (`logger.cpp`)
- Structured logging using spdlog
- Multiple output targets (console, file)
- Configurable log levels
- Thread-safe logging operations

## Data Flow

```
Input Files → VCS Args Parser → Translation Options
     ↓
Slang Frontend → Syntax Tree → AST Visitor
     ↓
SystemC Generator → Output Files (*.h, *.cpp)
```

### Detailed Flow:

1. **Argument Parsing**:
   ```cpp
   VCSArgsParser parser;
   parser.parse(argc, argv);
   TranslationOptions options = parser.getArguments();
   ```

2. **File Processing**:
   ```cpp
   SourceManager sourceManager;
   SyntaxTree syntaxTree = SyntaxTree::fromText(content);
   ```

3. **AST Traversal**:
   ```cpp
   SystemCCodeGenerator generator;
   SVToSCVisitor visitor(generator);
   syntaxTree->root().visit(visitor);
   ```

4. **Code Generation**:
   ```cpp
   generator.writeToFile(headerPath, implPath);
   ```

## Design Patterns

### 1. Visitor Pattern
Used for AST traversal, allowing clean separation of tree structure from operations:
```cpp
class SVToSCVisitor : public slang::syntax::SyntaxVisitor<SVToSCVisitor> {
public:
    void handle(const ModuleDeclarationSyntax& node);
    void handle(const PortDeclarationSyntax& node);
    // ... other handlers
};
```

### 2. PIMPL Idiom
Used in main translator class to hide implementation details:
```cpp
class SystemVerilogToSystemCTranslator {
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
```

### 3. Strategy Pattern
Different code generation strategies for different SystemVerilog constructs:
- Port generation strategy
- Signal generation strategy  
- Assignment generation strategy

### 4. Builder Pattern
SystemC code built incrementally:
```cpp
generator.beginModule("counter");
generator.addPort(port);
generator.addSignal(signal);
generator.addBlockingAssignment(lhs, rhs);
generator.endModule();
```

## Error Handling

### Exception Safety
- RAII for resource management
- Smart pointers for automatic cleanup
- Exception propagation with context

### Error Reporting
- Structured error collection
- Warning vs error classification
- Line number and file context preservation

### Validation Layers
1. **Input Validation**: File existence, argument validity
2. **Parse Validation**: Syntax error detection
3. **Semantic Validation**: Type checking, scope validation
4. **Output Validation**: File writing permissions

## Extensibility

### Adding New SystemVerilog Constructs

1. **Add Handler to Visitor**:
   ```cpp
   void SVToSCVisitor::handle(const NewConstructSyntax& node) {
       // Process the construct
       // Delegate to code generator
   }
   ```

2. **Extend Code Generator**:
   ```cpp
   void SystemCCodeGenerator::addNewConstruct(const ConstructData& data) {
       // Generate appropriate SystemC code
   }
   ```

3. **Update Type Mapping**:
   ```cpp
   std::string mapDataType(SystemCDataType type, int width) const {
       // Add new type mappings
   }
   ```

### Adding New Command Line Options

1. **Extend VCSArguments Structure**:
   ```cpp
   struct VCSArguments {
       // ... existing fields
       bool newOption = false;
   };
   ```

2. **Add Parser Option**:
   ```cpp
   app.add_flag("--new-option", args_.newOption, "Description");
   ```

3. **Process in Translation**:
   ```cpp
   if (options.newOption) {
       // Handle the new option
   }
   ```

## Performance Considerations

### Memory Management
- Smart pointers for automatic cleanup
- Move semantics for large objects
- Efficient string handling with string_view

### Parsing Performance
- Single-pass AST traversal
- Lazy evaluation where possible
- Minimal string copying

### Code Generation
- Stream-based output for memory efficiency
- Template-based type mapping
- Incremental code building

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies for isolation
- Comprehensive edge case coverage

### Integration Tests
- End-to-end translation testing
- Real SystemVerilog file processing
- Output validation

### Example Tests
- Practical usage scenarios
- Performance benchmarking
- Compatibility verification

## Future Enhancements

### Planned Features
1. **SystemVerilog Interfaces**: Interface and modport support
2. **Classes and Objects**: OOP construct translation
3. **Assertions**: SVA to SystemC assertion mapping
4. **Coverage**: Functional coverage translation
5. **DPI Functions**: SystemVerilog DPI integration

### Architecture Improvements
1. **Plugin System**: Extensible translation rules
2. **Configuration Files**: Translation behavior customization
3. **Incremental Compilation**: Large project support
4. **Parallel Processing**: Multi-file concurrent translation