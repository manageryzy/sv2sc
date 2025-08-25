# Slang API Documentation for sv2sc

This directory contains comprehensive documentation for the Slang SystemVerilog frontend APIs used in the sv2sc project.

## About Slang

Slang is a high-performance SystemVerilog frontend that provides complete parsing, elaboration, and analysis capabilities for SystemVerilog designs. It is developed by Mike Popoloski and is used as the parsing engine in the sv2sc SystemVerilog-to-SystemC translator.

- **Website**: https://sv-lang.com/
- **GitHub**: https://github.com/MikePopoloski/slang
- **License**: MIT

## Documentation Structure

| File | Description |
|------|-------------|
| **[00-api-overview.md](00-api-overview.md)** | 🏠 **Start Here** - Comprehensive overview and quick start guide |
| **[01-core-compilation-apis.md](01-core-compilation-apis.md)** | Core compilation, parsing, and syntax tree APIs |
| **[02-ast-visitor-apis.md](02-ast-visitor-apis.md)** | AST visitor patterns and symbol processing |
| **[03-expression-apis.md](03-expression-apis.md)** | Expression handling, operators, and literals |
| **[04-type-system-apis.md](04-type-system-apis.md)** | SystemVerilog type system and SystemC mapping |
| **[05-preprocessing-apis.md](05-preprocessing-apis.md)** | Preprocessor directives and configuration |

## Quick Navigation

### For New Users
1. Start with **[API Overview](00-api-overview.md)** for basic concepts
2. Review **[Core Compilation APIs](01-core-compilation-apis.md)** for setup
3. Explore **[AST Visitor APIs](02-ast-visitor-apis.md)** for traversal patterns

### For Implementation
1. **[Expression APIs](03-expression-apis.md)** - Converting SystemVerilog expressions
2. **[Type System APIs](04-type-system-apis.md)** - Type analysis and mapping
3. **[Preprocessing APIs](05-preprocessing-apis.md)** - VCS compatibility

### By Use Case

#### Basic SystemVerilog Parsing
- [Core Compilation APIs](01-core-compilation-apis.md#slangastcompilation)
- [Preprocessing APIs](05-preprocessing-apis.md#slangparsingpreprocessoroptions)

#### AST Analysis and Traversal  
- [AST Visitor APIs](02-ast-visitor-apis.md#slangastastvisitertrue-true)
- [Expression APIs](03-expression-apis.md#slangastexpression)

#### SystemC Code Generation
- [Type System APIs](04-type-system-apis.md#type-mapping-to-systemc)
- [Expression APIs](03-expression-apis.md#systemc-signal-handling)

#### VCS Compatibility
- [Preprocessing APIs](05-preprocessing-apis.md#vcs-compatible-define-processing)
- [Core Compilation APIs](01-core-compilation-apis.md#slangbag)

## Key API Categories

### 🔧 Core Classes
- `slang::ast::Compilation` - Main compilation unit
- `slang::syntax::SyntaxTree` - Parsed source representation  
- `slang::Bag` - Configuration container
- `slang::parsing::PreprocessorOptions` - Preprocessing configuration

### 🚶‍♂️ Visitor Pattern
- `slang::ast::ASTVisitor<T, VisitStatements, VisitExpressions>` - Base visitor
- Symbol handlers: `InstanceSymbol`, `PortSymbol`, `VariableSymbol`, etc.
- Context tracking and selective traversal

### 📝 Expression System
- `slang::ast::Expression` - Base expression class
- Literals: `IntegerLiteral`, `RealLiteral`, `StringLiteral`
- Operators: `UnaryExpression`, `BinaryExpression`, `ConditionalExpression`
- Complex: `ConcatenationExpression`, `CallExpression`

### 🏗️ Type System
- `slang::ast::Type` - Base type class
- Array types: `FixedSizeUnpackedArrayType`, `PackedArrayType`
- Scalar types: Logic, bit, integer variants
- Complex types: Structs, enums, strings

## Integration Examples

### Basic Compilation Setup
```cpp
// Configure preprocessing
slang::parsing::PreprocessorOptions preprocessorOpts;
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");
preprocessorOpts.additionalIncludePaths.emplace_back("./includes");

// Create compilation
slang::Bag options;
options.set(preprocessorOpts);
slang::ast::Compilation compilation(options);

// Parse and add files
auto syntaxTree = slang::syntax::SyntaxTree::fromText(content, filename, filename);
compilation.addSyntaxTree(syntaxTree);
```

### AST Processing
```cpp
class MyVisitor : public slang::ast::ASTVisitor<MyVisitor, true, true> {
public:
    void handle(const slang::ast::InstanceSymbol& node) {
        std::string moduleName = std::string(node.name);
        visitDefault(node);  // Continue traversal
    }
};

MyVisitor visitor;
for (auto& instance : compilation.getRoot().topInstances) {
    visitor.visit(*instance);
}
```

## sv2sc Integration

The sv2sc project leverages Slang for:

1. **Parsing** - Complete SystemVerilog language support
2. **Elaboration** - Design hierarchy analysis  
3. **Type Analysis** - SystemVerilog to SystemC type mapping
4. **Expression Processing** - Converting operators and literals
5. **Preprocessing** - VCS-compatible macro and include handling
6. **Error Reporting** - Comprehensive diagnostics

## Development Workflow

1. **Setup**: Configure compilation with preprocessor options
2. **Parse**: Add SystemVerilog source files to compilation
3. **Elaborate**: Let Slang build the design hierarchy
4. **Visit**: Use custom AST visitor to traverse design
5. **Analyze**: Extract ports, signals, types, and expressions
6. **Generate**: Convert to equivalent SystemC constructs
7. **Validate**: Check diagnostics and handle errors

## Contributing

When extending the sv2sc project with new Slang API usage:

1. **Document New APIs**: Add to appropriate documentation file
2. **Provide Examples**: Include practical usage examples
3. **Update Overview**: Reference new APIs in the overview
4. **Test Integration**: Ensure proper error handling
5. **Consider Performance**: Use appropriate visitor templates

## Support and Resources

- **Slang Issues**: https://github.com/MikePopoloski/slang/issues
- **SystemVerilog Standard**: IEEE 1800-2017  
- **sv2sc Project**: See main project documentation
- **SystemC Standard**: IEEE 1666-2011

---

**Last Updated**: Generated for sv2sc project integration
**Slang Version**: Compatible with latest stable release
**Audience**: sv2sc developers and SystemVerilog-to-SystemC conversion engineers
