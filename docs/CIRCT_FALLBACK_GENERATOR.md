# CIRCT-Compatible Fallback SystemC Generator

## Overview

The sv2sc project now includes a comprehensive CIRCT-compatible SystemC generator that serves as the **default implementation** for SystemC code generation. This provides all the functionality of CIRCT's ExportSystemC with additional customization capabilities and improved reliability.

## Architecture

### Core Components

1. **CIRCTCompatibleEmitter** - Main emitter class that provides CIRCT-compatible SystemC emission
2. **SystemCEmissionPatterns** - Comprehensive emission patterns for all SystemC dialect operations
3. **Emission Pattern System** - Modular pattern-based emission similar to CIRCT's approach

### Key Features

- **Default SystemC Generator**: Primary implementation for all SystemC code generation
- **Complete SystemC Dialect Support**: All SystemC operations, types, and attributes
- **Precedence-Based Expression Emission**: Proper parenthesization for complex expressions
- **Split File Generation**: Generate separate header and implementation files
- **Unified File Generation**: Generate single combined files
- **Pattern-Based Architecture**: Extensible emission patterns
- **MLIR Integration**: Full integration with MLIR infrastructure when available
- **Superior Reliability**: More robust than external CIRCT dependencies

## Supported SystemC Features

### Operations

#### Module Operations
- `systemc.module` - SC_MODULE declaration
- `systemc.ctor` - SC_CTOR constructor
- `systemc.func` - SystemC member functions

#### Process Operations
- `systemc.method` - SC_METHOD process registration
- `systemc.thread` - SC_THREAD process registration
- `systemc.sensitive` - Sensitivity list specification

#### Signal Operations
- `systemc.signal` - Signal declarations
- `systemc.signal.read` - Signal read operations (inlinable)
- `systemc.signal.write` - Signal write operations

#### Instance Operations
- `systemc.instance.decl` - Module instance declarations
- `systemc.instance.bind_port` - Port binding operations

#### C++ Operations
- `systemc.cpp.func` - C++ function declarations
- `systemc.cpp.variable` - Variable declarations
- `systemc.cpp.assign` - Assignment operations
- `systemc.cpp.member_access` - Member access (inlinable)
- `systemc.cpp.call` - Function calls (inlinable/statement)
- `systemc.cpp.return` - Return statements
- `systemc.cpp.new` - Object allocation (inlinable)
- `systemc.cpp.delete` - Object deallocation

### Types

#### Port Types
- `systemc.in<T>` → `sc_in<T>`
- `systemc.out<T>` → `sc_out<T>`
- `systemc.inout<T>` → `sc_inout<T>`

#### Signal Types
- `systemc.signal<T>` → `sc_signal<T>`

#### Integer Types
- `systemc.int<N>` → `sc_int<N>`
- `systemc.uint<N>` → `sc_uint<N>`
- `systemc.bigint<N>` → `sc_bigint<N>`
- `systemc.biguint<N>` → `sc_biguint<N>`

#### Vector Types
- `systemc.bv<N>` → `sc_bv<N>`
- `systemc.lv<N>` → `sc_lv<N>`
- `systemc.logic` → `sc_logic`

#### Module Types
- `systemc.module<Name>` → `Name`

### Attributes

- Integer literals
- String literals
- Boolean values
- Type attributes

## Usage

### Basic Usage

```cpp
#include "mlir/systemc/CIRCTCompatibleEmitter.h"

// Create emitter
CIRCTCompatibleEmitter emitter;

// Emit split files (header + implementation)
auto result = emitter.emitSplit(module, outputDirectory);

// Emit unified file
auto result = emitter.emitUnified(module, outputPath);
```

### Integration with Existing SystemCEmitter

The existing `SystemCEmitter` automatically uses the CIRCT-compatible emitter as a fallback:

```cpp
SystemCEmitter emitter;
auto result = emitter.emitSplit(module, outputDirectory);
// Automatically tries CIRCT-compatible emitter first, falls back to simple emitter
```

### Custom Pattern Registration

```cpp
// Create custom emission patterns
class CustomOpPattern : public OpEmissionPatternBase {
    // Implementation
};

// Register patterns
OpEmissionPatternSet patterns;
patterns.addPattern(std::make_unique<CustomOpPattern>());
emitter.registerOpPatterns(patterns);
```

## Generated Code Structure

### Header File Structure

```cpp
// module.h
#ifndef MODULE_H
#define MODULE_H

#include <systemc.h>

SC_MODULE(Module) {
public:
    // Ports
    sc_in<bool> clk;
    sc_out<sc_uint<8>> data;
    
    // Internal signals
    sc_signal<bool> internal_sig;
    
    // Module instances
    SubModule* sub_inst;
    
    SC_HAS_PROCESS(Module);
    Module(sc_module_name name);
    
private:
    void process();
};

#endif // MODULE_H
```

### Implementation File Structure

```cpp
// module.cpp
#include "module.h"

Module::Module(sc_module_name name) : sc_module(name) {
    // Process registration
    SC_METHOD(process);
    sensitive << clk.pos();
    
    // Instance initialization
    sub_inst = new SubModule("sub_inst");
    sub_inst->port(signal);
}

void Module::process() {
    // Process implementation
    if (clk.read()) {
        data.write(internal_sig.read());
    }
}
```

## Expression Precedence

The emitter uses a comprehensive precedence system to ensure correct parenthesization:

```cpp
enum class Precedence {
    LIT = 0,           // Literals
    VAR = 0,           // Variables
    FUNCTION_CALL = 2, // Function calls
    MEMBER_ACCESS = 2, // Member access
    // ... (complete precedence hierarchy)
    COMMA = 17         // Comma operator
};
```

### Precedence-Based Emission

```cpp
// Automatic parenthesization based on precedence
auto inlineEmitter = emitter.getInlinable(valueId);
inlineEmitter.emitWithParensOnLowerPrecedence(Precedence::ADD);
```

## Error Handling

The emitter provides comprehensive error handling:

```cpp
// Check for errors
if (emitter.hasErrors()) {
    auto errors = emitter.getErrors();
    for (const auto& error : errors) {
        std::cerr << "Error: " << error << std::endl;
    }
}
```

## Extension Points

### Adding New Operations

1. Create emission pattern class:
```cpp
class NewOpPattern : public OpEmissionPatternBase {
public:
    NewOpPattern() : OpEmissionPatternBase("dialect.new_op") {}
    
    MatchResult matchInlinable(const std::string& valueId) override {
        return MatchResult(Precedence::FUNCTION_CALL);
    }
    
    bool matchStatement(const std::string& opName) override {
        return opName == "dialect.new_op";
    }
    
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override {
        p << "new_op_inline()";
    }
    
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override {
        p << p.getIndent() << "new_op_statement();\n";
    }
};
```

2. Register pattern:
```cpp
patterns.addPattern(std::make_unique<NewOpPattern>());
```

### Adding New Types

```cpp
class NewTypePattern : public TypeEmissionPatternBase {
public:
    NewTypePattern() : TypeEmissionPatternBase("dialect.new_type") {}
    
    bool match(const std::string& typeName) override {
        return typeName.find("dialect.new_type") != std::string::npos;
    }
    
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override {
        p << "NewType";
    }
};
```

## Performance Considerations

- **Pattern Matching**: O(n) pattern search, optimized for common cases
- **Memory Usage**: Minimal memory overhead with streaming output
- **Build Time**: Fast compilation with header-only templates where possible

## Comparison with CIRCT ExportSystemC

| Feature | CIRCT ExportSystemC | sv2sc Fallback | Notes |
|---------|-------------------|----------------|-------|
| SystemC Dialect Support | ✅ Complete | ✅ Complete | Full compatibility |
| Precedence Handling | ✅ Yes | ✅ Yes | Same precedence rules |
| Split File Generation | ✅ Yes | ✅ Yes | Header + implementation |
| Pattern-Based Architecture | ✅ Yes | ✅ Yes | Extensible patterns |
| MLIR Integration | ✅ Required | ⚠️ Optional | Works with/without MLIR |
| Custom Extensions | ⚠️ Limited | ✅ Full | Easy to extend |
| Build Dependencies | ❌ Heavy | ✅ Light | Minimal dependencies |

## Future Enhancements

1. **Advanced Optimizations**: Dead code elimination, constant folding
2. **Template Support**: Generic module generation
3. **Verification Integration**: Automatic testbench generation
4. **Performance Profiling**: Built-in performance analysis
5. **Custom Dialects**: Support for user-defined dialects

## Troubleshooting

### Common Issues

1. **Missing Patterns**: Add custom patterns for unsupported operations
2. **Precedence Issues**: Check expression precedence settings
3. **File Generation**: Verify output directory permissions
4. **MLIR Integration**: Ensure proper MLIR context setup

### Debug Mode

Enable debug output for detailed emission information:

```cpp
emitter.setDebugMode(true);
```

## Contributing

To add new emission patterns:

1. Create pattern class inheriting from appropriate base
2. Implement required virtual methods
3. Register pattern with emitter
4. Add tests for new functionality
5. Update documentation

## License

This fallback generator is part of the sv2sc project and follows the same license terms.
