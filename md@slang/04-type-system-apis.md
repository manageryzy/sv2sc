# Slang Type System APIs

This document describes the Slang type system APIs used for analyzing and converting SystemVerilog data types to SystemC equivalents.

## Overview

Slang provides a comprehensive type system that represents all SystemVerilog data types. The sv2sc project uses these APIs to determine appropriate SystemC type mappings and generate correct signal declarations.

## Core Type Classes

### `slang::ast::Type`

Base class for all types in the Slang type system.

```cpp
#include <slang/ast/types/AllTypes.h>

// Basic type analysis
void analyzeType(const slang::ast::Type& type) {
    // Check basic type properties
    bool isPacked = type.isPackedArray();
    bool isFourState = type.isFourState();
    bool isIntegral = type.isIntegral();
    bool isNumeric = type.isNumeric();
    
    // Get bit width for packed types
    if (isPacked) {
        size_t bitWidth = type.getBitWidth();
    }
    
    // Check specific type kind
    if (type.isKind(slang::ast::SymbolKind::FixedSizeUnpackedArrayType)) {
        // Handle array type
    }
}
```

**Key Methods:**
- `isPackedArray()` - Check if type is a packed array/vector
- `isFourState()` - Check if type supports 4-state logic (0,1,X,Z)
- `isIntegral()` - Check if type is an integral type
- `isNumeric()` - Check if type is numeric
- `getBitWidth()` - Get bit width for packed types
- `isKind(kind)` - Check specific type kind

### Type Mapping to SystemC

```cpp
// SystemVerilog to SystemC type mapping
SystemCDataType mapToSystemCType(const slang::ast::Type& type) {
    if (type.isPackedArray()) {
        if (type.isFourState()) {
            return SystemCDataType::SC_LV;  // sc_lv<N>
        } else {
            return SystemCDataType::SC_BV;  // sc_bv<N>
        }
    } else if (type.isFourState()) {
        return SystemCDataType::SC_LOGIC;   // sc_logic
    } else {
        return SystemCDataType::SC_BIT;     // sc_bit
    }
}

int getTypeWidth(const slang::ast::Type& type) {
    if (type.isPackedArray()) {
        return static_cast<int>(type.getBitWidth());
    }
    return 1;  // Single bit
}
```

## Specific Type Classes

### Array Types

#### `slang::ast::FixedSizeUnpackedArrayType`

Represents unpacked arrays.

```cpp
if (type.isKind(slang::ast::SymbolKind::FixedSizeUnpackedArrayType)) {
    auto& arrayType = type.as<slang::ast::FixedSizeUnpackedArrayType>();
    
    // Get element type
    auto& elementType = arrayType.elementType;
    
    // Get array dimensions
    auto& dimension = arrayType.dimension;
    
    // Process array bounds and generate SystemC array declaration
}
```

#### `slang::ast::PackedArrayType`

Represents packed arrays (vectors).

```cpp
if (type.isPackedArray()) {
    // This is a packed array/vector type
    size_t width = type.getBitWidth();
    bool fourState = type.isFourState();
    
    if (fourState) {
        // Use sc_lv<width>
    } else {
        // Use sc_bv<width>
    }
}
```

### Scalar Types

#### Logic Types

```cpp
// SystemVerilog logic types
if (type.isFourState()) {
    if (type.getBitWidth() == 1) {
        // sc_logic - single bit 4-state
    } else {
        // sc_lv<N> - multi-bit 4-state vector
    }
}
```

#### Bit Types

```cpp
// SystemVerilog bit types  
if (!type.isFourState()) {
    if (type.getBitWidth() == 1) {
        // sc_bit - single bit 2-state
    } else {
        // sc_bv<N> - multi-bit 2-state vector
    }
}
```

### Integer Types

#### `slang::ast::IntegerType`

Represents SystemVerilog integer types.

```cpp
// Handle integer, int, shortint, longint, byte
if (type.isKind(slang::ast::SymbolKind::IntegerType)) {
    auto& intType = type.as<slang::ast::IntegerType>();
    
    switch (intType.integerKind) {
        case slang::ast::IntegerKind::TwoState:
            // 2-state integer type
            break;
        case slang::ast::IntegerKind::FourState:
            // 4-state integer type  
            break;
    }
    
    // Get bit width
    size_t width = intType.getBitWidth();
    
    // Map to SystemC sc_int<N> or sc_uint<N>
}
```

### Enumerated Types

#### `slang::ast::EnumType`

Represents SystemVerilog enumerated types.

```cpp
if (type.isKind(slang::ast::SymbolKind::EnumType)) {
    auto& enumType = type.as<slang::ast::EnumType>();
    
    // Get base type
    auto& baseType = enumType.baseType;
    
    // Get enumeration values
    for (auto& member : enumType.members()) {
        std::string memberName = std::string(member.name);
        // Process enum member
    }
    
    // Map to SystemC enum or integer type
}
```

### Struct and Union Types

#### `slang::ast::PackedStructType`

Represents packed struct types.

```cpp
if (type.isKind(slang::ast::SymbolKind::PackedStructType)) {
    auto& structType = type.as<slang::ast::PackedStructType>();
    
    // Process struct members
    for (auto& member : structType.members) {
        std::string memberName = std::string(member->name);
        auto& memberType = member->getType();
        
        // Generate SystemC struct member
    }
}
```

#### `slang::ast::UnpackedStructType`

Represents unpacked struct types.

```cpp
if (type.isKind(slang::ast::SymbolKind::UnpackedStructType)) {
    auto& structType = type.as<slang::ast::UnpackedStructType>();
    
    // Similar processing to packed struct
    // But different SystemC mapping
}
```

### String Types

#### `slang::ast::StringType`

Represents SystemVerilog string type.

```cpp
if (type.isKind(slang::ast::SymbolKind::StringType)) {
    // Map to std::string in SystemC
    return "std::string";
}
```

## Type Analysis Utilities

### Signal Usage Analysis

```cpp
class TypeAnalyzer {
private:
    std::set<std::string> arithmeticSignals_;
    std::set<std::string> logicSignals_;

public:
    void analyzeSignalUsage(const std::string& signalName, 
                           const slang::ast::Expression& expr) {
        // Analyze how signal is used to determine optimal SystemC type
        
        if (isArithmeticContext(expr)) {
            markSignalArithmetic(signalName);
        } else {
            markSignalLogic(signalName);
        }
    }
    
    SystemCDataType getOptimalType(const std::string& signalName,
                                  const slang::ast::Type& originalType) {
        if (isArithmeticSignal(signalName) && originalType.isPackedArray()) {
            // Use sc_uint/sc_int for arithmetic operations
            return originalType.isSigned() ? 
                SystemCDataType::SC_INT : SystemCDataType::SC_UINT;
        }
        
        // Default mapping
        return mapToSystemCType(originalType);
    }
};
```

### Type Conversion Utilities

```cpp
class TypeConverter {
public:
    std::string generateTypeDeclaration(const slang::ast::Type& type, 
                                       const std::string& signalName) {
        if (type.isPackedArray()) {
            size_t width = type.getBitWidth();
            
            if (type.isFourState()) {
                return fmt::format("sc_lv<{}> {}", width, signalName);
            } else {
                return fmt::format("sc_bv<{}> {}", width, signalName);
            }
        } else if (type.isFourState()) {
            return fmt::format("sc_logic {}", signalName);
        } else {
            return fmt::format("sc_bit {}", signalName);
        }
    }
    
    std::string generatePortDeclaration(const slang::ast::Type& type,
                                       const std::string& portName,
                                       PortDirection direction) {
        std::string scType = getSystemCPortType(type);
        std::string dirStr = getDirectionString(direction);
        
        return fmt::format("{}<{}> {}", dirStr, scType, portName);
    }
};
```

## Advanced Type Features

### Parameter Types

```cpp
// Handle parameterized types
if (type.isKind(slang::ast::SymbolKind::TypeParameterType)) {
    auto& paramType = type.as<slang::ast::TypeParameterType>();
    
    // Get parameter information
    std::string paramName = std::string(paramType.name);
    
    // Handle parameterized SystemC types
    // May need template parameters in generated code
}
```

### Custom Type Definitions

```cpp
// Handle typedef declarations
if (type.isKind(slang::ast::SymbolKind::TypeAliasType)) {
    auto& aliasType = type.as<slang::ast::TypeAliasType>();
    
    // Get underlying type
    auto& targetType = aliasType.targetType.getType();
    
    // Generate SystemC typedef
    std::string aliasName = std::string(aliasType.name);
    std::string targetTypeName = generateTypeDeclaration(targetType, "");
    
    return fmt::format("typedef {} {}", targetTypeName, aliasName);
}
```

## SystemC Type Mapping Table

| SystemVerilog Type | SystemC Equivalent | Notes |
|-------------------|-------------------|-------|
| `logic` | `sc_logic` | 4-state single bit |
| `logic [N:0]` | `sc_lv<N+1>` | 4-state vector |
| `bit` | `sc_bit` | 2-state single bit |
| `bit [N:0]` | `sc_bv<N+1>` | 2-state vector |
| `int` | `sc_int<32>` | 32-bit signed |
| `integer` | `sc_int<32>` | 32-bit 4-state |
| `shortint` | `sc_int<16>` | 16-bit signed |
| `longint` | `sc_int<64>` | 64-bit signed |
| `byte` | `sc_int<8>` | 8-bit signed |
| `string` | `std::string` | Dynamic string |
| `real` | `double` | Floating point |
| `time` | `sc_time` | SystemC time |

## Usage Patterns

### Port Type Analysis

```cpp
void processPort(const slang::ast::PortSymbol& port) {
    auto& type = port.getType();
    
    // Analyze type characteristics
    bool isPacked = type.isPackedArray();
    bool isFourState = type.isFourState();
    int width = isPacked ? static_cast<int>(type.getBitWidth()) : 1;
    
    // Generate SystemC port declaration
    SystemCDataType scType = mapToSystemCType(type);
    std::string portDecl = generatePortDeclaration(type, port.name, port.direction);
}
```

### Signal Type Optimization

```cpp
void optimizeSignalTypes() {
    // Analyze signal usage patterns
    for (auto& [signalName, type] : declaredSignals) {
        if (isArithmeticSignal(signalName)) {
            // Convert sc_lv to sc_uint for arithmetic efficiency
            if (type == SystemCDataType::SC_LV) {
                type = SystemCDataType::SC_UINT;
            }
        }
    }
}
```

## Integration Notes

- Comprehensive SystemVerilog type system support
- Intelligent SystemC type mapping based on usage analysis
- Support for complex types including arrays, structs, and enums
- Performance optimization through arithmetic signal detection
- Proper handling of 2-state vs 4-state logic requirements