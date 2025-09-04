//===- CIRCTCompatibleEmitter.h - CIRCT-Compatible SystemC Emitter -------===//
//
// This file implements a comprehensive SystemC emitter that supports all
// CIRCT ExportSystemC features as a fallback when CIRCT is not available.
//
//===----------------------------------------------------------------------===//

#ifndef SV2SC_CIRCT_COMPATIBLE_EMITTER_H
#define SV2SC_CIRCT_COMPATIBLE_EMITTER_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <sstream>

// Forward declarations for MLIR types
namespace mlir {
    class Operation;
    class Value;
    class Type;
    class Attribute;
    class Region;
    class Block;
    class ModuleOp;
}

namespace sv2sc::mlir_support {

//===----------------------------------------------------------------------===//
// Precedence and Expression Handling
//===----------------------------------------------------------------------===//

/// Expression precedence levels for proper parenthesization
enum class Precedence {
    LIT = 0,
    VAR = 0,
    SCOPE_RESOLUTION = 1,
    POSTFIX_INC = 2,
    POSTFIX_DEC = 2,
    FUNCTIONAL_CAST = 2,
    FUNCTION_CALL = 2,
    SUBSCRIPT = 2,
    MEMBER_ACCESS = 2,
    PREFIX_INC = 3,
    PREFIX_DEC = 3,
    NOT = 3,
    CAST = 3,
    DEREFERENCE = 3,
    ADDRESS_OF = 3,
    SIZEOF = 3,
    NEW = 3,
    DELETE = 3,
    POINTER_TO_MEMBER = 4,
    MUL = 5,
    DIV = 5,
    MOD = 5,
    ADD = 6,
    SUB = 6,
    SHL = 7,
    SHR = 7,
    RELATIONAL = 9,
    EQUALITY = 10,
    BITWISE_AND = 11,
    BITWISE_XOR = 12,
    BITWISE_OR = 13,
    LOGICAL_AND = 14,
    LOGICAL_OR = 15,
    TERNARY = 16,
    THROW = 16,
    ASSIGN = 16,
    COMMA = 17
};

/// Result of pattern matching for inlinable expressions
class MatchResult {
public:
    MatchResult() : isFailure(true), precedence(Precedence::VAR) {}
    MatchResult(Precedence precedence) : isFailure(false), precedence(precedence) {}
    
    bool failed() const { return isFailure; }
    Precedence getPrecedence() const { return precedence; }
    
private:
    bool isFailure;
    Precedence precedence;
};

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

class CIRCTCompatibleEmitter;

//===----------------------------------------------------------------------===//
// Inline Expression Emitter
//===----------------------------------------------------------------------===//

/// Helper class for emitting inline expressions with proper precedence
class InlineEmitter {
public:
    InlineEmitter(std::function<void()> emitter, Precedence precedence, CIRCTCompatibleEmitter& printer)
        : precedence(precedence), emitter(std::move(emitter)), printer(printer) {}
    
    Precedence getPrecedence() const { return precedence; }
    void emit() const { emitter(); }
    void emitWithParensOnLowerPrecedence(Precedence prec, const std::string& lParen = "(", 
                                        const std::string& rParen = ")") const;
    
private:
    Precedence precedence;
    std::function<void()> emitter;
    CIRCTCompatibleEmitter& printer;
};

//===----------------------------------------------------------------------===//
// Emission Pattern Base Classes
//===----------------------------------------------------------------------===//

/// Base class for all emission patterns
class PatternBase {
public:
    explicit PatternBase(const void* rootValue) : rootValue(rootValue) {}
    virtual ~PatternBase() = default;
    
    const void* getRootValue() const { return rootValue; }
    
private:
    const void* rootValue;
};

/// Base class for operation emission patterns
class OpEmissionPatternBase : public PatternBase {
public:
    OpEmissionPatternBase(const std::string& operationName) 
        : PatternBase(operationName.c_str()) {}
    virtual ~OpEmissionPatternBase() = default;
    
    /// Check if this pattern can emit the value as an inlinable expression
    virtual MatchResult matchInlinable(const std::string& valueId) = 0;
    
    /// Check if this pattern can emit the operation as a statement
    virtual bool matchStatement(const std::string& opName) = 0;
    
    /// Emit the expression for the given value
    virtual void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) = 0;
    
    /// Emit zero or more statements for the given operation
    virtual void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) = 0;
};

/// Base class for type emission patterns
class TypeEmissionPatternBase : public PatternBase {
public:
    explicit TypeEmissionPatternBase(const std::string& typeName)
        : PatternBase(typeName.c_str()) {}
    virtual ~TypeEmissionPatternBase() = default;
    
    /// Check if this pattern can emit the given type
    virtual bool match(const std::string& typeName) = 0;
    
    /// Emit the given type
    virtual void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) = 0;
};

/// Base class for attribute emission patterns
class AttrEmissionPatternBase : public PatternBase {
public:
    explicit AttrEmissionPatternBase(const std::string& attrName)
        : PatternBase(attrName.c_str()) {}
    virtual ~AttrEmissionPatternBase() = default;
    
    /// Check if this pattern can emit the given attribute
    virtual bool match(const std::string& attrName) = 0;
    
    /// Emit the given attribute
    virtual void emitAttr(const std::string& attrName, CIRCTCompatibleEmitter& p) = 0;
};

//===----------------------------------------------------------------------===//
// Pattern Collections
//===----------------------------------------------------------------------===//

/// Collection of operation emission patterns
class OpEmissionPatternSet {
public:
    void addPattern(std::unique_ptr<OpEmissionPatternBase> pattern);
    std::vector<std::unique_ptr<OpEmissionPatternBase>>& getPatterns() { return patterns; }
    
private:
    std::vector<std::unique_ptr<OpEmissionPatternBase>> patterns;
};

/// Collection of type emission patterns
class TypeEmissionPatternSet {
public:
    void addPattern(std::unique_ptr<TypeEmissionPatternBase> pattern);
    std::vector<std::unique_ptr<TypeEmissionPatternBase>>& getPatterns() { return patterns; }
    
private:
    std::vector<std::unique_ptr<TypeEmissionPatternBase>> patterns;
};

/// Collection of attribute emission patterns
class AttrEmissionPatternSet {
public:
    void addPattern(std::unique_ptr<AttrEmissionPatternBase> pattern);
    std::vector<std::unique_ptr<AttrEmissionPatternBase>>& getPatterns() { return patterns; }
    
private:
    std::vector<std::unique_ptr<AttrEmissionPatternBase>> patterns;
};

//===----------------------------------------------------------------------===//
// Main CIRCT-Compatible Emitter
//===----------------------------------------------------------------------===//

/// Main emitter class that provides CIRCT-compatible SystemC emission
class CIRCTCompatibleEmitter {
public:
    struct EmitResult {
        bool success;
        std::string headerPath;
        std::string implPath;
        std::string errorMessage;
    };
    
    CIRCTCompatibleEmitter();
    ~CIRCTCompatibleEmitter() = default;
    
    /// Main emission methods
    EmitResult emitSplit(mlir::ModuleOp module, const std::string& outDir);
    EmitResult emitUnified(mlir::ModuleOp module, const std::string& outPath);
    
    /// Pattern registration
    void registerOpPatterns(OpEmissionPatternSet& patterns);
    void registerTypePatterns(TypeEmissionPatternSet& patterns);
    void registerAttrPatterns(AttrEmissionPatternSet& patterns);
    
    /// Output stream operations
    CIRCTCompatibleEmitter& operator<<(const std::string& str);
    CIRCTCompatibleEmitter& operator<<(int64_t num);
    
    /// Expression emission
    InlineEmitter getInlinable(const std::string& valueId);
    void emitType(const std::string& typeName);
    void emitAttr(const std::string& attrName);
    
    /// Statement emission
    void emitOp(const std::string& opName);
    void emitRegion(const std::string& regionName);
    
    /// Utility methods
    void setIndentLevel(int level) { indentLevel = level; }
    int getIndentLevel() const { return indentLevel; }
    std::string getIndent() const;
    
    /// Error handling
    void emitError(const std::string& message);
    bool hasErrors() const { return !errors.empty(); }
    std::vector<std::string> getErrors() const { return errors; }
    
    /// Write content to file (for testing)
    void writeFile(const std::string& path, const std::string& content);
    
private:
    // Pattern storage
    std::unique_ptr<OpEmissionPatternSet> opPatterns;
    std::unique_ptr<TypeEmissionPatternSet> typePatterns;
    std::unique_ptr<AttrEmissionPatternSet> attrPatterns;
    
    // Output management
    std::stringstream output;
    int indentLevel = 0;
    std::vector<std::string> errors;
    
    // Internal emission helpers
    void initializePatterns();
    std::string pathToMacroName(const std::string& path);
    
    // MLIR-specific helpers
    void emitMLIROperation(mlir::Operation* op);
    void emitMLIRType(mlir::Type type);
    void emitMLIRAttribute(mlir::Attribute attr);
    void emitMLIRRegion(mlir::Region& region);
    std::string getMLIRValueId(mlir::Value value);
    std::string getMLIROpName(mlir::Operation* op);
};

//===----------------------------------------------------------------------===//
// Built-in Emission Patterns
//===----------------------------------------------------------------------===//

/// Register all built-in emission patterns
void registerBuiltinOpEmitters(OpEmissionPatternSet& patterns);
void registerBuiltinTypeEmitters(TypeEmissionPatternSet& patterns);
void registerBuiltinAttrEmitters(AttrEmissionPatternSet& patterns);

/// Register SystemC-specific emission patterns
void registerSystemCOpEmitters(OpEmissionPatternSet& patterns);
void registerSystemCTypeEmitters(TypeEmissionPatternSet& patterns);

/// Register HW dialect emission patterns
void registerHWOpEmitters(OpEmissionPatternSet& patterns);

/// Register EmitC dialect emission patterns
void registerEmitCOpEmitters(OpEmissionPatternSet& patterns);
void registerEmitCTypeEmitters(TypeEmissionPatternSet& patterns);
void registerEmitCAttrEmitters(AttrEmissionPatternSet& patterns);

/// Register Arithmetic dialect emission patterns
void registerArithOpEmitters(OpEmissionPatternSet& patterns);

/// Register all emission patterns
void registerAllEmissionPatterns(CIRCTCompatibleEmitter& emitter);

} // namespace sv2sc::mlir_support

#endif // SV2SC_CIRCT_COMPATIBLE_EMITTER_H
