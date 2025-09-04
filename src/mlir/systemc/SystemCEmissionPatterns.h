//===- SystemCEmissionPatterns.h - SystemC Emission Patterns -------------===//
//
// This file defines comprehensive emission patterns for all SystemC dialect
// operations, types, and attributes, providing CIRCT-compatible functionality.
//
//===----------------------------------------------------------------------===//

#ifndef SV2SC_SYSTEMC_EMISSION_PATTERNS_H
#define SV2SC_SYSTEMC_EMISSION_PATTERNS_H

#include "CIRCTCompatibleEmitter.h"
#include <memory>
#include <string>

namespace sv2sc::mlir_support {

//===----------------------------------------------------------------------===//
// SystemC Operation Emission Patterns
//===----------------------------------------------------------------------===//

/// SC_MODULE emission pattern
class SCModuleEmissionPattern : public OpEmissionPatternBase {
public:
    SCModuleEmissionPattern() : OpEmissionPatternBase("systemc.module") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// SC_CTOR emission pattern
class CtorEmissionPattern : public OpEmissionPatternBase {
public:
    CtorEmissionPattern() : OpEmissionPatternBase("systemc.ctor") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// SC_METHOD emission pattern
class MethodEmissionPattern : public OpEmissionPatternBase {
public:
    MethodEmissionPattern() : OpEmissionPatternBase("systemc.method") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// SC_THREAD emission pattern
class ThreadEmissionPattern : public OpEmissionPatternBase {
public:
    ThreadEmissionPattern() : OpEmissionPatternBase("systemc.thread") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Signal write emission pattern
class SignalWriteEmissionPattern : public OpEmissionPatternBase {
public:
    SignalWriteEmissionPattern() : OpEmissionPatternBase("systemc.signal.write") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Signal read emission pattern
class SignalReadEmissionPattern : public OpEmissionPatternBase {
public:
    SignalReadEmissionPattern() : OpEmissionPatternBase("systemc.signal.read") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Signal declaration emission pattern
class SignalEmissionPattern : public OpEmissionPatternBase {
public:
    SignalEmissionPattern() : OpEmissionPatternBase("systemc.signal") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Sensitivity list emission pattern
class SensitiveEmissionPattern : public OpEmissionPatternBase {
public:
    SensitiveEmissionPattern() : OpEmissionPatternBase("systemc.sensitive") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Instance declaration emission pattern
class InstanceDeclEmissionPattern : public OpEmissionPatternBase {
public:
    InstanceDeclEmissionPattern() : OpEmissionPatternBase("systemc.instance.decl") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Port binding emission pattern
class BindPortEmissionPattern : public OpEmissionPatternBase {
public:
    BindPortEmissionPattern() : OpEmissionPatternBase("systemc.instance.bind_port") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// Function emission pattern
class SCFuncEmissionPattern : public OpEmissionPatternBase {
public:
    SCFuncEmissionPattern() : OpEmissionPatternBase("systemc.func") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ function emission pattern
class FuncEmissionPattern : public OpEmissionPatternBase {
public:
    FuncEmissionPattern() : OpEmissionPatternBase("systemc.cpp.func") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ variable emission pattern
class VariableEmissionPattern : public OpEmissionPatternBase {
public:
    VariableEmissionPattern() : OpEmissionPatternBase("systemc.cpp.variable") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ assignment emission pattern
class AssignEmissionPattern : public OpEmissionPatternBase {
public:
    AssignEmissionPattern() : OpEmissionPatternBase("systemc.cpp.assign") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ member access emission pattern
class MemberAccessEmissionPattern : public OpEmissionPatternBase {
public:
    MemberAccessEmissionPattern() : OpEmissionPatternBase("systemc.cpp.member_access") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ call emission pattern
class CallEmissionPattern : public OpEmissionPatternBase {
public:
    CallEmissionPattern() : OpEmissionPatternBase("systemc.cpp.call") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ return emission pattern
class ReturnEmissionPattern : public OpEmissionPatternBase {
public:
    ReturnEmissionPattern() : OpEmissionPatternBase("systemc.cpp.return") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ new emission pattern
class NewEmissionPattern : public OpEmissionPatternBase {
public:
    NewEmissionPattern() : OpEmissionPatternBase("systemc.cpp.new") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// C++ delete emission pattern
class DeleteEmissionPattern : public OpEmissionPatternBase {
public:
    DeleteEmissionPattern() : OpEmissionPatternBase("systemc.cpp.delete") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC conversion operation emission pattern
class ConvertEmissionPattern : public OpEmissionPatternBase {
public:
    ConvertEmissionPattern() : OpEmissionPatternBase("systemc.convert") {}
    
    MatchResult matchInlinable(const std::string& valueId) override;
    bool matchStatement(const std::string& opName) override;
    void emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) override;
    void emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) override;
};

//===----------------------------------------------------------------------===//
// SystemC Type Emission Patterns
//===----------------------------------------------------------------------===//

/// SystemC input port type pattern
class InputTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    InputTypeEmissionPattern() : TypeEmissionPatternBase("systemc.in") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC output port type pattern
class OutputTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    OutputTypeEmissionPattern() : TypeEmissionPatternBase("systemc.out") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC inout port type pattern
class InOutTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    InOutTypeEmissionPattern() : TypeEmissionPatternBase("systemc.inout") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC signal type pattern
class SignalTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    SignalTypeEmissionPattern() : TypeEmissionPatternBase("systemc.signal") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC integer types with static width
class IntTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    IntTypeEmissionPattern() : TypeEmissionPatternBase("systemc.int") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

class UIntTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    UIntTypeEmissionPattern() : TypeEmissionPatternBase("systemc.uint") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

class BigIntTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    BigIntTypeEmissionPattern() : TypeEmissionPatternBase("systemc.bigint") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

class BigUIntTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    BigUIntTypeEmissionPattern() : TypeEmissionPatternBase("systemc.biguint") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC bit vector types
class BitVectorTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    BitVectorTypeEmissionPattern() : TypeEmissionPatternBase("systemc.bv") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

class LogicVectorTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    LogicVectorTypeEmissionPattern() : TypeEmissionPatternBase("systemc.lv") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC logic type
class LogicTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    LogicTypeEmissionPattern() : TypeEmissionPatternBase("systemc.logic") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

/// SystemC module type
class ModuleTypeEmissionPattern : public TypeEmissionPatternBase {
public:
    ModuleTypeEmissionPattern() : TypeEmissionPatternBase("systemc.module") {}
    
    bool match(const std::string& typeName) override;
    void emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) override;
};

//===----------------------------------------------------------------------===//
// Registration Functions
//===----------------------------------------------------------------------===//

/// Register all SystemC operation emission patterns
void registerAllSystemCOpEmitters(OpEmissionPatternSet& patterns);

/// Register all SystemC type emission patterns
void registerAllSystemCTypeEmitters(TypeEmissionPatternSet& patterns);

/// Register all SystemC attribute emission patterns
void registerAllSystemCAttrEmitters(AttrEmissionPatternSet& patterns);

/// Register all SystemC emission patterns
void registerAllSystemCEmitters(CIRCTCompatibleEmitter& emitter);

} // namespace sv2sc::mlir_support

#endif // SV2SC_SYSTEMC_EMISSION_PATTERNS_H
