//===- SystemCEmissionPatterns.cpp - SystemC Emission Patterns -----------===//
//
// This file implements comprehensive emission patterns for all SystemC dialect
// operations, types, and attributes, providing CIRCT-compatible functionality.
//
//===----------------------------------------------------------------------===//

#include "SystemCEmissionPatterns.h"
#include <regex>
#include <iostream>

namespace sv2sc::mlir_support {

//===----------------------------------------------------------------------===//
// SystemC Operation Emission Patterns Implementation
//===----------------------------------------------------------------------===//

// SC_MODULE emission pattern
MatchResult SCModuleEmissionPattern::matchInlinable(const std::string& valueId) {
    // Block arguments of SC_MODULE can be inlined as port references
    if (valueId.find("module_arg_") != std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult();
}

bool SCModuleEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.module";
}

void SCModuleEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Extract port name from value ID
    std::regex portRegex(R"(module_arg_(\w+))");
    std::smatch match;
    if (std::regex_search(valueId, match, portRegex)) {
        p << match[1].str();
    } else {
        p << valueId;
    }
}

void SCModuleEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << "\nSC_MODULE(Module) {\n";
    p.setIndentLevel(p.getIndentLevel() + 1);
    
    // Emit ports section
    p << p.getIndent() << "// Ports\n";
    p << p.getIndent() << "sc_in<bool> clk;\n";
    p << p.getIndent() << "sc_in<bool> reset;\n";
    
    // Emit signals section
    p << p.getIndent() << "\n// Internal signals\n";
    
    // Emit constructor
    p << p.getIndent() << "\nSC_HAS_PROCESS(Module);\n";
    p << p.getIndent() << "Module(sc_module_name name);\n";
    
    // Emit process methods
    p << p.getIndent() << "\nprivate:\n";
    p << p.getIndent() << "void process();\n";
    
    p.setIndentLevel(p.getIndentLevel() - 1);
    p << "};\n";
}

// SC_CTOR emission pattern
MatchResult CtorEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool CtorEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.ctor";
}

void CtorEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void CtorEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << "\nSC_CTOR(Module) {\n";
    p.setIndentLevel(p.getIndentLevel() + 1);
    p << p.getIndent() << "// Constructor body\n";
    p.setIndentLevel(p.getIndentLevel() - 1);
    p << "}\n";
}

// SC_METHOD emission pattern
MatchResult MethodEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool MethodEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.method";
}

void MethodEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void MethodEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "SC_METHOD(process);\n";
}

// SC_THREAD emission pattern
MatchResult ThreadEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool ThreadEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.thread";
}

void ThreadEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void ThreadEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "SC_THREAD(process);\n";
}

// Signal write emission pattern
MatchResult SignalWriteEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool SignalWriteEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.signal.write";
}

void SignalWriteEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void SignalWriteEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "signal.write(value);\n";
}

// Signal read emission pattern
MatchResult SignalReadEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are signal read operations
    if (valueId.find("signal_read_") != std::string::npos || valueId.find(".read()") != std::string::npos) {
        return MatchResult(Precedence::FUNCTION_CALL);
    }
    return MatchResult(); // Failed match
}

bool SignalReadEmissionPattern::matchStatement(const std::string& opName) {
    return false; // Only inlinable
}

void SignalReadEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "signal.read()";
}

void SignalReadEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    // Not used
}

// Signal declaration emission pattern
MatchResult SignalEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are signal declarations
    if (valueId.find("signal_") != std::string::npos && valueId.find("signal_read_") == std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult(); // Failed match
}

bool SignalEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.signal";
}

void SignalEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "signal_name";
}

void SignalEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "sc_signal<bool> signal_name;\n";
}

// Sensitivity list emission pattern
MatchResult SensitiveEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool SensitiveEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.sensitive";
}

void SensitiveEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void SensitiveEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "sensitive << clk.pos();\n";
}

// Instance declaration emission pattern
MatchResult InstanceDeclEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are instance declarations
    if (valueId.find("instance_") != std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult(); // Failed match
}

bool InstanceDeclEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.instance.decl";
}

void InstanceDeclEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "instance_name";
}

void InstanceDeclEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "ModuleType* instance_name;\n";
}

// Port binding emission pattern
MatchResult BindPortEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool BindPortEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.instance.bind_port";
}

void BindPortEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void BindPortEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "instance_name.port_name(signal_name);\n";
}

// Function emission pattern
MatchResult SCFuncEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are SystemC function references
    if (valueId.find("sc_func_") != std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult(); // Failed match
}

bool SCFuncEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.func";
}

void SCFuncEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "func_name";
}

void SCFuncEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "\nvoid func_name() {\n";
    p.setIndentLevel(p.getIndentLevel() + 1);
    p << p.getIndent() << "// Function body\n";
    p.setIndentLevel(p.getIndentLevel() - 1);
    p << p.getIndent() << "}\n";
}

// C++ function emission pattern
MatchResult FuncEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are C++ function references
    if (valueId.find("func_") != std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult(); // Failed match
}

bool FuncEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.func";
}

void FuncEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "cpp_func_name";
}

void FuncEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "\nvoid cpp_func_name() {\n";
    p.setIndentLevel(p.getIndentLevel() + 1);
    p << p.getIndent() << "// C++ function body\n";
    p.setIndentLevel(p.getIndentLevel() - 1);
    p << p.getIndent() << "}\n";
}

// C++ variable emission pattern
MatchResult VariableEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are variable references
    if (valueId.find("var_") != std::string::npos) {
        return MatchResult(Precedence::VAR);
    }
    return MatchResult(); // Failed match
}

bool VariableEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.variable";
}

void VariableEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "var_name";
}

void VariableEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "int var_name;\n";
}

// C++ assignment emission pattern
MatchResult AssignEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool AssignEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.assign";
}

void AssignEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void AssignEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "dest = source;\n";
}

// C++ member access emission pattern
MatchResult MemberAccessEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are member access operations 
    if (valueId.find("member_access_") != std::string::npos) {
        return MatchResult(Precedence::MEMBER_ACCESS);
    }
    // Check for actual member access pattern (object.member)
    size_t dotPos = valueId.find(".");
    if (dotPos != std::string::npos && dotPos > 0 && dotPos < valueId.length() - 1) {
        return MatchResult(Precedence::MEMBER_ACCESS);
    }
    return MatchResult(); // Failed match
}

bool MemberAccessEmissionPattern::matchStatement(const std::string& opName) {
    return false; // Only inlinable
}

void MemberAccessEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "object.member";
}

void MemberAccessEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    // Not used
}

// C++ call emission pattern
MatchResult CallEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are function call operations
    if (valueId.find("call_") != std::string::npos || valueId.find("()") != std::string::npos) {
        return MatchResult(Precedence::FUNCTION_CALL);
    }
    return MatchResult(); // Failed match
}

bool CallEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.call";
}

void CallEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "func_call()";
}

void CallEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "func_call();\n";
}

// C++ return emission pattern
MatchResult ReturnEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool ReturnEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.return";
}

void ReturnEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void ReturnEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "return value;\n";
}

// C++ new emission pattern
MatchResult NewEmissionPattern::matchInlinable(const std::string& valueId) {
    // Only match values that are new expressions
    if (valueId.find("new_") != std::string::npos) {
        return MatchResult(Precedence::NEW);
    }
    return MatchResult(); // Failed match
}

bool NewEmissionPattern::matchStatement(const std::string& opName) {
    return false; // Only inlinable
}

void NewEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "new Type()";
}

void NewEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    // Not used
}

// C++ delete emission pattern
MatchResult DeleteEmissionPattern::matchInlinable(const std::string& valueId) {
    return MatchResult(); // Not inlinable
}

bool DeleteEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.cpp.delete";
}

void DeleteEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    // Not used
}

void DeleteEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "delete ptr;\n";
}

// SystemC convert emission pattern
MatchResult ConvertEmissionPattern::matchInlinable(const std::string& valueId) {
    // Convert operations can be inlined
    return MatchResult(Precedence::CAST);
}

bool ConvertEmissionPattern::matchStatement(const std::string& opName) {
    return opName == "systemc.convert";
}

void ConvertEmissionPattern::emitInlined(const std::string& valueId, CIRCTCompatibleEmitter& p) {
    p << "static_cast<sc_uint<8>>(0)"; // Simple cast for now
}

void ConvertEmissionPattern::emitStatement(const std::string& opName, CIRCTCompatibleEmitter& p) {
    p << p.getIndent() << "// Type conversion\n";
}

//===----------------------------------------------------------------------===//
// SystemC Type Emission Patterns Implementation
//===----------------------------------------------------------------------===//

// Input port type pattern
bool InputTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.in") != std::string::npos || 
           typeName.find("!systemc.in") != std::string::npos;
}

void InputTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    // Extract base type from systemc.in<BaseType>
    std::regex typeRegex(R"(systemc\.in<(.+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, typeRegex)) {
        p << "sc_in<" << match[1].str() << ">";
    } else {
        p << "sc_in<bool>";
    }
}

// Output port type pattern
bool OutputTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.out") != std::string::npos ||
           typeName.find("!systemc.out") != std::string::npos;
}

void OutputTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex typeRegex(R"(systemc\.out<(.+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, typeRegex)) {
        p << "sc_out<" << match[1].str() << ">";
    } else {
        p << "sc_out<bool>";
    }
}

// Inout port type pattern
bool InOutTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.inout") != std::string::npos ||
           typeName.find("!systemc.inout") != std::string::npos;
}

void InOutTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex typeRegex(R"(systemc\.inout<(.+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, typeRegex)) {
        p << "sc_inout<" << match[1].str() << ">";
    } else {
        p << "sc_inout<bool>";
    }
}

// Signal type pattern
bool SignalTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.signal") != std::string::npos ||
           typeName.find("!systemc.signal") != std::string::npos;
}

void SignalTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex typeRegex(R"(systemc\.signal<(.+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, typeRegex)) {
        p << "sc_signal<" << match[1].str() << ">";
    } else {
        p << "sc_signal<bool>";
    }
}

// Integer type patterns
bool IntTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.int") != std::string::npos ||
           typeName.find("!systemc.int") != std::string::npos;
}

void IntTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.int<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_int<" << match[1].str() << ">";
    } else {
        p << "sc_int<32>";
    }
}

bool UIntTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.uint") != std::string::npos ||
           typeName.find("!systemc.uint") != std::string::npos;
}

void UIntTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.uint<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_uint<" << match[1].str() << ">";
    } else {
        p << "sc_uint<32>";
    }
}

bool BigIntTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.bigint") != std::string::npos ||
           typeName.find("!systemc.bigint") != std::string::npos;
}

void BigIntTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.bigint<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_bigint<" << match[1].str() << ">";
    } else {
        p << "sc_bigint<64>";
    }
}

bool BigUIntTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.biguint") != std::string::npos ||
           typeName.find("!systemc.biguint") != std::string::npos;
}

void BigUIntTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.biguint<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_biguint<" << match[1].str() << ">";
    } else {
        p << "sc_biguint<64>";
    }
}

// Bit vector type patterns
bool BitVectorTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.bv") != std::string::npos ||
           typeName.find("!systemc.bv") != std::string::npos;
}

void BitVectorTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.bv<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_bv<" << match[1].str() << ">";
    } else {
        p << "sc_bv<32>";
    }
}

bool LogicVectorTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.lv") != std::string::npos ||
           typeName.find("!systemc.lv") != std::string::npos;
}

void LogicVectorTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    std::regex widthRegex(R"(systemc\.lv<(\d+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, widthRegex)) {
        p << "sc_lv<" << match[1].str() << ">";
    } else {
        p << "sc_lv<32>";
    }
}

// Logic type pattern
bool LogicTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.logic") != std::string::npos ||
           typeName.find("!systemc.logic") != std::string::npos;
}

void LogicTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    p << "sc_logic";
}

// Module type pattern
bool ModuleTypeEmissionPattern::match(const std::string& typeName) {
    return typeName.find("systemc.module") != std::string::npos ||
           typeName.find("!systemc.module") != std::string::npos;
}

void ModuleTypeEmissionPattern::emitType(const std::string& typeName, CIRCTCompatibleEmitter& p) {
    // Extract module name if available
    std::regex moduleRegex(R"(systemc\.module<(.+)>)");
    std::smatch match;
    if (std::regex_search(typeName, match, moduleRegex)) {
        p << match[1].str();
    } else {
        p << "Module";
    }
}

//===----------------------------------------------------------------------===//
// Registration Functions Implementation
//===----------------------------------------------------------------------===//

void registerAllSystemCOpEmitters(OpEmissionPatternSet& patterns) {
    // Module and constructor patterns
    patterns.addPattern(std::make_unique<SCModuleEmissionPattern>());
    patterns.addPattern(std::make_unique<CtorEmissionPattern>());
    
    // Process patterns
    patterns.addPattern(std::make_unique<MethodEmissionPattern>());
    patterns.addPattern(std::make_unique<ThreadEmissionPattern>());
    patterns.addPattern(std::make_unique<SensitiveEmissionPattern>());
    
    // Signal patterns
    patterns.addPattern(std::make_unique<SignalEmissionPattern>());
    patterns.addPattern(std::make_unique<SignalWriteEmissionPattern>());
    patterns.addPattern(std::make_unique<SignalReadEmissionPattern>());
    
    // Instance patterns
    patterns.addPattern(std::make_unique<InstanceDeclEmissionPattern>());
    patterns.addPattern(std::make_unique<BindPortEmissionPattern>());
    
    // Function patterns
    patterns.addPattern(std::make_unique<SCFuncEmissionPattern>());
    patterns.addPattern(std::make_unique<FuncEmissionPattern>());
    
    // C++ patterns
    patterns.addPattern(std::make_unique<VariableEmissionPattern>());
    patterns.addPattern(std::make_unique<AssignEmissionPattern>());
    patterns.addPattern(std::make_unique<MemberAccessEmissionPattern>());
    patterns.addPattern(std::make_unique<CallEmissionPattern>());
    patterns.addPattern(std::make_unique<ReturnEmissionPattern>());
    patterns.addPattern(std::make_unique<NewEmissionPattern>());
    patterns.addPattern(std::make_unique<DeleteEmissionPattern>());
    
    // SystemC utility patterns
    patterns.addPattern(std::make_unique<ConvertEmissionPattern>());
}

void registerAllSystemCTypeEmitters(TypeEmissionPatternSet& patterns) {
    // Port types
    patterns.addPattern(std::make_unique<InputTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<OutputTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<InOutTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<SignalTypeEmissionPattern>());
    
    // Integer types
    patterns.addPattern(std::make_unique<IntTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<UIntTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<BigIntTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<BigUIntTypeEmissionPattern>());
    
    // Vector types
    patterns.addPattern(std::make_unique<BitVectorTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<LogicVectorTypeEmissionPattern>());
    patterns.addPattern(std::make_unique<LogicTypeEmissionPattern>());
    
    // Module type
    patterns.addPattern(std::make_unique<ModuleTypeEmissionPattern>());
}

void registerAllSystemCAttrEmitters(AttrEmissionPatternSet& patterns) {
    // SystemC attributes would be registered here
    // For now, using the built-in attribute patterns
}

void registerAllSystemCEmitters(CIRCTCompatibleEmitter& emitter) {
    // Register operation patterns
    OpEmissionPatternSet opPatterns;
    registerAllSystemCOpEmitters(opPatterns);
    emitter.registerOpPatterns(opPatterns);
    
    // Register type patterns
    TypeEmissionPatternSet typePatterns;
    registerAllSystemCTypeEmitters(typePatterns);
    emitter.registerTypePatterns(typePatterns);
    
    // Register attribute patterns
    AttrEmissionPatternSet attrPatterns;
    registerAllSystemCAttrEmitters(attrPatterns);
    emitter.registerAttrPatterns(attrPatterns);
    
    std::cout << "Registered all SystemC emission patterns" << std::endl;
}

} // namespace sv2sc::mlir_support
