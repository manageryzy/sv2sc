#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>
#include <sstream>
#include "codegen/template_engine.h"

namespace sv2sc::codegen {

enum class SystemCDataType {
    SC_BIT,
    SC_LOGIC,
    SC_INT,
    SC_UINT,
    SC_BIGINT,
    SC_BIGUINT,
    SC_BV,
    SC_LV
};

enum class PortDirection {
    INPUT,
    OUTPUT,
    INOUT
};

struct Port {
    std::string name;
    PortDirection direction;
    SystemCDataType dataType;
    int width = 1;
    std::string widthExpression; // Parameter expression like "WIDTH" or "WIDTH+1"
    bool isArray = false;
    std::vector<int> arrayDimensions;
};

struct Signal {
    std::string name;
    SystemCDataType dataType;
    int width = 1;
    std::string widthExpression; // Parameter expression like "WIDTH" or "WIDTH+1"
    bool isArray = false;
    std::vector<int> arrayDimensions;
    std::string initialValue;
    bool preferArithmetic = false;  // Hint for arithmetic vs logic type selection
};

struct ModuleInstance {
    std::string instanceName;
    std::string moduleName;
    std::vector<std::pair<std::string, std::string>> portConnections; // port -> signal mapping
};

struct Parameter {
    std::string name;
    std::string value;
};

struct ModuleData {
    std::string name;
    std::vector<Port> ports;
    std::vector<Signal> signals;
    std::vector<Parameter> parameters;
    std::vector<ModuleInstance> instances; // Modules instantiated within this module
    std::set<std::string> dependencies;    // Module names this module depends on
    std::stringstream headerCode;
    std::stringstream implCode;
    std::stringstream processCode;          // Legacy single process code (for backward compatibility)
    std::stringstream combProcessCode;      // Combinational logic (always_comb, assign statements)
    std::stringstream seqProcessCode;       // Sequential logic (always_ff, clocked processes)
    bool isComplete = false;
    bool hasCombProcess = false;            // Track if we have combinational processes
    bool hasSeqProcess = false;             // Track if we have sequential processes
    
    // Signal usage tracking for precise sensitivity lists
    std::set<std::string> combSensitiveSignals;  // Signals read in combinational processes
    std::set<std::string> seqSensitiveSignals;   // Signals read in sequential processes (excluding clk/reset)
};

class SystemCCodeGenerator {
public:
    SystemCCodeGenerator();
    
    // Template-based generation methods
    void enableTemplateMode(bool enable = true);
    void setTemplateDirectory(const std::string& templateDir);
    std::string generateModuleUsingTemplate(const std::string& moduleName);

    void beginModule(const std::string& moduleName);
    void endModule();
    
    void addPort(const Port& port);
    void addSignal(const Signal& signal);
    void addParameter(const std::string& name, const std::string& value);
    void updateSignalType(const std::string& signalName, bool preferArithmetic);
    void addHeaderComment(const std::string& comment);
    
    // Module dependency and instantiation methods
    std::string addModuleInstance(const std::string& instanceName, const std::string& moduleName);
    void addModuleDependency(const std::string& dependencyModule);
    void addPortConnection(const std::string& instanceName, const std::string& portName, const std::string& signalName);
    
    void addBlockingAssignment(const std::string& lhs, const std::string& rhs);
    void addNonBlockingAssignment(const std::string& lhs, const std::string& rhs);
    void addDelayedAssignment(const std::string& lhs, const std::string& rhs, const std::string& delay);
    
    // Process-specific assignment methods
    void addCombinationalAssignment(const std::string& lhs, const std::string& rhs);
    void addSequentialAssignment(const std::string& lhs, const std::string& rhs);
    
    // Conditional logic support
    void addConditionalStart(const std::string& condition, bool isSequential = false);
    void addElseClause(bool isSequential = false);
    void addConditionalEnd(bool isSequential = false);
    
    // Signal usage tracking for sensitivity lists
    void addCombSensitiveSignal(const std::string& signalName);
    void addSeqSensitiveSignal(const std::string& signalName);
    
    void beginGenerateBlock(const std::string& label = "");
    void endGenerateBlock();
    
    void addComment(const std::string& comment);
    void addRawCode(const std::string& code);
    
    void beginConditional(const std::string& condition);
    void addElse();
    void endConditional();
    
    std::string generateHeader() const;
    std::string generateImplementation() const;
    
    // New methods for multi-module support
    std::string generateModuleHeader(const std::string& moduleName) const;
    std::string generateModuleImplementation(const std::string& moduleName) const;
    std::vector<std::string> getGeneratedModuleNames() const;
    
    bool writeToFile(const std::string& headerPath, const std::string& implPath) const;
    bool writeModuleFiles(const std::string& moduleName, const std::string& outputDir) const;
    bool writeAllModuleFiles(const std::string& outputDir) const;
    bool generateMainHeader(const std::string& outputDir, const std::string& mainHeaderName = "all_modules.h") const;

private:
    std::string currentModule_;
    std::unordered_map<std::string, std::unique_ptr<ModuleData>> modules_;
    int indentLevel_ = 0;
    
    // Template engine support
    TemplateEngine templateEngine_;
    bool templateMode_ = false;
    std::string templateDirectory_ = "templates";
    
    // Helper methods for multi-module support
    ModuleData* getCurrentModuleData();
    const ModuleData* getCurrentModuleData() const;
    std::string generateForwardDeclarations(const ModuleData& module) const;
    std::string generateIncludes(const ModuleData& module) const;
    std::string generateModuleInstances(const ModuleData& module) const;
    
    // Type conversion helpers
    std::string applyTypeConversion(const std::string& lhs, const std::string& rhs) const;
    std::string convertIntegerToScLogic(const std::string& value) const;
    std::string getSignalType(const std::string& signalName) const;
    
    std::string getIndent() const;
    std::string mapDataType(SystemCDataType type, int width = 1) const;
    std::string mapDataType(SystemCDataType type, int width, const std::string& widthExpression) const;
    std::string generatePortDeclaration(const Port& port) const;
    std::string generateSignalDeclaration(const Signal& signal) const;
    void regenerateSignalDeclarations();
    std::string generateConstructor() const;
    std::string generateProcessMethods() const;
    std::string generateConstructorForModule(const ModuleData& module) const;
    std::string generateProcessMethodsForModule(const ModuleData& module) const;
    bool isSkippingModule() const;
};

} // namespace sv2sc::codegen