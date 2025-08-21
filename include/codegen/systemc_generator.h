#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>

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
    bool isArray = false;
    std::vector<int> arrayDimensions;
};

struct Signal {
    std::string name;
    SystemCDataType dataType;
    int width = 1;
    bool isArray = false;
    std::vector<int> arrayDimensions;
    std::string initialValue;
};

class SystemCCodeGenerator {
public:
    SystemCCodeGenerator();

    void beginModule(const std::string& moduleName);
    void endModule();
    
    void addPort(const Port& port);
    void addSignal(const Signal& signal);
    
    void addBlockingAssignment(const std::string& lhs, const std::string& rhs);
    void addNonBlockingAssignment(const std::string& lhs, const std::string& rhs);
    void addDelayedAssignment(const std::string& lhs, const std::string& rhs, const std::string& delay);
    
    void beginGenerateBlock(const std::string& label = "");
    void endGenerateBlock();
    
    void addComment(const std::string& comment);
    void addRawCode(const std::string& code);
    
    std::string generateHeader() const;
    std::string generateImplementation() const;
    
    bool writeToFile(const std::string& headerPath, const std::string& implPath) const;

private:
    std::string currentModule_;
    std::vector<Port> ports_;
    std::vector<Signal> signals_;
    std::stringstream headerCode_;
    std::stringstream implCode_;
    std::stringstream processCode_;
    int indentLevel_ = 0;
    
    std::string getIndent() const;
    std::string mapDataType(SystemCDataType type, int width = 1) const;
    std::string generatePortDeclaration(const Port& port) const;
    std::string generateSignalDeclaration(const Signal& signal) const;
    std::string generateConstructor() const;
    std::string generateProcessMethods() const;
};

} // namespace sv2sc::codegen