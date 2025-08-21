#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sv2sc {

struct TranslationOptions {
    std::vector<std::string> inputFiles;
    std::string outputDir = "./output";
    std::string topModule;
    bool generateTestbench = false;
    bool enableDebug = false;
    bool enableVerbose = false;
    std::vector<std::string> includePaths;
    std::vector<std::string> defines;
    std::vector<std::string> libraryPaths;
};

class SystemVerilogToSystemCTranslator {
public:
    explicit SystemVerilogToSystemCTranslator(const TranslationOptions& options);
    ~SystemVerilogToSystemCTranslator();

    bool translate();
    const std::vector<std::string>& getErrors() const;
    const std::vector<std::string>& getWarnings() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace sv2sc