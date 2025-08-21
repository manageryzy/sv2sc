#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sv2sc::translator {

struct VCSArguments {
    std::vector<std::string> inputFiles;
    std::string topModule;
    std::string outputDir = "./output";
    std::vector<std::string> includePaths;
    std::vector<std::string> libraryPaths;
    std::vector<std::string> defines;
    std::vector<std::string> undefines;
    std::string timescale;
    bool enableDebug = false;
    bool enableVerbose = false;
    bool enableElaboration = true;
    bool enableSynthesis = false;
    bool generateTestbench = false;
    std::string clockSignal = "clk";
    std::string resetSignal = "reset";
};

class VCSArgsParser {
public:
    VCSArgsParser();
    
    bool parse(int argc, char* argv[]);
    const VCSArguments& getArguments() const { return args_; }
    
    void printHelp() const;
    void printVersion() const;

private:
    VCSArguments args_;
    
    void setupParser();
    bool validateArguments();
    void processDefines(const std::vector<std::string>& defines);
    std::string expandPath(const std::string& path);
};

} // namespace sv2sc::translator