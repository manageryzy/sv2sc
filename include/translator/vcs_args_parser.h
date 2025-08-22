#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sv2sc::translator {

struct VCSArguments {
    // Input files and top module
    std::vector<std::string> inputFiles;
    std::string topModule;
    
    // Output control
    std::string outputDir = "./output";
    std::string outputName = "simv";
    bool runAfterCompile = false;
    bool enable64Bit = false;
    
    // File and library management
    std::vector<std::string> includePaths;           // -I, +incdir+
    std::vector<std::string> libraryFiles;           // -v
    std::vector<std::string> libraryPaths;           // -y
    std::vector<std::string> fileListFiles;          // -f
    std::vector<std::string> libraryExtensions;      // +libext+
    bool libraryOrder = false;                       // +liborder
    bool libraryRescan = false;                      // +librescan
    bool libraryVerbose = false;                     // +libverbose
    
    // Preprocessor defines and undefines
    std::unordered_map<std::string, std::string> defineMap;  // +define+, -D
    std::vector<std::string> defines;                // Legacy format
    std::vector<std::string> undefines;              // +undefine+, -U
    
    // Language standard selection
    bool enableSystemVerilog = false;               // -sverilog
    std::string systemVerilogExt;                   // +systemverilogext+
    std::string verilog2001Ext;                     // +verilog2001ext+
    std::string verilog1995Ext;                     // +verilog1995ext+
    bool disableVerilog2001 = false;                // -v95
    bool extIncludeVersion = false;                 // -extinclude
    
    // Timing and other options
    std::string timescale;
    bool enableDebug = false;
    bool enableVerbose = false;
    bool enableElaboration = true;
    bool enableSynthesis = false;
    bool generateTestbench = false;
    std::string clockSignal = "clk";
    std::string resetSignal = "reset";
    
    // Phase 2: Advanced File Management
    std::vector<std::string> advancedFileListFiles;  // -F (advanced file list)
    std::vector<std::string> namedFileListFiles;     // -file (named file list)
    std::vector<std::string> liblistFiles;           // +liblist (library list)
    std::string libmapFile;                          // -libmap (library mapping)
    std::string workDirectory;                       // -work (work directory)
    
    // Phase 2: Compilation Control
    bool enableMCG = false;                          // -mcg (multi-cycle generation)
    int optimizationLevel = 0;                       // -O0, -O1, -O2, -O3
    bool enableDiskOpt = false;                      // -diskopt (disk optimization)
    bool disableIncrComp = false;                    // -noincrcomp (disable incremental)
    std::string makeDirectory;                       // -Mdirectory (make directory)
    bool enableMakeUpdate = false;                   // -Mupdate (make update)
    std::string makepFile;                           // -Mmakep (make dependencies)
    
    // Phase 2: Debug Options
    std::string debugAccess;                         // -debug_access (debug access control)
    bool enableKDB = false;                          // -kdb (kernel debug)
    bool enableGUI = false;                          // -gui (graphical interface)
    std::string gfile;                               // -gfile (GUI file)
    
    // Phase 2: Runtime Control
    std::vector<std::string> simArgs;                // +simargs (simulation arguments)
    std::string saveFile;                            // -save (save simulation)
    bool quietMode = false;                          // -q (quiet mode)
    
    // Phase 2: Error and Warning Control
    std::vector<std::string> ignorePatterns;         // -ignore (ignore patterns)
    bool warnAll = false;                            // +warn=all
    std::vector<std::string> warnDisable;            // +warn=no* (disable warnings)
    std::vector<std::string> errorPatterns;          // -error=* (error patterns)
    
    // Phase 3: SystemVerilog Assertions
    bool assertDisable = false;                      // -assert disable
    bool assertEnableDiag = false;                   // -assert enable_diag
    std::string assertHierFile;                      // -assert hier=<filename>
    bool assertFilterPast = false;                   // -assert filter_past
    bool assertOffendingValues = false;              // -assert offending_values
    bool assertDumpOff = false;                      // -assert dumpoff
    bool assertVpiSeqBeginTime = false;              // -assert vpiSeqBeginTime
    bool assertVpiSeqFail = false;                   // -assert vpiSeqFail
    bool assertAsyncDisable = false;                 // -assert async_disable
    bool assertDisableCover = false;                 // -assert disable_cover
    bool assertDisableAssert = false;                // -assert disable_assert
    bool assertEnableHier = false;                   // -assert enable_hier
    bool assertDisableRepOpt = false;                // -assert disable_rep_opt
    int assertMaxFail = 0;                           // -assert maxfail=<N>
    bool assertFinishMaxFail = false;                // -assert finish_maxfail
    
    // Phase 3: Timing and SDF Annotation
    std::vector<std::string> sdfFiles;               // -sdf annotations
    bool maxDelays = false;                          // +maxdelays
    bool minDelays = false;                          // +mindelays
    bool typDelays = false;                          // +typdelays
    bool allMtm = false;                             // +allmtm
    bool delayModePath = false;                      // +delay_mode_path
    bool delayModeZero = false;                      // +delay_mode_zero
    bool delayModeUnit = false;                      // +delay_mode_unit
    bool delayModeDistributed = false;               // +delay_mode_distributed
    bool transportPathDelays = false;                // +transport_path_delays
    bool transportIntDelays = false;                 // +transport_int_delays
    bool sdfRetain = false;                          // -sdfretain
    bool pathPulse = false;                          // +pathpulse
    bool noSpecify = false;                          // +nospecify
    bool noTimingCheck = false;                      // +notimingcheck
    
    // Phase 3: Code Coverage
    std::vector<std::string> coverageMetrics;        // -cm <metric>
    bool coverageBranch = false;                     // -cm branch
    bool coverageCond = false;                       // -cm cond
    bool coverageFsm = false;                        // -cm fsm
    bool coverageToggle = false;                     // -cm tgl
    bool coverageLine = false;                       // -cm line
    bool coverageAssert = false;                     // -cm assert
    std::string coverageDir;                         // -cm_dir <directory>
    std::string coverageName;                        // -cm_name <name>
    std::string coverageHierFile;                    // -cm_hier <filename>
    std::vector<std::string> coverageLibs;           // -cm_libs <library>
    std::string coverageExcludeFile;                 // -cm_exclude <filename>
    bool coverageCondBasic = false;                  // -cm_cond basic
    bool coverageReport = false;                     // -cm_report
    bool coverageStats = false;                      // -cm_stats
    
    // Phase 3: Advanced Debug Features
    bool kdbOnly = false;                            // -kdb=only
    bool debugRegion = false;                        // -debug_region
    bool fsdbFormat = false;                         // +fsdb+
    std::string fgpMode;                             // -fgp, -fgp=single, -fgp=multi
    bool enableFrames = false;                       // -frames
    bool enableGvalue = false;                       // -gvalue
    
    // Phase 4: Power Analysis
    bool enablePower = false;                        // -power
    std::string powerFormat;                         // -power=UPF, -power=dump, etc.
    std::string upfFile;                             // -upf
    std::string powerTopModule;                      // -power_top
    std::vector<std::string> powerOptions;           // Various -power= options
    
    // Phase 4: OpenVera/NTB 
    bool enableNTB = false;                          // -ntb
    std::vector<std::string> ntbDefines;             // -ntb_define
    std::vector<std::string> ntbFileExt;             // -ntb_filext
    std::vector<std::string> ntbOpts;                // -ntb_opts
    bool ntbShellOnly = false;                       // -ntb_shell_only
    std::string ntbShellFilename;                    // -ntb_sfname
    std::string ntbShellModule;                      // -ntb_sname
    std::string ntbShellPath;                        // -ntb_spath
    std::vector<std::string> ntbVipExt;              // -ntb_vipext
    bool ntbNoShell = false;                         // -ntb_noshell
    bool ntbEnableCoverage = false;                  // +ntb_enable_coverage
    bool ntbFuncEnable = false;                      // +ntb_func_enable
    bool ntbSolveControl = false;                    // +ntb_solve_control
    
    // Phase 4: Advanced Optimization
    bool enableRad = false;                          // +rad
    std::string optConfigFile;                       // +optconfigfile+
    std::vector<std::string> hsOptOptions;           // -hsopt=
    bool plusOptimization = false;                   // +plus-optimization
    bool partialCompilation = false;                 // -partcomp
    bool fastPartialCompilation = false;             // -fastpartcomp
    std::vector<std::string> sparseOptions;          // -sparse+
    
    // Phase 4: Distributed Simulation
    bool enableDistSim = false;                      // -distsim
    std::string distSimMode;                         // -distsim=setup/run/collect/etc.
    
    // Phase 4: SystemC Integration
    bool enableSysC = false;                         // -sysc
    std::string sysCMode;                            // -sysc=show/incr/adjust/etc.
    std::vector<std::string> vcAbstract;             // +vc+[abstract]
    bool sysCRunConfigure = false;                   // -systemcrunconfigure
    
    // Phase 4: Verification Methodology
    bool enableUVM = false;                          // +UVM
    std::string uvmVersion;                          // -ntb_opts uvm-1.1/1.2/ieee/etc.
    bool uvmVcsRecord = false;                       // +define+UVM_VCS_RECORD
    bool uvmPhaseRecord = false;                     // +UVM_PHASE_RECORD
    bool enableVera = false;                         // -vera
    bool enablePSL = false;                          // -psl
    std::string ovaFile;                             // -ova_file
    bool assertCount = false;                        // +assert_count
    
    // Phase 4: Advanced File Handling
    bool enableProtect = false;                      // +protect
    bool protect123 = false;                         // -protect123
    bool autoProtect = false;                        // +autoprotect
    bool auto2Protect = false;                       // -auto2protect
    bool auto3Protect = false;                       // -auto3protect
    bool putProtect = false;                         // +putprotect/-putprotect
    bool ipProtect = false;                          // +ipprotect
    std::string ipOutFile;                           // -ipout
    std::vector<std::string> ipOptOptions;           // -ipopt=
    bool enableEncrypt = false;                      // +encrypt
    bool enableDecrypt = false;                      // +decrypt
};

// Plus argument parsing helper
struct PlusArg {
    std::string command;                    // "incdir", "define", "libext"
    std::vector<std::string> values;        // parsed values  
    bool hasTerminator = false;             // ends with '+'
};

class VCSArgsParser {
public:
    VCSArgsParser();
    
    bool parse(int argc, char* argv[]);
    bool parse(int argc, const char* argv[]); // const version for testing
    const VCSArguments& getArguments() const { return args_; }
    
    void printHelp() const;
    void printVersion() const;

private:
    VCSArguments args_;
    
    void setupParser();
    bool validateArguments();
    void processDefines(const std::vector<std::string>& defines);
    std::string expandPath(const std::string& path);
    
    // Plus argument parsing
    PlusArg parsePlusArg(const std::string& arg);
    void handlePlusIncdir(const PlusArg& plusArg);
    void handlePlusDefine(const PlusArg& plusArg);
    void handlePlusUndefine(const PlusArg& plusArg);
    void handlePlusLibext(const PlusArg& plusArg);
    void handlePlusSystemVerilogExt(const PlusArg& plusArg);
    void handlePlusVerilog2001Ext(const PlusArg& plusArg);
    void handlePlusVerilog1995Ext(const PlusArg& plusArg);
    
    // Phase 2: Plus argument handlers
    void handlePlusLiblist(const PlusArg& plusArg);
    void handlePlusSimargs(const PlusArg& plusArg);
    void handlePlusWarnAll(const PlusArg& plusArg);
    void handlePlusWarnDisable(const PlusArg& plusArg);
    
    // Command handlers
    bool handleDashCommand(const std::string& arg, int& i, int argc, const char* argv[]);
    bool handlePlusCommand(const std::string& arg);
    
    // Validation helpers
    bool fileExists(const std::string& path) const;
    bool validateFileList(const std::vector<std::string>& files, const char* fileType) const;
    bool validateOptionalFile(const std::string& file, const char* fileType) const;
    void expandPathList(std::vector<std::string>& paths);
    void createOutputDirectory();
    
    // File list processing
    bool processFileListFiles();
    bool processFileListType(const std::vector<std::string>& fileListPaths, const char* listTypeName);
    std::vector<std::string> parseFileList(const std::string& filepath) const;
    std::string trimAndRemoveComments(const std::string& line) const;
    std::string expandEnvironmentVariables(const std::string& path) const;
};

} // namespace sv2sc::translator