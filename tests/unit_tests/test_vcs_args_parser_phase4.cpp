#include <catch2/catch_test_macros.hpp>
#include "translator/vcs_args_parser.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>

using namespace sv2sc::translator;

// Helper class for accessing static test files
class TestFileHelper {
private:
    static std::filesystem::path findProjectRoot() {
        std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path projectRoot;
        
        // If we're in build/tests/unit_tests, go up to project root
        if (cwd.filename() == "unit_tests" && 
            cwd.parent_path().filename() == "tests" &&
            cwd.parent_path().parent_path().filename() == "build") {
            projectRoot = cwd.parent_path().parent_path().parent_path();
        } else {
            // Fallback: try to find project root by looking for CMakeLists.txt
            projectRoot = cwd;
            while (!projectRoot.empty() && projectRoot != projectRoot.root_path()) {
                if (std::filesystem::exists(projectRoot / "CMakeLists.txt") && 
                    std::filesystem::exists(projectRoot / "tests")) {
                    break;
                }
                projectRoot = projectRoot.parent_path();
            }
        }
        return projectRoot;
    }
    
public:
    static std::string getFileListPath(const std::string& listname) {
        std::filesystem::path projectRoot = findProjectRoot();
        
        // Set environment variable for file list resolution
        std::string projectRootStr = projectRoot.string();
        setenv("SV2SC_PROJECT_ROOT", projectRootStr.c_str(), 1);
        
        std::filesystem::path fileListPath = projectRoot / "tests" / "data" / "file_lists" / listname;
        return fileListPath.string();
    }
    
    static std::string getTestDataFile(const std::string& filename) {
        std::filesystem::path projectRoot = findProjectRoot();
        
        // Set environment variable for consistency
        std::string projectRootStr = projectRoot.string();
        setenv("SV2SC_PROJECT_ROOT", projectRootStr.c_str(), 1);
        
        std::filesystem::path testDataPath = projectRoot / "tests" / "basic_tests" / filename;
        return testDataPath.string();
    }
};

TEST_CASE("VCS Args Parser - Phase 4: Power Analysis", "[vcs][phase4][power]") {
    VCSArgsParser parser;
    
    SECTION("-power basic enable") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-power", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enablePower == true);
    }
    
    SECTION("-power=UPF format specification") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-power=UPF", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enablePower == true);
        REQUIRE(arguments.powerFormat == "UPF");
        REQUIRE(arguments.powerOptions.size() == 1);
        REQUIRE(arguments.powerOptions[0] == "UPF");
    }
    
    SECTION("-power= different formats") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test dump format
        const char* args1[] = {"sv2sc", "-power=dump", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().powerFormat == "dump");
        
        // Test verify format
        const char* args2[] = {"sv2sc", "-power=verify", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().powerFormat == "verify");
        
        // Test report format
        const char* args3[] = {"sv2sc", "-power=report", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().powerFormat == "report");
    }
    
    SECTION("-upf UPF file specification") {
        std::string upfFile = TestFileHelper::getTestDataFile("power_config.upf");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-upf", upfFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enablePower == true);
        REQUIRE(arguments.upfFile == upfFile);
        
        // Static files don't need cleanup: TestFileHelper::cleanup(upfFile);
    }
    
    SECTION("-power_top power top module") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-power_top", "cpu_top", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enablePower == true);
        REQUIRE(arguments.powerTopModule == "cpu_top");
    }
    
    SECTION("Combined power analysis options") {
        std::string upfFile = TestFileHelper::getTestDataFile("power_config.upf");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-power=verify", "-upf", upfFile.c_str(),
                              "-power_top", "system_top", testFile.c_str()};
        
        REQUIRE(parser.parse(7, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enablePower == true);
        REQUIRE(arguments.powerFormat == "verify");
        REQUIRE(arguments.upfFile == upfFile);
        REQUIRE(arguments.powerTopModule == "system_top");
        
        // Static files don't need cleanup: TestFileHelper::cleanup(upfFile);
    }
}

TEST_CASE("VCS Args Parser - Phase 4: OpenVera/NTB", "[vcs][phase4][ntb]") {
    VCSArgsParser parser;
    
    SECTION("-ntb basic enable") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ntb", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableNTB == true);
    }
    
    SECTION("-ntb_define macro definition") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ntb_define", "TESTBENCH_MODE", 
                              "-ntb_define", "DEBUG_LEVEL=2", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableNTB == true);
        REQUIRE(arguments.ntbDefines.size() == 2);
        REQUIRE(arguments.ntbDefines[0] == "TESTBENCH_MODE");
        REQUIRE(arguments.ntbDefines[1] == "DEBUG_LEVEL=2");
    }
    
    SECTION("-ntb_filext file extensions") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ntb_filext", ".vr", "-ntb_filext", ".vrh", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableNTB == true);
        REQUIRE(arguments.ntbFileExt.size() == 2);
        REQUIRE(arguments.ntbFileExt[0] == ".vr");
        REQUIRE(arguments.ntbFileExt[1] == ".vrh");
    }
    
    SECTION("-ntb_opts UVM version detection") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test UVM 1.1
        const char* args1[] = {"sv2sc", "-ntb_opts", "uvm-1.1", testFile.c_str()};
        REQUIRE(parser.parse(4, args1));
        REQUIRE(parser.getArguments().enableNTB == true);
        REQUIRE(parser.getArguments().enableUVM == true);
        REQUIRE(parser.getArguments().uvmVersion == "uvm-1.1");
        
        // Test UVM 1.2
        const char* args2[] = {"sv2sc", "-ntb_opts", "uvm-1.2", testFile.c_str()};
        REQUIRE(parser.parse(4, args2));
        REQUIRE(parser.getArguments().enableUVM == true);
        REQUIRE(parser.getArguments().uvmVersion == "uvm-1.2");
        
        // Test UVM IEEE
        const char* args3[] = {"sv2sc", "-ntb_opts", "uvm-ieee", testFile.c_str()};
        REQUIRE(parser.parse(4, args3));
        REQUIRE(parser.getArguments().enableUVM == true);
        REQUIRE(parser.getArguments().uvmVersion == "uvm-ieee");
    }
    
    SECTION("NTB shell configuration") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ntb_shell_only", "-ntb_sfname", "shell.vr",
                              "-ntb_sname", "test_shell", "-ntb_spath", "./shell",
                              testFile.c_str()};
        
        REQUIRE(parser.parse(9, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableNTB == true);
        REQUIRE(arguments.ntbShellOnly == true);
        REQUIRE(arguments.ntbShellFilename == "shell.vr");
        REQUIRE(arguments.ntbShellModule == "test_shell");
        REQUIRE(arguments.ntbShellPath == "./shell");
    }
    
    SECTION("NTB plus commands") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+ntb_enable_coverage", "+ntb_func_enable",
                              "+ntb_solve_control", testFile.c_str()};
        
        REQUIRE(parser.parse(5, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableNTB == true);
        REQUIRE(arguments.ntbEnableCoverage == true);
        REQUIRE(arguments.ntbFuncEnable == true);
        REQUIRE(arguments.ntbSolveControl == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 4: Advanced Optimization", "[vcs][phase4][optimization]") {
    VCSArgsParser parser;
    
    SECTION("+rad optimization") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+rad", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableRad == true);
    }
    
    SECTION("+optconfigfile+ optimization config") {
        std::string configFile = TestFileHelper::getTestDataFile("optimization.config");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        std::string configArg = "+optconfigfile+" + configFile + "+";
        
        const char* args[] = {"sv2sc", configArg.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.optConfigFile == configFile);
        
        // Static files don't need cleanup: TestFileHelper::cleanup(configFile);
    }
    
    SECTION("-hsopt= high-speed optimization") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-hsopt=race", "-hsopt=gate", 
                              "-hsopt=j4", testFile.c_str()};
        
        REQUIRE(parser.parse(5, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.hsOptOptions.size() == 3);
        REQUIRE(arguments.hsOptOptions[0] == "race");
        REQUIRE(arguments.hsOptOptions[1] == "gate");
        REQUIRE(arguments.hsOptOptions[2] == "j4");
    }
    
    SECTION("Compilation optimization") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-partcomp", "-fastpartcomp", 
                              "+plus-optimization", testFile.c_str()};
        
        REQUIRE(parser.parse(5, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.partialCompilation == true);
        REQUIRE(arguments.fastPartialCompilation == true);
        REQUIRE(arguments.plusOptimization == true);
    }
    
    SECTION("-sparse+ sparse matrix optimization") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-sparse+matrix", "-sparse+solver", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.sparseOptions.size() == 2);
        REQUIRE(arguments.sparseOptions[0] == "matrix");
        REQUIRE(arguments.sparseOptions[1] == "solver");
    }
}

TEST_CASE("VCS Args Parser - Phase 4: Distributed Simulation", "[vcs][phase4][distsim]") {
    VCSArgsParser parser;
    
    SECTION("-distsim basic enable") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-distsim", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableDistSim == true);
        REQUIRE(arguments.distSimMode == "default");
    }
    
    SECTION("-distsim= mode specification") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test different modes
        const char* args1[] = {"sv2sc", "-distsim=setup", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().enableDistSim == true);
        REQUIRE(parser.getArguments().distSimMode == "setup");
        
        const char* args2[] = {"sv2sc", "-distsim=run", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().distSimMode == "run");
        
        const char* args3[] = {"sv2sc", "-distsim=collect", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().distSimMode == "collect");
        
        const char* args4[] = {"sv2sc", "-distsim=farm", testFile.c_str()};
        REQUIRE(parser.parse(3, args4));
        REQUIRE(parser.getArguments().distSimMode == "farm");
    }
}

TEST_CASE("VCS Args Parser - Phase 4: SystemC Integration", "[vcs][phase4][systemc]") {
    VCSArgsParser parser;
    
    SECTION("-sysc basic enable") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-sysc", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableSysC == true);
        REQUIRE(arguments.sysCMode == "default");
    }
    
    SECTION("-sysc= mode specification") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test different SystemC modes
        const char* args1[] = {"sv2sc", "-sysc=show", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().enableSysC == true);
        REQUIRE(parser.getArguments().sysCMode == "show");
        
        const char* args2[] = {"sv2sc", "-sysc=incr", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().sysCMode == "incr");
        
        const char* args3[] = {"sv2sc", "-sysc=adjust", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().sysCMode == "adjust");
    }
    
    SECTION("+vc+ SystemC abstraction") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+vc+abstract", "+vc+timing", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableSysC == true);
        REQUIRE(arguments.vcAbstract.size() == 2);
        REQUIRE(arguments.vcAbstract[0] == "abstract");
        REQUIRE(arguments.vcAbstract[1] == "timing");
    }
    
    SECTION("-systemcrunconfigure runtime config") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-systemcrunconfigure", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableSysC == true);
        REQUIRE(arguments.sysCRunConfigure == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 4: Verification Methodology", "[vcs][phase4][verification]") {
    VCSArgsParser parser;
    
    SECTION("+UVM methodology") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+UVM", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableUVM == true);
    }
    
    SECTION("+define+UVM_VCS_RECORD UVM recording") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+define+UVM_VCS_RECORD+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableUVM == true);
        REQUIRE(arguments.uvmVcsRecord == true);
    }
    
    SECTION("+UVM_PHASE_RECORD phase recording") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+UVM_PHASE_RECORD", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableUVM == true);
        REQUIRE(arguments.uvmPhaseRecord == true);
    }
    
    SECTION("Verification methodologies") {
        std::string ovaFile = TestFileHelper::getTestDataFile("verification.ova");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-vera", "-psl", "-ova_file", ovaFile.c_str(),
                              "+assert_count", testFile.c_str()};
        
        REQUIRE(parser.parse(7, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableVera == true);
        REQUIRE(arguments.enablePSL == true);
        REQUIRE(arguments.ovaFile == ovaFile);
        REQUIRE(arguments.assertCount == true);
        
        // Static files don't need cleanup: TestFileHelper::cleanup(ovaFile);
    }
}

TEST_CASE("VCS Args Parser - Phase 4: Advanced File Handling", "[vcs][phase4][filehandling]") {
    VCSArgsParser parser;
    
    SECTION("File protection commands") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+protect", "-protect123", "+autoprotect",
                              "+putprotect", "+ipprotect", testFile.c_str()};
        
        REQUIRE(parser.parse(7, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableProtect == true);
        REQUIRE(arguments.protect123 == true);
        REQUIRE(arguments.autoProtect == true);
        REQUIRE(arguments.putProtect == true);
        REQUIRE(arguments.ipProtect == true);
    }
    
    SECTION("Auto protection levels") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-auto2protect", "-auto3protect", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.auto2Protect == true);
        REQUIRE(arguments.auto3Protect == true);
    }
    
    SECTION("IP handling") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ipout", "/tmp/ip_output.vp", 
                              "-ipopt=encrypt", "-ipopt=compress", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.ipOutFile == "/tmp/ip_output.vp");
        REQUIRE(arguments.ipOptOptions.size() == 2);
        REQUIRE(arguments.ipOptOptions[0] == "encrypt");
        REQUIRE(arguments.ipOptOptions[1] == "compress");
    }
    
    SECTION("Encryption and decryption") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+encrypt", "+decrypt", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableEncrypt == true);
        REQUIRE(arguments.enableDecrypt == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 4: File Validation", "[vcs][phase4][validation]") {
    VCSArgsParser parser;
    
    SECTION("Non-existent UPF file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-upf", "nonexistent.upf", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent OVA file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ova_file", "nonexistent.ova", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
}

TEST_CASE("VCS Args Parser - Phase 4: Complex Integration", "[vcs][phase4][integration]") {
    VCSArgsParser parser;
    
    SECTION("Complete Phase 1+2+3+4 example") {
        // Use test data files from data directory
        std::string advancedList = TestFileHelper::getFileListPath("advanced_list.lst");
        std::string upfFile = TestFileHelper::getTestDataFile("power_config.upf");
        std::string configFile = TestFileHelper::getTestDataFile("optimization.config");
        std::string ovaFile = TestFileHelper::getTestDataFile("verification.ova");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        std::string configArg = "+optconfigfile+" + configFile + "+";
        
        // Store all dynamic strings to ensure they remain valid
        std::vector<std::string> argStrings = {
            "sv2sc", "-sverilog", "+incdir+./include+./rtl+", "+define+WIDTH=32+",
            "-F", advancedList, "-O3", "-mcg",
            "-assert", "enable_diag", "-cm", "branch", "+fsdb+",
            "-power=verify", "-upf", upfFile, "-power_top", "cpu_top",
            "+UVM", "-ntb_opts", "uvm-ieee", "+ntb_enable_coverage",
            "+rad", configArg, "-hsopt=gate", "-partcomp",
            "-distsim=farm", "-sysc=incr", "+vc+abstract",
            "-vera", "-ova_file", ovaFile, "+protect", "+encrypt",
            testFile
        };
        
        // Convert to const char* array
        std::vector<const char*> args;
        for (const auto& str : argStrings) {
            args.push_back(str.c_str());
        }
        
        REQUIRE(parser.parse(static_cast<int>(args.size()), args.data()));
        
        const auto& arguments = parser.getArguments();
        
        // Verify Phase 1 options
        REQUIRE(arguments.enableSystemVerilog == true);
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.defineMap.find("WIDTH") != arguments.defineMap.end());
        
        // Verify Phase 2 options
        REQUIRE(arguments.advancedFileListFiles.size() == 1);
        REQUIRE(arguments.optimizationLevel == 3);
        REQUIRE(arguments.enableMCG == true);
        
        // Verify Phase 3 options
        REQUIRE(arguments.assertEnableDiag == true);
        REQUIRE(arguments.coverageBranch == true);
        REQUIRE(arguments.fsdbFormat == true);
        
        // Verify Phase 4 power options
        REQUIRE(arguments.enablePower == true);
        REQUIRE(arguments.powerFormat == "verify");
        REQUIRE(arguments.upfFile == upfFile);
        REQUIRE(arguments.powerTopModule == "cpu_top");
        
        // Verify Phase 4 UVM/NTB options
        REQUIRE(arguments.enableUVM == true);
        REQUIRE(arguments.uvmVersion == "uvm-ieee");
        REQUIRE(arguments.ntbEnableCoverage == true);
        
        // Verify Phase 4 optimization options
        REQUIRE(arguments.enableRad == true);
        REQUIRE(arguments.optConfigFile == configFile);
        REQUIRE(arguments.hsOptOptions.size() == 1);
        REQUIRE(arguments.hsOptOptions[0] == "gate");
        REQUIRE(arguments.partialCompilation == true);
        
        // Verify Phase 4 distributed sim and SystemC options
        REQUIRE(arguments.enableDistSim == true);
        REQUIRE(arguments.distSimMode == "farm");
        REQUIRE(arguments.enableSysC == true);
        REQUIRE(arguments.sysCMode == "incr");
        REQUIRE(arguments.vcAbstract.size() == 1);
        REQUIRE(arguments.vcAbstract[0] == "abstract");
        
        // Verify Phase 4 verification and file handling options
        REQUIRE(arguments.enableVera == true);
        REQUIRE(arguments.ovaFile == ovaFile);
        REQUIRE(arguments.enableProtect == true);
        REQUIRE(arguments.enableEncrypt == true);
        
        // Cleanup
        // Static files don't need cleanup: TestFileHelper::cleanup(advancedList);
        // Static files don't need cleanup: TestFileHelper::cleanup(upfFile);
        // Static files don't need cleanup: TestFileHelper::cleanup(configFile);
        // Static files don't need cleanup: TestFileHelper::cleanup(ovaFile);
    }
}