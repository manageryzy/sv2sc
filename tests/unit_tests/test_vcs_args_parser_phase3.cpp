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

TEST_CASE("VCS Args Parser - Phase 3: SystemVerilog Assertions", "[vcs][phase3][assert]") {
    VCSArgsParser parser;
    
    SECTION("-assert disable") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "disable", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertDisable == true);
    }
    
    SECTION("-assert enable_diag") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "enable_diag", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertEnableDiag == true);
    }
    
    SECTION("-assert hier=<filename>") {
        std::string hierFile = TestFileHelper::getTestDataFile("assert_hierarchy.hier");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        std::string hierArg = "hier=" + hierFile;
        
        const char* args[] = {"sv2sc", "-assert", hierArg.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertHierFile == hierFile);
        
        // Static files don't need cleanup
    }
    
    SECTION("-assert filter_past") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "filter_past", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertFilterPast == true);
    }
    
    SECTION("-assert offending_values") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "offending_values", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertOffendingValues == true);
    }
    
    SECTION("-assert maxfail=<N>") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "maxfail=100", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertMaxFail == 100);
    }
    
    SECTION("-assert finish_maxfail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "finish_maxfail", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertFinishMaxFail == true);
    }
    
    SECTION("Multiple assertion options") {
        std::string hierFile = TestFileHelper::getTestDataFile("assert_hierarchy.hier");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        std::string hierArg = "hier=" + hierFile;
        
        const char* args[] = {"sv2sc", "-assert", "enable_diag", "-assert", hierArg.c_str(),
                              "-assert", "maxfail=50", testFile.c_str()};
        
        REQUIRE(parser.parse(8, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.assertEnableDiag == true);
        REQUIRE(arguments.assertHierFile == hierFile);
        REQUIRE(arguments.assertMaxFail == 50);
        
        // Static files don't need cleanup
    }
}

TEST_CASE("VCS Args Parser - Phase 3: Timing and SDF", "[vcs][phase3][timing]") {
    VCSArgsParser parser;
    
    SECTION("-sdf timing annotation") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-sdf", "max:cpu:timing.sdf", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.sdfFiles.size() == 1);
        REQUIRE(arguments.sdfFiles[0] == "max:cpu:timing.sdf");
    }
    
    SECTION("Multiple SDF files") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-sdf", "min:cpu:timing_min.sdf",
                              "-sdf", "typ:cpu:timing_typ.sdf",
                              "-sdf", "max:cpu:timing_max.sdf", testFile.c_str()};
        
        REQUIRE(parser.parse(8, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.sdfFiles.size() == 3);
        REQUIRE(arguments.sdfFiles[0] == "min:cpu:timing_min.sdf");
        REQUIRE(arguments.sdfFiles[1] == "typ:cpu:timing_typ.sdf");
        REQUIRE(arguments.sdfFiles[2] == "max:cpu:timing_max.sdf");
    }
    
    SECTION("+maxdelays") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+maxdelays", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.maxDelays == true);
    }
    
    SECTION("+mindelays") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+mindelays", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.minDelays == true);
    }
    
    SECTION("+typdelays") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+typdelays", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.typDelays == true);
    }
    
    SECTION("Delay mode options") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test +delay_mode_path
        const char* args1[] = {"sv2sc", "+delay_mode_path", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().delayModePath == true);
        
        // Test +delay_mode_zero
        const char* args2[] = {"sv2sc", "+delay_mode_zero", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().delayModeZero == true);
        
        // Test +delay_mode_unit
        const char* args3[] = {"sv2sc", "+delay_mode_unit", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().delayModeUnit == true);
        
        // Test +delay_mode_distributed
        const char* args4[] = {"sv2sc", "+delay_mode_distributed", testFile.c_str()};
        REQUIRE(parser.parse(3, args4));
        REQUIRE(parser.getArguments().delayModeDistributed == true);
    }
    
    SECTION("-sdfretain") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-sdfretain", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.sdfRetain == true);
    }
    
    SECTION("+pathpulse") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+pathpulse", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.pathPulse == true);
    }
    
    SECTION("+nospecify and +notimingcheck") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+nospecify", "+notimingcheck", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.noSpecify == true);
        REQUIRE(arguments.noTimingCheck == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 3: Code Coverage", "[vcs][phase3][coverage]") {
    VCSArgsParser parser;
    
    SECTION("-cm branch") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "branch", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageBranch == true);
    }
    
    SECTION("-cm cond") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "cond", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageCond == true);
    }
    
    SECTION("-cm fsm") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "fsm", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageFsm == true);
    }
    
    SECTION("-cm tgl") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "tgl", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageToggle == true);
    }
    
    SECTION("-cm line") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "line", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageLine == true);
    }
    
    SECTION("-cm assert") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "assert", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageAssert == true);
    }
    
    SECTION("-cm custom metric") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "custom_metric", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageMetrics.size() == 1);
        REQUIRE(arguments.coverageMetrics[0] == "custom_metric");
    }
    
    SECTION("-cm_dir coverage directory") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_dir", "./coverage_db", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageDir == "./coverage_db");
    }
    
    SECTION("-cm_name coverage name") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_name", "design_coverage", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageName == "design_coverage");
    }
    
    SECTION("-cm_hier coverage hierarchy") {
        std::string hierFile = TestFileHelper::getTestDataFile("coverage_hierarchy.hier");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_hier", hierFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageHierFile == hierFile);
        
        // Static files don't need cleanup
    }
    
    SECTION("-cm_libs coverage libraries") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_libs", "work", "-cm_libs", "tech_lib", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageLibs.size() == 2);
        REQUIRE(arguments.coverageLibs[0] == "work");
        REQUIRE(arguments.coverageLibs[1] == "tech_lib");
    }
    
    SECTION("-cm_exclude coverage exclude") {
        std::string excludeFile = TestFileHelper::getTestDataFile("coverage_exclude.exclude");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_exclude", excludeFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageExcludeFile == excludeFile);
        
        // Static files don't need cleanup
    }
    
    SECTION("-cm_cond basic") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_cond", "basic", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageCondBasic == true);
    }
    
    SECTION("-cm_report and -cm_stats") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_report", "-cm_stats", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageReport == true);
        REQUIRE(arguments.coverageStats == true);
    }
    
    SECTION("Complete coverage configuration") {
        std::string hierFile = TestFileHelper::getTestDataFile("coverage_hierarchy.hier");
        std::string excludeFile = TestFileHelper::getTestDataFile("coverage_exclude.exclude");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm", "branch", "-cm", "cond", "-cm", "fsm",
                              "-cm_dir", "./cov_db", "-cm_name", "full_cov",
                              "-cm_hier", hierFile.c_str(),
                              "-cm_exclude", excludeFile.c_str(),
                              "-cm_cond", "basic", "-cm_report", "-cm_stats",
                              testFile.c_str()};
        
        REQUIRE(parser.parse(20, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.coverageBranch == true);
        REQUIRE(arguments.coverageCond == true);
        REQUIRE(arguments.coverageFsm == true);
        REQUIRE(arguments.coverageDir == "./cov_db");
        REQUIRE(arguments.coverageName == "full_cov");
        REQUIRE(arguments.coverageHierFile == hierFile);
        REQUIRE(arguments.coverageExcludeFile == excludeFile);
        REQUIRE(arguments.coverageCondBasic == true);
        REQUIRE(arguments.coverageReport == true);
        REQUIRE(arguments.coverageStats == true);
        
        // Static files don't need cleanup
        // Static files don't need cleanup
    }
}

TEST_CASE("VCS Args Parser - Phase 3: Advanced Debug Features", "[vcs][phase3][debug]") {
    VCSArgsParser parser;
    
    SECTION("-kdb=only") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-kdb=only", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.kdbOnly == true);
    }
    
    SECTION("-debug_region") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-debug_region", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.debugRegion == true);
    }
    
    SECTION("+fsdb+") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+fsdb+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.fsdbFormat == true);
    }
    
    SECTION("-fgp fine grain parallelism") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test -fgp (default)
        const char* args1[] = {"sv2sc", "-fgp", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().fgpMode == "default");
        
        // Test -fgp=single
        const char* args2[] = {"sv2sc", "-fgp=single", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().fgpMode == "single");
        
        // Test -fgp=multi
        const char* args3[] = {"sv2sc", "-fgp=multi", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().fgpMode == "multi");
    }
    
    SECTION("-frames stack frames") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-frames", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableFrames == true);
    }
    
    SECTION("-gvalue value display") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-gvalue", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableGvalue == true);
    }
    
    SECTION("Combined advanced debug options") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-kdb=only", "-debug_region", "+fsdb+", 
                              "-fgp=multi", "-frames", "-gvalue", testFile.c_str()};
        
        REQUIRE(parser.parse(8, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.kdbOnly == true);
        REQUIRE(arguments.debugRegion == true);
        REQUIRE(arguments.fsdbFormat == true);
        REQUIRE(arguments.fgpMode == "multi");
        REQUIRE(arguments.enableFrames == true);
        REQUIRE(arguments.enableGvalue == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 3: File Validation", "[vcs][phase3][validation]") {
    VCSArgsParser parser;
    
    SECTION("Non-existent assertion hierarchy file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-assert", "hier=nonexistent.hier", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent coverage hierarchy file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_hier", "nonexistent.hier", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent coverage exclude file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-cm_exclude", "nonexistent.exclude", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
}

TEST_CASE("VCS Args Parser - Phase 3: Complex Integration", "[vcs][phase3][integration]") {
    VCSArgsParser parser;
    
    SECTION("Complete Phase 1+2+3 example") {
        // Use test data files from data directory  
        std::string advancedList = TestFileHelper::getFileListPath("advanced_list.lst");
        std::string libMap = TestFileHelper::getFileListPath("library.map");
        std::string guiConfig = TestFileHelper::getFileListPath("gui_config.cfg");
        std::string assertHier = TestFileHelper::getTestDataFile("assert_hierarchy.hier");
        std::string coverageHier = TestFileHelper::getTestDataFile("coverage_hierarchy.hier");
        std::string coverageExclude = TestFileHelper::getTestDataFile("coverage_exclude.exclude");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        std::string assertHierArg = "hier=" + assertHier;
        
        const char* args[] = {"sv2sc",
                              // Phase 1 options
                              "-sverilog", "+incdir+./include+./rtl+", "+define+WIDTH=32+",
                              // Phase 2 options
                              "-F", advancedList.c_str(), "-O3", "-mcg", "-gui", "-gfile", guiConfig.c_str(),
                              // Phase 3 assertion options
                              "-assert", "enable_diag", "-assert", assertHierArg.c_str(), 
                              "-assert", "maxfail=100",
                              // Phase 3 timing options
                              "-sdf", "max:cpu:timing.sdf", "+maxdelays", "+transport_path_delays",
                              // Phase 3 coverage options
                              "-cm", "branch", "-cm", "cond", "-cm_dir", "./coverage", 
                              "-cm_hier", coverageHier.c_str(), "-cm_exclude", coverageExclude.c_str(),
                              // Phase 3 debug options
                              "-kdb=only", "+fsdb+", "-debug_region", "-fgp=multi",
                              // Input file
                              testFile.c_str()};
        
        REQUIRE(parser.parse(36, args));
        
        const auto& arguments = parser.getArguments();
        
        // Verify Phase 1 options
        REQUIRE(arguments.enableSystemVerilog == true);
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.defineMap.find("WIDTH") != arguments.defineMap.end());
        
        // Verify Phase 2 options
        REQUIRE(arguments.advancedFileListFiles.size() == 1);
        REQUIRE(arguments.optimizationLevel == 3);
        REQUIRE(arguments.enableMCG == true);
        REQUIRE(arguments.enableGUI == true);
        
        // Verify Phase 3 assertion options
        REQUIRE(arguments.assertEnableDiag == true);
        REQUIRE(arguments.assertHierFile == assertHier);
        REQUIRE(arguments.assertMaxFail == 100);
        
        // Verify Phase 3 timing options
        REQUIRE(arguments.sdfFiles.size() == 1);
        REQUIRE(arguments.sdfFiles[0] == "max:cpu:timing.sdf");
        REQUIRE(arguments.maxDelays == true);
        REQUIRE(arguments.transportPathDelays == true);
        
        // Verify Phase 3 coverage options
        REQUIRE(arguments.coverageBranch == true);
        REQUIRE(arguments.coverageCond == true);
        REQUIRE(arguments.coverageDir == "./coverage");
        REQUIRE(arguments.coverageHierFile == coverageHier);
        REQUIRE(arguments.coverageExcludeFile == coverageExclude);
        
        // Verify Phase 3 debug options
        REQUIRE(arguments.kdbOnly == true);
        REQUIRE(arguments.fsdbFormat == true);
        REQUIRE(arguments.debugRegion == true);
        REQUIRE(arguments.fgpMode == "multi");
        
        // Cleanup
        // Static files don't need cleanup
        // Static files don't need cleanup
    }
}