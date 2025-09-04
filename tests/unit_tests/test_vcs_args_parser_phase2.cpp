#include <catch2/catch_test_macros.hpp>
#include "translator/vcs_args_parser.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>  // for setenv, unsetenv

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

TEST_CASE("VCS Args Parser - Phase 2: Advanced File Management", "[vcs][phase2][file]") {
    VCSArgsParser parser;
    
    SECTION("-F advanced file list") {
        std::string advancedFileList = TestFileHelper::getFileListPath("core_modules.f");
        
        const char* args[] = {"sv2sc", "-F", advancedFileList.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.advancedFileListFiles.size() == 1);
        REQUIRE(arguments.advancedFileListFiles[0] == advancedFileList);
    }
    
    SECTION("-file named file list") {
        std::string namedFileList = TestFileHelper::getFileListPath("multiext.files");
        
        const char* args[] = {"sv2sc", "-file", namedFileList.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.namedFileListFiles.size() == 1);
        REQUIRE(arguments.namedFileListFiles[0] == namedFileList);
    }
    
    SECTION("+liblist library list") {
        std::string liblistFile = TestFileHelper::getFileListPath("library.liblist");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        std::string liblistArg = "+liblist+" + liblistFile + "+";
        const char* args[] = {"sv2sc", liblistArg.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.liblistFiles.size() == 1);
        REQUIRE(arguments.liblistFiles[0] == liblistFile);
    }
    
    SECTION("-libmap library mapping") {
        std::string libmapFile = TestFileHelper::getFileListPath("library.map");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-libmap", libmapFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.libmapFile == libmapFile);
    }
    
    SECTION("-work directory") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-work", "/tmp/work_dir", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.workDirectory == "/tmp/work_dir");
    }
    
    SECTION("Multiple file management options combined") {
        std::string advancedList = TestFileHelper::getFileListPath("core_modules.f");
        std::string namedList = TestFileHelper::getFileListPath("multiext.files");
        std::string testFile = TestFileHelper::getTestDataFile("testbench.sv");
        
        const char* args[] = {"sv2sc", "-F", advancedList.c_str(), "-file", namedList.c_str(),
                              testFile.c_str()};
        
        REQUIRE(parser.parse(5, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.advancedFileListFiles.size() == 1);
        REQUIRE(arguments.namedFileListFiles.size() == 1);
    }
}

TEST_CASE("VCS Args Parser - Phase 2: Compilation Control", "[vcs][phase2][compilation]") {
    VCSArgsParser parser;
    
    SECTION("-mcg multi-cycle generation") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-mcg", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableMCG == true);
    }
    
    SECTION("Optimization levels -O0 to -O3") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test -O0
        const char* args0[] = {"sv2sc", "-O0", testFile.c_str()};
        REQUIRE(parser.parse(3, args0));
        REQUIRE(parser.getArguments().optimizationLevel == 0);
        
        // Test -O1
        const char* args1[] = {"sv2sc", "-O1", testFile.c_str()};
        REQUIRE(parser.parse(3, args1));
        REQUIRE(parser.getArguments().optimizationLevel == 1);
        
        // Test -O2
        const char* args2[] = {"sv2sc", "-O2", testFile.c_str()};
        REQUIRE(parser.parse(3, args2));
        REQUIRE(parser.getArguments().optimizationLevel == 2);
        
        // Test -O3
        const char* args3[] = {"sv2sc", "-O3", testFile.c_str()};
        REQUIRE(parser.parse(3, args3));
        REQUIRE(parser.getArguments().optimizationLevel == 3);
    }
    
    SECTION("-diskopt disk optimization") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-diskopt", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableDiskOpt == true);
    }
    
    SECTION("-noincrcomp disable incremental compilation") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-noincrcomp", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.disableIncrComp == true);
    }
    
    SECTION("-Mdirectory make directory") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-Mdirectory", "/tmp/make_deps", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.makeDirectory == "/tmp/make_deps");
    }
    
    SECTION("-Mupdate make update") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-Mupdate", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableMakeUpdate == true);
    }
    
    SECTION("-Mmakep make dependencies") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-Mmakep", "deps.makefile", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.makepFile == "deps.makefile");
    }
    
    SECTION("Combined compilation control options") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-mcg", "-O3", "-diskopt", "-noincrcomp", 
                              "-Mdirectory", "/tmp/deps", "-Mupdate", 
                              "-Mmakep", "project.deps", testFile.c_str()};
        
        REQUIRE(parser.parse(11, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableMCG == true);
        REQUIRE(arguments.optimizationLevel == 3);
        REQUIRE(arguments.enableDiskOpt == true);
        REQUIRE(arguments.disableIncrComp == true);
        REQUIRE(arguments.makeDirectory == "/tmp/deps");
        REQUIRE(arguments.enableMakeUpdate == true);
        REQUIRE(arguments.makepFile == "project.deps");
    }
}

TEST_CASE("VCS Args Parser - Phase 2: Debug Options", "[vcs][phase2][debug]") {
    VCSArgsParser parser;
    
    SECTION("-debug_access debug access control") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        // Test different access levels
        const char* args1[] = {"sv2sc", "-debug_access", "line", testFile.c_str()};
        REQUIRE(parser.parse(4, args1));
        REQUIRE(parser.getArguments().debugAccess == "line");
        
        const char* args2[] = {"sv2sc", "-debug_access", "class", testFile.c_str()};
        REQUIRE(parser.parse(4, args2));
        REQUIRE(parser.getArguments().debugAccess == "class");
        
        const char* args3[] = {"sv2sc", "-debug_access", "task", testFile.c_str()};
        REQUIRE(parser.parse(4, args3));
        REQUIRE(parser.getArguments().debugAccess == "task");
        
        const char* args4[] = {"sv2sc", "-debug_access", "function", testFile.c_str()};
        REQUIRE(parser.parse(4, args4));
        REQUIRE(parser.getArguments().debugAccess == "function");
    }
    
    SECTION("-kdb kernel debug") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-kdb", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableKDB == true);
    }
    
    SECTION("-gui graphical interface") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-gui", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.enableGUI == true);
    }
    
    SECTION("-gfile GUI configuration file") {
        std::string guiConfigFile = TestFileHelper::getFileListPath("gui_config.cfg");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-gfile", guiConfigFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.gfile == guiConfigFile);
    }
    
    SECTION("Combined debug options") {
        std::string guiConfigFile = TestFileHelper::getFileListPath("gui_config.cfg");
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-debug_access", "class", "-kdb", "-gui", 
                              "-gfile", guiConfigFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(8, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.debugAccess == "class");
        REQUIRE(arguments.enableKDB == true);
        REQUIRE(arguments.enableGUI == true);
        REQUIRE(arguments.gfile == guiConfigFile);
    }
}

TEST_CASE("VCS Args Parser - Phase 2: Runtime Control", "[vcs][phase2][runtime]") {
    VCSArgsParser parser;
    
    SECTION("+simargs simulation arguments") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+simargs+-arg1+-arg2+value2+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.simArgs.size() == 3);
        REQUIRE(arguments.simArgs[0] == "-arg1");
        REQUIRE(arguments.simArgs[1] == "-arg2");
        REQUIRE(arguments.simArgs[2] == "value2");
    }
    
    SECTION("-save simulation state") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-save", "simulation.state", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.saveFile == "simulation.state");
    }
    
    SECTION("-q quiet mode") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-q", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.quietMode == true);
    }
    
    SECTION("Combined runtime control options") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+simargs+-verbose+-timeout+100+", 
                              "-save", "sim.state", "-q", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.simArgs.size() == 3);
        REQUIRE(arguments.simArgs[0] == "-verbose");
        REQUIRE(arguments.simArgs[1] == "-timeout");
        REQUIRE(arguments.simArgs[2] == "100");
        REQUIRE(arguments.saveFile == "sim.state");
        REQUIRE(arguments.quietMode == true);
    }
}

TEST_CASE("VCS Args Parser - Phase 2: Error and Warning Control", "[vcs][phase2][warning]") {
    VCSArgsParser parser;
    
    SECTION("-ignore error patterns") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-ignore", "LINT_*", "-ignore", "WARNING_123", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.ignorePatterns.size() == 2);
        REQUIRE(arguments.ignorePatterns[0] == "LINT_*");
        REQUIRE(arguments.ignorePatterns[1] == "WARNING_123");
    }
    
    SECTION("+warn=all enable all warnings") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+warn=all", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.warnAll == true);
    }
    
    SECTION("+warn=no* disable specific warnings") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+warn=noLINT", "+warn=noFORMAT", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.warnDisable.size() == 2);
        REQUIRE(arguments.warnDisable[0] == "noLINT");
        REQUIRE(arguments.warnDisable[1] == "noFORMAT");
    }
    
    SECTION("-error=* error patterns") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-error=FATAL_*", "-error=SYNTAX_ERROR", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.errorPatterns.size() == 2);
        REQUIRE(arguments.errorPatterns[0] == "FATAL_*");
        REQUIRE(arguments.errorPatterns[1] == "SYNTAX_ERROR");
    }
    
    SECTION("Combined error and warning control") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+warn=all", "-ignore", "LINT_*", 
                              "+warn=noFORMAT", "-error=FATAL_*", testFile.c_str()};
        
        REQUIRE(parser.parse(7, args));
        
        const auto& arguments = parser.getArguments();
        REQUIRE(arguments.warnAll == true);
        REQUIRE(arguments.ignorePatterns.size() == 1);
        REQUIRE(arguments.ignorePatterns[0] == "LINT_*");
        REQUIRE(arguments.warnDisable.size() == 1);
        REQUIRE(arguments.warnDisable[0] == "noFORMAT");
        REQUIRE(arguments.errorPatterns.size() == 1);
        REQUIRE(arguments.errorPatterns[0] == "FATAL_*");
    }
}

TEST_CASE("VCS Args Parser - Phase 2: File Validation", "[vcs][phase2][validation]") {
    VCSArgsParser parser;
    
    SECTION("Non-existent advanced file list should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-F", "nonexistent.lst", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent named file list should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-file", "nonexistent.files", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent liblist should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "+liblist", "nonexistent.liblist", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent libmap should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-libmap", "nonexistent.map", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("Non-existent GUI file should fail") {
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", "-gfile", "nonexistent.cfg", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
}

TEST_CASE("VCS Args Parser - Phase 2: Complex Combined Usage", "[vcs][phase2][integration]") {
    VCSArgsParser parser;
    
    SECTION("Complete Phase 2 example") {
        // Use test data files from data directory
        std::string advancedList = TestFileHelper::getFileListPath("advanced_list.lst");
        std::string libMap = TestFileHelper::getFileListPath("library.map");
        std::string guiConfig = TestFileHelper::getFileListPath("gui_config.cfg");
        
        std::string testFile = TestFileHelper::getTestDataFile("core.sv");
        
        const char* args[] = {"sv2sc", 
                              // Phase 1 options
                              "-sverilog", "+incdir+./include+./rtl+", "+define+WIDTH=32+",
                              // Phase 2 advanced file management
                              "-F", advancedList.c_str(), "-libmap", libMap.c_str(), 
                              "-work", "./build",
                              // Phase 2 compilation control
                              "-O3", "-mcg", "-diskopt",
                              // Phase 2 debug options
                              "-debug_access", "class", "-kdb", "-gui", "-gfile", guiConfig.c_str(),
                              // Phase 2 runtime control
                              "+simargs+-verbose+-timeout+1000+", "-q",
                              // Phase 2 error control
                              "+warn=all", "-ignore", "LINT_*", "-error=FATAL_*",
                              // Input file
                              testFile.c_str()};
        
        REQUIRE(parser.parse(26, args));
        
        const auto& arguments = parser.getArguments();
        
        // Verify Phase 1 options
        REQUIRE(arguments.enableSystemVerilog == true);
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.defineMap.find("WIDTH") != arguments.defineMap.end());
        
        // Verify Phase 2 file management
        REQUIRE(arguments.advancedFileListFiles.size() == 1);
        REQUIRE(!arguments.libmapFile.empty());
        REQUIRE(arguments.workDirectory == "./build");
        
        // Verify Phase 2 compilation control
        REQUIRE(arguments.optimizationLevel == 3);
        REQUIRE(arguments.enableMCG == true);
        REQUIRE(arguments.enableDiskOpt == true);
        
        // Verify Phase 2 debug options
        REQUIRE(arguments.debugAccess == "class");
        REQUIRE(arguments.enableKDB == true);
        REQUIRE(arguments.enableGUI == true);
        REQUIRE(!arguments.gfile.empty());
        
        // Verify Phase 2 runtime control
        REQUIRE(arguments.simArgs.size() == 3);
        REQUIRE(arguments.quietMode == true);
        
        // Verify Phase 2 error control
        REQUIRE(arguments.warnAll == true);
        REQUIRE(arguments.ignorePatterns.size() == 1);
        REQUIRE(arguments.errorPatterns.size() == 1);
        
        // Static files don't need cleanup
    }
}