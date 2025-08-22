#include <catch2/catch_test_macros.hpp>
#include "translator/vcs_args_parser.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>  // for setenv, unsetenv, getenv

using namespace sv2sc::translator;

// Helper function to create temporary test files
class TempFileHelper {
public:
    static std::string createTempFile(const std::string& content, const std::string& suffix = ".sv") {
        static int counter = 0;
        std::string filename = "/tmp/test_" + std::to_string(counter++) + suffix;
        std::ofstream file(filename);
        file << content;
        file.close();
        return filename;
    }
    
    // Get path to existing test data file using environment variable or relative path
    static std::string getTestDataFile(const std::string& filename) {
        const char* testDataDir = std::getenv("SV2SC_TEST_DATA_DIR");
        if (testDataDir != nullptr) {
            return std::string(testDataDir) + "/" + filename;
        }
        // Fallback to relative path from project root
        return "tests/data/vcs_test_files/" + filename;
    }
    
    // Create a file list pointing to test data files
    static std::string createTestFileList(const std::vector<std::string>& filenames, const std::string& suffix = ".f") {
        static int counter = 0;
        std::string listFile = "/tmp/test_list_" + std::to_string(counter++) + suffix;
        std::ofstream file(listFile);
        
        for (const auto& filename : filenames) {
            file << getTestDataFile(filename) << "\n";
        }
        
        file.close();
        return listFile;
    }
    
    static void cleanup(const std::string& filename) {
        std::filesystem::remove(filename);
    }
};

TEST_CASE("VCS Args Parser - Phase 1: Basic File Handling", "[vcs][phase1]") {
    VCSArgsParser parser;
    
    SECTION("Input files are parsed correctly") {
        std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
        const char* args[] = {"sv2sc", testFile.c_str()};
        
        REQUIRE(parser.parse(2, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 1);
        REQUIRE(arguments.inputFiles[0] == testFile);
        
        TempFileHelper::cleanup(testFile);
    }
    
    SECTION("Multiple input files") {
        std::string file1 = TempFileHelper::createTempFile("module test1; endmodule");
        std::string file2 = TempFileHelper::createTempFile("module test2; endmodule");
        const char* args[] = {"sv2sc", file1.c_str(), file2.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 2);
        REQUIRE(arguments.inputFiles[0] == file1);
        REQUIRE(arguments.inputFiles[1] == file2);
        
        TempFileHelper::cleanup(file1);
        TempFileHelper::cleanup(file2);
    }
}

TEST_CASE("VCS Args Parser - Phase 1: Library Management (-v)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-v single library file") {
        std::string libFile = TempFileHelper::createTempFile("module lib; endmodule");
        const char* args[] = {"sv2sc", "-v", libFile.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryFiles[0] == libFile);
        
        TempFileHelper::cleanup(libFile);
    }
    
    SECTION("-v multiple library files") {
        std::string libFile1 = TempFileHelper::createTempFile("module lib1; endmodule", ".sv");
        std::string libFile2 = TempFileHelper::createTempFile("module lib2; endmodule", ".sv");
        std::string libFile3 = TempFileHelper::createTempFile("module lib3; endmodule", ".v");
        
        const char* args[] = {
            "sv2sc", 
            "-v", libFile1.c_str(),
            "-v", libFile2.c_str(),
            "-v", libFile3.c_str(),
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryFiles.size() == 3);
        REQUIRE(arguments.libraryFiles[0] == libFile1);
        REQUIRE(arguments.libraryFiles[1] == libFile2);
        REQUIRE(arguments.libraryFiles[2] == libFile3);
        
        TempFileHelper::cleanup(libFile1);
        TempFileHelper::cleanup(libFile2);
        TempFileHelper::cleanup(libFile3);
    }
    
    SECTION("-v with different file extensions") {
        std::string svFile = TempFileHelper::createTempFile("module sv_lib; endmodule", ".sv");
        std::string vFile = TempFileHelper::createTempFile("module v_lib; endmodule", ".v");
        std::string vhFile = TempFileHelper::createTempFile("`define LIB_VERSION 1", ".vh");
        
        const char* args[] = {
            "sv2sc", 
            "-v", svFile.c_str(),
            "-v", vFile.c_str(),
            "-v", vhFile.c_str(),
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryFiles.size() == 3);
        REQUIRE(arguments.libraryFiles[0] == svFile);
        REQUIRE(arguments.libraryFiles[1] == vFile);
        REQUIRE(arguments.libraryFiles[2] == vhFile);
        
        TempFileHelper::cleanup(svFile);
        TempFileHelper::cleanup(vFile);
        TempFileHelper::cleanup(vhFile);
    }
    
    SECTION("-v non-existent file should fail validation") {
        const char* args[] = {"sv2sc", "-v", "/non/existent/lib.sv", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("-v with relative and absolute paths") {
        std::string libFile1 = TempFileHelper::createTempFile("module lib1; endmodule");
        
        const char* args[] = {
            "sv2sc", 
            "-v", libFile1.c_str(),  // absolute path
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryFiles[0] == libFile1);
        
        TempFileHelper::cleanup(libFile1);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Library Directories (-y)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-y single directory") {
        const char* args[] = {"sv2sc", "-y", "/tmp", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 1);
        REQUIRE(arguments.libraryPaths[0].find("/tmp") != std::string::npos);
    }
    
    SECTION("-y multiple directories") {
        const char* args[] = {
            "sv2sc", 
            "-y", "/usr/local/lib",
            "-y", "/opt/verilog/lib",
            "-y", "./local_libs",
            "-y", "/tmp",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(10, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 4);
        REQUIRE(arguments.libraryPaths[0].find("/usr/local/lib") != std::string::npos);
        REQUIRE(arguments.libraryPaths[1].find("/opt/verilog/lib") != std::string::npos);
        REQUIRE(arguments.libraryPaths[2].find("local_libs") != std::string::npos);
        REQUIRE(arguments.libraryPaths[3].find("/tmp") != std::string::npos);
    }
    
    SECTION("-y with relative paths") {
        const char* args[] = {
            "sv2sc", 
            "-y", "./libs",
            "-y", "../common/libs",
            "-y", "../../vendor/libs",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 3);
        // Paths should be expanded to absolute
        REQUIRE(arguments.libraryPaths[0].find("libs") != std::string::npos);
        REQUIRE(arguments.libraryPaths[1].find("common/libs") != std::string::npos);
        REQUIRE(arguments.libraryPaths[2].find("vendor/libs") != std::string::npos);
    }
    
    SECTION("-y with paths containing spaces") {
        // Note: This would typically require shell quoting in real usage
        std::filesystem::create_directories("/tmp/lib with spaces");
        
        const char* args[] = {"sv2sc", "-y", "/tmp/lib with spaces", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 1);
        REQUIRE(arguments.libraryPaths[0].find("lib with spaces") != std::string::npos);
        
        std::filesystem::remove_all("/tmp/lib with spaces");
    }
    
    SECTION("-y combined with -v") {
        std::string libFile = TempFileHelper::createTempFile("module specific_lib; endmodule");
        
        const char* args[] = {
            "sv2sc", 
            "-y", "/usr/lib/verilog",
            "-y", "/opt/libs", 
            "-v", libFile.c_str(),
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 2);
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryPaths[0].find("/usr/lib/verilog") != std::string::npos);
        REQUIRE(arguments.libraryPaths[1].find("/opt/libs") != std::string::npos);
        REQUIRE(arguments.libraryFiles[0] == libFile);
        
        TempFileHelper::cleanup(libFile);
    }
    
    SECTION("-y order preservation") {
        const char* args[] = {
            "sv2sc", 
            "-y", "/first/priority",
            "-y", "/second/priority",
            "-y", "/third/priority",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryPaths.size() == 3);
        // Order should be preserved for library search precedence
        REQUIRE(arguments.libraryPaths[0].find("first/priority") != std::string::npos);
        REQUIRE(arguments.libraryPaths[1].find("second/priority") != std::string::npos);
        REQUIRE(arguments.libraryPaths[2].find("third/priority") != std::string::npos);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: File Lists (-f)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-f single file list") {
        std::string fileList = TempFileHelper::createTestFileList({
            "design1.sv",
            "testbench.sv", 
            "memory.sv"
        });
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 1);
        REQUIRE(arguments.fileListFiles[0] == fileList);
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("-f multiple file lists") {
        std::string fileList1 = TempFileHelper::createTestFileList({
            "core.sv",
            "design1.sv"
        });
        std::string fileList2 = TempFileHelper::createTestFileList({
            "memory.sv",
            "interfaces.sv"
        });
        std::string fileList3 = TempFileHelper::createTestFileList({
            "testbench.sv",
            "design2.sv"
        });
        
        const char* args[] = {
            "sv2sc", 
            "-f", fileList1.c_str(),
            "-f", fileList2.c_str(),
            "-f", fileList3.c_str(),
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 3);
        REQUIRE(arguments.fileListFiles[0] == fileList1);
        REQUIRE(arguments.fileListFiles[1] == fileList2);
        REQUIRE(arguments.fileListFiles[2] == fileList3);
        
        TempFileHelper::cleanup(fileList1);
        TempFileHelper::cleanup(fileList2);
        TempFileHelper::cleanup(fileList3);
    }
    
    SECTION("-f with different extensions") {
        std::string dotF = TempFileHelper::createTestFileList({"design1.sv"}, ".f");
        std::string dotList = TempFileHelper::createTestFileList({"design2.sv"}, ".list");
        std::string dotFiles = TempFileHelper::createTestFileList({"design3.v"}, ".files");
        
        const char* args[] = {
            "sv2sc", 
            "-f", dotF.c_str(),
            "-f", dotList.c_str(),
            "-f", dotFiles.c_str(),
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 3);
        REQUIRE(arguments.fileListFiles[0] == dotF);
        REQUIRE(arguments.fileListFiles[1] == dotList);
        REQUIRE(arguments.fileListFiles[2] == dotFiles);
        
        TempFileHelper::cleanup(dotF);
        TempFileHelper::cleanup(dotList);
        TempFileHelper::cleanup(dotFiles);
    }
    
    SECTION("-f non-existent file should fail validation") {
        const char* args[] = {"sv2sc", "-f", "/non/existent/files.f", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(4, args));
    }
    
    SECTION("-f combined with other file options") {
        std::string fileList = TempFileHelper::createTestFileList({
            "core_modules.sv",
            "interfaces.sv"
        });
        std::string libFile = TempFileHelper::createTempFile("library_module.sv");
        
        const char* args[] = {
            "sv2sc", 
            "-f", fileList.c_str(),
            "-v", libFile.c_str(),
            "-y", "/usr/local/verilog",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 1);
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryPaths.size() == 1);
        REQUIRE(arguments.inputFiles.size() >= 1); // May have additional files from file list
        
        TempFileHelper::cleanup(fileList);
        TempFileHelper::cleanup(libFile);
    }
    
    SECTION("-f with relative and absolute paths") {
        // Create file list with relative path to existing test data
        std::string fileList = TempFileHelper::createTempFile(
            "relative/path/design.sv\n"   // relative path within test data
            + TempFileHelper::getTestDataFile("testbench.sv") + "\n"  // absolute path
            + TempFileHelper::getTestDataFile("core.sv"),
            ".f"
        );
        
        // Change to test data directory for relative path resolution
        std::string originalDir = std::filesystem::current_path();
        std::filesystem::current_path("/home/mana/workspace/sv2sc/tests/data/vcs_test_files");
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 1);
        REQUIRE(arguments.fileListFiles[0] == fileList);
        
        // Restore original directory
        std::filesystem::current_path(originalDir);
        TempFileHelper::cleanup(fileList);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: File List Content Parsing (-f)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string mainFile = TempFileHelper::createTempFile("module main; endmodule");
    
    SECTION("-f processes file list contents correctly") {
        // Create file list that references test data files
        std::string fileList = TempFileHelper::createTempFile(
            "// File list for simulation\n"
            + TempFileHelper::getTestDataFile("design1.sv") + "\n"
            + TempFileHelper::getTestDataFile("design2.sv") + "\n" 
            + TempFileHelper::getTestDataFile("design3.v"),
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        // Check that file list is recorded
        REQUIRE(arguments.fileListFiles.size() == 1);
        REQUIRE(arguments.fileListFiles[0] == fileList);
        
        // Check that files from the list were added to input files
        REQUIRE(arguments.inputFiles.size() == 4); // 3 from list + 1 main file
        
        // Verify the files from the list are included
        bool foundFile1 = false, foundFile2 = false, foundFile3 = false, foundMain = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("design1") != std::string::npos) foundFile1 = true;
            if (inputFile.find("design2") != std::string::npos) foundFile2 = true;
            if (inputFile.find("design3") != std::string::npos) foundFile3 = true;
            if (inputFile == mainFile) foundMain = true;
        }
        
        REQUIRE(foundFile1);
        REQUIRE(foundFile2);
        REQUIRE(foundFile3);
        REQUIRE(foundMain);
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("-f handles comments and empty lines") {
        std::string fileList = TempFileHelper::createTempFile(
            "// This is a comment\n"
            "# This is also a comment\n"
            "\n"  // empty line
            + TempFileHelper::getTestDataFile("core.sv") + "\n"
            "   \n"  // whitespace only line
            + TempFileHelper::getTestDataFile("memory.sv") + "  // inline comment\n"
            "# Another comment line\n"
            "\n",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        // Should have processed only the valid files, ignoring comments and empty lines
        REQUIRE(arguments.inputFiles.size() == 3); // 2 from list + 1 main file
        
        bool foundCore = false, foundMemory = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("core.sv") != std::string::npos) foundCore = true;
            if (inputFile.find("memory.sv") != std::string::npos) foundMemory = true;
        }
        
        REQUIRE(foundCore);
        REQUIRE(foundMemory);
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("-f handles quoted filenames") {
        std::string fileList = TempFileHelper::createTempFile(
            "\"" + TempFileHelper::getTestDataFile("design1.sv") + "\"\n"
            "'" + TempFileHelper::getTestDataFile("design2.sv") + "'",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 3); // 2 from list + 1 main file
        
        bool foundFile1 = false, foundFile2 = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("design1") != std::string::npos) foundFile1 = true;
            if (inputFile.find("design2") != std::string::npos) foundFile2 = true;
        }
        
        REQUIRE(foundFile1);
        REQUIRE(foundFile2);
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("-f fails when referenced files don't exist") {
        std::string fileList = TempFileHelper::createTempFile(
            "/non/existent/file1.sv\n"
            "/non/existent/file2.v",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        // Should fail validation when files in list don't exist
        REQUIRE_FALSE(parser.parse(4, args));
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("-f with multiple file lists") {
        std::string fileList1 = TempFileHelper::createTempFile(
            "// Core modules\n" 
            + TempFileHelper::getTestDataFile("core.sv") + "\n" 
            + TempFileHelper::getTestDataFile("design1.sv"),
            ".f"
        );
        std::string fileList2 = TempFileHelper::createTempFile(
            "// Testbench modules\n" 
            + TempFileHelper::getTestDataFile("testbench.sv") + "\n" 
            + TempFileHelper::getTestDataFile("design2.sv"),
            ".f"
        );
        
        const char* args[] = {
            "sv2sc", 
            "-f", fileList1.c_str(),
            "-f", fileList2.c_str(),
            mainFile.c_str()
        };
        
        REQUIRE(parser.parse(6, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.fileListFiles.size() == 2);
        REQUIRE(arguments.inputFiles.size() == 5); // 4 from lists + 1 main file
        
        // Verify all files are included
        bool foundCore = false, foundDesign1 = false, foundTestbench = false, foundDesign2 = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("core.sv") != std::string::npos) foundCore = true;
            if (inputFile.find("design1.sv") != std::string::npos) foundDesign1 = true;
            if (inputFile.find("testbench.sv") != std::string::npos) foundTestbench = true;
            if (inputFile.find("design2.sv") != std::string::npos) foundDesign2 = true;
        }
        
        REQUIRE(foundCore);
        REQUIRE(foundDesign1);
        REQUIRE(foundTestbench);
        REQUIRE(foundDesign2);
        
        TempFileHelper::cleanup(fileList1);
        TempFileHelper::cleanup(fileList2);
    }
    
    SECTION("-f with whitespace handling") {
        std::string fileList = TempFileHelper::createTempFile(
            "   " + TempFileHelper::getTestDataFile("core.sv") + "   \n"  // leading and trailing spaces
            "\t" + TempFileHelper::getTestDataFile("memory.sv") + "\t\n", // leading and trailing tabs
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 3); // 2 from list + 1 main file
        
        bool foundCore = false, foundMemory = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("core.sv") != std::string::npos) foundCore = true;
            if (inputFile.find("memory.sv") != std::string::npos) foundMemory = true;
        }
        
        REQUIRE(foundCore);
        REQUIRE(foundMemory);
        
        TempFileHelper::cleanup(fileList);
    }
    
    TempFileHelper::cleanup(mainFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Environment Variable Support in File Lists", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string mainFile = TempFileHelper::createTempFile("module main; endmodule");
    
    SECTION("Environment variables in file list - $VAR format") {
        // Set test environment variable
        setenv("TEST_DATA_PATH", "/home/mana/workspace/sv2sc/tests/data/vcs_test_files", 1);
        
        std::string fileList = TempFileHelper::createTempFile(
            "$TEST_DATA_PATH/core.sv\n"
            "$TEST_DATA_PATH/design1.sv",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        // Should have processed the files with expanded environment variables
        REQUIRE(arguments.inputFiles.size() == 3); // 2 from list + 1 main file
        
        bool foundCore = false, foundDesign1 = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("core.sv") != std::string::npos) foundCore = true;
            if (inputFile.find("design1.sv") != std::string::npos) foundDesign1 = true;
        }
        
        REQUIRE(foundCore);
        REQUIRE(foundDesign1);
        
        // Clean up
        unsetenv("TEST_DATA_PATH");
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("Environment variables in file list - ${VAR} format") {
        // Set test environment variables
        setenv("PROJECT_ROOT", "/home/mana/workspace/sv2sc", 1);
        setenv("RTL_DIR", "tests/data/vcs_test_files", 1);
        
        std::string fileList = TempFileHelper::createTempFile(
            "${PROJECT_ROOT}/${RTL_DIR}/memory.sv\n"
            "${PROJECT_ROOT}/${RTL_DIR}/interfaces.sv",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 3); // 2 from list + 1 main file
        
        bool foundMemory = false, foundInterfaces = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("memory.sv") != std::string::npos) foundMemory = true;
            if (inputFile.find("interfaces.sv") != std::string::npos) foundInterfaces = true;
        }
        
        REQUIRE(foundMemory);
        REQUIRE(foundInterfaces);
        
        // Clean up
        unsetenv("PROJECT_ROOT");
        unsetenv("RTL_DIR");
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("Mixed environment variables and regular paths") {
        setenv("TEST_RTL", "/home/mana/workspace/sv2sc/tests/data/vcs_test_files", 1);
        
        std::string fileList = TempFileHelper::createTempFile(
            "$TEST_RTL/core.sv\n"
            + TempFileHelper::getTestDataFile("design2.sv") + "\n"  // regular absolute path
            "${TEST_RTL}/testbench.sv",
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.inputFiles.size() == 4); // 3 from list + 1 main file
        
        bool foundCore = false, foundDesign2 = false, foundTestbench = false;
        for (const auto& inputFile : arguments.inputFiles) {
            if (inputFile.find("core.sv") != std::string::npos) foundCore = true;
            if (inputFile.find("design2.sv") != std::string::npos) foundDesign2 = true;
            if (inputFile.find("testbench.sv") != std::string::npos) foundTestbench = true;
        }
        
        REQUIRE(foundCore);
        REQUIRE(foundDesign2);
        REQUIRE(foundTestbench);
        
        unsetenv("TEST_RTL");
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("Undefined environment variables remain unexpanded") {
        std::string fileList = TempFileHelper::createTempFile(
            "$UNDEFINED_VAR/some_file.sv\n"
            + TempFileHelper::getTestDataFile("core.sv"),  // fallback to working file
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        // This should fail because $UNDEFINED_VAR/some_file.sv won't exist
        REQUIRE_FALSE(parser.parse(4, args));
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("Environment variables with HOME") {
        // Use common HOME environment variable
        const char* homeValue = std::getenv("HOME");
        REQUIRE(homeValue != nullptr); // HOME should be defined in test environment
        
        std::string fileList = TempFileHelper::createTempFile(
            TempFileHelper::getTestDataFile("core.sv") + "\n"  // working file first
            "$HOME/non_existent_file.sv",  // this will cause failure but demonstrates expansion
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        // This should fail because $HOME/non_existent_file.sv likely doesn't exist
        // but it shows that HOME was expanded (different error than undefined var)
        REQUIRE_FALSE(parser.parse(4, args));
        
        TempFileHelper::cleanup(fileList);
    }
    
    SECTION("Environment variables in comments should not be expanded") {
        setenv("TEST_COMMENT_VAR", "/some/path", 1);
        
        std::string fileList = TempFileHelper::createTempFile(
            "// This has $TEST_COMMENT_VAR but should not be expanded\n"
            "# Another comment with ${TEST_COMMENT_VAR}\n"
            + TempFileHelper::getTestDataFile("core.sv"),
            ".f"
        );
        
        const char* args[] = {"sv2sc", "-f", fileList.c_str(), mainFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        // Should only have core.sv + main file (comments with env vars ignored)
        REQUIRE(arguments.inputFiles.size() == 2);
        
        unsetenv("TEST_COMMENT_VAR");
        TempFileHelper::cleanup(fileList);
    }
    
    TempFileHelper::cleanup(mainFile);
}
    
TEST_CASE("VCS Args Parser - Phase 1: Library Extensions (+libext+)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("+libext+ basic extensions") {
        const char* args[] = {"sv2sc", "+libext+.sv+.v+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryExtensions.size() == 2);
        REQUIRE(arguments.libraryExtensions[0] == ".sv");
        REQUIRE(arguments.libraryExtensions[1] == ".v");
    }
    
    SECTION("+libext+ comprehensive extensions") {
        const char* args[] = {"sv2sc", "+libext+.sv+.v+.vh+.svh+.vlib+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryExtensions.size() == 5);
        REQUIRE(arguments.libraryExtensions[0] == ".sv");
        REQUIRE(arguments.libraryExtensions[1] == ".v");
        REQUIRE(arguments.libraryExtensions[2] == ".vh");
        REQUIRE(arguments.libraryExtensions[3] == ".svh");
        REQUIRE(arguments.libraryExtensions[4] == ".vlib");
    }
    
    SECTION("+libext+ multiple separate arguments") {
        const char* args[] = {
            "sv2sc", 
            "+libext+.sv+.v+",
            "+libext+.vh+.svh+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryExtensions.size() == 4);
        REQUIRE(arguments.libraryExtensions[0] == ".sv");
        REQUIRE(arguments.libraryExtensions[1] == ".v");
        REQUIRE(arguments.libraryExtensions[2] == ".vh");
        REQUIRE(arguments.libraryExtensions[3] == ".svh");
    }
    
    SECTION("+libext+ without leading dots") {
        const char* args[] = {"sv2sc", "+libext+sv+v+vh+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryExtensions.size() == 3);
        REQUIRE(arguments.libraryExtensions[0] == "sv");
        REQUIRE(arguments.libraryExtensions[1] == "v");
        REQUIRE(arguments.libraryExtensions[2] == "vh");
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Include Directories (-I)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-I single include directory") {
        const char* args[] = {"sv2sc", "-I", "/usr/include", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 1);
        REQUIRE(arguments.includePaths[0].find("/usr/include") != std::string::npos);
    }
    
    SECTION("-I multiple include directories") {
        const char* args[] = {
            "sv2sc", 
            "-I", "/usr/include/systemverilog",
            "-I", "/opt/verilog/include",
            "-I", "./project/include",
            "-I", "../common/include",
            "-I", "/tmp/headers",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(12, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 5);
        REQUIRE(arguments.includePaths[0].find("/usr/include/systemverilog") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("/opt/verilog/include") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("project/include") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("common/include") != std::string::npos);
        REQUIRE(arguments.includePaths[4].find("/tmp/headers") != std::string::npos);
    }
    
    SECTION("-I with relative paths") {
        const char* args[] = {
            "sv2sc", 
            "-I", "./include",
            "-I", "./src/headers", 
            "-I", "../shared/include",
            "-I", "../../vendor/include",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(10, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 4);
        // Paths should be expanded to absolute
        for (const auto& path : arguments.includePaths) {
            REQUIRE(std::filesystem::path(path).is_absolute());
        }
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("headers") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("shared") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("vendor") != std::string::npos);
    }
    
    SECTION("-I with paths containing spaces and special characters") {
        std::filesystem::create_directories("/tmp/include with spaces");
        std::filesystem::create_directories("/tmp/include-with-dashes");
        std::filesystem::create_directories("/tmp/include_with_underscores");
        
        const char* args[] = {
            "sv2sc", 
            "-I", "/tmp/include with spaces",
            "-I", "/tmp/include-with-dashes",
            "-I", "/tmp/include_with_underscores",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 3);
        REQUIRE(arguments.includePaths[0].find("include with spaces") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("include-with-dashes") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("include_with_underscores") != std::string::npos);
        
        std::filesystem::remove_all("/tmp/include with spaces");
        std::filesystem::remove_all("/tmp/include-with-dashes");
        std::filesystem::remove_all("/tmp/include_with_underscores");
    }
    
    SECTION("-I order preservation for search precedence") {
        const char* args[] = {
            "sv2sc", 
            "-I", "/first/priority",
            "-I", "/second/priority",
            "-I", "/third/priority",
            "-I", "/fourth/priority",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(10, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 4);
        // Order matters for include search precedence
        REQUIRE(arguments.includePaths[0].find("first/priority") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("second/priority") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("third/priority") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("fourth/priority") != std::string::npos);
    }
    
    SECTION("-I combined with other options") {
        std::string libFile = TempFileHelper::createTempFile("module lib; endmodule");
        
        const char* args[] = {
            "sv2sc", 
            "-I", "./include",
            "-I", "./headers",
            "-v", libFile.c_str(),
            "-y", "/opt/libs",
            "-D", "INCLUDE_TEST=1",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(12, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryPaths.size() == 1);
        REQUIRE(arguments.defineMap.size() == 1);
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("headers") != std::string::npos);
        
        TempFileHelper::cleanup(libFile);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Include Directories (+incdir+)", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("+incdir+ single directory") {
        const char* args[] = {"sv2sc", "+incdir+/usr/include+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 1);
        REQUIRE(arguments.includePaths[0].find("/usr/include") != std::string::npos);
    }
    
    SECTION("+incdir+ multiple directories in single argument") {
        const char* args[] = {"sv2sc", "+incdir+./include+./rtl+./tb+/usr/local/include+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 4);
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("rtl") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("tb") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("/usr/local/include") != std::string::npos);
    }
    
    SECTION("+incdir+ multiple separate arguments") {
        const char* args[] = {
            "sv2sc", 
            "+incdir+./src/include+./src/headers+",
            "+incdir+./rtl/include+",
            "+incdir+/system/include+/vendor/include+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(5, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 5);
        REQUIRE(arguments.includePaths[0].find("src/include") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("src/headers") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("rtl/include") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("system/include") != std::string::npos);
        REQUIRE(arguments.includePaths[4].find("vendor/include") != std::string::npos);
    }
    
    SECTION("+incdir+ without trailing plus") {
        const char* args[] = {"sv2sc", "+incdir+./include", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 1);
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
    }
    
    SECTION("+incdir+ with empty paths (should be ignored)") {
        const char* args[] = {"sv2sc", "+incdir+./include++./headers+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        // Empty paths between '++' should be ignored
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("headers") != std::string::npos);
    }
    
    SECTION("+incdir+ paths with special characters") {
        std::filesystem::create_directories("/tmp/inc-with-dash");
        std::filesystem::create_directories("/tmp/inc_with_underscore");
        std::filesystem::create_directories("/tmp/inc.with.dots");
        
        const char* args[] = {
            "sv2sc", 
            "+incdir+/tmp/inc-with-dash+/tmp/inc_with_underscore+/tmp/inc.with.dots+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 3);
        REQUIRE(arguments.includePaths[0].find("inc-with-dash") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("inc_with_underscore") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("inc.with.dots") != std::string::npos);
        
        std::filesystem::remove_all("/tmp/inc-with-dash");
        std::filesystem::remove_all("/tmp/inc_with_underscore");
        std::filesystem::remove_all("/tmp/inc.with.dots");
    }
    
    SECTION("Mixed -I and +incdir+ with order preservation") {
        const char* args[] = {
            "sv2sc", 
            "-I", "/first/via/dash/I",
            "+incdir+./second/via/plus+./third/via/plus+",
            "-I", "/fourth/via/dash/I",
            "+incdir+./fifth/via/plus+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(8, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 5);
        // Order should be preserved across mixed syntax
        REQUIRE(arguments.includePaths[0].find("first/via/dash") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("second/via/plus") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("third/via/plus") != std::string::npos);
        REQUIRE(arguments.includePaths[3].find("fourth/via/dash") != std::string::npos);
        REQUIRE(arguments.includePaths[4].find("fifth/via/plus") != std::string::npos);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Preprocessor Defines", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-D macro without value") {
        const char* args[] = {"sv2sc", "-D", "DEBUG", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 1);
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
    }
    
    SECTION("-D macro with value") {
        const char* args[] = {"sv2sc", "-D", "WIDTH=32", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 1);
        REQUIRE(arguments.defineMap.at("WIDTH") == "32");
    }
    
    SECTION("+define+ single macro") {
        const char* args[] = {"sv2sc", "+define+DEBUG=1+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 1);
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
    }
    
    SECTION("+define+ multiple macros") {
        const char* args[] = {"sv2sc", "+define+WIDTH=32+HEIGHT=64+DEBUG+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 3);
        REQUIRE(arguments.defineMap.at("WIDTH") == "32");
        REQUIRE(arguments.defineMap.at("HEIGHT") == "64");
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
    }
    
    SECTION("Mixed -D and +define+") {
        const char* args[] = {"sv2sc", "-D", "SYNTHESIS", "+define+WIDTH=8+CLK_FREQ=100+", testFile.c_str()};
        
        REQUIRE(parser.parse(5, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 3);
        REQUIRE(arguments.defineMap.at("SYNTHESIS") == "1");
        REQUIRE(arguments.defineMap.at("WIDTH") == "8");
        REQUIRE(arguments.defineMap.at("CLK_FREQ") == "100");
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Undefines", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-U undefine") {
        const char* args[] = {"sv2sc", "-D", "TEST", "-U", "TEST", testFile.c_str()};
        
        REQUIRE(parser.parse(6, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.undefines.size() == 1);
        REQUIRE(arguments.undefines[0] == "TEST");
        REQUIRE(arguments.defineMap.find("TEST") == arguments.defineMap.end());
    }
    
    SECTION("+undefine+ format") {
        const char* args[] = {"sv2sc", "+define+TEST=1+", "+undefine+TEST+", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.undefines.size() == 1);
        REQUIRE(arguments.undefines[0] == "TEST");
        REQUIRE(arguments.defineMap.find("TEST") == arguments.defineMap.end());
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Language Standards", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-sverilog flag") {
        const char* args[] = {"sv2sc", "-sverilog", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.enableSystemVerilog == true);
    }
    
    SECTION("-v95 flag") {
        const char* args[] = {"sv2sc", "-v95", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.disableVerilog2001 == true);
    }
    
    SECTION("-extinclude flag") {
        const char* args[] = {"sv2sc", "-extinclude", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.extIncludeVersion == true);
    }
    
    SECTION("+systemverilogext+ extension") {
        const char* args[] = {"sv2sc", "+systemverilogext+.sv+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.systemVerilogExt == ".sv");
    }
    
    SECTION("+verilog2001ext+ extension") {
        const char* args[] = {"sv2sc", "+verilog2001ext+.v+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.verilog2001Ext == ".v");
    }
    
    SECTION("+verilog1995ext+ extension") {
        const char* args[] = {"sv2sc", "+verilog1995ext+.v95+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.verilog1995Ext == ".v95");
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Output Control", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("-o output name") {
        const char* args[] = {"sv2sc", "-o", "cpu_test", testFile.c_str()};
        
        REQUIRE(parser.parse(4, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.outputName == "cpu_test");
    }
    
    SECTION("-R run after compile") {
        const char* args[] = {"sv2sc", "-R", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.runAfterCompile == true);
    }
    
    SECTION("-full64 flag") {
        const char* args[] = {"sv2sc", "-full64", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.enable64Bit == true);
    }
    
    TempFileHelper::cleanup(testFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Complex Real-World Examples", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    std::string libFile = TempFileHelper::createTempFile("module lib; endmodule");
    
    SECTION("Complete VCS-style command line") {
        const char* args[] = {
            "sv2sc",
            "-sverilog",
            "+incdir+./rtl+./include+./tb+",
            "+define+WIDTH=32+CLK_FREQ=100+DEBUG=1+",
            "-v", libFile.c_str(),
            "-y", "/tmp",
            "+libext+.sv+.v+.vh+",
            "+liborder",
            "+libverbose",
            "-o", "cpu_sim",
            "-R",
            "-full64",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(16, args));
        auto& arguments = parser.getArguments();
        
        // Check all parsed values
        REQUIRE(arguments.enableSystemVerilog == true);
        REQUIRE(arguments.includePaths.size() == 3);
        REQUIRE(arguments.defineMap.size() == 3);
        REQUIRE(arguments.defineMap.at("WIDTH") == "32");
        REQUIRE(arguments.defineMap.at("CLK_FREQ") == "100");
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryPaths.size() == 1);
        REQUIRE(arguments.libraryExtensions.size() == 3);
        REQUIRE(arguments.libraryOrder == true);
        REQUIRE(arguments.libraryVerbose == true);
        REQUIRE(arguments.outputName == "cpu_sim");
        REQUIRE(arguments.runAfterCompile == true);
        REQUIRE(arguments.enable64Bit == true);
        REQUIRE(arguments.inputFiles.size() == 1);
    }
    
    SECTION("Mixed syntax compatibility") {
        const char* args[] = {
            "sv2sc",
            "-I", "./include",
            "+incdir+./rtl+",
            "-D", "SYNTHESIS=1",
            "+define+WIDTH=8+DEBUG+",
            "-v", libFile.c_str(),
            "+libext+.sv+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(11, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 2);
        REQUIRE(arguments.defineMap.size() == 3);
        REQUIRE(arguments.defineMap.at("SYNTHESIS") == "1");
        REQUIRE(arguments.defineMap.at("WIDTH") == "8");
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
        REQUIRE(arguments.libraryFiles.size() == 1);
        REQUIRE(arguments.libraryExtensions.size() == 1);
    }
    
    TempFileHelper::cleanup(testFile);
    TempFileHelper::cleanup(libFile);
}

TEST_CASE("VCS Args Parser - Phase 1: Error Handling", "[vcs][phase1]") {
    VCSArgsParser parser;
    
    SECTION("Missing input files") {
        const char* args[] = {"sv2sc", "-sverilog"};
        
        REQUIRE_FALSE(parser.parse(2, args));
    }
    
    SECTION("Non-existent input file") {
        const char* args[] = {"sv2sc", "/non/existent/file.sv"};
        
        REQUIRE_FALSE(parser.parse(2, args));
    }
    
    SECTION("Unknown dash argument") {
        std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
        const char* args[] = {"sv2sc", "-unknown", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(3, args));
        
        TempFileHelper::cleanup(testFile);
    }
    
    SECTION("Unknown plus argument") {
        std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
        const char* args[] = {"sv2sc", "+unknown+arg+", testFile.c_str()};
        
        REQUIRE_FALSE(parser.parse(3, args));
        
        TempFileHelper::cleanup(testFile);
    }
}

TEST_CASE("VCS Args Parser - Phase 1: Plus Argument Parsing Edge Cases", "[vcs][phase1]") {
    VCSArgsParser parser;
    std::string testFile = TempFileHelper::createTempFile("module test; endmodule");
    
    SECTION("Plus argument without values") {
        const char* args[] = {"sv2sc", "+liborder", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.libraryOrder == true);
    }
    
    SECTION("Plus argument with empty values") {
        const char* args[] = {"sv2sc", "+incdir++", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        // Should not add empty paths
        REQUIRE(arguments.includePaths.empty());
    }
    
    SECTION("Plus argument with single value") {
        const char* args[] = {"sv2sc", "+define+DEBUG+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 1);
        REQUIRE(arguments.defineMap.at("DEBUG") == "1");
    }
    
    SECTION("Plus argument without trailing plus") {
        const char* args[] = {"sv2sc", "+incdir+./include", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.includePaths.size() == 1);
        REQUIRE(arguments.includePaths[0].find("include") != std::string::npos);
    }
    
    SECTION("Plus argument with consecutive plus signs") {
        const char* args[] = {"sv2sc", "+incdir+./dir1++./dir2+++./dir3+", testFile.c_str()};
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        // Empty values between '++' and '+++' should be ignored
        REQUIRE(arguments.includePaths.size() == 3);
        REQUIRE(arguments.includePaths[0].find("dir1") != std::string::npos);
        REQUIRE(arguments.includePaths[1].find("dir2") != std::string::npos);
        REQUIRE(arguments.includePaths[2].find("dir3") != std::string::npos);
    }
    
    SECTION("Plus argument with complex macro values") {
        const char* args[] = {
            "sv2sc", 
            "+define+COMPLEX_MACRO=`ifdef DEBUG ? 1 : 0`+ANOTHER_MACRO=2**WIDTH+",
            testFile.c_str()
        };
        
        REQUIRE(parser.parse(3, args));
        auto& arguments = parser.getArguments();
        
        REQUIRE(arguments.defineMap.size() == 2);
        REQUIRE(arguments.defineMap.at("COMPLEX_MACRO") == "`ifdef DEBUG ? 1 : 0`");
        REQUIRE(arguments.defineMap.at("ANOTHER_MACRO") == "2**WIDTH");
    }
    
    TempFileHelper::cleanup(testFile);
}