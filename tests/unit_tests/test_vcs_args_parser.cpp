#include <catch2/catch_test_macros.hpp>
#include "translator/vcs_args_parser.h"

using namespace sv2sc::translator;

TEST_CASE("VCS Args Parser Basic Functionality", "[vcs_args_parser]") {
    VCSArgsParser parser;
    
    SECTION("Default arguments") {
        const auto& args = parser.getArguments();
        REQUIRE(args.outputDir == "./output");
        REQUIRE(args.clockSignal == "clk");
        REQUIRE(args.resetSignal == "reset");
        REQUIRE(args.enableElaboration == true);
        REQUIRE(args.enableSynthesis == false);
    }
    
    SECTION("Argument structure validation") {
        const auto& args = parser.getArguments();
        // Test that the structure is properly initialized
        REQUIRE(args.inputFiles.empty());
        REQUIRE(args.topModule.empty());
        REQUIRE(args.includePaths.empty());
        REQUIRE(args.libraryPaths.empty());
        REQUIRE(args.defines.empty());
        REQUIRE(!args.timescale.empty() == false); // Should be empty initially
    }
}