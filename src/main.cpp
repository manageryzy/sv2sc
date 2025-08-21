#include "sv2sc/sv2sc.h"
#include "translator/vcs_args_parser.h"
#include "utils/logger.h"
#include <iostream>
#include <cstdlib>

using namespace sv2sc;

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        translator::VCSArgsParser parser;
        if (!parser.parse(argc, argv)) {
            return EXIT_FAILURE;
        }
        
        const auto& args = parser.getArguments();
        
        // Convert VCS arguments to translation options
        TranslationOptions options;
        options.inputFiles = args.inputFiles;
        options.outputDir = args.outputDir;
        options.topModule = args.topModule;
        options.generateTestbench = args.generateTestbench;
        options.enableDebug = args.enableDebug;
        options.enableVerbose = args.enableVerbose;
        options.includePaths = args.includePaths;
        options.defines = args.defines;
        options.libraryPaths = args.libraryPaths;
        
        // Create and run translator
        SystemVerilogToSystemCTranslator translator(options);
        
        LOG_INFO("sv2sc - SystemVerilog to SystemC Translator v1.0.0");
        LOG_INFO("Processing {} input file(s)", options.inputFiles.size());
        
        bool success = translator.translate();
        
        // Report results
        const auto& errors = translator.getErrors();
        const auto& warnings = translator.getWarnings();
        
        if (!warnings.empty()) {
            std::cout << "\nWarnings (" << warnings.size() << "):\n";
            for (const auto& warning : warnings) {
                std::cout << "  " << warning << "\n";
            }
        }
        
        if (!errors.empty()) {
            std::cout << "\nErrors (" << errors.size() << "):\n";
            for (const auto& error : errors) {
                std::cout << "  " << error << "\n";
            }
        }
        
        if (success) {
            std::cout << "\nTranslation completed successfully!\n";
            std::cout << "Output files generated in: " << options.outputDir << "\n";
            return EXIT_SUCCESS;
        } else {
            std::cout << "\nTranslation failed with " << errors.size() << " error(s)\n";
            return EXIT_FAILURE;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return EXIT_FAILURE;
    }
}