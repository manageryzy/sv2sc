#include "sv2sc/sv2sc.h"
#include "translator/vcs_args_parser.h"
#include "utils/logger.h"
#include <iostream>
#include <cstdlib>

using namespace sv2sc;

// Convert VCS arguments to translation options
TranslationOptions convertVCSToTranslationOptions(const translator::VCSArguments& args) {
    TranslationOptions options;
    
    // Core file processing
    options.inputFiles = args.inputFiles;
    options.outputDir = args.outputDir;
    options.topModule = args.topModule;
    
    // SystemC generation options
    options.generateTestbench = args.generateTestbench;
    options.clockSignal = args.clockSignal;
    options.resetSignal = args.resetSignal;
    options.enableSynthesis = args.enableSynthesis;
    
    // Language and compilation options  
    options.enableSystemVerilog = args.enableSystemVerilog;
    options.timescale = args.timescale;
    options.enable64Bit = args.enable64Bit;
    
    // Preprocessor and include handling
    options.includePaths = args.includePaths;
    options.defineMap = args.defineMap;
    options.undefines = args.undefines;
    
    // Library and dependency management
    options.libraryPaths = args.libraryPaths;
    options.libraryFiles = args.libraryFiles;
    options.fileExtensions = args.libraryExtensions;
    
    // Debug and output control
    options.enableDebug = args.enableDebug;
    options.enableVerbose = args.enableVerbose;
    options.outputName = args.outputName;
    
    // Legacy defines (for backward compatibility)
    options.defines = args.defines;
    
    return options;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        translator::VCSArgsParser parser;
        if (!parser.parse(argc, argv)) {
            return EXIT_FAILURE;
        }
        
        const auto& args = parser.getArguments();
        
        // Convert VCS arguments to translation options
        TranslationOptions options = convertVCSToTranslationOptions(args);
        
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