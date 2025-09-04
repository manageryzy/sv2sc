#include "sv2sc/sv2sc.h"
#include "translator/vcs_args_parser.h"
#include "utils/logger.h"
#include "utils/error_reporter.h"
#include "utils/performance_profiler.h"
#include <iostream>
#include <iomanip>
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
    
    // MLIR pipeline options
    options.useMLIRPipeline = args.useMLIRPipeline;
    options.enableMLIRDiagnostics = args.enableMLIRDiagnostics;
    options.dumpMLIR = args.dumpMLIR;
    options.optimizationLevel = args.optimizationLevel;
    
    // Legacy defines (for backward compatibility)
    options.defines = args.defines;
    
    return options;
}

int main(int argc, char* argv[]) {
    try {
        PROFILE_SCOPE("Total Translation Time");
        
        // Parse command line arguments
        translator::VCSArgsParser parser;
        TranslationOptions options;
        {
            PROFILE_SCOPE("Command Line Parsing");
            if (!parser.parse(argc, argv)) {
                return EXIT_FAILURE;
            }
            
            const auto& args = parser.getArguments();
            options = convertVCSToTranslationOptions(args);
        }
        
        // Create and run translator
        SystemVerilogToSystemCTranslator translator(options);
        
        LOG_INFO("sv2sc - SystemVerilog to SystemC Translator v1.0.0");
        LOG_INFO("Processing {} input file(s)", options.inputFiles.size());
        
        bool success;
        {
            PROFILE_SCOPE("SystemVerilog Translation");
            success = translator.translate();
        }
        
        // Get enhanced error reporting
        auto& errorReporter = utils::getGlobalErrorReporter();
        
        // Report results using enhanced error reporter
        const auto& errors = translator.getErrors();
        const auto& warnings = translator.getWarnings();
        
        // Add legacy errors/warnings to the error reporter for consistent formatting
        for (const auto& warning : warnings) {
            errorReporter.warning(warning);
        }
        
        for (const auto& error : errors) {
            errorReporter.error(error);
        }
        
        // Print all diagnostics with enhanced formatting
        if (errorReporter.getDiagnosticCount() > 0) {
            std::cout << "\n=== Translation Diagnostics ===\n";
            errorReporter.printDiagnostics();
            std::cout << "\n" << errorReporter.getSummary() << "\n";
        }
        
        // Print performance report if enabled
        auto& profiler = utils::getGlobalProfiler();
        if (profiler.isEnabled() && options.enableVerbose) {
            profiler.printSummary();
        }
        
        if (success && !errorReporter.hasErrors()) {
            std::cout << "\nTranslation completed successfully!\n";
            std::cout << "Output files generated in: " << options.outputDir << "\n";
            
            // Show brief performance info
            if (profiler.isEnabled()) {
                double totalTime = profiler.getTotalTime();
                if (totalTime > 0) {
                    std::cout << "Translation time: " << std::fixed << std::setprecision(2) 
                             << totalTime << "ms\n";
                }
            }
            
            return EXIT_SUCCESS;
        } else {
            if (errorReporter.hasErrors()) {
                std::cout << "\nTranslation failed due to errors. Please fix the issues above.\n";
            } else {
                std::cout << "\nTranslation failed with " << errors.size() << " error(s)\n";
            }
            return EXIT_FAILURE;
        }
        
    } catch (const std::exception& e) {
        auto& errorReporter = utils::getGlobalErrorReporter();
        errorReporter.fatal("Unhandled exception occurred", {}, 
                           "This is likely a bug in the translator. Please report this issue.");
        errorReporter.error(e.what());
        errorReporter.printDiagnostics();
        return EXIT_FAILURE;
    } catch (...) {
        auto& errorReporter = utils::getGlobalErrorReporter();
        errorReporter.fatal("Unknown fatal error occurred", {}, 
                           "This is likely a serious bug. Please report this issue with your input files.");
        errorReporter.printDiagnostics();
        return EXIT_FAILURE;
    }
}