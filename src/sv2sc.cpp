#include "sv2sc/sv2sc.h"
#include "core/ast_visitor.h"
#include "codegen/systemc_generator.h"
#include "utils/logger.h"
#include <slang/syntax/SyntaxTree.h>
#include <slang/text/SourceManager.h>
#include <slang/ast/Compilation.h>
#include <filesystem>
#include <fstream>

namespace sv2sc {

class SystemVerilogToSystemCTranslator::Impl {
public:
    explicit Impl(const TranslationOptions& options) : options_(options) {
        setupLogger();
    }
    
    bool translate() {
        try {
            LOG_INFO("Starting SystemVerilog to SystemC translation");
            
            // Process each input file
            for (const auto& inputFile : options_.inputFiles) {
                if (!processFile(inputFile)) {
                    return false;
                }
            }
            
            LOG_INFO("Translation completed successfully");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Translation failed with exception: {}", e.what());
            errors_.push_back(fmt::format("Exception: {}", e.what()));
            return false;
        }
    }
    
    const std::vector<std::string>& getErrors() const { return errors_; }
    const std::vector<std::string>& getWarnings() const { return warnings_; }

private:
    TranslationOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;
    
    void setupLogger() {
        auto& logger = utils::Logger::getInstance();
        
        if (options_.enableDebug) {
            logger.setLevel(spdlog::level::debug);
        } else if (options_.enableVerbose) {
            logger.setLevel(spdlog::level::info);
        } else {
            logger.setLevel(spdlog::level::warn);
        }
        
        // Enable file logging in output directory
        std::filesystem::create_directories(options_.outputDir);
        std::string logFile = options_.outputDir + "/sv2sc.log";
        logger.enableFileLogging(logFile);
    }
    
    bool processFile(const std::string& inputFile) {
        LOG_INFO("Processing file: {}", inputFile);
        
        try {
            // Create source manager
            slang::SourceManager sourceManager;
            
            // Load the file
            std::ifstream file(inputFile);
            if (!file.is_open()) {
                std::string error = fmt::format("Cannot open input file: {}", inputFile);
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            std::string content((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            file.close();
            
            // Create syntax tree
            auto syntaxTree = slang::syntax::SyntaxTree::fromText(content);
            
            // Check for parsing errors
            if (syntaxTree->diagnostics().size() > 0) {
                LOG_WARN("Parsing diagnostics found:");
                for (const auto& diag : syntaxTree->diagnostics()) {
                    std::string diagMsg = fmt::format("{}:{} - {}", 
                        inputFile, diag.location.offset(), "Parse diagnostic");
                    LOG_WARN(diagMsg);
                    warnings_.push_back(diagMsg);
                }
            }
            
            // Create compilation from syntax tree
            slang::ast::Compilation compilation;
            compilation.addSyntaxTree(syntaxTree);
            
            // Create code generator
            codegen::SystemCCodeGenerator generator;
            
            // Create visitor and process the compilation units
            core::SVToSCVisitor visitor(generator);
            
            // Visit the root of the compilation
            auto& root = compilation.getRoot();
            visitor.visit(root);
            
            // Generate output files
            std::string baseName = std::filesystem::path(inputFile).stem();
            std::string headerPath = options_.outputDir + "/" + baseName + ".h";
            std::string implPath = options_.outputDir + "/" + baseName + ".cpp";
            
            if (!generator.writeToFile(headerPath, implPath)) {
                std::string error = fmt::format("Failed to write output files for: {}", inputFile);
                LOG_ERROR(error);
                errors_.push_back(error);
                return false;
            }
            
            LOG_INFO("Successfully translated {} to SystemC", inputFile);
            return true;
            
        } catch (const std::exception& e) {
            std::string error = fmt::format("Error processing {}: {}", inputFile, e.what());
            LOG_ERROR(error);
            errors_.push_back(error);
            return false;
        }
    }
};

SystemVerilogToSystemCTranslator::SystemVerilogToSystemCTranslator(const TranslationOptions& options)
    : pImpl(std::make_unique<Impl>(options)) {
}

SystemVerilogToSystemCTranslator::~SystemVerilogToSystemCTranslator() = default;

bool SystemVerilogToSystemCTranslator::translate() {
    return pImpl->translate();
}

const std::vector<std::string>& SystemVerilogToSystemCTranslator::getErrors() const {
    return pImpl->getErrors();
}

const std::vector<std::string>& SystemVerilogToSystemCTranslator::getWarnings() const {
    return pImpl->getWarnings();
}

} // namespace sv2sc