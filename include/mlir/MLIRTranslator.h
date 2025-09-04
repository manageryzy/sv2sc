#pragma once

#include "sv2sc/sv2sc.h"
#include "mlir/MLIRContextManager.h"
#include "mlir/SVToHWBuilder.h"
#include "mlir/pipeline/SV2SCPassPipeline.h"
#include <memory>
#include <vector>
#include <string>
#include <fstream>

// MLIR includes - we always have real CIRCT
#include "mlir/IR/BuiltinOps.h"

namespace sv2sc::mlir_support {

/**
 * @brief MLIR-based SystemVerilog to SystemC translator
 * 
 * This class implements an alternative translation pipeline using MLIR/CIRCT
 * infrastructure. It follows the same interface as the existing translator
 * but uses a multi-level IR approach for better optimization and analysis.
 */
class MLIRSystemVerilogToSystemCTranslator {
public:
    explicit MLIRSystemVerilogToSystemCTranslator(const TranslationOptions& options);
    ~MLIRSystemVerilogToSystemCTranslator() = default;

    // Non-copyable
    MLIRSystemVerilogToSystemCTranslator(const MLIRSystemVerilogToSystemCTranslator&) = delete;
    MLIRSystemVerilogToSystemCTranslator& operator=(const MLIRSystemVerilogToSystemCTranslator&) = delete;

    /**
     * @brief Run the MLIR-based translation pipeline
     * @return true if translation succeeded, false otherwise
     */
    bool translate();

    /**
     * @brief Get translation errors
     * @return Vector of error messages
     */
    const std::vector<std::string>& getErrors() const { return errors_; }

    /**
     * @brief Get translation warnings  
     * @return Vector of warning messages
     */
    const std::vector<std::string>& getWarnings() const { return warnings_; }

private:
    TranslationOptions options_;
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;
    
    std::unique_ptr<MLIRContextManager> contextManager_;
    std::unique_ptr<SVToHWBuilder> hwBuilder_;
    std::unique_ptr<SV2SCPassPipeline> passPipeline_;
    
    // Translation pipeline methods
    bool initializeMLIRInfrastructure();
    bool processDesignWithMLIR();
    bool processFileWithMLIR(const std::string& inputFile);
    bool runMLIRPasses(mlir::ModuleOp module);
    bool emitSystemCFromMLIR(mlir::ModuleOp module);
    
    // Error handling
    void addError(const std::string& error);
    void addWarning(const std::string& warning);
    
    // SystemC generation fallback
    bool generateSystemCFromModule(mlir::ModuleOp module);
    
    // Pipeline phases
    bool runHWDialectGeneration();
    bool runAnalysisPasses();
    bool runTransformationPasses(); 
    bool runSystemCEmission();
};

/**
 * @brief Check if MLIR support is available and properly configured
 * @return true if MLIR/CIRCT libraries are available
 */
bool isMLIRSupportAvailable();

/**
 * @brief Get MLIR version information
 * @return Version string for logging/diagnostics
 */
std::string getMLIRVersionInfo();

} // namespace sv2sc::mlir_support

