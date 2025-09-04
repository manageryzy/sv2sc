#pragma once

#include <memory>

// MLIR includes - we always have real CIRCT
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"

namespace sv2sc::mlir_support {

/**
 * @brief Pass pipeline manager for sv2sc MLIR-based translation
 * 
 * This class manages the MLIR pass pipeline for transforming SystemVerilog
 * designs through multiple levels of IR (HW dialect -> SystemC dialect).
 */
class SV2SCPassPipeline {
public:
    SV2SCPassPipeline();
    ~SV2SCPassPipeline() = default;

    // Non-copyable
    SV2SCPassPipeline(const SV2SCPassPipeline&) = delete;
    SV2SCPassPipeline& operator=(const SV2SCPassPipeline&) = delete;

    /**
     * @brief Build the complete sv2sc pass pipeline
     * @param pm Pass manager to configure
     * @param optimizationLevel Optimization level (0-3)
     */
    void buildPipeline(mlir::OpPassManager& pm, int optimizationLevel = 1);

    /**
     * @brief Build analysis-only passes
     * @param pm Pass manager to configure
     */
    void buildAnalysisPasses(mlir::OpPassManager& pm);

    /**
     * @brief Build transformation passes
     * @param pm Pass manager to configure
     * @param optimizationLevel Optimization level (0-3)
     */
    void buildTransformationPasses(mlir::OpPassManager& pm, int optimizationLevel = 1);

    /**
     * @brief Build lowering passes (HW -> SystemC)
     * @param pm Pass manager to configure
     */
    void buildLoweringPasses(mlir::OpPassManager& pm);

    /**
     * @brief Build standard MLIR optimization passes
     * @param pm Pass manager to configure
     * @param optimizationLevel Optimization level (0-3)
     */
    void buildStandardOptimizations(mlir::OpPassManager& pm, int optimizationLevel = 1);

    /**
     * @brief Run the complete pipeline on a module
     * @param module MLIR module to process
     * @param optimizationLevel Optimization level (0-3)
     * @return true if pipeline succeeded
     */
    bool runPipeline(mlir::ModuleOp module, int optimizationLevel = 1);

    /**
     * @brief Enable pass timing statistics
     * @param enable Whether to enable timing
     */
    void enableTiming(bool enable = true) { enableTiming_ = enable; }

    /**
     * @brief Enable pass failure diagnostics
     * @param enable Whether to enable diagnostics
     */
    void enableDiagnostics(bool enable = true) { enableDiagnostics_ = enable; }

    /**
     * @brief Enable IR dumping for debugging
     * @param enable Whether to enable IR dumping
     */
    void enableIRDumping(bool enable = true) { enableIRDumping_ = enable; }

private:
    bool enableTiming_ = false;
    bool enableDiagnostics_ = true;
    bool enableIRDumping_ = false;
    
    void configurePipeline(mlir::PassManager& pm);
};

} // namespace sv2sc::mlir_support

