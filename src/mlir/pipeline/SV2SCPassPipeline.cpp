#include "mlir/pipeline/SV2SCPassPipeline.h"

// MLIR includes
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

// CIRCT includes
#include "circt/Conversion/HWToSystemC.h"

// sv2sc MLIR passes
#include "mlir/passes/HWToSystemCLoweringPass.h"

// System includes for stack size detection
#include <sys/resource.h>
#include <cstdlib>

// Logging
#include "utils/logger.h"

namespace sv2sc::mlir_support {

// Check current stack size limit and determine if we can handle complex designs
static bool hasAdequateStackSize() {
    struct rlimit stackLimit;
    if (getrlimit(RLIMIT_STACK, &stackLimit) != 0) {
        LOG_WARN("Could not determine stack size limit, assuming conservative limits");
        return false;
    }
    
    // Convert to MB for easier logging
    size_t stackSizeMB = stackLimit.rlim_cur / (1024 * 1024);
    
    // We need at least 8MB stack for complex MLIR processing
    // Most systems default to 8MB, but some may have less
    const size_t MIN_STACK_MB = 8;
    
    if (stackLimit.rlim_cur == RLIM_INFINITY) {
        LOG_DEBUG("Unlimited stack size detected - can handle any design complexity");
        return true;
    } else if (stackSizeMB >= MIN_STACK_MB) {
        LOG_DEBUG("Adequate stack size detected: {}MB - can handle complex designs", stackSizeMB);
        return true;
    } else {
        LOG_WARN("Limited stack size detected: {}MB (minimum {}MB recommended)", 
                stackSizeMB, MIN_STACK_MB);
        LOG_WARN("Consider running: ulimit -s {} before translation", MIN_STACK_MB * 1024);
        return false;
    }
}

SV2SCPassPipeline::SV2SCPassPipeline() {
    LOG_DEBUG("Initializing sv2sc MLIR pass pipeline");
}

void SV2SCPassPipeline::buildPipeline(mlir::OpPassManager& pm, int optimizationLevel) {
    LOG_INFO("Building sv2sc MLIR pass pipeline with optimization level: {}", optimizationLevel);
    
    // Phase 1: Analysis passes
    buildAnalysisPasses(pm);
    
    // Phase 2: Standard optimizations
    if (optimizationLevel > 0) {
        buildStandardOptimizations(pm, optimizationLevel);
    }
    
    // Phase 3: Custom transformation passes
    buildTransformationPasses(pm, optimizationLevel);
    
    // Phase 4: Lowering passes (HW -> SystemC)
    buildLoweringPasses(pm);
    
    // Phase 5: Final cleanup
    if (optimizationLevel > 0) {
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
    }
    
    LOG_DEBUG("sv2sc MLIR pass pipeline configured with {} phases", 5);
}

void SV2SCPassPipeline::buildAnalysisPasses(mlir::OpPassManager& pm) {
    LOG_DEBUG("Building analysis passes");
    
    // TODO: Add custom analysis passes
    // - SystemC type analysis pass
    // - Dependency analysis pass
    // - Sensitivity analysis pass
    // - Clock domain analysis pass
    
    LOG_DEBUG("Analysis passes configured");
}

void SV2SCPassPipeline::buildTransformationPasses(mlir::OpPassManager& pm, int optimizationLevel) {
    LOG_DEBUG("Building transformation passes with optimization level: {}", optimizationLevel);
    
    // TODO: Add custom transformation passes
    // - Process optimization pass
    // - Signal optimization pass
    // - Expression simplification pass
    
    LOG_DEBUG("Transformation passes configured");
}

void SV2SCPassPipeline::buildLoweringPasses(mlir::OpPassManager& pm) {
    LOG_DEBUG("Building lowering passes (HW -> SystemC)");

    // Use CIRCT's HW->SystemC conversion with IR dumping for debugging
    // Add IR dump before CIRCT conversion
    pm.addPass(createDebugIRDumpPass("1-before-circt-conversion"));
    
    pm.addPass(circt::createConvertHWToSystemCPass());
    LOG_DEBUG("Added CIRCT ConvertHWToSystemCPass");
    
    // Add IR dump after CIRCT conversion to see the invalid IR
    pm.addPass(createDebugIRDumpPass("2-after-circt-conversion"));
    
    // Add custom pass immediately to fix invalid signal reads from output ports
    pm.addPass(createFixSystemCSignalReadPass());
    LOG_DEBUG("Added FixSystemCSignalReadPass to fix invalid signal reads");
    
    // Add IR dump after our fix pass
    pm.addPass(createDebugIRDumpPass("3-after-signal-read-fix"));
    
    // Add custom pass to fold systemc.convert operations (CIRCT ExportSystemC doesn't handle them)
    pm.addPass(createFoldSystemCConvertPass());
    LOG_DEBUG("Added FoldSystemCConvertPass to fold systemc.convert operations");
    
    // Add IR dump after convert folding
    pm.addPass(createDebugIRDumpPass("4-after-convert-folding"));
    
    // Basic cleanup passes
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    LOG_DEBUG("Added cleanup passes");
    
    // Add final IR dump after all fixes
    pm.addPass(createDebugIRDumpPass("5-final-cleaned-ir"));
    
    // Re-enable verification for final check (after our fixes)
    // Note: This doesn't work with the current MLIR API, so we'll do manual verification in runPipeline()
    LOG_DEBUG("Will perform manual verification after pipeline completion");

    LOG_DEBUG("Lowering passes configured");
}

void SV2SCPassPipeline::buildStandardOptimizations(mlir::OpPassManager& pm, int optimizationLevel) {
    LOG_DEBUG("Building standard MLIR optimizations with level: {}", optimizationLevel);
    
    // Level 0: No optimizations
    if (optimizationLevel == 0) {
        return;
    }
    
    // Level 1: Basic optimizations
    if (optimizationLevel >= 1) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
    }
    
    // Level 2: More aggressive optimizations
    if (optimizationLevel >= 2) {
        pm.addPass(mlir::createInlinerPass());
    }
    
    // Level 3: Maximum optimizations
    if (optimizationLevel >= 3) {
        // Add more aggressive optimization passes
    }
    
    LOG_DEBUG("Standard optimizations configured for level: {}", optimizationLevel);
}

bool SV2SCPassPipeline::runPipeline(mlir::ModuleOp module, int optimizationLevel) {
    LOG_INFO("Running sv2sc MLIR pass pipeline on module");
    
    if (!module) {
        LOG_ERROR("Invalid MLIR module provided to pass pipeline");
        return false;
    }
    
    // Check stack size instead of artificially limiting design complexity
    bool stackOk = hasAdequateStackSize();
    
    // Count operations for informational purposes only
    size_t opCount = 0;
    try {
        module.walk([&opCount](mlir::Operation* op) {
            opCount++;
            return mlir::WalkResult::advance();
        });
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during module walk: {}", e.what());
        return false;
    } catch (...) {
        LOG_ERROR("Unknown exception during module walk");
        return false;
    }
    
    LOG_INFO("Module contains {} MLIR operations", opCount);
    
    // Warn about potential stack issues for very complex designs
    if (!stackOk && opCount > 50000) {
        LOG_WARN("Large design ({} operations) with limited stack size may cause issues", opCount);
        LOG_WARN("If translation fails with stack overflow, increase stack size:");
        LOG_WARN("  ulimit -s 16384  # Set 16MB stack");
        LOG_WARN("Then retry the translation");
    }
    
    // Allow user to override stack safety check if they know what they're doing
    bool forceRun = false;
    if (const char* forceEnv = std::getenv("SV2SC_FORCE_MLIR_PASSES")) {
        forceRun = (std::string(forceEnv) == "1" || std::string(forceEnv) == "true");
        if (forceRun) {
            LOG_WARN("Forcing MLIR passes despite stack size concerns (SV2SC_FORCE_MLIR_PASSES={})", forceEnv);
        }
    }
    
    if (!stackOk && !forceRun && opCount > 50000) {
        LOG_WARN("Skipping MLIR passes due to stack size limitations");
        LOG_INFO("Set SV2SC_FORCE_MLIR_PASSES=1 to override (may cause crashes)");
        return true; // Return success but skip passes
    }
    
    // Create pass manager
    mlir::PassManager pm(module.getContext());
    
    // Configure the pass manager
    configurePipeline(pm);
    
    // Build the pipeline
    buildPipeline(pm, optimizationLevel);
    
    // Run the pipeline
    LOG_DEBUG("Executing pass pipeline...");
    
    try {
        if (mlir::failed(pm.run(module))) {
            LOG_ERROR("sv2sc MLIR pass pipeline failed");
            return false;
        }
        
        // Perform manual verification after our fix passes have run
        LOG_DEBUG("Performing manual verification of fixed IR");
        try {
            if (mlir::failed(mlir::verify(module))) {
                LOG_ERROR("Module verification failed after CIRCT bug fixes - this indicates our fix didn't work");
                return false;
            } else {
                LOG_DEBUG("Manual verification passed - CIRCT bug successfully worked around");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception during manual verification: {}", e.what());
            return false;
        }
        
        LOG_INFO("sv2sc MLIR pass pipeline completed successfully with CIRCT bug workaround");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during pass pipeline execution: {}", e.what());
        return false;
    }
}

void SV2SCPassPipeline::configurePipeline(mlir::PassManager& pm) {
    LOG_DEBUG("Configuring pass manager");
    
    // Enable timing if requested
    if (enableTiming_) {
        pm.enableTiming();
        LOG_DEBUG("Pass timing enabled");
    }
    
    // Configure diagnostics - disable verification temporarily for CIRCT bug workaround
    if (enableDiagnostics_) {
        // Temporarily disable verification to allow CIRCT to generate invalid IR
        // that we'll fix with our custom FixSystemCSignalReadPass
        pm.enableVerifier(false);
        LOG_DEBUG("Pass verification disabled temporarily to work around CIRCT HW-to-SystemC bug");
    }
    
    // Configure IR dumping if requested
    if (enableIRDumping_) {
        pm.enableIRPrinting();
        LOG_DEBUG("IR printing enabled");
    }
    
    LOG_DEBUG("Pass manager configured");
}

} // namespace sv2sc::mlir_support