#pragma once

#include <memory>
#include <string>

// MLIR includes - we always have real CIRCT
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace sv2sc::mlir_support {

/**
 * @brief Manages MLIR context and dialect loading for sv2sc
 * 
 * This class encapsulates MLIR context initialization and provides
 * a centralized point for managing MLIR-related infrastructure.
 */
class MLIRContextManager {
public:
    MLIRContextManager();
    ~MLIRContextManager() = default;

    // Non-copyable, non-movable for now
    MLIRContextManager(const MLIRContextManager&) = delete;
    MLIRContextManager& operator=(const MLIRContextManager&) = delete;
    MLIRContextManager(MLIRContextManager&&) = delete;
    MLIRContextManager& operator=(MLIRContextManager&&) = delete;

    /**
     * @brief Get the MLIR context
     * @return Reference to the managed MLIR context
     */
    mlir::MLIRContext& getContext();

    /**
     * @brief Load all required dialects for sv2sc
     */
    void loadRequiredDialects();

    /**
     * @brief Check if MLIR support is properly initialized
     * @return true if MLIR context and dialects are ready
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Create an empty MLIR module
     * @param name Module name (optional)
     * @return OwningOpRef to the created module
     */
    mlir::OwningOpRef<mlir::ModuleOp> createModule(const std::string& name = "sv2sc_module");

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    bool initialized_ = false;

    void initializeContext();
};

} // namespace sv2sc::mlir_support