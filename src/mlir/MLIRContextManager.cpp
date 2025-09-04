#include "mlir/MLIRContextManager.h"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h" 
#include "mlir/IR/BuiltinOps.h"

// Standard MLIR dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

// CIRCT dialects
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"

// Logging
#include "utils/logger.h"

namespace sv2sc::mlir_support {

MLIRContextManager::MLIRContextManager() {
    initializeContext();
}

mlir::MLIRContext& MLIRContextManager::getContext() {
    if (!context_) {
        initializeContext();
    }
    return *context_;
}

void MLIRContextManager::loadRequiredDialects() {
    LOG_DEBUG("Loading required MLIR dialects for sv2sc");
    
    auto& context = getContext();
    
    // Load standard MLIR dialects
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::emitc::EmitCDialect>();
    
    // Load CIRCT dialects
    context.loadDialect<circt::hw::HWDialect>();
    context.loadDialect<circt::seq::SeqDialect>();
    context.loadDialect<circt::comb::CombDialect>();
    context.loadDialect<circt::sv::SVDialect>();
    context.loadDialect<circt::systemc::SystemCDialect>();
    
    LOG_INFO("Successfully loaded MLIR dialects: Arith, Func, SCF, EmitC, HW, Seq, Comb, SV, SystemC");
    initialized_ = true;
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRContextManager::createModule(const std::string& name) {
    if (!initialized_) {
        loadRequiredDialects();
    }
    
    auto& context = getContext();
    mlir::OpBuilder builder(&context);
    
    // Create module at unknown location for now
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<mlir::ModuleOp>(loc);
    
    // Set module name if provided
    if (!name.empty()) {
        module->setAttr("sym_name", builder.getStringAttr(name));
    }
    
    LOG_DEBUG("Created MLIR module: {}", name);
    return module;
}

void MLIRContextManager::initializeContext() {
    LOG_DEBUG("Initializing MLIR context for sv2sc");
    
    context_ = std::make_unique<mlir::MLIRContext>();
    
    // Disable multi-threading to allow IR printing/instrumentation during pass pipelines
    context_->disableMultithreading();
    
    // Load dialects immediately
    loadRequiredDialects();
    
    LOG_INFO("MLIR context initialized successfully");
}

} // namespace sv2sc::mlir_support