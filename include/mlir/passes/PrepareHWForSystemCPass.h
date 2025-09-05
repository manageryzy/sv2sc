#pragma once

// MLIR includes
#include "mlir/Pass/Pass.h"

// CIRCT includes  
#include "circt/Dialect/HW/HWOps.h"

namespace sv2sc::mlir_support {

/// PrepareHWForSystemCPass - Phase 3 CIRCT Bug Workaround Pass
/// 
/// This pass works around the CIRCT HWToSystemC conversion bug that creates
/// invalid systemc.signal.read operations for output ports.
///
/// CIRCT Bug Location: third-party/circt/lib/Conversion/HWToSystemC/HWToSystemC.cpp:101-109
/// Bug: Creates signal reads for ALL module arguments including outputs:
///   for (size_t i = 0, e = scFunc.getRegion().getNumArguments(); i < e; ++i) {
///     auto inputRead = SignalReadOp::create(rewriter, scFunc.getLoc(),
///                                          scModule.getArgument(i))  // BUG: should check !isa<OutputType>
///
/// Error: 'systemc.signal.read' op operand #0 must be a SystemC sc_in<T> type or a SystemC sc_inout<T> type 
///        or a SystemC sc_signal<T> type, but got '!systemc.out<!systemc.uint<8>>'
///
/// Workaround Strategy:
/// 1. Pre-process HW modules before CIRCT conversion
/// 2. Add markers/attributes to identify problematic patterns  
/// 3. Coordinate with FixSystemCSignalReadPass for post-processing
/// 4. Preserve seq.compreg operations from Phase 2
class PrepareHWForSystemCPass : public mlir::PassWrapper<PrepareHWForSystemCPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    PrepareHWForSystemCPass() = default;
    
    void runOnOperation() override;
    
    llvm::StringRef getArgument() const override {
        return "prepare-hw-for-systemc";
    }
    
    llvm::StringRef getDescription() const override {
        return "Prepare HW modules for CIRCT SystemC conversion (Phase 3 bug workaround)";
    }

private:
    /// Apply the CIRCT bug workaround to a specific HW module
    /// Adds attributes and markers to prevent invalid signal reads
    void applyCirctBugWorkaround(circt::hw::HWModuleOp hwModule);
};

/// Factory function to create the PrepareHWForSystemCPass
std::unique_ptr<mlir::Pass> createPrepareHWForSystemCPass();

} // namespace sv2sc::mlir_support