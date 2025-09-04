#pragma once

#include <memory>

// MLIR includes - we always have real CIRCT
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

// CIRCT includes
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"

namespace sv2sc::mlir_support {

/**
 * @brief Pass for lowering HW dialect operations to SystemC dialect
 * 
 * This pass converts CIRCT HW dialect constructs to SystemC dialect operations,
 * enabling the generation of SystemC code from hardware descriptions.
 */
class HWToSystemCLoweringPass : public mlir::PassWrapper<HWToSystemCLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    HWToSystemCLoweringPass() = default;
    
    void runOnOperation() override;
    
    llvm::StringRef getArgument() const override {
        return "hw-to-systemc";
    }
    
    llvm::StringRef getDescription() const override {
        return "Lower HW dialect operations to SystemC dialect";
    }

private:
    void configureConversionTarget(mlir::ConversionTarget& target);
    void populateRewritePatterns(mlir::RewritePatternSet& patterns);
};

/**
 * @brief Create an instance of the HW to SystemC lowering pass
 * @return Unique pointer to the pass
 */
std::unique_ptr<mlir::Pass> createHWToSystemCLoweringPass();

/**
 * @brief Create an instance of the SystemC convert folding pass
 * @return Unique pointer to the pass
 */
std::unique_ptr<mlir::Pass> createFoldSystemCConvertPass();

/**
 * @brief Create an instance of the SystemC signal read fix pass
 * @return Unique pointer to the pass
 */
std::unique_ptr<mlir::Pass> createFixSystemCSignalReadPass();

/**
 * @brief Create an instance of the debug IR dump pass
 * @param filename The filename (without extension) to dump IR to
 * @return Unique pointer to the pass
 */
std::unique_ptr<mlir::Pass> createDebugIRDumpPass(const std::string& filename);

} // namespace sv2sc::mlir_support

namespace sv2sc::mlir_support::patterns {

/**
 * @brief Pattern to convert HW module operations to SystemC modules
 */
class ModuleOpLowering : public mlir::OpConversionPattern<circt::hw::HWModuleOp> {
public:
    ModuleOpLowering(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context)
        : OpConversionPattern<circt::hw::HWModuleOp>(typeConverter, context) {}
    
    mlir::LogicalResult matchAndRewrite(
        circt::hw::HWModuleOp op,
        typename circt::hw::HWModuleOp::Adaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override;
};

/**
 * @brief Pattern to convert HW constant operations to SystemC constants
 */
class ConstantOpLowering : public mlir::OpConversionPattern<circt::hw::ConstantOp> {
public:
    ConstantOpLowering(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context)
        : OpConversionPattern<circt::hw::ConstantOp>(typeConverter, context) {}
    
    mlir::LogicalResult matchAndRewrite(
        circt::hw::ConstantOp op,
        typename circt::hw::ConstantOp::Adaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override;
};

/**
 * @brief Pattern to convert HW output operations to SystemC outputs
 */
class OutputOpLowering : public mlir::OpConversionPattern<circt::hw::OutputOp> {
public:
    OutputOpLowering(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context)
        : OpConversionPattern<circt::hw::OutputOp>(typeConverter, context) {}
    
    mlir::LogicalResult matchAndRewrite(
        circt::hw::OutputOp op,
        typename circt::hw::OutputOp::Adaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override;
};

} // namespace sv2sc::mlir_support::patterns

