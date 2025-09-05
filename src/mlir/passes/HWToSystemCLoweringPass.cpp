#include "mlir/passes/HWToSystemCLoweringPass.h"

// MLIR includes
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

// CIRCT includes
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/SystemC/SystemCTypes.h"

// MLIR Arith dialect for constants
#include "mlir/Dialect/Arith/IR/Arith.h"

// Logging
#include "utils/logger.h"

namespace sv2sc::mlir_support {

void HWToSystemCLoweringPass::runOnOperation() {
    LOG_DEBUG("Running HW to SystemC lowering pass");
    
    auto module = getOperation();
    auto& context = getContext();
    
    // Create conversion target
    mlir::ConversionTarget target(context);
    configureConversionTarget(target);
    
    // Create type converter
    mlir::TypeConverter typeConverter;
    // TODO: Configure type converter for HW -> SystemC type mapping
    
    // Populate rewrite patterns
    mlir::RewritePatternSet patterns(&context);
    populateRewritePatterns(patterns);
    
    // Apply conversion
    LOG_DEBUG("Applying HW to SystemC conversion patterns");
    
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        LOG_ERROR("HW to SystemC lowering failed");
        signalPassFailure();
        return;
    }
    
    LOG_DEBUG("HW to SystemC lowering completed successfully");
}

void HWToSystemCLoweringPass::configureConversionTarget(mlir::ConversionTarget& target) {
    LOG_DEBUG("Configuring conversion target for HW to SystemC lowering");
    
    // Mark SystemC dialect as legal target
    target.addLegalDialect<circt::systemc::SystemCDialect>();
    
    // Mark HW dialect operations as illegal (to be converted)
    target.addIllegalDialect<circt::hw::HWDialect>();
    
    // Allow some operations to remain as-is
    target.addLegalOp<mlir::ModuleOp>();
    
    LOG_DEBUG("Conversion target configured");
}

void HWToSystemCLoweringPass::populateRewritePatterns(mlir::RewritePatternSet& patterns) {
    LOG_DEBUG("Populating HW to SystemC rewrite patterns");
    
    auto& context = getContext();
    mlir::TypeConverter typeConverter;
    
    // Add conversion patterns - use insert method for OpConversionPattern
    patterns.insert<patterns::ModuleOpLowering>(typeConverter, &context);
    patterns.insert<patterns::ConstantOpLowering>(typeConverter, &context);
    patterns.insert<patterns::OutputOpLowering>(typeConverter, &context);
    
    LOG_DEBUG("Rewrite patterns populated");
}

std::unique_ptr<mlir::Pass> createHWToSystemCLoweringPass() {
    return std::make_unique<HWToSystemCLoweringPass>();
}

// SystemC Convert Folding Pass
class FoldSystemCConvertPass : public mlir::PassWrapper<FoldSystemCConvertPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    FoldSystemCConvertPass() = default;
    
    void runOnOperation() override {
        LOG_DEBUG("Running SystemC convert folding pass");
        
        auto module = getOperation();
        auto context = module.getContext();
        
        // Handle systemc.convert operations for CIRCT ExportSystemC compatibility  
        llvm::SmallVector<mlir::Operation*> toErase;
        
        module.walk([&](mlir::Operation* op) {
            if (op->getName().getStringRef() == "systemc.convert") {
                LOG_DEBUG("Found systemc.convert operation - folding");
                
                // For conversions, replace with the input value
                if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
                    auto input = op->getOperand(0);
                    auto result = op->getResult(0);
                    
                    // Replace all uses of the result with the input
                    result.replaceAllUsesWith(input);
                    toErase.push_back(op);
                    LOG_DEBUG("Folded systemc.convert operation");
                }
            }
        });
        
        // Erase operations after walking
        for (auto* op : toErase) {
            op->erase();
        }
        
        // Second pass: Handle hw.constant operations and fix type mismatches in systemc.signal.write operations
        llvm::SmallVector<mlir::Operation*> hwConstantsToErase;
        
        module.walk([&](mlir::Operation* op) {
            if (op->getName().getStringRef() == "hw.constant") {
                LOG_DEBUG("Found hw.constant operation - converting to arith.constant");
                
                mlir::OpBuilder builder(op);
                auto hwConstOp = mlir::cast<circt::hw::ConstantOp>(op);
                auto value = hwConstOp.getValue();
                auto hwType = hwConstOp.getType();
                
                // Create arith.constant with the same value and type
                mlir::Value newConstant;
                if (auto intType = llvm::dyn_cast<mlir::IntegerType>(hwType)) {
                    auto attr = builder.getIntegerAttr(intType, value);
                    newConstant = builder.create<mlir::arith::ConstantOp>(op->getLoc(), attr);
                    LOG_DEBUG("Created arith.constant for integer type with width: {}", intType.getWidth());
                } else {
                    LOG_WARN("Unsupported hw.constant type - keeping original");
                    return;
                }
                
                // Replace all uses
                op->getResult(0).replaceAllUsesWith(newConstant);
                hwConstantsToErase.push_back(op);
            } else if (op->getName().getStringRef() == "systemc.signal.write") {
                LOG_DEBUG("Found systemc.signal.write operation - checking types");
                
                if (op->getNumOperands() >= 2) {
                    auto signal = op->getOperand(0);
                    auto value = op->getOperand(1);
                    auto signalType = signal.getType();
                    auto valueType = value.getType();
                    
                    // Check if we need to convert i8 to !systemc.uint<8>
                    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(valueType)) {
                        if (intType.getWidth() != 1) { // Not i1
                            // Create systemc.convert operation
                            mlir::OpBuilder builder(op);
                            auto systemcType = circt::systemc::UIntType::get(builder.getContext(), intType.getWidth());
                            auto convertOp = builder.create<circt::systemc::ConvertOp>(op->getLoc(), systemcType, value);
                            
                            // Replace the operand
                            op->setOperand(1, convertOp.getResult());
                            LOG_DEBUG("Added systemc.convert for signal write with uint<{}>", intType.getWidth());
                        }
                    }
                }
            }
        });
        
        // Erase hw.constant operations after second pass
        for (auto* op : hwConstantsToErase) {
            op->erase();
        }
        
        LOG_DEBUG("SystemC convert folding pass completed");
    }
    
    llvm::StringRef getArgument() const override {
        return "fold-systemc-convert";
    }
    
    llvm::StringRef getDescription() const override {
        return "Fold identity systemc.convert operations";
    }
};

std::unique_ptr<mlir::Pass> createFoldSystemCConvertPass() {
    return std::make_unique<FoldSystemCConvertPass>();
}

// Fix SystemC Signal Read Pass - removes invalid reads from output ports
class FixSystemCSignalReadPass : public mlir::PassWrapper<FixSystemCSignalReadPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    FixSystemCSignalReadPass() = default;
    
    void runOnOperation() override {
        LOG_ERROR("**** RUNNING SystemC signal read fix pass (Phase 3 Enhanced) ****");
        LOG_DEBUG("Running SystemC signal read fix pass with PrepareHWForSystemCPass coordination");
        
        auto module = getOperation();
        auto context = module.getContext();
        
        // Check if modules were prepared by PrepareHWForSystemCPass
        bool foundPreparedModules = false;
        module.walk([&](circt::systemc::SCModuleOp scModule) {
            if (scModule->hasAttr("sv2sc.prepared_for_systemc")) {
                foundPreparedModules = true;
                LOG_DEBUG("Found prepared SystemC module: {}", scModule.getModuleName());
            }
        });
        
        if (foundPreparedModules) {
            LOG_DEBUG("Processing modules prepared by PrepareHWForSystemCPass");
        } else {
            LOG_WARN("No prepared modules found - running standard fix");
        }
        
        // Handle invalid systemc.signal.read operations from output ports
        llvm::SmallVector<mlir::Operation*> toErase;
        
        module.walk([&](mlir::Operation* op) {
            if (op->getName().getStringRef() == "systemc.signal.read") {
                LOG_DEBUG("Found systemc.signal.read operation - checking port type");
                
                if (op->getNumOperands() >= 1 && op->getNumResults() >= 1) {
                    auto operand = op->getOperand(0);
                    auto operandType = operand.getType();
                    
                    // Check if this is reading from an output port - use string matching as fallback
                    std::string operandTypeStr = "";
                    llvm::raw_string_ostream typeOS(operandTypeStr);
                    operandType.print(typeOS);
                    typeOS.flush();
                    
                    if (operandTypeStr.find("!systemc.out<") != std::string::npos) {
                        LOG_DEBUG("Invalid signal read from output port - removing operation");
                        
                        // Replace all uses of the result with a constant zero value
                        auto result = op->getResult(0);
                        auto resultType = result.getType();
                        
                        mlir::OpBuilder builder(op);
                        mlir::Value replacement;
                        
                        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(resultType)) {
                            // Create zero constant for integer types
                            auto zeroAttr = builder.getIntegerAttr(intType, 0);
                            replacement = builder.create<mlir::arith::ConstantOp>(op->getLoc(), zeroAttr);
                            LOG_DEBUG("Replaced invalid signal read with zero constant (i{})", intType.getWidth());
                        } else {
                            // For SystemC types, check type string and create appropriate constants
                            std::string resultTypeStr = "";
                            llvm::raw_string_ostream resultTypeOS(resultTypeStr);
                            resultType.print(resultTypeOS);
                            resultTypeOS.flush();
                            
                            if (resultTypeStr.find("!systemc.uint<") != std::string::npos) {
                                // Extract width from type string like "!systemc.uint<8>"
                                size_t widthStart = resultTypeStr.find('<');
                                size_t widthEnd = resultTypeStr.find('>');
                                if (widthStart != std::string::npos && widthEnd != std::string::npos) {
                                    std::string widthStr = resultTypeStr.substr(widthStart + 1, widthEnd - widthStart - 1);
                                    int width = std::stoi(widthStr);
                                    auto intType = builder.getIntegerType(width);
                                    auto zeroAttr = builder.getIntegerAttr(intType, 0);
                                    auto zeroConstant = builder.create<mlir::arith::ConstantOp>(op->getLoc(), zeroAttr);
                                    replacement = builder.create<circt::systemc::ConvertOp>(op->getLoc(), resultType, zeroConstant);
                                    LOG_DEBUG("Replaced invalid signal read with zero constant (uint<{}>)", width);
                                } else {
                                    LOG_WARN("Could not parse SystemC uint width, using default");
                                    return;
                                }
                            } else {
                                LOG_WARN("Unknown SystemC result type: {} - skipping", resultTypeStr);
                                return;
                            }
                        }
                        
                        // Replace all uses
                        result.replaceAllUsesWith(replacement);
                        toErase.push_back(op);
                    }
                }
            }
        });
        
        // Erase operations after walking
        for (auto* op : toErase) {
            op->erase();
        }
        
        LOG_DEBUG("SystemC signal read fix pass completed - removed {} invalid reads", toErase.size());
    }
    
    llvm::StringRef getArgument() const override {
        return "fix-systemc-signal-read";
    }
    
    llvm::StringRef getDescription() const override {
        return "Fix invalid systemc.signal.read operations from output ports";
    }
};

std::unique_ptr<mlir::Pass> createFixSystemCSignalReadPass() {
    return std::make_unique<FixSystemCSignalReadPass>();
}

// Debug IR Dump Pass - saves IR to specific files for debugging
class DebugIRDumpPass : public mlir::PassWrapper<DebugIRDumpPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    DebugIRDumpPass(const std::string& filename) : filename_(filename) {}
    
    void runOnOperation() override {
        auto module = getOperation();
        
        // Create output directory if it doesn't exist
        std::string outputDir = "./output/debug/";
        system(("mkdir -p " + outputDir).c_str());
        
        // Save IR to file
        std::string fullPath = outputDir + filename_ + ".mlir";
        std::error_code EC;
        llvm::raw_fd_ostream file(fullPath, EC);
        if (!EC) {
            module.print(file);
            file.close();
            LOG_DEBUG("Dumped IR to: {}", fullPath);
        } else {
            LOG_ERROR("Failed to write IR dump to: {}", fullPath);
        }
    }
    
    llvm::StringRef getArgument() const override {
        return "debug-ir-dump";
    }
    
    llvm::StringRef getDescription() const override {
        return "Dump IR to file for debugging";
    }

private:
    std::string filename_;
};

std::unique_ptr<mlir::Pass> createDebugIRDumpPass(const std::string& filename) {
    return std::make_unique<DebugIRDumpPass>(filename);
}

} // namespace sv2sc::mlir_support

namespace sv2sc::mlir_support::patterns {

// Common utilities for pattern implementations
static mlir::Value convertSignalType(mlir::Value hwValue, mlir::ConversionPatternRewriter& rewriter) {
    // TODO: Implement signal type conversion
    LOG_DEBUG("Converting HW signal type to SystemC (placeholder)");
    return hwValue;
}

static mlir::Type convertHWTypeToSystemC(mlir::Type hwType) {
    // TODO: Implement type conversion mapping
    LOG_DEBUG("Converting HW type to SystemC type (placeholder)");
    return hwType;
}

mlir::LogicalResult ModuleOpLowering::matchAndRewrite(
    circt::hw::HWModuleOp op,
    typename circt::hw::HWModuleOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) const {
    
    LOG_DEBUG("Converting HW module: {}", op.getModuleName().str());
    
    // TODO: Implement HW module -> SystemC module conversion
    // This would involve:
    // 1. Creating SystemC module operation
    // 2. Converting port list
    // 3. Converting module body
    // 4. Handling module instantiations
    
    // For now, just mark as converted without replacement
    rewriter.eraseOp(op);
    LOG_DEBUG("HW module conversion completed (placeholder)");
    
    return mlir::success();
}

mlir::LogicalResult ConstantOpLowering::matchAndRewrite(
    circt::hw::ConstantOp op,
    typename circt::hw::ConstantOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) const {
    
    LOG_DEBUG("Converting HW constant operation");
    
    // TODO: Implement HW constant -> SystemC constant conversion
    // This would involve creating appropriate SystemC constant operations
    
    // For now, just mark as converted
    rewriter.eraseOp(op);
    LOG_DEBUG("HW constant conversion completed (placeholder)");
    
    return mlir::success();
}

mlir::LogicalResult OutputOpLowering::matchAndRewrite(
    circt::hw::OutputOp op,
    typename circt::hw::OutputOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) const {
    
    LOG_DEBUG("Converting HW output operation");
    
    // TODO: Implement HW output -> SystemC output conversion
    // This would involve creating SystemC output assignment operations
    
    // For now, just mark as converted
    rewriter.eraseOp(op);
    LOG_DEBUG("HW output conversion completed (placeholder)");
    
    return mlir::success();
}

} // namespace sv2sc::mlir_support::patterns