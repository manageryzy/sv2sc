#include "mlir/systemc/SystemCEmitter.h"
#include "CIRCTCompatibleEmitter.h"
#include "SystemCEmissionPatterns.h"
#include <filesystem>
#include <fstream>
#include "utils/logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"

namespace sv2sc::mlir_support {

static void writeFile(const std::string& path, const std::string& content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream os(path);
    os << content;
}

static std::string scTypeFromType(mlir::Type type) {
    std::string s;
    llvm::raw_string_ostream os(s);
    type.print(os);
    os.flush();
    // naive mapping for !systemc.uint<N>
    if (s.find("!systemc.uint<") != std::string::npos) {
        auto lt = s.find('<');
        auto gt = s.find('>');
        auto n = (lt != std::string::npos && gt != std::string::npos && gt > lt) ? s.substr(lt+1, gt-lt-1) : "32";
        return "sc_out< sc_uint<" + n + "> >";
    }
    if (s.find("!systemc.in<") != std::string::npos) return "sc_in<bool>";
    if (s.find("!systemc.out<") != std::string::npos) return "sc_out<bool>";
    return "sc_signal<bool>";
}

SystemCEmitter::EmitResult SystemCEmitter::emitSplit(mlir::ModuleOp module, const std::string& outDir) {
    try {
        // Use our sv2sc CIRCT-compatible emitter as the default implementation
        CIRCTCompatibleEmitter circtEmitter;
        
        // Register all emission patterns
        registerAllEmissionPatterns(circtEmitter);
        
        auto circtResult = circtEmitter.emitSplit(module, outDir);
        
        if (circtResult.success) {
            LOG_INFO("sv2sc CIRCT-compatible emitter successfully generated files");
            return {circtResult.success, circtResult.headerPath, circtResult.implPath, circtResult.errorMessage};
        } else {
            LOG_WARN("sv2sc CIRCT-compatible emitter failed: {}, falling back to simple emitter", circtResult.errorMessage);
        }
        
        // Fallback to simple emitter
        std::string topName = "module";
        if (auto attr = module->getAttrOfType<mlir::StringAttr>("sym_name")) topName = attr.str();
        
        // Try to extract module name from systemc.module operation
        module.walk([&](mlir::Operation* op){
            std::string opName = op->getName().getStringRef().str();
            LOG_INFO("DEBUG: Found operation: {}", opName);
            if (opName == "systemc.module") {
                LOG_INFO("DEBUG: Found systemc.module operation");
                
                // Try multiple ways to extract the module name
                if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
                    topName = nameAttr.str();
                    LOG_INFO("Extracted SystemC module name from sym_name: {}", topName);
                    return mlir::WalkResult::interrupt();
                } else if (auto symbolAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("symbol_name")) {
                    topName = symbolAttr.getValue().str();
                    LOG_INFO("Extracted SystemC module name from symbol_name: {}", topName);
                    return mlir::WalkResult::interrupt();
                } else {
                    // Try to extract from operation string representation as fallback
                    std::string opStr;
                    llvm::raw_string_ostream os(opStr);
                    op->print(os);
                    os.flush();
                    
                    // Look for pattern like "systemc.module @counter"
                    auto pos = opStr.find("systemc.module @");
                    if (pos != std::string::npos) {
                        auto nameStart = pos + 16; // After "systemc.module @"
                        auto nameEnd = opStr.find(' ', nameStart);
                        if (nameEnd == std::string::npos) nameEnd = opStr.find('(', nameStart);
                        if (nameEnd == std::string::npos) nameEnd = opStr.find('\n', nameStart);
                        if (nameEnd != std::string::npos && nameEnd > nameStart) {
                            topName = opStr.substr(nameStart, nameEnd - nameStart);
                            LOG_INFO("Extracted SystemC module name from operation string: {}", topName);
                            return mlir::WalkResult::interrupt();
                        }
                    }
                    LOG_WARN("systemc.module operation has no extractable name attribute");
                }
            }
            return mlir::WalkResult::advance();
        });
        const std::string headerPath = outDir + "/" + topName + ".h";
        const std::string implPath = outDir + "/" + topName + ".cpp";

        std::string header;
        header += "#pragma once\n#include <systemc.h>\n\n";
        
        // Find first systemc.module to create a basic module shell
        mlir::Operation* scModuleOp = nullptr;
        module.walk([&](mlir::Operation* op){
            if (op->getName().getStringRef().starts_with("systemc.module")) { 
                scModuleOp = op; 
                return mlir::WalkResult::interrupt(); 
            } 
            return mlir::WalkResult::advance();
        });

        if (!scModuleOp) {
            // fallback empty
            header += "SC_MODULE(" + topName + ") {\n  SC_CTOR(" + topName + ") {}\n};\n";
            writeFile(headerPath, header);
            writeFile(implPath, "#include \"" + topName + ".h\"\n");
            return {true, headerPath, implPath, {}};
        }

        // Ports from block arguments
        header += "SC_MODULE(" + topName + ") {\n";
        if (!scModuleOp->getRegions().empty() && !scModuleOp->getRegion(0).empty()) {
            auto &block = scModuleOp->getRegion(0).front();
            for (auto arg : block.getArguments()) {
                header += "  " + scTypeFromType(arg.getType()) + " port_" + std::to_string(arg.getArgNumber()) + ";\n";
            }
        }
        header += "\n  SC_CTOR(" + topName + ") { }\n";
        header += "};\n";

        writeFile(headerPath, header);
        writeFile(implPath, "#include \"" + topName + ".h\"\n");
        LOG_INFO("Fallback SystemCEmitter wrote: {} and {}", headerPath, implPath);
        return {true, headerPath, implPath, {}};
    } catch (const std::exception& e) {
        return {false, {}, {}, e.what()};
    }
}

} // namespace sv2sc::mlir_support


