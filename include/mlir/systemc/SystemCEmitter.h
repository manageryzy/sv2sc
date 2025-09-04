#pragma once

#include <string>
#include <ostream>
#include <vector>

// MLIR includes - we always have real CIRCT
#include "mlir/IR/BuiltinOps.h"

namespace sv2sc {
namespace mlir_support {

class SystemCEmitter {
public:
    struct EmitResult {
        bool success;
        std::string headerPath;
        std::string implPath;
        std::string errorMessage;
    };

    static EmitResult emitSplit(mlir::ModuleOp module, const std::string& outDir);
};

} // namespace mlir_support
} // namespace sv2sc
