# Claude Code Memory - SV2SC Project

## Project Overview
sv2sc is a SystemVerilog to SystemC translator built with modern C++20 and CMake, featuring both standard and MLIR/CIRCT-based translation pipelines.

## Build Commands

```bash
cmake -B build -DSV2SC_ENABLE_MLIR=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Testing Commands
```bash
# Quick test
./build/src/sv2sc --use-mlir -top memory tests/examples/memory_array/memory.sv

# All tests
ctest --test-dir build

# PicoRV32 test
./build/src/sv2sc --use-mlir -top picorv32 third-party/picorv32/picorv32.v
```

## Critical MLIR Rules

1. **Never** call `buildStatement()` on procedural block bodies - causes infinite recursion
2. Always use `mlir::OpBuilder::InsertionGuard` for nested operations
3. Check operation count limits to prevent stack overflow
4. Load required CIRCT dialects: HWDialect, SeqDialect, CombDialect, SVDialect, SystemCDialect
5. Always add debug information to the code.

## Important Notes

1. **Never** delete the `build/` directory or modify it in any way. Ask user to do it.
2. Always use ninja to build the project.
3. Always use `-debug` and `-verbose` flags to get more information about the CIRCT bug.
4. **Always** read circt header for types before using them.
5. **Always** read and write your memory and plan in `memo` directory.
6. **Always** write memory and plan in markdown format after planing
7. **Always** break down the problem into smaller problems and solve them one by one.
8. Always write warning information printing in unfinished code.
9. Always read full code in planing.
10. Always write `FIXME` in unfinished code.
11. Always write all branches and conditions in code. Especially for switch cases.

## Language Rules

1. Always Use c++ 17 features if possible.
2. Do not use exception or RTTI. If some code must throw exception, try to handle it in the same function.
3. Use namespace for all code.
4. Use struct only for POD-like aggregates.
5. Use class when you need to add methods to the type.
6. Prefer final class.
7. No virtual destructor unless you need to override it.
8. Avoid Multiple Inheritance.
9. Use RAII for all resources.
10. Do not use std::shared_ptr unless you have to.
11. Use llvm::BumpPtrAllocator or llvm::ArenaAllocator for dynamic memory allocation.
12. Use llvm::SmallVector or llvm::SmallString for small collections.
13. Use StringRef for all string operations.
14. Use ArrayRef for read only view.
15. Avoid using c++ iostreams.
16. Do not use C-style casts.
17. Do not rely on undefined behavior.

## Key Files

- Main application: `src/sv2sc.cpp`
- Code Generator: `src/codegen/`
- MLIR translator: `src/mlir/MLIRTranslator.cpp`
- HW builder: `src/mlir/SVToHWBuilder.cpp`
- Pass pipeline: `src/mlir/pipeline/SV2SCPassPipeline.cpp`
- Context manager: `src/mlir/MLIRContextManager.cpp`
- circt: `third-party/circt/`
- circt header files: `third-party/circt/include/circt/`
- cmake rules: `cmake/`

## Documentation

- Complete reports in `output/` directory
- Implementation guide: `output/MLIR_FINAL_IMPLEMENTATION_REPORT.md`
- PicoRV32 fixes: `output/PICORV32_FIX_REPORT.md`

### Debug Infrastructure

- IR dumping enabled at each pipeline stage: `./output/debug/`
- FixSystemCSignalReadPass implemented but unreachable due to early verification failure
- Manual verification framework in place for post-fix validation
- Use `-debug` and `-verbose` flags to get more information about the CIRCT bug.

## Temporary Files

All test files stored in `tmp/` directory with automatic cleanup after tests.

## Build System Rules

- **ALWAYS** wait for cmake builds to complete before proceeding with tests or further development
- **NEVER** interrupt running builds - they can take 15-30 minutes for full LLVM/CIRCT compilation
- Full in-tree CIRCT builds are necessary for MLIR pipeline development and testing
- **Never** delete the `build/` directory or modify it in any way. Ask user to do it.
- Always use ninja to build the project.

## Test Infrastructure Rules

- Testing are in `tests/` directory.
- Use cmake test infrastructure to run tests.
- Testing rules are in cmake functions in `cmake/` directory.
- If you are smoking test, directly run commands like `./build/src/sv2sc --use-mlir -top <module> tests/examples/<module>/<module>.sv` to run test.
- Always write test for new features.
- Always write test for bug fixes.
