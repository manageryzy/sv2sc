# NBA Splitting Implementation Summary

## What Was Implemented

### 1. New Data Structures ✅
Added to `include/codegen/systemc_generator.h`:
- **ProcessBlock struct**: Represents individual process blocks with:
  - Name, code, sensitivity list
  - Input/output signal tracking
  - Sequential/combinational flag
  - Clock and reset signals
  - Line count for splitting heuristics

- **ModuleData enhancements**:
  - `processBlocks` vector for multiple processes
  - `signalToProcess` map for ownership tracking
  - `enableProcessSplitting` flag
  - `maxProcessLines` threshold

### 2. New API Methods ✅
Added to SystemCCodeGenerator:
- `beginProcessBlock()`: Start a new process block
- `endProcessBlock()`: Finalize and optionally split large blocks
- `addAssignmentToCurrentBlock()`: Add assignments to current block
- `setCurrentBlockSensitivity()`: Set sensitivity signals
- `setCurrentBlockClock()`: Set clock signal
- `setCurrentBlockReset()`: Set reset signal
- `enableProcessSplitting()`: Enable/disable splitting
- `setMaxProcessLines()`: Configure splitting threshold

### 3. Process Splitting Logic ✅
- `splitLargeProcess()`: Splits processes exceeding line threshold
- `analyzeAndSplitProcess()`: Analyzes dependencies and creates sub-processes
- Automatic splitting when process > 50 lines (configurable)
- Groups assignments into ~20 line blocks

### 4. Code Generation Updates ✅
Modified generation to support multiple processes:
- Constructor generation handles multiple SC_METHOD registrations
- Process method generation creates separate methods for each block
- Proper sensitivity list generation per process
- Clock edge handling for sequential processes

## Current Status

### What Works
✅ Infrastructure for multiple process blocks is complete
✅ Code generator can handle multiple processes
✅ Splitting algorithm implemented
✅ Backward compatibility maintained

### What Needs Integration
The AST visitor needs updating to use the new API. Currently it still uses the legacy single-process approach.

## Next Steps for Full Integration

### 1. Update AST Visitor
The AST visitor (`src/core/ast_visitor.cpp`) needs modification to:
```cpp
// In handle(ProceduralBlockSymbol):
void SVToSCVisitor::handle(const slang::ast::ProceduralBlockSymbol& node) {
    // Create unique process block for each always_ff
    std::string blockName = generateUniqueBlockName(node);
    codeGen_.beginProcessBlock(blockName, isSequential);
    
    // Visit body
    node.getBody().visit(*this);
    
    // Set sensitivity based on timing control
    extractAndSetSensitivity(node);
    
    codeGen_.endProcessBlock();
}
```

### 2. Assignment Routing
Update assignment generation to use new API:
```cpp
// Instead of:
codeGen_.addSequentialAssignment(lhs, rhs);

// Use:
codeGen_.addAssignmentToCurrentBlock(lhs, rhs);
```

### 3. Sensitivity Extraction
Extract sensitivity from timing controls:
```cpp
// Extract from @(posedge clk) or @(*)
auto sensitivity = extractSensitivityList(timingControl);
codeGen_.setCurrentBlockSensitivity(sensitivity);
```

## Performance Benefits (Expected)

### Before (Single Process)
```systemc
void seq_proc() {
    // 200+ lines of mixed logic
    // ALL evaluated every clock cycle
}
```

### After (Multiple Processes)
```systemc
void alu_proc() {
    // 30 lines - ALU logic only
}

void pc_proc() {
    // 25 lines - PC logic only
}

void state_proc() {
    // 40 lines - State machine only
}

void mem_proc() {
    // 35 lines - Memory interface only
}

void regfile_proc() {
    // 20 lines - Register file only
}
```

### Expected Improvements
- **Simulation Speed**: 2-5x faster for large designs
- **Cache Efficiency**: Better locality of reference
- **Debug-ability**: Easier to trace specific functionality
- **Maintainability**: Cleaner generated code

## Configuration Options

Users can control splitting behavior:
```cpp
// In main.cpp or via command line
generator.enableProcessSplitting(true);
generator.setMaxProcessLines(30);  // Split if > 30 lines
```

## Testing

Created test case: `tests/integration/test_nba_splitting.sv`
- Complex processor with 5 always_ff blocks
- Each block handles different functionality
- Should generate 5 separate processes when fully integrated

## Summary

The infrastructure for NBA splitting is **fully implemented** in the code generator. The remaining work is to integrate it with the AST visitor to actually use these new capabilities. This will transform the current monolithic process generation into efficient, modular SystemC code with significant performance benefits.
