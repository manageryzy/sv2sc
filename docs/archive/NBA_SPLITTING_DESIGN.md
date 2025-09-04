# Non-Blocking Assignment (NBA) Splitting Design

## Problem Statement
Currently, all non-blocking assignments from `always_ff` blocks are placed into a single `seq_proc()` method in SystemC. This creates:
- **Performance bottleneck**: Single large method evaluates on every clock edge
- **Poor locality**: Unrelated logic mixed together
- **Cache misses**: Large method footprint
- **Simulation overhead**: All assignments evaluated even if inputs haven't changed

## Proposed Solution

### 1. Process Grouping Strategy

Split NBA logic into multiple SC_METHOD processes based on:

#### A. **Clock Domain Grouping**
```systemverilog
always_ff @(posedge clk1) begin ... end  // → clk1_proc()
always_ff @(posedge clk2) begin ... end  // → clk2_proc()
```

#### B. **Functional Unit Grouping**
```systemverilog
// Group by logical function
always_ff @(posedge clk) begin
    // ALU logic → alu_proc()
    if (alu_enable) alu_result <= ...;
    
    // Register file → regfile_proc()
    if (reg_write) registers[addr] <= ...;
    
    // Control logic → control_proc()
    state <= next_state;
end
```

#### C. **Sensitivity-Based Grouping**
```systemverilog
// Group by input dependencies
always_ff @(posedge clk) begin
    if (reset) begin
        // Reset logic → reset_proc()
        counter <= 0;
        state <= IDLE;
    end else begin
        // Normal logic → main_proc()
        counter <= counter + 1;
    end
end
```

### 2. Implementation Architecture

#### New Data Structures
```cpp
// In systemc_generator.h
struct ProcessBlock {
    std::string name;                    // e.g., "alu_proc", "ctrl_proc"
    std::stringstream code;               // Process body
    std::set<std::string> sensitivity;   // Sensitivity list
    std::set<std::string> outputs;       // Signals written
    std::set<std::string> inputs;        // Signals read
    bool isSequential;                   // true for always_ff
    std::string clockSignal;             // Clock for sequential
    std::string resetSignal;             // Reset signal if any
};

struct ModuleData {
    // ... existing fields ...
    std::vector<ProcessBlock> processBlocks;  // Multiple process blocks
    std::map<std::string, std::string> signalToProcess;  // Track which process owns each signal
};
```

#### Process Splitting Algorithm
```cpp
class ProcessSplitter {
public:
    std::vector<ProcessBlock> splitProcess(
        const std::string& originalCode,
        const std::set<std::string>& signals
    ) {
        std::vector<ProcessBlock> blocks;
        
        // Step 1: Parse assignments and dependencies
        auto assignments = parseAssignments(originalCode);
        
        // Step 2: Build dependency graph
        auto depGraph = buildDependencyGraph(assignments);
        
        // Step 3: Identify independent groups
        auto groups = findIndependentGroups(depGraph);
        
        // Step 4: Create process blocks
        for (const auto& group : groups) {
            blocks.push_back(createProcessBlock(group));
        }
        
        return blocks;
    }
};
```

### 3. Generation Strategy

#### Before (Single Process)
```cpp
SC_MODULE(cpu) {
    // ... ports ...
    
    SC_CTOR(cpu) {
        SC_METHOD(seq_proc);
        sensitive << clk.pos();
    }
    
private:
    void seq_proc() {
        // ALL sequential logic here
        if (reset.read()) {
            pc.write(0);
            state.write(IDLE);
            alu_result.write(0);
            // ... 100+ more assignments
        } else {
            // ... massive if-else chain
        }
    }
};
```

#### After (Multiple Processes)
```cpp
SC_MODULE(cpu) {
    // ... ports ...
    
    SC_CTOR(cpu) {
        // Program counter process
        SC_METHOD(pc_proc);
        sensitive << clk.pos() << reset;
        
        // ALU process
        SC_METHOD(alu_proc);
        sensitive << clk.pos() << alu_enable;
        
        // Control state machine
        SC_METHOD(ctrl_proc);
        sensitive << clk.pos() << state;
        
        // Register file process
        SC_METHOD(regfile_proc);
        sensitive << clk.pos() << reg_write;
    }
    
private:
    void pc_proc() {
        if (reset.read()) {
            pc.write(0);
        } else if (pc_enable.read()) {
            pc.write(next_pc.read());
        }
    }
    
    void alu_proc() {
        if (alu_enable.read()) {
            alu_result.write(compute_alu());
        }
    }
    
    void ctrl_proc() {
        state.write(next_state.read());
    }
    
    void regfile_proc() {
        if (reg_write.read()) {
            registers[addr.read()].write(data.read());
        }
    }
};
```

### 4. Heuristics for Splitting

#### Automatic Detection Rules
1. **Separate always_ff blocks** → Separate processes
2. **Different clock domains** → Separate processes
3. **Independent if-else branches** → Can be separate processes
4. **Array/memory writes** → Group in memory process
5. **State machine logic** → Dedicated FSM process

#### Size Thresholds
- Split if process > 50 lines
- Split if > 20 signals written
- Keep together if strong data dependencies

### 5. Benefits

#### Performance Improvements
- **Reduced evaluation overhead**: Only affected processes run
- **Better cache locality**: Smaller method footprints
- **Parallel evaluation potential**: Independent processes
- **Cleaner generated code**: Easier to debug and understand

#### Simulation Speed
- **Before**: 100% of logic evaluated every clock
- **After**: 20-40% evaluated (only changed portions)
- **Expected speedup**: 2-5x for large designs

### 6. Implementation Plan

1. **Phase 1**: Extend ModuleData structure
2. **Phase 2**: Implement process splitter
3. **Phase 3**: Update AST visitor to track blocks
4. **Phase 4**: Modify code generator
5. **Phase 5**: Add optimization passes
6. **Phase 6**: Performance testing

### 7. Configuration Options

```cpp
// User control over splitting
struct SplittingConfig {
    bool enableSplitting = true;
    int maxProcessSize = 50;          // Lines
    int maxSignalsPerProcess = 20;    // Signals
    bool splitByClockDomain = true;
    bool splitByFunctionalUnit = true;
    bool aggressiveSplitting = false;  // More but smaller processes
};
```

## Example: PicoRV32 Optimization

### Current (Single Process)
- **Lines**: 500+ in seq_proc()
- **Signals**: 80+ updated
- **Evaluation**: Every cycle, all logic

### After Splitting
- **pc_proc()**: 15 lines, PC logic only
- **alu_proc()**: 40 lines, ALU operations
- **decode_proc()**: 30 lines, instruction decode
- **mem_proc()**: 25 lines, memory interface
- **ctrl_proc()**: 35 lines, control FSM

### Expected Results
- **Simulation speedup**: 3-4x
- **Compilation time**: Slightly longer (more methods)
- **Debug-ability**: Much improved
- **Code clarity**: Significantly better
