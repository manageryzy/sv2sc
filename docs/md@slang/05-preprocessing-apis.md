# Slang Preprocessing APIs

This document describes the Slang preprocessing APIs used for handling SystemVerilog preprocessor directives, defines, and includes.

## Overview

Slang provides comprehensive preprocessing support for SystemVerilog, including macro definitions, file inclusion, conditional compilation, and other preprocessor directives. The sv2sc project leverages these APIs to properly handle VCS-compatible preprocessor options.

## Core Preprocessing Classes

### `slang::parsing::PreprocessorOptions`

Configuration class for preprocessor behavior.

```cpp
#include <slang/parsing/ParserOptions.h>

// Create preprocessor options
slang::parsing::PreprocessorOptions preprocessorOpts;

// Configure defines
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");
preprocessorOpts.predefines.emplace_back("WIDTH=32");
preprocessorOpts.predefines.emplace_back("DEBUG");

// Configure include paths
preprocessorOpts.additionalIncludePaths.emplace_back("./includes");
preprocessorOpts.additionalIncludePaths.emplace_back("../common");

// Set maximum include depth
preprocessorOpts.maxIncludeDepth = 100;
```

**Key Properties:**
- `predefines` - Vector of macro definitions
- `additionalIncludePaths` - Vector of include search directories
- `maxIncludeDepth` - Maximum include nesting level
- `ignoreDirectives` - Set of directives to ignore

### VCS-Compatible Define Processing

```cpp
// Process VCS-style defines from command line
void processVCSDefines(slang::parsing::PreprocessorOptions& opts,
                      const std::map<std::string, std::string>& defineMap) {
    for (const auto& [name, value] : defineMap) {
        std::string defineStr;
        
        if (value.empty()) {
            // Simple define: +define+NAME
            defineStr = name;
        } else {
            // Define with value: +define+NAME=VALUE
            defineStr = name + "=" + value;
        }
        
        opts.predefines.emplace_back(defineStr);
        LOG_DEBUG("Added preprocessor define: {}", defineStr);
    }
}
```

### Include Path Management

```cpp
// Process VCS-style include paths
void processVCSIncludes(slang::parsing::PreprocessorOptions& opts,
                       const std::vector<std::string>& includePaths) {
    for (const auto& includePath : includePaths) {
        // Validate path exists
        if (std::filesystem::exists(includePath)) {
            opts.additionalIncludePaths.emplace_back(includePath);
            LOG_DEBUG("Added include path: {}", includePath);
        } else {
            LOG_WARN("Include path does not exist: {}", includePath);
        }
    }
}
```

## Preprocessor Directives

### Macro Definitions

#### `define Directive

```systemverilog
`define WIDTH 32
`define RESET_VALUE 1'b0
`define MAX(a,b) ((a) > (b) ? (a) : (b))
```

```cpp
// Handle macro definitions in preprocessing
// These are automatically processed by Slang when using predefines

// Functional macros with parameters
preprocessorOpts.predefines.emplace_back("MAX(a,b)=((a) > (b) ? (a) : (b))");

// Simple value macros
preprocessorOpts.predefines.emplace_back("WIDTH=32");
preprocessorOpts.predefines.emplace_back("RESET_VALUE=1'b0");
```

#### `undef Directive

```systemverilog
`undef WIDTH
```

Slang automatically handles `undef` directives during preprocessing.

### File Inclusion

#### `include Directive

```systemverilog
`include "definitions.sv"
`include "interfaces/axi_if.sv"
```

```cpp
// Include paths are searched automatically
// Configure search paths in preprocessor options
preprocessorOpts.additionalIncludePaths.emplace_back("./definitions");
preprocessorOpts.additionalIncludePaths.emplace_back("./interfaces");
```

### Conditional Compilation

#### `ifdef, `ifndef, `else, `endif

```systemverilog
`ifdef SYNTHESIS
    // Synthesis-specific code
`else
    // Simulation-specific code
`endif

`ifndef DEBUG
    `define DEBUG 0
`endif
```

Slang automatically evaluates conditional compilation based on defined macros.

```cpp
// Control conditional compilation through defines
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");
// This will cause `ifdef SYNTHESIS blocks to be included
```

### Other Directives

#### `timescale Directive

```systemverilog
`timescale 1ns/1ps
```

```cpp
// Handle timescale in compilation options
if (!timescale.empty()) {
    LOG_DEBUG("Setting timescale: {}", timescale);
    // Parse timescale format: "1ns/1ps"
    // Configure compilation accordingly
}
```

#### `resetall Directive

```systemverilog
`resetall
```

Automatically handled by Slang preprocessing.

## Advanced Preprocessing Features

### Library Search Paths

```cpp
// VCS library path support (-y option)
void configureLibraryPaths(slang::parsing::PreprocessorOptions& opts,
                          const std::vector<std::string>& libraryPaths) {
    // Add library directories to include paths
    for (const auto& libPath : libraryPaths) {
        opts.additionalIncludePaths.emplace_back(libPath);
        LOG_DEBUG("Added library path: {}", libPath);
    }
}
```

### File Extensions

```cpp
// VCS library extension support (+libext+ option)
void configureFileExtensions(slang::parsing::PreprocessorOptions& opts,
                            const std::vector<std::string>& extensions) {
    // Slang handles file extensions automatically
    // Log configured extensions
    for (const auto& ext : extensions) {
        LOG_DEBUG("Library extension: {}", ext);
    }
}
```

### Error Handling

```cpp
// Configure preprocessor error handling
preprocessorOpts.ignoreDirectives.insert(slang::parsing::DirectiveKind::Line);

// Check for preprocessing errors after compilation
void checkPreprocessingErrors(const slang::ast::Compilation& compilation) {
    auto& diagnostics = compilation.getAllDiagnostics();
    
    for (const auto& diag : diagnostics) {
        if (diag.isError()) {
            LOG_ERROR("Preprocessing error: {}", diag.formattedMessage);
        } else if (diag.isWarning()) {
            LOG_WARN("Preprocessing warning: {}", diag.formattedMessage);
        }
    }
}
```

## Integration with sv2sc

### VCS Command Line Compatibility

```cpp
class VCSArgsProcessor {
public:
    void processPreprocessorArgs(const std::vector<std::string>& args,
                                slang::parsing::PreprocessorOptions& opts) {
        for (const auto& arg : args) {
            if (arg.starts_with("+define+")) {
                processDefineArg(arg.substr(8), opts);
            } else if (arg.starts_with("+incdir+")) {
                processIncdirArg(arg.substr(8), opts);
            } else if (arg.starts_with("-I")) {
                processIncdirArg(arg.substr(2), opts);
            } else if (arg.starts_with("-D")) {
                processDefineArg(arg.substr(2), opts);
            } else if (arg.starts_with("-y")) {
                processLibraryPath(arg.substr(2), opts);
            }
        }
    }

private:
    void processDefineArg(const std::string& define,
                         slang::parsing::PreprocessorOptions& opts) {
        opts.predefines.emplace_back(define);
    }
    
    void processIncdirArg(const std::string& path,
                         slang::parsing::PreprocessorOptions& opts) {
        opts.additionalIncludePaths.emplace_back(path);
    }
};
```

### Configuration Example

```cpp
slang::parsing::PreprocessorOptions createPreprocessorOptions(
    const TranslationOptions& options) {
    
    slang::parsing::PreprocessorOptions preprocessorOpts;
    
    // Process defines from +define+ and -D options
    for (const auto& [name, value] : options.defineMap) {
        std::string defineStr = name;
        if (!value.empty()) {
            defineStr += "=" + value;
        }
        preprocessorOpts.predefines.emplace_back(defineStr);
    }
    
    // Process include paths from +incdir+ and -I options
    for (const auto& includePath : options.includePaths) {
        preprocessorOpts.additionalIncludePaths.emplace_back(includePath);
    }
    
    // Configure maximum include depth
    preprocessorOpts.maxIncludeDepth = 100;
    
    return preprocessorOpts;
}
```

## Common Preprocessor Patterns

### Synthesis vs Simulation

```cpp
// Common pattern: Different code for synthesis and simulation
preprocessorOpts.predefines.emplace_back("SYNTHESIS=1");

// In SystemVerilog:
// `ifdef SYNTHESIS
//     // Optimized synthesis code
// `else  
//     // Simulation assertions and debug code
// `endif
```

### Parameterized Designs

```cpp
// Configure design parameters via defines
preprocessorOpts.predefines.emplace_back("DATA_WIDTH=32");
preprocessorOpts.predefines.emplace_back("ADDR_WIDTH=16");
preprocessorOpts.predefines.emplace_back("ENABLE_CACHE=1");
```

### Debug and Tracing

```cpp
// Conditional debug features
if (enableDebug) {
    preprocessorOpts.predefines.emplace_back("DEBUG=1");
    preprocessorOpts.predefines.emplace_back("ENABLE_TRACE=1");
}
```

## Error Handling and Diagnostics

```cpp
void handlePreprocessingDiagnostics(const slang::ast::Compilation& compilation) {
    for (const auto& diag : compilation.getAllDiagnostics()) {
        switch (diag.severity) {
            case slang::DiagnosticSeverity::Error:
                LOG_ERROR("Preprocessor error: {} at {}", 
                         diag.formattedMessage, diag.location);
                break;
                
            case slang::DiagnosticSeverity::Warning:
                LOG_WARN("Preprocessor warning: {} at {}", 
                        diag.formattedMessage, diag.location);
                break;
                
            case slang::DiagnosticSeverity::Note:
                LOG_INFO("Preprocessor note: {} at {}", 
                        diag.formattedMessage, diag.location);
                break;
        }
    }
}
```

## Best Practices

1. **Validate Include Paths**: Check that include directories exist before adding them
2. **Handle Missing Files**: Provide meaningful error messages for missing includes
3. **Manage Macro Conflicts**: Be careful with macro redefinition warnings
4. **Performance**: Limit include depth to prevent infinite recursion
5. **Compatibility**: Support both VCS-style (+define+) and GCC-style (-D) options

## Integration Notes

- Full VCS command-line compatibility for preprocessor options
- Automatic handling of SystemVerilog preprocessor directives
- Support for conditional compilation and macro expansion
- Comprehensive error reporting and diagnostics
- Integration with file search paths and library management
