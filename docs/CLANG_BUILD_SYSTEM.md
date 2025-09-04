# Clang Build System for sv2sc

This document describes the comprehensive Clang-enforced build system for the sv2sc project, ensuring consistent use of the Clang toolchain throughout the development and CI/CD processes.

## Overview

The Clang build system consists of several coordinated scripts that:

1. **Setup and Configure** - Detect and configure Clang environment
2. **Build and Validate** - Enforce Clang usage during compilation
3. **Test and Verify** - Validate toolchain compatibility
4. **Clean and Reset** - Provide comprehensive cleanup capabilities
5. **CI/CD Integration** - Automated continuous integration with Clang

## Build Scripts

### 1. Environment Setup Script

**File**: `scripts/setup_clang_environment.sh`

**Purpose**: Detects, configures, and validates Clang toolchain setup.

**Features**:
- Auto-detects available Clang versions (10-18)
- Configures ccache for faster builds
- Creates CMake toolchain file
- Validates C/C++ compilation
- Generates environment activation script

**Usage**:
```bash
./scripts/setup_clang_environment.sh
```

**Outputs**:
- `.clang_env` - Environment configuration
- `cmake/ClangToolchain.cmake` - CMake toolchain file
- `activate_clang_env.sh` - Environment activation script

### 2. Clang Build Script

**File**: `scripts/build_with_clang.sh`

**Purpose**: Enforces Clang usage throughout the build process.

**Features**:
- Loads Clang environment configuration
- Configures CMake with Clang toolchain
- Builds with enforced Clang usage
- Validates executable compilation
- Runs optional tests

**Usage**:
```bash
# Standard release build
./scripts/build_with_clang.sh

# Debug build
BUILD_TYPE=Debug ./scripts/build_with_clang.sh

# Build with tests
ENABLE_TESTS=ON ./scripts/build_with_clang.sh

# Clean build
./scripts/build_with_clang.sh --clean
```

**Environment Variables**:
- `BUILD_TYPE` - Debug or Release (default: Release)
- `ENABLE_MLIR` - ON or OFF (default: ON)
- `ENABLE_TESTS` - ON or OFF (default: OFF)
- `ENABLE_EXAMPLES` - ON or OFF (default: ON)
- `PARALLEL_JOBS` - Number of parallel jobs (default: nproc)

### 3. Toolchain Validation Script

**File**: `scripts/validate_clang_toolchain.sh`

**Purpose**: Comprehensive validation of Clang toolchain setup.

**Features**:
- Tests Clang compiler availability
- Validates C++ standards support (14, 17, 20)
- Tests LLVM/MLIR compatibility features
- Validates SystemC compatibility
- Tests CMake toolchain integration
- Checks optimization levels
- Tests ccache integration

**Usage**:
```bash
./scripts/validate_clang_toolchain.sh
```

**Output**: 
- Console validation report
- `clang_validation_report.txt` - Detailed report

### 4. Clean Build Script

**File**: `scripts/clean_build_clang.sh`

**Purpose**: Comprehensive cleanup of build artifacts and caches.

**Features**:
- Cleans all build directories
- Removes CMake generated files
- Clears ccache contents
- Cleans temporary files
- Removes object files and binaries
- Cleans third-party build artifacts
- Optional environment reset

**Usage**:
```bash
# Interactive cleanup
./scripts/clean_build_clang.sh

# Non-interactive cleanup
./scripts/clean_build_clang.sh --force

# Cleanup and reset environment
./scripts/clean_build_clang.sh --reset-env
```

### 5. CI/CD Build Script

**File**: `scripts/ci_clang_build.sh`

**Purpose**: Automated CI/CD builds with Clang enforcement.

**Features**:
- Auto-detects CI environment (GitHub Actions, GitLab CI, etc.)
- Installs system dependencies
- Configures Clang for CI
- Handles git submodules
- Generates build artifacts
- Collects build statistics
- Provides failure diagnostics

**Usage**:
```bash
# Standard CI build
./scripts/ci_clang_build.sh

# Skip dependency installation
SKIP_DEPS=true ./scripts/ci_clang_build.sh
```

## GitHub Actions Workflow

**File**: `.github/workflows/clang-build.yml`

**Features**:
- Matrix builds across Clang versions (13-16)
- Builds in both Debug and Release modes
- Comprehensive testing
- ccache integration for speed
- Static analysis with clang-tidy
- Performance benchmarking
- Artifact generation and upload

**Triggered by**:
- Push to main/master/develop branches
- Pull requests to main/master
- Nightly scheduled builds

## CMake Toolchain Integration

**File**: `cmake/ClangToolchain.cmake`

**Purpose**: Enforces Clang usage at the CMake level.

**Features**:
- Sets Clang as the C/C++ compiler
- Configures RTTI and exceptions for LLVM
- Sets up atomic library linking
- Applies consistent compiler flags
- Enables optimization settings

**Usage**:
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/ClangToolchain.cmake
```

## Environment Configuration

### Clang Environment File

**File**: `.clang_env`

Contains environment variables for Clang usage:
```bash
export CC="clang-15"
export CXX="clang++-15"
export CMAKE_C_COMPILER="clang-15"
export CMAKE_CXX_COMPILER="clang++-15"
```

### Activation Script

**File**: `activate_clang_env.sh`

Activates the Clang environment:
```bash
source ./activate_clang_env.sh
```

## Supported Clang Versions

The build system supports Clang versions 10 through 18, with automatic detection and fallback:

1. **Preferred**: clang-15, clang-16, clang-17, clang-18
2. **Supported**: clang-10 through clang-14
3. **Fallback**: Generic `clang` if versioned not available

## Build Modes

### 1. Development Build
```bash
source ./activate_clang_env.sh
./scripts/build_with_clang.sh
```

### 2. Debug Build
```bash
BUILD_TYPE=Debug ./scripts/build_with_clang.sh
```

### 3. Release with Tests
```bash
BUILD_TYPE=Release ENABLE_TESTS=ON ./scripts/build_with_clang.sh
```

### 4. CI Build
```bash
./scripts/ci_clang_build.sh
```

## Performance Optimizations

### ccache Integration

The build system automatically configures ccache for faster rebuilds:

- **Cache Size**: Configurable (default 10GB for development, 5GB for CI)
- **Compression**: Enabled for space efficiency
- **Automatic Setup**: Creates ccache symlinks for Clang

### Parallel Building

- Uses `ninja` build system for optimal parallelization
- Defaults to `$(nproc)` parallel jobs
- Configurable via `PARALLEL_JOBS` environment variable

## Validation and Testing

### Toolchain Validation

The validation script performs comprehensive tests:

1. **Compiler Availability**: Tests C and C++ compiler presence
2. **Standards Support**: Validates C++14, C++17, and C++20 support
3. **LLVM Compatibility**: Tests atomic operations, RTTI, exceptions
4. **SystemC Compatibility**: Tests C++14 compatibility requirements
5. **CMake Integration**: Tests toolchain file functionality
6. **Optimization**: Tests different optimization levels
7. **ccache Integration**: Validates cache functionality

### Build Validation

Each build includes validation steps:

1. **Executable Creation**: Verifies sv2sc executable exists
2. **Basic Functionality**: Tests --version and --help commands
3. **Dependencies**: Checks dynamic library dependencies
4. **File Information**: Displays executable metadata

## Troubleshooting

### Common Issues

1. **Clang Not Found**
   ```bash
   # Install Clang
   sudo apt-get install clang
   # Re-run setup
   ./scripts/setup_clang_environment.sh
   ```

2. **Submodules Missing**
   ```bash
   git submodule update --init --recursive
   ```

3. **Build Failures**
   ```bash
   # Clean and rebuild
   ./scripts/clean_build_clang.sh --force
   ./scripts/setup_clang_environment.sh
   ./scripts/build_with_clang.sh
   ```

4. **ccache Issues**
   ```bash
   # Clear ccache
   ccache --clear
   # Or disable ccache
   export CCACHE_DISABLE=1
   ```

### Diagnostic Commands

```bash
# Validate toolchain
./scripts/validate_clang_toolchain.sh

# Check environment
source ./activate_clang_env.sh
echo $CC $CXX

# CMake configuration test
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/ClangToolchain.cmake --help
```

## Integration with Development Workflow

### Daily Development

1. **Setup** (once): `./scripts/setup_clang_environment.sh`
2. **Activate**: `source ./activate_clang_env.sh`
3. **Build**: `./scripts/build_with_clang.sh`
4. **Test**: `ENABLE_TESTS=ON ./scripts/build_with_clang.sh`

### Clean Development

1. **Clean**: `./scripts/clean_build_clang.sh`
2. **Rebuild**: `./scripts/build_with_clang.sh --clean`

### Validation Workflow

1. **Validate**: `./scripts/validate_clang_toolchain.sh`
2. **Review**: Check `clang_validation_report.txt`
3. **Fix Issues**: Based on validation results

## CI/CD Integration

### GitHub Actions

The provided workflow supports:
- Multiple Clang versions (13-16)
- Debug and Release builds
- Comprehensive testing
- Static analysis
- Performance benchmarking
- Artifact collection

### Other CI Systems

The `ci_clang_build.sh` script supports:
- GitLab CI
- Jenkins
- CircleCI
- Generic CI systems

## File Structure

```
sv2sc/
├── scripts/
│   ├── setup_clang_environment.sh    # Environment setup
│   ├── build_with_clang.sh           # Clang-enforced build
│   ├── validate_clang_toolchain.sh   # Toolchain validation
│   ├── clean_build_clang.sh          # Comprehensive cleanup
│   └── ci_clang_build.sh             # CI/CD build script
├── cmake/
│   └── ClangToolchain.cmake          # CMake toolchain file
├── .github/workflows/
│   └── clang-build.yml               # GitHub Actions workflow
├── docs/
│   └── CLANG_BUILD_SYSTEM.md         # This documentation
├── .clang_env                        # Environment configuration
├── activate_clang_env.sh             # Environment activation
└── build_info_clang.txt              # Build information
```

## Best Practices

1. **Always use the provided scripts** for consistent Clang enforcement
2. **Activate the environment** before manual CMake operations
3. **Validate the toolchain** after system changes
4. **Clean builds** when switching between configurations
5. **Use ccache** for faster incremental builds
6. **Check build artifacts** in CI for deployment readiness

## Support and Maintenance

### Updating Clang Version

1. Install new Clang version
2. Re-run setup: `./scripts/setup_clang_environment.sh`
3. Validate: `./scripts/validate_clang_toolchain.sh`
4. Update CI configuration if needed

### Adding New Build Features

1. Update relevant script(s)
2. Update CMake toolchain if needed
3. Update documentation
4. Test with validation script
5. Update CI workflow

This Clang build system ensures consistent, reliable builds with the Clang toolchain while providing comprehensive tooling for development, testing, and continuous integration.