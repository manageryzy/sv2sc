# Clang Toolchain Migration Guide

**Date**: 2025-08-27  
**Status**: CRITICAL - Required for C++20 and MLIR integration  
**Priority**: HIGH

## Executive Summary

The sv2sc project requires migration to Clang toolchain to resolve critical build failures related to C++20 standard compatibility and MLIR integration. The current GCC setup cannot handle the advanced C++20 features required by the slang library and MLIR dependencies.

## Critical Issues Addressed

### 1. C++20 Standard Compatibility
- **Problem**: slang library requires C++20 features (`std::source_location`, `std::convertible_to`)
- **Solution**: Clang 14+ provides better C++20 support than current GCC setup

### 2. MLIR Integration Requirements
- **Problem**: MLIR/CIRCT integration requires specific compiler capabilities
- **Solution**: Clang is the recommended compiler for LLVM/MLIR projects

### 3. Build System Conflicts
- **Problem**: Mixed C++ standard requirements across dependencies
- **Solution**: Clang provides better C++ standard management

## Installation Guide

### Ubuntu/Debian Systems

```bash
# Update package list
sudo apt update

# Install Clang 14 (recommended minimum version)
sudo apt install clang-14 clang-tools-14 libc++-14-dev libc++abi-14-dev

# Optional: Install additional tools
sudo apt install ninja-build ccache

# Set Clang as default (optional)
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# Verify installation
clang++ --version
```

### CentOS/RHEL Systems

```bash
# Install EPEL repository
sudo yum install epel-release

# Install Clang
sudo yum install clang clang-tools-extra

# Or for newer versions
sudo dnf install clang clang-tools-extra libc++-devel libc++abi-devel

# Verify installation
clang++ --version
```

### macOS Systems

```bash
# Install via Homebrew
brew install llvm

# Add to PATH
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
clang++ --version
```

## Build Configuration

### Environment Setup

```bash
# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export CC=clang
export CXX=clang++
export CXXFLAGS="-std=c++20"
```

### CMake Configuration

```bash
# Create build directory
mkdir build && cd build

# Configure with Clang
cmake .. \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON

# Build project
make -j$(nproc)
```

### Advanced Configuration

For projects requiring specific libc++ usage:

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_BUILD_TYPE=Release
```

## Dependency Updates

### CMakeLists.txt Updates

Update main CMakeLists.txt to enforce Clang:

```cmake
# Enforce Clang compiler
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "This project requires Clang compiler. Please set CMAKE_CXX_COMPILER to clang++")
endif()

# Ensure C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Clang-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
endif()
```

### Dependencies.cmake Updates

```cmake
# Configure slang with C++20 for Clang
set(SLANG_USE_MIMALLOC OFF CACHE BOOL "Disable mimalloc for slang" FORCE)
set(SLANG_USE_CPPTRACE OFF CACHE BOOL "Disable cpptrace for slang" FORCE)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "Disable slang tests" FORCE)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++20 required for slang" FORCE)
```

## Validation Steps

### 1. Compiler Verification

```bash
# Check Clang installation
clang++ --version | head -1
# Expected: clang version 14.0.0 or higher

# Check C++20 support
echo '#include <source_location>' | clang++ -x c++ -std=c++20 -c -
# Should compile without errors
```

### 2. Build Verification

```bash
# Clean build test
rm -rf build && mkdir build && cd build
export CC=clang && export CXX=clang++
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
make -j$(nproc)
```

### 3. Functionality Verification

```bash
# Run basic tests
./sv2sc --version
./sv2sc --help

# Run unit tests (if available)
ctest -V
```

## Common Issues and Solutions

### Issue 1: "clang++: command not found"

**Solution**: Install Clang package for your system

```bash
# Ubuntu/Debian
sudo apt install clang

# CentOS/RHEL
sudo yum install clang
```

### Issue 2: libc++ not found

**Solution**: Install libc++ development packages

```bash
# Ubuntu/Debian
sudo apt install libc++-dev libc++abi-dev

# CentOS/RHEL
sudo yum install libc++-devel libc++abi-devel
```

### Issue 3: C++20 features not recognized

**Solution**: Use correct flags and newer Clang version

```bash
# Ensure using Clang 14+
clang++ --version

# Use explicit flags
export CXXFLAGS="-std=c++20 -stdlib=libc++"
```

### Issue 4: MLIR build failures

**Solution**: Enable full LLVM/CIRCT build or use mock mode

```bash
# For full MLIR support
cmake .. -DSV2SC_ENABLE_FULL_LLVM=ON -DSV2SC_ENABLE_FULL_CIRCT=ON

# For development with mock MLIR
cmake .. -DSV2SC_ENABLE_FULL_LLVM=OFF -DSV2SC_ENABLE_FULL_CIRCT=OFF
```

## Performance Optimizations

### Compilation Speed

```bash
# Use Ninja build system (faster than Make)
cmake .. -G Ninja -DCMAKE_CXX_COMPILER=clang++
ninja

# Use ccache for faster rebuilds
sudo apt install ccache
export CXX="ccache clang++"
```

### Runtime Performance

```bash
# Enable LTO (Link Time Optimization)
cmake .. -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -flto"

# Use specific CPU optimizations
cmake .. -DCMAKE_CXX_FLAGS="-march=native"
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install Clang
  run: |
    sudo apt update
    sudo apt install clang-14 clang-tools-14 libc++-14-dev libc++abi-14-dev
    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

- name: Build with Clang
  env:
    CC: clang
    CXX: clang++
  run: |
    mkdir build && cd build
    cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
    make -j$(nproc)
```

## Testing Strategy

### 1. Component Testing
- Test individual libraries with Clang
- Verify C++20 feature compatibility
- Test MLIR integration components

### 2. Integration Testing
- Full build verification
- Runtime functionality testing
- Performance regression testing

### 3. Compatibility Testing
- Test on different Clang versions (14, 15, 16)
- Test on different operating systems
- Test with different libc++ versions

## Migration Timeline

### Phase 1: Immediate (Hours)
1. Install Clang toolchain on all development systems
2. Update build scripts and documentation
3. Test basic compilation

### Phase 2: Short-term (1-2 Days)
1. Resolve any remaining build issues
2. Update CI/CD pipelines
3. Verify all tests pass

### Phase 3: Medium-term (1 Week)
1. Optimize build performance
2. Documentation and training
3. Establish best practices

## Success Criteria

- ✅ Clang++ compiler available and working
- ✅ Project builds successfully with C++20 standard
- ✅ All unit tests pass
- ✅ MLIR integration functional (or properly mocked)
- ✅ Build time comparable to or better than GCC
- ✅ No runtime performance regressions

## Support and Resources

### Documentation
- [Clang C++20 Status](https://clang.llvm.org/cxx_status.html)
- [MLIR with Clang](https://mlir.llvm.org/getting_started/)
- [libc++ Documentation](https://libcxx.llvm.org/)

### Community
- LLVM Discourse: https://discourse.llvm.org/
- Stack Overflow: [clang] tag
- GitHub Issues: Project-specific issues

### Troubleshooting
- Check compiler version compatibility
- Verify all dependencies are built with same compiler
- Use verbose CMake output for debugging: `cmake .. --verbose`

---

**Next Steps**: Execute installation commands and update build configuration files according to this guide.