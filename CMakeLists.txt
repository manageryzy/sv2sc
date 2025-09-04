cmake_minimum_required(VERSION 3.20)

# Force Clang compiler detection before project() call
if(NOT CMAKE_C_COMPILER AND NOT CMAKE_CXX_COMPILER)
    # Try to find clang first
    find_program(CLANG_C_COMPILER clang)
    find_program(CLANG_CXX_COMPILER clang++)
    
    if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
        set(CMAKE_C_COMPILER ${CLANG_C_COMPILER} CACHE FILEPATH "Clang C compiler" FORCE)
        set(CMAKE_CXX_COMPILER ${CLANG_CXX_COMPILER} CACHE FILEPATH "Clang C++ compiler" FORCE)
        message(STATUS "Found and using Clang compilers:")
        message(STATUS "  C: ${CLANG_C_COMPILER}")
        message(STATUS "  CXX: ${CLANG_CXX_COMPILER}")
    else()
        message(WARNING "Clang not found, falling back to system default compilers")
        message(WARNING "Install clang for better compatibility: sudo apt install clang")
    endif()
endif()

# Default to Clang toolchain if none specified (can be overridden via -DCMAKE_TOOLCHAIN_FILE=...)
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolchain.cmake" CACHE FILEPATH "Default to Clang toolchain")
endif()

project(sv2sc VERSION 1.0.0 LANGUAGES CXX)

# Enable parallel builds by default - use all available CPU cores
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
    message(STATUS "Enabling parallel builds with ${N} cores")
endif()

# Set C++ standard to C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enhanced Clang toolchain detection and configuration
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "✅ Using Clang compiler: ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "   Compiler path: ${CMAKE_CXX_COMPILER}")
    
    # Verify Clang version is compatible (14+)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(WARNING "Clang version ${CMAKE_CXX_COMPILER_VERSION} may not fully support C++20. Consider upgrading.")
    endif()
    
    # Enhanced Clang-specific settings for slang compatibility
    # Note: stdlib selection is handled in the toolchain file
    
    # Enable color diagnostics for better debugging
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    
    # Set toolchain-specific definitions
    add_compile_definitions(SV2SC_USING_CLANG=1)
    add_compile_definitions(SV2SC_CPP20_SUPPORT=1)
    
else()
    message(WARNING "⚠️  GCC detected: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(WARNING "   This may cause compatibility issues with slang library.")
    message(WARNING "   Consider using Clang toolchain: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/ClangToolchain.cmake")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure ccache for faster compilation
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(WARNING "ccache not found - consider installing for faster builds")
endif()

# Enhanced compiler flags with toolchain-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wno-unused-parameter -fstandalone-debug")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto=thin")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -fstandalone-debug")
    
    # Use lld linker if available
    find_program(LLD_LINKER ld.lld)
    if(LLD_LINKER)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fuse-ld=lld")
        message(STATUS "Using LLD linker for optimized builds")
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    
    # Add GCC-specific compatibility flags for slang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -fcoroutines")
    
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Build configuration options
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
# MLIR is always enabled now - removed option
set(SV2SC_ENABLE_MLIR ON CACHE INTERNAL "MLIR is always enabled")
option(SV2SC_USE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
option(SV2SC_ENABLE_LTO "Enable Link Time Optimization" OFF)

# Advanced Clang-specific build options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    option(SV2SC_ENABLE_ASAN "Enable AddressSanitizer" OFF)
    option(SV2SC_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
    option(SV2SC_ENABLE_UBSAN "Enable UBSanitizer" OFF)
    option(SV2SC_ENABLE_MSAN "Enable MemorySanitizer" OFF)
    
    # Apply sanitizer flags
    if(SV2SC_ENABLE_ASAN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        message(STATUS "AddressSanitizer enabled")
    endif()
    
    if(SV2SC_ENABLE_TSAN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
        message(STATUS "ThreadSanitizer enabled")
    endif()
    
    if(SV2SC_ENABLE_UBSAN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
        message(STATUS "UBSanitizer enabled")
    endif()
    
    if(SV2SC_ENABLE_MSAN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=memory")
        message(STATUS "MemorySanitizer enabled")
    endif()
endif()

# Toolchain validation and configuration summary
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "=== Clang Toolchain Configuration ===")
    message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
    message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")
endif()

# Toolchain file is set above; users can override via -DCMAKE_TOOLCHAIN_FILE
include(cmake/Dependencies.cmake)
include(cmake/SystemCTestUtils.cmake)

include_directories(include)
include_directories(external)

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples are now included in the main tests directory
# BUILD_EXAMPLES is handled within tests/CMakeLists.txt