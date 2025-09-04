# Clang toolchain file for sv2sc project
# This file configures CMake to use Clang/Clang++ compilers

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Find and set Clang compilers
find_program(CMAKE_C_COMPILER clang REQUIRED)
find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)

if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Clang toolchain file requires clang and clang++ to be installed")
endif()

# Set compiler flags for better compatibility with slang and MLIR
set(CMAKE_C_FLAGS_INIT "-fPIC")
set(CMAKE_CXX_FLAGS_INIT "-fPIC -stdlib=libc++")

# Enable color diagnostics
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -fcolor-diagnostics")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -fcolor-diagnostics")

# Ensure libc++ is used consistently
set(CMAKE_EXE_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")

# Set the sysroot (optional, system default)
# set(CMAKE_SYSROOT /usr)

# Configure find_program to use the cross-compiler tools
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Ensure we use the correct linker
set(CMAKE_LINKER clang++)
