# Dependencies using FetchContent for local third-party directories
include(FetchContent)

# Check if submodules are initialized
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/slang/CMakeLists.txt")
    message(FATAL_ERROR "Git submodules not found. Please run: git submodule update --init --recursive")
endif()

# Check if CIRCT submodule is initialized when MLIR is enabled
if(SV2SC_ENABLE_MLIR AND NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/circt/CMakeLists.txt")
    message(FATAL_ERROR "CIRCT submodule not found. Please run: git submodule update --init --recursive")
endif()

# Declare local dependencies using FetchContent
FetchContent_Declare(
    fmt
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/fmt
)

FetchContent_Declare(
    CLI11
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/CLI11
)

FetchContent_Declare(
    spdlog
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/spdlog
)

FetchContent_Declare(
    SystemC
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/SystemC
)

FetchContent_Declare(
    slang
    SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/slang
)

# CIRCT and LLVM (only if MLIR is enabled)
if(SV2SC_ENABLE_MLIR)
    # Always use in-tree CIRCT/LLVM build
    message(STATUS "Configuring in-tree CIRCT/LLVM build from third-party/circt")
    FetchContent_Declare(
        llvm-project
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/circt/llvm
    )
endif()

# Configure slang with Clang-specific settings
set(SLANG_USE_MIMALLOC OFF CACHE BOOL "Disable mimalloc for slang" FORCE)
set(SLANG_USE_CPPTRACE OFF CACHE BOOL "Disable cpptrace for slang" FORCE)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "Disable slang tests" FORCE)

# Ensure slang uses C++20 standard for compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Force C++20 for slang when using Clang
    set(CMAKE_CXX_STANDARD 20 CACHE STRING "Force C++20 for slang" FORCE)
    message(STATUS "Configuring slang with Clang C++20 support")
    
    # Set additional Clang-specific flags for slang compilation
    set(SLANG_CXX_FLAGS "-stdlib=libc++ -std=c++20")
    set(SLANG_LINKER_FLAGS "-stdlib=libc++ -lc++abi")
else()
    message(WARNING "slang may not compile correctly with GCC 13. Consider using Clang toolchain.")
    message(WARNING "To use Clang: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/ClangToolchain.cmake")
endif()

# Make all dependencies available - always use in-tree versions
message(STATUS "Building in-tree dependencies: fmt, CLI11, spdlog, slang")
FetchContent_MakeAvailable(fmt CLI11 spdlog slang)

# Ensure we use the in-tree fmt
if(TARGET fmt::fmt)
    message(STATUS "Using in-tree fmt library")
else()
    message(STATUS "Creating fmt::fmt alias for in-tree fmt")
    add_library(fmt::fmt ALIAS fmt)
endif()

# Handle SystemC separately with C++14 compatibility
set(CMAKE_CXX_STANDARD_TEMP ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD 14)

# Configure SystemC build options
set(SYSTEMC_BUILD_SHARED_LIBS OFF CACHE BOOL "Build SystemC as shared library" FORCE)
set(SYSTEMC_ENABLE_PTHREADS ON CACHE BOOL "Enable pthreads for SystemC" FORCE)
set(SYSTEMC_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions in SystemC" FORCE)

# Save current compiler flags
set(ORIGINAL_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# Add compiler flags to suppress SystemC warnings - enhanced for Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds -Wno-cast-function-type -Wno-unused-parameter -Wno-deprecated-declarations -Wno-unused-variable -Wno-sign-compare")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds -Wno-cast-function-type -Wno-unused-parameter")
endif()

FetchContent_MakeAvailable(SystemC)

# Restore original compiler flags after SystemC build
set(CMAKE_CXX_FLAGS ${ORIGINAL_CXX_FLAGS})

# Restore original C++ standard
set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_TEMP})

# Add Catch2 for testing (only if tests are enabled)
if(BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third-party/Catch2
    )
    
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/third-party/Catch2/extras)
    include(Catch)
endif()

# Create aliases for consistent naming if they don't already exist
if(NOT TARGET SystemC::systemc)
    add_library(SystemC::systemc ALIAS systemc)
endif()

# MLIR/CIRCT Dependencies (optional)
if(SV2SC_ENABLE_MLIR)
    message(STATUS "MLIR-based translation pipeline enabled")
    
    # Use real CIRCT from third-party source
    message(STATUS "Using real CIRCT from third-party source")
    
    # Set paths to CIRCT source
    set(CIRCT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/circt")
    set(LLVM_SOURCE_DIR "${CIRCT_SOURCE_DIR}/llvm")
    
    # Set include directories for MLIR and CIRCT
    set(LLVM_INCLUDE_DIRS
        "${LLVM_SOURCE_DIR}/llvm/include"
        "${CMAKE_BINARY_DIR}/llvm/include"
    )
    set(MLIR_INCLUDE_DIRS
        "${LLVM_SOURCE_DIR}/mlir/include"
        "${CMAKE_BINARY_DIR}/llvm/tools/mlir/include"
    )
    set(CIRCT_INCLUDE_DIRS
        "${CIRCT_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}/circt/include"
    )
    
    # Configure LLVM build options
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "Enable MLIR project" FORCE)
    set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "Build for host target only" FORCE)
    set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI" FORCE)
    set(LLVM_ENABLE_EH ON CACHE BOOL "Enable exception handling" FORCE)
    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
    set(LLVM_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Don't include tests" FORCE)
    
    # Use C++20 for modern LLVM/MLIR/slang compatibility
    # Note: LLVM 15+ supports C++20, keeping C++17 as fallback for older versions
    if(LLVM_VERSION_MAJOR GREATER_EQUAL 15)
        set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard version" FORCE)
    else()
        set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard version" FORCE)
        message(STATUS "Using C++17 for LLVM ${LLVM_VERSION_MAJOR} compatibility")
    endif()
    
    # Enhanced atomic library linking for Clang and GCC
    find_library(ATOMIC_LIB atomic)
    if(ATOMIC_LIB OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang-specific atomic linking
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ flags" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic" CACHE STRING "Linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -latomic" CACHE STRING "Shared linker flags" FORCE)
        else()
            # GCC-style atomic linking
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -latomic" CACHE STRING "C flags" FORCE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -latomic" CACHE STRING "C++ flags" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic" CACHE STRING "Linker flags" FORCE)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -latomic" CACHE STRING "Shared linker flags" FORCE)
        endif()
        set(CMAKE_REQUIRED_LIBRARIES "atomic" CACHE STRING "Required libraries for atomic operations" FORCE)
        list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
    endif()
    
    # Set LLVM/MLIR variables needed by CIRCT for integrated build
    set(LLVM_MAIN_SRC_DIR "${LLVM_SOURCE_DIR}/llvm")
    set(MLIR_MAIN_SRC_DIR "${LLVM_SOURCE_DIR}/mlir")
    set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "Enable MLIR execution engine" FORCE)
    
    # Always build LLVM in-tree for better compatibility with Clang
    message(STATUS "Building LLVM/MLIR from source (in-tree) with Clang - this may take some time")
    message(STATUS "LLVM source directory: ${LLVM_SOURCE_DIR}")
    message(STATUS "CIRCT source directory: ${CIRCT_SOURCE_DIR}")
    
    # Temporarily sanitize global flags for third-party superprojects (avoid LTO/leaking flags)
    set(SV2SC_ORIG_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(SV2SC_ORIG_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(SV2SC_ORIG_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    set(SV2SC_ORIG_CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(SV2SC_ORIG_CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

    # Remove LTO flags to avoid toolchain mismatch inside LLVM/CIRCT builds
    string(REPLACE "-flto=thin" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    # Disable building tests/python bindings and Z3 in subprojects to simplify deps
    set(LLVM_BUILD_TESTS OFF CACHE BOOL "Disable LLVM tests" FORCE)
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Disable LLVM include tests" FORCE)
    set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "Disable LLVM Z3 solver" FORCE)
    set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable MLIR Python bindings" FORCE)
    set(CIRCT_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable CIRCT Python bindings" FORCE)
    set(CIRCT_ENABLE_Z3_SOLVER OFF CACHE BOOL "Disable CIRCT Z3 solver" FORCE)
    set(CIRCT_BUILD_TESTING OFF CACHE BOOL "Disable CIRCT testing" FORCE)
    # Do not build CIRCT tools (circt-opt, circt-translate) as part of ALL
    set(CIRCT_BUILD_TOOLS OFF CACHE BOOL "Disable CIRCT tools" FORCE)
    set(CIRCT_BUILD_EXAMPLES OFF CACHE BOOL "Disable CIRCT examples" FORCE)

    # Add LLVM as subdirectory with proper dependency ordering (part of ALL)
    add_subdirectory(${LLVM_SOURCE_DIR}/llvm ${CMAKE_BINARY_DIR}/llvm)
    
    # Create dependency targets to ensure proper build order
    add_custom_target(llvm_deps
        DEPENDS llvm-tblgen mlir-tblgen
        COMMENT "Building LLVM dependencies"
    )
    
    # Ensure LLVM config is generated before other targets
    if(TARGET llvm-config)
        add_dependencies(llvm_deps llvm-config)
    endif()
    
    # Ensure core tablegen targets are available
    if(TARGET llvm-tblgen)
        set(LLVM_TABLEGEN_EXE llvm-tblgen)
    endif()
    if(TARGET mlir-tblgen)
        set(MLIR_TABLEGEN_EXE mlir-tblgen)
    endif()
    
    # Set up LLVM/MLIR paths after build
    set(LLVM_DIR "${CMAKE_BINARY_DIR}/llvm/lib/cmake/llvm")
    set(MLIR_DIR "${CMAKE_BINARY_DIR}/llvm/lib/cmake/mlir")
    
    # Include LLVM and MLIR directories
    include_directories(SYSTEM "${LLVM_SOURCE_DIR}/llvm/include")
    include_directories(SYSTEM "${CMAKE_BINARY_DIR}/llvm/include")
    include_directories(SYSTEM "${LLVM_SOURCE_DIR}/mlir/include")
    include_directories(SYSTEM "${CMAKE_BINARY_DIR}/llvm/tools/mlir/include")
    
    # Always build CIRCT in-tree after LLVM for compatibility
    message(STATUS "Building CIRCT from source - this requires LLVM to be built first")
    
    include_directories(SYSTEM "${CIRCT_SOURCE_DIR}/include")
    add_subdirectory(${CIRCT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/circt)
    
    # Create dependency target to ensure proper build order for CIRCT core
    add_custom_target(circt_deps
        DEPENDS
            circt-headers
            CIRCTSupport
            CIRCTHW
            CIRCTSeq
            CIRCTComb
            CIRCTSV
            CIRCTSystemC
        COMMENT "Building CIRCT core libraries and headers"
    )
    
    # Ensure CIRCT depends on LLVM
    if(TARGET llvm_deps)
        add_dependencies(circt_deps llvm_deps)
    endif()
    
    # Define real MLIR/CIRCT libraries with correct names
    set(MLIR_LIBRARIES
        MLIRIR
        MLIRSupport
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRFuncDialect
        MLIRArithDialect
        MLIRSCFDialect
    )
    
    set(CIRCT_LIBRARIES
        CIRCTHW
        CIRCTSystemC
        CIRCTSupport
    )
    
    # Create interface library with real dependencies
    add_library(MLIR::Dependencies INTERFACE IMPORTED)
    set_property(TARGET MLIR::Dependencies PROPERTY
        INTERFACE_LINK_LIBRARIES ${MLIR_LIBRARIES} ${CIRCT_LIBRARIES})
    set_property(TARGET MLIR::Dependencies PROPERTY
        INTERFACE_COMPILE_DEFINITIONS SV2SC_HAS_MLIR SV2SC_HAS_REAL_CIRCT)
    
    # Restore original flags after adding subprojects
    set(CMAKE_CXX_FLAGS "${SV2SC_ORIG_CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${SV2SC_ORIG_CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${SV2SC_ORIG_CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_EXE_LINKER_FLAGS "${SV2SC_ORIG_CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${SV2SC_ORIG_CMAKE_SHARED_LINKER_FLAGS}")
    
    # Integration status reporting
    message(STATUS "MLIR/CIRCT integration configured successfully (in-tree)")
else()
    message(STATUS "MLIR-based translation pipeline disabled")
endif()