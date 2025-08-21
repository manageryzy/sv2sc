# Dependencies using FetchContent for local third-party directories
include(FetchContent)

# Check if submodules are initialized
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/slang/CMakeLists.txt")
    message(FATAL_ERROR "Git submodules not found. Please run: git submodule update --init --recursive")
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

# Configure slang to use minimal features
set(SLANG_USE_MIMALLOC OFF CACHE BOOL "Disable mimalloc for slang" FORCE)
set(SLANG_USE_CPPTRACE OFF CACHE BOOL "Disable cpptrace for slang" FORCE)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "Disable slang tests" FORCE)

# Make all dependencies available
FetchContent_MakeAvailable(fmt CLI11 spdlog slang)

# Handle SystemC separately with C++14 compatibility
set(CMAKE_CXX_STANDARD_TEMP ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD 14)

# Configure SystemC build options
set(SYSTEMC_BUILD_SHARED_LIBS OFF CACHE BOOL "Build SystemC as shared library" FORCE)
set(SYSTEMC_ENABLE_PTHREADS ON CACHE BOOL "Enable pthreads for SystemC" FORCE)
set(SYSTEMC_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions in SystemC" FORCE)

FetchContent_MakeAvailable(SystemC)

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