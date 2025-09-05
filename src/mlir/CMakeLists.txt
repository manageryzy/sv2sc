# MLIR-based translation infrastructure
# MLIR is always required now

message(STATUS "Building MLIR translation infrastructure")

    set(MLIR_SOURCES
        MLIRContextManager.cpp
        MLIRTranslator.cpp
        SVToHWBuilder.cpp
        pipeline/SV2SCPassPipeline.cpp
        passes/HWToSystemCLoweringPass.cpp
        passes/PrepareHWForSystemCPass.cpp
        systemc/SystemCEmitter.cpp
        systemc/CIRCTCompatibleEmitter.cpp
        systemc/SystemCEmissionPatterns.cpp
    )

    set(MLIR_HEADERS
        ${CMAKE_SOURCE_DIR}/include/mlir/MLIRContextManager.h
        ${CMAKE_SOURCE_DIR}/include/mlir/SVToHWBuilder.h
        ${CMAKE_SOURCE_DIR}/include/mlir/pipeline/SV2SCPassPipeline.h
        ${CMAKE_SOURCE_DIR}/include/mlir/passes/HWToSystemCLoweringPass.h
        ${CMAKE_SOURCE_DIR}/include/mlir/passes/PrepareHWForSystemCPass.h
        ${CMAKE_SOURCE_DIR}/include/mlir/systemc/SystemCEmitter.h
        systemc/CIRCTCompatibleEmitter.h
        systemc/SystemCEmissionPatterns.h
    )

    # Create MLIR support library
    add_library(sv2sc_mlir STATIC ${MLIR_SOURCES})
    
    # Add dependency on LLVM/CIRCT build targets if they exist
    if(TARGET llvm_deps)
        add_dependencies(sv2sc_mlir llvm_deps)
    endif()
    if(TARGET circt_deps)
        add_dependencies(sv2sc_mlir circt_deps)
    endif()
    
    # Add explicit dependencies on MLIR header generation
    if(TARGET mlir-headers)
        add_dependencies(sv2sc_mlir mlir-headers)
    endif()
    if(TARGET mlir-generic-headers)
        add_dependencies(sv2sc_mlir mlir-generic-headers)
    endif()
    
    # Require real CIRCT libraries; fail hard if missing
    if(NOT TARGET CIRCTHW OR NOT TARGET CIRCTSystemC)
        message(FATAL_ERROR "CIRCT targets not found. Ensure submodules are initialized and in-tree LLVM/MLIR/CIRCT are added before building sv2sc MLIR components.")
    endif()

    # Create comprehensive MLIR/CIRCT dependency target that depends on ALL MLIR headers
    add_custom_target(sv2sc_mlir_deps_complete
        COMMENT "Ensuring all MLIR/CIRCT dependencies are built"
    )
    
    # Depend on the comprehensive MLIR header generation
    if(TARGET mlir-generic-headers)
        add_dependencies(sv2sc_mlir_deps_complete mlir-generic-headers)
    endif()
    if(TARGET mlir-headers)
        add_dependencies(sv2sc_mlir_deps_complete mlir-headers)
    endif()
    
    # Add dependency on core CIRCT libraries instead of tools
    set(CIRCT_CORE_LIBS
        CIRCTHW
        CIRCTSeq
        CIRCTComb
        CIRCTSV
        CIRCTSystemC
        CIRCTSystemCTransforms
        CIRCTSupport
        CIRCTHWToSystemC
        CIRCTExportSystemC
    )
    
    foreach(circt_lib IN LISTS CIRCT_CORE_LIBS)
        if(TARGET ${circt_lib})
            add_dependencies(sv2sc_mlir_deps_complete ${circt_lib})
        endif()
    endforeach()
    
    # Add dependency on LLVM core targets
    if(TARGET llvm_deps)
        add_dependencies(sv2sc_mlir_deps_complete llvm_deps)
    endif()
    
    # Add dependency on CIRCT core targets
    if(TARGET circt_deps)
        add_dependencies(sv2sc_mlir_deps_complete circt_deps)
    endif()
    
    # Add the comprehensive dependency to sv2sc_mlir
    add_dependencies(sv2sc_mlir sv2sc_mlir_deps_complete)

    # Include directories
    target_include_directories(sv2sc_mlir PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third-party/slang/include  # Add slang headers
    )
    # Add MLIR/CIRCT include directories for real implementation
    target_include_directories(sv2sc_mlir PUBLIC
        # Generated headers from build directory (tablegen output)
        ${CMAKE_BINARY_DIR}/llvm/tools/mlir/include
        ${CMAKE_BINARY_DIR}/llvm/include
        ${CMAKE_BINARY_DIR}/circt/include
        # Source directories
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${CIRCT_INCLUDE_DIRS}
    )

    # Link libraries based on availability
    target_link_libraries(sv2sc_mlir PUBLIC
        sv2sc_utils  # For logging
        slang::slang  # For SystemVerilog AST
    )
    
    # Link real MLIR/CIRCT libraries
    target_link_libraries(sv2sc_mlir PUBLIC
        MLIRIR
        MLIRSupport
        MLIRFuncDialect
        MLIRArithDialect
        MLIRSCFDialect
        MLIREmitCDialect
        CIRCTHW
        CIRCTSeq
        CIRCTComb
        CIRCTSV
        CIRCTSystemC
        CIRCTSystemCTransforms
        CIRCTSupport
        CIRCTHWToSystemC
        CIRCTExportSystemC
    )
    # MLIR and CIRCT are always enabled now - removed conditional compilation definitions

    # Set C++ standard specifically for MLIR to C++20
    target_compile_features(sv2sc_mlir PUBLIC cxx_std_20)
    message(STATUS "Using C++20 for MLIR components")

    # Enhanced compiler flags for LLVM/MLIR compatibility with Clang support
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(sv2sc_mlir PRIVATE 
            -Wno-unused-parameter
            -Wno-comment
            -Wno-deprecated-declarations
            -Wno-unused-variable
            -Wno-sign-compare
            -Wno-unused-function
            -fno-rtti-data
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(sv2sc_mlir PRIVATE 
            -Wno-unused-parameter
            -Wno-comment
            -Wno-deprecated-declarations
        )
    endif()

    # Create alias for consistent naming
    add_library(sv2sc::mlir ALIAS sv2sc_mlir)

    message(STATUS "MLIR translation infrastructure configured")