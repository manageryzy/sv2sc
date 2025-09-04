add_subdirectory(core)
add_subdirectory(translator)
add_subdirectory(codegen)
add_subdirectory(utils)
add_subdirectory(mlir)

set(SV2SC_SOURCES
    main.cpp
    sv2sc.cpp
)

add_executable(sv2sc ${SV2SC_SOURCES})

# Add explicit dependency on MLIR build targets (always required now)
if(TARGET llvm_deps)
    add_dependencies(sv2sc llvm_deps)
endif()
if(TARGET mlir-headers)
    add_dependencies(sv2sc mlir-headers)
endif()
if(TARGET mlir-generic-headers)
    add_dependencies(sv2sc mlir-generic-headers)
endif()

target_link_libraries(sv2sc PRIVATE
    sv2sc_core
    sv2sc_translator
    sv2sc_codegen
    sv2sc_utils
    sv2sc_mlir
    CIRCTSystemCTransforms
    slang::slang
    fmt::fmt
    spdlog::spdlog
)

target_include_directories(sv2sc PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Ensure C++20 standard propagation and Clang compatibility
target_compile_features(sv2sc PRIVATE cxx_std_20)

# Clang-specific configuration applied via toolchain file

# Enable sanitizers if requested (Clang-specific)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(SV2SC_ENABLE_ASAN)
        target_compile_options(sv2sc PRIVATE -fsanitize=address)
        target_link_options(sv2sc PRIVATE -fsanitize=address)
    endif()
    if(SV2SC_ENABLE_TSAN)
        target_compile_options(sv2sc PRIVATE -fsanitize=thread)
        target_link_options(sv2sc PRIVATE -fsanitize=thread)
    endif()
    if(SV2SC_ENABLE_UBSAN)
        target_compile_options(sv2sc PRIVATE -fsanitize=undefined)
        target_link_options(sv2sc PRIVATE -fsanitize=undefined)
    endif()
    if(SV2SC_ENABLE_MSAN)
        target_compile_options(sv2sc PRIVATE -fsanitize=memory)
        target_link_options(sv2sc PRIVATE -fsanitize=memory)
    endif()
endif()

# MLIR is always enabled now - removed conditional compilation definitions