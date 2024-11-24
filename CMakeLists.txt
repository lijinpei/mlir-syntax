cmake_minimum_required(VERSION 3.20)

project(MLIRSyntax)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(MLIR_SYNTAX_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(MLIR_SYNTAX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLIR_SYNTAX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include(find_llvm_mlir_clang)

add_subdirectory(tools)

include(export_llvm_capi_to_rust)
include(export_mlir_capi_to_rust)
