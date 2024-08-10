cmake_minimum_required(VERSION 3.20)

project(MLIRSyntax)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(MLIR_SYNTAX_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(MLIR_SYNTAX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLIR_SYNTAX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

find_package(MLIR REQUIRED MODULE)
find_package(LLVM REQUIRED CONFIG
  HINTS "${MLIR_CMAKE_DIR}/../llvm")
find_package(Clang REQUIRED CONFIG
  HINTS "${MLIR_CMAKE_DIR}/../clang")

include(AddLLVM)

add_subdirectory(tools)
include(export_mlir_capi_to_rust)
include(export_llvm_capi_to_rust)
