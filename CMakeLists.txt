cmake_minimum_required(VERSION 3.20)

project(MLIRSyntax)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(MLIR REQUIRED MODULE)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include(AddLLVM)

add_subdirectory(tools)
