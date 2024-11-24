find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG
  HINTS "${LLVM_LIBRARY_DIR}/cmake/mlir")
find_package(Clang REQUIRED CONFIG
  HINTS "${LLVM_LIBRARY_DIR}/cmake/clang")
include(AddLLVM)
