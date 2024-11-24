include(TableGen)
set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)
add_tablegen(rust-ops-generator OpGen
  EXPORT OpGen
  rust_ops_generator.cpp
  op_gen.cpp)
target_link_libraries(rust-ops-generator
PRIVATE
MLIRTblgenLib)
target_include_directories(rust-ops-generator PRIVATE "${MLIR_INCLUDE_DIRS}" "${LLVM_INCLUDE_DIRS}")
include(gen_rust_ops)
