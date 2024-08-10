add_llvm_library(mlir_capi_exporter MODULE mlir_capi_exporter.cpp PLUGIN_TOOL clang)
#target_link_libraries(mlir_capi_exporter PRIVATE clangAST clangBasic clangFrontend)
target_include_directories(mlir_capi_exporter PRIVATE ${CLANG_INCLUDE_DIRS})

include(ExportSchemeFFI)
# FIXME: remove python
find_package (Python COMPONENTS Development)
export_scheme_ffi("${MLIR_INCLUDE_DIRS}/mlir-c" "${CMAKE_CURRENT_BINARY_DIR}/mlir-c" mlir_c_ffi)
target_link_libraries(mlir_c_ffi PUBLIC MLIR-C Python::Module)
target_include_directories(mlir_c_ffi PRIVATE ${MLIR_INCLUDE_DIRS})
