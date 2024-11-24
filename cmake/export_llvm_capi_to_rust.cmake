include(CAPI_FFI_Gen)

set(all_capi_headers)
set(extra_opts)
foreach(inc_dir ${LLVM_INCLUDE_DIRS})
  file(GLOB_RECURSE capi_headers "${inc_dir}/llvm-c/*.h")
  list(APPEND all_capi_headers ${capi_headers})
  list(APPEND extra_opts -I "${inc_dir}")
endforeach()

set(SRC_DIR "llvm-c")
set(src_file "${MLIR_SYNTAX_SOURCE_DIR}/cmake/llvm.yaml.in")
set(conf_file "${MLIR_SYNTAX_BINARY_DIR}/ffi/llvm.yaml")
set(LINK_LIB "LLVM")
add_custom_command(OUTPUT "${conf_file}"
  COMMAND "${CMAKE_COMMAND}" -D "SRC=${src_file}" -D "DEST=${conf_file}" -D "SRC_DIR=${SRC_DIR}" -D "LINK_LIB=${LINK_LIB}" -P ${MLIR_SYNTAX_SOURCE_DIR}/cmake/gen_configure.cmake
  DEPENDS "${src_file}")

export_ffi("${all_capi_headers}" RUST TARGET llvm_ffi_rust SRC_DIR "${LLVM_MAIN_INCLUDE_DIR}/llvm-c/" DEST_DIR "${MLIR_SYNTAX_BINARY_DIR}/ffi/llvm/src/" CONFIG "${conf_file}" EXTRA_OPTS ${extra_opts})
