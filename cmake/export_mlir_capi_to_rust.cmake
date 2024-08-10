include(CAPI_FFI_Gen)

set(all_capi_headers)
set(extra_opts)
foreach(inc_dir ${MLIR_INCLUDE_DIRS})
  file(GLOB_RECURSE capi_headers "${inc_dir}/mlir-c/*.h")
  foreach(header ${capi_headers})
    get_filename_component(dir "${header}" DIRECTORY)
    get_filename_component(dir "${dir}" NAME)
    if (NOT dir STREQUAL "Python")
      list(APPEND all_capi_headers "${header}")
    endif()
  endforeach()
  list(APPEND extra_opts -I "${inc_dir}")
endforeach()
foreach(inc_dir ${LLVM_INCLUDE_DIRS})
  list(APPEND extra_opts -I "${inc_dir}")
endforeach()

set(SRC_DIR "mlir-c")
set(src_file "${MLIR_SYNTAX_SOURCE_DIR}/cmake/mlir.yaml.in")
set(conf_file "${MLIR_SYNTAX_BINARY_DIR}/ffi/mlir.yaml")
set(LINK_LIB "MLIR-C")
add_custom_command(OUTPUT "${conf_file}"
  COMMAND "${CMAKE_COMMAND}" -D "SRC=${src_file}" -D "DEST=${conf_file}" -D "SRC_DIR=${SRC_DIR}" -D "LINK_LIB=${LINK_LIB}" -P ${MLIR_SYNTAX_SOURCE_DIR}/cmake/gen_capi_configure.cmake
  DEPENDS "${src_file}")

# FIXME: this is heuristic
list(GET MLIR_INCLUDE_DIRS 0 mlir_main_inc_dir)
export_ffi("${all_capi_headers}" RUST TARGET mlir_ffi_rust SRC_DIR "${mlir_main_inc_dir}/mlir-c/" DEST_DIR "${MLIR_SYNTAX_BINARY_DIR}/ffi/mlir/src/" CONFIG "${conf_file}" EXTRA_OPTS ${extra_opts})
