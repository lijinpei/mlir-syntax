function (export_scheme_ffi src_dir dest_dir target)
  # TODO: let user pass extra clang flags
  # FIXME: remove python
  find_package (Python COMPONENTS Development)
  file(GLOB_RECURSE src_headers RELATIVE ${src_dir} "${src_dir}/*.h")
  set(all_out_stub_files "")
  foreach (header ${src_headers})
    set(header_file "${src_dir}/${header}")
    string(REGEX REPLACE "[.]h$" ".s" cs_file "${dest_dir}/${header}")
    string(REGEX REPLACE "[.]h$" ".cpp" cpp_file "${dest_dir}/${header}")
    add_custom_command(OUTPUT "${cs_file}" "${cpp_file}" COMMAND "${LLVM_TOOLS_BINARY_DIR}/clang++" -x c++ -I "${Python_INCLUDE_DIRS}" -I "${LLVM_INCLUDE_DIRS}" -fplugin="$<TARGET_FILE:mlir_capi_exporter>" -Xclang -plugin -Xclang mlir-scheme-ffi -Xclang -plugin-arg-mlir-scheme-ffi -Xclang "${cs_file}" -Xclang -plugin-arg-mlir-scheme-ffi -Xclang "${cpp_file}" -fsyntax-only "${header_file}" WORKING_DIRECTORY "${dest_dir}" DEPENDS mlir_capi_exporter)
    list(APPEND all_out_stub_files "${cpp_file}")
  endforeach()
  add_library(${target} SHARED ${all_out_stub_files})
endfunction()
