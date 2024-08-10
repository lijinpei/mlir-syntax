function (export_ffi headers)
  cmake_parse_arguments(PARSE_ARGV 1 export_ffi_arg "RUST;CHEZSCHEME" "CONFIG;TARGET;SRC_DIR;DEST_DIR" "EXTRA_OPTS")
  if (NOT DEFINED export_ffi_arg_CONFIG)
    set(export_ffi_arg_CONFIG "${MLIR_SYNTAX_SOURCE_DIR}/tools/CAPI-FFI-gen/config.yaml")
  endif()
  if (NOT DEFINED export_ffi_arg_SRC_DIR)
    set(export_ffi_arg_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if (NOT DEFINED export_ffi_arg_DEST_DIR)
    set(export_ffi_arg_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if (NOT DEFINED export_ffi_arg_TARGET)
    set(export_ffi_arg_TARGET "${export_ffi_arg_SRC_DIR}.ffi")
  endif()
  set(out_files)
  set(all_output_files)
  foreach (header ${headers})
    set(lang_opts)
    set(output_files)
    cmake_path(RELATIVE_PATH header BASE_DIRECTORY "${export_ffi_arg_SRC_DIR}"
                                    OUTPUT_VARIABLE header_path)
    get_filename_component(header_path "${header_path}" DIRECTORY)
    get_filename_component(header_name "${header}" NAME_WLE)
    if (export_ffi_arg_RUST)
      string(REPLACE "/" ";" header_path_list "${header_path}")
      list(TRANSFORM header_path_list APPEND "_" OUTPUT_VARIABLE header_path_list)
      list(JOIN header_path_list "/" rust_header_path)
      if (NOT rust_header_path STREQUAL "")
        string(APPEND rust_header_path "/")
      endif()
      set(rust_output_file "${rust_header_path}${header_name}.rs")
      list(APPEND lang_opts
        "-Xclang" "-plugin-arg-capi_ffi_gen" "-Xclang" "rust"
        "-Xclang" "-plugin-arg-capi_ffi_gen" "-Xclang" "${rust_output_file}")
      list(APPEND output_files "${export_ffi_arg_DEST_DIR}/${rust_output_file}")
    endif()
    if (export_ffi_arg_CHEZSCHEME)
      set(chez_scheme_output_file "${export_ffi_arg_DEST_DIR}/${header}.cs")
      list(APPEND lang_opts
        "-Xclang" "-plugin-arg-capi_ffi_gen" "-Xclang" "chez-scheme"
        "-Xclang" "-plugin-arg-capi_ffi_gen" "-Xclang" "${chez_scheme_output_file}")
      list(APPEND output_files "${chez_scheme_output_file}")
    endif()
    set(input_file "${header}")
    add_custom_command(OUTPUT "${output_files}"
      COMMAND "${LLVM_TOOLS_BINARY_DIR}/clang" -fsyntax-only
      -fplugin="$<TARGET_FILE:capi_ffi_generator>" -Xclang -plugin -Xclang capi_ffi_gen
      "-Xclang" "-plugin-arg-capi_ffi_gen" "-Xclang" "${export_ffi_arg_CONFIG}"
      ${lang_opts}
      ${export_ffi_arg_EXTRA_OPTS}
      "${input_file}"
      WORKING_DIRECTORY "${export_ffi_arg_DEST_DIR}"
      DEPENDS capi_ffi_generator ${export_ffi_arg_CONFIG})
    list(APPEND all_output_files "${output_files}")
  endforeach()
  add_custom_target(${export_ffi_arg_TARGET} DEPENDS "${all_output_files}")
endfunction()
