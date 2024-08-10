get_property(found GLOBAL PROPERTY MLIR_SYNTAX_FMT_FOUND SET)
if (found)
  return()
endif()

add_subdirectory("${MLIR_SYNTAX_THIRD_PARTY_DIR}/fmt" "${MLIR_SYNTAX_BINARY_DIR}/third_party/fmt")
target_compile_options(fmt PUBLIC -fPIC)
set_property(GLOBAL PROPERTY MLIR_SYNTAX_FMT_FOUND)
