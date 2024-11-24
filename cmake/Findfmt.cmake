include_guard(GLOBAL)
add_subdirectory("${MLIR_SYNTAX_THIRD_PARTY_DIR}/fmt" "${MLIR_SYNTAX_BINARY_DIR}/third_party/fmt")
target_compile_options(fmt PUBLIC -fPIC)
