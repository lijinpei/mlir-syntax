get_property(found GLOBAL PROPERTY MLIR_SYNTAX_YAML_CAPP_FOUND SET)
if (found)
  return()
endif()

add_subdirectory("${MLIR_SYNTAX_THIRD_PARTY_DIR}/yaml-cpp" "${MLIR_SYNTAX_BINARY_DIR}/third_party/yaml-cpp")
set_property(GLOBAL PROPERTY MLIR_SYNTAX_YAML_CAPP_FOUND)
