get_property(found GLOBAL PROPERTY MLIR_SYNTAX_MLIR_FOUND SET)
if (found)
  return()
endif()

find_package(MLIR REQUIRED CONFIG)
set_property(GLOBAL PROPERTY MLIR_SYNTAX_MLIR_FOUND)
