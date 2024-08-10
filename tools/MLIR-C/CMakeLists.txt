list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

if (NOT TARGET MLIR-C)
	add_library(MLIR-C SHARED stub.cpp)
	get_property(mlir_capi_libs GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

	set(embed_libs)
	foreach (lib ${mlir_capi_libs})
		list(APPEND embed_libs "${LLVM_LIBRARY_DIRS}/lib${lib}.a")
	endforeach()
	target_link_libraries(MLIR-C -Wl,--whole-archive ${embed_libs} -Wl,--no-whole-archive MLIR)
endif()
