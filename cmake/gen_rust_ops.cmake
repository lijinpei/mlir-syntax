# FIXME: this is heuristic
list(GET MLIR_INCLUDE_DIRS 0 mlir_main_inc_dir)

function (gen_rust_ops dialect dialect_name)
  cmake_parse_arguments(PARSE_ARGV 1 gen_rust_args "" "SRC_DIR;DEST_DIR;TD;CONFIG" "")
  if (NOT DEFINED gen_rust_args_SRC_DIR)
    set(gen_rust_args_SRC_DIR "${mlir_main_inc_dir}")
  endif()
  set(output_file "${gen_rust_args_DEST_DIR}/${dialect}.rs")
  add_custom_command(OUTPUT "${output_file}"
    COMMAND rust-ops-generator
    --gen-rust-ops --rust-config "${gen_rust_args_CONFIG}"
    -I "${mlir_main_inc_dir}"
    -I "${mlir_main_inc_dir}/mlir/Dialect/${dialect}/IR/"
    "${gen_rust_args_TD}"
    -dialect ${dialect_name}
    -o "${output_file}"
    WORKING_DIRECTORY "${gen_rust_args_DEST_DIR}"
    DEPENDS rust-ops-generator ${gen_rust_args_CONFIG})
  add_custom_target(rust_ops_gen_${dialect} DEPENDS "${output_file}")
  set(_output_file "${output_file}" PARENT_SCOPE)
endfunction()

set(gen_rust_ops_dir "${MLIR_SYNTAX_BINARY_DIR}/mlir_ops_rust/")
set(src_file "${MLIR_SYNTAX_SOURCE_DIR}/cmake/rust_ops.yaml.in")
set(conf_file "${gen_rust_ops_dir}/rust_ops.yaml")
add_custom_command(OUTPUT "${conf_file}"
  COMMAND
  "${CMAKE_COMMAND}"
  -D "SRC=${src_file}" -D "DEST=${conf_file}"
  -P ${MLIR_SYNTAX_SOURCE_DIR}/cmake/gen_configure.cmake
  DEPENDS "${src_file}")

set(dialects
Affine ArmSME Func LLVMIR MPI PDL SCF Tosa Vector
AMDGPU ArmSVE Complex GPU Math NVGPU PDLInterp Shape X86Vector
AMX Async ControlFlow Index MemRef OpenACC Polynomial SparseTensor Transform XeGPU
Arith Bufferization DLTI IRDL Mesh Ptr SPIRV UB
ArmNeon EmitC Linalg MLProgram OpenMP Quant Tensor)

function (get_extra_opts dialect)
  set(dialect_dir "${mlir_main_inc_dir}/mlir/Dialect/${dialect}/")
  set(extra_opts)
  set(td_file "${dialect_dir}/IR/${dialect}Ops.td")

  set(dialect_name ${dialect})
  string(REGEX REPLACE "Ops$" "" dialect_name ${dialect_name})
  string(TOLOWER ${dialect_name} dialect_name)

  if (dialect STREQUAL "ArmSME")
    set(dialect_name arm_sme)
  endif()
  if (dialect STREQUAL "ArmSVE")
    set(dialect_name arm_sve)
  endif()
  if (dialect STREQUAL "ArmNeon")
    set(dialect_name arm_neon)
  endif()
  if (dialect STREQUAL "ControlFlow")
    set(dialect_name cf)
  endif()
  if (dialect STREQUAL "LLVMIR")
    set(dialect_name llvm)
  endif()
  if (dialect STREQUAL "MLProgram")
    set(dialect_name ml_program)
  endif()
  if (dialect STREQUAL "PDLInterp")
    set(dialect_name pdl_interp)
  endif()
  if (dialect STREQUAL "SparseTensor")
    set(dialect_name sparse_tensor)
  endif()

  set(_tmp_list_1 "AMDGPU;EmitC;Polynomial;NVGPU;ArmSVE")
  if (dialect IN_LIST _tmp_list_1)
    set(td_file "${dialect_dir}/IR/${dialect}.td")
  endif()
  set(_tmp_list_2 "OpenACC;OpenMP")
  if (dialect IN_LIST _tmp_list_2)
    set(td_file "${dialect_dir}/${dialect}Ops.td")
  endif()
  set(_tmp_list_3 "ArmNeon;AMX;DLTI;X86Vector")
  if (dialect IN_LIST _tmp_list_3)
    set(td_file "${dialect_dir}/${dialect}.td")
  endif()
  if (dialect STREQUAL "LLVMIR")
    set(td_file "${dialect_dir}/LLVMOps.td")
  endif()
  set(_extra_opts "${extra_opts}" PARENT_SCOPE)
  set(_td_file "${td_file}" PARENT_SCOPE)
  set(_dialect_name "${dialect_name}" PARENT_SCOPE)
endfunction()

set(gen_rust_ops_all)
set(dialect_skip_list "OpenACC;OpenMP")
foreach (dialect ${dialects})
  if (dialect IN_LIST dialect_skip_list)
    # FIXME:
    continue()
  endif()
  get_extra_opts(${dialect})
  gen_rust_ops(${dialect} ${_dialect_name} DEST_DIR "${gen_rust_ops_dir}/mlir_ops/src/" TD "${_td_file}" CONFIG "${conf_file}" ${_extra_opts})
  list(APPEND gen_rust_ops_all "${_output_file}")
endforeach()
add_custom_target(all_rust_ops_gen DEPENDS ${gen_rust_ops_all})
