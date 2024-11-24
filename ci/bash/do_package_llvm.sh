source "${SCRIPT_DIR}/detect_tools.sh"

llvm_dir=
build_dir=
build_type=debug

while : ; do
  if [ "$#" -lt 1 ] ;
  then
    break
  fi
  arg="$1"
  shift
  case "$arg" in
    --llvm-dir)
      llvm_dir="$1"
      shift
      ;;
    --build-dir)
      build_dir="$1"
      shift
      ;;
    --build-type)
      build_type="$1"
      shift
      ;;
    *)
      invalid_argument
      ;;
  esac
done

if [ -z "${llvm_dir}" ] || [ -z "${build_dir}" ] ;
then
  invalid_argument
fi

if command -v ccache >/dev/null 2>&1
then
  ccache_args="-DLLVM_CCACHE_BUILD=ON"
else
  ccache_args=
fi

rm -rf "${build_dir}"
cmake -S "${llvm_dir}/llvm" -B "${build_dir}" -G Ninja -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" -DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_C_COMPILER="${CLANG_CC}" -DCMAKE_CXX_COMPILER="${CLANG_CXX}" -DLLVM_ENABLE_LLD=ON -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_TARGETS_TO_BUILD=all -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(which python3) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCPACK_GENERATOR=TZST -DLLVM_BUILD_LLVM_DYLIB=ON "${ccache_args}"
cmake --build ${build_dir} -t package
