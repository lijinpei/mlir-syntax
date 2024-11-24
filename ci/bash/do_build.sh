source "${SCRIPT_DIR}/detect_tools.sh"

proj_dir="${SCRIPT_DIR}/../.."
build_dir="${proj_dir}/build"
src_dir="${proj_dir}"
do_clean=false
num_jobs=
mlir=

while : ; do
  if [ "$#" -eq 0 ]
  then
    break
  elif [ "$#" -eq 1 ]
  then
    invalid_argument
  fi
  arg="$1"
  shift
  case "$arg" in
    --src-dir)
      src_dir="$1"
      shift
      ;;
    --build-dir)
      build_dir="$1"
      shift
      ;;
    -j)
      num_jobs="$2"
      shift
      ;;
    --clean|-c)
      do_clean=true
      ;;
    --mlir)
      mlir="$1"
      shift
      ;;
    *)
      invalid_argument
      ;;
  esac
done

if [ "${do_clean}" = "true" ]
then
  do_action clean "${build_dir}"
fi

if [ -z "${mlir}" ] ;
then
  for ver in 18 19 20 ;
  do
    if [ -d "/usr/lib/llvm-${ver}" ]
    then
      mlir=/usr/lib/llvm-${ver}
      break
    fi
  done
  if [ -z "${mlir}" ]
  then
    failed_to_detect_mlir
  fi
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
cmake -G Ninja -B ${build_dir} -S ${src_dir} -DMLIR_DIR="${mlir}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER="${CLANG_CC}" -DCMAKE_CXX_COMPILER="${CLANG_CXX}"
if [ -z "${num_jobs}" ] ;
then
  num_jobs_arg=""
else
  num_jobs_arg="-j ${num_jobs}"
fi
cmake --build ${build_dir} -j ${num_jobs}
