distro=
registry=
proxy=
llvm_ver=${DEFAULT_LLVM_VER}
tag=

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
    --distro)
      distro="$1"
      shift
      ;;
    --registry)
      registry="$1"
      shift
      ;;
    --llvm)
      llvm_ver="$1"
      shift
      ;;
    --proxy)
      proxy="$1"
      shift
      ;;
    *)
      invalid_argument
      ;;
  esac
done

if [ -z "${registry}" ]
then
  invalid_argument
fi

build_debian_image () {
  docker build -t "${image_name}" --no-cache ${proxy_arg} --build-arg APT_SOURCE_TUNA=on --build-arg LLVM_VER=${llvm_ver} --network host .
}

build_archlinux_image() {
  docker build -t "${image_name}" --no-cache ${proxy_arg} --build-arg LLVM_VER=${llvm_ver} --network host .
}

build_image() {
  if [ ! -z "${proxy}" ]
  then
    echo "using proxy ${proxy}"
    proxy_arg="--build-arg http_proxy=${proxy} --build-arg https_proxy=${proxy} --build-arg HTTP_PROXY=${proxy} --build-arg HTTPS_PROXY=${proxy}"
    export http_proxy="${proxy}"
    export https_proxy="${proxy}"
    export HTTP_PROXY="${proxy}"
    export HTTPS_PROXY="${proxy}"
  fi
  if [ -z ${llvm_ver} ]
  then
    invalid_argument
  fi
  if [ -z "${tag}" ]
  then
    tag=$(date +%Y_%m_%d.%s)
  fi
  env_dir="${SCRIPT_DIR}/../environment/${distro}"
  cd ${env_dir}
  image_name="${registry}/mlir-syntax-${distro}:${tag}"
  build_${distro}_image
  docker push "${image_name}"
  docker rmi "${image_name}"
}

case "${distro}" in
  debian|archlinux)
    build_image
    ;;
  *)
    invalid_argument
    ;;
esac
