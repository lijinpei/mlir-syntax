if [ "$#" -eq 1 ] ;
then
  build_dir="$1"
else
  build_dir="${proj_dir}/build"
fi
rm -rf "${build_dir}"
