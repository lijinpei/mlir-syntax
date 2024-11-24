SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"
for i in "${SCRIPT_DIR}/src/"*.rs
do
  name="${i#"${SCRIPT_DIR}/src/"}"
  name="${name%".rs"}"
  if [[ "$name" == "common" || "$name" == "lib" ]]
  then
    continue
  fi
  echo ${name}
  cargo run --bin ${name} 2>&1 |FileCheck-20 src/${name}.rs
done
# cmake .. -G Ninja -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir/
# export RUSTFLAGS="-Lnative=/development/build/tools/MLIR-C -Lnative=/usr/lib/llvm-20/lib"
# export LD_LIBRARY_PATH="/development/build/tools/MLIR-C:/usr/lib/llvm-20/lib"
