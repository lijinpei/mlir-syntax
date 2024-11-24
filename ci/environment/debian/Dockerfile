FROM debian:sid
RUN apt-get update && apt-get install -y ca-certificates
ARG APT_SOURCE_TUNA=off
ARG LLVM_VER=20
ADD --chown=0:0 --chmod=644 debian.tuna.sources /etc/apt/sources.list.d/debian.tuna.sources.backup
RUN if [ "$APT_SOURCE_TUNA" != "off" ] ; then cd /etc/apt/sources.list.d/; mv debian.tuna.sources.backup debian.sources ; fi
RUN apt-get update && apt-get install -y cmake ninja-build pybind11-dev cargo python3 python3-dev python3-pip python3-venv libzstd-dev wget zstd clang
#ADD --chown=0:0 --chmod=644 llvm.sources /etc/apt/sources.list.d/llvm.sources
#ADD --chown=0:0 --chmod=644 https://apt.llvm.org/llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc
#RUN apt-get update && apt-get install -y llvm-${LLVM_VER} clang-${LLVM_VER} lld-${LLVM_VER}
#RUN apt-get update && apt-get install -y libllvm${LLVM_VER} libmlir-${LLVM_VER} libmlir-${LLVM_VER}-dev mlir-${LLVM_VER}-tools llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev
RUN <<EOF
set -exo pipefail
wget -q http://192.168.1.6:8080/job/mlir-syntax/job/package-llvm/lastSuccessfulBuild/artifact/workspace/build_llvm/LLVM-20.0.0git-Linux.tar.zst
mkdir /opt/llvm-${LLVM_VER}/
tar xvf LLVM-20.0.0git-Linux.tar.zst -C /opt/llvm-${LLVM_VER}/ --strip-components=1
rm -rf LLVM-20.0.0git-Linux.tar.zst
EOF
