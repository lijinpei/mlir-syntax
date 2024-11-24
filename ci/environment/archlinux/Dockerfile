FROM archlinux:latest
ARG LLVM_VER=20
RUN pacman -Syu --noconfirm wget python python-pip python-virtualenv clang cmake ninja rust && pacman -Syu --noconfirm cargo
RUN <<EOF
set -exo pipefail
wget -q http://192.168.1.6:8080/job/mlir-syntax/job/package-llvm/lastSuccessfulBuild/artifact/workspace/build_llvm/LLVM-20.0.0git-Linux.tar.zst
mkdir /opt/llvm-${LLVM_VER}/
tar xvf LLVM-20.0.0git-Linux.tar.zst -C /opt/llvm-${LLVM_VER}/ --strip-components=1
rm -rf LLVM-20.0.0git-Linux.tar.zst
EOF
