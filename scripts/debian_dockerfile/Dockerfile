FROM debian:sid
RUN apt-get update && apt-get install -y ca-certificates
ARG APT_SOURCE_TUNA=off
ARG LLVM_VER=20
ADD --chown=0:0 --chmod=644 debian.tuna.sources /etc/apt/sources.list.d/debian.tuna.sources.backup
RUN if [ "$APT_SOURCE_TUNA" != "off" ] ; then cd /etc/apt/sources.list.d/; mv debian.tuna.sources.backup debian.sources ; fi
ADD --chown=0:0 --chmod=644 llvm.sources /etc/apt/sources.list.d/llvm.sources
ADD --chown=0:0 --chmod=644 https://apt.llvm.org/llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN apt-get update && apt-get install -y cmake ninja-build
RUN apt-get update && apt-get install -y llvm-${LLVM_VER} clang-${LLVM_VER} lld-${LLVM_VER}
RUN apt-get update && apt-get install -y libllvm${LLVM_VER} libmlir-${LLVM_VER} libmlir-${LLVM_VER}-dev mlir-${LLVM_VER}-tools llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev python3-dev
