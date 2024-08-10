FROM debian:sid
RUN apt-get update && apt-get install -y ca-certificates
ADD --chown=0:0 --chmod=644 llvm.sources /etc/apt/sources.list.d/llvm.sources
ADD --chown=0:0 --chmod=644 https://apt.llvm.org/llvm-snapshot.gpg.key /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN apt-get update && apt-get install -y chezscheme libmlir-20
