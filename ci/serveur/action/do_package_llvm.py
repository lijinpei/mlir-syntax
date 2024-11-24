from serveur.action import Action
from serveur.common import get_system_clang
import shutil
import subprocess
import shlex
import sys
import shutil


class PackageLlvm(Action):
    @classmethod
    def desc(cls):
        return "Build and package llvm"

    def add_arguments(self, parser):
        parser.add_argument('--llvm-dir', required=True)
        parser.add_argument('--build-dir', required=True)
        parser.add_argument('--build-type', default='Debug',
                            choices=['Release', 'Debug'])

    def get_ccache_arg(self):
        if shutil.which('ccache') is not None:
            return '-DLLVM_CCACHE_BUILD=ON'
        return ''

    def do_action(self, args):
        subprocess.run(['rm', '-rf', args.build_dir], check=True)
        clang_cc = get_system_clang()
        clang_cxx = clang_cc + '++'
        cmake_invoke = f'''cmake
            -S {args.llvm_dir}/llvm
            -B {args.build_dir}
            -G Ninja
            -DLLVM_ENABLE_PROJECTS="clang;mlir;lld"
            -DCMAKE_BUILD_TYPE={args.build_type}
            -DCMAKE_C_COMPILER="{clang_cc}"
            -DCMAKE_CXX_COMPILER="{clang_cxx}"
            -DLLVM_ENABLE_LLD=ON
            -DLLVM_TARGETS_TO_BUILD=all
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON
            -DPython3_EXECUTABLE={sys.executable}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            -DCPACK_GENERATOR=TZST
            -DLLVM_BUILD_LLVM_DYLIB=ON
            -DLLVM_LINK_LLVM_DYLIB=ON
            -DLLVM_BUILD_UTILS=ON
            -DLLVM_INSTALL_UTILS=ON
            -DLLVM_DISTRIBUTION_COMPONENTS="LLVM;llvm-headers;cmake-exports;MLIR;mlir-headers;mlir-cmake-exports;MLIR-C;FileCheck;libclang;libclang-headers;clang-cmake-exports"
            -DCMAKE_INSTALL_PREFIX=/
            -DMLIR_BUILD_MLIR_C_DYLIB=ON
            -DLLVM_OPTIMIZED_TABLEGEN=ON
            {self.get_ccache_arg()}'''
        subprocess.run(shlex.split(cmake_invoke), check=True)
        subprocess.run(
            ['cmake',  '--build',  f'{args.build_dir}',  '-t', 'package'], check=True)
