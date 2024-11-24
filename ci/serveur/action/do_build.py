from serveur.action import Action
from serveur.common import get_llvm_install_dir, get_project_root_dir, get_system_clang
import subprocess
import shlex
import os


class Build(Action):
    def __init__(self):
        pass

    @classmethod
    def desc(cls):
        return "Build the project"

    def add_arguments(self, parser):
        parser.add_argument('--src-dir', default=None)
        parser.add_argument('--build-dir', default=None)
        parser.add_argument('--num-jobs', default=None)
        parser.add_argument('--clean', default=False, action='store_true')
        parser.add_argument('--llvm', default=None)

    def do_action(self, args):
        build_dir = args.build_dir
        if build_dir is None:
            build_dir = os.path.join(get_project_root_dir(), 'build')
        if args.clean:
            subprocess.run(['rm', '-rf', build_dir], check=True)
        llvm_install = args.llvm
        if llvm_install is None:
            llvm_install = get_llvm_install_dir()
        assert llvm_install is not None
        clang_cc = get_system_clang()
        clang_cxx = clang_cc + '++'
        src_dir = args.src_dir
        if src_dir is None:
            src_dir = get_project_root_dir()
        cmake_invoke = f'''cmake
        -G Ninja
        -B {build_dir}
        -S {src_dir}
        -DLLVM_DIR="{llvm_install}/lib/cmake/llvm"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCMAKE_C_COMPILER="{clang_cc}"
        -DCMAKE_CXX_COMPILER="{clang_cxx}"'''
        subprocess.run(shlex.split(cmake_invoke), check=True)
        num_jobs = args.num_jobs
        if num_jobs is not None:
            num_jobs_args = ['-j', f'{num_jobs}']
        else:
            num_jobs_args = []
        subprocess.run(
            ['cmake', '--build', f'{build_dir}', *num_jobs_args], check=True)
