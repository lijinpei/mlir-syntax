from serveur.action import Action
from serveur.common import get_llvm_install_dir, get_project_root_dir
import subprocess
import shlex
import os.path


class Test(Action):
    def __init__(self):
        pass

    @classmethod
    def desc(cls):
        return "Test the project"

    @classmethod
    def append_lib_path(cls, paths, lib_path):
        ld_lib_path, rust_flags = paths
        if ld_lib_path:
            ld_lib_path += ':' + lib_path
        else:
            ld_lib_path = lib_path
        new_flag = f"-Lnative={lib_path}"
        if rust_flags:
            rust_flags += f" {new_flag}"
        else:
            rust_flags = new_flag
        return ld_lib_path, rust_flags

    def add_arguments(self, parser):
        parser.add_argument('--llvm', default=None)

    def do_action(self, args):
        llvm_install = args.llvm
        if llvm_install is None:
            llvm_install = get_llvm_install_dir()
        assert llvm_install is not None
        llvm_lib_dir = os.path.join(llvm_install, 'lib')
        env = dict(os.environ)
        lib_paths = (env.get('LD_LIBRARY_PATH', ''), env.get('RUSTFLAGS', ''))
        lib_paths = self.append_lib_path(lib_paths, llvm_lib_dir)
        if not os.path.isfile(os.path.join(llvm_lib_dir, 'libMLIR-C.so')):
            mlir_c_dir = os.path.join(
                get_project_root_dir(), 'build', 'tools', 'MLIR-C')
            lib_paths = self.append_lib_path(lib_paths, mlir_c_dir)
        env['LD_LIBRARY_PATH'] = lib_paths[0]
        env['RUSTFLAGS'] = lib_paths[1]
        lit_test_dir = os.path.join(
            get_project_root_dir(), 'tests', 'mlir-ffi-rs', 'src', 'bin')
        subprocess.run(['lit', lit_test_dir, '-v'], env=env, check=True)
