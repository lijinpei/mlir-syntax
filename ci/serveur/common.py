import functools
import shutil
import os


@functools.cache
def get_common_arg_parser_args():
    return {'fromfile_prefix_chars': '@', 'allow_abbrev': False}


@functools.cache
def get_project_root_dir():
    res = get_project_ci_dir()
    res = os.path.dirname(res)
    return res


@functools.cache
def get_project_ci_dir():
    res = os.path.abspath(__file__)
    res = os.path.dirname(res)
    res = os.path.dirname(res)
    return res


@functools.cache
def get_llvm_install_dir():
    llvm_vers = [20, 19, 18]
    llvm_dirs = ['/opt/llvm-{ver}', '/opt/llvm', '/usr/lib/llvm-{ver}']
    for dir_pat in llvm_dirs:
        for ver in llvm_vers:
            ins_dir = (dir_pat + '/bin/clang').format(ver=ver)
            if os.path.isfile(ins_dir):
                return dir_pat.format(ver=ver)
    return None


@functools.cache
def get_system_clang():
    """This means to return a release build clang, which is used to compile things."""
    if shutil.which('clang') is not None:
        return 'clang'
    return os.path.join(get_llvm_install_dir(), 'bin', 'clang')
