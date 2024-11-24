import lit.formats
import os
import shutil

config.name = "MLIR CAPI Rust"
config.test_format = lit.formats.ShTest()
config.suffixes = [".rs"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.dirname(__file__)
config.environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
config.environment["RUSTFLAGS"] = os.environ["RUSTFLAGS"]
config.environment["HOME"] = os.environ["HOME"]


def maybe_export(x):
    x_v = os.environ.get(x, None)
    if x_v is not None:
        config.environment[x] = x_v


maybe_export('http_proxy')
maybe_export('https_proxy')
maybe_export('HTTP_PROXY')
maybe_export('HTTPS_PROXY')
this_dir_path = os.path.dirname(os.path.realpath(__file__))
serveur_common_path = os.path.join(
    this_dir_path, '..', '..', '..', '..', 'ci', 'serveur', 'common.py')
file_check = 'FileCheck'
if shutil.which(file_check) is None:
    found_suffix = None
    for suffix in [20, 19, 18]:
        file_check_with_suffix = f'FileCheck-{suffix}'
        if shutil.which(file_check_with_suffix) is not None:
            found_suffix = file_check_with_suffix
            break
    if found_suffix is not None:
        file_check = found_suffix
    else:
        with open(serveur_common_path, 'r') as serveur_common_file:
            serveur_common = serveur_common_file.read()
        globals = {}
        exec(serveur_common, globals)
        llvm_install_dir = globals['get_llvm_install_dir']()
        if llvm_install_dir is not None:
            file_check = os.path.join(llvm_install_dir, 'bin', 'FileCheck')
config.substitutions.append(('%FileCheck', file_check))
