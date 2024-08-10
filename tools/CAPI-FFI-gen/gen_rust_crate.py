import sys
import os
import os.path


def gen_module(root_path, ofile_name="mod.rs"):
    all_mods = []
    for file in os.listdir(root_path):
        if file == "lib.rs" or file == "mod.rs":
            continue
        path = os.path.join(root_path, file)
        if os.path.isfile(path):
            all_mods.append(file[:-len(".rs")])
        else:
            all_mods.append(file)
            gen_module(path)
    with open(os.path.join(root_path, ofile_name), "w") as fout:
        fout.write(f"#![allow(non_snake_case)]\n")
        for mod in sorted(all_mods):
            fout.write(f"pub mod {mod};\n")


def gen_cargo_proj(root_path):
    proj = root_path.split('/')[-1]
    with open(root_path + "/Cargo.toml", "w") as fout:
        fout.write(f"""
[package]
name = "{proj}"
version = "0.1.0"
edition = "2021"

[dependencies]
""")
        for x in sys.argv[2:]:
            fout.write(x)
    gen_module(root_path + "/src/", "lib.rs")


if __name__ == '__main__':
    gen_cargo_proj(sys.argv[1])
