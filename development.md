## Prerequisite

- Rust
- Prebuilt LLVM

## Supported OS

All functionalities are tested on the following OS:
    - debian-sid
    - archlinux
with llvm of version
    - latest release
    - latest trunk
and installed by
    - package.sh
    - official deb packages
    - a build tree

Of course, archlinux has no LLVM official debian pacakges, which means that combination should be eliminated from the support matrix.

A llvm build tree has different include path, cmake setup from an installed llvm tree.

Generally, llvm package from linux distribution packagers, miss some clang and mlir development components, lag behind the latest trunk. So build against distro package is not supported. To get a working llvm installment, see https://apt.llvm.org/ or our prebuilt package. (FIXME: put our prebuilt llvm package somewhere.)

## Requested Services
    - Http proxy
        * For access to github, llvm deb packages, docker hub etc.
    - Git server
        * Access to github is slow for large projects even with proxy.
        * To debug CI pipeline without needing to push WIP commits to github.
        * There is in fact no server running some service, just a public gituser account with some bare repo.
    - Docker image registry
        * Must-have for CI, optional for local workflow.
    - Jenkins server
        * Holds multiple pipelines.
        * Also holds some artifacts (a.k.a. LLVM.sh), maybe needs a separate object storage.
