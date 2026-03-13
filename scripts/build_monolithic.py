"""
Build monolithic ngsolve-sparsesolv wheel.

This script:
1. Clones official NGSolve at the version pinned in NGSOLVE_VERSION
2. Applies SetGeomInfo patch to netgen submodule
3. Copies SparseSolv source into the ngsolve tree
4. Patches ngsolve's CMakeLists.txt and setup.py
5. Builds a single wheel containing netgen + ngsolve + sparsesolv_ngsolve

Usage:
    python scripts/build_monolithic.py [--skip-clone] [--skip-patch]

Requirements:
    pip install build scikit-build wheel numpy pybind11-stubgen==2.5
    pip install netgen-occt-devel==7.8.1 netgen-occt==7.8.1
    pip install mkl-devel mkl intel-cmplr-lib-rt
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BUILD_DIR = REPO_ROOT / "build-monolithic"
NGSOLVE_SRC = BUILD_DIR / "ngsolve-src"
DIST_DIR = REPO_ROOT / "dist"

PACKAGE_NAME = "ngsolve-sparsesolv"


def read_ngsolve_version():
    """Read pinned NGSolve version from NGSOLVE_VERSION file."""
    version_file = REPO_ROOT / "NGSOLVE_VERSION"
    version = version_file.read_text().strip()
    print(f"NGSolve version: {version}")
    return version


def clone_ngsolve(version, skip=False):
    """Clone official NGSolve at the pinned version."""
    if skip and NGSOLVE_SRC.exists():
        print(f"Skipping clone (--skip-clone), using existing {NGSOLVE_SRC}")
        return

    if NGSOLVE_SRC.exists():
        print(f"Removing existing {NGSOLVE_SRC}")
        shutil.rmtree(NGSOLVE_SRC, ignore_errors=True)

    print(f"Cloning NGSolve {version}...")
    subprocess.check_call([
        "git", "clone",
        "--branch", version,
        "--recurse-submodules",
        "--depth", "1",
        "https://github.com/NGSolve/ngsolve.git",
        str(NGSOLVE_SRC),
    ])
    print("Clone complete.")


def apply_patches(skip=False):
    """Apply SetGeomInfo and MSVC patches to netgen submodule."""
    if skip:
        print("Skipping patches (--skip-patch)")
        return

    netgen_dir = NGSOLVE_SRC / "external_dependencies" / "netgen"
    patches_dir = REPO_ROOT / "patches"

    for patch_file in sorted(patches_dir.glob("*.patch")):
        print(f"Applying patch: {patch_file.name}")
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=str(netgen_dir),
            capture_output=True,
        )
        if result.returncode == 0:
            subprocess.check_call(
                ["git", "apply", str(patch_file)],
                cwd=str(netgen_dir),
            )
            print(f"  Applied successfully.")
        else:
            print(f"  Patch already applied or not applicable, skipping.")


def copy_sparsesolv():
    """Copy SparseSolv source into the ngsolve tree."""
    sparsesolv_dest = NGSOLVE_SRC / "sparsesolv"
    if sparsesolv_dest.exists():
        shutil.rmtree(sparsesolv_dest)
    sparsesolv_dest.mkdir()

    # Copy headers
    src_include = REPO_ROOT / "include"
    dst_include = sparsesolv_dest / "include"
    shutil.copytree(src_include, dst_include)
    print(f"Copied headers: {src_include} -> {dst_include}")

    # Copy pybind11 module source
    src_module = REPO_ROOT / "ngsolve" / "python_module.cpp"
    shutil.copy2(src_module, sparsesolv_dest / "python_module.cpp")
    print(f"Copied: {src_module}")

    # Copy type stubs and py.typed
    for f in ["sparsesolv_ngsolve.pyi", "py.typed"]:
        src = REPO_ROOT / f
        if src.exists():
            shutil.copy2(src, sparsesolv_dest / f)
            print(f"Copied: {src}")

    # Create CMakeLists.txt for sparsesolv inside ngsolve tree
    cmake_content = _generate_sparsesolv_cmake()
    (sparsesolv_dest / "CMakeLists.txt").write_text(cmake_content)
    print(f"Created: {sparsesolv_dest / 'CMakeLists.txt'}")


def _generate_sparsesolv_cmake():
    """Generate CMakeLists.txt for building sparsesolv inside ngsolve tree."""
    return r"""# sparsesolv/CMakeLists.txt -- SparseSolv module built inside ngsolve tree
# Produces sparsesolv_ngsolve.pyd alongside ngsolve

if(NETGEN_USE_PYTHON)
    add_library(sparsesolv_ngsolve SHARED python_module.cpp)

    find_package(Python3 REQUIRED COMPONENTS Development)
    target_link_libraries(sparsesolv_ngsolve PRIVATE Python3::Module)
    # On Windows, DLL symbols are not re-exported transitively.
    # Must explicitly link against each NGSolve library whose symbols we reference.
    target_link_libraries(sparsesolv_ngsolve PUBLIC ngsolve ngcomp ngfem ngla ngbla ngstd)

    set_target_properties(sparsesolv_ngsolve PROPERTIES
        PREFIX ""
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    if(WIN32)
        set_target_properties(sparsesolv_ngsolve PROPERTIES SUFFIX ".pyd")
    else()
        set_target_properties(sparsesolv_ngsolve PROPERTIES SUFFIX ".so")
    endif()

    target_include_directories(sparsesolv_ngsolve PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(sparsesolv_ngsolve PRIVATE /arch:AVX2 /bigobj)
    else()
        target_compile_options(sparsesolv_ngsolve PRIVATE -mavx2 -mfma)
    endif()

    target_compile_definitions(sparsesolv_ngsolve PRIVATE
        SPARSESOLV_USE_NGSOLVE_TASKMANAGER
    )

    if(USE_MKL)
        if(MKL_INCLUDE_DIR)
            target_include_directories(sparsesolv_ngsolve PRIVATE ${MKL_INCLUDE_DIR})
        endif()
        if(MKL_LIBRARY)
            target_link_libraries(sparsesolv_ngsolve PRIVATE ${MKL_LIBRARY})
        endif()
        target_compile_definitions(sparsesolv_ngsolve PRIVATE SPARSESOLV_USE_MKL)
    endif()

    # Install .pyd into sparsesolv_ngsolve/ package directory
    install(TARGETS sparsesolv_ngsolve
        LIBRARY DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
        RUNTIME DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
        COMPONENT ngsolve
    )

    # Generate and install __init__.py
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
         "from .sparsesolv_ngsolve import *\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
            DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
            COMPONENT ngsolve)

    # Install type stubs
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sparsesolv_ngsolve.pyi")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/sparsesolv_ngsolve.pyi"
                DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
                RENAME __init__.pyi
                COMPONENT ngsolve)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/py.typed")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/py.typed"
                DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/sparsesolv_ngsolve
                COMPONENT ngsolve)
    endif()

    message(STATUS "SparseSolv: building sparsesolv_ngsolve module")
endif()
"""


def patch_ngsolve_cmake():
    """Add add_subdirectory(sparsesolv) to ngsolve's CMakeLists.txt."""
    cmake_path = NGSOLVE_SRC / "CMakeLists.txt"
    content = cmake_path.read_text()

    marker = "add_subdirectory(sparsesolv)"
    if marker in content:
        print("CMakeLists.txt already patched.")
        return

    # Add after add_subdirectory(emscripten) -- last existing subdirectory
    old = "add_subdirectory(emscripten)"
    new = f"add_subdirectory(emscripten)\nadd_subdirectory(sparsesolv)"

    if old not in content:
        # Fallback: add at the very end
        content += f"\n{marker}\n"
    else:
        content = content.replace(old, new, 1)

    cmake_path.write_text(content)
    print("Patched CMakeLists.txt: added add_subdirectory(sparsesolv)")


def patch_ngsolve_setup():
    """Patch setup.py to change package name to ngsolve-sparsesolv."""
    setup_path = NGSOLVE_SRC / "setup.py"
    content = setup_path.read_text()

    # Change name from 'ngsolve' to 'ngsolve-sparsesolv'
    # Two locations in setup.py where name is set
    replacements = [
        (
            'name = netgen_name.replace("netgen-mesher", "ngsolve")',
            f'name = "{PACKAGE_NAME}"',
        ),
        (
            "name = 'ngsolve'",
            f"name = '{PACKAGE_NAME}'",
        ),
        (
            'description="NGSolve"',
            f'description="NGSolve + SparseSolv: MKL build, SetGeomInfo, Compact AMS/COCR solvers"',
        ),
    ]

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            print(f"  Replaced: {old[:60]}...")

    # Remove the PyPI version check (we're a different package)
    # Remove lines 57-69 that check if version exists on pypi
    lines = content.split('\n')
    filtered = []
    skip = False
    for line in lines:
        if '# check if release already exists on pypi' in line:
            skip = True
        elif skip and line.startswith('except'):
            # Keep going until we find the 'pass' after except
            filtered.append(line)
            continue
        elif skip and line.strip() == 'pass':
            skip = False
            filtered.append(line)
            continue
        elif skip:
            continue
        else:
            filtered.append(line)
    content = '\n'.join(filtered)

    setup_path.write_text(content)
    print("Patched setup.py: changed package name and removed PyPI check")


def build_netgen():
    """Build and install netgen from the submodule."""
    netgen_dir = NGSOLVE_SRC / "external_dependencies" / "netgen"
    print("Building netgen...")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=str(netgen_dir),
    )
    # Install the built wheel
    wheels = list((netgen_dir / "dist").glob("*.whl"))
    if not wheels:
        raise RuntimeError("No netgen wheel found!")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(wheels[0]), "--force-reinstall"],
    )
    print(f"Installed netgen: {wheels[0].name}")


def build_ngsolve_sparsesolv():
    """Build the monolithic ngsolve-sparsesolv wheel."""
    print("Building ngsolve-sparsesolv wheel...")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=str(NGSOLVE_SRC),
    )
    # Copy wheels to dist/
    DIST_DIR.mkdir(exist_ok=True)
    for whl in (NGSOLVE_SRC / "dist").glob("*.whl"):
        dest = DIST_DIR / whl.name
        shutil.copy2(whl, dest)
        print(f"Output: {dest}")


def verify():
    """Verify the built wheel contains all expected modules."""
    import zipfile

    wheels = list(DIST_DIR.glob("ngsolve*sparsesolv*.whl"))
    if not wheels:
        print("WARNING: No wheel found in dist/")
        return False

    whl = wheels[-1]
    print(f"\nVerifying wheel: {whl.name}")

    with zipfile.ZipFile(whl, 'r') as z:
        names = z.namelist()

    has_ngsolve = any("ngsolve/" in n for n in names)
    has_sparsesolv = any("sparsesolv_ngsolve/" in n for n in names)
    has_pyd = any(n.endswith(".pyd") or n.endswith(".so") for n in names)

    print(f"  ngsolve package:        {'OK' if has_ngsolve else 'MISSING'}")
    print(f"  sparsesolv_ngsolve:     {'OK' if has_sparsesolv else 'MISSING'}")
    print(f"  Compiled extensions:    {'OK' if has_pyd else 'MISSING'}")

    # List .pyd files
    for n in sorted(names):
        if n.endswith(".pyd") or n.endswith(".so"):
            print(f"    {n}")

    return has_ngsolve and has_sparsesolv and has_pyd


def main():
    parser = argparse.ArgumentParser(description="Build monolithic ngsolve-sparsesolv wheel")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning NGSolve (use existing)")
    parser.add_argument("--skip-patch", action="store_true",
                        help="Skip applying patches")
    parser.add_argument("--skip-netgen", action="store_true",
                        help="Skip netgen build (use pre-installed)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing wheel")
    args = parser.parse_args()

    if args.verify_only:
        verify()
        return

    BUILD_DIR.mkdir(exist_ok=True)

    version = read_ngsolve_version()

    # Step 1: Clone official NGSolve
    clone_ngsolve(version, skip=args.skip_clone)

    # Step 2: Apply patches
    apply_patches(skip=args.skip_patch)

    # Step 3: Copy SparseSolv into ngsolve tree
    copy_sparsesolv()

    # Step 4: Patch ngsolve build files
    patch_ngsolve_cmake()
    patch_ngsolve_setup()

    # Step 5: Build netgen
    if not args.skip_netgen:
        build_netgen()

    # Step 6: Build monolithic wheel
    build_ngsolve_sparsesolv()

    # Step 7: Verify
    verify()

    print("\nBuild complete!")
    print(f"Wheels are in: {DIST_DIR}")


if __name__ == "__main__":
    main()
