"""
Build monolithic ngsolve-sparsesolv wheel.

Strategy: Build netgen and ngsolve as separate wheels first, install them,
then build sparsesolv standalone against the installed ngsolve, and finally
merge everything into a single wheel.

This avoids Windows DLL template export issues that occur when building
sparsesolv inside the ngsolve CMake tree.

Usage:
    python scripts/build_monolithic.py [--skip-clone] [--skip-patch]

Requirements:
    pip install build scikit-build scikit-build-core wheel numpy "pybind11-stubgen==2.5"
    pip install "netgen-occt-devel==7.8.1" "netgen-occt==7.8.1"
    pip install packaging requests
    pip install mkl-devel mkl intel-cmplr-lib-rt
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
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
    """Apply SetGeomInfo patch to netgen submodule."""
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


def build_netgen():
    """Build and install netgen from the submodule."""
    netgen_dir = NGSOLVE_SRC / "external_dependencies" / "netgen"
    print("\n=== Building netgen ===")
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
    return wheels[0]


def build_ngsolve():
    """Build and install ngsolve (without sparsesolv)."""
    print("\n=== Building ngsolve ===")

    # Export ALL symbols from libngsolve.dll so that downstream pybind11
    # modules (sparsesolv) can link against template instantiations
    # (DifferentialOperator SIMD methods etc.) that aren't explicitly marked
    # with __declspec(dllexport) in the NGSolve headers.
    env = os.environ.copy()
    env["CMAKE_ARGS"] = env.get("CMAKE_ARGS", "") + " -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"

    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=str(NGSOLVE_SRC),
        env=env,
    )
    wheels = list((NGSOLVE_SRC / "dist").glob("*.whl"))
    if not wheels:
        raise RuntimeError("No ngsolve wheel found!")
    # Install the ngsolve wheel
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(wheels[0]), "--force-reinstall"],
    )
    print(f"Installed ngsolve: {wheels[0].name}")
    return wheels[0]


def build_sparsesolv():
    """Build sparsesolv standalone against installed ngsolve."""
    print("\n=== Building sparsesolv (standalone) ===")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(REPO_ROOT), "-v"],
    )
    print("SparseSolv installed.")

    # Find the installed sparsesolv files
    result = subprocess.run(
        [sys.executable, "-c",
         "import sparsesolv_ngsolve; import os; print(os.path.dirname(sparsesolv_ngsolve.__file__))"],
        capture_output=True, text=True,
    )
    sparsesolv_dir = Path(result.stdout.strip())
    print(f"SparseSolv installed at: {sparsesolv_dir}")
    return sparsesolv_dir


def merge_into_wheel(ngsolve_wheel, sparsesolv_dir):
    """Merge sparsesolv files into the ngsolve wheel, rename to ngsolve-sparsesolv."""
    print("\n=== Merging into monolithic wheel ===")
    DIST_DIR.mkdir(exist_ok=True)

    # Read ngsolve wheel version from filename
    # e.g., ngsolve-6.2.2601-cp312-cp312-win_amd64.whl
    ngsolve_whl_name = ngsolve_wheel.name
    # Extract version and platform tags
    parts = ngsolve_whl_name.replace(".whl", "").split("-")
    # parts: [name, version, pythonver, abi, platform]
    ngsolve_version = parts[1]
    py_tag = parts[2]
    abi_tag = parts[3]
    plat_tag = parts[4]

    # Read sparsesolv version from our pyproject.toml
    import tomllib
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    sparsesolv_version = pyproject["project"]["version"]

    # Combined version: sparsesolv_version+ngsolve_version
    # e.g., 3.0.0+ngsolve6.2.2601
    combined_version = f"{sparsesolv_version}"

    out_name = f"ngsolve_sparsesolv-{combined_version}-{py_tag}-{abi_tag}-{plat_tag}.whl"
    out_path = DIST_DIR / out_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract ngsolve wheel
        print(f"  Extracting: {ngsolve_wheel.name}")
        with zipfile.ZipFile(ngsolve_wheel, 'r') as z:
            z.extractall(tmpdir)

        # Copy sparsesolv_ngsolve package
        dest_sparsesolv = tmpdir / "sparsesolv_ngsolve"
        if dest_sparsesolv.exists():
            shutil.rmtree(dest_sparsesolv)
        shutil.copytree(sparsesolv_dir, dest_sparsesolv)
        print(f"  Added: sparsesolv_ngsolve/")

        # Update METADATA
        dist_info_dirs = list(tmpdir.glob("*.dist-info"))
        if dist_info_dirs:
            old_dist_info = dist_info_dirs[0]
            new_dist_info_name = f"ngsolve_sparsesolv-{combined_version}.dist-info"
            new_dist_info = tmpdir / new_dist_info_name

            # Rename dist-info directory
            old_dist_info.rename(new_dist_info)

            # Update METADATA file
            metadata_path = new_dist_info / "METADATA"
            if metadata_path.exists():
                content = metadata_path.read_text(encoding="utf-8")
                content = re.sub(r"^Name: .*$", f"Name: {PACKAGE_NAME}", content, flags=re.MULTILINE)
                content = re.sub(r"^Version: .*$", f"Version: {combined_version}", content, flags=re.MULTILINE)
                # Add mkl dependency if not present
                if "mkl" not in content:
                    content += "Requires-Dist: mkl>=2024.2.0\n"
                    content += "Requires-Dist: intel-cmplr-lib-rt\n"
                metadata_path.write_text(content, encoding="utf-8")
                print(f"  Updated METADATA: {PACKAGE_NAME} {combined_version}")

            # Update WHEEL file
            wheel_path = new_dist_info / "WHEEL"
            if wheel_path.exists():
                content = wheel_path.read_text(encoding="utf-8")
                # Keep existing content (tags, generator, etc.)
                wheel_path.write_text(content, encoding="utf-8")

            # Regenerate RECORD
            record_path = new_dist_info / "RECORD"
            _generate_record(tmpdir, record_path, new_dist_info_name)

        # Create the merged wheel
        print(f"  Creating: {out_name}")
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for file_path in sorted(tmpdir.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir).as_posix()
                    zout.write(file_path, arcname)

    print(f"  Output: {out_path}")
    return out_path


def _generate_record(wheel_dir, record_path, dist_info_name):
    """Regenerate RECORD file for the wheel."""
    import hashlib
    import base64

    records = []
    for file_path in sorted(wheel_dir.rglob("*")):
        if file_path.is_file():
            arcname = file_path.relative_to(wheel_dir).as_posix()
            if arcname == f"{dist_info_name}/RECORD":
                records.append(f"{arcname},,")
                continue
            data = file_path.read_bytes()
            digest = hashlib.sha256(data).digest()
            b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            size = len(data)
            records.append(f"{arcname},sha256={b64},{size}")

    record_path.write_text("\n".join(records) + "\n", encoding="utf-8")


def verify(wheel_path=None):
    """Verify the built wheel contains all expected modules."""
    if wheel_path is None:
        wheels = list(DIST_DIR.glob("ngsolve*sparsesolv*.whl"))
        if not wheels:
            print("WARNING: No wheel found in dist/")
            return False
        wheel_path = wheels[-1]

    print(f"\nVerifying wheel: {wheel_path.name}")

    with zipfile.ZipFile(wheel_path, 'r') as z:
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

    ok = has_ngsolve and has_sparsesolv and has_pyd
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Build monolithic ngsolve-sparsesolv wheel")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning NGSolve (use existing)")
    parser.add_argument("--skip-patch", action="store_true",
                        help="Skip applying patches")
    parser.add_argument("--skip-netgen", action="store_true",
                        help="Skip netgen build (use pre-installed)")
    parser.add_argument("--skip-ngsolve", action="store_true",
                        help="Skip ngsolve build (use pre-installed)")
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

    # Step 2: Apply patches (SetGeomInfo to netgen)
    apply_patches(skip=args.skip_patch)

    # Step 3: Build and install netgen
    if not args.skip_netgen:
        build_netgen()

    # Step 4: Build and install ngsolve
    if not args.skip_ngsolve:
        ngsolve_wheel = build_ngsolve()
    else:
        # Find existing ngsolve wheel
        wheels = list((NGSOLVE_SRC / "dist").glob("ngsolve-*.whl"))
        if not wheels:
            raise RuntimeError("No ngsolve wheel found (--skip-ngsolve)")
        ngsolve_wheel = wheels[0]

    # Step 5: Build sparsesolv standalone (against installed ngsolve)
    sparsesolv_dir = build_sparsesolv()

    # Step 6: Merge into monolithic wheel
    merged_wheel = merge_into_wheel(ngsolve_wheel, sparsesolv_dir)

    # Step 7: Verify
    verify(merged_wheel)

    print("\nBuild complete!")
    print(f"Wheel: {merged_wheel}")


if __name__ == "__main__":
    main()
