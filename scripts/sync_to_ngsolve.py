#!/usr/bin/env python3
"""
Sync SparseSolv headers to NGSolve linalg directory.

Usage:
    python sync_to_ngsolve.py [--ngsolve-dir PATH]

Default ngsolve-dir: S:/NGSolve/01_GitHub/ngsolve_ksugahar
"""

import argparse
import shutil
from pathlib import Path

# Default paths
DEFAULT_NGSOLVE_DIR = Path("S:/NGSolve/01_GitHub/ngsolve_ksugahar")
SCRIPT_DIR = Path(__file__).parent
SPARSESOLV_ROOT = SCRIPT_DIR.parent
INCLUDE_DIR = SPARSESOLV_ROOT / "include" / "sparsesolv"


def sync_headers(ngsolve_dir: Path, dry_run: bool = False):
    """Sync SparseSolv headers to NGSolve linalg directory."""
    linalg_dir = ngsolve_dir / "linalg"
    dest_sparsesolv = linalg_dir / "sparsesolv"

    if not linalg_dir.exists():
        raise FileNotFoundError(f"NGSolve linalg directory not found: {linalg_dir}")

    # Subdirectories to sync
    subdirs = ["core", "preconditioners", "solvers"]

    # 1. Sync sparsesolv.hpp (main header)
    src = INCLUDE_DIR / "sparsesolv.hpp"
    dst = dest_sparsesolv / "sparsesolv.hpp"
    _copy_file(src, dst, dry_run)

    # 2. Sync subdirectories
    for subdir in subdirs:
        src_dir = INCLUDE_DIR / subdir
        dst_dir = dest_sparsesolv / subdir
        if src_dir.exists():
            for hpp_file in sorted(src_dir.glob("*.hpp")):
                dst_file = dst_dir / hpp_file.name
                _copy_file(hpp_file, dst_file, dry_run)

    # 3. Sync sparsesolv_precond.hpp (NGSolve wrapper)
    #    Adjust include path: "../sparsesolv.hpp" -> "sparsesolv/sparsesolv.hpp"
    src_precond = INCLUDE_DIR / "ngsolve" / "sparsesolv_precond.hpp"
    dst_precond = linalg_dir / "sparsesolv_precond.hpp"
    if src_precond.exists():
        content = src_precond.read_text(encoding="utf-8")
        content = content.replace(
            '#include "../sparsesolv.hpp"',
            '#include "sparsesolv/sparsesolv.hpp"'
        )
        if dry_run:
            print(f"  [WRITE] {dst_precond} (with path adjustment)")
        else:
            dst_precond.parent.mkdir(parents=True, exist_ok=True)
            dst_precond.write_text(content, encoding="utf-8")
            print(f"  [WRITE] {dst_precond}")

    # 4. Sync test file
    src_test = SPARSESOLV_ROOT / "tests" / "test_sparsesolv.py"
    dst_test = ngsolve_dir / "tests" / "pytest" / "test_sparsesolv.py"
    if src_test.exists():
        _copy_file(src_test, dst_test, dry_run)

    print("\nSync complete.")


def _copy_file(src: Path, dst: Path, dry_run: bool):
    """Copy a single file, creating parent directories as needed."""
    if not src.exists():
        print(f"  [SKIP] {src} (not found)")
        return

    if dry_run:
        if dst.exists():
            # Check if contents differ
            src_content = src.read_bytes()
            dst_content = dst.read_bytes()
            status = "UNCHANGED" if src_content == dst_content else "UPDATE"
        else:
            status = "NEW"
        print(f"  [{status}] {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [COPY] {src.name} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Sync SparseSolv to NGSolve")
    parser.add_argument(
        "--ngsolve-dir",
        type=Path,
        default=DEFAULT_NGSOLVE_DIR,
        help=f"NGSolve source directory (default: {DEFAULT_NGSOLVE_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    args = parser.parse_args()

    print(f"SparseSolv: {SPARSESOLV_ROOT}")
    print(f"NGSolve:    {args.ngsolve_dir}")
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'SYNC'}")
    print()

    sync_headers(args.ngsolve_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
