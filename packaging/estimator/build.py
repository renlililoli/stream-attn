"""Build on the target OS using an isolated environment with PyInstaller installed."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stage_sources(destination: Path):
    """Copy the unchanged estimator under a lightweight standalone package root.

    The normal core __init__ exports CUDA runners and imports torch. The standalone
    distribution deliberately includes only estimation; do not install core or
    duplicate its engine implementation to build this application.
    """
    package = destination / "seqattn_core"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text('"""Standalone estimator bundle."""\n')
    shutil.copytree(
        ROOT / "src/seqattn_core/estimation",
        package / "estimation",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    shutil.copy2(ROOT / "packaging/estimator/launcher.py", destination / "launcher.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    work = ROOT / "build/estimator"
    source = work / "source"
    if source.exists():
        shutil.rmtree(source)
    stage_sources(source)
    if args.stage_only:
        return
    output = ROOT / "dist/seqattn-estimator"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--console",
            "--name",
            "seqattn-estimator",
            "--paths",
            str(source),
            "--distpath",
            str(output),
            "--workpath",
            str(work / "pyinstaller"),
            "--specpath",
            str(work),
            "--exclude-module",
            "torch",
            "--exclude-module",
            "triton",
            str(source / "launcher.py"),
        ],
        check=True,
    )
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE.txt")
    shutil.copy2(ROOT / "packaging/estimator/README.txt", output / "README.txt")
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        shutil.copy2(python_license, output / "PYTHON-LICENSE.txt")


if __name__ == "__main__":
    main()
