#!/usr/bin/env python3
"""Install Evidence Transition GraphRAG into another HippoRAG-style repo.

Run from the source checkout or after copying this package directory:

    python evidence_transition_graphrag/install_overlay.py --target /path/to/HippoRAG

The installer copies the method package and merges the portable overlay files
needed by the fresh end-to-end runner. It intentionally does not delete target
repository directories such as ``src/hipporag``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = PACKAGE_DIR / "portable_overlay"


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _merge_dir(source: Path, target: Path) -> None:
    """Merge ``source`` into ``target`` without deleting unrelated target files."""

    for child in source.iterdir():
        if child.name in {"__pycache__", ".pytest_cache"}:
            continue
        child_target = target / child.name
        if child.is_dir():
            _merge_dir(child, child_target)
            continue
        if child.suffix == ".pyc":
            continue
        _copy_file(child, child_target)


def _copy_overlay_path(source: Path, target: Path) -> None:
    if source.is_dir():
        _merge_dir(source, target)
        return
    _copy_file(source, target)


def _overlay_sources() -> Iterable[Path]:
    if not OVERLAY_DIR.exists():
        return ()
    return sorted(path for path in OVERLAY_DIR.iterdir() if path.name != "__pycache__")


def install(target_root: Path) -> None:
    target = target_root.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    package_target = target / PACKAGE_DIR.name
    if package_target.resolve() != PACKAGE_DIR.resolve():
        if package_target.exists():
            shutil.rmtree(package_target)
        shutil.copytree(
            PACKAGE_DIR,
            package_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    for source in _overlay_sources():
        _copy_overlay_path(source, target / source.name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Target HippoRAG repository root.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    install(Path(args.target))
    print(f"Installed Evidence Transition GraphRAG overlay into {Path(args.target).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
