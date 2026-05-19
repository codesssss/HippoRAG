#!/usr/bin/env python3
"""Verify the PCEC frozen ETv3 expander snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROOT = Path("evidenceflow/frozen_etv3_variable_flow")
DEFAULT_MANIFEST = DEFAULT_ROOT / "FREEZE_MANIFEST.json"
FORBIDDEN_IMPORTS = (
    "from evidence_transition_graphragv3_variable_flow",
    "import evidence_transition_graphragv3_variable_flow",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_include(path: Path, manifest_path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    if path.suffix == ".pyc":
        return False
    if path.resolve() == manifest_path.resolve():
        return False
    return path.is_file()


def collect_files(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not should_include(path, manifest_path):
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def find_live_imports(root: Path) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_IMPORTS:
            if needle in text:
                offenders.append({"path": str(path.relative_to(root)), "needle": needle})
    return offenders


def build_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    return {
        "freeze_date": "2026-05-10",
        "source_package": "evidence_transition_graphragv3_variable_flow",
        "frozen_package": "evidenceflow.frozen_etv3_variable_flow",
        "manifest_version": 1,
        "file_count": len(collect_files(root, manifest_path)),
        "files": collect_files(root, manifest_path),
    }


def compare_manifest(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_files = {str(row.get("path")): dict(row) for row in list(expected.get("files") or [])}
    actual_files = {str(row.get("path")): dict(row) for row in list(actual.get("files") or [])}
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    if missing:
        errors.append(f"missing files: {missing[:10]}")
    if extra:
        errors.append(f"extra files: {extra[:10]}")
    for path in sorted(set(expected_files) & set(actual_files)):
        expected_row = expected_files[path]
        actual_row = actual_files[path]
        if expected_row.get("sha256") != actual_row.get("sha256"):
            errors.append(f"sha256 mismatch: {path}")
        if int(expected_row.get("bytes", -1)) != int(actual_row.get("bytes", -2)):
            errors.append(f"size mismatch: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest)
    if not root.exists():
        raise SystemExit(f"Frozen root does not exist: {root}")
    offenders = find_live_imports(root)
    if offenders:
        raise SystemExit(f"Frozen snapshot imports live ETv3 package: {offenders[:10]}")
    actual = build_manifest(root, manifest_path)
    if bool(args.write_manifest):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {manifest_path} files={actual['file_count']}")
        return 0
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = compare_manifest(expected, actual)
    if errors:
        raise SystemExit("Frozen manifest mismatch:\n" + "\n".join(errors[:20]))
    print(f"Frozen expander verified: {root} files={actual['file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
