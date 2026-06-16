"""Verify the exported thesis handoff zip against its SHA-256 sidecar."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_ARCHIVE = Path("docs/thesis_export_package.zip")


def verify_thesis_package(
    archive_path: Path = DEFAULT_ARCHIVE,
    sha256_path: Path | None = None,
) -> dict[str, object]:
    archive_path = Path(archive_path)
    sha256_path = Path(sha256_path) if sha256_path is not None else archive_path.with_suffix(archive_path.suffix + ".sha256")
    expected_sha256, sidecar_archive = _read_sha256_sidecar(sha256_path)
    if sidecar_archive is not None and archive_path == DEFAULT_ARCHIVE:
        archive_path = sha256_path.parent / sidecar_archive

    actual_sha256 = _sha256_file(archive_path) if archive_path.exists() else ""
    status = "verified" if archive_path.exists() and actual_sha256 == expected_sha256 else "failed"
    return {
        "status": status,
        "archive": str(archive_path),
        "sha256_file": str(sha256_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "size_bytes": int(archive_path.stat().st_size) if archive_path.exists() else 0,
    }


def _read_sha256_sidecar(path: Path) -> tuple[str, str | None]:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if not parts:
        raise ValueError(f"Empty SHA-256 sidecar: {path}")
    expected = parts[0]
    if len(expected) != 64:
        raise ValueError(f"Invalid SHA-256 digest in sidecar: {path}")
    sidecar_archive = parts[1] if len(parts) >= 2 else None
    return expected, sidecar_archive


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--sha256", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_thesis_package(archive_path=args.archive, sha256_path=args.sha256)
    if result["status"] == "verified":
        print(
            "Verified thesis package: "
            f"{result['archive']} ({result['size_bytes']} bytes, sha256={result['actual_sha256']})"
        )
        return
    print(
        "Thesis package verification failed: "
        f"expected {result['expected_sha256']}, got {result['actual_sha256']}"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
