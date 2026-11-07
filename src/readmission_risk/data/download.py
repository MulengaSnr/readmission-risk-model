"""Download and verify the UCI Diabetes 130-US Hospitals dataset.

Run from the project root:

    uv run python -m readmission_risk.data.download

The archive is checked against a pinned SHA-256 hash so the analysis always runs
on exactly the same data. Only the expected CSV files are extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from readmission_risk.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_TIMEOUT_SECONDS = 60


class ChecksumError(RuntimeError):
    """Raised when a file's SHA-256 hash does not match the expected value."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    """Raise ``ChecksumError`` if ``path`` does not have the expected hash.

    Args:
        path: File to check.
        expected_sha256: Expected hex digest.

    Raises:
        ChecksumError: If the digests differ.
    """
    actual = sha256_of(path)
    if actual != expected_sha256.lower():
        raise ChecksumError(
            f"Checksum mismatch for {path.name}: expected {expected_sha256}, "
            f"got {actual}. The source file may have changed."
        )


def fetch(url: str, dest: Path, expected_sha256: str) -> Path:
    """Download ``url`` to ``dest`` atomically, verifying its checksum.

    The file is written to a temporary path in the same directory and only
    moved into place once the checksum passes, so an interrupted or corrupted
    download never leaves a file that looks valid.

    Args:
        url: Source URL.
        dest: Final path for the downloaded file.
        expected_sha256: Expected hex digest of the file.

    Returns:
        ``dest``.

    Raises:
        ChecksumError: If the downloaded file's hash does not match.
        urllib.error.URLError: If the download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        logger.info("Downloading %s", url)
        with (
            urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response,
            tmp_path.open("wb") as out,
        ):
            for chunk in iter(lambda: response.read(_CHUNK_SIZE), b""):
                out.write(chunk)
        verify_checksum(tmp_path, expected_sha256)
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("Saved and verified %s", dest)
    return dest


def extract_members(archive: Path, members: list[str], dest_dir: Path) -> list[Path]:
    """Extract only the named members of a zip archive.

    Extracting by explicit name (rather than ``extractall``) means an archive
    cannot write files outside ``dest_dir`` or add unexpected files.

    Args:
        archive: Zip file to read.
        members: Exact member names to extract (must be top-level files).
        dest_dir: Directory to write them to.

    Returns:
        Paths of the extracted files, in the order requested.

    Raises:
        KeyError: If a requested member is missing from the archive.
        ValueError: If a member name contains a directory component.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        available = set(zf.namelist())
        for name in members:
            if Path(name).name != name:
                raise ValueError(f"Refusing to extract nested path: {name!r}")
            if name not in available:
                raise KeyError(f"{name!r} not found in {archive.name}")
            target = dest_dir / name
            target.write_bytes(zf.read(name))
            extracted.append(target)
    return extracted


def download_dataset(force: bool = False) -> list[Path]:
    """Download (if needed), verify and extract the dataset into ``data/raw``.

    Args:
        force: Re-download even if a verified archive already exists.

    Returns:
        Paths of the extracted CSV files.
    """
    config = load_config()
    ds = config["dataset"]
    raw_dir = PROJECT_ROOT / config["paths"]["raw"]
    archive = raw_dir / ds["archive_name"]

    if archive.exists() and not force:
        try:
            verify_checksum(archive, ds["sha256"])
            logger.info("Verified archive already present: %s", archive)
        except ChecksumError:
            logger.warning("Existing archive failed checksum; downloading again.")
            fetch(ds["url"], archive, ds["sha256"])
    else:
        fetch(ds["url"], archive, ds["sha256"])

    return extract_members(archive, ds["members"], raw_dir)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download the archive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in download_dataset(force=args.force):
        logger.info("Ready: %s", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
