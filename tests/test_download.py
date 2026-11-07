import hashlib
import zipfile

import pytest

from readmission_risk.data.download import (
    ChecksumError,
    extract_members,
    fetch,
    sha256_of,
    verify_checksum,
)


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"readmission" * 1000)
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_of_matches_hashlib(sample_file):
    assert sha256_of(sample_file) == _sha(sample_file.read_bytes())


def test_verify_checksum_accepts_uppercase(sample_file):
    verify_checksum(sample_file, sha256_of(sample_file).upper())


def test_verify_checksum_rejects_mismatch(sample_file):
    with pytest.raises(ChecksumError):
        verify_checksum(sample_file, "0" * 64)


def test_fetch_downloads_and_verifies(sample_file, tmp_path):
    dest = tmp_path / "out" / "copy.bin"
    fetch(sample_file.as_uri(), dest, sha256_of(sample_file))
    assert dest.read_bytes() == sample_file.read_bytes()
    assert list(dest.parent.iterdir()) == [dest]  # no temp files left behind


def test_fetch_bad_checksum_leaves_nothing(sample_file, tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(ChecksumError):
        fetch(sample_file.as_uri(), out_dir / "copy.bin", "0" * 64)
    assert list(out_dir.iterdir()) == []


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.csv", "x,y\n1,2\n")
        zf.writestr("b.csv", "z\n3\n")
        zf.writestr("extra.txt", "not wanted")
    return path


def test_extract_members_only_requested(archive, tmp_path):
    out = tmp_path / "raw"
    paths = extract_members(archive, ["a.csv", "b.csv"], out)
    assert [p.name for p in paths] == ["a.csv", "b.csv"]
    assert sorted(p.name for p in out.iterdir()) == ["a.csv", "b.csv"]
    assert (out / "a.csv").read_text() == "x,y\n1,2\n"


def test_extract_members_missing_raises(archive, tmp_path):
    with pytest.raises(KeyError):
        extract_members(archive, ["missing.csv"], tmp_path / "raw")


@pytest.mark.parametrize("name", ["../evil.csv", "sub/a.csv"])
def test_extract_members_rejects_nested_paths(archive, tmp_path, name):
    with pytest.raises(ValueError):
        extract_members(archive, [name], tmp_path / "raw")
