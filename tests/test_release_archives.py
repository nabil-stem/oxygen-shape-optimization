import json
import zipfile

import pytest

from scripts.build_release import batches, write_part
from scripts.verify_release import PREFIX, safe_relative, verify_part


def test_round_trip_archive_hashes_and_size_partition(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"input{index}.txt"
        path.write_bytes(b"abcdefgh")
        paths.append((path, f"data/input{index}.txt"))
    assert [len(group) for group in batches(paths, 16)] == [2, 1]
    result = write_part(tmp_path, "part.zip", paths)
    assert verify_part(tmp_path / "part.zip", result, tmp_path / "extracted") == 3
    assert (tmp_path / "extracted" / PREFIX / "data/input2.txt").read_bytes() == b"abcdefgh"
    with pytest.raises(ValueError, match="exceeds"):
        list(batches(paths, 7))
    (tmp_path / "part.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_part(tmp_path / "part.zip", result)


@pytest.mark.parametrize("name", ["../secret", "/absolute", "C:/absolute", r"data\bad", "a/../b"])
def test_unsafe_archive_paths_rejected(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_unmanifested_extra_file_rejected(tmp_path):
    file = tmp_path / "value.txt"
    file.write_text("value")
    write_part(tmp_path, "part.zip", [(file, "value.txt")])
    with zipfile.ZipFile(tmp_path / "part.zip", "a") as zipped:
        zipped.writestr(f"{PREFIX}/unmanifested.txt", "injected")
    with pytest.raises(ValueError, match="contents differ"):
        verify_part(tmp_path / "part.zip")


def test_embedded_content_hash_is_checked(tmp_path):
    archive = tmp_path / "part.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(f"{PREFIX}/value.txt", "tampered")
        zipped.writestr(
            f"{PREFIX}/release_metadata/part.zip.manifest.json",
            json.dumps({"files": {"value.txt": {"bytes": 8, "sha256": "0" * 64}}}),
        )
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        verify_part(archive)
