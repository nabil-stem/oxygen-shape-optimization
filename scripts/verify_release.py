"""Verify release archives and optionally extract them to a new empty folder.

Uses only the Python standard library. No solver is imported or run.
"""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

CHUNK = 1024 * 1024
PREFIX = "oxygen_shape_release"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in (".", "..", "") for part in name.split("/"))
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def verify_part(archive, expected=None, extract=None):
    if expected is not None:
        if archive.stat().st_size != expected["bytes"]:
            raise ValueError(f"Archive size mismatch: {archive.name}")
        if sha256_file(archive) != expected["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive.name}")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        manifests = [
            name
            for name in names
            if name.startswith(f"{PREFIX}/release_metadata/") and name.endswith(".manifest.json")
        ]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one embedded part manifest")
        manifest = json.loads(zipped.read(manifests[0]))
        required = {f"{PREFIX}/{name}" for name in manifest["files"]} | {manifests[0]}
        if set(names) != required:
            raise ValueError("Archive contents differ from the embedded manifest")
        for name in names:
            safe_relative(name)
            info = zipped.getinfo(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlinks are not accepted in evidence archives")
        for relative, identity in manifest["files"].items():
            safe_relative(relative)
            name = f"{PREFIX}/{relative}"
            digest, size = hashlib.sha256(), 0
            target = None
            if extract is not None:
                target = extract / name
                target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as source:
                destination = target.open("xb") if target is not None else None
                try:
                    for chunk in iter(lambda: source.read(CHUNK), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        if destination is not None:
                            destination.write(chunk)
                finally:
                    if destination is not None:
                        destination.close()
            if size != identity["bytes"] or digest.hexdigest() != identity["sha256"]:
                raise ValueError(f"Evidence hash mismatch: {name}")
        if expected is not None and len(manifest["files"]) != expected["files"]:
            raise ValueError("Unexpected part file count")
        if extract is not None:
            target = extract / manifests[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(manifests[0]))
    return len(manifest["files"])


def run(directory, extract=None):
    directory = directory.resolve()
    release = json.loads((directory / "release_manifest.json").read_text())
    if extract is not None:
        extract = extract.resolve()
        if extract.exists():
            raise FileExistsError("Use a new extraction directory to preserve existing files")
        extract.mkdir(parents=True)
    total, seen = 0, set()
    for part in release["parts"]:
        safe_relative(part["name"])
        if Path(part["name"]).name != part["name"] or part["name"] in seen:
            raise ValueError("Invalid or repeated archive filename")
        seen.add(part["name"])
        total += verify_part(directory / part["name"], part, extract)
        print(f"Verified {part['name']}: {part['files']} files", flush=True)
    if total != release["payload_files"]:
        raise ValueError("Release total file count mismatch")
    if extract is not None:
        destination = extract / PREFIX / "release_metadata/release_manifest.json"
        destination.write_bytes((directory / "release_manifest.json").read_bytes())
    print(f"Verified {total} payload files; no PDE solves", flush=True)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    run(args.archives, args.extract)
