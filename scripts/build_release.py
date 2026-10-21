"""Build hashed, size-bounded code/data archives from a clean source commit."""

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path

from scripts.verify_release import CHUNK, PREFIX, safe_relative, sha256_file, verify_part

MAIN = "20261021T122626095890Z_morphology_98e4aafa"
CLASSICAL = "20261021T141950978652Z_classical_72101312"
PAPER = "paper_review_20261021"
VERSION = "v0.1.0"


def batches(files, limit):
    group, size = [], 0
    for path, relative in files:
        weight = path.stat().st_size
        if weight > limit:
            raise ValueError(f"One file exceeds the part budget: {relative}")
        if group and size + weight > limit:
            yield group
            group, size = [], 0
        group.append((path, relative))
        size += weight
    if group:
        yield group


def write_part(output, name, files):
    destination = output / name
    temporary = destination.with_suffix(".zip.partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    manifest = {"format": 1, "files": {}}
    with zipfile.ZipFile(temporary, "x", allowZip64=True) as zipped:
        for index, (source, relative) in enumerate(files):
            safe_relative(relative)
            if relative in manifest["files"] or source.is_symlink():
                raise ValueError(f"Repeated path or symbolic link: {relative}")
            info = zipfile.ZipInfo(f"{PREFIX}/{relative}", date_time=(2026, 10, 21, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = (
                zipfile.ZIP_STORED
                if source.suffix.lower() in (".npz", ".png", ".pdf", ".zip")
                else zipfile.ZIP_DEFLATED
            )
            before = source.stat()
            digest, size = hashlib.sha256(), 0
            with source.open("rb") as stream, zipped.open(info, "w", force_zip64=True) as target:
                for chunk in iter(lambda: stream.read(CHUNK), b""):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while archiving: {relative}")
            manifest["files"][relative] = {"bytes": size, "sha256": digest.hexdigest()}
            if index % 5000 == 0:
                print(f"{name}: {index + 1}/{len(files)} files", flush=True)
        embedded = f"{PREFIX}/release_metadata/{name}.manifest.json"
        zipped.writestr(embedded, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_part(temporary)
    temporary.replace(destination)
    result = dict(
        name=name,
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        files=len(files),
        payload_bytes=sum(row["bytes"] for row in manifest["files"].values()),
    )
    print(f"Completed and read-back verified {name}: {result['bytes']} bytes", flush=True)
    return result


def run(repository, output, limit=1_400_000_000):
    repository, output = repository.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new archive destination")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])
    commit = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("Commit the reviewed source changes before archiving")
    source_files = [
        (repository / name, name)
        for name in git("ls-files", "-z").decode().split("\0")
        if name and not name.startswith("experiments/")
    ]
    groups = [("code", source_files)]
    for label, directory in (("main", MAIN), ("classical", CLASSICAL), ("review", PAPER)):
        root = repository / "experiments" / directory
        files = [
            (path, path.relative_to(repository).as_posix())
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        ]
        if not files:
            raise ValueError(f"Missing evidence: {directory}")
        groups.append((label, files))
    summary = json.loads((repository / "experiments" / PAPER / "summary.json").read_text())
    if (
        summary["total_committed_solver_calls"] != 50721
        or summary["unique_returned_endpoints"] != 285
    ):
        raise ValueError("Evidence scope changed")
    output.mkdir(parents=True)
    parts = []
    for label, files in groups:
        for index, batch in enumerate(batches(files, limit), 1):
            name = f"oxygen-shape-{VERSION}-{label}-{index:02d}.zip"
            parts.append(write_part(output, name, batch))
    release = dict(
        format=1,
        version=VERSION,
        source_commit=commit,
        author="Nabil ES-SAADI",
        affiliation="Independent researcher",
        source_license="MIT",
        original_data_figures_license="CC-BY-4.0",
        status="Research release candidate; unreviewed manuscript; no DOI assigned",
        committed_production_solver_calls=50721,
        returned_endpoints=285,
        payload_files=sum(part["files"] for part in parts),
        payload_bytes=sum(part["payload_bytes"] for part in parts),
        parts=parts,
    )
    (output / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    sums = "".join(f"{part['sha256']}  {part['name']}\n" for part in parts)
    sums += f"{sha256_file(output / 'release_manifest.json')}  release_manifest.json\n"
    (output / "SHA256SUMS.txt").write_text(sums)
    print(json.dumps(release, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--part-bytes", type=int, default=1_400_000_000)
    args = parser.parse_args()
    run(args.repository, args.output, args.part_bytes)
