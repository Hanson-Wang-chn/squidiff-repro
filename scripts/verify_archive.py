"""Verify the portable archive using only Python's standard library.

Maintainers: stage the intended files, run --write, then stage the manifest.
--write inventories Git's index but hashes working-tree bytes; commit only after
the normal verification succeeds and all intended changes have been staged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/archive_manifest.json"
MAX_FILE_BYTES = 25 * 1024**2
MAX_TOTAL_BYTES = 350 * 1024**2
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tracked_paths():
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, stdout=subprocess.PIPE
    )
    return sorted(p.decode("utf-8") for p in result.stdout.split(b"\0") if p)


def forbidden(name):
    path = Path(name)
    return path.suffix.lower() in WEIGHT_SUFFIXES or (
        path.parts[0] == "data" and path.name != "manifest.json"
    )


def write_manifest():
    names = tracked_paths()
    entries = []
    for name in names:
        if name == MANIFEST.relative_to(ROOT).as_posix():
            continue
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular file: {name}")
        size = path.stat().st_size
        if forbidden(name) or size > MAX_FILE_BYTES:
            raise ValueError(f"Dataset, checkpoint or oversized tracked file: {name}")
        entries.append({"path": name, "bytes": size, "sha256": digest(path)})
    total = sum(e["bytes"] for e in entries)
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Archive exceeds the 350 MiB project budget")
    excluded = []
    for parent in (ROOT / "data", ROOT / "results"):
        for path in sorted(parent.rglob("*")):
            if path.is_file() and forbidden(path.relative_to(ROOT).as_posix()):
                excluded.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "reason": "checkpoint" if path.suffix.lower() in WEIGHT_SUFFIXES else "dataset_or_cache",
                })
    result = {
        "schema": 1,
        "note": "Historical archive. The manifest excludes itself; its bytes are protected by Git. Excluded files are not required for archive verification.",
        "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES},
        "file_count": len(entries),
        "total_bytes": total,
        "files": entries,
        "intentionally_excluded": excluded,
    }
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} files, {total / 1024**2:.2f} MiB (plus manifest)")


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)) or len(entries) != manifest["file_count"]:
        errors.append("Manifest has duplicate paths or an inconsistent file count")
    for entry in entries:
        name = entry["path"]
        path = ROOT / name
        if Path(name).is_absolute() or ROOT not in path.resolve().parents:
            errors.append(f"Unsafe manifest path: {name}")
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"Missing or non-regular file: {name}")
            continue
        if forbidden(name) or entry["bytes"] > MAX_FILE_BYTES:
            errors.append(f"Disallowed archive file: {name}")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            errors.append(f"Size/SHA256 mismatch: {name}")
    total = sum(entry["bytes"] for entry in entries)
    if total != manifest["total_bytes"] or total + MANIFEST.stat().st_size > MAX_TOTAL_BYTES:
        errors.append("Inconsistent size or archive exceeds the 350 MiB project budget")
    # A source ZIP has no .git directory; checksum verification still works there.
    if (ROOT / ".git").exists():
        expected = set(names) | {MANIFEST.relative_to(ROOT).as_posix()}
        actual = set(tracked_paths())
        for name in sorted(actual ^ expected):
            errors.append(f"Git index/manifest file-list mismatch: {name}")
    if errors:
        print("FAIL\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(entries)} files, {total / 1024**2:.2f} MiB; all sizes and SHA256 hashes match.")
    print("Datasets and checkpoints are intentionally absent from the archive; predictions and historical reports are included.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate manifest from staged paths (maintainers only)")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        sys.exit(verify())
