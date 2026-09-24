"""Generate/check hashes for tracked delivery files, not model/data artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_SHA256.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--refresh", action="store_true",
                   help="Explicitly regenerate this generated manifest after source edits")
    args = p.parse_args()
    if args.check:
        entries = json.loads(MANIFEST.read_text())
        changed = []
        for name, expected in entries.items():
            path = (ROOT / name).resolve()
            if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
                changed.append(name)
        if changed:
            raise SystemExit(f"Changed/missing files: {changed}")
        print(f"Verified {len(entries)} delivery files")
    else:
        paths = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        entries = {name: digest(ROOT / name) for name in sorted(paths)
                   if name and name != MANIFEST.name}
        with MANIFEST.open("w" if args.refresh else "x") as stream:
            json.dump(entries, stream, indent=2)
        print(f"Wrote {len(entries)} delivery file hashes")


if __name__ == "__main__":
    main()
