"""Extract only verified safe members into a new directory; preserve originals."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def validate_member(member, expected_root):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or path.parts[0] != expected_root or not (member.isfile() or member.isdir())):
        raise ValueError(f"Unsafe/unexpected archive member: {member.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    archives = [ROOT / "data/downloads" / f"libero_{s}_no_noops_lerobot.tar.gz"
                for s in ("spatial", "object", "goal", "10")]
    target = ROOT / "data/libero"
    print("Extract four legacy LeRobot archives into:", target)
    if not args.execute:
        print("Preview only; add --execute.")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to mix/overwrite data: {target}")
    for path in archives:
        with tarfile.open(path) as archive:
            for member in archive:
                validate_member(member, path.name.removesuffix(".tar.gz"))
    target.mkdir(parents=True)
    for path in archives:
        with tarfile.open(path) as archive:
            # Revalidate on extraction, not just during the first pass.
            for member in archive:
                validate_member(member, path.name.removesuffix(".tar.gz"))
                archive.extract(member, target)
    print("Extracted; source archives retained.")


if __name__ == "__main__":
    main()
