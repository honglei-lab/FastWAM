"""Pinned official downloads; preview by default. No automatic dataset extraction."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", choices=["wan", "tokenizer", "libero", "released-policy"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/assets.lock.json").read_text())
    for name in args.assets:
        item = lock[name]
        print(json.dumps(item, indent=2))
        if args.execute:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                              repo_type=item["repo_type"], allow_patterns=item["patterns"],
                              local_dir=str(ROOT / item["directory"]))
    if not args.execute:
        print("Preview only; add --execute to download. Models/data are not in this repository.")


if __name__ == "__main__":
    main()
