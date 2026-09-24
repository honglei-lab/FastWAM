"""Read-only comparison of four expert branches' pre-training identities."""
import argparse
import json
from pathlib import Path


def check(directories):
    roots = [Path(p).resolve() for p in directories]
    if len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError("Supply four distinct expert output directories")
    identities = set()
    for root in roots:
        files = sorted(root.glob("common-base-init-rank*.json"))
        if not files:
            raise ValueError(f"Missing initialization receipts: {root}")
        rows = [json.loads(p.read_text()) for p in files]
        if {r["rank"] for r in rows} != set(range(len(rows))):
            raise ValueError(f"Incomplete/duplicate rank coverage: {root}")
        for row in rows:
            if (row["world_size"] != len(rows) or not row["exact_readback"]
                    or not row["distributed_identity_verified"] or row["start_step"] != 0
                    or row["optimizer_restored"] or row["scheduler_restored"]):
                raise ValueError(f"Initialization contract failed: {root}")
            identities.add((row["checkpoint_sha256"], row["initial_model"]["sha256"]))
    if len(identities) != 1:
        raise ValueError("Experts did not start from identical complete model tensors")
    source_sha, model_sha = identities.pop()
    return {"accepted": True, "expert_branches": 4, "checkpoint_sha256": source_sha,
            "initial_model_sha256": model_sha, "directories": list(map(str, roots))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs=4)
    args = parser.parse_args()
    print(json.dumps(check(args.directories), indent=2))
