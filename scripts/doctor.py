"""Read-only environment/data readiness check. Never constructs a model."""
import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from workflow import ROOT, ACTION, SUITES, asset_files, check_data, composed_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", action="store_true")
    p.add_argument("--data", action="store_true")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    failures = []
    versions = {"python": sys.version}
    for name in ("torch", "torchvision", "torchcodec", "deepspeed", "accelerate",
                 "transformers", "datasets", "hydra-core", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing distribution: {name}")
    sys.path.insert(0, str(ROOT))
    for name in ("fastwam.runtime", "fastwam.models.wan22.fastwam", "torchcodec",
                 "ftfy", "sentencepiece"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"Import {name}: {exc}")
    if not shutil.which("ffmpeg"):
        failures.append("ffmpeg executable missing")
    try:
        cfg = composed_config()
        assert cfg.model.action_dit_config.action_dim == 7
        assert cfg.model.proprio_dim == 8
        if args.cuda:
            import torch
            if not torch.cuda.is_available():
                failures.append("CUDA unavailable")
            else:
                versions["gpus"] = [torch.cuda.get_device_name(i)
                                   for i in range(torch.cuda.device_count())]
                if not torch.cuda.is_bf16_supported():
                    failures.append("BF16 unsupported on selected GPU")
        if args.assets:
            versions["asset_files"] = len(asset_files())
            if not ACTION.is_file():
                failures.append("ActionDiT preprocessing output missing")
        if args.data:
            check_data(SUITES)
            if not list((ROOT / "data/text_embeds_cache/libero").glob("*.pt")):
                failures.append("Text embedding cache missing")
    except Exception as exc:
        failures.append(str(exc))
    print(json.dumps({"versions": versions, "failures": failures,
                      "ready": not failures,
                      "scope": "imports/files only; not a training or complete-cache validation"},
                     indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
