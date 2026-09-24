"""Strict weights-only initialization, separate from training-state resume."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def seed_initialization(seed):
    # No rank offset: all ranks/experts must construct the same initial model.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_options(checkpoint, expected_sha256, resume):
    if not checkpoint:
        if expected_sha256:
            raise ValueError("init_checkpoint_sha256 requires init_checkpoint")
        return
    if resume:
        raise ValueError("init_checkpoint and resume are mutually exclusive")
    if not expected_sha256 or len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256.lower()
    ):
        raise ValueError("Common-base initialization requires a pinned SHA256")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)


def model_fingerprint(model):
    """Includes frozen parameters/buffers, omitting duplicate registered aliases."""
    digest = hashlib.sha256()
    count = 0
    for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in iterator:
            tensor = value.detach().cpu().contiguous()
            header = json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)])
            digest.update(header.encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
    return {"sha256": digest.hexdigest(), "unique_parameters_and_buffers": count}


def initialize_from_checkpoint(model, checkpoint, expected_sha256):
    validate_options(checkpoint, expected_sha256, None)
    path = Path(checkpoint).resolve()
    identity_before = (path.stat().st_size, path.stat().st_mtime_ns)
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256.lower():
        raise ValueError("Common-base checkpoint SHA256 mismatch")
    payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    modules = {"mot": model.mot}
    if getattr(model, "proprio_encoder", None) is not None:
        modules["proprio_encoder"] = model.proprio_encoder
    elif "proprio_encoder" in payload:
        raise ValueError("Checkpoint has proprio weights but model has no proprio encoder")
    # Validate every component before changing any model tensor. No legacy dit fallback.
    for name, module in modules.items():
        if name not in payload:
            raise ValueError(f"Common base missing required component: {name}")
        target = module.state_dict()
        source = payload[name]
        if set(target) != set(source):
            raise ValueError(f"Common base tensor keys differ for {name}")
        for key, value in target.items():
            original = source[key]
            if not isinstance(original, torch.Tensor) or original.shape != value.shape:
                raise ValueError(f"Common base tensor shape differs: {name}.{key}")
            if original.is_floating_point() != value.is_floating_point():
                raise ValueError(f"Common base tensor type differs: {name}.{key}")
            if not original.is_floating_point() and original.dtype != value.dtype:
                raise ValueError(f"Common base nonfloating dtype differs: {name}.{key}")
            if original.is_floating_point() and not torch.isfinite(original.to(value.dtype)).all():
                raise ValueError(f"Common base nonfinite tensor: {name}.{key}")
    for name, module in modules.items():
        module.load_state_dict(payload[name], strict=True)
        for key, value in module.state_dict().items():
            if not torch.equal(value.detach().cpu(), payload[name][key].to(value.dtype)):
                raise ValueError(f"Common base exact readback failed: {name}.{key}")
    if identity_before != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("Common-base file changed during loading")
    return {"schema": "fastwam_common_base_initialization_v1", "checkpoint": str(path),
            "checkpoint_sha256": actual_sha, "source_step": payload.get("step"),
            "loaded_components": list(modules), "exact_readback": True,
            "initial_model": model_fingerprint(model),
            "optimizer_restored": False, "scheduler_restored": False, "start_step": 0}


def verify_and_record_identity(report, output_dir, trainer):
    if trainer.global_step != 0:
        raise ValueError("Common-base warm start must begin at training step zero")
    identity = (report["checkpoint_sha256"], report["initial_model"]["sha256"])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        identities = [None] * world_size
        torch.distributed.all_gather_object(identities, identity)
        if any(other != identity for other in identities):
            raise ValueError("Common-base initial model differs across training ranks")
        rank = torch.distributed.get_rank()
    else:
        rank = 0
        world_size = 1
    out = Path(output_dir) / f"common-base-init-rank{rank}.json"
    report = {**report, "rank": rank, "world_size": world_size,
              "distributed_identity_verified": True}
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
