"""Portable, preview-first preparation and common-base training entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"spatial": "spatial", "object": "object", "goal": "goal", "long": "10"}
ACTION = ROOT / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
TASK = "task=libero_uncond_2cam224_1e-4"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def config_identity():
    # Model structure, preprocessing and trainer defaults must not drift after sealing.
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "configs").rglob("*"))
            if p.is_file()}


def asset_files():
    wan = ROOT / "assets/Wan-AI/Wan2.2-TI2V-5B"
    shards = sorted(wan.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError("Wan diffusion shards missing; run download_assets.py")
    index = wan / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        if not expected.issubset({p.name for p in shards}):
            raise FileNotFoundError("Incomplete Wan diffusion shard set")
    else:
        # The pinned Wan release uses three numbered shards; do not accept a partial set.
        expected = {f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in range(1, 4)}
        if {p.name for p in shards} != expected:
            raise FileNotFoundError("Expected the three complete pinned Wan diffusion shards")
    files = shards + [wan / "Wan2.2_VAE.pth", wan / "models_t5_umt5-xxl-enc-bf16.pth"]
    tokenizer = ROOT / "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    files += [tokenizer / name for name in ("spiece.model", "tokenizer_config.json")]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return sorted(set(files + [p for p in tokenizer.iterdir() if p.is_file()]))


def offline_env(gpus=None):
    env = dict(os.environ)
    env.update(DIFFSYNTH_MODEL_BASE_PATH=str(ROOT / "assets"),
               DIFFSYNTH_SKIP_DOWNLOAD="True", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
               FASTWAM_SKIP_FINAL_SAVE="0")
    if gpus is not None:
        ids = gpus.split(",")
        if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("--gpus must be distinct physical numeric GPU IDs, e.g. 4,5,6,7")
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def dataset_dir(suite):
    return ROOT / "data/libero" / f"libero_{SUITES[suite]}_no_noops_lerobot"


def check_data(suites):
    for suite in suites:
        folder = dataset_dir(suite)
        for name in ("meta/info.json", "meta/tasks.jsonl"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        if not any(folder.glob("data/**/*.parquet")) or not any(folder.glob("videos/**/*.mp4")):
            raise FileNotFoundError(f"No parquet/video files in {folder}")


def composed_config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[TASK])
    OmegaConf.resolve(cfg)
    return cfg


def run(command, args, env=None):
    print(shlex.join([str(s) for s in command]), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, env=env or offline_env(), check=True)
    else:
        print("Preview only; add --execute. No training/download/write performed.")


def prepare(args):
    env = offline_env(args.gpus)
    if args.kind == "text":
        if args.execute:
            asset_files()
            check_data(SUITES)
        run([sys.executable, str(ROOT / "scripts/precompute_text_embeds.py"), TASK,
             "+overwrite=false"], args, env)
        return
    if ACTION.exists():
        raise FileExistsError(f"Refusing to overwrite ActionDiT asset: {ACTION}")
    resolved = ROOT / "artifacts/resolved-model.yaml"
    if args.execute:
        asset_files()
        from omegaconf import OmegaConf
        cfg = composed_config()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg.model, resolved)
    run([sys.executable, str(ROOT / "scripts/preprocess_action_dit_backbone.py"),
         "--model-config", str(resolved), "--output", str(ACTION),
         "--device", "cpu", "--dtype", "float32"], args, env)


def seal(args):
    manifest = Path(args.manifest).resolve()
    if manifest.exists():
        raise FileExistsError(f"Manifest already exists: {manifest}")
    source = Path(args.checkpoint).resolve() if args.checkpoint else None
    checkpoint = source or manifest.parent / "common-base.pt"
    if source and not source.is_file():
        raise FileNotFoundError(source)
    if not source and checkpoint.exists():
        raise FileExistsError(checkpoint)
    print(f"Seal {'existing trusted full policy' if source else 'new Wan + ActionDiT + random interfaces'}")
    print(f"checkpoint={checkpoint}\nmanifest={manifest}\ninitialization_seed={args.seed}")
    if not args.execute:
        print("Preview only; add --execute. Checkpoint selection is not automatic.")
        return
    files = asset_files()
    if not ACTION.is_file():
        raise FileNotFoundError(ACTION)
    os.environ.update(offline_env())
    sys.path.insert(0, str(ROOT))
    import torch
    from hydra.utils import instantiate
    from fastwam.common_base import seed_initialization, initialize_from_checkpoint
    cfg = composed_config()
    seed_initialization(args.seed)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cpu")
    if not source:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create prevents accidentally truncating an existing base.
        with checkpoint.open("xb") as stream:
            model.save_checkpoint(stream, step=0)
    digest = sha(checkpoint)
    report = initialize_from_checkpoint(model, checkpoint, digest)
    # Relative paths make a package-created base movable together with the repository.
    location = os.path.relpath(checkpoint, manifest.parent)
    value = {"schema": "fastwam_sealed_base_v1", "checkpoint": location,
             "checkpoint_sha256": digest, "initial_model_sha256": report["initial_model"]["sha256"],
             "source_step": report["source_step"], "seed": args.seed,
             "origin": "existing-policy" if source else "new-wan-initialization",
             "config_sha256": config_identity(),
             "assets_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files + [ACTION]}}
    write_new(manifest, value)
    print("SEALED: exact component load/readback and full-model fingerprint passed.")


def read_manifest(path, verify=False):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    if m.get("schema") != "fastwam_sealed_base_v1":
        raise ValueError("Unsupported or unsealed base manifest")
    checkpoint = (path.parent / m["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if m["config_sha256"] != config_identity():
        raise ValueError("Configs changed since base sealing; do not mix initialization recipes")
    if verify:
        if sha(checkpoint) != m["checkpoint_sha256"]:
            raise ValueError("Common base checkpoint SHA256 mismatch")
        for name, digest in m["assets_sha256"].items():
            asset = (ROOT / name).resolve()
            if ROOT not in asset.parents:
                raise ValueError("Manifest asset path escapes repository")
            if sha(asset) != digest:
                raise ValueError(f"Asset changed since sealing: {name}")
    return m, checkpoint


def launch_prefix(gpus, port):
    offline_env(gpus)
    return [sys.executable, "-m", "accelerate.commands.launch",
            "--config_file", str(ROOT / "configs/accelerate-zero1.yaml"),
            "--num_processes", str(len(gpus.split(","))), "--main_process_port", str(port)]


def train(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite: {output}")
    m, checkpoint = read_manifest(args.manifest, verify=args.execute)
    if output == checkpoint.parent or output in checkpoint.parents:
        raise ValueError("Output directory must not contain source checkpoint")
    if args.execute:
        check_data([args.suite])
        cache = ROOT / "data/text_embeds_cache/libero"
        if not list(cache.glob("*.pt")):
            raise FileNotFoundError("Text cache missing; run prepare text first")
    command = launch_prefix(args.gpus, args.port) + [
        str(ROOT / "scripts/train.py"), TASK, f"output_dir={output}",
        f"data.train.dataset_dirs=[{json.dumps(str(dataset_dir(args.suite)))}]",
        f"init_checkpoint={checkpoint}", f"init_checkpoint_sha256={m['checkpoint_sha256']}",
        f"init_expected_model_sha256={m['initial_model_sha256']}", f"init_seed={m['seed']}",
        f"seed={args.seed}", f"batch_size={args.batch_size}", f"num_workers={args.workers}",
        f"max_steps={args.steps}", f"save_every={args.save_every}",
        f"gradient_accumulation_steps={args.grad_accum}", "eval_every=0",
        "model.mot_checkpoint_mixed_attn=true", "wandb.enabled=false"]
    print(f"Effective global batch: {args.batch_size * args.grad_accum * len(args.gpus.split(','))}")
    if args.execute:
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "launch.json", {"schema": "fastwam_commonbase_launch_v1",
                  "suite": args.suite, "manifest": str(Path(args.manifest).resolve()),
                  "gpus": args.gpus, "world_size": len(args.gpus.split(",")),
                  "command": command, "base": m})
    run(command, args, offline_env(args.gpus))


def resume(args):
    output = Path(args.output).resolve()
    state = Path(args.state).resolve()
    launch = json.loads((output / "launch.json").read_text())
    if launch.get("schema") != "fastwam_commonbase_launch_v1":
        raise ValueError("Only common-base runs created by this entry point may be resumed")
    if len(args.gpus.split(",")) != launch["world_size"]:
        raise ValueError("Keep the original world size for training-state resume")
    if state.parent != output / "checkpoints/state":
        raise ValueError("Resume state must belong to this run's checkpoints/state directory")
    if not (state / "trainer_state.json").is_file():
        raise FileNotFoundError("Need a full training-state directory, not a weights .pt")
    if not (output / "config.yaml").is_file():
        raise FileNotFoundError(output / "config.yaml")
    if args.execute:
        read_manifest(launch["manifest"], verify=True)
    command = launch_prefix(args.gpus, args.port) + [
        str(ROOT / "scripts/train.py"), "--config-path", str(output), "--config-name", "config",
        f"resume={state}", "init_checkpoint=null", "init_checkpoint_sha256=null",
        "init_expected_model_sha256=null"]
    run(command, args, offline_env(args.gpus))


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("kind", choices=("action", "text"))
    p.add_argument("--gpus", default="0")
    p.set_defaults(func=prepare)
    p = sub.add_parser("seal")
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument("--fresh-wan", action="store_true")
    origin.add_argument("--checkpoint", help="Explicit trusted complete policy; never selected automatically")
    p.add_argument("--manifest", required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.set_defaults(func=seal)
    p = sub.add_parser("train")
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", required=True, type=positive)
    p.add_argument("--batch-size", type=positive, default=1, help="PER GPU")
    p.add_argument("--grad-accum", type=positive, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-every", type=positive, default=1000)
    p.add_argument("--seed", type=int, default=3407)
    p.set_defaults(func=train)
    p = sub.add_parser("resume")
    p.add_argument("--output", required=True)
    p.add_argument("--state", required=True)
    p.set_defaults(func=resume)
    for name, p in sub.choices.items():
        p.add_argument("--execute", action="store_true")
        if name in ("train", "resume"):
            p.add_argument("--gpus", required=True)
            p.add_argument("--port", type=positive, default=29501)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
