import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


wf = module("workflow")
extract = module("extract_libero")


class WorkflowTests(unittest.TestCase):
    def test_data_config_is_shipped_not_hidden_by_dataset_ignore(self):
        self.assertTrue((ROOT / "configs/data/libero_2cam.yaml").is_file())
        ignored = (ROOT / ".gitignore").read_text().splitlines()
        self.assertNotIn("data/", ignored)
        self.assertIn("/data/", ignored)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def manifest(self):
        ckpt = self.folder / "base.pt"
        ckpt.write_bytes(b"test fixture, not a real checkpoint")
        m = {"schema": "fastwam_sealed_base_v1", "checkpoint": "base.pt",
             "checkpoint_sha256": wf.sha(ckpt), "config_sha256": wf.config_identity(),
             "initial_model_sha256": "a" * 64, "seed": 3407, "assets_sha256": {}}
        path = self.folder / "base.json"
        path.write_text(json.dumps(m))
        return path, ckpt, m

    def test_manifest_detects_checkpoint_mutation(self):
        path, ckpt, _ = self.manifest()
        wf.read_manifest(path, verify=True)
        ckpt.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            wf.read_manifest(path, verify=True)

    def test_manifest_detects_config_drift(self):
        path, _, m = self.manifest()
        m["config_sha256"] = {}
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "Configs changed"):
            wf.read_manifest(path)

    def test_manifest_rejects_external_asset_paths(self):
        path, _, m = self.manifest()
        m["assets_sha256"] = {"../outside": "x"}
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "escapes"):
            wf.read_manifest(path, verify=True)

    def test_safe_archive_validation(self):
        root = "libero_spatial_no_noops_lerobot"
        extract.validate_member(tarfile.TarInfo(root + "/meta/info.json"), root)
        for name in ("/tmp/out", "../out", root + "/../out", "wrong/data"):
            with self.assertRaises(ValueError):
                extract.validate_member(tarfile.TarInfo(name), root)
        member = tarfile.TarInfo(root + "/link")
        member.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            extract.validate_member(member, root)

    def test_gpu_list_and_offline_environment(self):
        for gpus in ("", "0,0", "-1", "0;touch /tmp/bad"):
            with self.assertRaises(ValueError):
                wf.offline_env(gpus)
        env = wf.offline_env("4,5,6,7")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(env["DIFFSYNTH_SKIP_DOWNLOAD"].lower(), "true")

    def test_manifest_exclusive_create(self):
        path = self.folder / "manifest.json"
        wf.write_new(path, {"original": True})
        with self.assertRaises(FileExistsError):
            wf.write_new(path, {})
        self.assertTrue(json.loads(path.read_text())["original"])

    def test_four_suites_share_one_source_and_hash(self):
        path, ckpt, m = self.manifest()
        for suite in wf.SUITES:
            args = SimpleNamespace(output=str(self.folder / suite), manifest=str(path),
                suite=suite, gpus="4,5,6,7", port=29501, execute=False, batch_size=1,
                grad_accum=4, workers=2, steps=2, save_every=1000, seed=3407)
            with patch.object(wf, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                wf.train(args)
                command = run.call_args.args[0]
                self.assertIn(f"init_checkpoint={ckpt}", command)
                self.assertIn(f"init_checkpoint_sha256={m['checkpoint_sha256']}", command)
                self.assertIn(f"init_expected_model_sha256={m['initial_model_sha256']}", command)
                self.assertIn("model.mot_checkpoint_mixed_attn=true", command)
            self.assertFalse(Path(args.output).exists())

    def test_hydra_config_resolves_and_launch_overrides_compose(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        cfg = wf.composed_config()
        self.assertEqual(cfg.model.proprio_dim, 8)
        self.assertEqual(cfg.model.action_dit_config.action_dim, 7)
        self.assertFalse(cfg.model.redirect_common_files)
        path, _, _ = self.manifest()
        args = SimpleNamespace(output=str(self.folder / "run"), manifest=str(path),
            suite="long", gpus="4,5,6,7", port=29501, execute=False, batch_size=1,
            grad_accum=4, workers=2, steps=2, save_every=1000, seed=3407)
        with patch.object(wf, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            wf.train(args)
            command = run.call_args.args[0]
        overrides = command[command.index(str(ROOT / "scripts/train.py")) + 1:]
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)
            OmegaConf.resolve(cfg)
        self.assertEqual(cfg.max_steps, 2)
        self.assertEqual(cfg.batch_size, 1)
        self.assertTrue(cfg.data.train.dataset_dirs[0].endswith("libero_10_no_noops_lerobot"))

    def test_resume_rejects_other_run_and_changed_world_size(self):
        out = self.folder / "run"
        out.mkdir()
        (out / "launch.json").write_text(json.dumps(
            {"schema": "fastwam_commonbase_launch_v1", "world_size": 4}))
        args = SimpleNamespace(output=str(out), state=str(self.folder / "other"),
                               gpus="0", execute=False, port=29501)
        with self.assertRaisesRegex(ValueError, "world size"):
            wf.resume(args)
        args.gpus = "0,1,2,3"
        with self.assertRaisesRegex(ValueError, "belong to this run"):
            wf.resume(args)

    def test_seal_requires_explicit_origin_and_previews_without_write(self):
        import subprocess
        import sys
        command = [sys.executable, str(ROOT / "scripts/workflow.py"), "seal",
                   "--manifest", str(self.folder / "sealed/base.json")]
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        result = subprocess.run(command + ["--fresh-wan"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.folder / "sealed").exists())

    def test_full_seal_roundtrip_with_tiny_model_no_gpu(self):
        import os
        import torch
        from test_common_base import Model
        class SavableModel(Model):
            def save_checkpoint(self, stream, step=None):
                torch.save({"mot": self.mot.state_dict(),
                            "proprio_encoder": self.proprio_encoder.state_dict(),
                            "step": step}, stream)
        action = self.folder / "action.pt"
        action.write_bytes(b"tiny asset fixture")
        manifest = self.folder / "sealed/base.json"
        args = SimpleNamespace(manifest=str(manifest), checkpoint=None, seed=3407,
                               execute=True)
        with patch.object(wf, "ROOT", self.folder), patch.object(wf, "ACTION", action), \
             patch.object(wf, "asset_files", return_value=[]), \
             patch.object(wf, "config_identity", return_value={}), \
             patch.object(wf, "composed_config", return_value=SimpleNamespace(model={})), \
             patch("hydra.utils.instantiate", side_effect=lambda *a, **kw: SavableModel()), \
             patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            wf.seal(args)
            value, checkpoint = wf.read_manifest(manifest, verify=True)
            self.assertEqual(value["origin"], "new-wan-initialization")
            self.assertEqual(value["source_step"], 0)
            self.assertEqual(checkpoint.name, "common-base.pt")
            with self.assertRaises(FileExistsError):
                wf.seal(args)

    def test_resume_command_clears_init_and_uses_saved_config(self):
        out = self.folder / "run"
        state = out / "checkpoints/state/step_000001"
        state.mkdir(parents=True)
        (state / "trainer_state.json").write_text('{"global_step": 1}')
        (out / "config.yaml").write_text("max_steps: 3\n")
        (out / "launch.json").write_text(json.dumps(
            {"schema": "fastwam_commonbase_launch_v1", "world_size": 4}))
        args = SimpleNamespace(output=str(out), state=str(state), gpus="4,5,6,7",
                               execute=False, port=29501)
        with patch.object(wf, "run") as run:
            wf.resume(args)
        command = run.call_args.args[0]
        self.assertIn(f"resume={state}", command)
        self.assertIn("init_checkpoint=null", command)
        self.assertIn(str(out), command)


if __name__ == "__main__":
    unittest.main()
