import ast
import importlib.util
import logging
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fastwam"
spec = importlib.util.spec_from_file_location("common_base", SOURCE / "common_base.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        self.proprio_encoder = torch.nn.Linear(2, 3)
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.register_buffer("counter", torch.tensor(0))


class Config(dict):
    __getattr__ = dict.__getitem__


class CommonBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "base.pt"
        cb.seed_initialization(9)
        self.source = Model()
        self.payload = {"mot": self.source.mot.state_dict(),
                        "proprio_encoder": self.source.proprio_encoder.state_dict(),
                        "step": 12000, "optimizer": {"old": "not to restore"}}
        self.save()

    def save(self):
        torch.save(self.payload, self.path)
        self.sha = cb.sha256_file(self.path)

    def load(self, model=None):
        return cb.initialize_from_checkpoint(model or Model(), self.path, self.sha)

    def test_complete_weights_load_and_do_not_restore_training_state(self):
        model = Model()
        result = self.load(model)
        for key, value in self.source.mot.state_dict().items():
            self.assertTrue(torch.equal(value, model.mot.state_dict()[key]))
        self.assertTrue(result["exact_readback"])
        self.assertEqual(result["source_step"], 12000)
        self.assertEqual(result["start_step"], 0)
        self.assertFalse(result["optimizer_restored"])
        self.assertFalse(result["scheduler_restored"])

    def test_reject_missing_proprio_before_any_mutation(self):
        del self.payload["proprio_encoder"]
        self.save()
        model = Model()
        before = cb.model_fingerprint(model)
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.load(model)
        self.assertEqual(before, cb.model_fingerprint(model))

    def test_reject_partial_mot(self):
        del self.payload["mot"]["0.bias"]
        self.save()
        with self.assertRaisesRegex(ValueError, "keys differ"):
            self.load()

    def test_reject_legacy_dit(self):
        self.payload["dit"] = self.payload.pop("mot")
        self.save()
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.load()

    def test_reject_bad_sha(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            cb.initialize_from_checkpoint(Model(), self.path, "0" * 64)

    def test_reject_shape_and_nonfinite(self):
        self.payload["mot"]["0.bias"] = torch.zeros(9)
        self.save()
        with self.assertRaisesRegex(ValueError, "shape differs"):
            self.load()
        self.payload["mot"]["0.bias"] = torch.full((4,), float("nan"))
        self.save()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.load()

    def test_options_are_mutually_exclusive_and_sha_is_required(self):
        cb.validate_options(None, None, "existing-resume")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            cb.validate_options(self.path, self.sha, "existing-resume")
        with self.assertRaisesRegex(ValueError, "pinned SHA256"):
            cb.validate_options(self.path, None, None)

    def test_fixed_seed_before_creation_is_reproducible(self):
        cb.seed_initialization(3407)
        a = Model()
        cb.seed_initialization(3407)
        b = Model()
        self.assertEqual(cb.model_fingerprint(a), cb.model_fingerprint(b))

    def test_fingerprint_includes_frozen_parameters_and_buffers(self):
        model = Model()
        before = cb.model_fingerprint(model)
        model.counter += 1
        self.assertNotEqual(before, cb.model_fingerprint(model))
        before = cb.model_fingerprint(model)
        with torch.no_grad():
            model.frozen.weight += 1
        self.assertNotEqual(before, cb.model_fingerprint(model))

    def test_receipt_does_not_overwrite_and_requires_zero_step(self):
        report = self.load()
        with self.assertRaisesRegex(ValueError, "step zero"):
            cb.verify_and_record_identity(report, self.folder, SimpleNamespace(global_step=1))
        cb.verify_and_record_identity(report, self.folder, SimpleNamespace(global_step=0))
        with self.assertRaises(FileExistsError):
            cb.verify_and_record_identity(report, self.folder, SimpleNamespace(global_step=0))

    def test_distributed_mismatch_is_rejected(self):
        report = self.load()
        def gather(out, identity):
            out[:] = [identity, ("different", "different")]
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=2), \
             patch.object(torch.distributed, "all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(ValueError, "differs across"):
                cb.verify_and_record_identity(report, self.folder, SimpleNamespace(global_step=0))

    def test_actual_runtime_orders_init_before_trainer_and_preserves_old_output(self):
        # Execute the actual runtime function with only heavyweight construction mocked.
        tree = ast.parse((SOURCE / "runtime.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
        events = []
        def instantiate(*args, **kwargs):
            events.append("model")
            return Model()
        def init(*args):
            events.append("load")
            return cb.initialize_from_checkpoint(*args)
        def trainer(**kwargs):
            events.append("optimizer")
            model = kwargs["model"]
            self.assertTrue(torch.equal(model.mot[0].weight, self.source.mot[0].weight))
            optimizer = torch.optim.AdamW(model.parameters())
            self.assertEqual(len(optimizer.state), 0)
            return SimpleNamespace(global_step=0, train=lambda: events.append("train"))
        env = {"DictConfig": Config, "Path": Path, "logging": logging, "torch": torch,
               "validate_options": cb.validate_options, "seed_initialization": cb.seed_initialization,
               "initialize_from_checkpoint": init, "verify_and_record_identity": cb.verify_and_record_identity,
               "setup_logging": lambda **kw: None, "instantiate": instantiate,
               "misc": SimpleNamespace(register_work_dir=lambda p: Path(p).mkdir(parents=True, exist_ok=True)),
               "OmegaConf": SimpleNamespace(to_container=lambda cfg, **kw: dict(cfg), save=lambda *args: None),
               "_resolve_train_device": lambda: "cpu", "_normalize_mixed_precision": lambda v: v,
               "_mixed_precision_to_model_dtype": lambda v: torch.float32,
               "build_datasets": lambda cfg: ([], None), "Wan22Trainer": trainer}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE / "runtime.py"), "exec"), env)
        cfg = Config(init_checkpoint=str(self.path), init_checkpoint_sha256=self.sha, resume=None,
                     output_dir=str(self.folder / "new-run"), init_seed=3407, seed=5,
                     mixed_precision="no", model={}, data={})
        env["run_training"](cfg)
        self.assertEqual(events, ["model", "load", "optimizer", "train"])
        with self.assertRaises(FileExistsError):
            env["run_training"](cfg)
        cfg["output_dir"] = str(self.folder)
        with self.assertRaisesRegex(ValueError, "contain its source"):
            env["run_training"](cfg)
        cfg["output_dir"] = str(self.folder / "wrong-frozen-base")
        cfg["init_expected_model_sha256"] = "incorrect"
        events.clear()
        with self.assertRaisesRegex(ValueError, "sealed common base"):
            env["run_training"](cfg)
        self.assertEqual(events, ["model", "load"])

    def test_four_branch_receipt_checker(self):
        check_spec = importlib.util.spec_from_file_location("checker", ROOT / "scripts/check_common_base_receipts.py")
        checker = importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(checker)
        report = self.load()
        roots = [self.folder / n for n in ("spatial", "object", "goal", "long")]
        for root in roots:
            root.mkdir()
            cb.verify_and_record_identity(report, root, SimpleNamespace(global_step=0))
        self.assertTrue(checker.check(roots)["accepted"])
        path = roots[-1] / "common-base-init-rank0.json"
        changed = json.loads(path.read_text())
        changed["initial_model"]["sha256"] = "different"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "did not start"):
            checker.check(roots)
        with self.assertRaisesRegex(ValueError, "distinct"):
            checker.check([roots[0]] * 4)

    def test_portable_preview_and_old_output_protection(self):
        out = self.folder / "preview-only"
        wf_spec = importlib.util.spec_from_file_location("workflow", ROOT / "scripts/workflow.py")
        wf = importlib.util.module_from_spec(wf_spec)
        wf_spec.loader.exec_module(wf)
        manifest = self.folder / "base.json"
        manifest.write_text(json.dumps({"schema": "fastwam_sealed_base_v1",
            "checkpoint": "base.pt", "checkpoint_sha256": self.sha,
            "config_sha256": wf.config_identity(), "initial_model_sha256": "abc",
            "assets_sha256": {}, "seed": 3407}))
        command = [sys.executable, str(ROOT / "scripts/workflow.py"), "train",
                   "--suite", "spatial", "--manifest", str(manifest),
                   "--output", str(out), "--gpus", "4,5,6,7", "--steps", "2"]
        p = subprocess.run(command, capture_output=True, text=True, cwd="/tmp")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("init_checkpoint=", p.stdout)
        self.assertFalse(out.exists())
        out.mkdir()
        (out / "old-result.txt").write_text("preserve")
        p = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("new output directory", p.stderr)
        self.assertEqual((out / "old-result.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
