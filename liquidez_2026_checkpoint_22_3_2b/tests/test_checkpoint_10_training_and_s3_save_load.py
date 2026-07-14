from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import torch

from global_curriculum import GlobalCurriculumConfig, GlobalCurriculumTrainingResult, state_dict_digest
from global_data import StaticFeatureEncoder
from global_manager import GlobalManager, GlobalRunDimensions
from global_models import build_global_model
from global_s3 import (
    CHECKSUMS_FILENAME,
    DEFAULT_FINANCIAL_GPT_S3_ROOT,
    LATEST_FILENAME,
    SUCCESS_FILENAME,
)
from global_training import GlobalTrainingConfig, GlobalValidationMetrics


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "code_03_GLOBAL_MLP_E_D.ipynb"


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_order: list[str] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.put_order.append(key)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        payload = self.objects[(bucket, key)]
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    def put_object(self, *, Bucket: str, Key: str, Body, **kwargs):
        payload = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[(Bucket, Key)] = payload
        self.put_order.append(Key)
        return {"ETag": "fake"}

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def delete_objects(self, *, Bucket: str, Delete):
        for item in Delete.get("Objects", []):
            self.objects.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete.get("Objects", [])}


def validation_metrics(value: float = 12.0) -> GlobalValidationMetrics:
    return GlobalValidationMetrics(
        macro_mae=value,
        macro_rmse=value,
        micro_mae=value,
        raw_macro_mae=value,
        raw_macro_rmse=value,
        raw_macro_wmape=value,
        raw_macro_smape=value,
        num_series=2,
        num_points=8,
        per_series={"A": {"raw_smape": value}, "B": {"raw_smape": value}},
    )


def make_manager() -> GlobalManager:
    model_config = {
        "latent_dim": 4,
        "enc_hidden_size": 8,
        "enc_num_layers": 1,
        "dec_hidden_size": 8,
        "dec_num_layers": 1,
        "dropout_rate": 0.0,
        "activation": "gelu",
    }
    training_config = GlobalTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        loss="huber",
        device="cpu",
    )
    curriculum_config = GlobalCurriculumConfig(
        warmup_epochs=1,
        fine_tune_epochs_per_level=1,
        consolidation_epochs=1,
    )
    static_encoder = StaticFeatureEncoder(("saldo", "variacion"), ("MXN",))
    model = build_global_model(
        "mlp",
        model_config,
        window_size=3,
        horizon=2,
        exogenous_dim=1,
        static_dim=static_encoder.dimension,
    )
    manager = GlobalManager(
        "mlp",
        base_training_config=training_config,
        curriculum_config=curriculum_config,
        seed=42,
    )
    manager.training_result = GlobalCurriculumTrainingResult(
        architecture="mlp",
        model=model,
        model_config=model_config,
        training_config=training_config,
        curriculum_config=curriculum_config,
        stages=(),
        history=(),
        validation={
            "validation_seen": validation_metrics(10.0),
            "validation_unseen": validation_metrics(20.0),
        },
        best_score=15.0,
        total_epochs=3,
    )
    manager.dimensions = GlobalRunDimensions(
        window_size=3,
        horizon=2,
        exogenous_dim=1,
        static_dim=static_encoder.dimension,
        exogenous_columns=("dia_habil",),
        static_feature_names=static_encoder.feature_names,
    )
    manager.static_feature_encoder = static_encoder
    manager.split_manifest = {
        "train_series": ["A", "B"],
        "validation_unseen_series": ["C"],
    }
    manager.run_metadata = {"purpose": "checkpoint-10-test"}
    manager.loaded_manifest = {
        "best_candidate": {
            "window_size": 3,
            "model_config": model_config,
            "training_config": training_config.__dict__,
        },
        "best_hpo_value": 9.0,
        "num_hpo_trials": 2,
    }
    manager.persisted_hpo_summary = {
        "best_value": 9.0,
        "num_trials": 2,
        "best_params": {"window_size": 3},
        "trials": [],
    }
    return manager


class TestCheckpoint10TrainingAndS3SaveLoad(unittest.TestCase):
    def test_default_root_is_exact_required_user_folder(self) -> None:
        self.assertEqual(
            DEFAULT_FINANCIAL_GPT_S3_ROOT,
            "s3://your-private-bucket/users/your-user/financial_gpt",
        )

    def test_atomic_save_writes_success_last_and_updates_latest(self) -> None:
        manager = make_manager()
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            reports.mkdir()
            (reports / "smoke.json").write_text('{"ok": true}', encoding="utf-8")
            run_uri = manager.save_model_s3(
                run_id="GLOBAL_MLP_TEST001",
                reports_dir=reports,
                s3_client=client,
            )

        self.assertEqual(
            run_uri,
            DEFAULT_FINANCIAL_GPT_S3_ROOT + "/mlp/runs/GLOBAL_MLP_TEST001",
        )
        bucket = "your-private-bucket"
        prefix = "users/your-user/financial_gpt/mlp/runs/GLOBAL_MLP_TEST001"
        expected = {
            f"{prefix}/{SUCCESS_FILENAME}",
            f"{prefix}/{CHECKSUMS_FILENAME}",
            f"{prefix}/model/model_state.pt",
            f"{prefix}/model/manifest.json",
            f"{prefix}/evidence/training_history.parquet",
            f"{prefix}/evidence/metrics.parquet",
            f"{prefix}/reports/smoke.json",
            "users/your-user/financial_gpt/mlp/latest.json",
        }
        keys = {key for stored_bucket, key in client.objects if stored_bucket == bucket}
        self.assertTrue(expected.issubset(keys))
        success_position = client.put_order.index(f"{prefix}/{SUCCESS_FILENAME}")
        checksum_position = client.put_order.index(f"{prefix}/{CHECKSUMS_FILENAME}")
        latest_position = client.put_order.index(
            "users/your-user/financial_gpt/mlp/latest.json"
        )
        self.assertLess(checksum_position, success_position)
        self.assertLess(success_position, latest_position)
        latest = json.loads(
            client.objects[(bucket, "users/your-user/financial_gpt/mlp/latest.json")]
        )
        self.assertEqual(latest["run_uri"], run_uri)

    def test_s3_roundtrip_and_latest_preserve_exact_model(self) -> None:
        manager = make_manager()
        client = FakeS3Client()
        run_uri = manager.save_model_s3(
            run_id="GLOBAL_MLP_TEST002",
            s3_client=client,
        )
        loaded = GlobalManager.load_model_s3(
            run_uri,
            map_location="cpu",
            s3_client=client,
        )
        latest = GlobalManager.load_latest_model_s3(
            "mlp",
            map_location="cpu",
            s3_client=client,
        )
        expected_digest = state_dict_digest(manager.model.state_dict())
        self.assertEqual(state_dict_digest(loaded.model.state_dict()), expected_digest)
        self.assertEqual(state_dict_digest(latest.model.state_dict()), expected_digest)
        self.assertEqual(loaded.s3_run_uri, run_uri)
        self.assertIn("metrics.parquet", loaded.persisted_s3_evidence)
        self.assertEqual(loaded.split_manifest, manager.split_manifest)

    def test_loader_rejects_incomplete_or_tampered_run(self) -> None:
        manager = make_manager()
        client = FakeS3Client()
        run_uri = manager.save_model_s3(
            run_id="GLOBAL_MLP_TEST003",
            s3_client=client,
        )
        bucket = "your-private-bucket"
        prefix = "users/your-user/financial_gpt/mlp/runs/GLOBAL_MLP_TEST003"
        success = client.objects.pop((bucket, f"{prefix}/{SUCCESS_FILENAME}"))
        with self.assertRaises(FileNotFoundError):
            GlobalManager.load_model_s3(run_uri, s3_client=client)
        client.objects[(bucket, f"{prefix}/{SUCCESS_FILENAME}")] = success
        model_key = f"{prefix}/model/model_state.pt"
        client.objects[(bucket, model_key)] += b"tamper"
        with self.assertRaisesRegex(ValueError, "size mismatch|checksum mismatch"):
            GlobalManager.load_model_s3(run_uri, s3_client=client)

    def test_committed_run_is_immutable(self) -> None:
        manager = make_manager()
        client = FakeS3Client()
        manager.save_model_s3(run_id="GLOBAL_MLP_TEST004", s3_client=client)
        with self.assertRaises(FileExistsError):
            manager.save_model_s3(run_id="GLOBAL_MLP_TEST004", s3_client=client)

    def test_notebook_uses_manager_s3_roundtrip_and_required_root(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            DEFAULT_FINANCIAL_GPT_S3_ROOT,
            "manager.save_model_s3(",
            "GlobalManager.load_model_s3(",
            "VERIFY_S3_ROUNDTRIP = True",
        ):
            self.assertIn(token, code)
        self.assertNotIn("upload_directory_to_s3(", code)
        self.assertTrue(
            all(
                not cell.get("outputs") and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        compile(code, str(NOTEBOOK), "exec")

    def test_four_local_notebooks_remain_and_checkpoint_11_is_registered(self) -> None:
        for name in (
            "code_02_MLP_E_D.ipynb",
            "code_02_MLP_VaE_D.ipynb",
            "code_02_RNN_E_D.ipynb",
            "code_02_RNNBi_E_D.ipynb",
        ):
            self.assertTrue((ROOT / name).is_file())
        checkpoints = (ROOT / "GLOBAL_MODEL_CHECKPOINTS.csv").read_text()
        self.assertIn(
            "11,completed,Preserve four local notebooks and create four explicit global notebooks",
            checkpoints,
        )


if __name__ == "__main__":
    unittest.main()
