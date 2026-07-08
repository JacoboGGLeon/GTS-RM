from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FACADE_MODULES = [
    "gts_rm.config",
    "gts_rm.data",
    "gts_rm.models",
    "gts_rm.training",
    "gts_rm.evaluation",
    "gts_rm.artifacts",
]
SUPPORTED_ARCHITECTURES = ["mlp", "mlp_vae", "rnn", "rnn_bi"]
SMOKE_FACADE_MODULES = ["gts_rm.config", "gts_rm.models"]
SMOKE_WORKFLOWS = {
    "mlp": "smoke_global_mlp",
    "mlp_vae": "smoke_global_mlp_vae",
    "rnn": "smoke_global_rnn",
    "rnn_bi": "smoke_global_rnn_bi",
}


def test_gts_rm_exports_cp20_core() -> None:
    import gts_rm

    assert gts_rm.CP20_BUNDLE_ROOT.exists()
    assert gts_rm.MAC3_TEST_ROOT.exists()
    assert gts_rm.list_global_models() == tuple(SUPPORTED_ARCHITECTURES)
    assert gts_rm.FinancialGPTStageConfig().flags.use_static_context is True


def test_cp20_wrapper_builds_global_model() -> None:
    from gts_rm import build_global_model

    model = build_global_model(
        "mlp",
        {
            "latent_dim": 8,
            "enc_hidden_size": 16,
            "enc_num_layers": 1,
            "dec_hidden_size": 16,
            "dec_num_layers": 1,
            "use_auxiliary_autoencoder": False,
        },
        window_size=4,
        horizon=2,
        exogenous_dim=1,
        static_dim=3,
    )

    assert model.dimensions.window_size == 4
    assert model.dimensions.horizon == 2


def test_mac3_test_contract_loads_and_validates() -> None:
    import gts_rm

    contract = gts_rm.load_use_case("MAC3_TEST")
    assert contract.name == "MAC3_TEST"
    assert contract.manifest["checkpoint"] == "CP27"
    assert contract.manifest["status"] == "acceptance_report"
    assert contract.contract_path.exists()
    assert contract.cp20_bundle_path == gts_rm.CP20_BUNDLE_ROOT
    assert contract.frozen_contract_path.exists()
    assert set(SMOKE_WORKFLOWS.values()).issubset(contract.manifest["workflows"])
    assert "smoke_all_global_models" in contract.manifest["workflows"]
    assert contract.manifest["config_migration"]["loader"] == "gts_rm.config.load_mac3_config_bundle"
    assert contract.manifest["data_contract_migration"]["loader"] == "gts_rm.data.load_mac3_data_contract"
    assert "gts_rm.models.build_global_model_from_config" in contract.manifest["model_training_facade_migration"]["model_entrypoints"]
    assert "gts_rm.training.build_mac3_trainer" in contract.manifest["model_training_facade_migration"]["training_entrypoints"]
    assert contract.manifest["acceptance_report"]["verdict"] == "accepted"


def test_mac3_test_configs_match_locked_cp20_contract() -> None:
    manifest = json.loads((ROOT / "MAC3_TEST" / "manifest.json").read_text(encoding="utf-8-sig"))
    base = json.loads((ROOT / manifest["configs"]["base"]).read_text(encoding="utf-8-sig"))
    acceptance = json.loads((ROOT / manifest["configs"]["acceptance"]).read_text(encoding="utf-8-sig"))
    smokes = {
        arch: json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
        for arch, path in manifest["configs"]["smokes"].items()
    }

    assert manifest["kind"] == "use_case"
    assert manifest["release_first"] is True
    assert manifest["tutorials_deferred"] is True
    assert manifest["library_facade"]["modules"] == FACADE_MODULES
    assert manifest["configs"]["stage"] == "MAC3_TEST/configs/stage_cp20.json"
    assert manifest["configs"]["training"] == "MAC3_TEST/configs/training_smoke.json"
    assert manifest["configs"]["candidates"] == "MAC3_TEST/configs/candidates_smoke.json"
    assert manifest["configs"]["notebooks"] == "MAC3_TEST/configs/notebooks_mac3.json"
    assert manifest["configs"]["data_contract"] == "MAC3_TEST/configs/data_contract.json"
    assert base["checkpoint"] == "CP27"
    assert base["facade_modules"] == FACADE_MODULES
    assert base["model_inputs"] == manifest["locked_cp20_contract"]["model_inputs"]
    assert base["supported_architectures"] == manifest["locked_cp20_contract"]["architectures"]
    assert base["output"] == manifest["locked_cp20_contract"]["output"]
    assert base["latent"] == manifest["locked_cp20_contract"]["latent"]
    assert base["config_loader"] == "gts_rm.config.load_mac3_config_bundle"
    assert base["data_contract"] == "MAC3_TEST/configs/data_contract.json"
    assert base["model_training_facade"]["model_entrypoint"] == "gts_rm.models.build_global_model_from_config"
    assert base["model_training_facade"]["trainer_entrypoint"] == "gts_rm.training.build_mac3_trainer"
    assert list(smokes) == SUPPORTED_ARCHITECTURES
    for architecture, smoke in smokes.items():
        assert smoke["checkpoint"] == "CP24"
        assert smoke["architecture"] == architecture
        assert smoke["expected_output_shape"] == [2, 3, 1]
    assert acceptance["checkpoint"] == "CP27"
    assert acceptance["metrics"]["primary"] == "robust_macro_mase"
    assert acceptance["release_gate"]["must_use_library_facade"] is True
    assert acceptance["release_gate"]["must_run_smoke_workflow"] is True
    assert acceptance["release_gate"]["must_load_migrated_configs"] is True
    assert acceptance["release_gate"]["must_load_data_contract"] is True
    assert acceptance["release_gate"]["must_load_model_training_facade"] is True
    assert acceptance["release_gate"]["must_have_acceptance_report"] is True


def test_cp22_facade_modules_import_and_expose_expected_symbols() -> None:
    modules = {name: importlib.import_module(name) for name in FACADE_MODULES}

    assert modules["gts_rm.config"].FinancialGPTStageConfig().flags.use_static_context is True
    assert modules["gts_rm.config"].CONFIG_CHECKPOINT == "CP24"
    assert modules["gts_rm.data"].MODEL_INPUT_FIELDS == ("y_context", "x_history", "x_future", "x_static")
    assert modules["gts_rm.data"].DATA_CONTRACT_CHECKPOINT == "CP25"
    assert modules["gts_rm.data"].ContextScaler is not None
    assert modules["gts_rm.models"].list_global_models() == tuple(SUPPORTED_ARCHITECTURES)
    assert modules["gts_rm.models"].GLOBAL_OUTPUT_FIELD == "y_pred"
    assert modules["gts_rm.models"].build_global_model_from_config is not None
    assert modules["gts_rm.models"].build_mac3_smoke_model is not None
    assert modules["gts_rm.training"].GlobalTrainingConfig is not None
    assert modules["gts_rm.training"].build_mac3_trainer is not None
    assert modules["gts_rm.training"].load_mac3_candidates is not None
    assert modules["gts_rm.training"].GlobalCurriculumConfig is not None
    assert modules["gts_rm.evaluation"].GlobalValidationMetrics is not None
    assert modules["gts_rm.evaluation"].FinancialGPTMonitorResult is not None
    assert modules["gts_rm.artifacts"].GlobalManager is not None
    assert modules["gts_rm.artifacts"].S3Location is not None


def test_cp24_migrated_config_bundle_loads_and_validates() -> None:
    from gts_rm import config

    bundle = config.load_mac3_config_bundle()

    assert tuple(bundle.candidates) == tuple(SUPPORTED_ARCHITECTURES)
    assert tuple(bundle.notebooks) == tuple(SUPPORTED_ARCHITECTURES)
    assert tuple(bundle.smokes) == tuple(SUPPORTED_ARCHITECTURES)
    assert bundle.stage.flags.use_causal_scaler is True
    assert bundle.stage.flags.use_patch_tokenizer is False
    assert bundle.training.selection_metric == "robust_macro_mase"
    for architecture in SUPPORTED_ARCHITECTURES:
        assert bundle.candidates[architecture].window_size == bundle.smokes[architecture]["window_size"]
        assert bundle.candidates[architecture].model_config == bundle.smokes[architecture]["model_config"]
        assert bundle.notebooks[architecture].architecture == architecture
        assert bundle.notebooks[architecture].artifact_root == f"MAC3_TEST/artifacts/{architecture}"


def test_cp25_data_contract_loads_and_matches_cp20_schema() -> None:
    from gts_rm import data

    contract = data.load_mac3_data_contract()
    summary = data.mac3_data_contract_summary()

    assert contract.checkpoint == "CP25"
    assert contract.global_long_required_columns == data.GLOBAL_LONG_REQUIRED_COLUMNS
    assert contract.model_input_fields == data.MODEL_INPUT_FIELDS
    assert contract.metadata_fields == data.MODEL_METADATA_FIELDS
    assert contract.forbidden_model_input_fields == data.FORBIDDEN_MODEL_INPUT_FIELDS
    assert contract.calendar_date_column == data.DATE_COLUMN
    assert contract.target_column == data.TARGET_COLUMN
    assert contract.split_unit == data.ACCOUNT_CURRENCY_ID_COLUMN
    assert contract.exogenous_columns == ("month_sin", "month_cos", "is_month_end")
    assert summary["future_known_exogenous"] is True
    assert summary["global_long_columns"] == len(data.GLOBAL_LONG_REQUIRED_COLUMNS)


def test_cp25_data_contract_matches_notebook_configs() -> None:
    from gts_rm import config, data

    data_contract = data.load_mac3_data_contract()
    notebooks = config.load_notebook_configs()

    for notebook in notebooks.values():
        assert notebook.global_long_uri == data_contract.global_long_uri
        assert notebook.calendar_uri == data_contract.calendar_uri
        assert notebook.calendar_date_column == data_contract.calendar_date_column
        assert tuple(notebook.exogenous_columns) == data_contract.exogenous_columns


def test_cp26_model_facade_builds_from_migrated_configs() -> None:
    from gts_rm import models

    specs = models.mac3_model_specs()

    assert tuple(specs) == tuple(SUPPORTED_ARCHITECTURES)
    for architecture in SUPPORTED_ARCHITECTURES:
        model = models.build_mac3_smoke_model(architecture)
        spec = specs[architecture]
        assert spec.architecture == architecture
        assert model.dimensions.window_size == spec.window_size
        assert model.dimensions.horizon == spec.horizon
        assert model.dimensions.exogenous_dim == spec.exogenous_dim
        assert model.dimensions.static_dim == spec.static_dim


def test_cp26_training_facade_builds_trainers_from_candidates() -> None:
    from gts_rm import training

    candidates = training.load_mac3_candidates()
    summary = training.mac3_training_facade_summary()

    assert tuple(candidates) == tuple(SUPPORTED_ARCHITECTURES)
    assert summary["architectures"] == tuple(SUPPORTED_ARCHITECTURES)
    assert summary["selection_metric"] == "robust_macro_mase"
    for architecture in SUPPORTED_ARCHITECTURES:
        candidate = training.get_mac3_candidate(architecture)
        trainer = training.build_mac3_trainer(architecture)
        assert candidate == candidates[architecture]
        assert trainer.architecture == architecture
        assert trainer.model_config == dict(candidate.model_config)
        assert trainer.training_config.selection_metric == "robust_macro_mase"


def test_cp23_smoke_workflows_run_with_facade(tmp_path) -> None:
    from MAC3_TEST.workflows.smoke_all_global_models import run_smoke_suite

    reports = run_smoke_suite(output_root=tmp_path)

    assert list(reports) == SUPPORTED_ARCHITECTURES
    for architecture, report in reports.items():
        name = SMOKE_WORKFLOWS[architecture]
        assert report["ok"] is True
        assert report["checkpoint"] == "CP24"
        assert report["architecture"] == architecture
        assert report["actual_output_shape"] == [2, 3, 1]
        assert report["finite_prediction"] is True
        assert report["finite_history_embedding"] is True
        assert report["facade_modules"] == SMOKE_FACADE_MODULES
        assert (tmp_path / "reports" / f"{name}.json").exists()
        assert (tmp_path / "runs" / f"{name}_run.json").exists()


def test_cp24_smoke_workflows_do_not_import_cp20_modules_directly() -> None:
    workflow_paths = [
        ROOT / "MAC3_TEST" / "workflows" / "_global_smoke.py",
        ROOT / "MAC3_TEST" / "workflows" / "smoke_global_mlp.py",
        ROOT / "MAC3_TEST" / "workflows" / "smoke_global_mlp_vae.py",
        ROOT / "MAC3_TEST" / "workflows" / "smoke_global_rnn.py",
        ROOT / "MAC3_TEST" / "workflows" / "smoke_global_rnn_bi.py",
        ROOT / "MAC3_TEST" / "workflows" / "smoke_all_global_models.py",
    ]
    forbidden_imports = [
        "from global_",
        "import global_",
        "from financial_gpt_",
        "import financial_gpt_",
    ]
    for path in workflow_paths:
        source = path.read_text(encoding="utf-8")
        for forbidden in forbidden_imports:
            assert forbidden not in source

def test_cp27_acceptance_report_and_badge_are_versioned() -> None:
    report = ROOT / "MAC3_TEST" / "reports" / "CP27_ACCEPTANCE_REPORT.md"
    badge = ROOT / "MAC3_TEST" / "reports" / "api_coverage.svg"

    assert report.exists()
    assert badge.exists()
    report_text = report.read_text(encoding="utf-8")
    badge_text = badge.read_text(encoding="utf-8")
    assert "CP27 - MAC3_TEST Acceptance Report" in report_text
    assert "166 passed, 1 warning, 5 subtests passed" in report_text
    assert "32.04%" in report_text
    assert "api coverage: 32.04%" in badge_text
