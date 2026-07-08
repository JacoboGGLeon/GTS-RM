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
    assert contract.manifest["checkpoint"] == "CP23"
    assert contract.contract_path.exists()
    assert contract.cp20_bundle_path == gts_rm.CP20_BUNDLE_ROOT
    assert contract.frozen_contract_path.exists()
    assert set(SMOKE_WORKFLOWS.values()).issubset(contract.manifest["workflows"])
    assert "smoke_all_global_models" in contract.manifest["workflows"]


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
    assert base["facade_modules"] == FACADE_MODULES
    assert base["model_inputs"] == manifest["locked_cp20_contract"]["model_inputs"]
    assert base["supported_architectures"] == manifest["locked_cp20_contract"]["architectures"]
    assert base["output"] == manifest["locked_cp20_contract"]["output"]
    assert base["latent"] == manifest["locked_cp20_contract"]["latent"]
    assert list(smokes) == SUPPORTED_ARCHITECTURES
    for architecture, smoke in smokes.items():
        assert smoke["checkpoint"] == "CP23"
        assert smoke["architecture"] == architecture
        assert smoke["expected_output_shape"] == [2, 3, 1]
    assert acceptance["metrics"]["primary"] == "robust_macro_mase"
    assert acceptance["release_gate"]["must_use_library_facade"] is True
    assert acceptance["release_gate"]["must_run_smoke_workflow"] is True


def test_cp22_facade_modules_import_and_expose_expected_symbols() -> None:
    modules = {name: importlib.import_module(name) for name in FACADE_MODULES}

    assert modules["gts_rm.config"].FinancialGPTStageConfig().flags.use_static_context is True
    assert modules["gts_rm.data"].MODEL_INPUT_FIELDS == ("y_context", "x_history", "x_future", "x_static")
    assert modules["gts_rm.data"].ContextScaler is not None
    assert modules["gts_rm.models"].list_global_models() == tuple(SUPPORTED_ARCHITECTURES)
    assert modules["gts_rm.models"].GLOBAL_OUTPUT_FIELD == "y_pred"
    assert modules["gts_rm.training"].GlobalTrainingConfig is not None
    assert modules["gts_rm.training"].GlobalCurriculumConfig is not None
    assert modules["gts_rm.evaluation"].GlobalValidationMetrics is not None
    assert modules["gts_rm.evaluation"].FinancialGPTMonitorResult is not None
    assert modules["gts_rm.artifacts"].GlobalManager is not None
    assert modules["gts_rm.artifacts"].S3Location is not None


def test_cp23_smoke_workflows_run_with_facade(tmp_path) -> None:
    from MAC3_TEST.workflows.smoke_all_global_models import run_smoke_suite

    reports = run_smoke_suite(output_root=tmp_path)

    assert list(reports) == SUPPORTED_ARCHITECTURES
    for architecture, report in reports.items():
        name = SMOKE_WORKFLOWS[architecture]
        assert report["ok"] is True
        assert report["checkpoint"] == "CP23"
        assert report["architecture"] == architecture
        assert report["actual_output_shape"] == [2, 3, 1]
        assert report["finite_prediction"] is True
        assert report["finite_history_embedding"] is True
        assert report["facade_modules"] == ["gts_rm.models"]
        assert (tmp_path / "reports" / f"{name}.json").exists()
        assert (tmp_path / "runs" / f"{name}_run.json").exists()


def test_cp23_smoke_workflows_do_not_import_cp20_modules_directly() -> None:
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
