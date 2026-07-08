from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_gts_rm_exports_cp20_core() -> None:
    import gts_rm

    assert gts_rm.CP20_BUNDLE_ROOT.exists()
    assert gts_rm.MAC3_TEST_ROOT.exists()
    assert gts_rm.list_global_models() == ("mlp", "mlp_vae", "rnn", "rnn_bi")
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
    assert contract.manifest["checkpoint"] == "CP21"
    assert contract.contract_path.exists()
    assert contract.cp20_bundle_path == gts_rm.CP20_BUNDLE_ROOT
    assert contract.frozen_contract_path.exists()


def test_mac3_test_configs_match_locked_cp20_contract() -> None:
    manifest = json.loads((ROOT / "MAC3_TEST" / "manifest.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / manifest["configs"]["base"]).read_text(encoding="utf-8"))
    acceptance = json.loads((ROOT / manifest["configs"]["acceptance"]).read_text(encoding="utf-8"))

    assert manifest["kind"] == "use_case"
    assert manifest["release_first"] is True
    assert manifest["tutorials_deferred"] is True
    assert base["model_inputs"] == manifest["locked_cp20_contract"]["model_inputs"]
    assert base["supported_architectures"] == manifest["locked_cp20_contract"]["architectures"]
    assert base["output"] == manifest["locked_cp20_contract"]["output"]
    assert base["latent"] == manifest["locked_cp20_contract"]["latent"]
    assert acceptance["metrics"]["primary"] == "robust_macro_mase"
