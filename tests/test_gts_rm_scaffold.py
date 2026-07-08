from __future__ import annotations

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
