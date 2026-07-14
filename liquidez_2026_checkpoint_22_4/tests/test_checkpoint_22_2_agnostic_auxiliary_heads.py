import unittest

import torch

from global_models import (
    DIRECTION_LOGITS_FIELD,
    EVENT_LOGITS_FIELD,
    MAGNITUDE_PRED_FIELD,
    GlobalMLPEncoderDecoder,
)
from gtrm_config import GTRMModelConfig


class TestCheckpoint222AgnosticAuxiliaryHeads(unittest.TestCase):
    def test_gtrm_config_accepts_auxiliary_heads_in_stage2(self):
        cfg = GTRMModelConfig(
            architecture="mlp",
            use_static_context=True,
            use_local_residual_decoder=True,
            use_event_head=True,
            use_magnitude_head=True,
            use_direction_head=True,
            event_threshold=1.0,
            magnitude_transform="asinh",
        )
        cfg.validate(stage=2)
        flags = cfg.stage_flags()
        self.assertTrue(flags["use_event_head"])
        self.assertTrue(flags["use_magnitude_head"])
        self.assertTrue(flags["use_direction_head"])
        self.assertEqual(cfg.dataset_kwargs()["magnitude_transform"], "asinh")

    def test_stage1_rejects_auxiliary_heads(self):
        cfg = GTRMModelConfig(architecture="mlp", use_event_head=True)
        with self.assertRaisesRegex(ValueError, "auxiliary heads"):
            cfg.validate(stage=1)

    def test_mlp_outputs_event_magnitude_direction_heads(self):
        model = GlobalMLPEncoderDecoder(
            {
                "latent_dim": 8,
                "enc_hidden_size": 16,
                "enc_num_layers": 1,
                "dec_hidden_size": 16,
                "dec_num_layers": 1,
                "dropout_rate": 0.0,
                "use_auxiliary_autoencoder": False,
                "use_local_residual_decoder": True,
                "local_residual_hidden_size": 8,
                "local_residual_num_layers": 1,
                "use_event_head": True,
                "use_magnitude_head": True,
                "use_direction_head": True,
                "auxiliary_head_hidden_size": 8,
                "auxiliary_head_num_layers": 1,
            },
            window_size=4,
            horizon=3,
            exogenous_dim=2,
            static_dim=5,
        )
        batch = 2
        output = model(
            torch.randn(batch, 4, 1),
            torch.randn(batch, 4, 2),
            torch.randn(batch, 3, 2),
            torch.randn(batch, 5),
        )
        extras = output["extras"]
        self.assertEqual(tuple(extras[EVENT_LOGITS_FIELD].shape), (batch, 3, 1))
        self.assertEqual(tuple(extras[MAGNITUDE_PRED_FIELD].shape), (batch, 3, 1))
        self.assertEqual(tuple(extras[DIRECTION_LOGITS_FIELD].shape), (batch, 3, 3))


if __name__ == "__main__":
    unittest.main()
