import importlib.util
import unittest

import torch

from global_models import EVENT_LOGITS_FIELD, MAGNITUDE_PRED_FIELD, DIRECTION_LOGITS_FIELD


@unittest.skipIf(importlib.util.find_spec("polars") is None, "polars is not installed")
class TestCheckpoint2221AuxiliaryLossBlockHPO(unittest.TestCase):
    def test_auxiliary_loss_block_requires_normalized_shares(self):
        from global_training import GlobalTrainingConfig

        cfg = GlobalTrainingConfig(
            use_event_head=True,
            use_magnitude_head=True,
            use_direction_head=True,
            use_auxiliary_loss_block=True,
            auxiliary_loss_weight=0.2,
            event_loss_share=0.4,
            magnitude_loss_share=0.4,
            direction_loss_share=0.2,
        )
        cfg.validate()

        bad = GlobalTrainingConfig(
            use_event_head=True,
            use_auxiliary_loss_block=True,
            event_loss_share=0.4,
            magnitude_loss_share=0.4,
            direction_loss_share=0.4,
        )
        with self.assertRaisesRegex(ValueError, "must equal 1.0"):
            bad.validate()

    def test_global_forecast_loss_uses_normalized_auxiliary_block(self):
        from global_training import global_forecast_loss

        target = torch.zeros(2, 3, 1)
        prediction = torch.zeros_like(target)
        event_logits = torch.zeros_like(target)
        magnitude_pred = torch.zeros_like(target)
        direction_logits = torch.zeros(2, 3, 3)

        output = {
            "y_pred": prediction,
            "extras": {
                "use_auxiliary_loss_block": True,
                "auxiliary_loss_weight": 0.20,
                "event_loss_share": 0.40,
                "magnitude_loss_share": 0.40,
                "direction_loss_share": 0.20,
                EVENT_LOGITS_FIELD: event_logits,
                MAGNITUDE_PRED_FIELD: magnitude_pred,
                DIRECTION_LOGITS_FIELD: direction_logits,
            },
        }
        auxiliary_targets = {
            "event_target": torch.ones_like(event_logits),
            "magnitude_target": torch.ones_like(magnitude_pred),
            "direction_target": torch.ones(2, 3, dtype=torch.long),
        }
        loss = global_forecast_loss(
            output,
            target,
            loss="huber",
            auxiliary_targets=auxiliary_targets,
        )
        expected = (
            0.20 * 0.40 * torch.nn.functional.binary_cross_entropy_with_logits(
                event_logits, auxiliary_targets["event_target"]
            )
            + 0.20 * 0.40 * torch.nn.functional.huber_loss(
                magnitude_pred, auxiliary_targets["magnitude_target"], delta=1.0
            )
            + 0.20 * 0.20 * torch.nn.functional.cross_entropy(
                direction_logits.reshape(-1, 3), auxiliary_targets["direction_target"].reshape(-1)
            )
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_legacy_per_head_weights_remain_supported(self):
        from global_training import global_forecast_loss

        target = torch.zeros(1, 2, 1)
        event_logits = torch.zeros_like(target)
        output = {
            "y_pred": target.clone(),
            "extras": {
                "use_auxiliary_loss_block": False,
                "event_loss_weight": 0.5,
                EVENT_LOGITS_FIELD: event_logits,
            },
        }
        auxiliary_targets = {"event_target": torch.ones_like(event_logits)}
        loss = global_forecast_loss(
            output,
            target,
            loss="huber",
            auxiliary_targets=auxiliary_targets,
        )
        expected = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            event_logits, auxiliary_targets["event_target"]
        )
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
