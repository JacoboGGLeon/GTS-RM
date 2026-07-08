from __future__ import annotations

import unittest

import torch

from financial_gpt_flags import FinancialGPTStageConfig
from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    SERIES_TYPE_COLUMN,
    SUPPORTED_ARCHITECTURES,
    validate_model_input_fields,
)
from global_models import (
    GLOBAL_LATENT_FIELD,
    GLOBAL_MODEL_REGISTRY,
    GLOBAL_OUTPUT_FIELD,
    build_global_model,
    list_global_models,
    validate_global_forward_contract,
)


class TestCP20BaselineLock(unittest.TestCase):
    def test_model_input_contract_is_frozen(self) -> None:
        self.assertEqual(
            MODEL_INPUT_FIELDS,
            ("y_context", "x_history", "x_future", "x_static"),
        )
        self.assertEqual(validate_model_input_fields(MODEL_INPUT_FIELDS), MODEL_INPUT_FIELDS)

    def test_identity_and_raw_categories_stay_out_of_forward(self) -> None:
        self.assertEqual(
            FORBIDDEN_MODEL_INPUT_FIELDS,
            (
                CROSS_KEY_COLUMN,
                ACCOUNT_CURRENCY_ID_COLUMN,
                CURRENCY_COLUMN,
                SERIES_TYPE_COLUMN,
                "serie",
            ),
        )
        for forbidden in FORBIDDEN_MODEL_INPUT_FIELDS:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    validate_model_input_fields((*MODEL_INPUT_FIELDS, forbidden))

    def test_supported_architectures_are_the_cp20_set(self) -> None:
        expected = ("mlp", "mlp_vae", "rnn", "rnn_bi")
        self.assertEqual(SUPPORTED_ARCHITECTURES, expected)
        self.assertEqual(list_global_models(), expected)
        self.assertEqual(tuple(GLOBAL_MODEL_REGISTRY), expected)
        for model_class in GLOBAL_MODEL_REGISTRY.values():
            self.assertEqual(validate_global_forward_contract(model_class), MODEL_INPUT_FIELDS)

    def test_default_feature_flags_preserve_cp20_behavior(self) -> None:
        config = FinancialGPTStageConfig()
        config.validate()
        self.assertTrue(config.flags.use_causal_scaler)
        self.assertTrue(config.flags.use_calendar_future)
        self.assertTrue(config.flags.use_static_context)
        self.assertTrue(config.flags.use_auxiliary_autoencoder)
        self.assertFalse(config.flags.use_observed_mask)
        self.assertFalse(config.flags.use_context_mask)
        self.assertFalse(config.flags.use_patch_tokenizer)
        self.assertFalse(config.flags.use_local_residual_decoder)
        self.assertFalse(config.flags.use_quantile_head)
        self.assertFalse(config.flags.use_self_supervised_pretraining)

    def test_global_model_output_contract_is_stable(self) -> None:
        torch.manual_seed(7)
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
            window_size=5,
            horizon=3,
            exogenous_dim=2,
            static_dim=4,
        )
        output = model(
            y_context=torch.zeros(2, 5, 1),
            x_history=torch.zeros(2, 5, 2),
            x_future=torch.zeros(2, 3, 2),
            x_static=torch.zeros(2, 4),
        )

        self.assertIn(GLOBAL_OUTPUT_FIELD, output)
        self.assertEqual(tuple(output[GLOBAL_OUTPUT_FIELD].shape), (2, 3, 1))
        self.assertIn("extras", output)
        self.assertIn(GLOBAL_LATENT_FIELD, output["extras"])
        self.assertEqual(output["extras"]["use_auxiliary_autoencoder"], False)


if __name__ == "__main__":
    unittest.main()
