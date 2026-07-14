from __future__ import annotations

import json
from pathlib import Path

from global_curriculum import GlobalTrainingScheduleConfig

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_NOTEBOOKS = (
    "code_03_GLOBAL_MLP_E_D.ipynb",
    "code_03_GLOBAL_MLP_VaE_D.ipynb",
    "code_03_GLOBAL_RNN_E_D.ipynb",
    "code_03_GLOBAL_RNNBi_E_D.ipynb",
)


def test_pooled_balanced_is_the_canonical_productive_schedule() -> None:
    config = GlobalTrainingScheduleConfig(
        pooled_train_epochs=60,
        pooled_continuation_epochs=5,
        pooled_continuation_lr_factor=0.2,
    )
    stages = config.build_stages((1, 2, 3))
    assert config.training_order == "pooled_balanced"
    assert [stage.phase for stage in stages] == [
        "productive_training",
        "pooled_continuation",
    ]
    assert stages[0].name == "pooled_full_training"
    assert stages[0].current_levels == (1, 2, 3)
    assert stages[0].replay_levels == ()
    assert stages[0].epochs == 60
    assert stages[1].epochs == 5
    assert stages[1].learning_rate_factor == 0.2


def test_global_notebooks_expose_only_the_standardized_pooled_budget() -> None:
    for notebook_name in GLOBAL_NOTEBOOKS:
        notebook = json.loads((ROOT / notebook_name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "Checkpoint 22.3" in code
        assert 'TRAINING_STRATEGY = "pooled_balanced"' in code
        assert "POOLED_TRAIN_EPOCHS = 60" in code
        assert "POOLED_TRAIN_BATCH = 512" in code
        assert "POOLED_CONTINUATION_EPOCHS = 0" in code
        assert "PooledTrainingRequest(" in code
        assert "workflow.run_hpo_and_train(" in code
        for stale in (
            "WARM_EPOCHS",
            "FINE_EPOCHS",
            "CONSOLIDATION_EPOCHS",
            "TRAINING_ORDER",
            "workflow.run_finetune(",
        ):
            assert stale not in code
