from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from global_curriculum import GlobalTrainingScheduleConfig
from global_manager import GlobalManager
from global_pipeline import PooledTrainingRequest
from global_training import GlobalTrainingConfig


def test_standard_schedule_has_one_primary_stage_and_optional_continuation() -> None:
    stages = GlobalTrainingScheduleConfig(
        pooled_train_epochs=60,
        pooled_continuation_epochs=7,
        pooled_continuation_lr_factor=0.25,
    ).build_stages((1, 2, 3))

    assert [(stage.name, stage.phase, stage.epochs) for stage in stages] == [
        ("pooled_full_training", "productive_training", 60),
        ("pooled_continuation", "pooled_continuation", 7),
    ]
    assert all(stage.current_levels == (1, 2, 3) for stage in stages)
    assert all(stage.replay_levels == () for stage in stages)
    assert all(stage.replay_fraction == 0.0 for stage in stages)


def test_pooled_training_request_rejects_invalid_continuation_budget() -> None:
    with pytest.raises(ValueError):
        PooledTrainingRequest(
            n_trials=1,
            train_epochs=1,
            continuation_epochs=-1,
            batch_size=2,
            study_name="invalid",
        )
    with pytest.raises(ValueError):
        PooledTrainingRequest(
            n_trials=1,
            train_epochs=1,
            continuation_lr_factor=1.1,
            batch_size=2,
            study_name="invalid",
        )


def test_manager_runs_hpo_and_both_pooled_stages_in_one_public_call() -> None:
    schedule = GlobalTrainingScheduleConfig(
        pooled_train_epochs=3,
        pooled_continuation_epochs=0,
    )
    manager = GlobalManager(
        "mlp",
        base_training_config=GlobalTrainingConfig(
            epochs=1,
            batch_size=2,
            patience=1,
            scheduler_patience=1,
        ),
        schedule_config=schedule,
    )
    candidate = SimpleNamespace(
        window_size=3,
        model_config={},
        training_config=GlobalTrainingConfig(
            epochs=1,
            batch_size=2,
            patience=1,
            scheduler_patience=1,
        ),
    )
    hpo = SimpleNamespace(best_candidate=candidate)
    train = SimpleNamespace(
        exogenous_columns=(),
        static_feature_names=("static",),
        static_feature_encoder=None,
    )
    validation = SimpleNamespace(exogenous_columns=())
    bundle = SimpleNamespace(
        train=train,
        validation_datasets={
            "validation_seen": validation,
            "validation_unseen": validation,
        },
        window_size=3,
        horizon=2,
        exogenous_dim=0,
        static_dim=1,
        static_feature_names=("static",),
        validate=Mock(),
    )
    factory = Mock(return_value=bundle)
    result = SimpleNamespace(stages=())
    session = Mock()
    session.run_phases.return_value = result

    with patch(
        "global_manager.GlobalHPOTrainer.search_and_fit",
        return_value=hpo,
    ), patch(
        "global_manager.GlobalCurriculumSession",
        return_value=session,
    ) as session_cls, patch.object(
        manager,
        "_phase_result",
        return_value={"stages": []},
    ):
        returned = manager.run_hpo_and_train(
            factory,
            n_trials=1,
            train_epochs=5,
            continuation_epochs=2,
            continuation_lr_factor=0.3,
            batch=4,
        )

    assert returned is result
    effective_schedule = session_cls.call_args.args[4]
    assert effective_schedule.training_order == "pooled_balanced"
    assert effective_schedule.pooled_train_epochs == 5
    assert effective_schedule.pooled_continuation_epochs == 2
    assert effective_schedule.pooled_continuation_lr_factor == 0.3
    session.run_phases.assert_called_once()
    assert session.run_phases.call_args.args[0] == (
        "productive_training",
        "pooled_continuation",
    )
    assert manager.run_results()["training"] == {"stages": []}


def test_curriculum_and_shuffled_remain_explicit_ablations() -> None:
    curriculum = GlobalTrainingScheduleConfig(
        training_order="curriculum",
        warmup_epochs=1,
        fine_tune_epochs_per_level=1,
        consolidation_epochs=1,
    ).build_stages((1, 2))
    shuffled = GlobalTrainingScheduleConfig(
        training_order="shuffled",
        warmup_epochs=1,
        fine_tune_epochs_per_level=1,
        consolidation_epochs=1,
    ).build_stages((1, 2))

    assert [stage.phase for stage in curriculum] == [
        "warmup",
        "finetune",
        "consolidation",
    ]
    assert [stage.phase for stage in shuffled] == [
        "warmup",
        "finetune",
        "consolidation",
    ]
