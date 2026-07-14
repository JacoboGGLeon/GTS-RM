from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from global_curriculum import GlobalTrainingScheduleConfig
from global_notebook import GlobalNotebookConfig
from global_pipeline import (
    BacktestRequest,
    ForecastRequest,
    GlobalNotebookRunContract,
    GlobalTrainingWorkflow,
    PooledTrainingRequest,
    TRAINING_PHASE_ORDER,
)
from global_training import GlobalHPOConfig, GlobalTrainingConfig
from gtrm_config import GTRMModelConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted(ROOT.glob("code_03_GLOBAL_*.ipynb")))


def _contract(
    *,
    model: GTRMModelConfig | None = None,
    training: GlobalTrainingConfig | None = None,
) -> GlobalNotebookRunContract:
    selected_model = model or GTRMModelConfig(architecture="rnn")
    notebook = GlobalNotebookConfig(
        architecture="rnn",
        global_long_uri="global.parquet",
        calendar_uri="calendar.csv",
        artifact_root="run",
        horizon=2,
        seen_validation_size=2,
        validation_unseen_fraction=0.2,
        test_unseen_fraction=0.2,
        stride=1,
        n_trials=1,
        max_window_size=3,
        model_config=selected_model,
        gtrm_stage=2,
    )
    return GlobalNotebookRunContract(
        notebook=notebook,
        model=selected_model,
        training=training
        or GlobalTrainingConfig(
            epochs=1,
            batch_size=2,
            patience=1,
            scheduler_patience=1,
        ),
        hpo=GlobalHPOConfig(
            epochs=1,
            min_resource=1,
            finalists=1,
            fidelity_epochs=1,
        ),
        schedule=GlobalTrainingScheduleConfig(
            pooled_train_epochs=1,
            pooled_continuation_epochs=0,
        ),
    )


class DummyManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_hpo_and_train(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.calls.append("hpo_training")
        return {"phase": "hpo_training"}

    def run_backtest(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.calls.append("backtest")
        return {"phase": "backtest"}

    def run_future_forecast(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.calls.append("forecast")
        return {"phase": "forecast"}


def test_run_contract_is_strict_pydantic_and_serializable() -> None:
    contract = _contract()
    assert isinstance(contract, BaseModel)
    assert contract.schema_version == "22.4"
    assert contract.phase_order == TRAINING_PHASE_ORDER
    payload = contract.to_dict()
    assert payload["schema_version"] == "22.4"
    assert payload["phase_order"] == [phase.value for phase in TRAINING_PHASE_ORDER]
    assert payload["training"]["selection_metric"] == payload["hpo"]["objective_metric"]
    assert payload["schedule"]["training_order"] == "pooled_balanced"

    with pytest.raises(ValidationError):
        GlobalNotebookRunContract(
            notebook=contract.notebook,
            model=contract.model,
            training=contract.training,
            hpo=contract.hpo,
            schedule=contract.schedule,
            unexpected=True,
        )


def test_run_contract_rejects_cross_config_head_mismatch() -> None:
    model = GTRMModelConfig(architecture="rnn", use_event_head=True)
    training = GlobalTrainingConfig(
        epochs=1,
        batch_size=2,
        patience=1,
        scheduler_patience=1,
        use_event_head=False,
    )
    with pytest.raises(ValidationError, match="use_event_head"):
        _contract(model=model, training=training)


def test_workflow_rejects_out_of_order_phases_and_completes_exact_order() -> None:
    manager = DummyManager()
    workflow = GlobalTrainingWorkflow(manager, _contract())

    with pytest.raises(RuntimeError, match="expected=hpo_and_pooled_training"):
        workflow.run_backtest(BacktestRequest(n_mc=2, batch_size=2))

    workflow.run_hpo_and_train(
        lambda window_size: window_size,
        PooledTrainingRequest(
            n_trials=1,
            train_epochs=1,
            batch_size=2,
            study_name="cp2231",
        ),
    )
    workflow.run_backtest(BacktestRequest(n_mc=2, batch_size=2))
    workflow.run_forecast(ForecastRequest(n_steps=2, n_mc=2, batch_size=2))

    assert manager.calls == ["hpo_training", "backtest", "forecast"]
    assert workflow.snapshot.completed_phases == TRAINING_PHASE_ORDER
    assert workflow.snapshot.next_phase is None
    assert workflow.snapshot.is_complete is True

    with pytest.raises(RuntimeError, match="expected=complete"):
        workflow.run_forecast(ForecastRequest(n_steps=2))


def test_forecast_request_requires_exactly_one_forecast_mode() -> None:
    assert ForecastRequest(n_steps=2).n_steps == 2
    assert ForecastRequest(start_date="2026-01-01", end_date="2026-01-02").start_date
    with pytest.raises(ValidationError, match="exactly one mode"):
        ForecastRequest()
    with pytest.raises(ValidationError, match="provided together"):
        ForecastRequest(start_date="2026-01-01")
    with pytest.raises(ValidationError, match="exactly one mode"):
        ForecastRequest(
            start_date="2026-01-01",
            end_date="2026-01-02",
            n_steps=2,
        )


def test_four_global_notebooks_use_public_ordered_contract_only() -> None:
    assert len(NOTEBOOKS) == 4
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            "GlobalNotebookRunContract(",
            "GlobalTrainingWorkflow(",
            "PooledTrainingRequest(",
            "BacktestRequest(",
            "ForecastRequest(",
            "workflow.run_hpo_and_train(",
            "workflow.run_backtest(",
            "workflow.run_forecast(",
            '"pydantic>=2.6"',
            '"notebook_run_contract.json"',
            '"workflow_snapshot.json"',
        ):
            assert token in code
        for stale_token in (
            "HPOWarmupRequest(",
            "FineTuneRequest(",
            "workflow.run_hpo_and_warmup(",
            "workflow.run_finetune(",
            "WARM_EPOCHS",
            "FINE_EPOCHS",
            "CONSOLIDATION_EPOCHS",
        ):
            assert stale_token not in code
        for private_token in (
            "manager._warmup_all(",
            "manager._finetune_all(",
            "manager._run_backtest(",
            "manager._run_forecast(",
            "manager._backtest_results",
            "manager._future_results",
            "manager._df_forecasts",
            "manager._df_outliers",
        ):
            assert private_token not in code
        assert all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(path), "exec")


def test_global_manager_exposes_standardized_public_phase_methods() -> None:
    from global_manager import GlobalManager

    for name in (
        "run_hpo_and_train",
        "run_backtest",
        "run_future_forecast",
        "run_results",
    ):
        assert callable(getattr(GlobalManager, name))
