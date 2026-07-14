"""MC-Dropout, forecast futuro, outliers y visualización por serie.

Este módulo adapta el único modelo global a los contratos analíticos del pipeline
legacy. Los pesos siguen siendo compartidos; únicamente resultados, métricas,
outliers y figuras se separan por ``cross_key_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from global_contracts import (
    ACCOUNT_CURRENCY_ID_COLUMN,
    CROSS_KEY_COLUMN,
    CURRENCY_COLUMN,
    DATE_COLUMN,
    SERIES_AGE_COLUMN,
    SERIES_TYPE_COLUMN,
    TARGET_COLUMN,
)
from global_data import (
    ContextScale,
    ContextScaler,
    GlobalWindowDataset,
    StaticFeatureEncoder,
)
from global_models import GlobalForecastModel
from global_long_schema import upgrade_global_long_checkpoint19
from tools import Tools
from temporal_axis import (
    ForecastRequest,
    InsufficientFutureContextError,
    TemporalAxis,
    TemporalWindowAligner,
)
from p0_diagnostics import interval_calibration_by_horizon


BACKTEST_COLUMNS: Tuple[str, ...] = (
    "date",
    "serie",
    "cross_key_id",
    "account_currency_id",
    "tipo_serie",
    "cutoff",
    "horizon_step",
    "isTrain",
    "actual_orig",
    "pred_orig",
    "lower_ci",
    "upper_ci",
    "bias2",
    "var_pred",
)

FORECAST_COLUMNS: Tuple[str, ...] = (
    "date",
    "serie",
    "cross_key_id",
    "account_currency_id",
    "tipo_serie",
    "pred_orig",
    "lower_ci",
    "upper_ci",
    "outlier_level",
)


@dataclass(frozen=True)
class MCDropoutConfig:
    n_mc: int = 100
    lower_quantile: float = 0.025
    upper_quantile: float = 0.975
    batch_size: int = 256
    device: str = "auto"

    def validate(self) -> None:
        if isinstance(self.n_mc, bool) or int(self.n_mc) <= 0:
            raise ValueError("n_mc must be a positive integer")
        if isinstance(self.batch_size, bool) or int(self.batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not 0.0 <= float(self.lower_quantile) < float(self.upper_quantile) <= 1.0:
            raise ValueError("MC quantiles must satisfy 0 <= lower < upper <= 1")
        if not str(self.device).strip():
            raise ValueError("device must not be empty")



@dataclass(frozen=True)
class BacktestRunReport:
    total_series: int
    processed_series: int
    skipped_series: int
    train_windows: int
    test_windows: int
    output_rows: int
    mc_samples: int
    elapsed_seconds: float

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "total_series": self.total_series,
            "processed_series": self.processed_series,
            "skipped_series": self.skipped_series,
            "train_windows": self.train_windows,
            "test_windows": self.test_windows,
            "output_rows": self.output_rows,
            "mc_samples": self.mc_samples,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def format_summary(self) -> str:
        return (
            "Backtest MC-Dropout completado\n"
            f"  Series: {self.processed_series}/{self.total_series} "
            f"(omitidas={self.skipped_series})\n"
            f"  Ventanas train/test: {self.train_windows}/{self.test_windows}\n"
            f"  Filas consolidadas: {self.output_rows}\n"
            f"  MC samples: {self.mc_samples}\n"
            f"  Tiempo: {self.elapsed_seconds:.2f} s"
        )

def mc_dropout_backtest(
    model: GlobalForecastModel,
    train_dataset: GlobalWindowDataset,
    validation_seen_dataset: GlobalWindowDataset,
    *,
    config: MCDropoutConfig | None = None,
) -> Mapping[str, Any]:
    """Rolling backtest train/test por ``cross_key_id`` con MC-Dropout.

    Las ventanas superpuestas se agregan por fecha para conservar una sola fila
    por serie y fecha, como espera la visualización histórica.
    """

    cfg = config or MCDropoutConfig()
    cfg.validate()
    started = time.perf_counter()
    train = _predict_window_dataset_mc(model, train_dataset, is_train=True, config=cfg)
    test = _predict_window_dataset_mc(
        model, validation_seen_dataset, is_train=False, config=cfg
    )
    raw = pd.concat([train, test], ignore_index=True)
    if raw.empty:
        consolidated = pd.DataFrame(columns=BACKTEST_COLUMNS)
    else:
        group_columns = [
            "date",
            "serie",
            "cross_key_id",
            "account_currency_id",
            "tipo_serie",
            "isTrain",
        ]
        consolidated = (
            raw.groupby(group_columns, as_index=False, dropna=False)
            .agg(
                cutoff=("cutoff", "max"),
                horizon_step=("horizon_step", "min"),
                actual_orig=("actual_orig", "mean"),
                pred_orig=("pred_orig", "mean"),
                lower_ci=("lower_ci", "mean"),
                upper_ci=("upper_ci", "mean"),
                bias2=("bias2", "mean"),
                var_pred=("var_pred", "mean"),
            )
            .sort_values(["serie", "date", "isTrain"])
            .reset_index(drop=True)
        )
        consolidated = consolidated.loc[:, list(BACKTEST_COLUMNS)]

    metric_rows: list[dict[str, Any]] = []
    by_series: Dict[str, Any] = {}
    train_bounds: Dict[str, Mapping[int, float]] = {}
    for serie, frame in consolidated.groupby("serie", sort=True):
        train_frame = frame.loc[frame["isTrain"].astype(bool)].copy()
        test_frame = frame.loc[~frame["isTrain"].astype(bool)].copy()
        reference = train_frame if not train_frame.empty else frame
        train_bounds[str(serie)] = Tools._compute_z_stats(
            reference["actual_orig"].to_numpy()
        )[0]
        evaluation = test_frame if not test_frame.empty else frame.iloc[0:0]
        if evaluation.empty:
            metrics = {name: float("nan") for name in (
                "MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "WMAPE (%)",
                "MASE", "MedAE", "MedAPE (%)", "EVS", "R2"
            )}
            metrics.update({"PICP": float("nan"), "MPIW": float("nan"), "Winkler": float("nan")})
        else:
            metrics = Tools.metrics_regression(
                evaluation["actual_orig"].to_numpy(),
                evaluation["pred_orig"].to_numpy(),
            )
            actual = evaluation["actual_orig"].to_numpy(dtype=float)
            lower = evaluation["lower_ci"].to_numpy(dtype=float)
            upper = evaluation["upper_ci"].to_numpy(dtype=float)
            alpha = max(1e-8, 1.0 - (cfg.upper_quantile - cfg.lower_quantile))
            covered = (actual >= lower) & (actual <= upper)
            width = upper - lower
            winkler = width.copy()
            below = actual < lower
            above = actual > upper
            winkler[below] += (2.0 / alpha) * (lower[below] - actual[below])
            winkler[above] += (2.0 / alpha) * (actual[above] - upper[above])
            metrics.update({
                "PICP": float(np.mean(covered)),
                "MPIW": float(np.mean(width)),
                "Winkler": float(np.mean(winkler)),
            })
        metrics["serie"] = str(serie)
        metrics["evaluation_scope"] = "test_only"
        metric_rows.append(metrics)
        by_series[str(serie)] = {
            "df_regression": frame.reset_index(drop=True),
            "df_regression_metrics": pd.DataFrame([metrics]),
        }
    all_series = set(train_dataset.series_ids) | set(validation_seen_dataset.series_ids)
    report = BacktestRunReport(
        total_series=len(all_series),
        processed_series=len(by_series),
        skipped_series=max(0, len(all_series) - len(by_series)),
        train_windows=len(train_dataset),
        test_windows=len(validation_seen_dataset),
        output_rows=len(consolidated),
        mc_samples=int(cfg.n_mc),
        elapsed_seconds=float(time.perf_counter() - started),
    )
    return {
        "df_regression": consolidated,
        "df_regression_metrics": pd.DataFrame(metric_rows),
        "interval_calibration_by_horizon": interval_calibration_by_horizon(
            consolidated, nominal_coverage=float(cfg.upper_quantile - cfg.lower_quantile)
        ) if not consolidated.empty else pd.DataFrame(),
        "by_series": by_series,
        "train_bounds": train_bounds,
        "run_report": report,
    }


def forecast_future_mc(
    model: GlobalForecastModel,
    global_long: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    window_size: int,
    horizon: int,
    exogenous_columns: Sequence[str],
    static_feature_encoder: StaticFeatureEncoder,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    n_steps: int | None = None,
    max_steps: int | None = None,
    series_ids: Sequence[str] | None = None,
    config: MCDropoutConfig | None = None,
) -> Tuple[Mapping[str, pd.DataFrame], pd.DataFrame]:
    """Forecast sobre pasos válidos del proveedor temporal.

    El núcleo no interpreta frecuencias ni días hábiles. En modo ``n_steps``
    toma las próximas N filas del ``TemporalAxis``; en modo rango utiliza sólo
    los timestamps del eje comprendidos entre ``start_date`` y ``end_date``.
    """

    cfg = config or MCDropoutConfig()
    cfg.validate()
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size and rollout chunk horizon must be positive")
    request = ForecastRequest(n_steps=n_steps, start=start_date, end=end_date)
    request.validate()
    if max_steps is not None:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer when provided")
        if n_steps is not None and int(n_steps) > int(max_steps):
            raise ValueError("n_steps cannot exceed max_steps")

    frame = upgrade_global_long_checkpoint19(global_long).sort([CROSS_KEY_COLUMN, DATE_COLUMN])
    required = {
        DATE_COLUMN,
        CROSS_KEY_COLUMN,
        ACCOUNT_CURRENCY_ID_COLUMN,
        CURRENCY_COLUMN,
        SERIES_AGE_COLUMN,
        SERIES_TYPE_COLUMN,
        TARGET_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"global_long is missing columns: {missing}")

    selected_ids = (
        tuple(str(value) for value in series_ids)
        if series_ids is not None
        else tuple(sorted(str(v) for v in frame[CROSS_KEY_COLUMN].unique().to_list()))
    )
    if not selected_ids:
        raise ValueError("At least one series is required for future forecasting")

    axis = TemporalAxis.from_frame(
        calendar,
        timestamp_column=DATE_COLUMN,
        feature_columns=exogenous_columns,
    )
    aligned_frame, _ = TemporalWindowAligner(axis).align_global_long(frame)

    device = _resolve_device(cfg.device)
    model = model.to(device)
    results: Dict[str, pd.DataFrame] = {}
    all_frames: list[pd.DataFrame] = []
    scaler = ContextScaler()

    for series_id in selected_ids:
        series = aligned_frame.filter(pl.col(CROSS_KEY_COLUMN) == series_id).sort(DATE_COLUMN)
        if series.height < window_size:
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(series[DATE_COLUMN].to_list()))
        values = series[TARGET_COLUMN].to_numpy().astype(np.float64, copy=True)
        first = series.row(0, named=True)
        series_type = str(first[SERIES_TYPE_COLUMN])
        currency = str(first[CURRENCY_COLUMN])
        last_observed_age = int(series[SERIES_AGE_COLUMN][-1])
        last_observed = pd.Timestamp(dates[-1])
        requested_dates = axis.resolve(request, anchor=last_observed)
        if max_steps is not None and len(requested_dates) > int(max_steps):
            raise ValueError(
                "requested forecast range exceeds max_steps: "
                f"requested={len(requested_dates)}, max_steps={int(max_steps)}"
            )
        requested_set = set(requested_dates)
        final_requested = pd.Timestamp(requested_dates[-1])

        history_dates = list(dates)
        history_values = list(values.astype(float))
        rows: list[dict[str, Any]] = []
        history_bounds = Tools._compute_z_stats(values)[0]
        emitted: set[pd.Timestamp] = set()
        generated_steps = 0

        while not requested_set.issubset(emitted):
            context_dates = pd.DatetimeIndex(history_dates[-window_size:])
            context_values = np.asarray(history_values[-window_size:], dtype=np.float64)
            if len(context_values) != window_size:
                raise RuntimeError(f"Insufficient context for {series_id!r}")

            # ``horizon`` es aquí el rollout chunk persistido del modelo, no el
            # horizonte total solicitado. El modelo emite K pasos conjuntamente;
            # si faltan más timestamps, la media del bloque se agrega al contexto
            # y se ejecuta otro bloque hasta cubrir ``n_steps`` o el rango.
            block_dates = axis.after(history_dates[-1], n_steps=horizon)
            parameters = scaler.fit(context_values, series_type=series_type)
            y_context = scaler.transform(context_values, parameters).astype(np.float32)
            x_history = axis.features_for(context_dates)
            x_future = axis.features_for(block_dates)
            cutoff_age = last_observed_age + generated_steps
            x_static = static_feature_encoder.encode(
                series_type=series_type,
                currency=currency,
                scale=parameters.scale,
                series_age=cutoff_age,
            )
            inputs = {
                "y_context": torch.as_tensor(y_context[:, None])[None, ...].to(device),
                "x_history": torch.as_tensor(x_history)[None, ...].to(device),
                "x_future": torch.as_tensor(x_future)[None, ...].to(device),
                "x_static": torch.as_tensor(x_static)[None, ...].to(device),
            }
            samples_scaled = _mc_forward(model, inputs, cfg.n_mc)
            samples_raw = ContextScaler.inverse_transform(
                samples_scaled[:, 0, :, 0], parameters
            )
            mean = samples_raw.mean(axis=0)
            lower = np.quantile(samples_raw, cfg.lower_quantile, axis=0)
            upper = np.quantile(samples_raw, cfg.upper_quantile, axis=0)
            flags = Tools._flag_from_bounds(pd.Series(mean, index=block_dates), history_bounds)

            for offset, target_date in enumerate(block_dates):
                target_ts = pd.Timestamp(target_date)
                if target_ts in requested_set:
                    rows.append(
                        {
                            "date": target_ts,
                            "serie": series_id,
                            "cross_key_id": series_id,
                            "account_currency_id": str(first[ACCOUNT_CURRENCY_ID_COLUMN]),
                            "tipo_serie": series_type,
                            "pred_orig": float(mean[offset]),
                            "lower_ci": float(lower[offset]),
                            "upper_ci": float(upper[offset]),
                            "outlier_level": int(flags.iloc[offset]),
                        }
                    )
                    emitted.add(target_ts)

            history_dates.extend(list(block_dates))
            history_values.extend(mean.astype(float).tolist())
            generated_steps += len(block_dates)
            if pd.Timestamp(block_dates[-1]) >= final_requested and not requested_set.issubset(emitted):
                missing_requested = sorted(requested_set.difference(emitted))
                raise InsufficientFutureContextError(
                    f"Unable to reach requested temporal steps for {series_id!r}; "
                    f"first_missing={missing_requested[0]}"
                )

        if rows:
            df = pd.DataFrame(rows).set_index("date").sort_index()
            df.index = pd.to_datetime(df.index)
            results[series_id] = df[["pred_orig", "lower_ci", "upper_ci", "outlier_level"]]
            all_frames.append(pd.DataFrame(rows))

    consolidated = (
        pd.concat(all_frames, ignore_index=True)
        if all_frames
        else pd.DataFrame(columns=FORECAST_COLUMNS)
    )
    if not consolidated.empty:
        consolidated = consolidated.loc[:, list(FORECAST_COLUMNS)]
    return results, consolidated


def build_train_reference_outliers(
    backtest_results: Mapping[str, Any],
) -> pd.DataFrame:
    """Hierarchical outliers fitted only on real training observations."""
    frame = backtest_results.get("df_regression")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    datasets: Dict[str, pd.DataFrame] = {}
    train = frame.loc[frame["isTrain"].astype(bool)].copy()
    for series_id, part in train.groupby("serie", sort=True):
        values = part[["date", "actual_orig"]].drop_duplicates("date").copy()
        values["date"] = pd.to_datetime(values["date"])
        values = values.set_index("date").sort_index()
        values["total_amount"] = values["actual_orig"].astype(float)
        values["y"] = values["total_amount"]
        datasets[str(series_id)] = values[["total_amount", "y"]]
    if not datasets:
        return pd.DataFrame()
    return Tools().build_hierarchical_outliers(datasets)


def build_legacy_series_and_outliers(
    global_long: pl.DataFrame,
) -> Tuple[Mapping[str, pd.DataFrame], pd.DataFrame]:
    """Reconstruye los objetos legacy usados por las tres figuras existentes."""

    datasets: Dict[str, pd.DataFrame] = {}
    for key, part in global_long.partition_by(CROSS_KEY_COLUMN, as_dict=True).items():
        series_id = str(key[0] if isinstance(key, tuple) else key)
        pandas = pd.DataFrame(part.select([DATE_COLUMN, TARGET_COLUMN]).sort(DATE_COLUMN).to_dicts())
        pandas[DATE_COLUMN] = pd.to_datetime(pandas[DATE_COLUMN])
        pandas = pandas.set_index(DATE_COLUMN)
        pandas["total_amount"] = pandas[TARGET_COLUMN].astype(float)
        pandas["y"] = pandas["total_amount"]
        datasets[series_id] = pandas[["total_amount", "y"]]
    if not datasets:
        return {}, pd.DataFrame()
    outliers = Tools().build_hierarchical_outliers(datasets)
    return datasets, outliers


def visualise_legacy_contract(
    *,
    backtest_results: Mapping[str, Any],
    future_results: Mapping[str, pd.DataFrame],
    global_long: pl.DataFrame,
    bt_start: str,
    bt_end: str,
    fc_start: str,
    fc_end: str,
    series_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Genera exactamente las tres figuras legacy por serie."""

    datasets, outliers = build_legacy_series_and_outliers(global_long)
    available = sorted(set(future_results).intersection(datasets))
    selected = available if series_ids is None else [str(v) for v in series_ids if str(v) in available]
    tools = Tools()
    train_bounds = dict(backtest_results.get("train_bounds", {}))
    for series_id in selected:
        bounds = train_bounds.get(series_id)
        tools.plot_backtest_for_serie(
            backtest_results=dict(backtest_results),
            serie=series_id,
            bounds=bounds,
        )
        tools.plot_forecast_with_outliers(
            serie_name=series_id,
            future_results=dict(future_results),
            dict_series_clean=dict(datasets),
            bounds=bounds,
        )
        tools.plot_backtest_forecast_with_outliers_for_serie(
            serie_name=series_id,
            backtest_results=dict(backtest_results),
            future_results=dict(future_results),
            dict_series_clean=dict(datasets),
            build_hierarchical_outliers=tools.build_hierarchical_outliers,
            bt_start=bt_start,
            bt_end=bt_end,
            fc_start=fc_start,
            fc_end=fc_end,
            bounds=bounds,
        )
    return build_train_reference_outliers(backtest_results)


def _predict_window_dataset_mc(
    model: GlobalForecastModel,
    dataset: GlobalWindowDataset,
    *,
    is_train: bool,
    config: MCDropoutConfig,
) -> pd.DataFrame:
    device = _resolve_device(config.device)
    model = model.to(device)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(dataset), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        inputs = {name: value.to(device) for name, value in batch["model_inputs"].items()}
        metadata = batch["metadata"]
        series_types = list(metadata[SERIES_TYPE_COLUMN])
        samples_scaled = _mc_forward(model, inputs, config.n_mc)
        actual = batch["targets"]["y_future_raw"].detach().cpu().numpy()
        centers = _as_numpy(metadata["center"])
        scales = _as_numpy(metadata["scale"])
        transforms = list(metadata.get("transform", ["identity"] * len(series_types)))
        samples_raw = np.empty_like(samples_scaled, dtype=np.float64)
        for row in range(samples_scaled.shape[1]):
            samples_raw[:, row] = ContextScaler.inverse_transform(
                samples_scaled[:, row],
                ContextScale(
                    center=float(centers[row]),
                    scale=float(scales[row]),
                    transform=str(transforms[row]),
                ),
            )
        mean = samples_raw.mean(axis=0)
        lower = np.quantile(samples_raw, config.lower_quantile, axis=0)
        upper = np.quantile(samples_raw, config.upper_quantile, axis=0)
        variance = samples_raw.var(axis=0)

        for local_index, dataset_index in enumerate(indices):
            sample = dataset[dataset_index]
            sample_meta = sample["metadata"]
            future_dates = dataset.future_dates(dataset_index)
            for horizon_index, target_date in enumerate(future_dates):
                actual_value = float(actual[local_index, horizon_index, 0])
                pred_value = float(mean[local_index, horizon_index, 0])
                rows.append(
                    {
                        "date": pd.Timestamp(str(target_date)),
                        "serie": str(sample_meta[CROSS_KEY_COLUMN]),
                        "cross_key_id": str(sample_meta[CROSS_KEY_COLUMN]),
                        "account_currency_id": str(sample_meta[ACCOUNT_CURRENCY_ID_COLUMN]),
                        "tipo_serie": str(sample_meta[SERIES_TYPE_COLUMN]),
                        "cutoff": pd.Timestamp(str(sample_meta["cutoff"])),
                        "horizon_step": horizon_index + 1,
                        "isTrain": bool(is_train),
                        "actual_orig": actual_value,
                        "pred_orig": pred_value,
                        "lower_ci": float(lower[local_index, horizon_index, 0]),
                        "upper_ci": float(upper[local_index, horizon_index, 0]),
                        "bias2": float((pred_value - actual_value) ** 2),
                        "var_pred": float(variance[local_index, horizon_index, 0]),
                    }
                )
    model.eval()
    return pd.DataFrame(rows, columns=BACKTEST_COLUMNS)


def _mc_forward(
    model: GlobalForecastModel,
    model_inputs: Mapping[str, torch.Tensor],
    n_mc: int,
) -> np.ndarray:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(int(n_mc)):
            output = model(**model_inputs)
            prediction = output.get("y_pred")
            if not isinstance(prediction, torch.Tensor):
                raise KeyError("Model output must contain 'y_pred'")
            samples.append(prediction.detach().cpu().numpy())
    model.eval()
    return np.stack(samples, axis=0)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
