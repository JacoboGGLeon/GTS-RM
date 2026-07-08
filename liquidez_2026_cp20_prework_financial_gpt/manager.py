# manager.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

from tools import Tools, ChecklistMixin
from engineer import Engineer
from scientist import Scientist


class Manager(ChecklistMixin):
    """High-level orchestrator coordinating Engineer and Scientist."""

    def __init__(
        self,
        engineer: Engineer,
        scientist: Scientist,
        *,
        n_mc: int = 1_000,
        test_size: int = 25,
    ) -> None:
        super().__init__()
        if not engineer.config.series_columns:
            raise ValueError("Engineer contains no configured series columns.")

        self.tools = Tools()
        self.engineer = engineer
        self.scientist = scientist
        self.n_mc = n_mc
        self.test_size = test_size

        # Results containers
        self._warmup_results: Dict[str, Any] = {"df_regression": [], "df_regression_metrics": []}
        self._finetune_results: Dict[str, Any] = {"df_regression": [], "df_regression_metrics": []}

        # Stable, case-agnostic public contract for notebooks/exporters:
        #   self._backtest_results["df_regression"] -> consolidated long DataFrame
        #   self._backtest_results["df_regression_metrics"] -> consolidated metrics DataFrame
        #   self._backtest_results["by_series"] -> raw payloads keyed by serie name
        # Do not expose serie names as top-level keys; that makes the library
        # leak a specific use case and breaks reusable downstream contracts.
        self._backtest_results: Dict[str, Any] = {
            "df_regression": pd.DataFrame(),
            "df_regression_metrics": pd.DataFrame(),
            "by_series": {},
        }
        self._future_results: Dict[str, pd.DataFrame] = {}  # dict per serie -> forecast df (index=date)
        self._df_forecasts: Optional[pd.DataFrame] = None
        self._df_outliers: Optional[pd.DataFrame] = None
        self._series_status: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Output helper (supports dict / tuple / tensor model outputs)
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_y_pred(out: Any) -> torch.Tensor:
        if isinstance(out, dict):
            return out["y_pred"]
        if isinstance(out, tuple):
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def _train_test_split_idx(self, ds: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
        if len(ds) < 2:
            raise ValueError("Dataset needs at least two rows.")
        effective_test = min(int(self.test_size), max(1, len(ds) // 3))
        train_idx = ds.index[:-effective_test]
        test_idx = ds.index[-effective_test:]
        if len(train_idx) == 0:
            train_idx = ds.index[:1]
            test_idx = ds.index[1:]
        return train_idx, test_idx

    def _extract_xy(self, serie: str, *, subset: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        ds = self.engineer.datasets[serie]
        train_idx, test_idx = self._train_test_split_idx(ds)

        if subset == "train":
            sub = ds.loc[train_idx]
        elif subset == "test":
            sub = ds.loc[test_idx]
        else:
            raise ValueError("subset must be 'train' or 'test'")

        feat_cols = [c for c in sub.columns if c not in ("y", "y_trans")]
        X = sub[feat_cols].values.astype(np.float32)
        y_trans = sub["y_trans"].values.astype(np.float32)
        return X, y_trans

    def _extract_full_xy(self, serie: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        ds = self.engineer.datasets[serie]
        feat_cols = [c for c in ds.columns if c not in ("y", "y_trans")]
        X = ds[feat_cols].values.astype(np.float32)
        y_trans = ds["y_trans"].values.astype(np.float32)
        return X, y_trans, feat_cols

    def _evaluate_full_supervised(self, serie: str) -> Optional[pd.DataFrame]:
        """Evaluate a trained model on all rows with Optuna-controlled lags."""
        model = self.scientist.models[serie]
        cfg = self.scientist.best_params[serie]["cfg"]
        ds = self.engineer.datasets[serie]
        X_full, y_full, _ = self._extract_full_xy(serie)

        Xw, yw, meta = self.scientist._make_supervised_windows(X_full, y_full, cfg)
        if len(Xw) == 0:
            return None

        with torch.no_grad():
            out = model(torch.tensor(Xw, dtype=torch.float32, device=self.scientist.device))
            y_pred = self._unwrap_y_pred(out)
            preds_t = y_pred[:, -1].detach().cpu().numpy()

        target_positions = np.asarray(meta["target_positions"], dtype=int)
        qt = self.engineer.transforms[serie]
        pred_orig = Tools.invert_transform(preds_t, qt)
        actual_orig = Tools.invert_transform(yw[:, -1], qt)
        dates = ds.index[target_positions]

        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "serie": serie,
                "pred_orig": pred_orig,
                "actual_orig": actual_orig,
            }
        )

    # ---------------------------------------------------------------------
    # Warm-up for all series
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Warm-up (all series)")
    def _warmup_all(
        self,
        *,
        n_trials: int = 20,
        max_epochs: int = 30,
        batch: int = 32,
        show_progress: bool = False,
    ) -> None:
        self._warmup_results = {"df_regression": [], "df_regression_metrics": []}

        it = self.engineer.config.series_columns
        iterator = tqdm(it, desc="Warm-up", unit="series") if show_progress else it

        for serie in iterator:
            try:
                X, y_t = self._extract_xy(serie, subset="train")
                safe_batch = max(1, min(batch, max(1, int(0.9 * len(X)))))

                self.scientist.warm_up(
                    serie,
                    X,
                    y_t,
                    n_trials=n_trials,
                    max_epochs=max_epochs,
                    batch=safe_batch,
                )

                df_r = self._evaluate_full_supervised(serie)
                if df_r is not None and not df_r.empty:
                    met = self.tools.metrics_regression(
                        df_r["actual_orig"].values,
                        df_r["pred_orig"].values,
                    )
                    df_met = pd.DataFrame([met])
                    df_met["serie"] = serie

                    self._warmup_results["df_regression"].append(df_r)
                    self._warmup_results["df_regression_metrics"].append(df_met)
                self._series_status[serie] = {"stage": "warmup", "status": "ok"}
            except Exception as exc:
                self._series_status[serie] = {"stage": "warmup", "status": "failed", "reason": str(exc)}
                print(f"[WARN] Warm-up skipped for {serie}: {exc}")
                continue

        # Aggregate
        self._warmup_results["df_regression"] = (
            pd.concat(self._warmup_results["df_regression"], ignore_index=True)
            if self._warmup_results["df_regression"]
            else pd.DataFrame()
        )
        self._warmup_results["df_regression_metrics"] = (
            pd.concat(self._warmup_results["df_regression_metrics"], ignore_index=True)
            if self._warmup_results["df_regression_metrics"]
            else pd.DataFrame()
        )

    # ---------------------------------------------------------------------
    # Fine-tune for all series
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Fine-tune (all series)")
    def _finetune_all(
        self,
        *,
        epochs: int = 50,
        batch: int = 128,
        show_progress: bool = False,
    ) -> None:
        self._finetune_results = {"df_regression": [], "df_regression_metrics": []}

        it = self.engineer.config.series_columns
        iterator = tqdm(it, desc="Fine-tune", unit="series") if show_progress else it

        for serie in iterator:
            if serie not in self.scientist.best_params:
                print(f"[WARN] Fine-tune skipped for {serie}: no warm-up params.")
                continue
            try:
                X, y_t = self._extract_xy(serie, subset="train")
                qt = self.engineer.transforms[serie]
                safe_batch = max(1, min(batch, max(1, int(0.9 * len(X)))))

                self.scientist.fine_tune(
                    serie,
                    X,
                    y_t,
                    qt=qt,
                    epochs=epochs,
                    batch=safe_batch,
                )

                df_r = self._evaluate_full_supervised(serie)
                if df_r is not None and not df_r.empty:
                    met = self.tools.metrics_regression(
                        df_r["actual_orig"].values,
                        df_r["pred_orig"].values,
                    )
                    df_met = pd.DataFrame([met])
                    df_met["serie"] = serie

                    self._finetune_results["df_regression"].append(df_r)
                    self._finetune_results["df_regression_metrics"].append(df_met)
                self._series_status[serie] = {"stage": "finetune", "status": "ok"}
            except Exception as exc:
                self._series_status[serie] = {"stage": "finetune", "status": "failed", "reason": str(exc)}
                print(f"[WARN] Fine-tune skipped for {serie}: {exc}")
                continue

        self._finetune_results["df_regression"] = (
            pd.concat(self._finetune_results["df_regression"], ignore_index=True)
            if self._finetune_results["df_regression"]
            else pd.DataFrame()
        )
        self._finetune_results["df_regression_metrics"] = (
            pd.concat(self._finetune_results["df_regression_metrics"], ignore_index=True)
            if self._finetune_results["df_regression_metrics"]
            else pd.DataFrame()
        )

    # ---------------------------------------------------------------------
    # Back-test MC-Dropout (all series)
    # ---------------------------------------------------------------------
    @staticmethod
    def _empty_backtest_results() -> Dict[str, Any]:
        """Return the stable public backtest result contract used by notebooks."""
        return {
            "df_regression": pd.DataFrame(),
            "df_regression_metrics": pd.DataFrame(),
            "by_series": {},
        }

    @ChecklistMixin.track("Back-test MC-Dropout (all series)")
    def _run_backtest(self) -> None:
        by_series: Dict[str, Any] = {}
        regression_frames: List[pd.DataFrame] = []
        metric_frames: List[pd.DataFrame] = []

        for serie in tqdm(self.engineer.config.series_columns, desc="Back-test", unit="series"):
            if serie not in self.scientist.best_params:
                print(f"[WARN] Back-test skipped for {serie}: no trained model.")
                continue
            try:
                in_dim = int(self.scientist.best_params[serie]["input_dim"])
                self.scientist.load_model(serie, in_dim, self.scientist.model_filepath("outputs", serie, "ft"))

                ds = self.engineer.datasets[serie]
                X_full, y_full, _ = self._extract_full_xy(serie)
                qt = self.engineer.transforms[serie]

                bt = self.scientist.backtest_mc(
                    serie,
                    X_full=X_full,
                    y_trans=y_full,
                    qt=qt,
                    n_mc=self.n_mc,
                )

                df_r = bt.get("df_regression")
                if isinstance(df_r, pd.DataFrame) and not df_r.empty:
                    df_r = df_r.copy()

                    if "target_pos" in df_r.columns:
                        pos = df_r["target_pos"].astype(int).to_numpy()
                        df_r["date"] = pd.to_datetime(ds.index[pos])
                    elif "date" not in df_r.columns:
                        df_r["date"] = pd.to_datetime(ds.index[: len(df_r)])

                    df_r["serie"] = serie

                    if "actual_orig" not in df_r.columns and "target_pos" in df_r.columns:
                        pos = df_r["target_pos"].astype(int).to_numpy()
                        df_r["actual_orig"] = Tools.invert_transform(ds["y_trans"].values[pos], qt)

                    first_cols = [
                        "date", "serie", "idx", "target_pos", "isTrain",
                        "actual_orig", "pred_orig", "lower_ci", "upper_ci",
                        "bias2", "var_pred",
                    ]
                    ordered = [c for c in first_cols if c in df_r.columns]
                    ordered += [c for c in df_r.columns if c not in ordered]
                    df_r = df_r[ordered]

                    bt["df_regression"] = df_r
                    regression_frames.append(df_r)

                df_m = bt.get("df_regression_metrics")
                if isinstance(df_m, pd.DataFrame) and not df_m.empty:
                    df_m = df_m.copy()
                    df_m["serie"] = serie
                    bt["df_regression_metrics"] = df_m
                    metric_frames.append(df_m)

                by_series[serie] = bt
                self._series_status[serie] = {"stage": "backtest", "status": "ok"}
            except Exception as exc:
                self._series_status[serie] = {"stage": "backtest", "status": "failed", "reason": str(exc)}
                print(f"[WARN] Back-test skipped for {serie}: {exc}")
                continue

        self._backtest_results = {
            "df_regression": (
                pd.concat(regression_frames, ignore_index=True)
                if regression_frames
                else pd.DataFrame()
            ),
            "df_regression_metrics": (
                pd.concat(metric_frames, ignore_index=True)
                if metric_frames
                else pd.DataFrame()
            ),
            "by_series": by_series,
        }

    # ---------------------------------------------------------------------
    # Forecast MC-Dropout (all series)
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Forecast MC-Dropout (all series)")
    def _run_forecast(self, *, start_date: str, end_date: str) -> None:
        self._future_results = {}
        self._df_forecasts = None

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        if end_ts < start_ts:
            raise ValueError("end_date must be greater than or equal to start_date.")

        cal = self.engineer.calendar.copy().replace({False: 0, True: 1})
        cal = cal.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        for serie in self.engineer.config.series_columns:
            if serie not in self.scientist.best_params:
                print(f"[WARN] Forecast skipped for {serie}: no trained model.")
                continue
            try:
                in_dim = int(self.scientist.best_params[serie]["input_dim"])
                self.scientist.load_model(serie, in_dim, self.scientist.model_filepath("outputs", serie, "ft"))

                qt = self.engineer.transforms[serie]
                ds = self.engineer.datasets[serie]
                X_hist, y_hist, feat_cols = self._extract_full_xy(serie)

                last_hist_date = pd.to_datetime(ds.index.max())
                first_forecast_date = last_hist_date + pd.Timedelta(days=1)
                if end_ts < first_forecast_date:
                    continue

                all_future_dates = pd.date_range(first_forecast_date, end_ts, freq="D")
                cal_needed = cal.reindex(all_future_dates).fillna(0)
                cal_needed = cal_needed.reindex(columns=feat_cols, fill_value=0)
                X_future_all = cal_needed.values.astype(np.float32)

                res = self.scientist.recursive_forecast_mc(
                    serie,
                    X_history=X_hist,
                    y_history=y_hist,
                    X_future=X_future_all,
                    qt=qt,
                    n_mc=self.n_mc,
                )

                keep = all_future_dates >= start_ts
                if not keep.any():
                    continue

                df_fc = pd.DataFrame(
                    {
                        "pred_orig": res["pred_orig"][keep],
                        "lower_ci": res["lower_ci"][keep],
                        "upper_ci": res["upper_ci"][keep],
                    },
                    index=pd.to_datetime(all_future_dates[keep]),
                )
                df_fc.index.name = "date"

                self._future_results[serie] = df_fc
                self._series_status[serie] = {"stage": "forecast", "status": "ok"}
            except Exception as exc:
                self._series_status[serie] = {"stage": "forecast", "status": "failed", "reason": str(exc)}
                print(f"[WARN] Forecast skipped for {serie}: {exc}")
                continue

        if self._future_results:
            tmp = []
            for serie, df_fc in self._future_results.items():
                df_tmp = df_fc.copy()
                df_tmp["serie"] = serie
                df_tmp = df_tmp.reset_index()
                tmp.append(df_tmp)
            self._df_forecasts = pd.concat(tmp, ignore_index=True)
        else:
            self._df_forecasts = pd.DataFrame()

        self._df_outliers = None
        try:
            self._df_outliers = self.engineer.outliers
        except Exception:
            self._df_outliers = None

    @property
    def series_status(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._series_status, orient="index")

    # ---------------------------------------------------------------------
    # Visualisation (NON-NEGOTIABLE: use existing Tools plots exactly)
    # ---------------------------------------------------------------------
    @ChecklistMixin.track("Visualización")
    def visualise(self, *, bt_start: str, bt_end: str, fc_start: str, fc_end: str) -> None:
        """
        No negociable: generar las 3 figuras por serie como antes:
        1) Backtest
        2) Forecast (+ outliers teóricos)
        3) Backtest + Forecast (+ outliers)
        """
        if not self._backtest_results:
            print("[Manager] No backtest results found. Run _run_backtest() first.")
            return
        if not self._future_results:
            print("[Manager] No forecast results found. Run _run_forecast() first.")
            return

        # Backtest results already follow the public consolidated contract.
        df_backtest = self._backtest_results.get("df_regression", pd.DataFrame())
        if not isinstance(df_backtest, pd.DataFrame) or df_backtest.empty:
            print("[Manager] Backtest has no df_regression frame to plot.")
            return
        backtest_results_global = {"df_regression": df_backtest.copy()}

        # dict_series_clean requerido por Tools (usa series limpias / datasets)
        dict_series_clean = self.engineer.datasets

        for serie in self.engineer.config.series_columns:
            if serie not in self._future_results:
                # si no hay forecast para esa serie, saltamos (o solo plot backtest)
                continue

            # 1) Backtest (solo histórico)
            self.tools.plot_backtest_for_serie(
                backtest_results=backtest_results_global,
                serie=serie,
            )

            # 2) Forecast (solo futuro + outliers teóricos)
            self.tools.plot_forecast_with_outliers(
                serie_name=serie,
                future_results=self._future_results,
                dict_series_clean=dict_series_clean,
            )

            # 3) Backtest + Forecast (combinado)
            self.tools.plot_backtest_forecast_with_outliers_for_serie(
                serie_name=serie,
                backtest_results=backtest_results_global,
                future_results=self._future_results,
                dict_series_clean=dict_series_clean,
                build_hierarchical_outliers=self.tools.build_hierarchical_outliers,  # API compatibility
                bt_start=bt_start,
                bt_end=bt_end,
                fc_start=fc_start,
                fc_end=fc_end,
            )