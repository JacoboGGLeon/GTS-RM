"""tools.py – Renovated utilities module
======================================
This file contains the **Tools** helper class plus a lightweight
ChecklistMixin.  The class groups together **data‑engineering helpers**,
**visual helpers** and **metric utilities** that are reused across the
whole library (Engineer → Scientist → Manager layers).

Changes introduced in the *renovated* version
---------------------------------------------
1. All docstrings and comments are now in **English**.
2. Each public function/method has a concise docstring that follows the
   *Google style* guide.
3. Constants and attributes that are useful to external users are
   explicitly declared at class level so they are discoverable with
   dir(Tools) or IDE autocompletion.
4. The file is notebook‑friendly: the first line uses the %%writefile
   cell‑magic so that you can execute the cell in a Jupyter notebook and
   have the module written to disk ready for import tools.
"""

# ================================================================
# Imports
# ================================================================
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.base import TransformerMixin
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
from dataclasses import dataclass, field


# ================================================================
# ChecklistMixin
# ================================================================
class ChecklistMixin:
    """Small mixin to keep a sequential log of the executed steps.

    Every method decorated with :py:meth:track will append a ✓ or ✗ plus
    the *action* name to the instance attribute :pyattr:checklist.
    This makes it trivial to print a full processing log at the end of a
    pipeline run.
    """

    def __init__(self) -> None:  # noqa: D401 – simple docstring is OK
        #: Keeps a textual log of the executed steps.
        self.checklist: List[str] = []

    @staticmethod
    def track(action: str):
        """Decorator that records success/failure of *action* in the checklist."""

        def decorator(func):
            def wrapper(self, *args, **kwargs):  # type: ignore[override]
                try:
                    out = func(self, *args, **kwargs)
                    self.checklist.append(f"✔️ {action}")
                    return out
                except Exception as exc:  # pylint: disable=broad-except
                    self.checklist.append(f"❌ {action}: {exc}")
                    raise

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator



class TargetTransformPipeline:
    """Composable target transformer: optional signed-log + sklearn scaler.

    The pipeline exposes the same fit/transform/inverse_transform API used by
    sklearn transformers, so the rest of the project can keep a single contract.
    Input/output values are always 2-D internally and flattened by callers when
    needed.
    """

    def __init__(self, scaler: TransformerMixin, *, target_transform: str = "identity") -> None:
        self.scaler = scaler
        self.target_transform = (target_transform or "identity").strip().lower()
        if self.target_transform not in {"identity", "none", "signed_log", "signed-log", "signedlog"}:
            raise ValueError(
                "target_transform must be one of: identity, none, signed_log. "
                f"Got {target_transform!r}."
            )
        self.feature_names_in_ = None

    @staticmethod
    def _to_2d(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)
        arr = arr.astype(np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _signed_log(X: np.ndarray) -> np.ndarray:
        return np.sign(X) * np.log1p(np.abs(X))

    @staticmethod
    def _inverse_signed_log(X: np.ndarray) -> np.ndarray:
        # Clip to avoid exp overflow on pathological predictions.
        X = np.clip(X, -80.0, 80.0)
        return np.sign(X) * np.expm1(np.abs(X))

    def _forward_target(self, X: np.ndarray) -> np.ndarray:
        if self.target_transform in {"signed_log", "signed-log", "signedlog"}:
            return self._signed_log(X)
        return X

    def _inverse_target(self, X: np.ndarray) -> np.ndarray:
        if self.target_transform in {"signed_log", "signed-log", "signedlog"}:
            return self._inverse_signed_log(X)
        return X

    def fit(self, X: Any, y: Any = None) -> "TargetTransformPipeline":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
        X2 = self._to_2d(X)
        self.scaler.fit(self._forward_target(X2))
        return self

    def transform(self, X: Any) -> np.ndarray:
        X2 = self._to_2d(X)
        return self.scaler.transform(self._forward_target(X2))

    def inverse_transform(self, X: Any) -> np.ndarray:
        X2 = self._to_2d(X)
        inv_scaled = self.scaler.inverse_transform(X2)
        return self._inverse_target(inv_scaled)

@dataclass(frozen=True)
class SystemConfig:
    series_columns: List[str]
    transformer_cls: Type[TransformerMixin] = QuantileTransformer
    transformer_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'output_distribution': 'normal',
        'n_quantiles': 1000,
        'random_state': 42,
        })
    # Target preprocessing. Use 'signed_log' for sparse/signed amount series.
    target_transform: str = 'identity'
    # Missing target policy: 'drop', 'zero', 'ffill_zero', or 'interpolate_zero'.
    missing_policy: str = 'drop'
    min_train_points: int = 2



# ================================================================
# Tools class
# ================================================================
class Tools(ChecklistMixin):
    """Collection of data‑engineering helpers and unified visual utilities.

    The class is *stateless*: every method is either `@staticmethod or
    relies only on its input arguments.  A few constants such as
    :pyattr:EPS_MAD or the z‑band colour map are exposed at class level
    so they can be reused elsewhere.
    """

    # ------------------------------------------------------------------
    # Public constants (discoverable by external code)
    # ------------------------------------------------------------------
    EPS_MAD: float = 1e-9
    """Small epsilon added to denominators to avoid division by zero."""

    # Z‑band colour palette (beyond ±3 σ → purple).
    colors_z: Dict[int, str] = {
        4: "purple",
        3: "red",
        2: "yellow",
        1: "white",
        0: "white",
        -1: "white",
        -2: "yellow",
        -3: "red",
        -4: "purple",
    }

    # =============================================================
    # 🔹  Weight initialisers (PyTorch) – may be used by *Scientist*
    # =============================================================
    @staticmethod
    def xavier_init_weights(m: nn.Module) -> None:  # noqa: D401
        """Apply Xavier/Glorot normal initialisation to *Linear* layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def sparse_init(m: nn.Module, sparsity: float = 0.1) -> None:  # noqa: D401
        """Initialise *Linear* weights with a given *sparsity* (PyTorch)."""
        if isinstance(m, nn.Linear):
            nn.init.sparse_(m.weight, sparsity=sparsity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def orthogonal_init(m: nn.Module) -> None:  # noqa: D401
        """Orthogonal initialisation with *ReLU* gain for *Linear* layers."""
        if isinstance(m, nn.Linear):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # =============================================================
    # 🔹  Z‑band helpers (common to all visual plots)
    # =============================================================
    @staticmethod
    def _compute_z_stats(vals: np.ndarray) -> Tuple[Dict[int, float], float, float]:
        """Return z‑band boundaries plus global `ymin/ymax.

        Parameters
        ----------
        vals
            1‑D array with the numerical data to analyse.

        Returns
        -------
        tuple
            *bounds*, *ymin*, *ymax* where *bounds* maps `{-3..3} → value.
        """
        mean_v, std_v = vals.mean(), vals.std()
        bounds = {
            3: mean_v + 3 * std_v,
            2: mean_v + 2 * std_v,
            1: mean_v + 1 * std_v,
            0: mean_v,
            -1: mean_v - 1 * std_v,
            -2: mean_v - 2 * std_v,
            -3: mean_v - 3 * std_v,
        }
        return bounds, min(vals.min(), bounds[-3]), max(vals.max(), bounds[3])

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_z_bands(
        ax,
        bounds: Dict[int, float],
        ymin: float,
        ymax: float,
        colors: Dict[int, str],
    ) -> None:
        """Paint seven coloured background bands and dashed σ‑lines."""

        intervals = [
            (ymin, bounds[-3], colors[-4]),
            (bounds[-3], bounds[-2], colors[-3]),
            (bounds[-2], bounds[-1], colors[-2]),
            (bounds[-1], bounds[1], colors[0]),
            (bounds[1], bounds[2], colors[2]),
            (bounds[2], bounds[3], colors[3]),
            (bounds[3], ymax, colors[4]),
        ]
        for lo, hi, col in intervals:
            ax.axhspan(lo, hi, color=col, alpha=0.20)

        for z in (-3, -2, -1, 1, 2, 3):
            ax.axhline(bounds[z], ls="--", lw=1, color="k")

    # =============================================================
    # 🔸  Outlier encoding helper (shared by several public plots)
    # =============================================================
    @staticmethod
    def _flag_from_bounds(
        s: pd.Series,
        bounds: Dict[int, float],
        thresholds: Tuple[int, ...] = (3, 2, 1),
    ) -> pd.Series:
        """Return a *flag* Series encoding the outlier level of each point."""
        mean_v = bounds[0]
        std_v = (bounds[3] - mean_v) / 3.0 + Tools.EPS_MAD
        z = (s - mean_v) / std_v

        def encode(zv: float) -> int:
            az = abs(zv)
            if az > 3:
                lvl = 4
            elif az >= 3:
                lvl = 3
            elif az >= 2:
                lvl = 2
            elif az >= 1:
                lvl = 1
            else:
                lvl = 0
            return int(np.sign(zv) * lvl)

        return z.apply(encode).astype(int)

    # =============================================================
    # 🔹  Data‑cleaning & dataset construction helpers
    # =============================================================
    @ChecklistMixin.track("Clean data")
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates/NaNs and cast every column to *float*."""
        return df.drop_duplicates().dropna().astype(float)

    @ChecklistMixin.track("Detect outliers (z‑score)")
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Return indexes whose |z‑score| ≥ *threshold* (classical z‑score)."""
        vals = df["total_amount"].values
        mean_v = vals.mean()
        std_v = vals.std() + self.EPS_MAD
        z = np.abs((vals - mean_v) / std_v)
        idx = np.where(z >= threshold)[0]
        return {"indexes": {df.index[i]: int(np.sign(vals[i] - mean_v)) for i in idx}}

    @ChecklistMixin.track("Hierarchical outliers build")
    def build_hierarchical_outliers(
        self,
        series_dict: Mapping[str, pd.DataFrame],
        *,
        thresholds: Sequence[int] = (3, 2, 1),
        suffix: str = "_outlier",
    ) -> pd.DataFrame:
        """Iteratively detect outliers at thresholds 3→2→1 σ for each series."""
        base = pd.DataFrame(index=next(iter(series_dict.values())).index)
        out = pd.concat(
            [base, pd.DataFrame(0, index=base.index, columns=[f"{n}{suffix}" for n in series_dict])],
            axis=1,
        )
        for name, df in series_dict.items():
            col = f"{name}{suffix}"
            tmp = df.copy()
            for thr in sorted(thresholds, reverse=True):
                res = self.detect_outliers(tmp, threshold=thr)["indexes"]
                for dt, sign in res.items():
                    if out.at[dt, col] == 0:
                        out.at[dt, col] = sign * thr
                tmp = tmp.drop(index=list(res.keys()), errors="ignore")
        return out

    def make_datasets(
        self,
        df_series: pd.DataFrame,
        df_features: pd.DataFrame,
        lags: int = 0,
    ) -> pd.DataFrame:
        """Merge series + features and add *y* and its lags."""
        df = df_series.join(df_features, how="inner").copy()
        df["y"] = df["total_amount"]
        for i in range(1, lags + 1):
            df[f"y_lag_{i}"] = df["y"].shift(i)
        return df

    # -------------------------------------------------------------
    # Sliding window transformation (usable by external scientists)
    # -------------------------------------------------------------
    @staticmethod
    def sliding_window_transform(
        X: np.ndarray,
        y: np.ndarray,
        window: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sequences into overlapping windows for RNN input."""
        X_new: List[np.ndarray] = []
        y_new: List[np.ndarray] = []
        for i in range(len(X) - window + 1):
            X_new.append(X[i : i + window])
            y_new.append(y[i : i + window])
        return np.array(X_new), np.array(y_new)

    # -------------------------------------------------------------
    # Transform helpers
    # -------------------------------------------------------------
    @staticmethod
    def make_target_transformer(
        scaler_cls: Type[TransformerMixin],
        scaler_kwargs: Optional[Dict[str, Any]] = None,
        *,
        target_transform: str = "identity",
    ) -> TargetTransformPipeline:
        """Build a reversible target transformer used by Engineer/Manager."""
        return TargetTransformPipeline(
            scaler_cls(**(scaler_kwargs or {})),
            target_transform=target_transform,
        )

    @staticmethod
    def invert_transform(arr_1d: np.ndarray, qt: Any) -> np.ndarray:
        """Inverse any fitted target transformer/scaler safely."""
        arr = np.asarray(arr_1d, dtype=np.float64).reshape(-1, 1)
        try:
            if getattr(qt, "feature_names_in_", None) is not None:
                arr_in = pd.DataFrame(arr, columns=qt.feature_names_in_)
                out = qt.inverse_transform(arr_in)
            else:
                out = qt.inverse_transform(arr)
        except Exception:
            out = qt.inverse_transform(arr)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        return np.nan_to_num(out, nan=0.0, posinf=np.finfo(np.float64).max / 10, neginf=np.finfo(np.float64).min / 10)

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------
    @staticmethod
    def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return robust regression metrics without crashing on sparse/constant series."""
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return {k: float("nan") for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WMAPE", "MASE", "MedAE", "MedAPE", "EVS", "R2"]}

        eps = 1e-9
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nz = np.abs(y_true) > eps
        mape = float(np.mean(np.abs(err[nz] / y_true[nz])) * 100.0) if np.any(nz) else float("nan")
        smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)
        wmape = float(np.sum(np.abs(err)) / (np.sum(np.abs(y_true)) + eps) * 100.0)
        denom = np.mean(np.abs(np.diff(y_true))) if len(y_true) > 1 else np.nan
        mase = float(mae / denom) if np.isfinite(denom) and denom > eps else float("nan")
        medae = float(np.median(np.abs(err)))
        medape = float(np.median(np.abs(err[nz] / y_true[nz])) * 100.0) if np.any(nz) else float("nan")
        try:
            evs = float(explained_variance_score(y_true, y_pred)) if len(y_true) >= 2 and np.std(y_true) > eps else float("nan")
        except Exception:
            evs = float("nan")
        try:
            r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.std(y_true) > eps else float("nan")
        except Exception:
            r2 = float("nan")
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WMAPE": wmape, "MASE": mase, "MedAE": medae, "MedAPE": medape, "EVS": evs, "R2": r2}

    # =============================================================
    # 🔹  Unified visualisations
    # =============================================================
    # NOTE: the public plotting functions purposefully *do not* return
    # a figure.  They immediately render the plot via plt.show() to
    # keep the notebook‑friendly behaviour.

    @ChecklistMixin.track("Plot series outliers")
    def plot_series_outliers(
        self,
        dict_datasets: Dict[str, pd.DataFrame],
        df_outliers: pd.DataFrame,  # kept for API compatibility
        series_name: str,
    ) -> None:
        """Plot one time‑series with coloured z‑bands and outlier markers."""
        if series_name not in dict_datasets:
            raise ValueError(f"Series '{series_name}' not found in dict_datasets.")

        df = dict_datasets[series_name].copy()
        if "total_amount" not in df.columns:
            df["total_amount"] = df["y"]

        vals = df["total_amount"].values
        if bounds is None:
            history = dict_series_clean.get(serie_name)
            reference = history["total_amount"].to_numpy() if history is not None else vals
            bounds, ymin, ymax = self._compute_z_stats(reference)
        else:
            _, ymin, ymax = self._compute_z_stats(np.asarray(list(bounds.values()), dtype=float))
            ymin = min(ymin, float(np.nanmin(vals)))
            ymax = max(ymax, float(np.nanmax(vals)))
        flags = self._flag_from_bounds(df["total_amount"], bounds)

        # Figure layout ------------------------------------------------
        fig, (ax_ts, ax_hist) = plt.subplots(  # noqa: F841 – kept for clarity
            1,
            2,
            figsize=(20, 6),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        self._draw_z_bands(ax_ts, bounds, ymin, ymax, self.colors_z)
        self._draw_z_bands(ax_hist, bounds, ymin, ymax, self.colors_z)

        # Time‑series --------------------------------------------------
        (line_real,) = ax_ts.plot(df.index, vals, "b-o", ms=4, label="Actual")
        for dt, flag in flags.items():
            if flag != 0:
                ax_ts.scatter(
                    dt,
                    df.at[dt, "total_amount"],
                    s=100,                                        # marker size
                    facecolors=self.colors_z[int(flag)],          # filled color
                    edgecolors="black",                           # black outline
                    linewidths=1.5,                               # outline width
                    zorder=5                                      # draw on top
                )


        ax_ts.set_ylim(ymin, ymax)
        ax_ts.set_title(f"{series_name}: Outliers & Distribution")
        ax_ts.tick_params(axis="x", rotation=45)
        ax_ts.grid(True)

        # Histogram ----------------------------------------------------
        ax_hist.hist(vals, bins=30, orientation="horizontal", color="blue", alpha=0.6)
        ax_hist.set_ylim(ymin, ymax)
        ax_hist.set_xlabel("Frequency")
        ax_hist.set_title("Distribution")
        ax_hist.grid(True)

        # Legend -------------------------------------------------------
        patches = [line_real] + [
            mpatches.Patch(color=self.colors_z[z], alpha=0.2, label=f"Z {'+' if z>0 else ''}{z}")
            for z in (3, 2, 1, 0, -1, -2, -3)
        ]
        ax_ts.legend(handles=patches, loc="upper left")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    @ChecklistMixin.track("Plot annual comparison")
    def plot_annual_comparison(
        self,
        dict_datasets: Dict[str, pd.DataFrame],
        df_outliers: pd.DataFrame,  # kept for API compatibility
        series_name: str,
        years: List[int] | Tuple[int, ...] = (2022, 2023, 2024, 2025),
    ) -> None:
        """Overlay multiple years on top of each other to spot seasonality."""
        if series_name not in dict_datasets:
            raise ValueError(f"Series '{series_name}' not found in dict_datasets.")

        df_src = dict_datasets[series_name].copy()
        if "total_amount" not in df_src.columns:
            df_src["total_amount"] = df_src["y"]

        bounds, ymin, ymax = self._compute_z_stats(df_src["total_amount"].values)
        fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(25, 6), gridspec_kw={"width_ratios": [3, 1]})
        self._draw_z_bands(ax, bounds, ymin, ymax, self.colors_z)
        self._draw_z_bands(ax_hist, bounds, ymin, ymax, self.colors_z)

        palette = ["y", "b", "r", "g", "m", "c"]
        for i, y in enumerate(years):
            df_y = self.compare_annual_series(df_src, y)
            seg = df_y[f"saldo_{y}"]
            ax.plot(seg.index, seg.values, f"{palette[i % len(palette)]}-o", ms=4, label=str(y))

            mask_y = df_src.index.year == y
            flags_y = self._flag_from_bounds(df_src.loc[mask_y, "total_amount"], bounds)
            for dt, flag in flags_y.items():
                if flag != 0:
                    ax.scatter(
                        dt.replace(year=1920),
                        seg.at[dt.replace(year=1920)],
                        s=100,
                        facecolors=self.colors_z[int(flag)],
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=5,
                    )


        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{series_name}: Annual Overlay")
        ax.set_ylabel("Total Amount")

        # Combined histogram -----------------------------------------
        all_vals = np.concatenate([self.compare_annual_series(df_src, y)[f"saldo_{y}"].values for y in years])
        ax_hist.hist(all_vals, bins=30, orientation="horizontal", color="blue", alpha=0.6)
        ax_hist.set_ylim(ymin, ymax)
        ax_hist.grid(True)
        ax_hist.set_title("Distribution")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    @ChecklistMixin.track("Plot series backtest")
    def plot_backtest_for_serie(
        self,
        backtest_results: Dict[str, Any],
        serie: str,
        figsize: Tuple[int, int] = (20, 6),
        bounds: Dict[int, float] | None = None,
    ) -> None:
        """Visualise backtest predictions vs. actuals for a single series."""
        df_r = backtest_results["df_regression"].query("serie == @serie").copy()
        if df_r.empty:
            print(f"[{serie}] No backtest data available.")
            return

        df_r["date"] = pd.to_datetime(df_r["date"])
        vals_real = df_r["actual_orig"].values
        vals_pred = df_r["pred_orig"].values
        if bounds is None:
            train_vals = df_r.loc[df_r.get("isTrain", True).astype(bool), "actual_orig"].to_numpy() if "isTrain" in df_r.columns else vals_real
            bounds, ymin, ymax = self._compute_z_stats(train_vals)
        else:
            _, ymin, ymax = self._compute_z_stats(np.asarray(list(bounds.values()), dtype=float))
            ymin = min(ymin, float(np.nanmin(np.concatenate([vals_real, vals_pred]))))
            ymax = max(ymax, float(np.nanmax(np.concatenate([vals_real, vals_pred]))))
        flags_pred = self._flag_from_bounds(pd.Series(vals_pred, index=df_r["date"]), bounds)
        mask_theo = flags_pred != 0

        fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})
        self._draw_z_bands(ax_ts, bounds, ymin, ymax, self.colors_z)
        self._draw_z_bands(ax_hist, bounds, ymin, ymax, self.colors_z)

        ax_ts.plot(df_r["date"], vals_real, "g-", label="Actual")
        ax_ts.plot(df_r["date"], vals_pred, "b-", label="Prediction")
        if {"lower_ci", "upper_ci"}.issubset(df_r.columns):
            ax_ts.fill_between(df_r["date"], df_r["lower_ci"], df_r["upper_ci"], color="blue", alpha=0.3)

        ax_ts.scatter(flags_pred.index[mask_theo], vals_pred[mask_theo], marker="o", s=100, facecolors="none", edgecolors="red", label="Outlier")
        if "isTrain" in df_r.columns:
            test_mask = ~df_r["isTrain"]
            test_dates = df_r.loc[test_mask, "date"]
            if not test_dates.empty:
                ax_ts.axvline(test_dates.min(), color="gray", lw=5, ls="--")

        ax_ts.set_ylim(ymin, ymax)
        ax_ts.margins(y=0.30)
        ax_ts.set_title(f"Backtest – {serie}")
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax_ts.tick_params(axis="x", rotation=45)
        ax_ts.grid(True)
        ax_ts.legend(loc="upper right")

        # Histogram ----------------------------------------------------
        ax_hist.hist(vals_real, bins=30, orientation="horizontal", color="green", alpha=0.6, label="Actual")
        ax_hist.hist(vals_pred, bins=30, orientation="horizontal", color="blue", alpha=0.3, label="Prediction")
        ax_hist.set_ylim(ymin, ymax)
        ax_hist.margins(y=0.30)
        ax_hist.set_xlabel("Frequency")
        ax_hist.set_title("Distribution")
        ax_hist.grid(True)
        ax_hist.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    @ChecklistMixin.track("Plot forecast")
    def plot_forecast_with_outliers(
        self,
        serie_name: str,
        future_results: Dict[str, pd.DataFrame],
        dict_series_clean: Dict[str, pd.DataFrame],
        *,
        thresholds: Sequence[int] = (3, 2, 1),
        figsize: Tuple[int, int] = (25, 6),
        bounds: Dict[int, float] | None = None,
    ) -> None:
        """Plot Monte‑Carlo forecast and highlight theoretical outliers."""
        df_fc = future_results[serie_name].copy()
        vals = df_fc["pred_orig"].values
        bounds, ymin, ymax = self._compute_z_stats(vals)
        flags_fc = self._flag_from_bounds(df_fc["pred_orig"], bounds)
        mask_theo = flags_fc != 0

        fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})
        self._draw_z_bands(ax_ts, bounds, ymin, ymax, self.colors_z)
        self._draw_z_bands(ax_hist, bounds, ymin, ymax, self.colors_z)

        ax_ts.plot(df_fc.index, vals, "-o", color="blue", label="Prediction")
        ax_ts.fill_between(df_fc.index, df_fc["lower_ci"], df_fc["upper_ci"], color="blue", alpha=0.3)
        ax_ts.scatter(df_fc.index[mask_theo], vals[mask_theo], marker="o", s=100, facecolors="none", edgecolors="red", linewidths=2, zorder=10, label="Outlier")

        ax_ts.set_ylim(ymin, ymax)
        ax_ts.margins(y=0.30)
        ax_ts.grid(True)
        ax_ts.set_title(f"Forecast – {serie_name}")
        ax_ts.tick_params(axis="x", rotation=45)
        ax_ts.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax_ts.legend(loc="upper right")

        # Histogram ----------------------------------------------------
        ax_hist.hist(vals, bins=30, orientation="horizontal", color="blue", alpha=0.6, label="Prediction")
        ax_hist.set_ylim(ymin, ymax)
        ax_hist.margins(y=0.30)
        ax_hist.set_xlabel("Frequency")
        ax_hist.set_title("Distribution")
        ax_hist.legend(loc="upper right")
        ax_hist.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    @ChecklistMixin.track("Plot backtest+forecast")
    def plot_backtest_forecast_with_outliers_for_serie(
        self,
        serie_name: str,
        backtest_results: Dict[str, Any],
        future_results: Dict[str, pd.DataFrame],
        dict_series_clean: Dict[str, pd.DataFrame],
        build_hierarchical_outliers,  # kept for API compatibility
        *,
        bt_start: str,
        bt_end: str,
        fc_start: str,
        fc_end: str,
        figsize: Tuple[int, int] = (25, 6),
        bounds: Dict[int, float] | None = None,
    ) -> None:
        """Combined visual: historical backtest + future forecast."""
        df_r = backtest_results["df_regression"].query(
            "(serie == @serie_name) and (date >= @bt_start) and (date <= @bt_end)"
        ).copy()
        if df_r.empty:
            print(f"[{serie_name}] No backtest data in selected range.")
            return

        df_r["date"] = pd.to_datetime(df_r["date"])
        df_r = df_r.set_index("date").sort_index()
        df_r["y_real"] = df_r["actual_orig"]
        df_r["y_pred_bt"] = df_r["pred_orig"]

        df_fc = future_results[serie_name].loc[fc_start:fc_end].copy()
        df_fc["y_pred_fc"] = df_fc["pred_orig"]
        df_fc["y_real"] = np.nan

        frames = []
        if df_r is not None and not df_r.empty:
            # elimina columnas 100% NA (por si algún merge/outlier dejó basura)
            frames.append(df_r.dropna(axis=1, how="all"))

        if df_fc is not None and not df_fc.empty:
            # en forecast "y_real" suele ser NaN; la quitamos para no afectar dtypes
            frames.append(df_fc.dropna(axis=1, how="all"))

        if not frames:
            print(f"[{serie_name}] Nothing to plot: empty backtest and forecast.")
            return

        df_plot = pd.concat(frames, axis=0, sort=False).sort_index()
        vals_all = np.concatenate([
            df_r["y_real"].dropna().values,
            df_r["y_pred_bt"].values,
            df_fc["y_pred_fc"].values,
        ])
        if bounds is None:
            train_reference = df_r.loc[df_r.get("isTrain", True).astype(bool), "y_real"].dropna().to_numpy() if "isTrain" in df_r.columns else df_r["y_real"].dropna().to_numpy()
            bounds, ymin, ymax = self._compute_z_stats(train_reference)
        else:
            _, ymin, ymax = self._compute_z_stats(np.asarray(list(bounds.values()), dtype=float))
            ymin = min(ymin, float(np.nanmin(vals_all)))
            ymax = max(ymax, float(np.nanmax(vals_all)))

        flags_fc = self._flag_from_bounds(df_fc["y_pred_fc"], bounds)
        mask_theo_fc = flags_fc != 0
        flags_bt = self._flag_from_bounds(df_r["y_pred_bt"], bounds)
        mask_theo_bt = flags_bt != 0

        fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})
        self._draw_z_bands(ax_ts, bounds, ymin, ymax, self.colors_z)
        self._draw_z_bands(ax_hist, bounds, ymin, ymax, self.colors_z)

        mask_bt_range = (df_plot.index >= pd.to_datetime(bt_start)) & (df_plot.index <= pd.to_datetime(bt_end))
        ax_ts.plot(df_plot.index[mask_bt_range], df_plot.loc[mask_bt_range, "y_real"], "-o", color="green", label="Actual (BT)")
        ax_ts.plot(df_plot.index[mask_bt_range], df_plot.loc[mask_bt_range, "y_pred_bt"], "-o", color="blue", label="Pred (BT)")
        if {"lower_ci", "upper_ci"}.issubset(df_r.columns):
            ax_ts.fill_between(df_r.index, df_r["lower_ci"], df_r["upper_ci"], color="blue", alpha=0.3)

        ax_ts.plot(df_fc.index, df_fc["y_pred_fc"], "-o", color="red", label="Pred (FC)")
        if {"lower_ci", "upper_ci"}.issubset(df_fc.columns):
            ax_ts.fill_between(df_fc.index, df_fc["lower_ci"], df_fc["upper_ci"], color="red", alpha=0.3)

        ax_ts.scatter(df_r.index[mask_theo_bt], df_r.loc[mask_theo_bt, "y_pred_bt"], marker="o", s=100, facecolors="none", edgecolors="red", linewidths=2, zorder=10, label="Outlier BT")
        ax_ts.scatter(df_fc.index[mask_theo_fc], df_fc.loc[mask_theo_fc, "y_pred_fc"], marker="o", s=100, facecolors="none", edgecolors="red", linewidths=2, zorder=10, label="Outlier FC")

        ax_ts.axvline(pd.to_datetime(bt_end), color="gray", lw=5, ls="--")
        ax_ts.axvline(df_fc.index.min(), color="gray", lw=5, ls="--")

        # if "isTrain" in df_plot.columns:
        #     test_mask = ~df_plot["isTrain"]
        #     test_dates = df_plot.loc[test_mask].index
        #     if not test_dates.empty:
        #         ax_ts.axvline(test_dates.min(), color="gray", lw=5, ls="--")

        if "isTrain" in df_plot.columns:
            # treat NaN (forecast rows) as False, then invert
            # is_train = df_plot["isTrain"].fillna(False).astype(bool)
            is_train = df_plot["isTrain"]\
                      .astype("boolean")\
                      .fillna(False)
            test_mask = ~is_train
            test_dates = df_plot.loc[test_mask].index
            if not test_dates.empty:
                ax_ts.axvline(test_dates.min(), color="gray", lw=5, ls="--")

        ax_ts.set_xlim(df_plot.index.min(), df_plot.index.max())
        ax_ts.set_ylim(ymin, ymax)
        ax_ts.margins(y=0.30)
        ax_ts.grid(True)
        ax_ts.tick_params(axis="x", rotation=45)
        ax_ts.set_title(f"Backtest + Forecast – {serie_name}")
        ax_ts.legend(loc="upper right")

        ax_hist.hist(df_r["y_real"].dropna().values, bins=30, orientation="horizontal", color="green", alpha=0.6, label="Actual (BT)")
        ax_hist.hist(df_r["y_pred_bt"].values, bins=30, orientation="horizontal", color="blue", alpha=0.6, label="Pred (BT)")
        ax_hist.hist(df_fc["y_pred_fc"].values, bins=30, orientation="horizontal", color="red", alpha=0.6, label="Pred (FC)")
        ax_hist.set_ylim(ymin, ymax)
        ax_hist.margins(y=0.30)
        ax_hist.set_xlabel("Frequency")
        ax_hist.set_title("Distribution")
        ax_hist.grid(True)
        ax_hist.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Utility to re‑index a single year to a common pivot (year 1920)
    # ------------------------------------------------------------------
    @staticmethod
    def compare_annual_series(df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Return a DataFrame with the selected *year* re‑based to 1920."""
        df_y = df[df.index.year == year].copy()
        df_y = df_y.rename(columns={"total_amount": f"saldo_{year}"})
        df_y.index = df_y.index.map(lambda dt: dt.replace(year=1920))
        return df_y
