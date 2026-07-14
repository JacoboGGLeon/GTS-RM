"""engineer.py – Engineer (data-engineering) layer.

The Engineer is intentionally model-agnostic: it cleans raw series, aligns
calendar features and builds per-series tabular datasets with `y` and
`y_trans`. Autoregressive lags are owned by Scientist because their depth is an
Optuna hyperparameter (`sliding_window`).
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import QuantileTransformer

# re-export configuration dataclass and mixin
from tools import Tools, SystemConfig, ChecklistMixin  # noqa: F401


class Engineer(ChecklistMixin):
    """Data-engineering pipeline for arbitrary target series.

    Stress-mode behavior is driven by SystemConfig:
    - `missing_policy`: handles sparse/missing target values without killing the run.
    - `target_transform`: optional reversible target preprocessing, e.g. signed_log.
    """

    def __init__(
        self,
        series: pd.DataFrame,
        calendar: pd.DataFrame,
        config: SystemConfig,
        test_size: int,
        *,
        qt_seed: int = 42,
    ) -> None:
        super().__init__()
        self.tools = Tools()
        self._series = series.copy()
        self._calendar = calendar.copy()
        self._config = config
        self._test_size = int(test_size)
        self._qt_seed = qt_seed
        self._series_status: Dict[str, Dict[str, Any]] = {}

        self._validate_columns()
        self._prepare(self._test_size)

    @ChecklistMixin.track("Validate columns")
    def _validate_columns(self) -> None:
        missing = [c for c in self._config.series_columns if c not in self._series.columns]
        if missing:
            raise ValueError(f"Missing series columns: {missing}")

    @staticmethod
    def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan)

    def _apply_missing_policy(self, s: pd.Series) -> pd.Series:
        policy = (self._config.missing_policy or "drop").strip().lower()
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if policy == "drop":
            return s.dropna()
        if policy == "zero":
            return s.fillna(0.0)
        if policy == "ffill_zero":
            return s.ffill().fillna(0.0)
        if policy == "interpolate_zero":
            return s.interpolate(limit_direction="both").fillna(0.0)
        raise ValueError("missing_policy must be one of: drop, zero, ffill_zero, interpolate_zero")

    @ChecklistMixin.track("Prepare pipeline")
    def _prepare(self, test_size: int) -> None:
        self._clean_dict = self._clean_series()
        self._outliers = self._compute_outliers()
        self._features, self._aligned_features = self._align_calendar()
        self._datasets, self._transforms = self._build_datasets(test_size)

    @ChecklistMixin.track("Clean series")
    def _clean_series(self) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for col in self._config.series_columns:
            raw = self._series[[col]].loc[~self._series.index.duplicated(keep="first")]
            y = self._apply_missing_policy(raw[col])
            df = y.astype(float).to_frame(name="total_amount")
            self._series_status[col] = {
                "raw_rows": int(len(raw)),
                "clean_rows": int(len(df)),
                "missing_policy": self._config.missing_policy,
                "n_unique": int(df["total_amount"].nunique(dropna=True)) if not df.empty else 0,
                "status": "cleaned" if not df.empty else "empty_after_clean",
            }
            result[col] = df
        return result

    @ChecklistMixin.track("Compute outliers")
    def _compute_outliers(self) -> pd.DataFrame:
        if not self._clean_dict:
            return pd.DataFrame()
        # Use the union of all indexes so sparse series do not inherit the first
        # series' calendar only.
        union_idx = None
        for df in self._clean_dict.values():
            union_idx = df.index if union_idx is None else union_idx.union(df.index)
        out = pd.DataFrame(index=union_idx.sort_values())
        for name, df in self._clean_dict.items():
            col = f"{name}_outlier"
            out[col] = 0
            if df.empty or df["total_amount"].nunique(dropna=True) <= 1:
                continue
            try:
                tmp = Tools().build_hierarchical_outliers({name: df})
                if col in tmp.columns:
                    out.loc[tmp.index, col] = tmp[col].astype(int)
            except Exception:
                # Outliers are diagnostic; never kill a stress-mode modeling run.
                out[col] = 0
        return out.fillna(0).astype(int)

    @ChecklistMixin.track("Align calendar")
    def _align_calendar(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feats = self._calendar.copy().replace({False: 0, True: 1})
        feats = self._coerce_numeric_frame(feats).fillna(0.0)
        idx = feats.index.intersection(self._outliers.index) if not self._outliers.empty else feats.index
        aligned = feats.loc[idx].copy()
        return feats, aligned

    @ChecklistMixin.track("Build datasets")
    def _build_datasets(self, test_size: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, TransformerMixin]]:
        datasets: Dict[str, pd.DataFrame] = {}
        transforms: Dict[str, TransformerMixin] = {}

        for name, clean_df in self._clean_dict.items():
            if clean_df.empty:
                self._series_status[name]["status"] = "skipped_empty"
                continue

            df = clean_df.join(self._features, how="inner")
            df = self._coerce_numeric_frame(df)
            feature_cols = [c for c in df.columns if c != "total_amount"]
            df[feature_cols] = df[feature_cols].fillna(0.0)
            df["y"] = df["total_amount"]
            df = df.drop(columns=["total_amount"])
            df = df.dropna(subset=["y"])
            if df.empty:
                self._series_status[name]["status"] = "skipped_no_target_after_join"
                continue

            effective_test = min(max(1, int(test_size)), max(1, len(df) - 1)) if len(df) > 1 else 1
            train_idx = df.index[:-effective_test] if len(df) > effective_test else df.index[:1]
            test_idx = df.index.difference(train_idx)
            if len(train_idx) == 0:
                train_idx = df.index
                test_idx = pd.Index([])

            cls = self._config.transformer_cls
            kwargs = dict(self._config.transformer_kwargs)
            if cls is QuantileTransformer:
                kwargs["n_quantiles"] = max(1, min(int(kwargs.get("n_quantiles", len(train_idx))), len(train_idx)))

            transformer = Tools.make_target_transformer(
                cls,
                kwargs,
                target_transform=self._config.target_transform,
            )
            transformer.fit(df.loc[train_idx, ["y"]])
            df.loc[train_idx, "y_trans"] = transformer.transform(df.loc[train_idx, ["y"]]).ravel()
            if len(test_idx):
                df.loc[test_idx, "y_trans"] = transformer.transform(df.loc[test_idx, ["y"]]).ravel()

            df["y_trans"] = pd.to_numeric(df["y_trans"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            datasets[name] = df
            transforms[name] = transformer
            self._series_status[name].update({
                "status": "ready",
                "rows": int(len(df)),
                "effective_test_size": int(effective_test),
                "target_transform": self._config.target_transform,
                "constant_series": bool(df["y"].nunique(dropna=True) <= 1),
            })

        return datasets, transforms

    @property
    def series(self) -> pd.DataFrame:
        return self._series

    @property
    def calendar(self) -> pd.DataFrame:
        return self._calendar

    @property
    def config(self) -> SystemConfig:
        return self._config

    @property
    def features(self) -> pd.DataFrame:
        return self._features

    @property
    def aligned_features(self) -> pd.DataFrame:
        return self._aligned_features

    @property
    def outliers(self) -> pd.DataFrame:
        return self._outliers

    @property
    def datasets(self) -> Dict[str, pd.DataFrame]:
        return self._datasets

    @property
    def transforms(self) -> Dict[str, TransformerMixin]:
        return self._transforms

    @property
    def series_status(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._series_status, orient="index")

    @ChecklistMixin.track("Show summary")
    def show_summary(self, head: int = 3) -> Dict[str, pd.DataFrame]:
        summary = {
            "series": pd.concat([self.series.head(head), self.series.tail(head)]),
            "calendar": pd.concat([self.calendar.head(head), self.calendar.tail(head)]),
            "features": pd.concat([self.features.head(head), self.features.tail(head)]),
            "outliers": pd.concat([self.outliers.head(head), self.outliers.tail(head)]),
            "series_status": self.series_status,
        }
        for k, df in summary.items():
            print(f"\n--- {k.upper()} ---")
            print(df)
        return summary

    @ChecklistMixin.track("Visualise data")
    def visualize(self, series_names: Optional[List[str]] = None) -> None:
        names = series_names or self._config.series_columns
        for name in names:
            if name not in self._datasets:
                continue
            self.tools.plot_series_outliers(self._datasets, self._outliers, name)
            self.tools.plot_annual_comparison(self._datasets, self._outliers, name)
