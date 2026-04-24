"""
XGBoost-based demand forecasting — **weekly granularity**.

Key design decisions:
  - Aggregate daily transactions to weekly sums per SKU.
    Weekly demand avg ~6.4 units (vs 1.7 daily); 0% zero-demand weeks (vs 51% daily).
    → Eliminates intermittent-demand MAPE inflation; target MAPE 20–35%.
  - log1p(demand) target → back-transform with expm1.
  - Rich weekly lag set: 1, 2, 4, 8, 13 weeks.
  - Rolling windows: 4, 8, 13 weeks.
  - EWM span=4 (short) and span=13 (quarter) for trend.
  - Cyclical encoding of week-of-year and month.
  - SKU target-encoding: historical weekly mean, sku×month seasonality.
  - Interaction features: sku_mean × month_sin, prior × ewm.
  - Forecast() returns one row per SKU per week (date = week Monday).
    Downstream sums are preserved: sum(forecasted_demand) ≈ 90-day total.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.settings import Settings, get_settings
from utils.logger import get_logger
from utils.metrics import mape, rmse, mae, r2


class DemandForecaster:
    """
    XGBoost-based demand forecasting with time-series feature engineering.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.logger = get_logger(__name__)
        self.model = None
        self.feature_cols: list = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._test_rmse: float = 5.0
        self._sku_list: list = []
        self._sku_mean_lookup:  dict = {}
        self._sku_log_lookup:   dict = {}
        self._sku_freq_lookup:  dict = {}
        self._sku_dow_lookup:   dict = {}
        self._sku_month_lookup: dict = {}
        self._sku_woy_lookup:   dict = {}

    # ─── feature engineering ──────────────────────────────────────────────────
    def prepare_features(
        self,
        sales_df: pd.DataFrame,
        products_df: pd.DataFrame,
        social_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate transactions to **weekly** SKU-level demand and engineer
        time-series features at weekly granularity.

        Weekly aggregation rationale:
          - Daily avg demand ≈ 1.7 units with ~51% zero days  → MAPE unreliable
          - Weekly avg demand ≈ 6.4 units with   0% zero weeks → smooth signal
          - Expected MAPE improvement: 50%+ daily → 20-35% weekly
        """
        self.logger.info("Preparing WEEKLY features for demand forecasting...")

        # ── 1. Daily → Weekly aggregation ────────────────────────────────────
        df = sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Compute week-start (Monday) for each transaction
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
        weekly_raw = (
            df.groupby(["sku_id", "week_start"])["quantity_sold"]
            .sum()
            .reset_index()
            .rename(columns={"week_start": "date", "quantity_sold": "demand"})
        )

        # Full week × SKU grid (no gaps — needed for correct rolling windows)
        all_skus  = weekly_raw["sku_id"].unique()
        all_weeks = pd.date_range(weekly_raw["date"].min(), weekly_raw["date"].max(), freq="W-MON")
        idx = pd.MultiIndex.from_product([all_skus, all_weeks], names=["sku_id", "date"])
        weekly = (
            weekly_raw.set_index(["sku_id", "date"])
            .reindex(idx, fill_value=0)
            .reset_index()
        )

        # ── 2. Weekly time features ───────────────────────────────────────────
        weekly["month"]        = weekly["date"].dt.month
        weekly["quarter"]      = weekly["date"].dt.quarter
        weekly["week_of_year"] = weekly["date"].dt.isocalendar().week.astype(int)
        weekly["year"]         = weekly["date"].dt.year

        # Cyclical encoding (periodicity without ordinal ordering)
        weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"]        / 12)
        weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"]        / 12)
        weekly["woy_sin"]   = np.sin(2 * np.pi * weekly["week_of_year"] / 52)
        weekly["woy_cos"]   = np.cos(2 * np.pi * weekly["week_of_year"] / 52)
        weekly["q_sin"]     = np.sin(2 * np.pi * weekly["quarter"]      / 4)
        weekly["q_cos"]     = np.cos(2 * np.pi * weekly["quarter"]      / 4)

        # Holiday-week flag (week contains a major holiday)
        holiday_months_days = {
            (1, 1), (1, 26), (8, 15), (10, 2),
            (11, 25), (11, 26), (12, 25), (12, 31),
            (7, 4), (5, 27),
        }
        def _has_holiday(week_start: pd.Timestamp) -> int:
            for d in range(7):
                dt = week_start + pd.Timedelta(days=d)
                if (dt.month, dt.day) in holiday_months_days:
                    return 1
            return 0
        weekly["has_holiday_week"] = weekly["date"].apply(_has_holiday)

        # Weeks since product launch
        if "launch_date" in products_df.columns:
            launch = products_df[["sku_id", "launch_date"]].copy()
            launch["launch_date"] = pd.to_datetime(launch["launch_date"])
            weekly = weekly.merge(launch, on="sku_id", how="left")
            weekly["weeks_since_launch"] = (
                (weekly["date"] - weekly["launch_date"]).dt.days / 7
            ).fillna(52).clip(lower=0)
        else:
            weekly["weeks_since_launch"] = 52.0

        # ── 3. Weekly lag features ────────────────────────────────────────────
        weekly = weekly.sort_values(["sku_id", "date"])
        grp = weekly.groupby("sku_id")["demand"]
        for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
            weekly[f"lag_{lag}w"] = grp.shift(lag).fillna(0)

        # ── 4. Rolling statistics (shifted by 1 week to avoid leakage) ────────
        weekly["_w_sh"] = grp.shift(1).fillna(0)
        _sh_grp = weekly.groupby("sku_id")["_w_sh"]
        for w in [4, 8, 13, 26]:
            weekly[f"rolling_mean_{w}w"] = (
                _sh_grp.rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True).fillna(0)
            )
        weekly["rolling_std_13w"] = (
            _sh_grp.rolling(13, min_periods=1).std()
            .reset_index(level=0, drop=True).fillna(0)
        )
        weekly["rolling_max_13w"] = (
            _sh_grp.rolling(13, min_periods=1).max()
            .reset_index(level=0, drop=True).fillna(0)
        )
        weekly["rolling_median_13w"] = (
            _sh_grp.rolling(13, min_periods=1).median()
            .reset_index(level=0, drop=True).fillna(0)
        )
        weekly = weekly.drop(columns=["_w_sh"])

        # ── 5. EWM — short (4-week) and long (13-week / quarter) trend ────────
        weekly["_w_sh2"] = weekly.groupby("sku_id")["demand"].shift(1).fillna(0)
        for span in [4, 13]:
            weekly[f"ewm_{span}w"] = (
                weekly.groupby("sku_id")["_w_sh2"]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
                .fillna(0)
            )
        weekly = weekly.drop(columns=["_w_sh2"])

        # ── 6. SKU-level target encoding ──────────────────────────────────────
        # Weekly mean demand per SKU (all weeks, including zeros for completeness)
        sku_stats = (
            weekly.groupby("sku_id")["demand"]
            .agg(sku_mean_demand="mean", sku_median_demand="median", sku_total="sum")
            .reset_index()
        )
        sku_stats["sku_log_mean"] = np.log1p(sku_stats["sku_mean_demand"])
        sku_stats["sku_cv"] = (
            sku_stats["sku_mean_demand"] / (sku_stats["sku_median_demand"] + 1e-6)
        ).fillna(1.0)
        weekly = weekly.merge(sku_stats[["sku_id", "sku_mean_demand", "sku_log_mean", "sku_cv"]],
                              on="sku_id", how="left")
        weekly["sku_mean_demand"] = weekly["sku_mean_demand"].fillna(0)
        weekly["sku_log_mean"]    = weekly["sku_log_mean"].fillna(0)
        weekly["sku_cv"]          = weekly["sku_cv"].fillna(1.0)

        # ── 6b. SKU × month prior (monthly seasonality) ───────────────────────
        sku_month = (
            weekly.groupby(["sku_id", "month"])["demand"].mean()
            .reset_index()
            .rename(columns={"demand": "sku_month_mean"})
        )
        weekly = weekly.merge(sku_month, on=["sku_id", "month"], how="left")
        weekly["sku_month_mean"] = weekly["sku_month_mean"].fillna(weekly["sku_mean_demand"])

        # ── 6c. SKU × week-of-year prior (intra-year seasonal index) ──────────
        sku_woy = (
            weekly.groupby(["sku_id", "week_of_year"])["demand"].mean()
            .reset_index()
            .rename(columns={"demand": "sku_woy_mean"})
        )
        weekly = weekly.merge(sku_woy, on=["sku_id", "week_of_year"], how="left")
        weekly["sku_woy_mean"] = weekly["sku_woy_mean"].fillna(weekly["sku_mean_demand"])

        # ── 7. Interaction features ───────────────────────────────────────────
        weekly["sku_mean_x_month_sin"] = weekly["sku_mean_demand"] * weekly["month_sin"]
        weekly["sku_mean_x_woy_sin"]   = weekly["sku_mean_demand"] * weekly["woy_sin"]
        weekly["sku_log_x_lag4w"]      = weekly["sku_log_mean"]    * weekly["lag_4w"]
        weekly["prior_x_ewm4"]         = weekly["sku_month_mean"]  * weekly["ewm_4w"]
        weekly["prior_x_ewm13"]        = weekly["sku_woy_mean"]    * weekly["ewm_13w"]

        # ── 8. Social signal features (lagged 1 week) ─────────────────────────
        if len(social_df) > 0:
            social = social_df.copy()
            social["week_start"] = pd.to_datetime(social["week_start"])
            # Normalise to Monday
            social["week_start"] = social["week_start"] - pd.to_timedelta(
                social["week_start"].dt.dayofweek, unit="D"
            )
            soc_agg = (
                social.groupby(["sku_id", "week_start"])
                .agg(
                    trend_index=("trend_index", "mean"),
                    sentiment_score=("sentiment_score", "mean"),
                    mention_count=("mention_count", "sum"),
                )
                .reset_index()
            )
            # Lag 1 week
            soc_agg["week_start"] = soc_agg["week_start"] + pd.Timedelta(weeks=1)
            soc_agg = soc_agg.rename(columns={"week_start": "date",
                                               "trend_index": "trend_index_lag1w",
                                               "sentiment_score": "sentiment_lag1w",
                                               "mention_count": "mention_lag1w"})
            weekly = weekly.merge(soc_agg, on=["sku_id", "date"], how="left")
            weekly["trend_index_lag1w"] = weekly["trend_index_lag1w"].fillna(50)
            weekly["sentiment_lag1w"]   = weekly["sentiment_lag1w"].fillna(0.6)
            weekly["mention_lag1w"]     = weekly["mention_lag1w"].fillna(100)
        else:
            weekly["trend_index_lag1w"] = 50.0
            weekly["sentiment_lag1w"]   = 0.6
            weekly["mention_lag1w"]     = 100.0

        # ── 9. Product features (label encoded) ───────────────────────────────
        cat_cols = ["price_point", "frame_shape", "material", "category", "gender"]
        prod_meta = products_df[
            ["sku_id"] + [c for c in cat_cols if c in products_df.columns]
        ].copy()
        weekly = weekly.merge(prod_meta, on="sku_id", how="left")
        for col in cat_cols:
            if col not in weekly.columns:
                continue
            if col not in self._label_encoders:
                le = LabelEncoder()
                weekly[f"{col}_enc"] = le.fit_transform(weekly[col].fillna("unknown").astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders[col]
                vals = weekly[col].fillna("unknown").astype(str)
                known = vals.isin(le.classes_)
                enc = np.full(len(weekly), -1, dtype=int)
                if known.any():
                    enc[known.values] = le.transform(vals[known]).astype(int)
                weekly[f"{col}_enc"] = enc

        # ── 10. Inventory features ────────────────────────────────────────────
        if len(inventory_df) > 0:
            inv_agg = inventory_df.groupby("sku_id").agg(
                days_of_supply=("days_of_supply", "mean"),
                stockout_risk_flag=("days_of_supply", lambda x: int((x < 7).any())),
            ).reset_index()
            weekly = weekly.merge(inv_agg, on="sku_id", how="left")
            weekly["days_of_supply"]     = weekly["days_of_supply"].fillna(60)
            weekly["stockout_risk_flag"] = weekly["stockout_risk_flag"].fillna(0)
        else:
            weekly["days_of_supply"]     = 60.0
            weekly["stockout_risk_flag"] = 0

        # ── Clean up ──────────────────────────────────────────────────────────
        drop_cols = ["launch_date", "sku_median_demand"] + cat_cols
        weekly = weekly.drop(columns=drop_cols, errors="ignore")
        weekly = weekly.fillna(0)

        self._sku_list = weekly["sku_id"].unique().tolist()

        # Store lookups for forecast()
        self._sku_mean_lookup  = weekly.groupby("sku_id")["sku_mean_demand"].first().to_dict()
        self._sku_log_lookup   = weekly.groupby("sku_id")["sku_log_mean"].first().to_dict()
        self._sku_freq_lookup  = {}  # not used in weekly model
        self._sku_month_lookup = (
            weekly.groupby(["sku_id", "month"])["sku_month_mean"].first().to_dict()
        )
        self._sku_woy_lookup = (
            weekly.groupby(["sku_id", "week_of_year"])["sku_woy_mean"].first().to_dict()
        )

        n_weeks = weekly["date"].nunique()
        n_skus  = weekly["sku_id"].nunique()
        zero_pct = (weekly["demand"] == 0).mean() * 100
        self.logger.info(
            "Weekly feature matrix ready",
            n_rows=len(weekly),
            n_weeks=n_weeks,
            n_skus=n_skus,
            zero_demand_pct=round(zero_pct, 1),
        )
        return weekly

    # ─── training ─────────────────────────────────────────────────────────────
    def train(
        self,
        features_df: pd.DataFrame,
        test_months: int = 3,
    ) -> dict:
        """
        Train on log1p(demand), evaluating only on non-zero test rows.
        This eliminates bias from sparse zero-demand days.
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required. pip install xgboost") from e

        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        cutoff = df["date"].max() - pd.DateOffset(months=test_months)
        train_df = df[df["date"] <= cutoff].copy()
        test_df  = df[df["date"] >  cutoff].copy()

        exclude_cols = {"sku_id", "date", "demand", "launch_date"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.feature_cols = feature_cols

        # Evaluate arrays
        X_test_all = test_df[feature_cols].values.astype(float)
        y_test_all = test_df["demand"].values.astype(float)

        # Train on ALL rows — zeros become log1p(0)=0, positives are log-compressed
        # This gives the model both "no demand" and "positive demand" signal
        X_train = train_df[feature_cols].values.astype(float)
        y_train = np.log1p(train_df["demand"].values.astype(float))  # log1p transform

        # Validation set for early stopping: last 10% of training data
        val_size = max(200, int(0.10 * len(X_train)))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_tr,  y_tr  = X_train[:-val_size], y_train[:-val_size]

        self.model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=5,
            subsample=0.75,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            early_stopping_rounds=40,
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Back-transform predictions
        y_pred_log = self.model.predict(X_test_all)
        y_pred = np.expm1(y_pred_log).clip(0)
        y_test_log = np.log1p(y_test_all)

        self._test_rmse = float(np.sqrt(np.mean((y_pred - y_test_all) ** 2)))

        # MAPE / wMAPE on positive-demand rows only (avoids div-by-zero)
        pos_mask = y_test_all > 0
        y_test_pos = y_test_all[pos_mask]
        y_pred_pos = y_pred[pos_mask]

        # wMAPE = sum(|actual-pred|) / sum(actual) — volume-weighted metric
        wm = float(np.sum(np.abs(y_test_pos - y_pred_pos)) / (np.sum(y_test_pos) + 1e-9) * 100)

        # R² in log space on ALL test rows — matches training objective, always valid
        ss_res = float(np.sum((y_test_log - y_pred_log) ** 2))
        ss_tot = float(np.sum((y_test_log - float(np.mean(y_test_log))) ** 2))
        r2_log = float(1.0 - ss_res / (ss_tot + 1e-9))

        metrics = {
            "mape":   mape(y_test_pos, y_pred_pos),
            "wmape":  round(wm, 4),
            "r2_log": round(r2_log, 4),
            "rmse":   self._test_rmse,
            "mae":    float(np.mean(np.abs(y_test_pos - y_pred_pos))),
        }

        fi = pd.DataFrame({
            "feature":    feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        result_test = test_df[["sku_id", "date", "demand"]].copy()
        result_test["predicted"] = y_pred

        self.logger.info("Model trained", **{k: round(float(v), 4) for k, v in metrics.items()})
        return {
            "model":                self.model,
            "feature_importance_df": fi,
            "metrics":              metrics,
            "test_df":              result_test,
        }

    # ─── forecasting ──────────────────────────────────────────────────────────
    def forecast(
        self,
        model,
        features_df: pd.DataFrame,
        horizon_days: int = 90,
    ) -> pd.DataFrame:
        """
        Vectorised weekly forward forecast per SKU.
        horizon_days is converted to weeks (rounded up).
        Returns one row per SKU per week with:
          date              = week start (Monday)
          forecasted_demand = predicted weekly demand
          lower_bound / upper_bound = 80% prediction interval
        Downstream sum-based consumption (recommendation engine, supply chain)
        is unaffected — weekly totals aggregate correctly over any horizon.
        """
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        last_week_start = df["date"].max()

        horizon_weeks = max(1, (horizon_days + 6) // 7)

        exclude_cols = {"sku_id", "date", "demand", "launch_date"}
        feature_cols = self.feature_cols or [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Seed from the last observed week per SKU
        base = df.sort_values("date").groupby("sku_id").last().reset_index()

        future_weeks = pd.date_range(
            last_week_start + pd.Timedelta(weeks=1),
            periods=horizon_weeks,
            freq="W-MON",
        )

        all_rows: list[dict] = []
        for _, sku_row in base.iterrows():
            sku = sku_row["sku_id"]
            sku_mean       = self._sku_mean_lookup.get(sku, float(sku_row.get("sku_mean_demand", 0)))
            sku_log        = self._sku_log_lookup.get(sku,  float(sku_row.get("sku_log_mean",  0)))

            for fdate in future_weeks:
                month  = fdate.month
                woy    = fdate.isocalendar()[1]
                feat   = sku_row.to_dict()

                feat["sku_id"]       = sku
                feat["date"]         = fdate
                feat["month"]        = month
                feat["quarter"]      = (month - 1) // 3 + 1
                feat["week_of_year"] = woy
                feat["year"]         = fdate.year

                feat["month_sin"] = np.sin(2 * np.pi * month / 12)
                feat["month_cos"] = np.cos(2 * np.pi * month / 12)
                feat["woy_sin"]   = np.sin(2 * np.pi * woy   / 52)
                feat["woy_cos"]   = np.cos(2 * np.pi * woy   / 52)
                feat["q_sin"]     = np.sin(2 * np.pi * feat["quarter"] / 4)
                feat["q_cos"]     = np.cos(2 * np.pi * feat["quarter"] / 4)

                # SKU seasonal priors
                sku_month_mean = self._sku_month_lookup.get((sku, month), sku_mean)
                sku_woy_mean   = self._sku_woy_lookup.get(  (sku, woy),   sku_mean)
                feat["sku_mean_demand"]    = sku_mean
                feat["sku_log_mean"]       = sku_log
                feat["sku_month_mean"]     = sku_month_mean
                feat["sku_woy_mean"]       = sku_woy_mean

                # Interaction features
                feat["sku_mean_x_month_sin"] = sku_mean        * feat["month_sin"]
                feat["sku_mean_x_woy_sin"]   = sku_mean        * feat["woy_sin"]
                feat["sku_log_x_lag4w"]      = sku_log         * float(feat.get("lag_4w", 0))
                feat["prior_x_ewm4"]         = sku_month_mean  * float(feat.get("ewm_4w", sku_mean))
                feat["prior_x_ewm13"]        = sku_woy_mean    * float(feat.get("ewm_13w", sku_mean))

                feat["weeks_since_launch"] = float(feat.get("weeks_since_launch", 52)) + 1

                all_rows.append(feat)

        if not all_rows:
            return pd.DataFrame(columns=["sku_id", "date", "forecasted_demand",
                                         "lower_bound", "upper_bound"])

        batch_df = pd.DataFrame(all_rows)
        X_batch = np.array([[r.get(c, 0) for c in feature_cols] for r in all_rows], dtype=float)

        # Predict in log space, back-transform
        preds_log = model.predict(X_batch)
        preds     = np.expm1(preds_log).clip(0)

        # 80% interval: ±1.28σ
        margin   = 1.28 * self._test_rmse
        sku_col  = batch_df["sku_id"].values
        date_col = batch_df["date"].dt.strftime("%Y-%m-%d").values

        rows = [
            {
                "sku_id":            sku,
                "date":              d,
                "forecasted_demand": round(float(p), 2),
                "lower_bound":       round(max(0.0, float(p) - margin), 2),
                "upper_bound":       round(float(p) + margin, 2),
            }
            for sku, d, p in zip(sku_col, date_col, preds)
        ]
        return pd.DataFrame(rows)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Returns: mape, rmse, mae, r2, bias_pct"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        bias = float(np.mean(y_pred - y_true))
        mean_true = float(np.mean(y_true))
        bias_pct = bias / mean_true * 100 if mean_true != 0 else 0.0
        return {
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "bias_pct": round(bias_pct, 4),
        }

    def save(self, path: str) -> None:
        """Save model + metadata to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("DemandForecaster saved", path=path)

    @classmethod
    def load(cls, path: str) -> "DemandForecaster":
        """Load from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_synthetic_data import (
        generate_products, generate_sales, generate_inventory
    )
    import pandas as pd

    print("Generating data...")
    products = generate_products(50)
    sales = generate_sales(products, 10_000)
    inventory = generate_inventory(products, sales)

    forecaster = DemandForecaster()
    print("Preparing features...")
    features = forecaster.prepare_features(sales, products, pd.DataFrame(), inventory)
    print(f"Feature matrix: {features.shape}")

    print("Training...")
    result = forecaster.train(features, test_months=2)
    print("Metrics:", result["metrics"])
    print("Top features:\n", result["feature_importance_df"].head(10))

    print("Forecasting 30 days...")
    fcast = forecaster.forecast(result["model"], features, horizon_days=30)
    print(fcast.head(5))
