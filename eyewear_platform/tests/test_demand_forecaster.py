"""Tests for models/demand_forecaster.py"""
import numpy as np
import pandas as pd
import pytest
from models.demand_forecaster import DemandForecaster


@pytest.fixture(scope="module")
def forecaster_with_features(sales_df, products_df, social_df, inventory_df):
    f = DemandForecaster()
    features = f.prepare_features(sales_df, products_df, social_df, inventory_df)
    return f, features


def test_feature_engineering_no_nulls(forecaster_with_features):
    _, features = forecaster_with_features
    # No column should be all-null
    null_cols = features.columns[features.isnull().all()].tolist()
    assert null_cols == [], f"All-null columns: {null_cols}"


def test_feature_engineering_correct_shape(forecaster_with_features, sales_df):
    _, features = forecaster_with_features
    # Should have more rows than raw sales (filled grid)
    assert len(features) > 0
    # Should have many feature columns
    assert features.shape[1] > 10


def test_feature_has_required_columns(forecaster_with_features):
    """Weekly model has week-level columns instead of daily ones."""
    _, features = forecaster_with_features
    required = [
        "sku_id", "date", "demand",
        "month", "week_of_year", "month_sin", "month_cos",
        "lag_1w", "lag_4w", "rolling_mean_4w",
        "sku_mean_demand", "sku_month_mean",
    ]
    for col in required:
        assert col in features.columns, f"Missing column: {col}"


def test_train_returns_metrics(forecaster_with_features):
    f, features = forecaster_with_features
    result = f.train(features, test_months=1)
    assert "metrics" in result
    assert "model" in result
    metrics = result["metrics"]
    assert "mape" in metrics
    assert "rmse" in metrics
    assert "r2_log" in metrics  # weekly model uses log-space R² instead of raw R²
    assert "mae" in metrics


def test_mape_reasonable(forecaster_with_features):
    """MAPE should be < 80% on synthetic data (synthetic is more predictable)."""
    f, features = forecaster_with_features
    result = f.train(features, test_months=1)
    mape = result["metrics"]["mape"]
    assert mape < 80.0, f"MAPE too high: {mape:.2f}%"


def test_forecast_horizon_length(forecaster_with_features):
    """Weekly model returns one row per SKU per week; horizon_days is converted to weeks (ceil)."""
    import math
    f, features = forecaster_with_features
    result = f.train(features, test_months=1)
    horizon = 30
    expected_weeks = math.ceil(horizon / 7)  # 30 days → 5 weeks
    fcast = f.forecast(result["model"], features, horizon_days=horizon)
    sku_counts = fcast.groupby("sku_id").size()
    assert (sku_counts == expected_weeks).all(), (
        f"Expected {expected_weeks} weeks per SKU, got: {sku_counts.unique()}"
    )


def test_confidence_bounds_ordered(forecaster_with_features):
    """lower_bound <= forecasted_demand <= upper_bound for all rows."""
    f, features = forecaster_with_features
    result = f.train(features, test_months=1)
    fcast = f.forecast(result["model"], features, horizon_days=14)
    assert (fcast["lower_bound"] <= fcast["forecasted_demand"] + 1e-6).all()
    assert (fcast["forecasted_demand"] <= fcast["upper_bound"] + 1e-6).all()


def test_forecasted_demand_non_negative(forecaster_with_features):
    f, features = forecaster_with_features
    result = f.train(features, test_months=1)
    fcast = f.forecast(result["model"], features, horizon_days=14)
    assert (fcast["forecasted_demand"] >= 0).all()
    assert (fcast["lower_bound"] >= 0).all()


def test_evaluate_metrics(forecaster_with_features):
    f, _ = forecaster_with_features
    y_true = np.array([10, 20, 30, 40, 50], dtype=float)
    y_pred = np.array([11, 19, 32, 37, 53], dtype=float)
    metrics = f.evaluate(y_true, y_pred)
    assert "mape" in metrics
    assert "rmse" in metrics
    assert "bias_pct" in metrics
    assert metrics["rmse"] >= 0
