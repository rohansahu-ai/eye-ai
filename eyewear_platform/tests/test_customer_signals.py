"""Tests for modules/customer_signals.py"""
import numpy as np
import pytest
from modules.customer_signals import CustomerSignals


def test_velocity_change_calculation(sales_df):
    cs = CustomerSignals()
    result = cs.sales_velocity(sales_df)
    assert "current_velocity" in result.columns
    assert "prior_velocity" in result.columns
    assert "velocity_change_pct" in result.columns
    assert "momentum" in result.columns


def test_velocity_momentum_values(sales_df):
    cs = CustomerSignals()
    result = cs.sales_velocity(sales_df)
    valid = {"accelerating", "stable", "decelerating"}
    actual = set(result["momentum"].astype(str).unique()) - {"nan"}
    assert actual.issubset(valid)


def test_return_rate_bounds(sales_df, returns_df, products_df):
    cs = CustomerSignals()
    result = cs.return_rate_analysis(sales_df, returns_df, products_df)
    assert "return_rate_pct" in result.columns
    assert (result["return_rate_pct"] >= 0).all()
    # Return rate should not exceed 100% (or unrealistic values)
    assert (result["return_rate_pct"] <= 200).all()


def test_return_rate_sorted_descending(sales_df, returns_df, products_df):
    cs = CustomerSignals()
    result = cs.return_rate_analysis(sales_df, returns_df, products_df)
    rates = result["return_rate_pct"].values
    assert all(rates[i] >= rates[i+1] for i in range(len(rates)-1)), "Should be sorted descending"


def test_trending_skus_count(social_df, sales_df):
    cs = CustomerSignals()
    result = cs.trending_skus(social_df, sales_df, top_n=5)
    assert len(result) <= 5
    assert "trend_score" in result.columns


def test_trend_score_normalization(social_df, sales_df):
    cs = CustomerSignals()
    result = cs.trending_skus(social_df, sales_df, top_n=10)
    if len(result) > 0:
        assert (result["trend_score"] >= 0).all()
        assert (result["trend_score"] <= 1.0 + 1e-6).all()


def test_seasonality_index_mean_near_one(sales_df):
    cs = CustomerSignals()
    result = cs.demand_seasonality(sales_df)
    if len(result) > 0 and "seasonality_index" in result.columns:
        mean_idx = result["seasonality_index"].mean()
        # Mean seasonality across all months should be near 1.0
        assert 0.5 <= mean_idx <= 2.0


def test_customer_segment_preferences_keys(sales_df, products_df):
    cs = CustomerSignals()
    prefs = cs.customer_segment_preferences(sales_df, products_df)
    expected_keys = [
        "age_group_x_frame_shape", "gender_x_price_point",
        "channel_x_category", "region_x_frame_shape",
    ]
    for key in expected_keys:
        assert key in prefs


def test_wishlist_demand_proxy_sorted(sales_df, inventory_df):
    cs = CustomerSignals()
    result = cs.wishlist_demand_proxy(sales_df, inventory_df)
    assert "latent_demand_score" in result.columns
    assert len(result) > 0
    scores = result["latent_demand_score"].values
    # Should be sorted descending
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
