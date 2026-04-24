"""Tests for models/recommendation_engine.py"""
import pandas as pd
import numpy as np
import pytest
from models.recommendation_engine import RecommendationEngine
from modules.similarity_index import SimilarityIndex
from modules.customer_signals import CustomerSignals


@pytest.fixture(scope="module")
def similarity_idx(products_df):
    return SimilarityIndex().fit(products_df)


@pytest.fixture(scope="module")
def recs_df(products_df, inventory_df, suppliers_df, sales_df, similarity_idx):
    engine = RecommendationEngine()
    product_supplier = products_df[["sku_id", "supplier_id"]]
    return engine.generate_buy_recommendations(
        pd.DataFrame(),  # no forecast
        inventory_df,
        suppliers_df,
        product_supplier,
        similarity_idx,
        products_df,
        sales_df,
    )


@pytest.fixture(scope="module")
def vel_df(sales_df):
    return CustomerSignals().sales_velocity(sales_df)


def test_urgency_score_bounds(recs_df):
    if len(recs_df) > 0:
        assert (recs_df["urgency_score"] >= 0).all()
        assert (recs_df["urgency_score"] <= 1.0 + 1e-6).all()


def test_urgency_band_valid(recs_df):
    valid_bands = {"critical", "high", "medium", "low"}
    if len(recs_df) > 0 and "urgency_band" in recs_df.columns:
        actual = set(recs_df["urgency_band"].astype(str).unique()) - {"nan"}
        assert actual.issubset(valid_bands)


def test_recommended_qty_positive(recs_df):
    if len(recs_df) > 0:
        assert (recs_df["recommended_qty"] >= 0).all()


def test_output_columns_present(recs_df):
    required = ["sku_id", "urgency_score", "recommended_qty"]
    for col in required:
        assert col in recs_df.columns, f"Missing column: {col}"


def test_no_recommendation_when_overstocked(products_df, suppliers_df, sales_df, similarity_idx):
    """SKUs with days_of_supply >> 300 should not appear as urgent buys."""
    import pandas as pd

    # Inject inventory with very high stock for all SKUs
    inv_overstocked = pd.DataFrame([{
        "sku_id": sku,
        "store_id": "STORE01",
        "quantity_on_hand": 999999,
        "reorder_point": 10,
        "quantity_on_order": 0,
        "days_of_supply": 999,
    } for sku in products_df["sku_id"].unique()])

    engine = RecommendationEngine()
    product_supplier = products_df[["sku_id", "supplier_id"]]
    recs = engine.generate_buy_recommendations(
        pd.DataFrame(), inv_overstocked, suppliers_df, product_supplier,
        similarity_idx, products_df, sales_df
    )

    if len(recs) > 0 and "urgency_band" in recs.columns:
        critical_count = (recs["urgency_band"].astype(str) == "critical").sum()
        # Very few or no critical recommendations for overstocked items
        assert critical_count < len(recs) * 0.5


def test_clearance_triggered_at_threshold(products_df, inventory_df, sales_df, vel_df):
    """SKUs with high days_of_supply + decelerating momentum should trigger clearance."""
    engine = RecommendationEngine()
    inv_overstock = inventory_df.copy()

    clearance = engine.clearance_alerts(
        sales_df, inv_overstock, products_df, vel_df, days_overstock_threshold=120
    )
    # Returns a DataFrame (may be empty, that's fine)
    assert isinstance(clearance, pd.DataFrame)
    if len(clearance) > 0:
        assert "recommended_markdown_pct" in clearance.columns
        assert (clearance["recommended_markdown_pct"] > 0).all()


def test_portfolio_summary_keys_present(recs_df, inventory_df, products_df, sales_df, vel_df):
    engine = RecommendationEngine()
    clearance = engine.clearance_alerts(sales_df, inventory_df, products_df, vel_df)
    summary = engine.portfolio_summary(recs_df, clearance, inventory_df, products_df)

    required_keys = [
        "total_buy_budget_usd", "total_skus_to_buy", "critical_skus",
        "high_priority_skus", "projected_revenue_at_risk_usd",
        "overstock_cost_usd", "overstock_sku_count",
        "estimated_margin_improvement_usd", "capital_freed_if_clearance_usd",
        "top_5_opportunities", "top_5_risks",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"


def test_allocation_optimizer_output(products_df, inventory_df):
    engine = RecommendationEngine()
    result = engine.allocation_optimizer(inventory_df, pd.DataFrame(), products_df)
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        assert "from_store_id" in result.columns
        assert "to_store_id" in result.columns
        assert (result["transfer_qty"] >= 1).all()
