"""
Business and forecast evaluation metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Handles zeros safely."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def inventory_turnover(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    period_days: int = 365,
) -> pd.DataFrame:
    """Per-SKU inventory turnover ratio = COGS / avg_inventory_value."""
    # Compute COGS proxy from sales
    if "unit_price" in sales_df.columns and "quantity_sold" in sales_df.columns:
        sales_df = sales_df.copy()
        sales_df["revenue"] = sales_df["unit_price"] * sales_df["quantity_sold"]
        cogs = sales_df.groupby("sku_id")["revenue"].sum() * 0.40  # assume 40% COGS ratio
    else:
        cogs = pd.Series(dtype=float)

    # Average inventory value
    if "quantity_on_hand" in inventory_df.columns:
        avg_inv = inventory_df.groupby("sku_id")["quantity_on_hand"].mean()
    else:
        avg_inv = pd.Series(dtype=float)

    result = pd.DataFrame({"cogs": cogs, "avg_inventory": avg_inv}).dropna()
    result["inventory_turnover"] = result["cogs"] / result["avg_inventory"].replace(0, np.nan)
    result["inventory_turnover"] = result["inventory_turnover"].fillna(0)
    return result.reset_index()


def stockout_rate(inventory_df: pd.DataFrame, sales_df: pd.DataFrame) -> float:
    """% of SKU-days where quantity_on_hand == 0 and demand > 0."""
    if "stockout_risk" in sales_df.columns:
        total = len(sales_df)
        stockouts = sales_df["stockout_risk"].sum() if total > 0 else 0
        return float(stockouts / total * 100) if total > 0 else 0.0
    # Fallback: check inventory directly
    if "quantity_on_hand" in inventory_df.columns:
        zero_stock = (inventory_df["quantity_on_hand"] == 0).sum()
        return float(zero_stock / len(inventory_df) * 100) if len(inventory_df) > 0 else 0.0
    return 0.0


def overstock_cost(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    holding_cost_rate: float = 0.25,
) -> float:
    """Total annual cost of excess inventory = excess_units * cost_price * holding_cost_rate."""
    merged = inventory_df.merge(products_df[["sku_id", "cost_price"]], on="sku_id", how="left")
    # Excess = quantity above 180 days of supply threshold
    if "days_of_supply" in merged.columns and "quantity_on_hand" in merged.columns:
        excess_mask = merged["days_of_supply"] > 180
        excess_rows = merged[excess_mask].copy()
        if len(excess_rows) == 0:
            return 0.0
        # Estimate excess units as 50% of quantity_on_hand for overstock items
        excess_rows["excess_units"] = excess_rows["quantity_on_hand"] * 0.5
        excess_rows["annual_holding_cost"] = (
            excess_rows["excess_units"] * excess_rows["cost_price"].fillna(50) * holding_cost_rate
        )
        return float(excess_rows["annual_holding_cost"].sum())
    return 0.0


def gross_margin_return_on_investment(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """GMROI per SKU = gross_margin / avg_inventory_cost. Target > 2.0."""
    merged_sales = sales_df.merge(
        products_df[["sku_id", "cost_price", "retail_price"]], on="sku_id", how="left"
    )
    if "quantity_sold" in merged_sales.columns:
        merged_sales["gross_margin"] = merged_sales["quantity_sold"] * (
            merged_sales["unit_price"].fillna(merged_sales["retail_price"])
            - merged_sales["cost_price"].fillna(0)
        )
        gm_by_sku = merged_sales.groupby("sku_id")["gross_margin"].sum()
    else:
        gm_by_sku = pd.Series(dtype=float)

    avg_inv_cost = inventory_df.merge(
        products_df[["sku_id", "cost_price"]], on="sku_id", how="left"
    )
    avg_inv_cost["inv_cost"] = avg_inv_cost["quantity_on_hand"] * avg_inv_cost["cost_price"].fillna(50)
    inv_cost_by_sku = avg_inv_cost.groupby("sku_id")["inv_cost"].mean()

    result = pd.DataFrame({"gross_margin": gm_by_sku, "avg_inv_cost": inv_cost_by_sku}).dropna()
    result["gmroi"] = result["gross_margin"] / result["avg_inv_cost"].replace(0, np.nan)
    result["gmroi"] = result["gmroi"].fillna(0).clip(0, 50)
    result["meets_target"] = result["gmroi"] >= 2.0
    return result.reset_index()


if __name__ == "__main__":
    y_t = np.array([10, 20, 30, 40, 50])
    y_p = np.array([11, 19, 31, 38, 52])
    print(f"MAPE: {mape(y_t, y_p):.2f}%")
    print(f"RMSE: {rmse(y_t, y_p):.4f}")
    print(f"MAE:  {mae(y_t, y_p):.4f}")
    print(f"R²:   {r2(y_t, y_p):.4f}")
