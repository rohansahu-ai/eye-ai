"""
Supply-side constraint analysis and reorder recommendation engine.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from utils.logger import get_logger


class SupplyChainIntelligence:
    """Supply-side constraint analysis and reorder recommendation engine."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.logger = get_logger(__name__)

    def supplier_risk_scores(self, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite risk = (1 - reliability_score)*0.40
                       + capacity_utilization*0.35
                       + (lead_time_days/120)*0.25
        Risk bands: low (<0.35), medium (0.35-0.60), high (>0.60)
        """
        df = suppliers_df.copy()
        df["risk_score"] = (
            (1 - df["reliability_score"]) * 0.40
            + df["capacity_utilization"] * 0.35
            + (df["lead_time_days"] / 120.0) * 0.25
        ).clip(0.0, 1.0).round(4)

        df["risk_band"] = pd.cut(
            df["risk_score"],
            bins=[-0.001, 0.35, 0.60, 1.001],
            labels=["low", "medium", "high"],
        )

        def _flags(row: pd.Series) -> list:
            flags = []
            if row["reliability_score"] < 0.5:
                flags.append("low_reliability")
            if row["capacity_utilization"] > 0.85:
                flags.append("high_capacity")
            if row["lead_time_days"] > 90:
                flags.append("long_lead_time")
            if row.get("on_time_delivery_rate", 1.0) < 0.70:
                flags.append("poor_on_time_delivery")
            return flags

        df["risk_flags"] = df.apply(_flags, axis=1)
        return df

    def reorder_recommendations(
        self,
        inventory_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        product_supplier_df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate urgency-ranked reorder recommendations using EOQ.
        """
        # Compute avg daily sales (rolling 30d)
        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        cutoff = sales_df["date"].max()
        recent = sales_df[sales_df["date"] >= cutoff - pd.Timedelta(days=30)]
        avg_daily = (
            recent.groupby("sku_id")["quantity_sold"].sum() / 30
        ).rename("avg_daily_sales")

        # Annual demand per SKU
        annual_daily = (
            sales_df.groupby("sku_id")["quantity_sold"].sum()
            / max(1, (sales_df["date"].max() - sales_df["date"].min()).days)
            * 365
        ).rename("annual_demand")

        # Aggregate inventory to SKU level (sum across stores)
        inv = inventory_df.groupby("sku_id").agg(
            quantity_on_hand=("quantity_on_hand", "sum"),
            quantity_on_order=("quantity_on_order", "sum"),
            days_of_supply=("days_of_supply", "mean"),
        ).reset_index()

        inv = inv.merge(avg_daily, on="sku_id", how="left")
        inv["avg_daily_sales"] = inv["avg_daily_sales"].fillna(0.1)
        inv = inv.merge(annual_daily, on="sku_id", how="left")
        inv["annual_demand"] = inv["annual_demand"].fillna(1.0)

        # Map supplier
        ps = product_supplier_df[["sku_id", "supplier_id"]].drop_duplicates()
        inv = inv.merge(ps, on="sku_id", how="left")

        sup_scored = self.supplier_risk_scores(suppliers_df)
        sup_info = sup_scored[
            ["supplier_id", "lead_time_days", "risk_band", "capacity_utilization", "avg_unit_cost_multiplier"]
        ].drop_duplicates("supplier_id")
        inv = inv.merge(sup_info, on="supplier_id", how="left")
        inv["lead_time_days"] = inv["lead_time_days"].fillna(60).astype(float)

        # Days until stockout
        inv["days_until_stockout"] = (
            inv["quantity_on_hand"] / inv["avg_daily_sales"].replace(0, 0.1)
        ).clip(0, 365)

        safety_stock_days = (
            inv["lead_time_days"] + self.config.REORDER_SAFETY_STOCK_DAYS
        )
        inv["trigger"] = inv["days_until_stockout"] < safety_stock_days

        # EOQ: sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
        ordering_cost = 150.0
        unit_cost_base = 80.0  # fallback unit cost
        inv["unit_cost"] = unit_cost_base * inv["avg_unit_cost_multiplier"].fillna(1.0)
        holding_cost = 0.25 * inv["unit_cost"]
        inv["eoq"] = np.sqrt(
            2 * inv["annual_demand"] * ordering_cost / holding_cost.replace(0, 1.0)
        ).clip(1).round(0).astype(int)

        # Gap vs forecast
        if forecast_df is not None and "forecasted_demand" in forecast_df.columns:
            projected = (
                forecast_df.groupby("sku_id")["forecasted_demand"].sum()
                .rename("projected_demand_90d")
                .reset_index()
            )
            inv = inv.merge(projected, on="sku_id", how="left")
            inv["projected_demand_90d"] = inv["projected_demand_90d"].fillna(
                inv["avg_daily_sales"] * 90
            )
        else:
            inv["projected_demand_90d"] = inv["avg_daily_sales"] * 90

        inv["recommended_order_qty"] = np.maximum(
            inv["projected_demand_90d"] - inv["quantity_on_hand"] - inv["quantity_on_order"],
            inv["eoq"],
        ).clip(0).round(0).astype(int)

        # Urgency score
        inv["urgency_score"] = (
            (1 - (inv["days_until_stockout"] / 90).clip(0, 1)) * 0.5
            + ((inv["avg_daily_sales"] / inv["avg_daily_sales"].max().clip(0.001)) * 0.3)
            + (inv["trigger"].astype(float) * 0.2)
        ).clip(0, 1).round(4)

        inv["estimated_cost"] = inv["recommended_order_qty"] * inv["unit_cost"]

        triggered = inv[inv["trigger"]].copy()
        triggered = triggered.sort_values("urgency_score", ascending=False)

        out_cols = [
            "sku_id", "days_until_stockout", "recommended_order_qty",
            "supplier_id", "risk_band", "lead_time_days",
            "estimated_cost", "urgency_score", "eoq",
        ]
        available = [c for c in out_cols if c in triggered.columns]
        result = triggered[available].rename(columns={"supplier_id": "preferred_supplier", "risk_band": "supplier_risk_band"})
        return result.reset_index(drop=True)

    def cost_opportunity_alerts(self, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Bulk discount candidates: capacity_utilization < 0.50
        Early payment discount: payment_terms_days > 60
        """
        rows = []
        for _, row in suppliers_df.iterrows():
            if row["capacity_utilization"] < 0.50:
                rows.append({
                    "supplier_id": row["supplier_id"],
                    "name": row.get("name", ""),
                    "alert_type": "bulk_discount",
                    "detail": f"Capacity at {row['capacity_utilization']*100:.0f}% — negotiate bulk pricing",
                    "estimated_savings_pct": round((0.50 - row["capacity_utilization"]) * 20, 1),
                })
            if row.get("payment_terms_days", 0) > 60:
                rows.append({
                    "supplier_id": row["supplier_id"],
                    "name": row.get("name", ""),
                    "alert_type": "early_payment_discount",
                    "detail": f"Payment terms at {row['payment_terms_days']} days — early payment likely yields 2-3%",
                    "estimated_savings_pct": 2.5,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["supplier_id", "name", "alert_type", "detail", "estimated_savings_pct"]
        )

    def material_risk_alerts(
        self, suppliers_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cross-reference material type with supplier country.
        Flag geopolitical risk for titanium (China), acetate (Italy).
        """
        GEOPOLITICAL_RISK = {
            ("titanium", "China"): "high",
            ("acetate", "Italy"): "medium",
            ("stainless_steel", "China"): "medium",
            ("tr90", "China"): "low",
        }

        # Map supplier country to products
        sup_country = suppliers_df[["supplier_id", "country"]].drop_duplicates()
        merged = products_df.merge(sup_country, on="supplier_id", how="left")

        rows = []
        for (mat, country), level in GEOPOLITICAL_RISK.items():
            count = len(merged[(merged["material"] == mat) & (merged["country"] == country)])
            if count > 0:
                rows.append({
                    "material": mat,
                    "country": country,
                    "sku_count": count,
                    "risk_level": level,
                    "reason": f"{mat.title()} sourced from {country} — geopolitical concentration risk",
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["material", "country", "sku_count", "risk_level", "reason"]
        )

    def supply_chain_health_dashboard(
        self,
        suppliers_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        sales_df: pd.DataFrame,
    ) -> dict:
        """Returns summary health dict."""
        scored = self.supplier_risk_scores(suppliers_df)

        # SKUs with low stock
        inv_agg = inventory_df.groupby("sku_id")["days_of_supply"].mean()
        at_risk = int((inv_agg < 30).sum())
        critical = int((inv_agg < 7).sum())
        overstock = int((inv_agg > 180).sum())

        avg_lead = float(suppliers_df["lead_time_days"].mean())
        high_risk_count = int((scored["risk_band"] == "high").sum())
        capacity_slack = scored[scored["capacity_utilization"] < 0.50]["supplier_id"].tolist()
        on_time_avg = float(suppliers_df.get("on_time_delivery_rate", pd.Series([0.75])).mean())

        # Rough total reorder value
        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        avg_daily = (
            sales_df.groupby("sku_id")["quantity_sold"].sum()
            / max(1, (sales_df["date"].max() - sales_df["date"].min()).days)
        )
        reorder_qty = (avg_daily * 60).clip(0)  # 60d reorder horizon
        unit_cost = 80.0  # avg unit cost proxy
        total_reorder_value = float(reorder_qty.sum() * unit_cost)

        return {
            "at_risk_sku_count": at_risk,
            "critical_sku_count": critical,
            "avg_lead_time_days": round(avg_lead, 1),
            "high_risk_supplier_count": high_risk_count,
            "capacity_slack_suppliers": capacity_slack,
            "total_reorder_value_usd": round(total_reorder_value, 2),
            "on_time_delivery_rate_avg": round(on_time_avg, 3),
            "overstock_sku_count": overstock,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_synthetic_data import (
        generate_products, generate_suppliers, generate_sales, generate_inventory
    )

    products = generate_products(50)
    suppliers = generate_suppliers(10)
    sales = generate_sales(products, 5000)
    inventory = generate_inventory(products, sales)

    sci = SupplyChainIntelligence()
    scored = sci.supplier_risk_scores(suppliers)
    print(scored[["supplier_id", "risk_score", "risk_band"]].head(10))

    product_supplier = products[["sku_id", "supplier_id"]].copy()
    recs = sci.reorder_recommendations(inventory, sales, suppliers, product_supplier)
    print(f"\nReorder recommendations: {len(recs)} SKUs")
    print(recs.head(5).to_string())
