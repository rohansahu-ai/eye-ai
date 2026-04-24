"""
Generates ranked buy recommendations, clearance alerts,
and portfolio optimization actions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from utils.logger import get_logger


class RecommendationEngine:
    """
    Generates ranked buy recommendations, clearance alerts,
    and portfolio optimization actions.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.logger = get_logger(__name__)

    def generate_buy_recommendations(
        self,
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        product_supplier_df: pd.DataFrame,
        similarity_index,
        products_df: pd.DataFrame,
        sales_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each SKU generate urgency-ranked buy recommendations.
        """
        # Projected demand (next 90d)
        if "forecasted_demand" in forecast_df.columns:
            projected = (
                forecast_df.groupby("sku_id")["forecasted_demand"]
                .sum()
                .rename("projected_demand_90d")
                .reset_index()
            )
        else:
            projected = pd.DataFrame(columns=["sku_id", "projected_demand_90d"])

        # Current stock
        inv_agg = inventory_df.groupby("sku_id").agg(
            quantity_on_hand=("quantity_on_hand", "sum"),
            quantity_on_order=("quantity_on_order", "sum"),
            days_of_supply=("days_of_supply", "mean"),
        ).reset_index()
        inv_agg["current_stock"] = inv_agg["quantity_on_hand"] + inv_agg["quantity_on_order"]

        df = inv_agg.merge(projected, on="sku_id", how="left")

        # Fallback projected demand
        sales_df_c = sales_df.copy()
        sales_df_c["date"] = pd.to_datetime(sales_df_c["date"])
        avg_daily = (
            sales_df_c.groupby("sku_id")["quantity_sold"].sum()
            / max(1, (sales_df_c["date"].max() - sales_df_c["date"].min()).days)
        )
        df["avg_daily_sales"] = df["sku_id"].map(avg_daily).fillna(0.1)
        df["projected_demand_90d"] = df["projected_demand_90d"].fillna(df["avg_daily_sales"] * 90)

        # Gap
        df["gap"] = (df["projected_demand_90d"] - df["current_stock"]).clip(0)

        # EOQ
        ordering_cost = 150.0
        unit_cost_base = 80.0
        annual_demand = df["avg_daily_sales"] * 365
        holding_cost = 0.25 * unit_cost_base
        df["eoq"] = np.sqrt(2 * annual_demand * ordering_cost / max(holding_cost, 0.01)).clip(1).round(0).astype(int)
        df["recommended_qty"] = np.maximum(df["gap"], df["eoq"]).round(0).astype(int)

        # Velocity score
        recent_sales = sales_df_c[sales_df_c["date"] >= sales_df_c["date"].max() - pd.Timedelta(days=30)]
        vel = recent_sales.groupby("sku_id")["quantity_sold"].sum().rename("vel30d")
        df = df.merge(vel, on="sku_id", how="left")
        df["vel30d"] = df["vel30d"].fillna(0)
        max_vel = df["vel30d"].max()
        df["velocity_score"] = (df["vel30d"] / max_vel.clip(0.001)).clip(0, 1)

        # Supplier info
        sup = suppliers_df[["supplier_id", "reliability_score", "lead_time_days"]].copy()
        ps = product_supplier_df[["sku_id", "supplier_id"]].drop_duplicates("sku_id")
        df = df.merge(ps, on="sku_id", how="left")
        df = df.merge(sup, on="supplier_id", how="left")
        df["reliability_score"] = df["reliability_score"].fillna(0.75)
        df["lead_time_days"] = df["lead_time_days"].fillna(60)

        # Supplier risk band
        from modules.supply_chain import SupplyChainIntelligence
        sci = SupplyChainIntelligence(self.config)
        scored_sup = sci.supplier_risk_scores(suppliers_df)[["supplier_id", "risk_band"]]
        df = df.merge(scored_sup, on="supplier_id", how="left")
        df["supplier_risk_band"] = df["risk_band"].fillna("unknown")

        # Urgency score
        df["urgency_score"] = (
            0.4 * (1 - (df["days_of_supply"] / 90).clip(0, 1))
            + 0.3 * df["velocity_score"]
            + 0.2 * (df["gap"] / df["projected_demand_90d"].clip(0.001)).clip(0, 1)
            + 0.1 * (1 - df["reliability_score"])
        ).clip(0, 1).round(4)

        urgency_bands = pd.cut(
            df["urgency_score"],
            bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
            labels=["low", "medium", "high", "critical"],
        )
        df["urgency_band"] = urgency_bands

        # Product metadata
        meta = products_df[["sku_id", "name", "frame_shape", "material", "color",
                             "price_point", "cost_price", "retail_price"]].copy()
        df = df.merge(meta, on="sku_id", how="left")

        # Financial
        df["cost_price"] = df["cost_price"].fillna(50)
        df["retail_price"] = df["retail_price"].fillna(150)
        df["estimated_cost_usd"] = (df["recommended_qty"] * df["cost_price"]).round(2)

        # Sell-through rate estimate from velocity percentile
        vel_pct = df["velocity_score"].rank(pct=True)
        df["sell_through_rate"] = (0.50 + 0.40 * vel_pct).clip(0.30, 0.95)
        df["estimated_margin_usd"] = (
            df["recommended_qty"]
            * (df["retail_price"] - df["cost_price"])
            * df["sell_through_rate"]
        ).round(2)
        df["margin_pct"] = (
            (df["retail_price"] - df["cost_price"]) / df["retail_price"].replace(0, np.nan) * 100
        ).round(1).fillna(0)

        # Demand signal drivers
        def _drivers(row: pd.Series) -> list:
            signals = []
            if row["velocity_score"] > 0.7:
                signals.append("high_velocity")
            if row["days_of_supply"] < 14:
                signals.append("critical_stock")
            if row["gap"] > 50:
                signals.append("demand_gap")
            if row["reliability_score"] < 0.6:
                signals.append("supplier_risk")
            return signals[:3] or ["standard_replenishment"]

        df["demand_signal_drivers"] = df.apply(_drivers, axis=1)

        # Similar alternatives
        def _similar(sku: str) -> list:
            try:
                similar = similarity_index.get_similar_skus(sku, top_n=3)
                if len(similar) > 0:
                    return similar["similar_sku_id"].tolist()
            except Exception:
                pass
            return []

        df["similar_alternatives"] = df["sku_id"].apply(_similar)
        df["confidence_score"] = (
            0.5 + 0.3 * df["velocity_score"] + 0.2 * (1 - (df["days_of_supply"] / 180).clip(0, 1))
        ).clip(0, 1).round(4)

        out_cols = [
            "sku_id", "name", "frame_shape", "material", "color", "price_point",
            "current_stock", "projected_demand_90d", "gap", "recommended_qty",
            "urgency_score", "urgency_band", "estimated_cost_usd", "estimated_margin_usd",
            "margin_pct", "supplier_id", "supplier_risk_band", "lead_time_days",
            "demand_signal_drivers", "similar_alternatives", "confidence_score",
        ]
        available = [c for c in out_cols if c in df.columns]
        result = df[available].rename(columns={"supplier_id": "preferred_supplier"})
        return result.sort_values("urgency_score", ascending=False).reset_index(drop=True)

    def clearance_alerts(
        self,
        sales_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        products_df: pd.DataFrame,
        velocity_df: pd.DataFrame,
        days_overstock_threshold: int = 120,
    ) -> pd.DataFrame:
        """
        Flag SKUs where days_of_supply > threshold AND momentum == 'decelerating'.
        """
        inv_agg = inventory_df.groupby("sku_id").agg(
            days_of_supply=("days_of_supply", "mean"),
            quantity_on_hand=("quantity_on_hand", "sum"),
        ).reset_index()

        overstocked = inv_agg[inv_agg["days_of_supply"] > days_overstock_threshold].copy()

        if "momentum" in velocity_df.columns:
            decel = velocity_df[velocity_df["momentum"] == "decelerating"][["sku_id"]]
            overstocked = overstocked.merge(decel, on="sku_id", how="inner")

        if len(overstocked) == 0:
            return pd.DataFrame(columns=[
                "sku_id", "sku_name", "days_of_supply", "quantity_on_hand",
                "carrying_cost_usd", "recommended_markdown_pct", "estimated_recovery_usd", "action"
            ])

        meta = products_df[["sku_id", "name", "cost_price", "retail_price"]].copy()
        overstocked = overstocked.merge(meta, on="sku_id", how="left")
        overstocked["cost_price"] = overstocked["cost_price"].fillna(50)
        overstocked["retail_price"] = overstocked["retail_price"].fillna(150)

        # Carrying cost = 25% * cost_price * qty * (days_overstock/365)
        overstocked["carrying_cost_usd"] = (
            0.25 * overstocked["cost_price"] * overstocked["quantity_on_hand"]
            * (overstocked["days_of_supply"] / 365)
        ).round(2)

        def _markdown(dos: float) -> tuple[float, str]:
            if dos < 180:
                return 15.0, "markdown"
            elif dos < 270:
                return 25.0, "markdown"
            else:
                return 40.0, "liquidate"

        md_pcts, actions = zip(*overstocked["days_of_supply"].apply(_markdown))
        overstocked["recommended_markdown_pct"] = list(md_pcts)
        overstocked["action"] = list(actions)
        overstocked["estimated_recovery_usd"] = (
            overstocked["quantity_on_hand"]
            * overstocked["retail_price"]
            * (1 - overstocked["recommended_markdown_pct"] / 100)
            * 0.7  # estimated sell-through after markdown
        ).round(2)

        return overstocked.rename(columns={"name": "sku_name"})[
            ["sku_id", "sku_name", "days_of_supply", "quantity_on_hand",
             "carrying_cost_usd", "recommended_markdown_pct", "estimated_recovery_usd", "action"]
        ].sort_values("days_of_supply", ascending=False).reset_index(drop=True)

    def allocation_optimizer(
        self,
        inventory_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For limited stock SKUs, distribute available stock to stores
        proportional to local forecast demand share.
        """
        # Find SKUs where total qty < reorder_point
        inv = inventory_df.copy()
        inv_total = inv.groupby("sku_id").agg(
            total_qty=("quantity_on_hand", "sum"),
            reorder_point=("reorder_point", "mean"),
        ).reset_index()
        limited = inv_total[inv_total["total_qty"] < inv_total["reorder_point"]]

        if len(limited) == 0:
            return pd.DataFrame(columns=["sku_id", "from_store_id", "to_store_id", "transfer_qty", "rationale"])

        # For simplicity, use avg daily demand as store demand share
        if "forecasted_demand" in forecast_df.columns:
            store_demand = forecast_df.groupby("sku_id")["forecasted_demand"].sum()
        else:
            store_demand = pd.Series(dtype=float)

        rows = []
        stores = inv["store_id"].unique()

        for _, lim_row in limited.iterrows():
            sku = lim_row["sku_id"]
            # Find the store with most stock (source) and least (destination)
            sku_inv = inv[inv["sku_id"] == sku][["store_id", "quantity_on_hand"]].copy()
            if len(sku_inv) < 2:
                continue
            max_store = sku_inv.loc[sku_inv["quantity_on_hand"].idxmax(), "store_id"]
            min_store = sku_inv.loc[sku_inv["quantity_on_hand"].idxmin(), "store_id"]
            max_qty = sku_inv["quantity_on_hand"].max()
            transfer = max(1, int(max_qty * 0.25))  # transfer 25% of max stock

            rows.append({
                "sku_id": sku,
                "from_store_id": max_store,
                "to_store_id": min_store,
                "transfer_qty": transfer,
                "rationale": f"Rebalance stock: {max_store} has surplus, {min_store} is below reorder point",
            })

        return pd.DataFrame(rows)

    def portfolio_summary(
        self,
        recs_df: pd.DataFrame,
        clearance_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> dict:
        """Returns overall portfolio summary dict."""
        total_buy_budget = float(recs_df["estimated_cost_usd"].sum()) if "estimated_cost_usd" in recs_df else 0.0
        total_skus = len(recs_df)
        critical = int((recs_df["urgency_band"] == "critical").sum()) if "urgency_band" in recs_df else 0
        high_priority = int((recs_df["urgency_band"] == "high").sum()) if "urgency_band" in recs_df else 0

        inv_agg = inventory_df.groupby("sku_id")["days_of_supply"].mean()
        critical_stock_skus = inv_agg[inv_agg < 7]
        meta = products_df[["sku_id", "retail_price"]].set_index("sku_id")
        avg_price = float(meta["retail_price"].mean()) if len(meta) > 0 else 100.0
        revenue_at_risk = float(len(critical_stock_skus) * avg_price * 30)  # 30d risk window

        overstock_cost = 0.0
        overstock_sku_count = 0
        capital_freed = 0.0
        if len(clearance_df) > 0:
            overstock_sku_count = len(clearance_df)
            if "carrying_cost_usd" in clearance_df.columns:
                overstock_cost = float(clearance_df["carrying_cost_usd"].sum())
            if "estimated_recovery_usd" in clearance_df.columns:
                capital_freed = float(clearance_df["estimated_recovery_usd"].sum())

        est_margin = float(recs_df["estimated_margin_usd"].sum()) if "estimated_margin_usd" in recs_df else 0.0

        if len(recs_df) > 0:
            _opps = recs_df.copy()
            _opps["urgency_score"] = pd.to_numeric(_opps["urgency_score"], errors="coerce").fillna(0)
            top5_opps = _opps.nlargest(min(5, len(_opps)), "urgency_score")["sku_id"].tolist()
        else:
            top5_opps = []
        if len(clearance_df) > 0:
            _clr = clearance_df.copy()
            _clr["days_of_supply"] = pd.to_numeric(_clr["days_of_supply"], errors="coerce").fillna(0)
            top5_risks = _clr.nlargest(min(5, len(_clr)), "days_of_supply")["sku_id"].tolist()
        else:
            top5_risks = []

        return {
            "total_buy_budget_usd": round(total_buy_budget, 2),
            "total_skus_to_buy": total_skus,
            "critical_skus": critical,
            "high_priority_skus": high_priority,
            "projected_revenue_at_risk_usd": round(revenue_at_risk, 2),
            "overstock_cost_usd": round(overstock_cost, 2),
            "overstock_sku_count": overstock_sku_count,
            "estimated_margin_improvement_usd": round(est_margin, 2),
            "capital_freed_if_clearance_usd": round(capital_freed, 2),
            "top_5_opportunities": top5_opps,
            "top_5_risks": top5_risks,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_synthetic_data import (
        generate_products, generate_suppliers, generate_sales, generate_inventory
    )
    from modules.similarity_index import SimilarityIndex
    from modules.customer_signals import CustomerSignals

    products = generate_products(50)
    suppliers = generate_suppliers(10)
    sales = generate_sales(products, 5000)
    inventory = generate_inventory(products, sales)

    si = SimilarityIndex().fit(products)
    cs = CustomerSignals()
    vel = cs.sales_velocity(sales)

    reco = RecommendationEngine()
    recs = reco.generate_buy_recommendations(
        pd.DataFrame(), inventory, suppliers, products[["sku_id", "supplier_id"]],
        si, products, sales
    )
    print(f"Buy recommendations: {len(recs)}")
    print(recs[["sku_id", "urgency_score", "urgency_band", "recommended_qty"]].head(5))

    clearance = reco.clearance_alerts(sales, inventory, products, vel)
    print(f"\nClearance alerts: {len(clearance)}")
    print(clearance.head(3).to_string())

    summary = reco.portfolio_summary(recs, clearance, inventory, products)
    print("\nPortfolio summary:", summary)
