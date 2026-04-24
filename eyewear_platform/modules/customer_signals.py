"""
Real-time and predictive demand signal extraction from sales, returns, and social data.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from utils.logger import get_logger


class CustomerSignals:
    """
    Real-time and predictive demand signal extraction from sales,
    returns, and social data.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.logger = get_logger(__name__)

    def sales_velocity(
        self, sales_df: pd.DataFrame, window_days: int = 30
    ) -> pd.DataFrame:
        """
        Per SKU velocity metrics with momentum classification.
        """
        df = sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()

        period_current = (max_date - pd.Timedelta(days=window_days), max_date)
        period_prior = (
            max_date - pd.Timedelta(days=2 * window_days),
            max_date - pd.Timedelta(days=window_days + 1),
        )

        current = (
            df[df["date"].between(*period_current)]
            .groupby("sku_id")["quantity_sold"]
            .sum()
            .rename("current_velocity")
        )
        prior = (
            df[df["date"].between(*period_prior)]
            .groupby("sku_id")["quantity_sold"]
            .sum()
            .rename("prior_velocity")
        )

        result = pd.concat([current, prior], axis=1).fillna(0)
        result["velocity_change_pct"] = np.where(
            result["prior_velocity"] > 0,
            (result["current_velocity"] - result["prior_velocity"]) / result["prior_velocity"] * 100,
            0.0,
        )
        result["momentum"] = pd.cut(
            result["velocity_change_pct"],
            bins=[-np.inf, -20, 20, np.inf],
            labels=["decelerating", "stable", "accelerating"],
        )
        return result.reset_index()

    def return_rate_analysis(
        self,
        sales_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Per SKU: return_rate_pct, top_return_reason, is_problematic.
        """
        total_sold = sales_df.groupby("sku_id")["quantity_sold"].sum().rename("total_sold")
        total_returned = returns_df.groupby("sku_id").size().rename("total_returned")
        top_reason = (
            returns_df.groupby("sku_id")["return_reason"]
            .agg(lambda x: x.value_counts().index[0])
            .rename("top_return_reason")
        )
        quality_pct = (
            returns_df[returns_df["return_reason"] == "quality_issue"]
            .groupby("sku_id")
            .size()
            / returns_df.groupby("sku_id").size()
        ).rename("quality_issue_pct").fillna(0)

        result = pd.concat(
            [total_sold, total_returned, top_reason, quality_pct], axis=1
        ).reset_index()
        result["total_sold"] = result["total_sold"].fillna(0)
        result["total_returned"] = result["total_returned"].fillna(0)
        result["return_rate_pct"] = (
            result["total_returned"] / result["total_sold"].replace(0, np.nan) * 100
        ).fillna(0).round(2)
        result["is_problematic"] = result["return_rate_pct"] > 15
        result["quality_flag"] = result["quality_issue_pct"] > 0.40

        if products_df is not None and "supplier_id" in products_df.columns:
            result = result.merge(
                products_df[["sku_id", "supplier_id"]], on="sku_id", how="left"
            )

        return result.sort_values("return_rate_pct", ascending=False).reset_index(drop=True)

    def trending_skus(
        self,
        social_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Composite trend_score using social + velocity signals.
        """
        social_df = social_df.copy()
        social_df["week_start"] = pd.to_datetime(social_df["week_start"])
        max_week = social_df["week_start"].max()
        recent_social = social_df[social_df["week_start"] >= max_week - pd.Timedelta(weeks=4)]

        agg_social = recent_social.groupby("sku_id").agg(
            mention_count_4wk_avg=("mention_count", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            influencer_mentions=("influencer_mentions", "sum"),
        ).reset_index()

        vel = self.sales_velocity(sales_df)

        merged = agg_social.merge(vel[["sku_id", "velocity_change_pct"]], on="sku_id", how="left")
        merged["velocity_change_pct"] = merged["velocity_change_pct"].fillna(0)

        def _norm(s: pd.Series) -> pd.Series:
            mn, mx = s.min(), s.max()
            if mx == mn:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - mn) / (mx - mn)

        merged["trend_score"] = (
            0.40 * _norm(merged["mention_count_4wk_avg"])
            + 0.30 * _norm(merged["sentiment_score"])
            + 0.20 * _norm(merged["velocity_change_pct"])
            + 0.10 * _norm(merged["influencer_mentions"])
        ).round(4)

        # Platform breakdown
        platform_breakdown = (
            recent_social.groupby(["sku_id", "platform"])["mention_count"]
            .sum()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        merged["platform_breakdown"] = merged["sku_id"].map(platform_breakdown)

        return merged.nlargest(top_n, "trend_score").reset_index(drop=True)

    def demand_seasonality(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Monthly seasonality index per (category x frame_shape).
        Uses STL decomposition proxy (ratio to mean).
        """
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            STL = None

        df = sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year

        # We need product-level category/frame_shape
        monthly = df.groupby(["sku_id", "year", "month"])["quantity_sold"].sum().reset_index()

        rows = []
        for (sku), grp in monthly.groupby("sku_id"):
            ts = grp.set_index("month")["quantity_sold"]
            mean_val = ts.mean()
            if mean_val > 0:
                for month, val in ts.items():
                    rows.append({"sku_id": sku, "month": month, "seasonality_index": round(val / mean_val, 4)})

        result = pd.DataFrame(rows)
        # Aggregate over all SKUs to get "category x frame_shape" level
        if "sku_id" in result.columns:
            result = result.groupby("month")["seasonality_index"].mean().reset_index()
            result.columns = ["month", "seasonality_index"]

        return result

    def customer_segment_preferences(
        self, sales_df: pd.DataFrame, products_df: pd.DataFrame
    ) -> dict:
        """
        Returns nested dict of preference heatmaps.
        """
        merged = sales_df.merge(
            products_df[["sku_id", "frame_shape", "price_point", "category", "material"]],
            on="sku_id",
            how="left",
        )

        def _pivot_normalized(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
            pivot = df.pivot_table(
                index=row_col, columns=col_col,
                values="quantity_sold", aggfunc="sum", fill_value=0
            )
            row_sums = pivot.sum(axis=1)
            return pivot.div(row_sums, axis=0).round(4)

        return {
            "age_group_x_frame_shape": _pivot_normalized(
                merged, "customer_age_group", "frame_shape"
            ) if "customer_age_group" in merged.columns else pd.DataFrame(),
            "gender_x_price_point": _pivot_normalized(
                merged, "customer_gender", "price_point"
            ) if "customer_gender" in merged.columns else pd.DataFrame(),
            "channel_x_category": _pivot_normalized(
                merged, "channel", "category"
            ) if "channel" in merged.columns else pd.DataFrame(),
            "region_x_frame_shape": _pivot_normalized(
                merged, "region", "frame_shape"
            ) if "region" in merged.columns else pd.DataFrame(),
        }

    def wishlist_demand_proxy(
        self, sales_df: pd.DataFrame, inventory_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Proxy latent demand using: high velocity + low inventory signal.
        """
        vel = self.sales_velocity(sales_df)
        vel["velocity_rank"] = vel["current_velocity"].rank(pct=True)

        if inventory_df is not None:
            inv_agg = inventory_df.groupby("sku_id")["days_of_supply"].mean().reset_index()
            inv_agg["dos_norm"] = (
                inv_agg["days_of_supply"] / inv_agg["days_of_supply"].max().clip(1)
            ).clip(0.01, 1.0)
            merged = vel.merge(inv_agg[["sku_id", "dos_norm"]], on="sku_id", how="left")
            merged["dos_norm"] = merged["dos_norm"].fillna(0.5)
            merged["latent_demand_score"] = merged["velocity_rank"] * (1 / merged["dos_norm"])
        else:
            merged = vel.copy()
            merged["latent_demand_score"] = merged["velocity_rank"]

        return merged.sort_values("latent_demand_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_synthetic_data import (
        generate_products, generate_sales, generate_returns
    )

    products = generate_products(50)
    sales = generate_sales(products, 5000)
    returns = generate_returns(sales, products, 500)

    cs = CustomerSignals()
    vel = cs.sales_velocity(sales)
    print("Velocity sample:")
    print(vel.head(5).to_string())

    rr = cs.return_rate_analysis(sales, returns, products)
    print("\nReturn rate (top 5):")
    print(rr.head(5).to_string())
