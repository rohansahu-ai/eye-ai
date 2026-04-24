"""
Synthetic data generator for the AI-Powered Eyewear Buying Platform.
Deterministic: seed=42. Run to regenerate all CSVs in DATA_DIR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings

RNG = np.random.default_rng(42)
SETTINGS = get_settings()


# ─── helpers ──────────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, name: str) -> None:
    out_dir = Path(SETTINGS.DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows → {path}")


# ─── products.csv ─────────────────────────────────────────────────────────────
def generate_products(n: int = 300) -> pd.DataFrame:
    price_bands = {
        "budget":  (50, 120),
        "mid":     (120, 250),
        "premium": (250, 500),
        "luxury":  (500, 1200),
    }
    price_band_labels = ["budget", "mid", "premium", "luxury"]
    price_band_weights = [0.30, 0.35, 0.25, 0.10]

    frame_shapes = ["aviator", "wayfarer", "round", "cat_eye", "rectangle", "square", "oval", "oversized"]
    materials = ["acetate", "titanium", "stainless_steel", "tr90", "wood", "aluminium"]
    colors = ["black", "tortoise", "gold", "silver", "rose_gold", "navy", "brown", "green", "clear", "red"]
    genders = ["unisex", "men", "women"]
    seasons = ["SS23", "AW23", "SS24", "AW24"]
    categories = ["sunglasses", "optical"]

    sku_ids = [f"SKU{str(i).zfill(4)}" for i in range(1, n + 1)]
    price_points = RNG.choice(price_band_labels, size=n, p=price_band_weights)
    retail_prices = np.array([
        round(RNG.uniform(*price_bands[pp]), 2) for pp in price_points
    ])
    cost_prices = np.round(retail_prices * RNG.uniform(0.25, 0.45, size=n), 2)

    supplier_ids = [f"SUP{str(RNG.integers(1, 26)).zfill(3)}" for _ in range(n)]
    launch_dates = pd.to_datetime(
        RNG.choice(pd.date_range("2022-01-01", "2024-06-01", freq="W").strftime("%Y-%m-%d"), size=n)
    )

    df = pd.DataFrame({
        "sku_id": sku_ids,
        "name": [
            f"{RNG.choice(materials).title()} {RNG.choice(frame_shapes).replace('_', ' ').title()} {i}"
            for i in range(1, n + 1)
        ],
        "frame_shape": RNG.choice(frame_shapes, size=n),
        "material":    RNG.choice(materials, size=n),
        "color":       RNG.choice(colors, size=n),
        "price_point": price_points,
        "gender":      RNG.choice(genders, size=n),
        "season_launched": RNG.choice(seasons, size=n),
        "is_active":   RNG.choice([True, False], size=n, p=[0.70, 0.30]),
        "supplier_id": supplier_ids,
        "cost_price":  cost_prices,
        "retail_price": retail_prices,
        "category":    RNG.choice(categories, size=n, p=[0.55, 0.45]),
        "launch_date": launch_dates.strftime("%Y-%m-%d"),
    })
    return df


# ─── suppliers.csv ────────────────────────────────────────────────────────────
def generate_suppliers(n: int = 25) -> pd.DataFrame:
    countries = ["China", "Italy", "Germany", "Japan", "South Korea", "France", "USA", "Taiwan"]
    payment_terms = [30, 45, 60, 90, 120]

    supplier_ids = [f"SUP{str(i).zfill(3)}" for i in range(1, n + 1)]

    # Lead time skewed toward 45-90
    lead_times = np.clip(
        RNG.normal(loc=65, scale=20, size=n).astype(int), 30, 120
    )

    # Reliability: beta distribution, mean ~0.78
    reliability = np.clip(RNG.beta(a=4, b=1.1, size=n), 0.0, 1.0)
    # Flag 5 as high-risk
    high_risk_idx = RNG.choice(n, size=5, replace=False)
    reliability[high_risk_idx] = RNG.uniform(0.2, 0.49, size=5)

    capacity = RNG.uniform(0.40, 0.95, size=n)
    on_time = np.clip(reliability * RNG.uniform(0.90, 1.10, size=n), 0.0, 1.0)

    df = pd.DataFrame({
        "supplier_id": supplier_ids,
        "name": [f"Supplier {chr(65 + (i % 26))}{i}" for i in range(n)],
        "country": RNG.choice(countries, size=n),
        "lead_time_days": lead_times,
        "reliability_score": np.round(reliability, 3),
        "capacity_utilization": np.round(capacity, 3),
        "avg_unit_cost_multiplier": np.round(RNG.uniform(0.85, 1.20, size=n), 3),
        "on_time_delivery_rate": np.round(on_time, 3),
        "min_order_qty": RNG.choice([50, 100, 150, 200, 300, 500], size=n),
        "payment_terms_days": RNG.choice(payment_terms, size=n),
    })
    return df


# ─── sales_transactions.csv ───────────────────────────────────────────────────
def generate_sales(products_df: pd.DataFrame, target_n: int = 80_000) -> pd.DataFrame:
    """
    Generate sales with proper per-SKU time-series patterns.
    Uses an AR(1) Poisson process with monthly/DOW seasonality so that
    lag features carry real predictive signal (per-SKU ACF >> 0).
    """
    date_range = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    channels = ["in_store", "online", "wholesale"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders_c = ["male", "female", "other"]
    regions = ["North", "South", "East", "West", "Online"]
    store_ids = [f"STORE{str(i).zfill(2)}" for i in range(1, 16)]

    active_skus = products_df[products_df["is_active"]]["sku_id"].values
    hero_count = max(1, int(len(active_skus) * 0.20))
    hero_skus: set = set(RNG.choice(active_skus, size=hero_count, replace=False))

    sku_to_category = products_df.set_index("sku_id")["category"].to_dict()
    sku_to_price    = products_df.set_index("sku_id")["retail_price"].to_dict()

    # Monthly seasonality (relative multipliers, mean ≈ 1.0)
    month_mult = {1: 0.72, 2: 0.68, 3: 0.80, 4: 0.78, 5: 0.82,
                  6: 1.35, 7: 1.42, 8: 1.38, 9: 0.78, 10: 0.88,
                  11: 1.28, 12: 1.40}
    sunglass_extra = {6: 0.70, 7: 0.80, 8: 0.70}   # extra for sunglasses in summer
    dow_mult       = {0: 0.88, 1: 0.86, 2: 0.87, 3: 0.90, 4: 0.96,
                      5: 1.28, 6: 1.32}

    n_days   = len(date_range)
    n_active = len(active_skus)
    # Expected demand per (sku, day):
    #   hero_frac=0.20 → avg_rate_factor = 0.20*3 + 0.80*1 = 1.40
    #   avg_seasonal ≈ 1.0 by construction
    #   AR(1) adds ~15 % extra volume on non-zero days
    base_rate = target_n / (n_active * n_days * 1.40 * 1.15)

    rows: list = []
    txn_id = 1

    for sku in active_skus:
        cat        = sku_to_category.get(sku, "optical")
        base_price = sku_to_price.get(sku, 150.0)
        is_hero    = sku in hero_skus
        rate_factor = 3.0 if is_hero else 1.0

        # AR(1) coefficient – creates week-level autocorrelation
        ar1_coef = RNG.uniform(0.30, 0.55)
        prev_demand = 0.0

        for date in date_range:
            month = date.month
            dow   = date.dayofweek
            year  = date.year

            m_mult = month_mult[month]
            if cat == "sunglasses" and month in sunglass_extra:
                m_mult += sunglass_extra[month]

            lam = max(0.01,
                      base_rate * rate_factor * m_mult * dow_mult[dow]
                      + ar1_coef * prev_demand)
            daily_qty = int(min(RNG.poisson(lam=lam), 20))
            prev_demand = float(daily_qty)

            if daily_qty == 0:
                continue

            is_stockout = int(RNG.random() < 0.08)
            if is_stockout:
                daily_qty = 1

            price   = round(base_price * RNG.uniform(0.92, 1.05), 2)
            online_p = 0.20 if year == 2023 else 0.23
            channel = RNG.choice(channels,
                                 p=[1 - online_p - 0.10, online_p, 0.10])

            rows.append({
                "transaction_id":     f"TXN{str(txn_id).zfill(7)}",
                "sku_id":             sku,
                "date":               date.strftime("%Y-%m-%d"),
                "store_id":           str(RNG.choice(store_ids)),
                "quantity_sold":      daily_qty,
                "unit_price":         price,
                "channel":            channel,
                "customer_age_group": str(RNG.choice(age_groups)),
                "customer_gender":    str(RNG.choice(genders_c)),
                "region":             str(RNG.choice(regions)),
                "stockout_risk":      is_stockout,
            })
            txn_id += 1

    df = pd.DataFrame(rows)
    print(f"    Generated {len(df):,} transactions from AR(1) time-series simulation")
    return df


# ─── inventory.csv ────────────────────────────────────────────────────────────
def generate_inventory(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    store_ids = [f"STORE{str(i).zfill(2)}" for i in range(1, 16)]
    active_skus = products_df[products_df["is_active"]]["sku_id"].values

    # Compute avg daily sales per SKU
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    total_days = (sales_df["date"].max() - sales_df["date"].min()).days + 1
    avg_daily = (
        sales_df.groupby("sku_id")["quantity_sold"].sum() / total_days
    ).to_dict()

    rows = []
    for sku in active_skus:
        daily = avg_daily.get(sku, 0.5)
        for store in store_ids:
            # Base days of supply
            dos_roll = RNG.random()
            if dos_roll < 0.15:
                dos = RNG.integers(1, 14)  # critical
            elif dos_roll < 0.25:
                dos = RNG.integers(181, 365)  # overstock
            else:
                dos = RNG.integers(14, 181)

            qty_on_hand = max(0, int(daily * dos))
            reorder_point = max(1, int(daily * 45))
            qty_on_order = int(daily * 30) if dos < 21 else 0
            last_restocked = pd.Timestamp("2024-12-31") - pd.Timedelta(days=int(RNG.integers(1, 90)))

            rows.append({
                "sku_id": sku,
                "store_id": store,
                "quantity_on_hand": qty_on_hand,
                "reorder_point": reorder_point,
                "quantity_on_order": qty_on_order,
                "days_of_supply": int(dos),
                "last_restocked_date": last_restocked.strftime("%Y-%m-%d"),
                "warehouse_location": f"WH-{RNG.choice(['A', 'B', 'C'])}{RNG.integers(1, 20):02d}",
            })

    return pd.DataFrame(rows)


# ─── returns.csv ──────────────────────────────────────────────────────────────
def generate_returns(sales_df: pd.DataFrame, products_df: pd.DataFrame, n: int = 8_000) -> pd.DataFrame:
    return_reasons = ["wrong_size", "quality_issue", "style_preference", "defective", "wrong_item"]
    reason_weights = [0.28, 0.22, 0.25, 0.15, 0.10]

    # Problematic SKUs (25%+ return rate)
    active_skus = products_df[products_df["is_active"]]["sku_id"].values
    problematic_skus = set(RNG.choice(active_skus, size=20, replace=False))

    # Select return transactions from sales
    sampled = sales_df.sample(n=n, random_state=42).copy()
    sampled["return_date"] = (
        pd.to_datetime(sampled["date"])
        + pd.to_timedelta(RNG.integers(1, 30, size=n), unit="D")
    ).dt.strftime("%Y-%m-%d")

    reasons = []
    for sku in sampled["sku_id"]:
        if sku in problematic_skus:
            # Problematic SKUs: elevated quality/defect
            r = RNG.choice(return_reasons, p=[0.15, 0.40, 0.20, 0.20, 0.05])
        else:
            r = RNG.choice(return_reasons, p=reason_weights)
        reasons.append(r)

    df = pd.DataFrame({
        "return_id": [f"RET{str(i).zfill(6)}" for i in range(1, n + 1)],
        "sku_id": sampled["sku_id"].values,
        "transaction_id": sampled["transaction_id"].values,
        "return_date": sampled["return_date"].values,
        "return_reason": reasons,
        "refund_amount": (sampled["unit_price"].values * RNG.uniform(0.85, 1.0, size=n)).round(2),
        "restocked": RNG.choice([True, False], size=n, p=[0.65, 0.35]),
    })
    return df


# ─── social_signals.csv ───────────────────────────────────────────────────────
def generate_social_signals(products_df: pd.DataFrame) -> pd.DataFrame:
    active_skus = products_df[products_df["is_active"]]["sku_id"].values
    platforms = ["instagram", "tiktok", "pinterest", "youtube"]
    weeks = pd.date_range("2023-01-02", periods=104, freq="W")

    # 15 viral SKUs
    viral_skus = set(RNG.choice(active_skus, size=15, replace=False))

    rows = []
    for sku in active_skus:
        base_mentions = int(RNG.integers(50, 500))
        base_sentiment = float(RNG.uniform(0.40, 0.85))
        is_viral = sku in viral_skus

        for w_idx, week in enumerate(weeks):
            for plat in platforms:
                mentions = int(RNG.poisson(lam=base_mentions))

                # Viral spike event (sudden 10x for 3 weeks)
                if is_viral and 20 <= w_idx <= 23:
                    mentions = int(mentions * RNG.uniform(8, 12))

                # Platform multipliers
                plat_mult = {"instagram": 1.5, "tiktok": 2.0, "pinterest": 0.8, "youtube": 0.6}
                mentions = int(mentions * plat_mult.get(plat, 1.0))

                sentiment = float(np.clip(
                    base_sentiment + RNG.normal(0, 0.1), -1.0, 1.0
                ))
                trend_index = min(100, max(0, int(mentions / 20)))
                influencer = int(RNG.poisson(lam=2)) if mentions > 200 else 0

                rows.append({
                    "sku_id": sku,
                    "week_start": week.strftime("%Y-%m-%d"),
                    "platform": plat,
                    "mention_count": mentions,
                    "sentiment_score": round(sentiment, 4),
                    "trend_index": trend_index,
                    "hashtag_reach": int(mentions * RNG.uniform(10, 50)),
                    "influencer_mentions": influencer,
                })

    return pd.DataFrame(rows)


# ─── main ─────────────────────────────────────────────────────────────────────
def generate_all() -> None:
    print("Generating synthetic data (seed=42)...")

    print("→ products.csv")
    products = generate_products(300)
    _save(products, "products.csv")

    print("→ suppliers.csv")
    suppliers = generate_suppliers(25)
    _save(suppliers, "suppliers.csv")

    print("→ sales_transactions.csv")
    sales = generate_sales(products, 80_000)
    _save(sales, "sales_transactions.csv")

    print("→ inventory.csv")
    inventory = generate_inventory(products, sales)
    _save(inventory, "inventory.csv")

    print("→ returns.csv")
    returns = generate_returns(sales, products, 8_000)
    _save(returns, "returns.csv")

    print("→ social_signals.csv")
    social = generate_social_signals(products)
    _save(social, "social_signals.csv")

    print("\nAll files generated successfully.")


if __name__ == "__main__":
    generate_all()
