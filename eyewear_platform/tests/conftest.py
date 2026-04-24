"""
Shared pytest fixtures for the EyeAI Buying Platform test suite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── tiny deterministic DataFrames ────────────────────────────────────────────

@pytest.fixture(scope="session")
def products_df():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "sku_id": [f"SKU{i:03d}" for i in range(1, n+1)],
        "name": [f"Product {i}" for i in range(1, n+1)],
        "frame_shape": rng.choice(["aviator", "round", "cat_eye", "rectangle"], size=n),
        "material": rng.choice(["acetate", "titanium", "tr90"], size=n),
        "color": rng.choice(["black", "gold", "tortoise"], size=n),
        "price_point": rng.choice(["budget", "mid", "premium", "luxury"], size=n),
        "gender": rng.choice(["men", "women", "unisex"], size=n),
        "season_launched": rng.choice(["SS23", "AW23"], size=n),
        "is_active": rng.choice([True, False], size=n, p=[0.7, 0.3]),
        "supplier_id": [f"SUP{rng.integers(1, 6):02d}" for _ in range(n)],
        "cost_price": rng.uniform(30, 200, n).round(2),
        "retail_price": rng.uniform(80, 600, n).round(2),
        "category": rng.choice(["sunglasses", "optical"], size=n),
        "launch_date": ["2023-01-01"] * n,
    })


@pytest.fixture(scope="session")
def suppliers_df():
    rng = np.random.default_rng(1)
    n = 10
    return pd.DataFrame({
        "supplier_id": [f"SUP{i:02d}" for i in range(1, n+1)],
        "name": [f"Supplier {i}" for i in range(1, n+1)],
        "country": rng.choice(["China", "Italy", "Germany"], size=n),
        "lead_time_days": rng.integers(30, 120, n),
        "reliability_score": rng.uniform(0.4, 1.0, n).round(3),
        "capacity_utilization": rng.uniform(0.3, 0.95, n).round(3),
        "avg_unit_cost_multiplier": rng.uniform(0.8, 1.2, n).round(3),
        "on_time_delivery_rate": rng.uniform(0.6, 1.0, n).round(3),
        "min_order_qty": rng.choice([100, 200, 500], size=n),
        "payment_terms_days": rng.choice([30, 60, 90], size=n),
    })


@pytest.fixture(scope="session")
def sales_df():
    rng = np.random.default_rng(2)
    n = 500
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({
        "transaction_id": [f"TXN{i:06d}" for i in range(1, n+1)],
        "sku_id": [f"SKU{rng.integers(1, 31):03d}" for _ in range(n)],
        "date": rng.choice(dates.strftime("%Y-%m-%d"), size=n),
        "store_id": rng.choice(["STORE01", "STORE02", "STORE03"], size=n),
        "quantity_sold": rng.integers(1, 10, n),
        "unit_price": rng.uniform(50, 500, n).round(2),
        "channel": rng.choice(["in_store", "online"], size=n),
        "customer_age_group": rng.choice(["18-24", "25-34", "35-44", "45-54"], size=n),
        "customer_gender": rng.choice(["male", "female"], size=n),
        "region": rng.choice(["North", "South", "East"], size=n),
        "stockout_risk": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    })


@pytest.fixture(scope="session")
def inventory_df():
    rng = np.random.default_rng(3)
    rows = []
    for sku_i in range(1, 31):
        for store in ["STORE01", "STORE02", "STORE03"]:
            dos = int(rng.integers(1, 300))
            rows.append({
                "sku_id": f"SKU{sku_i:03d}",
                "store_id": store,
                "quantity_on_hand": int(rng.integers(0, 200)),
                "reorder_point": 20,
                "quantity_on_order": int(rng.integers(0, 50)),
                "days_of_supply": dos,
                "last_restocked_date": "2024-11-01",
                "warehouse_location": f"WH-A{rng.integers(1,10):02d}",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def returns_df():
    rng = np.random.default_rng(4)
    n = 50
    return pd.DataFrame({
        "return_id": [f"RET{i:05d}" for i in range(1, n+1)],
        "sku_id": [f"SKU{rng.integers(1, 31):03d}" for _ in range(n)],
        "transaction_id": [f"TXN{rng.integers(1, 501):06d}" for _ in range(n)],
        "return_date": ["2024-06-15"] * n,
        "return_reason": rng.choice(
            ["wrong_size", "quality_issue", "style_preference", "defective"],
            size=n,
        ),
        "refund_amount": rng.uniform(20, 400, n).round(2),
        "restocked": rng.choice([True, False], size=n),
    })


@pytest.fixture(scope="session")
def social_df():
    rng = np.random.default_rng(5)
    rows = []
    for sku_i in range(1, 11):
        for week_offset in range(8):
            for platform in ["instagram", "tiktok"]:
                rows.append({
                    "sku_id": f"SKU{sku_i:03d}",
                    "week_start": (pd.Timestamp("2024-10-01") + pd.Timedelta(weeks=week_offset)).strftime("%Y-%m-%d"),
                    "platform": platform,
                    "mention_count": int(rng.integers(10, 500)),
                    "sentiment_score": round(float(rng.uniform(0.3, 0.9)), 4),
                    "trend_index": int(rng.integers(10, 90)),
                    "hashtag_reach": int(rng.integers(500, 50000)),
                    "influencer_mentions": int(rng.integers(0, 5)),
                })
    return pd.DataFrame(rows)
