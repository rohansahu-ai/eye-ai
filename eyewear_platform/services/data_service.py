"""
Centralized data loading, validation, and caching layer.
All modules access data through this service.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import Settings, get_settings
from utils.logger import get_logger


class DataValidationError(Exception):
    """Raised when a dataset fails schema validation."""
    pass


EXPECTED_SCHEMAS = {
    "products": [
        "sku_id", "name", "frame_shape", "material", "color", "price_point",
        "gender", "season_launched", "is_active", "supplier_id", "cost_price",
        "retail_price", "category", "launch_date",
    ],
    "suppliers": [
        "supplier_id", "name", "country", "lead_time_days", "reliability_score",
        "capacity_utilization", "avg_unit_cost_multiplier", "on_time_delivery_rate",
        "min_order_qty", "payment_terms_days",
    ],
    "sales": [
        "transaction_id", "sku_id", "date", "store_id", "quantity_sold",
        "unit_price", "channel", "customer_age_group", "customer_gender", "region",
    ],
    "inventory": [
        "sku_id", "store_id", "quantity_on_hand", "reorder_point",
        "quantity_on_order", "days_of_supply", "last_restocked_date",
    ],
    "returns": [
        "return_id", "sku_id", "transaction_id", "return_date",
        "return_reason", "refund_amount", "restocked",
    ],
    "social_signals": [
        "sku_id", "week_start", "platform", "mention_count",
        "sentiment_score", "trend_index", "influencer_mentions",
    ],
}


class DataService:
    """
    Centralized data loading, validation, and caching layer.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.data_dir = Path(self.config.DATA_DIR)
        self._cache: dict = {}
        self.logger = get_logger(__name__)
        self._last_refresh: Optional[datetime] = None

    def _load_csv(self, filename: str, schema_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
        cache_key = filename
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. "
                "Run `python data/generate_synthetic_data.py` first."
            )

        df = pd.read_csv(path, **kwargs)
        if schema_key:
            self.validate_schema(df, EXPECTED_SCHEMAS[schema_key], filename)

        self._cache[cache_key] = df
        self._last_refresh = datetime.now()
        self.logger.info("Loaded dataset", file=filename, rows=len(df))
        return df

    def load_products(self) -> pd.DataFrame:
        """Load + validate products.csv."""
        df = self._load_csv("products.csv", schema_key="products")
        df["is_active"] = df["is_active"].astype(bool)
        df["cost_price"] = df["cost_price"].astype(float)
        df["retail_price"] = df["retail_price"].astype(float)
        return df

    def load_sales(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load sales_transactions.csv with optional date filter."""
        df = self._load_csv("sales_transactions.csv", schema_key="sales")
        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df

    def load_inventory(self) -> pd.DataFrame:
        """Load inventory.csv."""
        df = self._load_csv("inventory.csv", schema_key="inventory")
        df["quantity_on_hand"] = df["quantity_on_hand"].astype(int)
        df["quantity_on_order"] = df["quantity_on_order"].astype(int)
        df["days_of_supply"] = df["days_of_supply"].astype(float)
        return df

    def load_suppliers(self) -> pd.DataFrame:
        """Load suppliers.csv."""
        return self._load_csv("suppliers.csv", schema_key="suppliers")

    def load_returns(self) -> pd.DataFrame:
        """Load returns.csv."""
        df = self._load_csv("returns.csv", schema_key="returns")
        df["return_date"] = pd.to_datetime(df["return_date"])
        df["restocked"] = df["restocked"].astype(bool)
        return df

    def load_social_signals(self) -> pd.DataFrame:
        """Load social_signals.csv."""
        df = self._load_csv("social_signals.csv", schema_key="social_signals")
        df["week_start"] = pd.to_datetime(df["week_start"])
        return df

    def validate_schema(
        self, df: pd.DataFrame, expected_cols: list, name: str
    ) -> None:
        """Raise DataValidationError with helpful message if columns missing."""
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Schema validation failed for '{name}'. "
                f"Missing columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self.logger.info("Data cache cleared")

    def data_quality_report(self) -> dict:
        """Returns per-dataset quality report."""
        report = {}
        file_map = {
            "products": "products.csv",
            "sales": "sales_transactions.csv",
            "inventory": "inventory.csv",
            "suppliers": "suppliers.csv",
            "returns": "returns.csv",
            "social_signals": "social_signals.csv",
        }

        loaders = {
            "products": self.load_products,
            "sales": self.load_sales,
            "inventory": self.load_inventory,
            "suppliers": self.load_suppliers,
            "returns": self.load_returns,
            "social_signals": self.load_social_signals,
        }

        for name, loader in loaders.items():
            try:
                df = loader()
                null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
                dup_count = int(df.duplicated().sum())

                date_col = next(
                    (c for c in df.columns if "date" in c.lower() or "week" in c.lower()), None
                )
                date_range = None
                if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    date_range = (
                        str(df[date_col].min().date()),
                        str(df[date_col].max().date()),
                    )

                report[name] = {
                    "row_count": len(df),
                    "null_pct_per_col": null_pct,
                    "duplicate_count": dup_count,
                    "date_range": date_range,
                }
            except Exception as e:
                report[name] = {"error": str(e)}

        return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Generate data if not exists
    from data.generate_synthetic_data import generate_all
    data_path = Path(get_settings().DATA_DIR) / "products.csv"
    if not data_path.exists():
        generate_all()

    svc = DataService()
    products = svc.load_products()
    print(f"Products: {len(products)} rows")

    sales = svc.load_sales(start_date="2024-01-01")
    print(f"Sales (2024): {len(sales)} rows")

    report = svc.data_quality_report()
    for ds, info in report.items():
        if "error" not in info:
            print(f"{ds}: {info['row_count']} rows, {info['duplicate_count']} dups")
