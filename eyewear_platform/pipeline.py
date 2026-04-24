"""
CLI pipeline runner for the AI-Powered Eyewear Buying Platform.

Usage:
  python pipeline.py --stage all
  python pipeline.py --stage data
  python pipeline.py --stage train
  python pipeline.py --stage recommend
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.logger import configure_logging, get_logger

cfg = get_settings()
configure_logging(cfg.LOG_LEVEL)
logger = get_logger(__name__)


def _timer(fn):
    """Decorator to time stage execution."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  ✓ {fn.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper


@_timer
def run_data_generation():
    """Generate all synthetic CSV files."""
    print("\n[Stage 1/6] Generating synthetic data...")
    from data.generate_synthetic_data import generate_all
    generate_all()


@_timer
def run_similarity_index():
    """Build product similarity index and save to disk."""
    print("\n[Stage 2/6] Building similarity index...")
    from services.data_service import DataService
    from modules.similarity_index import SimilarityIndex

    svc = DataService()
    products = svc.load_products()

    idx = SimilarityIndex(cfg).fit(products)
    save_path = Path(cfg.MODEL_DIR) / "similarity_index.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(save_path))
    print(f"  Similarity index saved → {save_path}")


@_timer
def run_supply_chain():
    """Compute supply chain risk scores and log summary."""
    print("\n[Stage 3/6] Running supply chain analysis...")
    from services.data_service import DataService
    from modules.supply_chain import SupplyChainIntelligence

    svc = DataService()
    suppliers = svc.load_suppliers()
    inventory = svc.load_inventory()
    sales = svc.load_sales()

    sci = SupplyChainIntelligence(cfg)
    health = sci.supply_chain_health_dashboard(suppliers, inventory, sales)

    print(f"  At-risk SKUs: {health['at_risk_sku_count']}")
    print(f"  Critical SKUs (< 7d): {health['critical_sku_count']}")
    print(f"  High-risk suppliers: {health['high_risk_supplier_count']}")
    print(f"  Total reorder value: ${health['total_reorder_value_usd']:,.0f}")


@_timer
def run_customer_signals():
    """Compute customer signals (velocity, trending, return rates)."""
    print("\n[Stage 4/6] Computing customer signals...")
    from services.data_service import DataService
    from modules.customer_signals import CustomerSignals

    svc = DataService()
    sales = svc.load_sales()
    returns = svc.load_returns()
    social = svc.load_social_signals()
    products = svc.load_products()

    cs = CustomerSignals(cfg)
    vel = cs.sales_velocity(sales)
    accelerating = (vel["momentum"] == "accelerating").sum()
    decelerating = (vel["momentum"] == "decelerating").sum()
    print(f"  Accelerating SKUs: {accelerating}")
    print(f"  Decelerating SKUs: {decelerating}")

    trending = cs.trending_skus(social, sales, top_n=5)
    if len(trending) > 0:
        print(f"  Top trending SKU: {trending.iloc[0]['sku_id']} (score={trending.iloc[0]['trend_score']:.3f})")

    rr = cs.return_rate_analysis(sales, returns, products)
    problematic = (rr["is_problematic"]).sum()
    print(f"  Problematic SKUs (>15% return rate): {problematic}")


@_timer
def run_demand_forecaster():
    """Train the XGBoost demand forecasting model."""
    print("\n[Stage 5/6] Training demand forecaster...")
    from services.data_service import DataService
    from models.demand_forecaster import DemandForecaster
    import pandas as pd

    svc = DataService()
    products = svc.load_products()
    sales = svc.load_sales()
    inventory = svc.load_inventory()
    try:
        social = svc.load_social_signals()
    except Exception:
        social = pd.DataFrame()

    forecaster = DemandForecaster(cfg)
    print("  Preparing features...")
    features = forecaster.prepare_features(sales, products, social, inventory)
    print(f"  Feature matrix shape: {features.shape}")

    print("  Training XGBoost model...")
    result = forecaster.train(features, test_months=3)
    metrics = result["metrics"]
    print(f"  MAPE: {metrics['mape']:.2f}% | wMAPE: {metrics.get('wmape', 0):.2f}% | RMSE: {metrics['rmse']:.3f} | R²(log): {metrics.get('r2_log', metrics.get('r2', 0)):.3f}")

    # Store metrics on object for app display
    forecaster._test_metrics = {"MAPE": f"{metrics['mape']:.1f}%", "wMAPE": f"{metrics.get('wmape', 0):.1f}%",
                                  "RMSE": f"{metrics['rmse']:.2f}",
                                  "R²(log)": f"{metrics.get('r2_log', 0):.3f}", "MAE": f"{metrics['mae']:.2f}"}

    save_path = Path(cfg.MODEL_DIR) / "demand_forecaster.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(str(save_path))
    print(f"  Model saved → {save_path}")


@_timer
def run_recommendation_engine():
    """Generate and save buy recommendations."""
    print("\n[Stage 6/6] Generating recommendations...")
    from services.data_service import DataService
    from modules.similarity_index import SimilarityIndex
    from modules.customer_signals import CustomerSignals
    from models.demand_forecaster import DemandForecaster
    from models.recommendation_engine import RecommendationEngine
    import pandas as pd

    svc = DataService()
    products = svc.load_products()
    sales = svc.load_sales()
    inventory = svc.load_inventory()
    suppliers = svc.load_suppliers()

    # Load similarity index
    si_path = Path(cfg.MODEL_DIR) / "similarity_index.pkl"
    if si_path.exists():
        si = SimilarityIndex.load(str(si_path))
    else:
        si = SimilarityIndex(cfg).fit(products)

    # Load or skip forecaster
    fcast_path = Path(cfg.MODEL_DIR) / "demand_forecaster.pkl"
    forecast_df = pd.DataFrame()
    if fcast_path.exists():
        forecaster = DemandForecaster.load(str(fcast_path))
        try:
            social = svc.load_social_signals()
        except Exception:
            social = pd.DataFrame()
        features = forecaster.prepare_features(sales, products, social, inventory)
        forecast_df = forecaster.forecast(forecaster.model, features, cfg.FORECAST_HORIZON_DAYS)
        print(f"  Forecast generated: {len(forecast_df)} rows")
    else:
        print("  No trained forecaster found. Using heuristic demand projection.")

    product_supplier = products[["sku_id", "supplier_id"]].copy()
    engine = RecommendationEngine(cfg)

    recs = engine.generate_buy_recommendations(
        forecast_df, inventory, suppliers, product_supplier, si, products, sales
    )
    cs = CustomerSignals(cfg)
    vel_df = cs.sales_velocity(sales)
    clearance = engine.clearance_alerts(sales, inventory, products, vel_df)
    summary = engine.portfolio_summary(recs, clearance, inventory, products)

    # Save recommendations
    out_dir = Path(cfg.DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_dir / "recommendations.csv", index=False)
    clearance.to_csv(out_dir / "clearance_alerts.csv", index=False)

    print(f"  Buy recommendations: {len(recs)} SKUs")
    print(f"  Clearance alerts: {len(clearance)} SKUs")
    print(f"  Total buy budget: ${summary['total_buy_budget_usd']:,.0f}")
    print(f"  Recommendations saved → {out_dir / 'recommendations.csv'}")


def run_all():
    """Run all stages in dependency order."""
    print("=" * 60)
    print("  EyeAI Buying Platform — Full Pipeline")
    print("=" * 60)
    t_start = time.time()

    run_data_generation()
    run_similarity_index()
    run_supply_chain()
    run_customer_signals()
    run_demand_forecaster()
    run_recommendation_engine()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {total:.1f}s")
    print(f"  Run: streamlit run app.py")
    print("=" * 60)


STAGES = {
    "data": run_data_generation,
    "similarity": run_similarity_index,
    "supply": run_supply_chain,
    "signals": run_customer_signals,
    "train": run_demand_forecaster,
    "recommend": run_recommendation_engine,
    "all": run_all,
}


def main():
    parser = argparse.ArgumentParser(
        description="EyeAI Buying Platform — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  data        Generate synthetic CSV data files
  similarity  Build product similarity index
  supply      Run supply chain analysis
  signals     Compute customer demand signals
  train       Train XGBoost demand forecasting model
  recommend   Generate buy/clearance recommendations
  all         Run all stages in order (default)
        """,
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
