"""Diagnose data patterns and model issues before any tuning."""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

sales    = pd.read_csv("data/synthetic/sales_transactions.csv")
products = pd.read_csv("data/synthetic/products.csv")
social   = pd.read_csv("data/synthetic/social_signals.csv")
inventory= pd.read_csv("data/synthetic/inventory.csv")

sales["date"] = pd.to_datetime(sales["date"])

# ─── 1. Demand sparsity ───────────────────────────────────────────────────────
print("=" * 60)
print("1. DEMAND DISTRIBUTION (raw transactions)")
print("=" * 60)
print(sales["quantity_sold"].describe())
print()

daily = sales.groupby(["sku_id", "date"])["quantity_sold"].sum().reset_index()
all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
all_skus  = daily["sku_id"].unique()
total_cells = len(all_skus) * len(all_dates)
nonzero_cells = len(daily)
print(f"Active SKUs in sales : {len(all_skus)}")
print(f"Date range           : {daily['date'].min().date()} → {daily['date'].max().date()} ({len(all_dates)} days)")
print(f"Total SKU-day grid   : {total_cells:,}")
print(f"Non-zero cells       : {nonzero_cells:,}  ({nonzero_cells/total_cells*100:.1f}% fill rate)")
print()

# Full grid including zeros
idx = pd.MultiIndex.from_product([all_skus, all_dates], names=["sku_id", "date"])
dfull = daily.set_index(["sku_id", "date"]).reindex(idx, fill_value=0).reset_index()
sku_stats = dfull.groupby("sku_id")["quantity_sold"].agg(["mean", "std", "max", "sum"])
print("Per-SKU stats (over full date grid):")
print(sku_stats.describe().round(3))
print()

# ─── 2. Autocorrelation ───────────────────────────────────────────────────────
print("=" * 60)
print("2. AUTOCORRELATION — is there temporal signal?")
print("=" * 60)
total_daily = dfull.groupby("date")["quantity_sold"].sum().sort_index()
acf = {lag: total_daily.autocorr(lag=lag) for lag in [1, 2, 7, 14, 30, 90]}
for lag, val in acf.items():
    print(f"  ACF lag={lag:>3}: {val:.4f}")
print()

# Per-SKU ACF (sample 20 skus)
sample_skus = np.random.default_rng(42).choice(all_skus, size=min(20, len(all_skus)), replace=False)
sku_acf7 = []
for sku in sample_skus:
    ts = dfull[dfull["sku_id"] == sku].sort_values("date")["quantity_sold"]
    if ts.std() > 0:
        sku_acf7.append(ts.autocorr(lag=7))
print(f"Sampled per-SKU ACF(lag=7): mean={np.nanmean(sku_acf7):.3f}, std={np.nanstd(sku_acf7):.3f}")
print()

# ─── 3. Seasonality ──────────────────────────────────────────────────────────
print("=" * 60)
print("3. SEASONALITY")
print("=" * 60)
sales["month"] = sales["date"].dt.month
sales["dow"]   = sales["date"].dt.dayofweek
by_month = sales.groupby("month")["quantity_sold"].sum()
by_dow   = sales.groupby("dow")["quantity_sold"].sum()
print("Demand by month (relative to mean):")
print((by_month / by_month.mean()).round(3).to_string())
print()
print("Demand by day-of-week (relative to mean):")
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for i, v in by_dow.items():
    print(f"  {dow_labels[i]}: {v/by_dow.mean():.3f}")
print()

# ─── 4. Hero SKU concentration ───────────────────────────────────────────────
print("=" * 60)
print("4. SKU DEMAND CONCENTRATION")
print("=" * 60)
sku_total = sales.groupby("sku_id")["quantity_sold"].sum().sort_values(ascending=False)
n = len(sku_total)
for pct in [0.1, 0.2, 0.5]:
    k = max(1, int(n * pct))
    share = sku_total.head(k).sum() / sku_total.sum()
    print(f"  Top {int(pct*100):>3}% SKUs ({k:>3}) → {share*100:.1f}% of total demand")
print(f"  Bottom 50 SKUs total demand: {sku_total.tail(50).sum():,} units")
print(f"  Top SKU: {sku_total.index[0]} = {sku_total.iloc[0]:,} units")
print(f"  Bottom SKU: {sku_total.index[-1]} = {sku_total.iloc[-1]:,} units")
print()

# ─── 5. Social signal correlation ────────────────────────────────────────────
print("=" * 60)
print("5. SOCIAL SIGNAL vs SALES CORRELATION")
print("=" * 60)
social["week_start"] = pd.to_datetime(social["week_start"])
soc_agg = social.groupby(["sku_id","week_start"])[["trend_index","mention_count","sentiment_score"]].mean().reset_index()
sales["week_start"] = sales["date"].dt.to_period("W").dt.start_time
sw = sales.groupby(["sku_id","week_start"])["quantity_sold"].sum().reset_index()
merged = sw.merge(soc_agg, on=["sku_id","week_start"])
corr = merged[["quantity_sold","trend_index","mention_count","sentiment_score"]].corr()
print(corr.round(4))
print()

# ─── 6. Feature vs demand Pearson correlations ───────────────────────────────
print("=" * 60)
print("6. FEATURE CORRELATION WITH DEMAND (daily grid)")
print("=" * 60)
test_df = dfull[dfull["quantity_sold"] > 0].copy()  # only sale days
test_df["month"]   = test_df["date"].dt.month
test_df["dow"]     = test_df["date"].dt.dayofweek
test_df["year"]    = test_df["date"].dt.year
test_df["weekend"] = (test_df["dow"] >= 5).astype(int)
test_df["q"]       = test_df["date"].dt.quarter

# Merge sku-level demand mean
smean = sku_stats[["mean"]].rename(columns={"mean": "sku_mean"})
test_df = test_df.merge(smean, on="sku_id")

for col in ["month", "dow", "year", "weekend", "q", "sku_mean"]:
    c = test_df["quantity_sold"].corr(test_df[col])
    print(f"  {col:<15}: r = {c:.4f}")

print()

# ─── 7. Model feature importance ─────────────────────────────────────────────
print("=" * 60)
print("7. MODEL FEATURE IMPORTANCE (top 20)")
print("=" * 60)
try:
    from models.demand_forecaster import DemandForecaster
    fc = DemandForecaster.load("models/artifacts/demand_forecaster.pkl")
    if fc.model is not None and fc.feature_cols:
        fi = pd.DataFrame({"feature": fc.feature_cols, "importance": fc.model.feature_importances_})
        fi = fi.sort_values("importance", ascending=False).head(20)
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
except Exception as e:
    print(f"  Could not load model: {e}")

print()

# ─── 8. Zero-demand vs nonzero predictability ────────────────────────────────
print("=" * 60)
print("8. ZERO-DAY ANALYSIS — can we predict demand=0?")
print("=" * 60)
# What fraction of zero-demand days have zero the previous 7 days?
dfull_sorted = dfull.sort_values(["sku_id","date"])
dfull_sorted["lag7"] = dfull_sorted.groupby("sku_id")["quantity_sold"].shift(7).fillna(0)
zero_days = dfull_sorted[dfull_sorted["quantity_sold"] == 0]
nonzero_days = dfull_sorted[dfull_sorted["quantity_sold"] > 0]
print(f"  Zero-demand days: {len(zero_days):,} ({len(zero_days)/len(dfull_sorted)*100:.1f}%)")
print(f"  Non-zero days   : {len(nonzero_days):,} ({len(nonzero_days)/len(dfull_sorted)*100:.1f}%)")
zero_and_prev_zero = (zero_days["lag7"] == 0).mean()
print(f"  Of zero days, lag7 was also 0: {zero_and_prev_zero*100:.1f}% — zero-inflation is {('predictable' if zero_and_prev_zero > 0.7 else 'unpredictable')}")
print()

print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
