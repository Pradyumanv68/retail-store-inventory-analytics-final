"""End-to-end pipeline:  python run_pipeline.py
raw CSV -> clean -> analytics -> model comparison -> 7-day forecast -> recommendations -> figures"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import analytics, forecasting, recommendations
from src.data_cleaning import clean
from src.generate_demo_data import generate

ROOT = Path(__file__).resolve().parent
RAW = next((p for p in [ROOT / "data" / "sales_data.csv", ROOT / "data" / "retail_store_inventory.csv"] if p.exists()),
           ROOT / "data" / "retail_store_inventory.csv")
CLEAN = ROOT / "data" / "processed" / "clean.csv.gz"
REP, VIZ = ROOT / "reports", ROOT / "visualizations"


def figures(df, sku, res, te):
    sns.set_theme(style="whitegrid")
    m = analytics.monthly_sales(df)
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(m["date"], m["revenue"] / 1e3, marker="o", color="#1f77b4")
    ax.set(title="Monthly Revenue", ylabel="Revenue (thousands)"); fig.tight_layout(); fig.savefig(VIZ / "monthly_revenue.png", dpi=150); plt.close(fig)
    c = analytics.revenue_by(df, "category")
    fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(data=c, x="category", y="revenue", ax=ax, color="#2ca02c")
    ax.set(title="Revenue by Category"); fig.tight_layout(); fig.savefig(VIZ / "revenue_by_category.png", dpi=150); plt.close(fig)
    order = ["Critical", "High Risk", "Medium", "Healthy", "Overstock"]
    colors = ["#d62728", "#ff7f0e", "#ffdd57", "#2ca02c", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(7, 4)); sns.countplot(data=sku, x="risk", order=order, palette=dict(zip(order, colors)), hue="risk", legend=False, ax=ax)
    ax.set(title="Inventory Risk Distribution (store-product pairs)"); fig.tight_layout(); fig.savefig(VIZ / "risk_distribution.png", dpi=150); plt.close(fig)
    hm = sku.pivot(index="product_id", columns="store_id", values="days_of_cover").clip(upper=60)
    fig, ax = plt.subplots(figsize=(6, 8)); sns.heatmap(hm, cmap="RdYlGn", annot=True, fmt=".0f", ax=ax)
    ax.set(title="Days of Cover Heatmap (capped at 60)"); fig.tight_layout(); fig.savefig(VIZ / "days_of_cover_heatmap.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=res, y="Model", x="RMSE", ax=ax, color="#9467bd")
    ax.set(title="Model Comparison (lower RMSE is better)"); fig.tight_layout(); fig.savefig(VIZ / "model_comparison.png", dpi=150); plt.close(fig)
    daily = te.groupby("date")[["demand_target", "predicted"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4)); daily.plot(ax=ax); ax.set(title="Actual vs Predicted Daily Demand (test period)")
    fig.tight_layout(); fig.savefig(VIZ / "actual_vs_predicted.png", dpi=150); plt.close(fig)


def main():
    if not RAW.exists():
        print("No dataset found - generating demo data with the Kaggle schema...")
        generate().to_csv(RAW, index=False)
    df = clean(pd.read_csv(RAW))
    print(f"Cleaned: {len(df):,} rows | {df['date'].min().date()} -> {df['date'].max().date()} | outliers flagged: {df['units_outlier'].sum()}")
    df.to_csv(CLEAN, index=False)
    sku = analytics.sku_summary(df)
    res, best, model, te = forecasting.train_and_compare(df)
    print(res.to_string(index=False)); print("Best model:", best)
    fc = forecasting.forecast_next(df, model, 7)
    rec = recommendations.build_recommendations(sku, fc)
    res.assign(best=res["Model"].eq(best)).to_csv(REP / "model_metrics.csv", index=False)
    fc.to_csv(REP / "forecast_next_7_days.csv", index=False)
    rec.to_csv(REP / "inventory_actions.csv", index=False)
    joblib.dump(model, ROOT / "models" / "best_model.joblib", compress=3)
    figures(df, sku, res, te)
    print(rec["risk"].value_counts().to_string()); print("Artifacts saved to reports/, visualizations/, models/")


if __name__ == "__main__":
    main()
