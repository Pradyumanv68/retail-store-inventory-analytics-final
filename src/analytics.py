"""Steps 3-5: KPIs, sales analytics, inventory risk score, fast/slow movers."""
import numpy as np
import pandas as pd
from .data_cleaning import KEYS

def kpis(df: pd.DataFrame) -> dict:
    last = df[df["date"] == df["date"].max()]
    days = max(df["date"].nunique(),1)
    avg_inv_total = df.groupby(KEYS)["inventory_level"].mean().sum()
    return {
        "total_revenue": df["revenue"].sum(), "total_units": int(df["units_sold"].sum()),
        "current_inventory": int(last["inventory_level"].sum()), "avg_revenue_per_txn": df["revenue"].mean(),
        "inventory_turnover": df["units_sold"].sum() / max(avg_inv_total,1) * 365 / days,
        "stockout_rate_pct": df["stockout_flag"].mean() * 100,
        "unmet_demand_pct": df["unmet_demand"].sum() / max(df["demand_target"].sum(),1) * 100,
        "lost_revenue": df["lost_revenue"].sum(),
    }

def revenue_by(df, col):
    return df.groupby(col).agg(revenue=("revenue","sum"),units=("units_sold","sum")).sort_values("revenue",ascending=False).reset_index()

def monthly_sales(df):
    m=df.groupby(df["date"].dt.to_period("M")).agg(revenue=("revenue","sum"),units=("units_sold","sum"))
    m.index=m.index.to_timestamp(); return m.reset_index()

def risk_level(days_cover, lead_time_days=3):
    L=lead_time_days
    if days_cover < .5*L: return "Critical"
    if days_cover < L: return "High Risk"
    if days_cover < 2*L: return "Medium"
    if days_cover <= 2.5*L: return "Healthy"
    return "Overstock"

def sku_summary(df, window=30, lead_time_days=3):
    last=df["date"].max(); rec=df[df["date"]>last-pd.Timedelta(days=window)]
    s=rec.groupby(KEYS).agg(avg_daily_sales=("units_sold","mean"),avg_daily_demand=("demand_target","mean"),
                             std_daily_demand=("demand_target","std"),recent_stockout_rate=("stockout_flag","mean")).reset_index()
    cur=df[df["date"]==last][KEYS+["inventory_level"]].rename(columns={"inventory_level":"current_inventory"})
    full=df.groupby(KEYS).agg(total_units=("units_sold","sum"),total_revenue=("revenue","sum"),avg_inventory=("inventory_level","mean"),
                              category=("category",lambda x:x.mode().iat[0]),region=("region",lambda x:x.mode().iat[0])).reset_index()
    full["inventory_turnover"]=full["total_units"]/full["avg_inventory"].clip(lower=.1)/max(df["date"].nunique(),1)*365
    s=s.merge(cur,on=KEYS).merge(full,on=KEYS); s["std_daily_demand"]=s["std_daily_demand"].fillna(0)
    s["days_of_cover"]=s["current_inventory"]/s["avg_daily_demand"].clip(lower=.1)
    s["risk"]=s["days_of_cover"].apply(risk_level,lead_time_days=lead_time_days)
    s["risk_score"]=(100*(1-(s["days_of_cover"]/(2*lead_time_days)).clip(0,1))).round(1)
    p10,p25,p75=s["avg_daily_sales"].quantile([.10,.25,.75])
    s["movement"]=np.select([s["avg_daily_sales"]>=p75,s["avg_daily_sales"]>=p25,s["avg_daily_sales"]>=p10],["Fast Mover","Medium Mover","Slow Mover"],"Dead Stock")
    return s
