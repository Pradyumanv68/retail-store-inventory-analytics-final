"""Steps 1-2: load, clean and feature-engineer the raw retail dataset."""
import re
import pandas as pd

NUM = ["inventory_level", "units_sold", "units_ordered", "price", "discount",
       "holiday_promotion", "competitor_pricing"]
OPT = ["demand_forecast", "epidemic", "demand"]
CAT = ["store_id", "product_id", "category", "region", "weather_condition", "seasonality"]
KEYS = ["store_id", "product_id"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df

def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw)
    if "holiday_promotion" not in df and "promotion" in df:
        df = df.rename(columns={"promotion": "holiday_promotion"})
    if "epidemic" not in df:
        df["epidemic"] = 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates()
    num = NUM + [c for c in OPT if c in df]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        if c in df:
            df[c] = df[c].fillna(df[c].mode().iat[0] if not df[c].mode().empty else "Unknown").astype(str).str.strip()
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    for c in num:
        if c in df:
            df[c] = df.groupby(KEYS)[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())
    df["holiday_promotion"] = df["holiday_promotion"].round().astype(int)
    df["epidemic"] = df["epidemic"].round().astype(int)
    df["units_sold"] = df["units_sold"].clip(lower=0)
    q1, q3 = df["units_sold"].quantile([.25, .75])
    df["units_outlier"] = (df["units_sold"] > q3 + 3 * (q3 - q1)).astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["discount_pct"] = df["discount"]
    df["revenue"] = df["units_sold"] * df["price"] * (1 - df["discount_pct"] / 100)
    df["demand_target"] = df["demand"] if "demand" in df else df["units_sold"]
    df["unmet_demand"] = (df["demand_target"] - df["units_sold"]).clip(lower=0)
    df["lost_revenue"] = df["unmet_demand"] * df["price"] * (1 - df["discount_pct"] / 100)
    df["stockout_flag"] = (df["units_sold"] >= df["inventory_level"]).astype(int)
    roll = df.groupby(KEYS)["units_sold"].transform(lambda s: s.rolling(30, min_periods=7).mean())
    df["stock_coverage_days"] = df["inventory_level"] / roll.clip(lower=0.1)
    return df
