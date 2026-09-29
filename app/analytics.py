import pandas as pd
import numpy as np

DATA_URL="https://raw.githubusercontent.com/HiraAli01/ASOS-dataset-for-machine-learning/main/demand_forecasting.csv"

def load_data(path):
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    return _clean(df)

def load_remote():
    return _clean(pd.read_csv(DATA_URL))

def _clean(df):
    df=df.copy()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    for c in ["Inventory Level","Units Sold","Units Ordered","Price","Discount","Competitor Pricing","Epidemic","Promotion","Demand"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    for c in ["Store ID","Product ID","Category","Region","Weather Condition","Seasonality"]:
        if c in df: df[c]=df[c].fillna("Unknown").astype(str)
    if "Revenue" not in df:
        df["Revenue"]=df["Units Sold"]*df["Price"]*(1-df["Discount"].clip(0,100)/100)
    if "Demand" not in df: df["Demand"]=df["Units Sold"]
    df["Demand_Gap"]=df["Demand"]-df["Units Sold"]
    df["Price_Gap"]=df["Price"]-df["Competitor Pricing"] if "Competitor Pricing" in df else 0
    return df.dropna(subset=["Date"]).copy()

def kpis(df):
    if not len(df):
        return {"Revenue":0,"Units Sold":0,"Demand":0,"Avg Selling Price":0,"Avg Inventory":0,
                "Stock Cover Days":0,"SKU Count":0,"Store Count":0,"Demand Gap":0}

    # Inventory is a point-in-time measure: use the latest observed stock per SKU/store,
    # rather than averaging every historical inventory observation.
    latest_date=df["Date"].max()
    latest=df[df["Date"]==latest_date].copy()
    if not len(latest):
        latest=df.copy()
    latest_stock=float(latest["Inventory Level"].sum())

    # Demand is a flow: convert it to average daily demand before calculating stock cover.
    days=max(df["Date"].nunique(),1)
    daily_demand=float(df["Demand"].sum())/days
    stock_cover=latest_stock/daily_demand if daily_demand>0 else 0

    sold=float(df["Units Sold"].sum())
    demand=float(df["Demand"].sum())
    return {
        "Revenue":float(df["Revenue"].sum()),
        "Units Sold":int(sold),
        "Demand":int(demand),
        "Avg Selling Price":float(df["Price"].mean()),
        "Avg Inventory":latest_stock,
        "Stock Cover Days":stock_cover,
        "SKU Count":int(df["Product ID"].nunique()),
        "Store Count":int(df["Store ID"].nunique()),
        "Demand Gap":int(max(0,demand-sold)),
    }

def monthly_sales(df):
    x=df.copy(); x["Month"]=x["Date"].dt.to_period("M").astype(str)
    return x.groupby("Month",as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum"),Demand=("Demand","sum"))

def top_products(df,n=10):
    return df.groupby(["Product ID","Category"],as_index=False).agg(Revenue=("Revenue","sum"),Units_Sold=("Units Sold","sum"),Demand=("Demand","sum")).sort_values("Revenue",ascending=False).head(n)

def demand_by_category(df):
    return df.groupby("Category",as_index=False).agg(Demand=("Demand","mean"),Units_Sold=("Units Sold","mean"),Revenue=("Revenue","sum")).sort_values("Demand",ascending=False)

def promotion_impact(df):
    return df.groupby("Promotion",as_index=False).agg(Avg_Demand=("Demand","mean"),Avg_Units_Sold=("Units Sold","mean"),Revenue=("Revenue","sum"))
