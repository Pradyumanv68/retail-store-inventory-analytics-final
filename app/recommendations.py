import pandas as pd,numpy as np

def inventory_status(df,forecast_days=7,safety=0.25):
    x=df.copy().sort_values("Date")
    latest_date=x["Date"].max()
    latest=x[x["Date"]==latest_date].copy()
    # Use current stock, while demand is estimated from the most recent 28 days.
    recent_start=latest_date-pd.Timedelta(days=27)
    recent=x[x["Date"]>=recent_start]
    demand=recent.groupby(["Store ID","Product ID"],as_index=False).agg(
        Daily_Demand=("Demand","mean"),Units_Ordered=("Units Ordered","mean"),Price=("Price","mean")
    )
    stock=latest.groupby(["Store ID","Product ID"],as_index=False).agg(Inventory_Level=("Inventory Level","sum"))
    d=demand.merge(stock,on=["Store ID","Product ID"],how="outer").fillna(0)
    d["Demand Forecast"]=d["Daily_Demand"]*forecast_days
    d["Safety Stock"]=d["Daily_Demand"]*forecast_days*safety
    d["Coverage Days"]=d["Inventory_Level"]/d["Daily_Demand"].replace(0,np.nan)
    d["Coverage Days"]=d["Coverage Days"].replace([np.inf,np.nan],999)
    d["Recommended Order"]=np.ceil(np.maximum(0,d["Demand Forecast"]+d["Safety Stock"]-d["Inventory_Level"]))
    def status(r):
        cov=r["Coverage Days"]
        if cov<1: return "CRITICAL"
        if cov<7: return "REPLENISH"
        if cov>30: return "OVERSTOCK"
        return "HEALTHY"
    d["Inventory_Status"]=d.apply(status,axis=1)
    return d.sort_values(["Inventory_Status","Recommended Order"],ascending=[True,False])

def generate_alerts(df,forecast_days=7):
    x=inventory_status(df,forecast_days,.25)
    x["Priority"]=x["Inventory_Status"].map({"CRITICAL":1,"REPLENISH":2,"OVERSTOCK":3,"HEALTHY":4})
    x["Action"]=x.apply(lambda r:"Expedite replenishment" if r["Inventory_Status"]=="CRITICAL" else ("Place replenishment order" if r["Inventory_Status"]=="REPLENISH" else ("Review markdown/promotion" if r["Inventory_Status"]=="OVERSTOCK" else "Monitor")),axis=1)
    return x.sort_values(["Priority","Recommended Order"],ascending=[True,False])

def narrative(r):
    return f"Inventory {r['Inventory_Level']:.0f} vs {r['Demand Forecast']:.0f} expected demand units. Coverage: {r['Coverage Days']:.1f} days. Recommended order: {r['Recommended Order']:.0f}. Action: {r['Action']}."
