import pandas as pd,numpy as np

def inventory_status(df,forecast_days=7,safety=0.25):
    d=df.groupby(["Store ID","Product ID"],as_index=False).agg(
        Daily_Demand=("Demand","mean"),Inventory_Level=("Inventory Level","mean"),
        Units_Ordered=("Units Ordered","mean"),Price=("Price","mean"))
    d["Demand Forecast"]=d["Daily_Demand"]*forecast_days
    d["Safety Stock"]=d["Demand Forecast"]*safety
    d["Coverage Days"]=d["Inventory_Level"]/d["Daily_Demand"].replace(0,np.nan)
    d["Coverage Days"]=d["Coverage Days"].replace([np.inf,np.nan],999)
    d["Recommended Order"]=np.ceil(np.maximum(0,d["Demand Forecast"]+d["Safety Stock"]-d["Inventory_Level"]))
    def status(r):
        if r["Inventory_Level"]<=max(1,r["Daily_Demand"]): return "CRITICAL"
        if r["Inventory_Level"]<r["Demand Forecast"]+r["Safety Stock"]: return "REPLENISH"
        if r["Inventory_Level"]>max(r["Demand Forecast"]*4,50): return "OVERSTOCK"
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
