import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

def ai_risk_scores(df, horizon=7):
    x=df.groupby(["Store ID","Product ID"],as_index=False).agg(
        Demand=("Demand","mean"),
        Inventory=("Inventory Level","mean"),
        Sales=("Units Sold","mean"),
        Price=("Price","mean"),
        Discount=("Discount","mean"),
        Promotion=("Promotion","mean"),
        Competitor=("Competitor Pricing","mean"),
    )
    x["Coverage"]=x["Inventory"]/(x["Demand"].replace(0,np.nan))
    x["Coverage"]=x["Coverage"].replace([np.inf,np.nan],99)
    x["Demand_Gap"]=np.maximum(0,x["Demand"]-x["Sales"])
    x["Price_Gap"]=x["Price"]-x["Competitor"]
    # Explainable composite risk: stock-out exposure + demand gap + competitive price pressure.
    low_stock=np.clip((horizon-x["Coverage"])/max(horizon,1),0,1)
    gap=np.clip(x["Demand_Gap"]/(x["Demand"]+1),0,1)
    price_pressure=np.clip(x["Price_Gap"]/(x["Price"].abs()+1)*2+0.5,0,1)
    x["AI_Risk_Score"]=np.round(100*(0.55*low_stock+0.30*gap+0.15*price_pressure),1)
    x["AI_Risk_Level"]=pd.cut(x["AI_Risk_Score"],[-1,30,60,100],labels=["LOW","MEDIUM","HIGH"])
    return x.sort_values("AI_Risk_Score",ascending=False)

def detect_anomalies(df):
    daily=df.groupby("Date",as_index=False).agg(
        Revenue=("Revenue","sum"),Demand=("Demand","sum"),Units_Sold=("Units Sold","sum"),
        Inventory=("Inventory Level","mean"),Price=("Price","mean")
    )
    if len(daily)<20:
        daily["Anomaly"]=False; daily["Anomaly_Score"]=0.0
        return daily
    features=daily[["Revenue","Demand","Units_Sold","Inventory","Price"]].replace([np.inf,-np.inf],np.nan).fillna(0)
    model=IsolationForest(n_estimators=200,contamination="auto",random_state=42)
    pred=model.fit_predict(features)
    raw=-model.decision_function(features)
    daily["Anomaly"]=pred==-1
    daily["Anomaly_Score"]=np.round(raw,3)
    return daily.sort_values("Anomaly_Score",ascending=False)

def sku_segments(df, n_clusters=4):
    x=df.groupby(["Store ID","Product ID","Category"],as_index=False).agg(
        Demand=("Demand","mean"),Sales=("Units Sold","mean"),Inventory=("Inventory Level","mean"),
        Revenue=("Revenue","sum"),Price=("Price","mean")
    )
    if len(x)<n_clusters:
        x["Segment"]="Portfolio"
        return x
    features=x[["Demand","Sales","Inventory","Revenue","Price"]].replace([np.inf,-np.inf],np.nan).fillna(0)
    # Robust scaling by median/IQR so high-revenue SKUs do not dominate every cluster.
    med=features.median()
    scale=(features.quantile(.75)-features.quantile(.25)).replace(0,1)
    z=(features-med)/scale
    km=KMeans(n_clusters=n_clusters,n_init=20,random_state=42)
    labels=km.fit_predict(z)
    x["Cluster"]=labels
    summary=x.groupby("Cluster").agg(Demand=("Demand","mean"),Inventory=("Inventory","mean"),Revenue=("Revenue","mean")).sort_values("Demand")
    names=["Value Builder","Growth Driver","Stable Core","Slow Mover"]
    mapping={cluster:names[i%len(names)] for i,cluster in enumerate(summary.index)}
    x["Segment"]=x["Cluster"].map(mapping)
    return x.sort_values("Revenue",ascending=False)

def feature_importance(df):
    cols=["Inventory Level","Units Sold","Price","Discount","Promotion","Competitor Pricing","Epidemic"]
    cols=[c for c in cols if c in df.columns]
    if len(df)<50 or len(cols)<3:
        return pd.DataFrame(columns=["Feature","Importance"])
    x=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=pd.to_numeric(df["Demand"],errors="coerce").fillna(0)
    model=RandomForestRegressor(n_estimators=120,max_depth=10,random_state=42,n_jobs=-1)
    model.fit(x,y)
    imp=pd.DataFrame({"Feature":cols,"Importance":model.feature_importances_})
    return imp.sort_values("Importance",ascending=False)

def scenario_demand(df, discount_delta=0, promotion=0, price_delta=0):
    base=float(df["Demand"].mean()) if len(df) else 0
    promo_lift=0
    if "Promotion" in df and df["Promotion"].nunique()>1:
        promoted=df.loc[df["Promotion"]==1,"Demand"].mean()
        regular=df.loc[df["Promotion"]==0,"Demand"].mean()
        if pd.notna(promoted) and pd.notna(regular) and regular>0:
            promo_lift=(promoted/regular)-1
    price_effect=0
    if "Price" in df and "Demand" in df and df["Price"].nunique()>1:
        corr=df["Price"].corr(df["Demand"])
        if pd.notna(corr):
            price_effect=-0.08*np.sign(corr)*price_delta
    discount_effect=0.004*discount_delta
    multiplier=1+promo_lift*promotion+discount_effect+price_effect
    return max(0,base*multiplier), promo_lift

def copilot_summary(df, risk, anomalies, segments):
    if len(df)==0:
        return "No data is available for the selected filters."
    revenue=float(df["Revenue"].sum())
    demand=float(df["Demand"].sum())
    sold=float(df["Units Sold"].sum())
    gap=max(0,demand-sold)
    high=int((risk["AI_Risk_Level"]=="HIGH").sum()) if len(risk) else 0
    anomaly_count=int(anomalies["Anomaly"].sum()) if len(anomalies) else 0
    top=risk.iloc[0] if len(risk) else None
    segment=segments.iloc[0]["Segment"] if len(segments) else "Portfolio"
    parts=[
        f"Revenue across the selected scope is ₹{revenue:,.0f} with {demand:,.0f} demand units and {sold:,.0f} units sold.",
        f"The model identifies {high} high-risk SKU/store combinations and {anomaly_count} unusual daily patterns.",
        f"Demand gap is {gap:,.0f} units, indicating potential unmet demand or stock constraints." if gap else "Demand and realized sales are closely aligned in the selected scope.",
        f"Highest current risk: {top['Store ID']} / {top['Product ID']} at {top['AI_Risk_Score']:.0f}/100." if top is not None else "No material SKU risk was detected.",
        f"The leading portfolio segment is {segment}; use the Action Center to prioritize interventions."
    ]
    return " ".join(parts)
