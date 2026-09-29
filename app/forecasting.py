import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def _features(x,target="Demand"):
    z=x.copy().sort_values("Date")
    z["dow"]=z["Date"].dt.dayofweek; z["month"]=z["Date"].dt.month; z["dayofyear"]=z["Date"].dt.dayofyear
    z["lag1"]=z[target].shift(1); z["lag7"]=z[target].shift(7); z["lag14"]=z[target].shift(14); z["roll7"]=z[target].rolling(7).mean()
    return z.dropna()

def chronological_model(df):
    daily=df.groupby("Date",as_index=False).agg({"Demand":"sum","Inventory Level":"mean","Price":"mean","Discount":"mean","Promotion":"mean","Competitor Pricing":"mean"})
    z=_features(daily,"Demand")
    feats=["dow","month","dayofyear","lag1","lag7","lag14","roll7","Inventory Level","Price","Discount","Promotion","Competitor Pricing"]
    split=max(int(len(z)*.8),1); Xtr,Xte=z[feats].iloc[:split],z[feats].iloc[split:]; ytr,yte=z["Demand"].iloc[:split],z["Demand"].iloc[split:]
    model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=18,l2_regularization=0.5,random_state=42).fit(Xtr,ytr)
    pred=model.predict(Xte) if len(Xte) else model.predict(Xtr); actual=yte if len(yte) else ytr
    metrics={"MAE":mean_absolute_error(actual,pred),"RMSE":mean_squared_error(actual,pred)**.5,"R2":r2_score(actual,pred) if len(actual)>1 else 0}
    out=z.iloc[split:].copy() if len(Xte) else z.copy(); out["Predicted_Demand"]=pred
    return model,metrics,out

def forecast_next_days(df,store,product,days=7):
    x=df[(df["Store ID"].astype(str)==str(store))&(df["Product ID"].astype(str)==str(product))].copy()
    if len(x)<21:
        base=max(x["Demand"].tail(7).mean() if len(x) else 0,0); dates=pd.date_range(df["Date"].max()+pd.Timedelta(days=1),periods=days)
        return pd.DataFrame({"Date":dates,"Predicted_Demand":np.repeat(round(base),days)})
    daily=x.groupby("Date",as_index=False).agg({"Demand":"sum"}); z=_features(daily,"Demand"); feats=["dow","month","dayofyear","lag1","lag7","lag14","roll7"]
    model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=15,random_state=42).fit(z[feats],z["Demand"])
    hist=z[["Date","Demand"]].copy(); rows=[]
    for _ in range(days):
        d=hist["Date"].max()+pd.Timedelta(days=1); vals=hist["Demand"].tolist()
        row=pd.DataFrame([{"dow":d.dayofweek,"month":d.month,"dayofyear":d.dayofyear,"lag1":vals[-1],"lag7":vals[-7],"lag14":vals[-14],"roll7":np.mean(vals[-7:])}])
        y=max(0,float(model.predict(row[feats])[0])); rows.append([d,y]); hist=pd.concat([hist,pd.DataFrame({"Date":[d],"Demand":[y]})],ignore_index=True)
    return pd.DataFrame(rows,columns=["Date","Predicted_Demand"])
