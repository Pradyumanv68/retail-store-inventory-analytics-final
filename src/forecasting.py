"""Step 6-7: demand forecasting - model comparison + recursive 7-day forecast."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data_cleaning import KEYS
from .generate_demo_data import season

CAT_F=["store_id","product_id","category","region","weather_condition","seasonality"]
TARGET="demand_target"
NUM_F=["price","discount","competitor_pricing","holiday_promotion","epidemic","month","day_of_week","day_of_year","lag_1","lag_7","roll_7","roll_28"]
FEATURES=CAT_F+NUM_F

def add_lag_features(df):
    df=df.sort_values(KEYS+["date"]).copy(); g=df.groupby(KEYS)[TARGET]
    df["lag_1"]=g.shift(1); df["lag_7"]=g.shift(7)
    df["roll_7"]=g.transform(lambda s:s.shift(1).rolling(7).mean())
    df["roll_28"]=g.transform(lambda s:s.shift(1).rolling(28).mean())
    return df.dropna(subset=["lag_1","lag_7","roll_7","roll_28"])

def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    return {"MAE":mean_absolute_error(y,p),"RMSE":float(np.sqrt(mean_squared_error(y,p))),
            "MAPE_%":float(np.mean(np.abs(y-p)/np.maximum(y,1))*100),"R2":r2_score(y,p)}

def _pipe(model,scale=False):
    num=StandardScaler() if scale else "passthrough"
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT_F),("num",num,NUM_F)])
    return Pipeline([("pre",pre),("model",model)])

def train_and_compare(df,test_days=90):
    d=add_lag_features(df); cutoff=d["date"].max()-pd.Timedelta(days=test_days)
    tr,te=d[d["date"]<=cutoff],d[d["date"]>cutoff]
    models={"Linear Regression":_pipe(LinearRegression(),scale=True),
            "Random Forest":_pipe(RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=3,n_jobs=-1,random_state=42)),
            "Gradient Boosting":_pipe(HistGradientBoostingRegressor(max_iter=300,learning_rate=.06,random_state=42))}
    rows=[{"Model":"Baseline (7-day moving avg)",**metrics(te[TARGET],te["roll_7"])}]; fitted={}
    for name,m in models.items():
        m.fit(tr[FEATURES],tr[TARGET]); fitted[name]=m; rows.append({"Model":name,**metrics(te[TARGET],m.predict(te[FEATURES]))})
    res=pd.DataFrame(rows).round(3); best=res[res["Model"].isin(fitted)].sort_values("RMSE").iloc[0]["Model"]
    te=te.assign(predicted=fitted[best].predict(te[FEATURES]))
    return res,best,fitted[best],te

def forecast_next(df,model,horizon=7):
    last=df["date"].max()
    piv=df[df["date"]>last-pd.Timedelta(days=28)].pivot_table(index="date",columns=KEYS,values=TARGET).ffill().bfill()
    W=piv.values[-28:]; recent=df[df["date"]>last-pd.Timedelta(days=30)]
    static=recent.groupby(KEYS).agg(price=("price","mean"),discount=("discount","mean"),competitor_pricing=("competitor_pricing","mean"),weather_condition=("weather_condition",lambda s:s.mode().iat[0])).reset_index()
    meta=df.groupby(KEYS).agg(category=("category",lambda x:x.mode().iat[0]),region=("region",lambda x:x.mode().iat[0]),seasonality=("seasonality",lambda x:x.mode().iat[0])).reset_index()
    base=piv.columns.to_frame(index=False).merge(static,on=KEYS,how="left").merge(meta,on=KEYS,how="left"); out=[]
    for h in range(1,horizon+1):
        d=last+pd.Timedelta(days=h); X=base.copy()
        X["holiday_promotion"]=0; X["epidemic"]=int(df.loc[df["date"]==last,"epidemic"].max())
        X["month"],X["day_of_week"],X["day_of_year"]=d.month,d.dayofweek,d.dayofyear; X["seasonality"]=season(d.month)
        X["lag_1"],X["lag_7"]=W[-1],W[-7]; X["roll_7"],X["roll_28"]=W[-7:].mean(0),W[-28:].mean(0)
        pred=np.clip(model.predict(X[FEATURES]),0,None); W=np.vstack([W,pred])
        out.append(X[KEYS].assign(date=d,predicted_units=pred))
    return pd.concat(out,ignore_index=True)
