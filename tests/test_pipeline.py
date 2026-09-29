import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import analytics, forecasting, recommendations
from src.data_cleaning import clean
from src.generate_demo_data import generate

def _small():
    raw=generate()
    return raw[raw["Product ID"].isin(["P0001","P0002","P0003"]) & raw["Store ID"].isin(["S001","S002"])]

def test_cleaning_removes_dupes_and_adds_features():
    raw=_small(); df=clean(raw.copy()._append(raw.head(50),ignore_index=True))
    assert len(df)==len(raw)
    assert {"revenue","week","stock_coverage_days"} <= set(df.columns)
    assert df["revenue"].ge(0).all()

def test_risk_levels():
    assert analytics.risk_level(1)=="Critical"
    assert analytics.risk_level(100)=="Overstock"
    assert analytics.risk_level(2)=="High Risk"
    assert analytics.risk_level(5)=="Medium"
    assert analytics.risk_level(7)=="Healthy"

def test_forecast_and_recommendations():
    df=clean(_small())
    sku=analytics.sku_summary(df)
    _,_,model,_=forecasting.train_and_compare(df)
    fc=forecasting.forecast_next(df,model,7)
    assert len(fc)==7*len(sku)
    assert fc["predicted_units"].ge(0).all()
    rec=recommendations.build_recommendations(sku,fc)
    assert (rec["recommended_order"]>=0).all()
    assert rec["action"].str.len().gt(0).all()
