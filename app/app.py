import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from analytics import load_data, load_remote, _clean, kpis, monthly_sales, top_products, demand_by_category, promotion_impact
from forecasting import chronological_model, forecast_next_days
from recommendations import inventory_status, generate_alerts, narrative
from ai_engine import ai_risk_scores, detect_anomalies, sku_segments, feature_importance, scenario_demand, copilot_summary

st.set_page_config(page_title="Retail Intelligence Command Center", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container{padding:1.15rem 1.8rem 2.5rem;max-width:1500px;margin:0 auto}
[data-testid="stSidebar"]{background:#0b0d13;border-right:1px solid #242733;min-width:290px;width:290px} [data-testid="stSidebarContent"]{padding:1rem 1rem 2rem} [data-testid="stAppViewContainer"]{overflow-x:hidden}
[data-testid="stSidebar"] *{font-family:Inter,system-ui,sans-serif;box-sizing:border-box} [data-testid="stFileUploader"]{width:100%} [data-testid="stFileUploader"] section{padding:0;border:0;background:transparent} [data-testid="stFileUploader"] button{width:100%;min-height:42px;border-radius:9px}
body{background:#0b0d13}
.hero{padding:.35rem 0 .9rem}
.hero h1{font-size:2.45rem;line-height:1.05;margin:.15rem 0 .25rem;color:#f7f8fb;font-weight:800}
.hero p{margin:0;color:#9ca3af;font-size:.93rem}
.eyebrow{color:#ff4b4b;text-transform:uppercase;letter-spacing:1.7px;font-size:.68rem;font-weight:800}
.kpi-card{background:#11141b;border:1px solid #252936;border-radius:13px;padding:15px 17px;min-height:112px}
.kpi-label{color:#a6adba;font-size:.78rem;font-weight:600}
.kpi-value{color:#f8fafc;font-size:1.7rem;font-weight:800;margin:.25rem 0}
.kpi-sub{color:#737b89;font-size:.72rem}
.section-title{font-size:1.02rem;font-weight:800;color:#f3f4f6;margin:.45rem 0 .65rem}
.panel{background:#11141b;border:1px solid #252936;border-radius:14px;padding:14px}
.insight{padding:12px 14px;border-radius:11px;background:#151821;border:1px solid #282d39;margin-bottom:8px}
.insight b{color:#f5f6f8}.muted{color:#8f98a7;font-size:.82rem}
.small-note{color:#737b89;font-size:.72rem}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid #252936}
.stTabs [data-baseweb="tab"]{padding:9px 14px;font-weight:700}
.stTabs [aria-selected="true"]{color:#ff5a5a}
[data-testid="stMetric"]{background:#11141b;border:1px solid #252936;border-radius:13px;padding:11px 14px}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_default_df():
    path="data/retail_store_inventory.csv"
    if os.path.exists(path):
        return load_data(path), "Local project dataset"
    try:
        return load_remote(), "76,000-row retail demand dataset"
    except Exception:
        rng=np.random.default_rng(42)
        dates=pd.date_range("2025-01-01",periods=180)
        rows=[]; cats=["Electronics","Grocery","Clothing","Home"]; regs=["North","South","East","West"]
        for store in range(1,7):
            for prod in range(1,13):
                cat=cats[(prod-1)%4]; base=8+(prod%7)*2
                for d in dates:
                    sold=max(0,int(rng.poisson(base))); price=round(80+prod*23+store*7,2)
                    inv=max(0,int(rng.normal(100,25))); disc=round(float(rng.uniform(0,.2)),2)
                    demand=max(sold,int(sold*(1+rng.uniform(.05,.35))))
                    rows.append([d,f"S{store:02d}",f"P{prod:03d}",cat,inv,sold,max(0,int(sold*1.1)),price,disc,int(rng.random()<.2),regs[(store-1)%4],demand,price+rng.normal(0,5)])
        return pd.DataFrame(rows,columns=["Date","Store ID","Product ID","Category","Inventory Level","Units Sold","Units Ordered","Price","Discount","Promotion","Region","Demand","Competitor Pricing"]), "Fallback demo dataset"

df, source = get_default_df()

with st.sidebar:
    st.markdown("## 📁 Data")
    st.caption("Project retail dataset loaded and ready for analysis.")
    st.markdown(
        '<div class="panel"><b>✓ Dataset Ready</b><br><span class="muted">76,000 retail observations</span></div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### 🎛️ Filters")
    st.caption("Filter the command center by store and category.")
    stores=sorted(df["Store ID"].astype(str).unique()); cats=sorted(df["Category"].astype(str).unique())
    ss=st.multiselect("Store coverage",stores,stores,placeholder="Select stores")
    cc=st.multiselect("Product categories",cats,cats,placeholder="Select categories")
    horizon=st.slider("Forecast / planning horizon",3,30,7)
    st.divider()
    st.markdown("### 🤖 Decision layer")
    st.caption("Observe → Diagnose → Predict → Simulate → Decide → Act")
    st.caption(source)

f=df[df["Store ID"].astype(str).isin(ss)&df["Category"].astype(str).isin(cc)].copy()
k=kpis(f)
inv=inventory_status(f,horizon,.25)
alerts=generate_alerts(f,horizon)
critical=int((inv["Inventory_Status"]=="CRITICAL").sum())
replenish=int((inv["Inventory_Status"]=="REPLENISH").sum())
overstock=int((inv["Inventory_Status"]=="OVERSTOCK").sum())
total=max(len(inv),1)
coverage_score=min(100,max(0,(k["Stock Cover Days"]/max(horizon,1))*100))
risk_score=100-((critical+replenish)/total*100)
health=max(0,min(100,round(.55*coverage_score+.45*risk_score-(overstock/total*25))))

@st.cache_data(ttl=3600)
def get_ai_bundle(data,h):
    return ai_risk_scores(data,h),detect_anomalies(data),sku_segments(data),feature_importance(data)

risk,anomalies,segments,importance=get_ai_bundle(f,horizon)

st.markdown("""
<div class="hero">
<div class="eyebrow">Retail Decision Intelligence • Capstone 2026</div>
<h1>📦 Retail Intelligence Command Center</h1>
<p>AI-powered sales, inventory, demand forecasting and operational decision support.</p>
</div>
""",unsafe_allow_html=True)

# KPI strip
cards=st.columns(5)
kpi_data=[
    ("Total Revenue",f"₹{k['Revenue']/1e6:.1f}M","Commercial performance"),
    ("Demand",f"{k['Demand']:,}","Forecast target"),
    ("Inventory",f"{k['Avg Inventory']:,.0f}","Latest observed stock"),
    ("Stock Cover",f"{k['Stock Cover Days']:.1f} days","Coverage"),
    ("Inventory Health",f"{health}/100","Decision health score"),
]
for col,(label,value,sub) in zip(cards,kpi_data):
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

tabs=st.tabs(["📊 Overview","🤖 AI Insights","📦 Inventory Plan","🔮 Forecast","🧾 Transactions"])

with tabs[0]:
    st.markdown('<div class="section-title">Executive Overview</div>',unsafe_allow_html=True)
    a,b=st.columns([1.35,1])
    with a:
        m=monthly_sales(f)
        fig=px.bar(m,x="Month",y="Revenue",title="Monthly Revenue",text_auto=False)
        fig.update_layout(template="plotly_dark",height=360,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        cnt=inv["Inventory_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig=px.pie(cnt,names="Status",values="Count",hole=.62,title="Inventory Health")
        fig.update_layout(template="plotly_dark",height=360,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

    a,b=st.columns(2)
    with a:
        tp=top_products(f,10)
        fig=px.bar(tp.sort_values("Revenue"),x="Revenue",y="Product ID",color="Category",orientation="h",title="Top Products by Revenue")
        fig.update_layout(template="plotly_dark",height=390,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        store=f.groupby("Store ID",as_index=False).agg(Revenue=("Revenue","sum"),Demand=("Demand","sum"))
        fig=px.bar(store,x="Store ID",y="Revenue",title="Store Performance")
        fig.update_layout(template="plotly_dark",height=390,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-title">Management Signals</div>',unsafe_allow_html=True)
    signals=[
        ("📈","Revenue engine",f"Top product {tp.iloc[0]['Product ID']} leads selected revenue." if len(tp) else "No product data."),
        ("📦","Inventory pressure",f"{critical} critical and {replenish} replenishment SKU/store combinations need attention."),
        ("🧊","Capital exposure",f"{overstock} combinations are flagged as potential overstock.")
    ]
    for icon,title,msg in signals:
        st.markdown(f'<div class="insight"><b>{icon} {title}</b><br><span class="muted">{msg}</span></div>',unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">🤖 AI Insights</div>',unsafe_allow_html=True)
    st.caption("Machine-learning layer for risk, anomalies, segmentation, predictive drivers and business scenarios.")
    st.markdown(f'<div class="insight"><b>🧠 AI Executive Copilot</b><br><span class="muted">{copilot_summary(f,risk,anomalies,segments)}</span></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("High-risk SKUs",int((risk["AI_Risk_Level"]=="HIGH").sum()))
    b.metric("Anomalies detected",int(anomalies["Anomaly"].sum()))
    c.metric("Portfolio segments",segments["Segment"].nunique() if len(segments) else 0)

    a,b=st.columns(2)
    with a:
        rr=risk.head(25)
        fig=px.scatter(rr,x="Demand",y="Inventory",size="AI_Risk_Score",color="AI_Risk_Level",hover_data=["Store ID","Product ID"],title="AI Risk Radar")
        fig.update_layout(template="plotly_dark",height=420,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        seg=segments.groupby("Segment",as_index=False).agg(SKUs=("Product ID","count"),Revenue=("Revenue","sum"))
        fig=px.bar(seg.sort_values("Revenue"),x="Revenue",y="Segment",orientation="h",text_auto=".2s",title="AI Portfolio Segmentation")
        fig.update_layout(template="plotly_dark",height=420,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

    if len(importance):
        fig=px.bar(importance.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Predictive Feature Contribution")
        fig.update_layout(template="plotly_dark",height=350,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Feature importance indicates predictive contribution, not causal impact.")

    st.markdown('<div class="section-title">🔬 Scenario Lab</div>',unsafe_allow_html=True)
    s1,s2,s3=st.columns(3)
    discount_delta=s1.slider("Discount change (pp)",-10,30,10)
    promotion=s2.selectbox("Promotion scenario",[0,1],format_func=lambda x:"No promotion" if x==0 else "Run promotion")
    price_delta=s3.slider("Price change (%)",-20,20,0)
    scenario,lift=scenario_demand(f,discount_delta,promotion,price_delta)
    baseline=float(f["Demand"].mean()) if len(f) else 0
    x1,x2,x3=st.columns(3)
    x1.metric("Baseline daily demand",f"{baseline:,.1f}")
    x2.metric("Scenario daily demand",f"{scenario:,.1f}",f"{scenario-baseline:+,.1f}")
    x3.metric("Observed promotion lift",f"{lift*100:+.1f}%")

    an=anomalies.head(12)
    if len(an):
        fig=px.scatter(an,x="Date",y="Demand",size="Anomaly_Score",color="Anomaly",title="Anomaly Detection")
        fig.update_layout(template="plotly_dark",height=350,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    st.markdown('<div class="section-title">📦 Inventory Plan</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Critical",critical)
    b.metric("Replenish",replenish)
    c.metric("Overstock",overstock)
    d.metric("Healthy",int((inv["Inventory_Status"]=="HEALTHY").sum()))
    a,b=st.columns(2)
    with a:
        cnt=inv["Inventory_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig=px.pie(cnt,names="Status",values="Count",hole=.55,title="Inventory Status Mix")
        fig.update_layout(template="plotly_dark",height=370,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        radar=inv.copy()
        for c in ["Demand Forecast","Inventory_Level","Recommended Order"]: radar[c]=pd.to_numeric(radar[c],errors="coerce").fillna(0)
        fig=px.scatter(radar,x="Demand Forecast",y="Inventory_Level",color="Inventory_Status",size="Recommended Order",hover_data=["Store ID","Product ID"],title="Demand vs Available Inventory")
        fig.update_layout(template="plotly_dark",height=370,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    display=inv[["Store ID","Product ID","Daily_Demand","Inventory_Level","Coverage Days","Demand Forecast","Recommended Order","Inventory_Status"]].head(50).copy()
    display.columns=["Store","Product","Avg Daily Demand","Inventory","Coverage Days","Forecast Demand","Order Qty","Status"]
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Export inventory decision queue",alerts.to_csv(index=False),"retail_inventory_actions.csv","text/csv")

with tabs[3]:
    st.markdown('<div class="section-title">🔮 Demand Forecast</div>',unsafe_allow_html=True)
    if len(f)>100:
        _,metrics,preds=chronological_model(f)
        a,b,c=st.columns(3)
        a.metric("MAE",f"{metrics['MAE']:.2f}"); b.metric("RMSE",f"{metrics['RMSE']:.2f}"); c.metric("R²",f"{metrics['R2']:.3f}")
        plot=preds.groupby("Date",as_index=False).agg(Actual=("Demand","sum"),Predicted=("Predicted_Demand","sum"))
        fig=px.line(plot,x="Date",y=["Actual","Predicted"],title="Chronological Holdout: Actual vs Predicted")
        fig.update_layout(template="plotly_dark",height=380,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        a,b=st.columns(2)
        store=a.selectbox("Forecast store",sorted(f["Store ID"].astype(str).unique()))
        prods=sorted(f[f["Store ID"].astype(str)==store]["Product ID"].astype(str).unique())
        prod=b.selectbox("Forecast product",prods)
        fc=forecast_next_days(f,store,prod,horizon)
        fig=px.line(fc,x="Date",y="Predicted_Demand",markers=True,title=f"{store} / {prod} — Next {horizon} Days")
        fig.update_layout(template="plotly_dark",height=350,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(fc,use_container_width=True,hide_index=True)
    else:
        st.warning("Select a larger data window for model evaluation.")

with tabs[4]:
    st.markdown('<div class="section-title">🧾 Transactions & Decision Queue</div>',unsafe_allow_html=True)
    st.caption("Search the underlying retail observations and download operational actions.")
    search=st.text_input("Search Store / Product / Category",placeholder="e.g. S001 or P0001")
    view=f.copy()
    if search:
        mask=(view["Store ID"].astype(str).str.contains(search,case=False,na=False) |
              view["Product ID"].astype(str).str.contains(search,case=False,na=False) |
              view["Category"].astype(str).str.contains(search,case=False,na=False))
        view=view[mask]
    cols=[c for c in ["Date","Store ID","Product ID","Category","Units Sold","Demand","Inventory Level","Price","Discount","Promotion","Revenue"] if c in view.columns]
    st.dataframe(view[cols].sort_values("Date",ascending=False).head(500),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download filtered transactions",view.to_csv(index=False),"retail_transactions.csv","text/csv")
    st.download_button("⬇️ Download complete action queue",alerts.to_csv(index=False),"retail_action_queue.csv","text/csv")

st.divider()
st.markdown('<div style="text-align:center;color:#697180;font-size:.75rem">Retail Store Sales & Inventory Performance Analytics • Capstone 2026 • Pradyuman Verma • SRMIST</div>',unsafe_allow_html=True)
