"""Step 8: turn forecast + inventory position into replenishment actions."""
import numpy as np
from .data_cleaning import KEYS

def _action(r):
    who=f"{r.product_id} @ {r.store_id}"
    if r.risk=="Overstock":
        return f"OVERSTOCK: {who} holds {r.days_of_cover:.0f} days of cover. Pause replenishment and run a promotional discount / inter-store transfer to release cash."
    if r.recommended_order>0:
        trend=f"Demand is {'rising' if r.demand_change_pct>0 else 'falling'} {abs(r.demand_change_pct):.0f}%. "
        return f"{r.risk.upper()}: {who} - {trend}Inventory ({r.current_inventory:.0f}) does not cover the 7-day forecast ({r.forecast_7d:.0f}). Order ~{r.recommended_order:.0f} units."
    return f"OK: {who} - stock ({r.current_inventory:.0f}) covers forecast demand ({r.forecast_7d:.0f})."

def build_recommendations(sku,fc,lead_time_days=3,service_z=1.65):
    f=fc.groupby(KEYS)["predicted_units"].sum().rename("forecast_7d").reset_index()
    d=sku.merge(f,on=KEYS)
    d["demand_change_pct"]=(d["forecast_7d"]/7/d["avg_daily_demand"].clip(lower=.1)-1)*100
    d["safety_stock"]=np.ceil(service_z*d["std_daily_demand"]*np.sqrt(lead_time_days))
    d["recommended_order"]=np.ceil((d["forecast_7d"]+d["safety_stock"]-d["current_inventory"]).clip(lower=0))
    d["action"]=d.apply(_action,axis=1)
    order={"Critical":0,"High Risk":1,"Medium":2,"Healthy":3,"Overstock":4}
    return d.sort_values(["risk","recommended_order"],key=lambda s:s.map(order) if s.name=="risk" else -s).reset_index(drop=True)
