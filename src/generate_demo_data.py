"""Generate a demo dataset with the same schema as the retail inventory dataset."""
import numpy as np
import pandas as pd

def season(month: int) -> str:
    return ("Winter" if month in (12, 1, 2) else "Spring" if month in (3, 4, 5)
            else "Summer" if month in (6, 7, 8) else "Autumn")

def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-01-01")
    stores = [f"S{i:03d}" for i in range(1, 6)]
    products = [f"P{i:04d}" for i in range(1, 21)]
    cats = ["Groceries", "Toys", "Electronics", "Furniture", "Clothing"]
    regions = ["North", "South", "East", "West"]
    idx = pd.MultiIndex.from_product([dates, stores, products], names=["Date", "Store ID", "Product ID"]).to_frame(index=False)
    idx["Category"] = idx["Product ID"].map({p: cats[i % 5] for i, p in enumerate(products)})
    idx["Region"] = idx["Store ID"].map({s: regions[i % 4] for i, s in enumerate(stores)})
    pop = {p: rng.uniform(0.5, 2.2) for p in products}
    size = {s: rng.uniform(0.8, 1.3) for s in stores}
    price0 = {p: rng.uniform(10, 100) for p in products}
    cover = {(s, p): rng.choice([0.7, 1.5, 3, 6, 12, 25, 60, 120], p=[.06, .08, .10, .15, .20, .20, .13, .08]) for s in stores for p in products}
    n = len(idx); month = idx["Date"].dt.month.values; dow = idx["Date"].dt.dayofweek.values
    seas = np.array([season(m) for m in range(1, 13)])[month - 1]
    seas_f = np.select([seas == "Winter", seas == "Summer", seas == "Autumn"], [1.15, 1.10, 0.95], 1.0)
    toy_peak = np.where((idx["Category"] == "Toys") & (month >= 11), 1.6, 1.0)
    promo = rng.binomial(1, 0.2, n)
    discount = rng.choice([0, 5, 10, 15, 20], n, p=[.4, .15, .2, .15, .1])
    price = idx["Product ID"].map(price0).values * rng.uniform(0.95, 1.05, n)
    comp = price * rng.uniform(0.9, 1.1, n)
    trend = 1 + 0.15 * (idx["Date"] - dates[0]).dt.days.values / 730
    mean_d = (idx["Product ID"].map(pop).values * idx["Store ID"].map(size).values * 12 * seas_f * toy_peak * trend *
              (1 + 0.25 * promo) * (1 + 0.012 * discount) * np.where(dow >= 5, 1.2, 1.0))
    demand = rng.poisson(mean_d * rng.uniform(0.9, 1.1, n))
    cov = np.array([cover[k] for k in zip(idx["Store ID"], idx["Product ID"])])
    inventory = np.maximum(1, (mean_d * cov * rng.uniform(0.6, 1.4, n)).astype(int))
    sold = np.minimum(demand, inventory)
    ordered = np.maximum(0, (mean_d * 5 - inventory) * rng.uniform(0.7, 1.2, n)).astype(int)
    return pd.DataFrame({
        "Date": idx["Date"].dt.strftime("%Y-%m-%d"), "Store ID": idx["Store ID"], "Product ID": idx["Product ID"],
        "Category": idx["Category"], "Region": idx["Region"], "Inventory Level": inventory, "Units Sold": sold,
        "Units Ordered": ordered, "Demand Forecast": (mean_d * rng.uniform(0.85, 1.15, n)).round(2),
        "Price": price.round(2), "Discount": discount, "Weather Condition": rng.choice(["Sunny","Rainy","Cloudy","Snowy"], n, p=[.35,.25,.3,.1]),
        "Holiday/Promotion": promo, "Competitor Pricing": comp.round(2), "Seasonality": seas
    })
