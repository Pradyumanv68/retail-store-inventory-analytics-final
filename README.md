# 📦 Retail Store Sales & Inventory Performance Analytics

> An intelligent retail decision-support system that turns historical sales and inventory data into **demand forecasts, inventory-risk alerts and replenishment recommendations**.

**Author:** Pradyuman Verma · Capstone Project · Data Science / Business Analytics

## Business problem
Retailers lose money in two opposite ways: **stockouts** (lost sales) and **overstock** (locked-up cash). This project answers, for every store-product pair:
*How is it selling? How many days of stock are left? What will demand be next week? How much should we order, or should we discount instead?*

## What's inside
| Stage | What it does | Code |
|---|---|---|
| 1. Cleaning | column standardisation, dates, duplicates, imputation (per store-product), IQR outlier flag | `src/data_cleaning.py` |
| 2. Feature engineering | year/month/week/day/day-of-week, revenue, discount %, stock-coverage days, stockout flag, lag & rolling demand features | `src/data_cleaning.py`, `src/forecasting.py` |
| 3. Sales analytics | revenue/units by store, category, product, month, season, weekday; promotion & discount impact | `src/analytics.py` |
| 4. Inventory risk engine | days of cover vs lead time -> **Critical / High Risk / Medium / Healthy / Overstock** + risk score, lost-sales analysis | `src/analytics.py` |
| 5. Movers | **Fast / Medium / Slow / Dead-stock** classification by sales velocity, plus annualised inventory turnover | `src/analytics.py` |
| 6. Forecasting | Baseline vs Linear Regression vs Random Forest vs Gradient Boosting, compared on MAE / RMSE / MAPE / R² with a **time-based split** (no future leakage) | `src/forecasting.py` |
| 7. 7-day demand forecast | recursive multi-step forecast per store-product | `src/forecasting.py` |
| 8. Recommendation engine | `order = forecast_7d + safety_stock − current_inventory`, safety stock = z·σ·√lead-time; auto-generated action text | `src/recommendations.py` |
| 9. Dashboard | 4-page Streamlit app + order calculator + CSV export | `dashboard/app.py` |

## Key findings (real dataset, 76,000 records)
- **17.5% of demand goes unmet** (about $83.7M of lost revenue): stock on hand averages only ~3 days of demand, so stockouts are the biggest problem, not overstock.
- **11.4%** of store-product-days end in a stockout, yet a few items sit on 8+ days of cover, so stock is misallocated as well as short.
- Of 100 store-product pairs, **28 are Critical and 33 High Risk** today; 6 are overstocked. The action queue ranks them by urgency with an order quantity for each.
- Forecasting **true demand** (not sales, which are capped by stock) is the right target, because sales understate what customers wanted.

## Model results (last 90 days held out, target = daily demand)
| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| Baseline (7-day moving avg) | 32.12 | 41.22 | 45.9 | 0.134 |
| Linear Regression | 25.94 | 33.44 | 37.3 | 0.430 |
| Random Forest | 21.75 | 29.21 | 31.5 | 0.565 |
| **Gradient Boosting** | **14.22** | **19.61** | **21.7** | **0.804** |

Gradient Boosting cuts RMSE by about 52% versus the naive moving-average baseline.

Example generated recommendations:
> `CRITICAL: P0009 @ S001 - Demand is rising 4%. Inventory (128) does not cover the 7-day forecast (1036). Order ~1042 units.`
> `OVERSTOCK: P0014 @ S002 holds 8 days of cover. Pause replenishment and run a promotional discount / inter-store transfer to release cash.`

## Visuals
![Monthly revenue](visualizations/monthly_revenue.png)
![Days of cover heatmap](visualizations/days_of_cover_heatmap.png)
![Risk distribution](visualizations/risk_distribution.png)
![Model comparison](visualizations/model_comparison.png)
![Actual vs predicted](visualizations/actual_vs_predicted.png)

## Run it
```bash
git clone https://github.com/Pradyumanv68/Retail-Store-Sales-Inventory-Performance-Analytics.git
cd Retail-Store-Sales-Inventory-Performance-Analytics
pip install -r requirements.txt
python run_pipeline.py            # clean -> analyse -> train -> forecast -> recommend -> figures
streamlit run dashboard/app.py    # interactive dashboard
pytest tests                      # unit tests
```

## Dataset
`data/sales_data.csv`: retail store inventory and demand data (76,000 rows; 5 stores × 20 products × 760 days, 2022-01-01 to 2024-01-30) with Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic and **Demand**. No missing values or duplicates were found.
The pipeline also accepts the other common Kaggle version of this dataset (with `Holiday/Promotion` and `Demand Forecast` columns): drop the file into `data/` and re-run `python run_pipeline.py`. If no file is present, a synthetic demo file with the same schema is generated.

## Project structure
```
├── data/                 sales_data.csv + processed/clean.csv.gz
├── notebooks/            Retail_Analytics_Walkthrough.ipynb (executed)
├── src/                  data_cleaning, analytics, forecasting, recommendations, generate_demo_data
├── dashboard/app.py      Streamlit app
├── reports/              model_metrics, forecast_next_7_days, inventory_actions (CSV)
├── visualizations/       exported charts
├── tests/                pytest suite
├── run_pipeline.py       one-command pipeline
└── requirements.txt
```

## Methodology notes
- **Days of cover** = current inventory ÷ 30-day average daily demand. Risk is set relative to the replenishment lead time L (default 3 days): <0.5L Critical, <L High Risk, <2L Medium, up to 2.5L Healthy, above that Overstock.
- **Forecast target is true demand**; unmet demand = demand − units sold, which gives lost sales and lost revenue.
- **Time-based validation** avoids the leakage a random split would cause in time-series data.
- **Outliers are flagged, not removed**, since demand spikes are genuine business events.
- Limitations: forecasts assume no future promotion; lead time and service level are configurable assumptions; no supplier constraints or holding costs.

## Future work
Prophet/ARIMA per-series models, hyper-parameter tuning, cost-based (EOQ) ordering, anomaly detection, Power BI companion dashboard.

## License
MIT

## Deploy the dashboard (Streamlit Community Cloud)
1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `dashboard/app.py`.
3. Deploy. The app reads the committed `data/processed/clean.csv.gz` and `reports/*.csv`, so no extra setup is needed.
Add your live link here: `https://<your-app-name>.streamlit.app`
