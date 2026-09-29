# 📦 Retail Intelligence Command Center

> **Retail Store Sales & Inventory Performance Analytics** — an AI-powered decision-intelligence capstone that connects sales analytics, demand forecasting, inventory risk, anomaly detection, scenario simulation and replenishment actions.

**Author:** Pradyuman Verma · B.Tech CSE (Data Science), SRMIST · Capstone 2026

## 🚀 What the system does

**OBSERVE → DIAGNOSE → PREDICT → SIMULATE → DECIDE → ACT**

| Layer | Capability |
|---|---|
| Executive Analytics | Revenue, demand, sales, inventory, stock cover and demand-gap KPIs |
| Sales Intelligence | Store, category, product and promotion performance |
| Forecasting | Chronological ML validation with MAE, RMSE, MAPE and R² |
| Model Benchmarking | Moving-average baseline vs Linear Regression vs Random Forest vs Gradient Boosting |
| Inventory Risk | Critical / High Risk / Medium / Healthy / Overstock |
| AI Risk Radar | 0–100 SKU/store risk score using coverage, demand gap and price pressure |
| Anomaly Detection | Isolation Forest for unusual demand/revenue/inventory patterns |
| SKU Segmentation | K-Means portfolio segmentation |
| Predictive Drivers | Random Forest feature contribution for demand |
| Scenario Lab | Promotion, discount and price what-if simulation |
| Replenishment | Safety stock + forecast-driven recommended order |
| Action Center | Prioritized operational actions with CSV export |
| AI Copilot | Concise executive decision brief from current model outputs |

## 🧠 Why it is different

A conventional dashboard tells a manager **what happened**.

This project is designed to answer:

- What is happening?
- Why should I care?
- Which SKU/store needs attention?
- What could happen next?
- What if I change price or promotion?
- How much should I order?
- Which action should happen first?

The result is a **decision-support prototype**, not only a visualization dashboard.

## 🏗️ Architecture

```
Retail Dataset
      ↓
Cleaning + Feature Engineering
      ↓
Sales / Demand Analytics
      ↓
ML Model Benchmarking
      ↓
Demand Forecast
      ↓
Inventory + AI Risk Engine
      ↓
Anomaly Detection + SKU Segmentation
      ↓
Scenario Lab + Executive Copilot
      ↓
Replenishment Recommendations
      ↓
Action Center
```

## 📁 Repository structure

```
├── app/
│   ├── app.py                  # Main AI Streamlit Command Center
│   ├── analytics.py            # Dashboard analytics
│   ├── forecasting.py          # Dashboard demand forecasting
│   ├── recommendations.py      # Inventory recommendations
│   └── ai_engine.py            # Risk, anomaly, segmentation & scenario AI
├── dashboard/
│   └── app.py                  # Streamlit compatibility entry point
├── src/
│   ├── data_cleaning.py
│   ├── analytics.py
│   ├── forecasting.py
│   ├── recommendations.py
│   └── generate_demo_data.py
├── tests/
│   └── test_pipeline.py
├── reports/
├── visualizations/
├── data/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

The original complete project archive is also retained in the repository as a reference package.

## 📊 Forecasting methodology

The forecasting pipeline uses a **time-based holdout**, keeping future observations out of training.

Models include:

- 7-day moving-average baseline
- Linear Regression
- Random Forest
- Gradient Boosting

Evaluation:

- MAE
- RMSE
- MAPE
- R²

The forecasting target is **true Demand when available**, rather than treating realized sales as demand. This matters because stockouts can cap sales below what customers actually wanted.

## 📦 Inventory intelligence

For each store-product combination, the system calculates:

- current inventory
- average demand
- demand variability
- days of cover
- stockout exposure
- risk level
- safety stock
- forecast demand
- recommended order quantity
- operational action

The recommendation logic is explicitly presented as a planning aid rather than a guaranteed business outcome.

## 🤖 AI layer

### AI Risk Radar
Produces a 0–100 risk score from:

- inventory coverage
- demand gap
- competitive price pressure

### Anomaly Detection
Isolation Forest identifies unusual combinations of:

- revenue
- demand
- units sold
- inventory
- price

### Portfolio Segmentation
K-Means groups SKU/store combinations into operational segments such as:

- Value Builder
- Growth Driver
- Stable Core
- Slow Mover

### Predictive Feature Contribution
Random Forest estimates the relative predictive contribution of available demand drivers.

### Scenario Lab
Allows interactive what-if planning around:

- discount changes
- promotions
- price changes

Scenario outputs are simulations based on observed relationships, not causal guarantees.

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Alternative Streamlit entry point:

```bash
streamlit run dashboard/app.py
```

Run the analytical pipeline:

```bash
python run_pipeline.py
```

Run tests:

```bash
pytest tests
```

## 🌐 Deployment

For Streamlit Community Cloud, use:

- Repository: `Pradyumanv68/retail-store-inventory-analytics-final`
- Branch: `main`
- Main file: `app/app.py`

The dashboard can use the packaged retail dataset when available and has a public-data fallback plus demo-data fallback so the application remains deployable.

## ⚠️ Analytical limitations

- Lead time and safety-stock service level are configurable assumptions.
- Scenario simulation is not causal inference.
- Feature importance represents predictive contribution, not causation.
- Supplier constraints, procurement cost and full EOQ economics are outside the current scope.
- Future promotions are not known by the forecasting model unless explicitly supplied.

## 📜 License

MIT
