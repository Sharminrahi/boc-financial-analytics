# Bank of Canada Financial Markets Analytics

Automated analytical workflows using public Bank of Canada data.  
No API key required — uses the free [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs).

![CI](https://github.com/Sharminrahi/boc-financial-analytics/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)

---

## What it does

| Workflow | Description |
|---|---|
| **Yield curve construction** | Government of Canada benchmark yields (2y, 5y, 10y, 30y) fetched daily |
| **2s10s spread analysis** | 2-year vs 10-year spread in basis points — primary recession indicator |
| **Term premium modelling** | 10-year yield minus overnight policy rate |
| **Yield curve regime** | deeply inverted / inverted / flat / normal / steep |
| **CAD/USD FX analysis** | Log returns, annualised 20d and 60d realised volatility, GARCH-inspired regime |
| **Cross-market correlation** | Rolling 60d correlation and 5-day lead-lag between 2s10s spread and FX returns |
| **Data quality pipeline** | Null rate monitoring, outlier detection, stale data flagging, structured quality report |

---

## Project structure

```
boc-financial-analytics/
├── src/
│   └── boc_financial_analytics.py   # Main analytics module (~430 lines)
├── tests/
│   └── test_financial_analytics.py  # 25 unit tests (fully offline)
├── data/                            # Output CSV and JSON files — gitignored
├── requirements.txt
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── ci.yml                   # Runs on every push to main
```

---

## Quickstart

### Step 1 — Clone

```bash
git clone https://github.com/Sharminrahi/boc-financial-analytics.git
cd boc-financial-analytics
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the pipeline

```bash
python src/boc_financial_analytics.py
```

**What happens:**
- Fetches live data from the Bank of Canada Valet API (free, no key needed)
- Falls back to synthetic data automatically if offline
- Runs all four analytical workflows
- Prints JSON summary report to console
- Writes CSV and JSON outputs to `data/`

**Expected console output:**
```
2024-01-15 [INFO] Building market dataset | 2020-01-01 -> 2024-01-15
2024-01-15 [INFO] Yield curve | 2s10s: -45.2bps | regime: inverted
2024-01-15 [INFO] ADF | stat=-18.432 p=0.0000 (stationary)
2024-01-15 [INFO] FX | CAD/USD: 0.7412 | 20d vol: 6.8%

Curve regime : inverted
2s10s spread : -45.2 bps
CAD/USD      : 0.7412
20d FX vol   : 6.8%
Outputs written to data/
```

### Step 5 — Run the tests

```bash
pytest tests/ -v
```

All 25 tests run fully offline. Expected result: `25 passed`.

---

## Outputs

| File | Contents |
|---|---|
| `data/market_data.csv` | Raw series: yields, policy rate, CAD/USD |
| `data/yield_curve_analysis.csv` | Spreads, regime, term premium, z-score, inversion counter |
| `data/fx_analysis.csv` | Log returns, realised vol (20d/60d), vol regime |
| `data/cross_market_analysis.csv` | Rolling correlations, lead-lag analysis |
| `data/summary_report.json` | Structured JSON summary of current market conditions |

---

## Data source

All data from the **Bank of Canada Valet API** — free, no registration required.

| Series | Valet ID | Description |
|---|---|---|
| 2-year GoC yield | V122540 | Government of Canada 2-year benchmark |
| 5-year GoC yield | V122541 | Government of Canada 5-year benchmark |
| 10-year GoC yield | V122543 | Government of Canada 10-year benchmark |
| 30-year GoC yield | V122544 | Government of Canada 30-year benchmark |
| Overnight policy rate | V39079 | Bank of Canada policy rate |
| CAD/USD exchange rate | FXCADUSD | Canadian dollar noon rate |

API docs: https://www.bankofcanada.ca/valet/docs

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **Log returns** | Time-additive and approximately normal for small moves — correct for volatility and correlation |
| **ADF stationarity test** | Non-stationary series produce spurious correlations — ADF confirms stationarity before cross-market analysis |
| **Forward-fill max 5 days** | Handles weekends and holidays without introducing staleness beyond practitioner tolerance |
| **Synthetic fallback** | Pipeline runs fully offline in CI — deterministic, reproducible |
| **Rolling z-score outliers (window=90)** | Detects anomalies relative to local context — appropriate for non-stationary financial series |

---

## CI/CD

Every push to `main` runs GitHub Actions:
1. Install all dependencies
2. Run 25 pytest unit tests
3. Run full pipeline in synthetic data mode

---

## Push to GitHub

```bash
cd boc-financial-analytics
git init
git add .
git commit -m "feat: Bank of Canada financial analytics — yield curve, FX volatility, cross-market, 25 tests"
git branch -M main
git remote add origin https://github.com/Sharminrahi/boc-financial-analytics.git
git push -u origin main
```

> Create the repo at github.com/new first — leave it **empty** (no README, no .gitignore).

---

## Author

**Sharmin Akhter** | github.com/Sharminrahi
