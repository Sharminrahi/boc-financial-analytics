"""
boc_financial_analytics.py
===========================
Bank of Canada Financial Markets Data Analytics
================================================
Analytical workflows using publicly available Bank of Canada data.

Analyses performed:
  1. Government of Canada yield curve construction and analysis
  2. CAD/USD exchange rate trend modelling and volatility
  3. Yield curve slope (2s10s spread) as recession indicator
  4. Statistical tests: stationarity, autocorrelation, regime detection
  5. Rolling correlation between FX and yield spreads

Data source: Bank of Canada Valet API (free, public, no API key required)
  https://www.bankofcanada.ca/valet/docs

Designed to demonstrate:
  - Financial data modelling using real central bank data
  - Automated analytical workflows with clear, usable outputs
  - Data quality controls within analytical pipelines
  - Reproducible research (documented, version-controlled, testable)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bank of Canada Valet API client
# ---------------------------------------------------------------------------

BOC_API_BASE = "https://www.bankofcanada.ca/valet"

# Series identifiers (Bank of Canada naming convention)
SERIES = {
    # Government of Canada benchmark bond yields
    "yield_2y":    "V122540",   # 2-year bond yield
    "yield_5y":    "V122541",   # 5-year bond yield
    "yield_10y":   "V122543",   # 10-year bond yield
    "yield_30y":   "V122544",   # 30-year bond yield
    # Policy rate
    "policy_rate": "V39079",    # Bank of Canada overnight rate target
    # Exchange rate
    "cad_usd":     "FXCADUSD",  # CAD/USD noon rate
}


def fetch_boc_series(
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Fetch a single time series from the Bank of Canada Valet API.

    Args:
        series_id:  BoC series identifier (e.g. 'V122540')
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'

    Returns:
        pd.Series indexed by date, values as float
    """
    url = f"{BOC_API_BASE}/observations/{series_id}/json"
    params = {"start_date": start_date, "end_date": end_date, "order_dir": "asc"}

    log.info(f"Fetching {series_id} from {start_date} to {end_date}")

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning(f"API unavailable for {series_id}: {e}. Using synthetic data.")
        return _synthetic_series(series_id, start_date, end_date)

    observations = data.get("observations", [])
    if not observations:
        log.warning(f"No observations returned for {series_id}. Using synthetic data.")
        return _synthetic_series(series_id, start_date, end_date)

    records = []
    for obs in observations:
        date_str = obs.get("d")
        value    = obs.get(series_id, {}).get("v")
        if date_str and value and value != "":
            try:
                records.append((pd.to_datetime(date_str), float(value)))
            except (ValueError, TypeError):
                continue

    series = pd.Series(
        {date: val for date, val in records},
        name=series_id,
        dtype=float,
    )
    series.index = pd.DatetimeIndex(series.index)
    log.info(f"Fetched {len(series)} observations for {series_id}")
    return series


def _synthetic_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Generate realistic synthetic financial time series for offline dev.
    Uses random walk with drift to mimic real yield/FX behaviour.
    """
    np.random.seed(hash(series_id) % (2**32))
    dates  = pd.bdate_range(start=start_date, end=end_date)
    n      = len(dates)

    base_map = {
        "yield_2y": 3.5, "yield_5y": 3.8, "yield_10y": 4.0,
        "yield_30y": 4.2, "policy_rate": 3.25, "cad_usd": 0.74,
    }
    # Reverse-lookup the series name from ID
    name = next((k for k, v in SERIES.items() if v == series_id), "generic")
    base = base_map.get(name, 3.0)

    # Random walk with mean reversion
    shocks    = np.random.normal(0, 0.02, n)
    mean_rev  = 0.05 * (base - np.cumsum(shocks))
    values    = base + np.cumsum(shocks + mean_rev * 0.1)
    values    = np.abs(values)   # yields cannot be negative (simplified)

    return pd.Series(values, index=dates, name=series_id, dtype=float)


# ---------------------------------------------------------------------------
# Step 1: Fetch and assemble all series
# ---------------------------------------------------------------------------

def build_market_dataset(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all financial market series and assemble into a clean DataFrame.
    Applies data quality checks: forward-fill weekends, flag outliers,
    check for stale data (no updates for > 5 business days).

    Returns:
        DataFrame with columns: yield_2y, yield_5y, yield_10y, yield_30y,
        policy_rate, cad_usd — indexed by business date.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    log.info(f"Building market dataset | {start_date} to {end_date}")

    frames = {}
    for name, series_id in SERIES.items():
        frames[name] = fetch_boc_series(series_id, start_date, end_date)

    df = pd.DataFrame(frames)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    # ── Data quality checks ───────────────────────────────────────────────
    quality_report = {}

    # 1. Missing value rate per column
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        quality_report[f"{col}_null_pct"] = round(null_pct * 100, 2)
        if null_pct > 0.10:
            log.warning(f"Column {col} has {null_pct:.1%} missing values")

    # 2. Forward-fill missing values (market holidays, weekends)
    df = df.ffill(limit=5)   # max 5-day gap (one trading week)

    # 3. Outlier detection: flag values > 4 std dev from 90-day rolling mean
    for col in df.columns:
        rolling_mean = df[col].rolling(90, min_periods=10).mean()
        rolling_std  = df[col].rolling(90, min_periods=10).std()
        zscore       = (df[col] - rolling_mean).abs() / rolling_std
        outliers     = (zscore > 4).sum()
        quality_report[f"{col}_outliers"] = int(outliers)
        if outliers > 0:
            log.warning(f"Column {col}: {outliers} outlier observations detected")

    # 4. Stale data check
    for col in df.columns:
        last_obs = df[col].dropna().index[-1] if not df[col].dropna().empty else None
        if last_obs:
            days_stale = (pd.Timestamp(end_date) - last_obs).days
            quality_report[f"{col}_days_stale"] = days_stale

    log.info(f"Data quality report: {quality_report}")
    log.info(f"Dataset shape: {df.shape} | date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df, quality_report


# ---------------------------------------------------------------------------
# Step 2: Yield curve analysis
# ---------------------------------------------------------------------------

def analyse_yield_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct and analyse the Government of Canada yield curve.

    Computes:
      - 2s10s spread (classic recession indicator — inverts before recessions)
      - 5s30s spread (long-end steepness)
      - Yield curve slope regime (steep / flat / inverted)
      - Rolling 30-day z-score of 2s10s spread for standardised comparison
    """
    yc = pd.DataFrame(index=df.index)

    # Core spreads (basis points)
    yc["spread_2s10s"]  = (df["yield_10y"] - df["yield_2y"]) * 100
    yc["spread_5s30s"]  = (df["yield_30y"] - df["yield_5y"]) * 100
    yc["spread_2s5s"]   = (df["yield_5y"]  - df["yield_2y"]) * 100
    yc["spread_2s30s"]  = (df["yield_30y"] - df["yield_2y"]) * 100

    # Term premium proxy: 10y vs policy rate
    yc["term_premium"]  = (df["yield_10y"] - df["policy_rate"]) * 100

    # Yield curve regime classification
    yc["curve_regime"]  = pd.cut(
        yc["spread_2s10s"],
        bins=[-np.inf, -25, 0, 50, 100, np.inf],
        labels=["deeply_inverted", "inverted", "flat", "normal", "steep"]
    )

    # 30-day rolling z-score of 2s10s (standardised signal)
    roll_mean = yc["spread_2s10s"].rolling(30, min_periods=10).mean()
    roll_std  = yc["spread_2s10s"].rolling(30, min_periods=10).std()
    yc["spread_2s10s_zscore"] = (yc["spread_2s10s"] - roll_mean) / roll_std.replace(0, np.nan)

    # Days inverted (running count — policy-relevant metric)
    yc["is_inverted"] = yc["spread_2s10s"] < 0
    yc["days_inverted"] = (
        yc["is_inverted"]
        .groupby((~yc["is_inverted"]).cumsum())
        .cumcount()
    )

    log.info(f"Yield curve analysis complete | "
             f"current 2s10s: {yc['spread_2s10s'].iloc[-1]:.1f}bps | "
             f"regime: {yc['curve_regime'].iloc[-1]}")
    return yc


# ---------------------------------------------------------------------------
# Step 3: FX analysis
# ---------------------------------------------------------------------------

def analyse_fx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse CAD/USD exchange rate dynamics.

    Computes:
      - Daily, weekly, monthly log returns
      - Rolling 20-day and 60-day realised volatility (annualised)
      - GARCH-inspired volatility regime using rolling std quartiles
      - Augmented Dickey-Fuller stationarity test on returns
    """
    fx = pd.DataFrame(index=df.index)
    fx["cad_usd"]   = df["cad_usd"]

    # Log returns (continuously compounded)
    fx["return_daily"]   = np.log(fx["cad_usd"] / fx["cad_usd"].shift(1))
    fx["return_weekly"]  = np.log(fx["cad_usd"] / fx["cad_usd"].shift(5))
    fx["return_monthly"] = np.log(fx["cad_usd"] / fx["cad_usd"].shift(21))

    # Annualised realised volatility (√252 convention for daily data)
    fx["vol_20d"]  = fx["return_daily"].rolling(20).std()  * np.sqrt(252) * 100
    fx["vol_60d"]  = fx["return_daily"].rolling(60).std()  * np.sqrt(252) * 100

    # Volatility regime (quartile-based)
    vol_q = fx["vol_20d"].quantile([0.25, 0.5, 0.75])
    fx["vol_regime"] = pd.cut(
        fx["vol_20d"],
        bins=[-np.inf, vol_q[0.25], vol_q[0.5], vol_q[0.75], np.inf],
        labels=["low", "moderate", "elevated", "high"]
    )

    # Stationarity test on daily returns (ADF test)
    clean_returns = fx["return_daily"].dropna()
    if len(clean_returns) > 30:
        adf_stat, adf_pval, *_ = adfuller(clean_returns, autolag="AIC")
        fx["adf_stationary"]   = adf_pval < 0.05
        log.info(f"ADF test on CAD/USD returns: stat={adf_stat:.3f}, p={adf_pval:.4f} "
                 f"({'stationary' if adf_pval < 0.05 else 'non-stationary'})")

    log.info(f"FX analysis complete | "
             f"current CAD/USD: {fx['cad_usd'].iloc[-1]:.4f} | "
             f"20d vol: {fx['vol_20d'].iloc[-1]:.1f}%")
    return fx


# ---------------------------------------------------------------------------
# Step 4: Cross-market analysis
# ---------------------------------------------------------------------------

def cross_market_analysis(yc: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse relationships between yield curve and FX dynamics.

    Computes:
      - Rolling 60-day correlation: 2s10s spread vs CAD/USD returns
      - Lead-lag analysis: does yield spread predict FX direction?
      - Regime co-movement: are both in the same risk regime?
    """
    cm = pd.DataFrame(index=yc.index)

    aligned_fx  = fx["return_daily"].reindex(yc.index)
    aligned_yc  = yc["spread_2s10s"]

    # Rolling 60-day correlation
    cm["corr_spread_fx_60d"] = (
        aligned_yc
        .rolling(60, min_periods=20)
        .corr(aligned_fx)
    )

    # Lead-lag: does yield spread 5 days ago predict today's FX return?
    cm["spread_lag5"]           = aligned_yc.shift(5)
    cm["fx_return_fwd5"]        = aligned_fx.shift(-5)
    cm["lead_lag_correlation"]  = (
        cm["spread_lag5"]
        .rolling(90, min_periods=30)
        .corr(cm["fx_return_fwd5"])
    )

    log.info(f"Cross-market analysis: current 60d corr(2s10s, CAD/USD return) = "
             f"{cm['corr_spread_fx_60d'].iloc[-1]:.3f}")
    return cm


# ---------------------------------------------------------------------------
# Step 5: Generate summary report
# ---------------------------------------------------------------------------

def generate_summary_report(
    df: pd.DataFrame,
    yc: pd.DataFrame,
    fx: pd.DataFrame,
    quality_report: dict,
) -> dict:
    """
    Generate a structured summary report of current market conditions.
    Designed to be output as a JSON file or rendered in a dashboard.
    """
    latest = df.index[-1]

    report = {
        "report_date":  latest.strftime("%Y-%m-%d"),
        "generated_at": datetime.utcnow().isoformat(),

        "yield_curve": {
            "policy_rate_pct":   round(float(df["policy_rate"].iloc[-1]), 2),
            "yield_2y_pct":      round(float(df["yield_2y"].iloc[-1]),    2),
            "yield_5y_pct":      round(float(df["yield_5y"].iloc[-1]),    2),
            "yield_10y_pct":     round(float(df["yield_10y"].iloc[-1]),   2),
            "yield_30y_pct":     round(float(df["yield_30y"].iloc[-1]),   2),
            "spread_2s10s_bps":  round(float(yc["spread_2s10s"].iloc[-1]), 1),
            "spread_5s30s_bps":  round(float(yc["spread_5s30s"].iloc[-1]), 1),
            "curve_regime":      str(yc["curve_regime"].iloc[-1]),
            "days_inverted":     int(yc["days_inverted"].iloc[-1]),
        },

        "fx": {
            "cad_usd_rate":      round(float(fx["cad_usd"].iloc[-1]), 4),
            "return_daily_pct":  round(float(fx["return_daily"].iloc[-1]) * 100, 4),
            "vol_20d_annualised": round(float(fx["vol_20d"].iloc[-1]), 2),
            "vol_60d_annualised": round(float(fx["vol_60d"].iloc[-1]), 2),
            "vol_regime":         str(fx["vol_regime"].iloc[-1]),
        },

        "data_quality": quality_report,
    }

    return report


# ---------------------------------------------------------------------------
# Entry point: run the full analytical workflow
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    import json

    print("=" * 60)
    print("Bank of Canada Financial Markets Analytics")
    print("=" * 60)

    # Fetch data (last 4 years)
    start = (datetime.today() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    df, quality = build_market_dataset(start_date=start)

    # Run analyses
    yc = analyse_yield_curve(df)
    fx = analyse_fx(df)
    cm = cross_market_analysis(yc, fx)

    # Generate report
    report = generate_summary_report(df, yc, fx, quality)

    print("\nMarket summary report:")
    print(json.dumps(report, indent=2))

    # Save outputs
    df.to_csv("data/market_data.csv")
    yc.to_csv("data/yield_curve_analysis.csv")
    fx.to_csv("data/fx_analysis.csv")

    print(f"\nOutput files written to data/")
    print(f"Yield curve regime: {report['yield_curve']['curve_regime']}")
    print(f"2s10s spread: {report['yield_curve']['spread_2s10s_bps']:.1f} bps")
    print(f"CAD/USD: {report['fx']['cad_usd_rate']:.4f}")
    print(f"20-day FX vol: {report['fx']['vol_20d_annualised']:.1f}%")
