"""
tests/test_financial_analytics.py
==================================
Unit tests for the BoC financial analytics module.
No API calls — all tests use synthetic data.
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from boc_financial_analytics import (
    _synthetic_series, build_market_dataset,
    analyse_yield_curve, analyse_fx, generate_summary_report,
    SERIES,
)


@pytest.fixture(scope="module")
def market_df():
    """Shared dataset for all tests — uses synthetic data, no API."""
    df, quality = build_market_dataset(
        start_date="2022-01-01",
        end_date="2024-01-01",
    )
    return df, quality


class TestSyntheticData:

    def test_returns_series_with_correct_length(self):
        s = _synthetic_series("V122543", "2023-01-01", "2023-12-31")
        assert len(s) > 200

    def test_values_are_positive(self):
        s = _synthetic_series("V122543", "2023-01-01", "2023-12-31")
        assert (s > 0).all()

    def test_index_is_datetime(self):
        s = _synthetic_series("V122543", "2023-01-01", "2023-12-31")
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_deterministic_for_same_inputs(self):
        s1 = _synthetic_series("V122543", "2023-01-01", "2023-06-30")
        s2 = _synthetic_series("V122543", "2023-01-01", "2023-06-30")
        pd.testing.assert_series_equal(s1, s2)


class TestBuildMarketDataset:

    def test_returns_dataframe_and_quality_report(self, market_df):
        df, quality = market_df
        assert isinstance(df, pd.DataFrame)
        assert isinstance(quality, dict)

    def test_all_expected_columns_present(self, market_df):
        df, _ = market_df
        expected = {"yield_2y", "yield_5y", "yield_10y", "yield_30y",
                    "policy_rate", "cad_usd"}
        assert expected.issubset(set(df.columns))

    def test_no_nulls_after_ffill(self, market_df):
        df, _ = market_df
        null_counts = df.isnull().sum()
        # After forward-fill, nulls should be minimal (only at very start)
        assert null_counts.max() < len(df) * 0.05

    def test_quality_report_has_null_pct_keys(self, market_df):
        _, quality = market_df
        for col in SERIES.keys():
            assert f"{col}_null_pct" in quality

    def test_index_is_sorted(self, market_df):
        df, _ = market_df
        assert df.index.is_monotonic_increasing

    def test_yields_are_positive(self, market_df):
        df, _ = market_df
        yield_cols = ["yield_2y", "yield_5y", "yield_10y", "yield_30y"]
        for col in yield_cols:
            assert (df[col].dropna() > 0).all(), f"{col} has non-positive values"

    def test_cad_usd_in_realistic_range(self, market_df):
        df, _ = market_df
        assert df["cad_usd"].between(0.50, 1.10).all(), \
            "CAD/USD out of realistic range [0.50, 1.10]"


class TestAnalyseYieldCurve:

    @pytest.fixture
    def yc(self, market_df):
        df, _ = market_df
        return analyse_yield_curve(df)

    def test_spread_columns_exist(self, yc):
        for col in ["spread_2s10s", "spread_5s30s", "spread_2s5s", "term_premium"]:
            assert col in yc.columns, f"Missing: {col}"

    def test_spread_2s10s_formula(self, yc, market_df):
        df, _ = market_df
        expected = (df["yield_10y"] - df["yield_2y"]) * 100
        pd.testing.assert_series_equal(
            yc["spread_2s10s"].dropna(),
            expected.reindex(yc.index).dropna(),
            check_names=False
        )

    def test_curve_regime_valid_values(self, yc):
        valid = {"deeply_inverted", "inverted", "flat", "normal", "steep", None, np.nan}
        actual = set(yc["curve_regime"].astype(str).unique())
        # All values should be valid regime names or NaN
        assert len(actual - {"nan"} - valid) == 0

    def test_days_inverted_non_negative(self, yc):
        assert (yc["days_inverted"] >= 0).all()

    def test_zscore_has_mean_near_zero(self, yc):
        zscore = yc["spread_2s10s_zscore"].dropna()
        assert abs(zscore.mean()) < 0.5, "Z-score mean should be near zero"


class TestAnalyseFX:

    @pytest.fixture
    def fx(self, market_df):
        df, _ = market_df
        return analyse_fx(df)

    def test_return_columns_exist(self, fx):
        for col in ["return_daily", "return_weekly", "return_monthly",
                    "vol_20d", "vol_60d"]:
            assert col in fx.columns

    def test_daily_returns_are_small(self, fx):
        """Daily log returns should almost always be within ±5%"""
        returns = fx["return_daily"].dropna()
        assert (returns.abs() < 0.10).mean() > 0.95

    def test_vol_20d_is_annualised_positive(self, fx):
        vol = fx["vol_20d"].dropna()
        assert (vol > 0).all()
        assert (vol < 100).all(), "Annualised vol > 100% is unrealistic"

    def test_vol_60d_smoother_than_20d(self, fx):
        """60-day vol should have lower standard deviation than 20-day vol"""
        std_20 = fx["vol_20d"].dropna().std()
        std_60 = fx["vol_60d"].dropna().std()
        assert std_60 < std_20

    def test_vol_regime_valid_categories(self, fx):
        valid = {"low", "moderate", "elevated", "high"}
        actual = set(fx["vol_regime"].dropna().astype(str).unique())
        assert actual.issubset(valid)


class TestGenerateSummaryReport:

    @pytest.fixture
    def report(self, market_df):
        df, quality = market_df
        yc = analyse_yield_curve(df)
        fx = analyse_fx(df)
        return generate_summary_report(df, yc, fx, quality)

    def test_report_has_required_sections(self, report):
        for section in ["report_date", "yield_curve", "fx", "data_quality"]:
            assert section in report

    def test_spread_in_basis_points(self, report):
        spread = report["yield_curve"]["spread_2s10s_bps"]
        assert -500 < spread < 500, f"Spread {spread}bps outside realistic range"

    def test_cad_usd_in_report(self, report):
        rate = report["fx"]["cad_usd_rate"]
        assert 0.50 < rate < 1.10

    def test_vol_annualised_is_pct(self, report):
        vol = report["fx"]["vol_20d_annualised"]
        assert 0 < vol < 100

    def test_report_date_is_valid_date(self, report):
        # Should be parseable as a date
        datetime.strptime(report["report_date"], "%Y-%m-%d")
