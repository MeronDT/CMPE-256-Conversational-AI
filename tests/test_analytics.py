import math

import numpy as np
import pandas as pd
import pytest

from econchat import charts
from econchat.analytics import Econ, cagr, parse_date, transform
from econchat.data import load_all, parse_fred_csv, resolve_series
from econchat.tools import TOOLS, ToolRunner


def synthetic(gap_month: str | None = "2023-10-01") -> Econ:
    """GDP grows exactly 1 % per quarter; core CPI exactly 0.25 % per month."""
    q = pd.date_range("2010-01-01", "2024-07-01", freq="QS")
    gdp = pd.Series(1000 * 1.01 ** np.arange(len(q)), index=q, name="GDP")
    m = pd.date_range("2010-01-01", "2024-10-01", freq="MS")
    cpi = pd.Series(200 * 1.0025 ** np.arange(len(m)), index=m, name="CPILFESL")
    if gap_month:
        cpi.loc[gap_month] = np.nan
    return Econ({"GDP": gdp, "CPILFESL": cpi})


def test_parse_fred_csv_handles_blank_and_dot():
    text = "observation_date,CPILFESL\n2025-09-01,330.4\n2025-10-01,\n2025-11-01,.\n2025-12-01,331.8\n"
    s = parse_fred_csv(text, "CPILFESL")
    assert len(s) == 4 and s.isna().sum() == 2
    assert s.iloc[-1] == pytest.approx(331.8)


def test_resolve_series_aliases():
    assert resolve_series("core cpi") == "CPILFESL"
    assert resolve_series("gdp") == "GDP"
    with pytest.raises(ValueError):
        resolve_series("unemployment")


@pytest.mark.parametrize("raw,end,expected", [
    ("2015", False, "2015-01-01"), ("2015", True, "2015-12-31"),
    ("2015-06", True, "2015-06-30"), ("2015 Q2", False, "2015-04-01"),
    ("2015Q2", True, "2015-06-30"), ("2015-06-15", False, "2015-06-15"),
])
def test_parse_date(raw, end, expected):
    assert parse_date(raw, end=end) == pd.Timestamp(expected)


def test_growth_transforms_exact():
    e = synthetic(gap_month=None)
    assert e.series("GDP", "qoq").dropna().iloc[-1] == pytest.approx(1.0)
    assert e.series("GDP", "qoq_ann").dropna().iloc[-1] == pytest.approx((1.01 ** 4 - 1) * 100)
    assert e.series("GDP", "yoy").dropna().iloc[-1] == pytest.approx((1.01 ** 4 - 1) * 100)
    assert e.series("CPILFESL", "yoy").dropna().iloc[-1] == pytest.approx((1.0025 ** 12 - 1) * 100)
    assert e.series("CPILFESL", "3m_ann").dropna().iloc[-1] == pytest.approx((1.0025 ** 12 - 1) * 100)


def test_missing_month_is_not_bridged():
    e = synthetic()
    yoy = e.series("CPILFESL", "yoy")
    assert math.isnan(yoy.loc["2023-10-01"])          # month itself missing
    assert math.isnan(yoy.loc["2024-10-01"])          # base month missing
    assert not math.isnan(yoy.loc["2024-09-01"])
    # estimated version interpolates the single gap and recovers the true rate
    est = e.cpi_yoy_with_interpolated_gaps().loc["2024-10-01"]
    assert est == pytest.approx((1.0025 ** 12 - 1) * 100, rel=1e-9)


def test_core_inflation_fallback_when_base_missing():
    out = synthetic().core_inflation()
    assert out["as_of_month"] == "2024-10"
    assert out["core_inflation_yoy_pct"] is None
    assert out["latest_computable_yoy"]["month"] == "2024-09"
    assert out["estimated_yoy_pct_interpolated_base"] == pytest.approx(3.04, abs=0.01)


def test_compute_change_years_back_and_cagr():
    out = synthetic(None).compute_change("GDP", years_back=10)
    assert out["start_period"] == "2014 Q3" and out["end_period"] == "2024 Q3"
    assert out["annualized_rate_pct"] == pytest.approx((1.01 ** 4 - 1) * 100, abs=0.02)
    assert cagr(100, 200, 10) == pytest.approx((2 ** 0.1 - 1) * 100)


def test_quarterly_cpi_needs_two_months():
    e = synthetic()
    q = e.quarterly
    assert not math.isnan(q.loc["2023-10-01", "cpi"])  # Nov + Dec present
    assert math.isnan(q.loc["2024-10-01", "cpi"])       # only October present


def test_out_of_range_window_gives_clear_error():
    res = ToolRunner(synthetic()).run("compute_change", {"series": "GDP", "start": "1930", "end": "1935"})
    assert "error" in res and "covers" in res["error"]


@pytest.fixture(scope="module")
def real():
    data, info = load_all()
    return Econ(data, info)


def test_every_tool_runs_on_real_data(real):
    runner = ToolRunner(real)
    calls = {
        "get_latest": {"series": "GDP"}, "core_inflation": {},
        "compute_change": {"series": "CPILFESL", "years_back": 5},
        "annual_table": {"series": "GDP", "years_back": 10},
        "get_observations": {"series": "CPILFESL", "transform": "yoy", "start": "2020"},
        "compare_growth_inflation": {"exclude_covid": True}, "trend_summary": {},
        "make_chart": {"kind": "comparison"},
    }
    assert set(calls) == {t["function"]["name"] for t in TOOLS}
    for name, args in calls.items():
        assert "error" not in runner.run(name, args), name
    assert len(runner.figures) == 1


@pytest.mark.parametrize("spec", [
    {"kind": "comparison"}, {"kind": "line", "series": ["GDP", "CPILFESL"]},
    {"kind": "line", "series": ["CPILFESL"], "transform": "yoy", "start": "2021"},
    {"kind": "scatter", "exclude_covid": True}, {"kind": "rolling_corr"},
    {"kind": "annual_bars", "series": ["CPILFESL"], "years_back": 5},
])
def test_charts_build(real, spec):
    fig, info = charts.build(real, spec)
    assert len(fig.data) >= 1 and info["kind"] == spec["kind"]


def test_real_gdp_yoy_matches_raw(real):
    g = real.raw["GDP"].dropna()
    expected = (g.iloc[-1] / g.iloc[-5] - 1) * 100
    assert real.get_latest("GDP")["yoy_growth_pct"] == pytest.approx(expected, abs=0.005)
