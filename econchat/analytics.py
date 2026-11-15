"""Deterministic economic calculations. The LLM never does arithmetic; every
number it reports comes from one of these functions.

Conventions
-----------
* Growth / inflation rates are percentages (3.1 means 3.1 %).
* "yoy"      : % change vs the same period one year earlier.
* "qoq_ann"  : quarter-over-quarter % change, compounded to an annual rate (GDP).
* "mom"      : month-over-month % change (CPI).
* "3m_ann"   : 3-month % change compounded to an annual rate (CPI momentum).
* Core CPI is averaged to quarters (needs >= 2 of 3 months) before comparing
  with quarterly GDP.
* "Real growth proxy" = nominal GDP YoY growth - core CPI YoY inflation. It is an
  approximation (the GDP deflator is the proper deflator), labelled as such.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .data import SERIES, resolve_series

FREQ = {"Q": "QS", "M": "MS"}
COVID_DISTORTED = (pd.Timestamp("2020-04-01"), pd.Timestamp("2021-06-30"))
ERAS = [
    ("1958-1983 high-inflation era", "1958-01-01", "1983-12-31"),
    ("1984-2007 Great Moderation", "1984-01-01", "2007-12-31"),
    ("2008-2019 post-financial-crisis", "2008-01-01", "2019-12-31"),
    ("2020-present pandemic and after", "2020-01-01", "2100-01-01"),
]
VALID_TRANSFORMS = {
    "GDP": ["level", "yoy", "qoq_ann", "qoq"],
    "CPILFESL": ["level", "yoy", "mom", "3m_ann", "6m_ann"],
}


# --------------------------------------------------------------------------- helpers
def r(x, nd: int = 2):
    """Round to a JSON-friendly float (None for NaN)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), nd)


def label(ts: pd.Timestamp, freq: str) -> str:
    ts = pd.Timestamp(ts)
    if freq == "Q":
        return f"{ts.year} Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime("%Y-%m")


def parse_date(value: str | None, *, end: bool = False) -> pd.Timestamp | None:
    """Accept 'YYYY', 'YYYY-MM', 'YYYY-MM-DD', 'YYYY Qn' / 'YYYYQn'."""
    if value in (None, ""):
        return None
    v = str(value).strip().upper().replace(" ", "")
    if "Q" in v and len(v) in (6, 7):
        year, q = v.replace("-", "").split("Q")
        start = pd.Timestamp(int(year), 3 * (int(q) - 1) + 1, 1)
        return start + pd.offsets.QuarterEnd(0) if end else start
    if len(v) == 4 and v.isdigit():
        return pd.Timestamp(int(v), 12, 31) if end else pd.Timestamp(int(v), 1, 1)
    ts = pd.Timestamp(v)
    if end and len(v) == 7:  # YYYY-MM -> end of month
        ts = ts + pd.offsets.MonthEnd(0)
    return ts


def regular(s: pd.Series, freq: str) -> pd.Series:
    """Put on a regular calendar so shift(n) means n periods even across gaps."""
    return s.asfreq(FREQ[freq])


def transform(s: pd.Series, kind: str, freq: str) -> pd.Series:
    ppy = 4 if freq == "Q" else 12
    s = regular(s, freq)
    if kind == "level":
        return s
    if kind == "yoy":
        return (s / s.shift(ppy) - 1) * 100
    if kind in ("qoq", "mom"):
        return (s / s.shift(1) - 1) * 100
    if kind == "qoq_ann":
        return ((s / s.shift(1)) ** ppy - 1) * 100
    if kind == "3m_ann":
        return ((s / s.shift(3)) ** 4 - 1) * 100
    if kind == "6m_ann":
        return ((s / s.shift(6)) ** 2 - 1) * 100
    raise ValueError(f"Unknown transform '{kind}'")


def first_on_or_after(s: pd.Series, ts: pd.Timestamp):
    s = s.dropna()
    s = s[s.index >= ts]
    return (s.index[0], s.iloc[0]) if len(s) else (None, None)


def last_on_or_before(s: pd.Series, ts: pd.Timestamp):
    s = s.dropna()
    s = s[s.index <= ts]
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, None)


def cagr(v0: float, v1: float, years: float) -> float | None:
    if years <= 0 or v0 <= 0:
        return None
    return ((v1 / v0) ** (1 / years) - 1) * 100


def _corr(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 8:
        return {"n": n, "pearson_r": None, "pearson_p": None, "spearman_rho": None}
    pr, pp = stats.pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    sr, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return {"n": n, "pearson_r": r(pr, 3), "pearson_p": float(f"{pp:.3g}"), "spearman_rho": r(sr, 3)}


# --------------------------------------------------------------------------- main object
class Econ:
    """Holds the two raw series plus derived frames; all tool logic lives here."""

    def __init__(self, data: dict[str, pd.Series], info: dict[str, dict] | None = None):
        self.raw = dict(data)  # NaNs kept so gaps stay visible
        self.info = info or {}

    # ------------------------------------------------------------- basics
    def series(self, sid: str, kind: str = "level") -> pd.Series:
        sid = resolve_series(sid)
        if kind not in VALID_TRANSFORMS[sid]:
            raise ValueError(f"Transform '{kind}' not valid for {sid}; use one of {VALID_TRANSFORMS[sid]}")
        return transform(self.raw[sid], kind, SERIES[sid].frequency)

    def latest_date(self, sid: str) -> pd.Timestamp:
        return self.raw[resolve_series(sid)].dropna().index[-1]

    @cached_property
    def quarterly(self) -> pd.DataFrame:
        """Quarterly frame aligning GDP with quarter-averaged core CPI."""
        gdp = regular(self.raw["GDP"], "Q")
        cpi_m = regular(self.raw["CPILFESL"], "M")
        grp = cpi_m.groupby(cpi_m.index.to_period("Q"))
        cpi_q = grp.mean().where(grp.count() >= 2)
        cpi_q.index = cpi_q.index.to_timestamp(how="start")
        df = pd.DataFrame({"gdp": gdp, "cpi": cpi_q}).asfreq("QS")
        df["gdp_yoy"] = (df["gdp"] / df["gdp"].shift(4) - 1) * 100
        df["cpi_yoy"] = (df["cpi"] / df["cpi"].shift(4) - 1) * 100
        df["gdp_qoq_ann"] = ((df["gdp"] / df["gdp"].shift(1)) ** 4 - 1) * 100
        df["cpi_qoq_ann"] = ((df["cpi"] / df["cpi"].shift(1)) ** 4 - 1) * 100
        df["real_proxy_yoy"] = df["gdp_yoy"] - df["cpi_yoy"]
        return df

    # ------------------------------------------------------------- tools
    def get_latest(self, sid: str) -> dict:
        sid = resolve_series(sid)
        meta = SERIES[sid]
        s = self.raw[sid].dropna()
        d, v = s.index[-1], s.iloc[-1]
        out = {
            "series": sid,
            "name": meta.name,
            "units": meta.units,
            "latest_period": label(d, meta.frequency),
            "latest_value": r(v, 3),
            "previous_period": label(s.index[-2], meta.frequency),
            "previous_value": r(s.iloc[-2], 3),
            "data_retrieved": self.info.get(sid, {}).get("fetched_at"),
            "notes": meta.notes,
        }
        if sid == "GDP":
            out["latest_value_trillions"] = r(v / 1000, 3)
            out["qoq_annualized_growth_pct"] = r(self.series(sid, "qoq_ann").loc[d])
            out["qoq_growth_pct_not_annualized"] = r(self.series(sid, "qoq").loc[d])
            out["yoy_growth_pct"] = r(self.series(sid, "yoy").loc[d])
        else:
            out.update(self.core_inflation())
        return out

    def core_inflation(self, as_of: str | None = None) -> dict:
        s = self.raw["CPILFESL"].dropna()
        d = s.index[-1] if as_of is None else last_on_or_before(s, parse_date(as_of, end=True))[0]
        if d is None:
            raise ValueError("No Core CPI observation on or before that date (series starts 1957-01).")
        yoy = self.series("CPILFESL", "yoy")
        out = {
            "as_of_month": label(d, "M"),
            "core_inflation_yoy_pct": r(yoy.loc[d]),
            "core_inflation_3m_annualized_pct": r(self.series("CPILFESL", "3m_ann").loc[d]),
            "core_inflation_6m_annualized_pct": r(self.series("CPILFESL", "6m_ann").loc[d]),
            "core_mom_pct": r(self.series("CPILFESL", "mom").loc[d], 3),
            "core_inflation_yoy_one_year_earlier_pct": r(yoy.get(d - pd.DateOffset(years=1))),
            "definition": "Headline figure = year-over-year % change in the CPILFESL index for the latest month.",
        }
        if pd.isna(yoy.loc[d]):
            vd, vv = last_on_or_before(yoy, d)
            est = self.cpi_yoy_with_interpolated_gaps().get(d)
            out["core_inflation_yoy_pct"] = None
            base = label(d - pd.DateOffset(years=1), "M")
            out["yoy_not_computable_reason"] = (
                f"The {label(d, 'M')} index IS published, but the comparison month {base} was never published "
                f"(BLS shutdown gap), so the official {label(d, 'M')} YoY rate cannot be computed."
            )
            out["latest_computable_yoy"] = {"month": label(vd, "M"), "pct": r(vv)}
            out["estimated_yoy_pct_interpolated_base"] = r(est)
            out["estimate_method"] = (
                "Missing base month filled by log-linear interpolation between its neighbours; "
                "an estimate, not an official BLS figure."
            )
        gaps = [label(i, "M") for i, v in self.raw["CPILFESL"].loc[:d].tail(13).items() if pd.isna(v)]
        if gaps:
            out["missing_months_in_last_year"] = gaps
            out["gap_note"] = (
                "Rates whose comparison month is missing are not computed (e.g. October 2025, "
                "not published by BLS due to the government shutdown)."
            )
        return out

    def cpi_yoy_with_interpolated_gaps(self) -> pd.Series:
        """YoY core inflation where isolated missing months are log-interpolated.
        Used only for clearly labelled estimates and for continuous chart lines."""
        s = regular(self.raw["CPILFESL"], "M")
        filled = np.exp(np.log(s).interpolate(limit=1, limit_area="inside"))
        return (filled / filled.shift(12) - 1) * 100

    def compute_change(self, sid: str, start: str | None = None, end: str | None = None,
                       years_back: float | None = None) -> dict:
        sid = resolve_series(sid)
        meta = SERIES[sid]
        s = self.raw[sid]
        if years_back is not None:
            d1, v1 = last_on_or_before(s, parse_date(end, end=True) or s.dropna().index[-1])
            d0, v0 = last_on_or_before(s, d1 - pd.DateOffset(months=round(12 * float(years_back))))
        else:
            d0, v0 = first_on_or_after(s, parse_date(start) or s.dropna().index[0])
            d1, v1 = last_on_or_before(s, parse_date(end, end=True) or s.dropna().index[-1])
        if d0 is None or d1 is None or d0 >= d1:
            first = label(s.dropna().index[0], meta.frequency)
            last = label(s.dropna().index[-1], meta.frequency)
            raise ValueError(f"No data for that window. {sid} covers {first} to {last}.")
        years = (d1 - d0).days / 365.25
        window = s.loc[d0:d1].dropna()
        kind = "yoy"
        growth = self.series(sid, kind).loc[d0:d1].dropna()
        out = {
            "series": sid,
            "units": meta.units,
            "start_period": label(d0, meta.frequency),
            "start_value": r(v0, 3),
            "end_period": label(d1, meta.frequency),
            "end_value": r(v1, 3),
            "years": r(years, 2),
            "absolute_change": r(v1 - v0, 3),
            "percent_change": r((v1 / v0 - 1) * 100),
            "annualized_rate_pct": r(cagr(v0, v1, years)),
            "window_min": {"period": label(window.idxmin(), meta.frequency), "value": r(window.min(), 3)},
            "window_max": {"period": label(window.idxmax(), meta.frequency), "value": r(window.max(), 3)},
            "yoy_rate_in_window": {
                "average_pct": r(growth.mean()),
                "highest": {"period": label(growth.idxmax(), meta.frequency), "pct": r(growth.max())} if len(growth) else None,
                "lowest": {"period": label(growth.idxmin(), meta.frequency), "pct": r(growth.min())} if len(growth) else None,
            },
        }
        yoy_full = self.series(sid, "yoy")
        ys_d, ys = first_on_or_after(yoy_full, d0)
        ye_d, ye = last_on_or_before(yoy_full, d1)
        if ys_d is not None and ye_d is not None:
            out["yoy_rate_at_start"] = {"period": label(ys_d, meta.frequency), "pct": r(ys)}
            out["yoy_rate_at_end"] = {"period": label(ye_d, meta.frequency), "pct": r(ye)}
        if sid == "GDP":
            q = self.series("GDP", "qoq").loc[d0:d1].dropna()
            neg = q[q < 0]
            out["quarters_with_nominal_decline"] = [
                {"period": label(i, "Q"), "qoq_pct": r(v)} for i, v in neg.items()
            ][:12]
            out["interpretation_hint"] = (
                "annualized_rate_pct is the compound annual growth rate (CAGR) of nominal GDP."
            )
        else:
            out["interpretation_hint"] = (
                "percent_change is cumulative core price increase; annualized_rate_pct is the "
                "average annual core inflation over the window."
            )
        return out

    def annual_table(self, sid: str, start: str | None = None, end: str | None = None,
                     years_back: int | None = None) -> dict:
        """Calendar-year table: year-end level and the year's YoY rate (Q4/Q4 or Dec/Dec),
        with the latest partial year using the latest available period."""
        sid = resolve_series(sid)
        meta = SERIES[sid]
        level = self.raw[sid]
        yoy = self.series(sid, "yoy")
        last = level.dropna().index[-1]
        y_end = parse_date(end, end=True).year if end else last.year
        y_start = (y_end - int(years_back)) if years_back else (parse_date(start).year if start else y_end - 10)
        rows = []
        for y in range(max(y_start, level.dropna().index[0].year), y_end + 1):
            d, v = last_on_or_before(level, pd.Timestamp(y, 12, 31))
            if d is None or d.year != y:
                continue
            g, gd = yoy.get(d), d
            if pd.isna(g):  # comparison month missing: use nearest earlier computable rate
                gd, g = last_on_or_before(yoy.loc[pd.Timestamp(y, 1, 1):d], d)
            row = {"year": y, "level_period": label(d, meta.frequency), "level": r(v, 3), "yoy_pct": r(g)}
            if gd is not None and gd != d:
                row["yoy_period"] = label(gd, meta.frequency)
            rows.append(row)
        if len(rows) > 70:
            rows = rows[-70:]
        return {"series": sid, "units": meta.units, "rate_definition":
                "yoy_pct = % change from the same period a year earlier (Q4/Q4 for GDP, Dec/Dec for CPI; "
                "latest partial year uses the latest period)", "rows": rows}

    def compare_growth_inflation(self, start: str | None = None, end: str | None = None,
                                 exclude_covid: bool = False) -> dict:
        q = self.quarterly.copy()
        s0, s1 = parse_date(start), parse_date(end, end=True)
        if s0 is not None:
            q = q[q.index >= s0]
        if s1 is not None:
            q = q[q.index <= s1]
        full = q
        if exclude_covid:
            q = q[(q.index < COVID_DISTORTED[0]) | (q.index > COVID_DISTORTED[1])]
        valid = q.dropna(subset=["gdp_yoy", "cpi_yoy"])
        if len(valid) < 8:
            raise ValueError("Too few overlapping quarters (core CPI starts 1957; YoY needs one prior year).")

        lead_lag = []
        for k in range(-8, 9):
            c = valid["gdp_yoy"].corr(q["cpi_yoy"].shift(-k).reindex(valid.index))
            lead_lag.append({"k_quarters": k, "corr": r(c, 3)})
        best = max(lead_lag, key=lambda x: abs(x["corr"] or 0))

        eras = []
        for name, a, b in ERAS:
            e = valid[(valid.index >= pd.Timestamp(a)) & (valid.index <= pd.Timestamp(b))]
            if len(e) >= 8:
                eras.append({
                    "era": name,
                    "quarters": len(e),
                    "avg_nominal_gdp_growth_yoy_pct": r(e["gdp_yoy"].mean()),
                    "avg_core_inflation_yoy_pct": r(e["cpi_yoy"].mean()),
                    "avg_real_growth_proxy_pct": r(e["real_proxy_yoy"].mean()),
                    "corr_gdp_growth_vs_core_inflation": r(e["gdp_yoy"].corr(e["cpi_yoy"]), 3),
                })

        roll = full["gdp_yoy"].rolling(40, min_periods=30).corr(full["cpi_yoy"]).dropna()
        last_row = valid.iloc[-1]
        return {
            "window": f"{label(valid.index[0], 'Q')} to {label(valid.index[-1], 'Q')}",
            "frequency": "quarterly (core CPI averaged within quarter)",
            "covid_quarters_excluded": exclude_covid,
            "nominal_gdp_growth_vs_core_inflation": _corr(valid["gdp_yoy"], valid["cpi_yoy"]),
            "real_growth_proxy_vs_core_inflation": _corr(valid["real_proxy_yoy"], valid["cpi_yoy"]),
            "averages_pct": {
                "nominal_gdp_growth_yoy": r(valid["gdp_yoy"].mean()),
                "core_inflation_yoy": r(valid["cpi_yoy"].mean()),
                "real_growth_proxy": r(valid["real_proxy_yoy"].mean()),
            },
            "latest_quarter": {
                "period": label(valid.index[-1], "Q"),
                "core_cpi_months_averaged": f"{valid.index[-1]:%Y-%m} to {(valid.index[-1] + pd.DateOffset(months=2)):%Y-%m}",
                "nominal_gdp_growth_yoy_pct": r(last_row["gdp_yoy"]),
                "core_inflation_yoy_pct": r(last_row["cpi_yoy"]),
                "real_growth_proxy_pct": r(last_row["real_proxy_yoy"]),
            },
            "lead_lag": {
                "definition": "corr(GDP growth at t, core inflation at t+k); k>0 means GDP growth leads inflation",
                "strongest": best,
                "selected": [x for x in lead_lag if x["k_quarters"] in (-8, -4, -2, 0, 2, 4, 8)],
            },
            "by_era": eras,
            "rolling_40q_corr": {
                "latest": r(roll.iloc[-1], 3) if len(roll) else None,
                "min": {"period": label(roll.idxmin(), "Q"), "corr": r(roll.min(), 3)} if len(roll) else None,
                "max": {"period": label(roll.idxmax(), "Q"), "corr": r(roll.max(), 3)} if len(roll) else None,
            },
            "caveats": [
                "GDP here is nominal, so part of the positive co-movement is mechanical: nominal GDP "
                "growth ~ real growth + inflation. The real-growth proxy removes that.",
                "Correlation is not causation; both series respond to monetary policy, supply shocks "
                "and demand shocks.",
                "2020 Q2-2021 Q2 YoY rates are dominated by the pandemic collapse and rebound; "
                "compare results with exclude_covid=true.",
            ],
        }

    def trend_summary(self) -> dict:
        g = self.raw["GDP"].dropna()
        last = g.index[-1]
        gdp_cagr = {}
        for y in (1, 5, 10, 20):
            d0, v0 = last_on_or_before(g, last - pd.DateOffset(years=y))
            gdp_cagr[f"{y}y"] = r(cagr(v0, g.iloc[-1], (last - d0).days / 365.25))
        gdp_cagr["since_1947"] = r(cagr(g.iloc[0], g.iloc[-1], (last - g.index[0]).days / 365.25))

        qd = self.series("GDP", "qoq").dropna()
        worst = qd.nsmallest(3)

        yoy = self.series("CPILFESL", "yoy").dropna()
        decade = yoy.groupby((yoy.index.year // 10) * 10).mean()
        peak_all = (label(yoy.idxmax(), "M"), r(yoy.max()))
        post2020 = yoy[yoy.index >= "2020-01-01"]
        pre_pand = yoy[(yoy.index >= "2010-01-01") & (yoy.index < "2020-01-01")]
        comp = self.compare_growth_inflation()
        comp_ex = self.compare_growth_inflation(exclude_covid=True)
        return {
            "gdp": {
                "latest": self.get_latest("GDP"),
                "nominal_cagr_pct": gdp_cagr,
                "largest_quarterly_declines": [{"period": label(i, "Q"), "qoq_pct": r(v)} for i, v in worst.items()],
            },
            "core_inflation": {
                "latest": self.core_inflation(),
                "average_yoy_by_decade_pct": {f"{int(k)}s": r(v) for k, v in decade.items()},
                "all_time_peak_yoy": {"month": peak_all[0], "pct": peak_all[1]},
                "post_2020_peak_yoy": {"month": label(post2020.idxmax(), "M"), "pct": r(post2020.max())},
                "2010s_average_yoy_pct": r(pre_pand.mean()),
            },
            "relationship": {
                "full_sample_corr": comp["nominal_gdp_growth_vs_core_inflation"],
                "ex_covid_corr": comp_ex["nominal_gdp_growth_vs_core_inflation"],
                "real_proxy_vs_inflation_corr": comp["real_growth_proxy_vs_core_inflation"],
                "by_era": comp["by_era"],
                "latest_quarter": comp["latest_quarter"],
            },
        }
