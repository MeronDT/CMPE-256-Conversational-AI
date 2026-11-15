"""Tool schemas (OpenAI function-calling format) and the dispatcher that runs
them against the analytics layer."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import charts
from .analytics import Econ, VALID_TRANSFORMS, label, parse_date, r
from .data import SERIES, resolve_series

SERIES_ENUM = ["GDP", "CPILFESL"]
_DATE = {"type": "string", "description": "Date like '2015', '2015-06', '2015-06-01' or '2015 Q2'."}


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [],
                       "additionalProperties": False}}}


TOOLS: list[dict] = [
    _fn("get_latest",
        "Latest reported observation of a series with its period, previous value and headline growth "
        "(GDP: QoQ annualized and YoY growth; CPILFESL: core inflation YoY, 3m/6m annualized, MoM).",
        {"series": {"type": "string", "enum": SERIES_ENUM}}, ["series"]),
    _fn("core_inflation",
        "Core CPI inflation rates (YoY, 3-month and 6-month annualized, MoM) as of the latest month or a given month. "
        "Use for 'current core inflation' questions.",
        {"as_of": {**_DATE, "description": "Optional month; omit for latest."}}),
    _fn("compute_change",
        "Change in a series between two dates: start/end values, absolute and percent change, annualized rate "
        "(CAGR / average annual inflation), min/max in window, YoY rate average/high/low. Use years_back=N for "
        "'past N years' (measured back from the latest observation).",
        {"series": {"type": "string", "enum": SERIES_ENUM}, "start": _DATE, "end": _DATE,
         "years_back": {"type": "number", "description": "Look back N years from the latest observation (or from end)."}},
        ["series"]),
    _fn("annual_table",
        "Calendar-year table of year-end level and annual YoY rate (Q4/Q4 GDP growth, Dec/Dec core inflation). "
        "Good for 'how has X changed year by year'.",
        {"series": {"type": "string", "enum": SERIES_ENUM}, "years_back": {"type": "integer"},
         "start": _DATE, "end": _DATE}, ["series"]),
    _fn("get_observations",
        "Raw observations of a series (optionally transformed) in a date window; at most ~60 rows, "
        "downsampled for long windows. Use for specific dates or values not covered by other tools.",
        {"series": {"type": "string", "enum": SERIES_ENUM},
         "transform": {"type": "string", "enum": ["level", "yoy", "qoq_ann", "qoq", "mom", "3m_ann", "6m_ann"]},
         "start": _DATE, "end": _DATE}, ["series"]),
    _fn("compare_growth_inflation",
        "Relationship between nominal GDP growth and core CPI inflation (quarterly YoY): correlations, "
        "real-growth proxy, lead-lag correlations, averages and correlations by era, rolling correlation, caveats.",
        {"start": _DATE, "end": _DATE,
         "exclude_covid": {"type": "boolean", "description": "Drop 2020 Q2-2021 Q2 distorted quarters."}}),
    _fn("trend_summary",
        "Fact sheet for summarizing key trends in both datasets: latest values, long-run GDP growth rates, "
        "biggest GDP declines, core inflation by decade and peaks, relationship by era.", {}),
    _fn("make_chart",
        "Create a chart that is shown to the user under your answer. Kinds: "
        "'comparison' = GDP growth vs core inflation (YoY %) over time [best for comparing the two]; "
        "'line' = one or both series with a transform (two series at level are indexed to 100); "
        "'scatter' = GDP growth vs core inflation by quarter, colored by era; "
        "'rolling_corr' = rolling correlation over time; "
        "'annual_bars' = annual growth/inflation bars for one series.",
        {"kind": {"type": "string", "enum": charts.CHART_KINDS},
         "series": {"type": "array", "items": {"type": "string", "enum": SERIES_ENUM}},
         "transform": {"type": "string", "enum": ["level", "yoy", "qoq_ann", "qoq", "mom", "3m_ann", "6m_ann"]},
         "start": _DATE, "end": _DATE,
         "years_back": {"type": "integer", "description": "annual_bars only"},
         "exclude_covid": {"type": "boolean"}, "show_real_proxy": {"type": "boolean"},
         "window": {"type": "integer", "description": "rolling_corr window in quarters (default 40)"}},
        ["kind"]),
]


class ToolRunner:
    def __init__(self, econ: Econ):
        self.econ = econ
        self.figures: list[tuple[Any, dict]] = []

    def get_observations(self, series, transform="level", start=None, end=None):
        sid = resolve_series(series)
        meta = SERIES[sid]
        if transform not in VALID_TRANSFORMS[sid]:
            raise ValueError(f"Transform '{transform}' not valid for {sid}; use one of {VALID_TRANSFORMS[sid]}")
        s = self.econ.series(sid, transform)
        s0, s1 = parse_date(start), parse_date(end, end=True)
        if s0 is not None:
            s = s[s.index >= s0]
        if s1 is not None:
            s = s[s.index <= s1]
        s = s.dropna()
        if s.empty:
            full = self.econ.raw[sid].dropna()
            raise ValueError(f"No observations in that window. {sid} covers "
                             f"{label(full.index[0], meta.frequency)} to {label(full.index[-1], meta.frequency)}.")
        note = None
        if len(s) > 60:
            step = int(len(s) / 60) + 1
            s = pd.concat([s.iloc[::step], s.iloc[[-1]]])
            s = s[~s.index.duplicated()]
            note = f"Downsampled to every {step}th observation; narrow the window for full detail."
        out = {"series": sid, "transform": transform,
               "units": meta.units if transform == "level" else "% change",
               "observations": [{"period": label(i, meta.frequency), "value": r(v, 3)} for i, v in s.items()]}
        if note:
            out["note"] = note
        return out

    def make_chart(self, **spec):
        fig, info = charts.build(self.econ, spec)
        self.figures.append((fig, spec))
        return {"status": "chart displayed to the user below your answer", "chart": info}

    def run(self, name: str, args: dict) -> dict:
        e = self.econ
        table = {
            "get_latest": lambda a: e.get_latest(a["series"]),
            "core_inflation": lambda a: e.core_inflation(a.get("as_of")),
            "compute_change": lambda a: e.compute_change(a["series"], a.get("start"), a.get("end"), a.get("years_back")),
            "annual_table": lambda a: e.annual_table(a["series"], a.get("start"), a.get("end"), a.get("years_back")),
            "get_observations": lambda a: self.get_observations(**a),
            "compare_growth_inflation": lambda a: e.compare_growth_inflation(a.get("start"), a.get("end"), bool(a.get("exclude_covid", False))),
            "trend_summary": lambda a: e.trend_summary(),
            "make_chart": lambda a: self.make_chart(**a),
        }
        if name not in table:
            return {"error": f"Unknown tool {name}"}
        try:
            args = {k: v for k, v in (args or {}).items() if v is not None}
            return table[name](args)
        except Exception as exc:  # returned to the model so it can recover / explain
            return {"error": str(exc)}


def to_json(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
