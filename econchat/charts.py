"""Chart builder: a small JSON spec -> Plotly figure. The LLM chooses the spec;
it never writes plotting code.

Design rules (dataviz skill): one y-axis per chart (two series of different
units are indexed to 100 or shown as rates), fixed colour per entity
(GDP blue, Core CPI orange, real proxy aqua), thin 2px lines, recessive grid,
recession shading, unified hover.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .analytics import COVID_DISTORTED, Econ, label, parse_date

COLORS = {"GDP": "#2a78d6", "CPILFESL": "#eb6834", "REAL": "#1baf7a"}
ERA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT2, GRID, SHADE = "#0b0b0b", "#52514e", "rgba(82,81,78,0.15)", "rgba(82,81,78,0.12)"
NAMES = {"GDP": "Nominal GDP", "CPILFESL": "Core CPI"}

# NBER business-cycle peaks/troughs (monthly) since 1947.
RECESSIONS = [
    ("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"),
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]
TRANSFORM_LABEL = {
    "level": "Level", "yoy": "% change from a year earlier", "qoq_ann": "% change, quarterly at annual rate",
    "qoq": "% change from previous quarter", "mom": "% change from previous month",
    "3m_ann": "3-month % change, annualized", "6m_ann": "6-month % change, annualized",
    "index100": "Index (start of window = 100)",
}
CHART_KINDS = ["comparison", "line", "scatter", "rolling_corr", "annual_bars"]


def _layout(fig: go.Figure, title: str, ytitle: str, subtitle: str = "") -> go.Figure:
    full_title = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{TEXT2}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        title={"text": full_title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": TEXT}},
        template="plotly_white",
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": TEXT2, "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "xanchor": "left"},
        margin={"l": 60, "r": 30, "t": 90 if subtitle else 70, "b": 60},
        height=460,
    )
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=True, zerolinecolor="rgba(82,81,78,0.45)", zerolinewidth=1)
    fig.update_xaxes(gridcolor=GRID, showgrid=False)
    fig.add_annotation(text="Source: BEA, BLS via FRED. Shaded: NBER recessions.", xref="paper", yref="paper",
                       x=0, y=-0.14, showarrow=False, font={"size": 10, "color": TEXT2}, xanchor="left")
    return fig


def _shade(fig: go.Figure, x0: pd.Timestamp, x1: pd.Timestamp, covid: bool = False) -> None:
    for a, b in RECESSIONS:
        a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
        if b >= x0 and a <= x1:
            fig.add_vrect(x0=max(a, x0), x1=min(b, x1), fillcolor=SHADE, line_width=0, layer="below")


def _window(start, end, default_start=None):
    return parse_date(start) or (pd.Timestamp(default_start) if default_start else None), parse_date(end, end=True)


def _clip(s: pd.Series, s0, s1) -> pd.Series:
    if s0 is not None:
        s = s[s.index >= s0]
    if s1 is not None:
        s = s[s.index <= s1]
    return s


def comparison(econ: Econ, start=None, end=None, show_real_proxy=True) -> tuple[go.Figure, dict]:
    s0, s1 = _window(start, end, "1958-01-01")
    q = _clip(econ.quarterly, s0, s1).dropna(subset=["gdp_yoy", "cpi_yoy"], how="all")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.index, y=q["gdp_yoy"], name="Nominal GDP growth",
                             line={"color": COLORS["GDP"], "width": 2}, hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=q.index, y=q["cpi_yoy"], name="Core CPI inflation",
                             line={"color": COLORS["CPILFESL"], "width": 2}, hovertemplate="%{y:.2f}%"))
    if show_real_proxy:
        fig.add_trace(go.Scatter(x=q.index, y=q["real_proxy_yoy"], name="Real growth proxy (GDP - core CPI)",
                                 line={"color": COLORS["REAL"], "width": 1.5, "dash": "dot"}, hovertemplate="%{y:.2f}%"))
    _shade(fig, q.index[0], q.index[-1])
    _layout(fig, "Nominal GDP growth vs Core CPI inflation",
            "% change from a year earlier",
            f"Quarterly, {label(q.index[0], 'Q')} to {label(q.index[-1], 'Q')}; core CPI averaged within each quarter")
    lines = {"Nominal GDP growth (YoY %)": "blue solid", "Core CPI inflation (YoY %)": "orange solid"}
    if show_real_proxy:
        lines["Real growth proxy (GDP growth minus core inflation)"] = "green dotted"
    return fig, {"kind": "comparison", "window": f"{label(q.index[0], 'Q')} to {label(q.index[-1], 'Q')}",
                 "points": int(len(q)), "lines": lines, "shading": "grey bands = NBER recessions"}


def line(econ: Econ, series: list[str], transform: str = "level", start=None, end=None) -> tuple[go.Figure, dict]:
    series = [("GDP" if x.upper() == "GDP" else "CPILFESL") for x in series] or ["GDP"]
    s0, s1 = _window(start, end)
    two = len(set(series)) > 1
    if two and transform == "level":
        transform = "index100"  # never put $bn and an index on one axis
    fig = go.Figure()
    info = {"kind": "line", "transform": transform, "series": series}
    for sid in dict.fromkeys(series):
        freq = "Q" if sid == "GDP" else "M"
        kind = "level" if transform == "index100" else transform
        s = _clip(econ.series(sid, kind), s0, s1)
        if transform in ("yoy",) and sid == "CPILFESL":
            s = _clip(econ.cpi_yoy_with_interpolated_gaps(), s0, s1)  # continuous line across the single missing month
        s = s.dropna() if transform == "index100" else s
        if transform == "index100":
            s = s / s.dropna().iloc[0] * 100
        fmt = "%{y:,.1f}" if kind == "level" else "%{y:.2f}%"
        fig.add_trace(go.Scatter(x=s.index, y=s, name=NAMES[sid], line={"color": COLORS[sid], "width": 2},
                                 hovertemplate=fmt, connectgaps=False))
        s_ok = s.dropna()
        pt = lambda i: {"period": label(i, freq), "value": round(float(s_ok.loc[i]), 2)}
        info[sid] = {"color": "blue" if sid == "GDP" else "orange",
                     "start": pt(s_ok.index[0]), "end": pt(s_ok.index[-1]),
                     "highest": pt(s_ok.idxmax()), "lowest": pt(s_ok.idxmin())}
        if transform == "yoy" and sid == "CPILFESL":
            official = econ.series(sid, "yoy").reindex(s_ok.index)
            est_pts = [label(i, freq) for i in s_ok.index if pd.isna(official.loc[i])]
            if est_pts:
                info[sid]["estimated_points"] = est_pts
                if info[sid]["end"]["period"] in est_pts:
                    info[sid]["end"]["status"] = "ESTIMATE, not official"
                    ok = official.dropna()
                    info[sid]["latest_official"] = {"period": label(ok.index[-1], freq), "value": round(float(ok.iloc[-1]), 2)}
                info[sid]["estimate_note"] = ("These points use an interpolated base month (Oct 2025 not published) "
                                              "and are estimates, not official rates.")
    xs = [t.x for t in fig.data]
    x0 = min(pd.Timestamp(x[0]) for x in xs)
    x1 = max(pd.Timestamp(x[-1]) for x in xs)
    _shade(fig, x0, x1)
    if transform == "level":
        sid = series[0]
        ytitle = "Billions of dollars (SAAR)" if sid == "GDP" else "Index, 1982-84 = 100"
        title = f"{NAMES[sid]}, level"
    else:
        ytitle = TRANSFORM_LABEL[transform]
        names = " and ".join(NAMES[s] for s in dict.fromkeys(series))
        title = f"{names} (indexed)" if transform == "index100" else f"{names}: {TRANSFORM_LABEL[transform].lower()}"
    note = "Core CPI YoY: Oct 2025 base month not published; line bridges it by interpolation." if (
        transform == "yoy" and "CPILFESL" in series) else ""
    _layout(fig, title, ytitle, note)
    fig.update_layout(showlegend=two)
    return fig, info


def scatter(econ: Econ, start=None, end=None, exclude_covid=False) -> tuple[go.Figure, dict]:
    s0, s1 = _window(start, end)
    q = _clip(econ.quarterly, s0, s1).dropna(subset=["gdp_yoy", "cpi_yoy"])
    if exclude_covid:
        q = q[(q.index < COVID_DISTORTED[0]) | (q.index > COVID_DISTORTED[1])]
    eras = [("Before 1984", None, "1983-12-31"), ("1984-2019", "1984-01-01", "2019-12-31"), ("2020 onward", "2020-01-01", None)]
    fig = go.Figure()
    for (name, a, b), color in zip(eras, ERA_COLORS):
        e = q
        if a:
            e = e[e.index >= a]
        if b:
            e = e[e.index <= b]
        if e.empty:
            continue
        fig.add_trace(go.Scatter(
            x=e["cpi_yoy"], y=e["gdp_yoy"], mode="markers", name=name,
            marker={"size": 8, "color": color, "opacity": 0.8, "line": {"width": 1, "color": "#fcfcfb"}},
            text=[label(i, "Q") for i in e.index],
            hovertemplate="%{text}<br>Core CPI %{x:.2f}%<br>GDP %{y:.2f}%<extra></extra>"))
    _layout(fig, "Nominal GDP growth vs Core CPI inflation, each quarter", "Nominal GDP growth, % y/y",
            "Each dot is one quarter" + (" (2020 Q2-2021 Q2 excluded)" if exclude_covid else ""))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Core CPI inflation, % y/y", showgrid=True)
    fig.layout.annotations[0].text = "Source: BEA, BLS via FRED."
    return fig, {"kind": "scatter", "points": int(len(q)),
                 "dot_colors": {"Before 1984": "blue", "1984-2019": "orange", "2020 onward": "green"}, "corr": round(float(q["gdp_yoy"].corr(q["cpi_yoy"])), 3)}


def rolling_corr(econ: Econ, window: int = 40, start=None, end=None) -> tuple[go.Figure, dict]:
    q = econ.quarterly
    rc = q["gdp_yoy"].rolling(window, min_periods=int(window * 0.75)).corr(q["cpi_yoy"]).dropna()
    s0, s1 = _window(start, end)
    rc = _clip(rc, s0, s1)
    fig = go.Figure(go.Scatter(x=rc.index, y=rc, name="Rolling correlation", line={"color": COLORS["GDP"], "width": 2},
                               hovertemplate="%{y:.2f}"))
    _shade(fig, rc.index[0], rc.index[-1])
    _layout(fig, f"Rolling {window}-quarter correlation: GDP growth vs core inflation",
            "Pearson correlation", "Both measured as % change from a year earlier")
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(showlegend=False)
    return fig, {"kind": "rolling_corr", "window_quarters": window, "latest": round(float(rc.iloc[-1]), 3)}


def annual_bars(econ: Econ, series: str = "GDP", years_back: int = 10) -> tuple[go.Figure, dict]:
    tbl = econ.annual_table(series, years_back=years_back)
    sid = tbl["series"]
    rows = tbl["rows"]
    last_p = "Q4" if sid == "GDP" else "-12"
    xlab = [str(x["year"]) + ("" if x["level_period"].endswith(last_p) else "*") for x in rows]
    fig = go.Figure(go.Bar(x=xlab, y=[x["yoy_pct"] for x in rows],
                           marker={"color": COLORS[sid], "cornerradius": 4}, name=NAMES[sid],
                           text=[f"{x['yoy_pct']:.1f}%" for x in rows], textposition="outside",
                           textfont={"color": TEXT2}, hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    what = "Q4 over Q4" if sid == "GDP" else "December over December"
    _layout(fig, f"{NAMES[sid]}: annual {'growth' if sid == 'GDP' else 'inflation'}", "% change",
            f"{what}; * partial year, uses latest available period")
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.35)
    fig.layout.annotations[0].text = "Source: BEA, BLS via FRED."
    return fig, {"kind": "annual_bars", "series": sid, "rows": rows}


def build(econ: Econ, spec: dict) -> tuple[go.Figure, dict]:
    kind = spec.get("kind", "comparison")
    if kind == "comparison":
        return comparison(econ, spec.get("start"), spec.get("end"), spec.get("show_real_proxy", True))
    if kind == "line":
        return line(econ, spec.get("series") or ["GDP"], spec.get("transform", "level"), spec.get("start"), spec.get("end"))
    if kind == "scatter":
        return scatter(econ, spec.get("start"), spec.get("end"), spec.get("exclude_covid", False))
    if kind == "rolling_corr":
        return rolling_corr(econ, int(spec.get("window", 40)), spec.get("start"), spec.get("end"))
    if kind == "annual_bars":
        return annual_bars(econ, (spec.get("series") or ["GDP"])[0], int(spec.get("years_back", 10)))
    raise ValueError(f"Unknown chart kind '{kind}'. Use one of {CHART_KINDS}.")
