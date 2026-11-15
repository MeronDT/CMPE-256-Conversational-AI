"""Regenerate the report figures in figures/report_*.png (no LLM needed).

    python -m scripts.make_figures
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from econchat import charts
from econchat.analytics import Econ
from econchat.data import load_all

OUT = Path(__file__).resolve().parent.parent / "figures"


def lead_lag(econ: Econ) -> go.Figure:
    q = econ.quarterly.dropna(subset=["gdp_yoy", "cpi_yoy"])
    ks = list(range(-8, 9))
    vals = [q["gdp_yoy"].corr(econ.quarterly["cpi_yoy"].shift(-k).reindex(q.index)) for k in ks]
    fig = go.Figure(go.Bar(x=ks, y=vals, marker={"color": charts.COLORS["GDP"], "cornerradius": 4},
                           hovertemplate="k=%{x}: r=%{y:.2f}<extra></extra>"))
    charts._layout(fig, "Lead-lag correlation: GDP growth at t vs core inflation at t+k",
                   "Pearson correlation", "k > 0: GDP growth leads core inflation by k quarters (full sample, quarterly YoY)")
    fig.update_xaxes(title="k (quarters)", dtick=1)
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.3)
    fig.layout.annotations[0].text = "Source: BEA, BLS via FRED."
    return fig


def main():
    data, info = load_all()
    econ = Econ(data, info)
    OUT.mkdir(exist_ok=True)
    specs = {
        "report_01_gdp_level": {"kind": "line", "series": ["GDP"]},
        "report_02_gdp_annual_growth_10y": {"kind": "annual_bars", "series": ["GDP"], "years_back": 10},
        "report_03_core_cpi_yoy_5y": {"kind": "line", "series": ["CPILFESL"], "transform": "yoy", "start": "2021-10"},
        "report_04_core_cpi_yoy_full": {"kind": "line", "series": ["CPILFESL"], "transform": "yoy", "start": "1958"},
        "report_05_gdp_vs_core_cpi": {"kind": "comparison"},
        "report_06_scatter_ex_covid": {"kind": "scatter", "exclude_covid": True},
        "report_07_rolling_corr": {"kind": "rolling_corr"},
    }
    for name, spec in specs.items():
        fig, _ = charts.build(econ, spec)
        fig.write_image(OUT / f"{name}.png", width=1100, height=500, scale=2)
        print("wrote", name)
    lead_lag(econ).write_image(OUT / "report_08_lead_lag.png", width=1100, height=500, scale=2)
    print("wrote report_08_lead_lag")


if __name__ == "__main__":
    main()
