"""Data layer: download FRED series without an API key, cache them locally,
and fall back to a bundled snapshot when offline.

FRED's public graph endpoint returns CSV with columns ``observation_date`` and
the series id. Missing observations come back as empty cells (or ``"."`` in
the API), e.g. October 2025 Core CPI, which BLS never published because of the
government shutdown. They are kept as NaN, never interpolated.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
CACHE_TTL = timedelta(hours=float(os.getenv("FRED_CACHE_HOURS", "12")))


@dataclass(frozen=True)
class SeriesMeta:
    id: str
    name: str
    units: str
    frequency: str  # "Q" or "M"
    periods_per_year: int
    seasonal_adjustment: str
    source: str
    url: str
    notes: str


SERIES: dict[str, SeriesMeta] = {
    "GDP": SeriesMeta(
        id="GDP",
        name="Gross Domestic Product",
        units="Billions of dollars, seasonally adjusted annual rate (nominal)",
        frequency="Q",
        periods_per_year=4,
        seasonal_adjustment="Seasonally adjusted annual rate",
        source="U.S. Bureau of Economic Analysis via FRED",
        url="https://fred.stlouisfed.org/series/GDP",
        notes=(
            "Nominal GDP: includes price changes, so its growth rate mixes real "
            "output growth and inflation. Dates are the first day of the quarter. "
            "Values are revised after release."
        ),
    ),
    "CPILFESL": SeriesMeta(
        id="CPILFESL",
        name="Consumer Price Index for All Urban Consumers: All Items Less Food and Energy (Core CPI)",
        units="Index 1982-1984=100",
        frequency="M",
        periods_per_year=12,
        seasonal_adjustment="Seasonally adjusted",
        source="U.S. Bureau of Labor Statistics via FRED",
        url="https://fred.stlouisfed.org/series/CPILFESL",
        notes=(
            "An index level, not an inflation rate. Core inflation is the % change "
            "of this index (conventionally year-over-year). October 2025 is missing "
            "because BLS did not publish it during the government shutdown."
        ),
    ),
}

ALIASES = {
    "gdp": "GDP",
    "gross domestic product": "GDP",
    "cpilfesl": "CPILFESL",
    "core cpi": "CPILFESL",
    "cpi": "CPILFESL",
    "core_cpi": "CPILFESL",
    "inflation": "CPILFESL",
}


def resolve_series(name: str) -> str:
    key = name.strip()
    if key.upper() in SERIES:
        return key.upper()
    try:
        return ALIASES[key.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown series '{name}'. Available: GDP, CPILFESL (Core CPI)."
        ) from exc


def parse_fred_csv(text: str, sid: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), na_values=["."], keep_default_na=True)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = sid if sid in df.columns else df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=sid,
    ).sort_index()
    s.index.name = "date"
    if s.dropna().empty:
        raise ValueError(f"No numeric observations parsed for {sid}")
    return s


def _cache_paths(sid: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{sid}.csv", CACHE_DIR / f"{sid}.meta.json"


def _read_local(path: Path, sid: str) -> pd.Series:
    return parse_fred_csv(path.read_text(encoding="utf-8"), sid)


def fetch_series(sid: str, force: bool = False, timeout: float = 20.0) -> tuple[pd.Series, dict]:
    """Return (series, info). info has source ('live'|'cache'|'snapshot') and fetched_at."""
    sid = resolve_series(sid)
    csv_path, meta_path = _cache_paths(sid)
    now = datetime.now(timezone.utc)

    if not force and csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fetched = datetime.fromisoformat(meta["fetched_at"])
        if now - fetched < CACHE_TTL:
            return _read_local(csv_path, sid), {"source": "cache", "fetched_at": meta["fetched_at"]}

    try:
        resp = requests.get(
            FRED_CSV_URL.format(sid=sid),
            timeout=timeout,
            headers={"User-Agent": "fred-econ-chat/1.0"},
        )
        resp.raise_for_status()
        series = parse_fred_csv(resp.text, sid)  # validate before caching
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(resp.text, encoding="utf-8")
        meta_path.write_text(json.dumps({"fetched_at": now.isoformat()}))
        return series, {"source": "live", "fetched_at": now.isoformat()}
    except Exception as exc:  # network down, FRED change, etc.
        if csv_path.exists():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            return _read_local(csv_path, sid), {
                "source": "cache (stale)",
                "fetched_at": meta.get("fetched_at", "unknown"),
                "warning": f"Live fetch failed: {exc}",
            }
        snap = SNAPSHOT_DIR / f"{sid}.csv"
        if snap.exists():
            snap_meta = SNAPSHOT_DIR / "snapshot.json"
            when = json.loads(snap_meta.read_text()).get("fetched_at", "unknown") if snap_meta.exists() else "unknown"
            return _read_local(snap, sid), {
                "source": "snapshot",
                "fetched_at": when,
                "warning": f"Live fetch failed: {exc}",
            }
        raise


def load_all(force: bool = False) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    data, info = {}, {}
    for sid in SERIES:
        data[sid], info[sid] = fetch_series(sid, force=force)
    return data, info
