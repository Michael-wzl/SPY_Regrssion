#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate and feature computation pipeline.

Primary mode: Use official RTH daily data from raw_data/spy_ohlcv_1drth_20141231_20250602.csv.

Optional mode: Re-aggregate RTH from provided processed intraday files with a flag to
control whether to include extended hours (pre/post market) in the aggregation.

CLI usage examples:
  - python3 aggregate.py                   # use official RTH, compute all features
  - python3 aggregate.py --from-intraday   # build RTH daily from processed intraday
  - python3 aggregate.py --from-intraday --include-extended false  # exclude pre/post
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

import basic_feats
import gaps as gaps_mod
import liquidity as liq_mod
import trend as trend_mod
import volatility as vol_mod


ROOT = os.path.dirname(__file__)
OFFICIAL_RTH = os.path.join(ROOT, "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")

# Optional: intraday sources if re-aggregating
HOURLY_CSV = os.path.join(ROOT, "processed_data", "spy_ohlcv_1h_20141231_20250602.csv")
MINUTE_PRIMARY = os.path.join(ROOT, "processed_data", "spy_ohlcv_1m910_20241231_20250602.csv")

OUT_RTH_1D = os.path.join(ROOT, "processed_data", "spy_ohlcv_rth_1d_20141231-20250602.csv")


def parse_as_eastern_walltime(s: pd.Series, tz: str = "America/New_York") -> pd.DatetimeIndex:
    dt = pd.to_datetime(s)
    try:
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    except AttributeError:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt.dt.tz_localize(tz)


def agg_ohlcv(df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    o = df.iloc[0]["open"]
    h = df["high"].max()
    l = df["low"].min()
    c = df.iloc[-1]["close"]
    v = df["volume"].sum()
    return float(o), float(h), float(l), float(c), float(v)


def build_rth_daily_from_intraday(include_extended: bool = True) -> pd.DataFrame:
    tz = "America/New_York"
    if not os.path.exists(HOURLY_CSV) or not os.path.exists(MINUTE_PRIMARY):
        raise FileNotFoundError("Missing processed intraday sources for re-aggregation.")
    h1 = pd.read_csv(HOURLY_CSV)
    m1 = pd.read_csv(MINUTE_PRIMARY)
    h1["dt_et"] = parse_as_eastern_walltime(h1["time"], tz)
    m1["dt_et"] = parse_as_eastern_walltime(m1["time"], tz)

    # Base RTH window 09:30-16:00 constructed by 09:30-10:00 minutes + 10-16 hours.
    m_0930_1000 = m1[(m1["dt_et"].dt.hour.eq(9) & m1["dt_et"].dt.minute.ge(30))].copy()
    m_0930_1000["gdate"] = m_0930_1000["dt_et"].dt.date
    m_day = (
        m_0930_1000.sort_values("dt_et")
        .groupby("gdate", as_index=False)
        .agg(open_m=("open", "first"),
             high_m=("high", "max"),
             low_m=("low", "min"),
             close_m=("close", "last"),
             volume_m=("volume", "sum"))
        .rename(columns={"gdate": "date"})
    )

    # 10:00-16:00 from hourly bars labeled 11..16 (end-of-hour labels)
    h1 = h1.sort_values("dt_et")
    h_sel = h1[h1["dt_et"].dt.hour.isin([11, 12, 13, 14, 15, 16])].copy()
    h_day = (
        h_sel.groupby(h_sel["dt_et"].dt.date)
        .agg(open_h=("open", "first"),
             high_h=("high", "max"),
             low_h=("low", "min"),
             close_h=("close", "last"),
             volume_h=("volume", "sum"))
        .reset_index(names=["date"])
    )

    daily = pd.merge(m_day, h_day, on="date", how="inner")
    daily["Date"] = pd.to_datetime(daily["date"])  # local calendar date
    daily["Open"] = daily["open_m"]
    daily["High"] = daily[["high_m", "high_h"]].max(axis=1)
    daily["Low"] = daily[["low_m", "low_h"]].min(axis=1)
    daily["Close"] = daily["close_h"]
    daily["Volume"] = daily["volume_m"] + daily["volume_h"]

    # Optionally include extended hours by expanding High/Low/Volume
    if include_extended:
        # crude proxy: add ranges from 09:00-09:30 (pre-open in minute slice) and 16:00 hour if present
        pre = m1[(m1["dt_et"].dt.hour.eq(9) & m1["dt_et"].dt.minute.lt(30))].copy()
        pre["gdate"] = pre["dt_et"].dt.date
        pre_day = pre.groupby("gdate").agg(high_pre=("high", "max"), low_pre=("low", "min"), vol_pre=("volume", "sum")).reset_index()
        daily = daily.merge(pre_day, left_on="date", right_on="gdate", how="left")
        daily["High"] = np.nanmax(np.vstack([daily["High"].values, daily["high_pre"].values]), axis=0)
        daily["Low"] = np.nanmin(np.vstack([daily["Low"].values, daily["low_pre"].values]), axis=0)
        daily["Volume"] = daily["Volume"] + daily["vol_pre"].fillna(0)
        daily.drop(columns=[c for c in ["gdate", "high_pre", "low_pre", "vol_pre"] if c in daily.columns], inplace=True)

    daily = daily[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)
    return daily


def load_official_rth() -> pd.DataFrame:
    if not os.path.exists(OFFICIAL_RTH):
        raise FileNotFoundError(f"Official RTH daily not found: {OFFICIAL_RTH}")
    df = pd.read_csv(OFFICIAL_RTH)

    # Normalize columns: accept lowercase schema (time, open, high, low, close, volume)
    cols = {c.lower(): c for c in df.columns}
    # Map to canonical names
    date_col = cols.get("date") or cols.get("time")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    volume_col = cols.get("volume")
    if date_col is None or open_col is None or high_col is None or low_col is None or close_col is None:
        raise ValueError("Official RTH CSV must include columns for date/time and ohlc (open/high/low/close).")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col]).dt.date.astype(str)
    out["Open"] = df[open_col].astype(float)
    out["High"] = df[high_col].astype(float)
    out["Low"] = df[low_col].astype(float)
    out["Close"] = df[close_col].astype(float)
    if volume_col is not None:
        out["Volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def compute_all_features(daily: pd.DataFrame) -> None:
    # basic indicators
    basic_out = basic_feats.compute_basic_indicators(daily)
    basic_path = os.path.join(ROOT, "processed_data", "spy_rth_indicators_20141231-20250602.csv")
    os.makedirs(os.path.dirname(basic_path), exist_ok=True)
    basic_out.to_csv(basic_path, index=False)

    # gaps/overnight
    gaps_out = gaps_mod.compute(daily)
    gaps_path = os.path.join(ROOT, "processed_data", "spy_rth_gaps_overnight_20141231-20250602.csv")
    gaps_out.to_csv(gaps_path, index=False)

    # liquidity / pressure
    liq_out = liq_mod.compute(daily)
    liq_path = os.path.join(ROOT, "processed_data", "spy_rth_trend_liquidity_pressure_20141231-20250602.csv")
    liq_out.to_csv(liq_path, index=False)

    # trend / mean reversion
    trend_out = trend_mod.compute(daily)
    trend_path = os.path.join(ROOT, "processed_data", "spy_rth_trend_meanrev_20141231-20250602.csv")
    trend_out.to_csv(trend_path, index=False)

    # volatility
    vol_out = vol_mod.compute_from_daily(daily)
    vol_path = os.path.join(ROOT, "processed_data", "spy_rth_volatility_20141231-20250602.csv")
    vol_out.to_csv(vol_path, index=False)

    print("Saved processed datasets:")
    for p in [basic_path, gaps_path, liq_path, trend_path, vol_path]:
        print(" -", p)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and compute SPY RTH features")
    parser.add_argument("--from-intraday", action="store_true", help="Build RTH daily from processed intraday instead of using official RTH file")
    parser.add_argument("--include-extended", type=str, default="true", help="When --from-intraday, include pre/post market (true/false)")
    args = parser.parse_args()

    if args.from_intraday:
        include_ext = str(args.include_extended).lower() in {"1", "true", "yes", "y"}
        daily = build_rth_daily_from_intraday(include_extended=include_ext)
        os.makedirs(os.path.dirname(OUT_RTH_1D), exist_ok=True)
        daily.to_csv(OUT_RTH_1D, index=False)
        print(f"Saved RTH daily (built): {len(daily)} rows -> {OUT_RTH_1D}")
    else:
        daily = load_official_rth()
        print(f"Loaded official RTH daily: {len(daily)} rows from {OFFICIAL_RTH}")

    # Ensure Date is string for consistency
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.date.astype(str)

    compute_all_features(daily)


if __name__ == "__main__":
    main()
