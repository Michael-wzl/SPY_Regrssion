#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaps & Overnight dynamics on RTH daily bars.

- Input default: raw_data/spy_ohlcv_1drth_20141231_20250602.csv
- Output default: processed_data/spy_rth_gaps_overnight_20141231-20250602.csv
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_log(x: pd.Series) -> pd.Series:
    x_valid = x.where(x > 0)
    return np.log(x_valid)


def _rolling_ratio(num: pd.Series, den: pd.Series, window: int) -> pd.Series:
    num_sum = num.rolling(window, min_periods=1).sum()
    den_sum = den.rolling(window, min_periods=1).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = num_sum / den_sum
    ratio = ratio.where(den_sum > 0)
    return ratio


def compute(df: pd.DataFrame, prob_windows: Iterable[int] = (20, 60, 120, 252)) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    # Gaps
    gap_abs = open_ - prev_close
    gap_pct = gap_abs / prev_close
    gap_log = _safe_log(open_) - _safe_log(prev_close)

    # Return decomposition
    overnight_ret = open_ / prev_close - 1.0
    intraday_ret = close / open_ - 1.0
    daily_ret = close / prev_close - 1.0
    overnight_logret = _safe_log(open_) - _safe_log(prev_close)
    intraday_logret = _safe_log(close) - _safe_log(open_)
    daily_logret = _safe_log(close) - _safe_log(prev_close)

    # Gap classes and fill
    is_gap_up = (open_ > prev_close)
    is_gap_down = (open_ < prev_close)
    is_gap = is_gap_up | is_gap_down
    filled_up = is_gap_up & (low <= prev_close)
    filled_down = is_gap_down & (high >= prev_close)
    gap_filled = filled_up | filled_down

    out = pd.DataFrame({
        "Date": df["Date"],
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": df.get("Volume", np.nan),
        "prev_Close": prev_close,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "gap_log": gap_log,
        "overnight_ret": overnight_ret,
        "overnight_logret": overnight_logret,
        "intraday_ret": intraday_ret,
        "intraday_logret": intraday_logret,
        "daily_ret": daily_ret,
        "daily_logret": daily_logret,
        "is_gap": is_gap.astype(float),
        "is_gap_up": is_gap_up.astype(float),
        "is_gap_down": is_gap_down.astype(float),
        "gap_filled": gap_filled.astype(float),
    })

    for w in prob_windows:
        out[f"p_fill_any_{w}d"] = _rolling_ratio((gap_filled & is_gap).astype(int), is_gap.astype(int), w)
        out[f"p_fill_up_{w}d"] = _rolling_ratio((gap_filled & is_gap_up).astype(int), is_gap_up.astype(int), w)
        out[f"p_fill_down_{w}d"] = _rolling_ratio((gap_filled & is_gap_down).astype(int), is_gap_down.astype(int), w)
    return out


def main():
    root = os.path.dirname(__file__)
    in_csv = os.path.join(root, "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")
    out_csv = os.path.join(root, "processed_data", "spy_rth_gaps_overnight_20141231-20250602.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Official RTH daily not found: {in_csv}")
    df_raw = pd.read_csv(in_csv)
    cols = {c.lower(): c for c in df_raw.columns}
    date_col = cols.get("date") or cols.get("time")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    volume_col = cols.get("volume")
    if date_col is None or open_col is None or high_col is None or low_col is None or close_col is None:
        raise ValueError("Input CSV must contain time/date and ohlc columns.")
    df = pd.DataFrame({
        "Date": pd.to_datetime(df_raw[date_col]).dt.date.astype(str),
        "Open": df_raw[open_col].astype(float),
        "High": df_raw[high_col].astype(float),
        "Low": df_raw[low_col].astype(float),
        "Close": df_raw[close_col].astype(float),
        "Volume": pd.to_numeric(df_raw.get(volume_col, np.nan), errors="coerce"),
    })
    out = compute(df)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
