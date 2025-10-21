#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend and mean-reversion indicators on RTH daily bars.

- Input default: raw_data/spy_ohlcv_1drth_20141231_20250602.csv
- Output default: processed_data/spy_rth_trend_meanrev_20141231-20250602.csv
"""

from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
import pandas as pd

import basic_feats as ind


def compute_price_momentum(close: pd.Series, max_n: int = 20) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for n in range(1, max_n + 1):
        out[f"mom_{n}d"] = close.pct_change(n)
    return out


def _time_since_last_extreme(x: np.ndarray, is_max: bool) -> float:
    if x.size == 0:
        return np.nan
    mask = ~np.isnan(x)
    if not np.any(mask):
        return np.nan
    vals = x[mask]
    idx_ext = np.argmax(vals) if is_max else np.argmin(vals)
    orig_idx = np.flatnonzero(mask)[idx_ext]
    return float(x.size - 1 - orig_idx)


def time_since_high_low(high: pd.Series, low: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=high.index)
    for L in windows:
        out[f"days_since_high_{L}d"] = high.rolling(L, min_periods=L).apply(lambda x: _time_since_last_extreme(x.values, True), raw=False)
        out[f"days_since_low_{L}d"] = low.rolling(L, min_periods=L).apply(lambda x: _time_since_last_extreme(x.values, False), raw=False)
    return out


def distance_from_ma(close: pd.Series, ma_windows: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    sma = ind.compute_sma(close, ma_windows)
    ema = ind.compute_ema(close, ma_windows)
    for n in ma_windows:
        s = sma[f"SMA_{n}"]
        e = ema[f"EMA_{n}"]
        df[f"dist_sma_{n}d"] = (close - s) / s.replace(0, np.nan)
        df[f"dist_ema_{n}d"] = (close - e) / e.replace(0, np.nan)
    return df


def compute_zscore_returns(logret: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=logret.index)
    for w in windows:
        mu = logret.rolling(w, min_periods=w).mean()
        sd = logret.rolling(w, min_periods=w).std(ddof=0)
        out[f"zret_{w}d"] = (logret - mu) / sd.replace(0, np.nan)
    return out


def rolling_autocorr(logret: pd.Series, windows: Iterable[int], lag: int = 1) -> pd.DataFrame:
    def _autocorr_window(x: np.ndarray) -> float:
        if np.any(np.isnan(x)):
            return np.nan
        if x.size <= lag:
            return np.nan
        a = x[lag:]
        b = x[:-lag]
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    out = pd.DataFrame(index=logret.index)
    for w in windows:
        out[f"autocorr_ret_lag{lag}_{w}d"] = logret.rolling(w, min_periods=w).apply(lambda x: _autocorr_window(x.values), raw=False)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    logret = np.log(close.replace(0, np.nan)).diff()
    ret_1d = close.pct_change()

    momentum_max_n = 20
    ts_ext_windows: List[int] = [20, 60, 120, 252]
    ma_windows: List[int] = [5, 10, 20, 50, 200]
    z_windows: List[int] = [5, 20, 60]
    ac_windows: List[int] = [20, 60, 120]

    mom = compute_price_momentum(close, max_n=momentum_max_n)
    ts_ext = time_since_high_low(high, low, windows=ts_ext_windows)
    dist_ma = distance_from_ma(close, ma_windows)
    zret = compute_zscore_returns(logret, z_windows)
    ac1 = rolling_autocorr(logret, ac_windows, lag=1)

    out = pd.DataFrame(index=df.index)
    out["Date"] = pd.to_datetime(df["Date"]) 
    out["ret_1d"] = ret_1d
    out["logret_1d"] = logret
    out = out.join(mom)
    out = out.join(ts_ext)
    out = out.join(dist_ma)
    out = out.join(zret)
    out = out.join(ac1)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            out[c] = df[c].values
    return out


def main():
    root = os.path.dirname(__file__)
    in_csv = os.path.join(root, "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")
    out_csv = os.path.join(root, "processed_data", "spy_rth_trend_meanrev_20141231-20250602.csv")
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
