#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic OHLCV technical indicators used across the project.

This module provides reusable functions and an optional CLI to compute a
compact set of common indicators from a given RTH daily OHLCV dataframe.

Input expectation (DataFrame): columns Date, Open, High, Low, Close, (Volume optional).

Outputs (when run as script): processed_data/spy_rth_indicators_20141231-20250602.csv
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------- Core indicator functions ---------------------- #

def compute_rolling_returns(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    daily = close.pct_change()
    one_plus = (1 + daily).replace({0.0: 1.0})
    for n in windows:
        roll = one_plus.rolling(n, min_periods=n).apply(np.prod, raw=True) - 1.0
        df[f"roll_ret_{n}d"] = roll
    return df


def compute_sma(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    for n in windows:
        df[f"SMA_{n}"] = close.rolling(n, min_periods=n).mean()
    return df


def compute_ema(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    for n in windows:
        df[f"EMA_{n}"] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    return df


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False, min_periods=slow + signal - 1).mean()
    hist = dif - dea
    return pd.DataFrame({
        "MACD_line": dif,
        "MACD_signal": dea,
        "MACD_hist": hist,
    }, index=close.index)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ad = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = au / ad.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return pd.DataFrame({f"RSI_{period}": rsi}, index=close.index)


def compute_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    lowest_n = low.rolling(k_period, min_periods=k_period).min()
    highest_n = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest_n - lowest_n).replace(0, np.nan)
    k = 100.0 * (close - lowest_n) / denom
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({
        f"Stoch_%K_{k_period}": k,
        f"Stoch_%D_{d_period}": d,
    }, index=close.index)


def compute_bbands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return pd.DataFrame({
        f"BB_Middle_{period}": mid,
        f"BB_Upper_{period}_{int(num_std)}": upper,
        f"BB_Lower_{period}_{int(num_std)}": lower,
        f"BB_Width_{period}": width,
    }, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({f"ATR_{period}": atr}, index=close.index)


def compute_basic_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute the set of basic OHLCV indicators on the provided dataframe.

    Contract:
    - Input: daily dataframe with columns: Date, Open, High, Low, Close[, Volume]
    - Output: DataFrame with Date and indicators, aligned to input index order.
    """
    df = daily.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    out = pd.DataFrame({"Date": pd.to_datetime(df["Date"])})
    out["ret_1d"] = close.pct_change()
    out["logret_1d"] = np.log(close.replace(0, np.nan)).diff()
    out = out.join(compute_rolling_returns(close, windows=[5, 10, 20]))
    out = out.join(compute_sma(close, windows=[5, 10, 20, 50, 200]))
    out = out.join(compute_ema(close, windows=[5, 10, 20, 50, 200]))
    out = out.join(compute_macd(close, fast=12, slow=26, signal=9))
    out = out.join(compute_rsi(close, period=14))
    out = out.join(compute_stoch(high, low, close, k_period=14, d_period=3))
    out = out.join(compute_bbands(close, period=20, num_std=2.0))
    out = out.join(compute_atr(high, low, close, period=14))
    # Reference OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            out[c] = df[c].values
    return out


def main():
    root = os.path.dirname(__file__)
    in_csv = os.path.join(root, "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")
    out_csv = os.path.join(root, "processed_data", "spy_rth_indicators_20141231-20250602.csv")

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Official RTH daily not found: {in_csv}")

    df_raw = pd.read_csv(in_csv)
    # Normalize lowercase schema if needed
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
    df = df.sort_values("Date").reset_index(drop=True)

    out = compute_basic_indicators(df)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
