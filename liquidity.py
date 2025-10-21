#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liquidity and trading pressure indicators on RTH daily OHLCV.

- Input default: raw_data/spy_ohlcv_1drth_20141231_20250602.csv
- Output default: processed_data/spy_rth_trend_liquidity_pressure_20141231-20250602.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def compute_volume_percentiles(volume: pd.Series, windows=(20, 60, 252)) -> pd.DataFrame:
    df = pd.DataFrame(index=volume.index)
    v = volume.astype(float)
    for n in windows:
        def _pct(x: np.ndarray) -> float:
            if len(x) < n or not np.isfinite(x[-1]):
                return np.nan
            return float(np.mean(x <= x[-1]))
        df[f"vol_pct_{n}"] = v.rolling(n, min_periods=n).apply(_pct, raw=True)
    return df


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    delta = close.diff()
    sign = np.sign(delta).replace({np.inf: np.nan, -np.inf: np.nan})
    sign = sign.fillna(0.0)
    obv = (sign * volume.fillna(0.0)).cumsum()
    obv_mean20 = obv.rolling(20, min_periods=20).mean()
    obv_std20 = obv.rolling(20, min_periods=20).std(ddof=0)
    obv_z20 = (obv - obv_mean20) / obv_std20
    return pd.DataFrame({"OBV": obv, "OBV_z20": obv_z20}, index=close.index)


def compute_volume_surge_flags(volume: pd.Series, windows=(20, 60), percentile: float = 0.95) -> pd.DataFrame:
    df = pd.DataFrame(index=volume.index)
    v = volume.astype(float)
    qname = int(round(percentile * 100))
    for n in windows:
        avg = v.rolling(n, min_periods=n).mean()
        std = v.rolling(n, min_periods=n).std(ddof=0)
        thr = v.rolling(n, min_periods=n).quantile(percentile)
        ratio = v / avg
        flag = v >= thr
        z = (v - avg) / std
        df[f"vol_avg_{n}"] = avg
        df[f"vol_ratio_{n}"] = ratio
        df[f"vol_p{qname}_{n}"] = thr
        df[f"vol_gt_p{qname}_{n}"] = flag.astype(float)
        df[f"vol_z_{n}"] = z
    return df


def compute_spread_proxies(high: pd.Series, low: pd.Series, close: pd.Series, roll_window: int = 20) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    dp = close.diff()
    cov = (dp.rolling(roll_window, min_periods=roll_window).cov(dp.shift(1)))
    roll_abs = (2.0 * (-cov).clip(lower=0.0).pow(0.5))
    roll_pct = roll_abs / close
    hl_rel = (high - low) / close
    hl_log = (np.log(high.replace(0, np.nan)) - np.log(low.replace(0, np.nan)))
    df[f"roll_spread_abs_{roll_window}"] = roll_abs
    df[f"roll_spread_pct_{roll_window}"] = roll_pct
    df["HL_rel_range"] = hl_rel
    df["HL_log_range"] = hl_log
    return df


def compute_amihud_illiq(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.DataFrame:
    ret = close.pct_change()
    dollar_vol = (close * volume).replace(0, np.nan)
    daily = ret.abs() / dollar_vol
    illiq_mean = daily.rolling(window, min_periods=window).mean()
    illiq_median = daily.rolling(window, min_periods=window).median()
    return pd.DataFrame({
        f"amihud_illiq_mean_{window}": illiq_mean,
        f"amihud_illiq_median_{window}": illiq_median,
    }, index=close.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df.get("Volume", pd.Series(index=df.index, dtype=float)).astype(float)

    out = pd.DataFrame(index=df.index)
    out["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    out = out.join(compute_volume_percentiles(vol, windows=(20, 60, 252)))
    out = out.join(compute_obv(close, vol))
    out = out.join(compute_volume_surge_flags(vol, windows=(20, 60), percentile=0.95))
    out = out.join(compute_spread_proxies(high, low, close, roll_window=20))
    out = out.join(compute_amihud_illiq(close, vol, window=20))

    # Reference OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            out[c] = df[c].values
    return out


def main():
    root = os.path.dirname(__file__)
    in_csv = os.path.join(root, "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")
    out_csv = os.path.join(root, "processed_data", "spy_rth_trend_liquidity_pressure_20141231-20250602.csv")
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
