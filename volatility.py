#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read SPY daily data (including OHLCV + dividend + forward returns) from data/spy_ohlcvd_1d.csv,
calculate various volatility-related indicators from 2015-01-01 to 2025-06-01 and export.

Output file: data/spy_volatility_20150101-20250601.csv

List of indicators (Chinese comments include "definition / formula / possible predictive role"):
1) Realized Vol (rolling standard deviation, annualized):
   - Definition: Use the standard deviation of daily log returns r_t = ln(C_t/C_{t-1}) within a rolling window n as volatility estimate;
   - Annualized: σ_annual = √252 × std(r_{t-n+1..t});
   - Predictive role: High volatility often corresponds to rising uncertainty and risk premium adjustment; volatility decline and "volatility clustering" phenomenon can be used for cross-sectional/temporal timing.
2) Parkinson volatility (using only high and low prices, no noisy open/close):
   - Single-day range variance: v_t = (ln(H_t/L_t))^2 / (4 ln 2);
   - Rolling n-day variance: V_t = average(v_{t-n+1..t}), annualized σ_P = √252 × √V_t;
   - Predictive role: Generally more effective than standard deviation of closing returns (utilizing high-low range information), can improve volatility estimation noise.
3) Garman–Klass volatility (using O/H/L/C, assuming zero drift):
   - Single-day variance estimate: v_t = 0.5[ln(H/L)]^2 − (2ln2−1)[ln(C/O)]^2;
   - Rolling annualized as above;
   - Predictive role: More efficient than simple closing return variance, commonly used for intraday range estimation in low-jump, low-drift scenarios.
4) Rogers–Satchell volatility (allows non-zero drift):
   - Single-day variance estimate: v_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O);
   - Rolling annualized as above;
   - Predictive role: More robust in the presence of trends (drift), commonly used in practical volatility modeling.
5) Intraday Range:
   - Absolute range: Range_abs = H−L; relative range: Range_rel = (H−L)/C; log range: Range_log = ln(H/L);
   - Predictive role: Extreme range expansion often corresponds to "announcements/surprises" or emotional volatility, subsequent mean reversion and risk premium changes can be referenced.
6) Realized Skew/Kurt (rolling skewness/kurtosis, based on daily log returns):
   - Skewness: Skew = E[(r−μ)^3]/σ^3;
   - Kurtosis (Fisher): Kurt_excess = E[(r−μ)^4]/σ^4 − 3; (normal = 0)
   - Predictive role:
	   · Negative skew (left tail fat) often corresponds to rising downside risk premium;
	   · High kurtosis (fat tails) indicates rising probability of extreme events, risk control and position sizing need adjustment.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), "raw_data", "spy_ohlcv_1drth_20141231_20250602.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "processed_data", "spy_rth_volatility_20141231-20250602.csv")


def _safe_log(x: pd.Series) -> pd.Series:
	"""Take logarithm of price series, automatically treat non-positive values as missing to avoid log(0) or negative numbers.

	- Input: Price series (e.g., Close, High, Low, Open)
	- Output: Log series, NaN for invalid inputs (<=0)
	"""
	x_valid = x.where(x > 0)
	return np.log(x_valid)


def realized_vol_from_logret(logret: pd.Series, windows: Iterable[int], trading_days: int = 252) -> pd.DataFrame:
	"""Realized Vol (rolling standard deviation, annualized).

	Definition/Formula:
	- Daily log returns: r_t = ln(C_t/C_{t-1})
	- Standard deviation within window: σ_n = std(r_{t-n+1..t})
	- Annualized: σ_annual = √252 × σ_n
	Predictive role:
	- Volatility increase usually accompanies rising risk aversion and valuation compression; volatility decline and "volatility clustering" feature can assist in timing and risk control.
	"""
	out = pd.DataFrame(index=logret.index)
	for n in windows:
		std_n = logret.rolling(n, min_periods=n).std(ddof=0)
		out[f"RealizedVol_{n}d_ann"] = np.sqrt(trading_days) * std_n
	return out


def parkinson_vol(high: pd.Series, low: pd.Series, windows: Iterable[int], trading_days: int = 252) -> pd.DataFrame:
	"""Parkinson volatility estimate (using only high and low prices).

	Formula: v_t = (ln(H_t/L_t))^2 / (4 ln 2), σ_n = √252 × sqrt( mean(v_t, window n) )
	Predictive role: Utilizing high-low range, generally more efficient than closing return std.
	"""
	hl_log = (_safe_log(high) - _safe_log(low))
	vt = (hl_log ** 2) / (4.0 * np.log(2.0))
	out = pd.DataFrame(index=high.index)
	for n in windows:
		mean_v = vt.rolling(n, min_periods=n).mean()
		out[f"ParkinsonVol_{n}d_ann"] = np.sqrt(trading_days * mean_v)
	return out


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
					 windows: Iterable[int], trading_days: int = 252) -> pd.DataFrame:
	"""Garman–Klass volatility estimate (O/H/L/C), assuming zero drift.

	Single-day variance: v_t = 0.5[ln(H/L)]^2 − (2ln2−1)[ln(C/O)]^2
	n-day rolling annualized: σ_n = √252 × sqrt( mean(v_t) )
	Predictive role: More efficient in scenarios without significant drift or jumps.
	"""
	log_hl = _safe_log(high) - _safe_log(low)
	log_co = _safe_log(close) - _safe_log(open_)
	vt = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
	out = pd.DataFrame(index=close.index)
	for n in windows:
		mean_v = vt.rolling(n, min_periods=n).mean()
		out[f"GarmanKlassVol_{n}d_ann"] = np.sqrt(trading_days * mean_v)
	return out


def rogers_satchell_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
						windows: Iterable[int], trading_days: int = 252) -> pd.DataFrame:
	"""Rogers–Satchell volatility (allows non-zero drift).

	Single-day variance: v_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
	n-day rolling annualized: σ_n = √252 × sqrt( mean(v_t) )
	Predictive role: More robust in the presence of trends, commonly used in practical volatility estimation.
	"""
	log_hc = _safe_log(high) - _safe_log(close)
	log_ho = _safe_log(high) - _safe_log(open_)
	log_lc = _safe_log(low) - _safe_log(close)
	log_lo = _safe_log(low) - _safe_log(open_)
	vt = (log_hc * log_ho) + (log_lc * log_lo)
	out = pd.DataFrame(index=close.index)
	for n in windows:
		mean_v = vt.rolling(n, min_periods=n).mean()
		out[f"RogersSatchellVol_{n}d_ann"] = np.sqrt(trading_days * mean_v)
	return out


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
	"""Intraday Range.

	- Absolute range: Range_abs = H−L
	- Relative range: Range_rel = (H−L)/C
	- Log range: Range_log = ln(H/L)
	Predictive role: Extreme range expansion often corresponds to news/emotional shocks, may accompany subsequent volatility decline or trend continuation, need to filter with trend/volume.
	"""
	abs_range = (high - low)
	rel_range = abs_range / close.replace(0, np.nan)
	log_range = (_safe_log(high) - _safe_log(low))
	return pd.DataFrame({
		"Range_abs": abs_range,
		"Range_rel_close": rel_range,
		"Range_log": log_range,
	}, index=close.index)


def realized_skew_kurt(logret: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
	"""Rolling skewness/kurtosis (based on daily log returns).

	- Skew: E[(r−μ)^3]/σ^3, measures asymmetry of distribution;
	- Kurt_excess (Fisher): E[(r−μ)^4]/σ^4 − 3, normal = 0;
	Predictive role: Negative skew and high kurtosis correspond to left-tail risk and tail thickness, often related to risk compensation and timing.
	Note: pandas' rolling().skew()/kurt() uses sample estimation, kurt returns Fisher definition (normal=0).
	"""
	out = pd.DataFrame(index=logret.index)
	for n in windows:
		out[f"RealizedSkew_{n}d"] = logret.rolling(n, min_periods=n).skew()
		out[f"RealizedKurtExcess_{n}d"] = logret.rolling(n, min_periods=n).kurt()
	return out


def compute_from_daily(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute volatility indicators from a given RTH daily dataframe.

	Contract:
	- Input df columns: Date, Open, High, Low, Close[, Volume]
	- Return: DataFrame with Date + volatility indicators + passthrough OHLCV if present.
	"""
	# Ensure datetime order
	df = df.sort_values("Date").reset_index(drop=True).copy()
	if not np.issubdtype(pd.to_datetime(df["Date"]).dtype, np.datetime64):
		# normalize Date type for joins and filtering
		df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

	open_ = df["Open"].astype(float)
	high = df["High"].astype(float)
	low = df["Low"].astype(float)
	close = df["Close"].astype(float)
	logret = _safe_log(close).diff()

	vol_windows = [5, 10, 20, 60, 120, 252]
	shape_windows = [20, 60, 120]

	realized = realized_vol_from_logret(logret, vol_windows)
	parkinson = parkinson_vol(high, low, vol_windows)
	gk = garman_klass_vol(open_, high, low, close, vol_windows)
	rs = rogers_satchell_vol(open_, high, low, close, vol_windows)
	rng = intraday_range(high, low, close)
	shape = realized_skew_kurt(logret, shape_windows)

	out = pd.DataFrame(index=df.index)
	out["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
	out = out.join(realized)
	out = out.join(parkinson)
	out = out.join(gk)
	out = out.join(rs)
	out = out.join(rng)
	out = out.join(shape)

	for c in ["Open", "High", "Low", "Close", "Volume"]:
		if c in df.columns:
			out[c] = df[c].values
	return out


def main():
	# Prefer official RTH daily file
	if not os.path.exists(DATA_PATH):
		raise FileNotFoundError(f"Official RTH daily not found: {DATA_PATH}")
	df_raw = pd.read_csv(DATA_PATH)
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

	out = compute_from_daily(df)
	# Filter to repo's common range
	mask = (pd.to_datetime(out["Date"]) >= pd.Timestamp("2014-12-31")) & (pd.to_datetime(out["Date"]) <= pd.Timestamp("2025-06-02"))
	out_range = out.loc[mask].reset_index(drop=True)

	os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
	out_range.to_csv(OUT_PATH, index=False)
	print(f"Saved {len(out_range)} rows to {OUT_PATH}")
	print(out_range.head(3))
	print("...")
	print(out_range.tail(3))


if __name__ == "__main__":
	main()

