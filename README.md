# SPY RTH Daily Feature Calculations and Data Description

This repository contains scripts and indicators computed from SPY RTH daily data (regular trading hours 09:30–16:00, America/New_York). The refactored and modularized files are:

- gaps.py: gap and overnight decomposition
- liquidity.py: liquidity and trading pressure
- basic_feats.py: basic OHLCV technical indicators (SMA/EMA/MACD/RSI/ATR, etc.)
- trend.py: trend and mean-reversion (momentum, distance-to-moving-average, extreme lookback, return autocorrelation, etc.)
- volatility.py: volatility-related metrics (Realized/PK/GK/RS, range, skewness/kurtosis)
- aggregate.py: aggregation and unified entry point (supports using the official RTH daily data or reconstructing RTH daily from intraday data, with an option to include pre-/post-market data)

## Usage

1) Use the official RTH daily data directly (recommended)

- Place the official RTH daily CSV at raw_data/spy_ohlcv_1drth_20141231_20250602.csv
- Run the main entry:

```bash
python3 aggregate.py
```

- Or run modules individually:

```bash
python3 basic_feats.py
python3 gaps.py
python3 liquidity.py
python3 trend.py
python3 volatility.py
```

2) Reconstruct RTH daily data from prepared intraday data

- Require processed_data/spy_ohlcv_1h_20141231_20250602.csv and processed_data/spy_ohlcv_1m910_20241231_20250602.csv
- Run:

```bash
python3 aggregate.py --from-intraday
```

- To include extended hours (pre-/post-market) when reconstructing daily high/low/volume:

```bash
python3 aggregate.py --from-intraday --include-extended true
```

- To exclude extended hours:

```bash
python3 aggregate.py --from-intraday --include-extended false
```

## Output location (processed_data/)

- spy_ohlcv_rth_1d_20141231-20250602.csv: RTH daily OHLCV (generated when using --from-intraday; not required when using the official RTH file)
- spy_rth_indicators_20141231-20250602.csv: basic OHLCV technical indicators
- spy_rth_gaps_overnight_20141231-20250602.csv: gap and overnight-related indicators
- spy_rth_trend_liquidity_pressure_20141231-20250602.csv: liquidity and trading pressure indicators
- spy_rth_trend_meanrev_20141231-20250602.csv: trend and mean-reversion indicators
- spy_rth_volatility_20141231-20250602.csv: volatility and distribution-shape indicators

## Raw data (raw_data/)

- spy_ohlcv_1drth_20141231_20250602.csv (official RTH daily)
	- Fields: time,close,high,low,open,volume
	- Time: trading hours 09:30–16:00 (RTH), America/New_York
	- Purpose: preferred input for indicator calculations to avoid biases from timezone or stitching issues
    - Source: Downloaded from QuantConnect with `download_from_QC.py`
- spy_ohlcv_1h_20141231_20250602.csv (hourly OHLCV including extended hours)
    - Fields: time,open,high,low,close,volume
    - Time: hourly bars including pre-/post-market, America/New_York
    - Purpose: used to reconstruct RTH daily OHLCV when official RTH daily is not available
    - Source: Downloaded from QuantConnect with `download_from_QC.py` and copy-pasted output manually
- spy_ohlcv_1m910_20241231_20250602.csv (minute-level 09:00–10:00 segment)
    - Fields: time,open,high,low,close,volume
    - Time: minute bars from 09:00 to 10:00 (pre-market segment), America/New_York
    - Purpose: used to reconstruct RTH daily OHLCV when official RTH daily is not available
    - Source: Downloaded from QuantConnect with `download_from_QC.py` and copy-pasted output manually

## Intermediate / reference data (processed_data/ and trash/data/)

- processed_data/ may contain historical builds or intermediate data:
	- spy_ohlcv_1h_20141231_20250602.csv: hourly OHLCV including extended hours
	- spy_ohlcv_1m910_20241231_20250602.csv: minute-level 09:00–10:00 segment
	- Use aggregate.py --from-intraday to reconstruct RTH daily from these files
- trash/data/ holds historical validation or source snippets for reference only.

## Notes

- Indicator windows and date ranges: by default the scripts do not enforce a specific date range. Filter as needed to align with other datasets.
- Gap fill determination is based on whether the daily High/Low touches the previous close; it does not capture intraday path or timing. For finer granularity use minute-level (including pre-/post-market) data.
- Timezone: official RTH daily data is naturally aligned to trading hours. When reconstructing from intraday data, ensure timestamps are handled in ET (America/New_York).

