import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def _safe_import_darts():
	try:
		from darts import TimeSeries  # noqa: F401
		from darts.dataprocessing.transformers import Scaler  # noqa: F401
		from darts.metrics import mse  # noqa: F401
		from darts.models import TFTModel  # noqa: F401
		return True
	except Exception as e:
		print("[ERROR] Failed to import 'darts'. Please install dependencies first:")
		print("        pip install -r requirements.txt")
		print(f"        Import error: {e}")
		return False


@dataclass
class Config:
	# Paths
	base_dir: str = os.path.dirname(os.path.abspath(__file__))
	data_dir: str = os.path.join(base_dir, "data")
	ohlcv_path: str = os.path.join(
		data_dir, "ohlcv", "spy_ohlcv_1drth_20141231_20250602.csv"
	)

	# Date ranges
	train_start: str = "2015-01-01"
	train_end: str = "2021-12-31"
	test_start: str = "2022-01-01"
	test_end: str = "2025-06-01"

	# Model params (kept modest for runtime)
	input_chunk_length: int = 90
	output_chunk_length: int = 7
	hidden_size: int = 32
	lstm_layers: int = 1
	num_attention_heads: int = 4
	dropout: float = 0.1
	batch_size: int = 64
	n_epochs: int = 30
	random_state: int = 42

	# Output
	out_dir: str = os.path.join(base_dir, "outputs")
	pred_csv: str = os.path.join(out_dir, "tft_predictions.csv")


def read_csv_with_any_date(path: str, date_cols: List[str]) -> Optional[pd.DataFrame]:
	if not os.path.exists(path):
		return None
	try:
		df = pd.read_csv(path)
		# Find a date column among candidates
		found = None
		for c in date_cols:
			if c in df.columns:
				found = c
				break
		if found is None:
			return None
		df = df.rename(columns={found: "date"})
		df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
		df = df.sort_values("date").drop_duplicates("date")
		return df
	except Exception:
		return None


def load_base_ohlcv(cfg: Config) -> pd.DataFrame:
	df = read_csv_with_any_date(cfg.ohlcv_path, ["time", "date", "Date"])
	if df is None:
		raise FileNotFoundError(f"Cannot read OHLCV file at {cfg.ohlcv_path}")
	# Keep canonical columns
	keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
	df = df[keep]
	return df


def load_covariates(cfg: Config) -> List[pd.DataFrame]:
	covs: List[pd.DataFrame] = []
	# Volatility and options
	for name in [
		("volatility_and_options/vix_history.csv", ["Date"], ["VIX"]),
		("volatility_and_options/vix3m_history.csv", ["Date"], ["VIX3M"]),
		("volatility_and_options/vvix_history.csv", ["Date"], ["VVIX"]),
		("volatility_and_options/skew_history.csv", ["Date"], ["SKEW"]),
	]:
		path = os.path.join(cfg.data_dir, name[0])
		df = read_csv_with_any_date(path, name[1])
		if df is not None:
			cols = ["date"] + [c for c in name[2] if c in df.columns]
			covs.append(df[cols])

	# Macro (drop OHLCV duplicates if present)
	macro_path = os.path.join(cfg.data_dir, "macro_and_growth", "spy_with_macro_positioning.csv")
	macro = read_csv_with_any_date(macro_path, ["Date", "date"])
	if macro is not None:
		drop_cols = {"AdjClose", "Close", "High", "Low", "Open", "Volume", "VIX"}
		keep_cols = ["date"] + [c for c in macro.columns if c not in drop_cols | {"date"}]
		covs.append(macro[keep_cols])

	# Sentiment
	senti_path = os.path.join(cfg.data_dir, "sentiment_and_news", "spy_average_sentiment_score.csv")
	senti = read_csv_with_any_date(senti_path, ["date", "Date"])  # file uses 'date'
	if senti is not None:
		cols = ["date"] + [c for c in ["sentiment_score"] if c in senti.columns]
		covs.append(senti[cols])

	# Sector features
	sector_path = os.path.join(cfg.data_dir, "sector_and_cross_assets", "spy_sector_data.csv")
	sector = read_csv_with_any_date(sector_path, ["Date", "date"])
	if sector is not None:
		# Prefer engineered features
		feature_cols = [c for c in sector.columns if c.startswith("Features_") or c.startswith("ZScores_")]
		if not feature_cols:
			# Fallback: all numeric except date
			feature_cols = [c for c in sector.columns if c != "date" and pd.api.types.is_numeric_dtype(sector[c])]
		covs.append(sector[["date"] + feature_cols])

	# Commodities & FX (drop OHLCV)
	fx_path = os.path.join(cfg.data_dir, "commodities_and_fx", "spy_fx_flows.csv")
	fx = read_csv_with_any_date(fx_path, ["time", "Date", "date"])  # file uses 'time'
	if fx is not None:
		drop_cols = {"open", "high", "low", "close", "volume"}
		keep_cols = ["date"] + [c for c in fx.columns if c not in drop_cols | {"date"}]
		covs.append(fx[keep_cols])

	return covs


def build_merged_frame(cfg: Config) -> pd.DataFrame:
	base = load_base_ohlcv(cfg)
	# normalize base dates to midnight and keep trading calendar
	base["date"] = pd.to_datetime(base["date"]).dt.normalize()
	trading_dates = base["date"].drop_duplicates().sort_values()

	covs = load_covariates(cfg)

	# Merge all on date (outer join then sort and ffill)
	df = base.copy()
	for cv in covs:
		df = pd.merge(df, cv, on="date", how="outer")

	df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
	# Normalize to midnight
	df["date"] = pd.to_datetime(df["date"]).dt.normalize()

	# Restrict to full window
	mask = (df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.test_end))
	df = df[mask].copy()

	# Reindex to continuous daily calendar and forward/backward fill covariates and target for modeling convenience
	# Ensure unique index before reindex
	df = df.drop_duplicates(subset=["date"]).set_index("date")
	full_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
	df = df.reindex(full_days)
	df.index.name = "date"
	df = df.sort_index().ffill().bfill().reset_index()

	# Mark original trading days for evaluation filtering
	df["is_trading_day"] = df["date"].isin(set(trading_dates))

	df = df.sort_values("date")
	# Clean numeric columns: remove infs, ensure no NaNs remain
	num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
	df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
	df[num_cols] = df[num_cols].ffill().bfill()
	# In rare case still NaN (all-missing column), fill with column medians or zeros
	med = df[num_cols].median(numeric_only=True)
	df[num_cols] = df[num_cols].fillna(med).fillna(0.0)

	# Ensure target has no NaN and is float
	df = df.dropna(subset=["close"]).copy()
	df["close"] = df["close"].astype(float)
	return df


def pandas_to_timeseries(df: pd.DataFrame, target_col: str, past_cov_cols: List[str]):
	from darts import TimeSeries

	# Use daily frequency and rely on prior reindexing to continuous daily calendar
	df2 = df.copy()
	ts_target = TimeSeries.from_dataframe(
		df2, time_col="date", value_cols=target_col, fill_missing_dates=True, freq="D"
	)

	ts_past = None
	if past_cov_cols:
		ts_past = TimeSeries.from_dataframe(
			df2, time_col="date", value_cols=past_cov_cols, fill_missing_dates=True, freq="D"
		)
		# Align covariates to target timeline
		ts_past = ts_past.slice_intersect(ts_target)

	return ts_target, ts_past


def split_series_by_date(series, start: str, end: str):
	return series.slice(pd.to_datetime(start), pd.to_datetime(end))


def train_and_evaluate(cfg: Config):
	if not _safe_import_darts():
		return

	from darts.dataprocessing.transformers import Scaler
	from sklearn.metrics import mean_squared_error
	from darts.models import TFTModel

	warnings.filterwarnings("ignore")

	print("[1/5] Loading and merging data...")
	df = build_merged_frame(cfg)

	# Identify past covariate columns: all numeric except target
	numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
	target_col = "close"
	past_cov_cols = [c for c in numeric_cols if c != target_col]

	print(f"    total rows: {len(df)}; features: {len(past_cov_cols)}")

	print("[2/5] Building TimeSeries...")
	ts_target, ts_past = pandas_to_timeseries(df, target_col, past_cov_cols)

	# Split into train/test ranges
	ts_train = split_series_by_date(ts_target, cfg.train_start, cfg.train_end)
	ts_test = split_series_by_date(ts_target, cfg.test_start, cfg.test_end)
	pc_train = split_series_by_date(ts_past, cfg.train_start, cfg.train_end) if ts_past is not None else None

	# Scale target and covariates
	print("[3/5] Scaling...")
	target_scaler = Scaler()
	ts_train_s = target_scaler.fit_transform(ts_train)
	ts_test_s = target_scaler.transform(ts_test)
	ts_full_s = target_scaler.transform(ts_target)

	cov_scaler = Scaler()
	pc_train_s = cov_scaler.fit_transform(pc_train) if pc_train is not None else None
	pc_full_s = cov_scaler.transform(ts_past) if ts_past is not None else None

	# Initialize TFT
	print("[4/5] Training TFT model...")
	model = TFTModel(
		input_chunk_length=cfg.input_chunk_length,
		output_chunk_length=cfg.output_chunk_length,
		hidden_size=cfg.hidden_size,
		lstm_layers=cfg.lstm_layers,
		num_attention_heads=cfg.num_attention_heads,
		dropout=cfg.dropout,
		batch_size=cfg.batch_size,
		n_epochs=cfg.n_epochs,
		add_relative_index=True,
		random_state=cfg.random_state,
		save_checkpoints=False,
		pl_trainer_kwargs={"enable_progress_bar": True, "accelerator": "cpu"},
	)

	model.fit(series=ts_train_s, past_covariates=pc_train_s, verbose=True)

	# Historical forecasts across the test window, predicting 1-step ahead iteratively
	print("[5/5] Forecasting and evaluating...")
	preds_s = model.historical_forecasts(
		series=ts_full_s,
		past_covariates=pc_full_s,
		start=pd.to_datetime(cfg.test_start),
		forecast_horizon=1,
		stride=1,
		retrain=False,
		last_points_only=True,
		verbose=True,
	)

	# Ensure TimeSeries output
	from darts import TimeSeries
	if isinstance(preds_s, list):
		preds_s = TimeSeries.from_times_and_values(
			times=[ts.time_index[-1] for ts in preds_s],
			values=np.array([ts.values()[-1] for ts in preds_s]).squeeze(),
		)

	# Align with test period
	preds = target_scaler.inverse_transform(preds_s)
	ts_test_aligned = ts_test.slice_intersect(preds)
	preds_aligned = preds.slice_intersect(ts_test)

	# Compute MSE on true trading days only
	# Build trading-day mask from original df
	eval_df = pd.DataFrame({
		"date": ts_test_aligned.time_index,
		"true_close": ts_test_aligned.values().squeeze(),
	})
	eval_df = eval_df.merge(
		pd.DataFrame({
			"date": preds_aligned.time_index,
			"pred_close": preds_aligned.values().squeeze(),
		}), on="date", how="inner"
	)
	# bring the trading-day indicator from the merged frame
	eval_df = eval_df.merge(df[["date", "is_trading_day"]], on="date", how="left")
	eval_df = eval_df[eval_df["is_trading_day"] == True]
	test_mse = float(mean_squared_error(eval_df["true_close"], eval_df["pred_close"]))
	print(f"\nMSE on test trading days [{cfg.test_start} .. {cfg.test_end}]: {test_mse:.6f}")

	# Save predictions
	os.makedirs(cfg.out_dir, exist_ok=True)
	df_pred = pd.DataFrame({
		"date": preds_aligned.time_index,
		"pred_close": preds_aligned.values().squeeze(),
	})
	df_true = pd.DataFrame({
		"date": ts_test_aligned.time_index,
		"true_close": ts_test_aligned.values().squeeze(),
	})
	out = pd.merge(df_true, df_pred, on="date", how="inner")
	out.to_csv(cfg.pred_csv, index=False)
	print(f"Predictions saved to: {cfg.pred_csv}")


def parse_args() -> Config:
	p = argparse.ArgumentParser(description="Train/evaluate TFT on SPY daily data.")
	p.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
	p.add_argument("--input-chunk-length", type=int, default=90, help="Encoder input length (lookback).")
	p.add_argument("--output-chunk-length", type=int, default=7, help="Decoder output length (multi-step horizon).")
	p.add_argument("--hidden-size", type=int, default=32, help="Hidden size of TFT model.")
	p.add_argument("--fast", action="store_true", help="Shortcut for quick smoke test (epochs=3, hidden-size=16).")
	args = p.parse_args()
	cfg = Config(
		input_chunk_length=args.input_chunk_length,
		output_chunk_length=args.output_chunk_length,
		hidden_size=args.hidden_size,
		n_epochs=args.epochs,
	)
	if args.fast:
		cfg.n_epochs = 3
		cfg.hidden_size = 16
	return cfg


def main():
	cfg = parse_args()
	train_and_evaluate(cfg)


if __name__ == "__main__":
	main()

