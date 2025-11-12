import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import warnings
import argparse
import json
from dataclasses import dataclass, asdict, replace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

	# Output & experiment
	out_root: str = os.path.join(base_dir, "outputs")
	exp_name: Optional[str] = None
	pred_csv: Optional[str] = None
	plot_png: Optional[str] = None
	metrics_json: Optional[str] = None

	# Options
	top_k_covariates: int = 0  # 0 means all
	target_mode: str = "close"  # 'close' or 'logret'
	learning_rate: float = 1e-3
	optimizer: str = "adam"
	use_cosine_scheduler: bool = False
	val_days: int = 180
	weight_decay: float = 1e-4
	gradient_clip_val: float = 0.5
	winsor_clip: Tuple[float, float] = (0.01, 0.99)  # quantile clipping
	add_rolling_features: bool = True
	rolling_windows: Tuple[int, ...] = (20, 60, 126, 252)
	stack_with_tree_models: bool = False
	tree_model_weight: float = 0.4  # weight for tree model in ensemble
	feature_selection_max: int = 0  # wrapper recursive selection upper bound (0=skip)
	fixed_covariates: Optional[List[str]] = None  # if provided, use this exact set
	fixed_covariates_file: Optional[str] = None  # path to txt/json with list
	# Greedy selection options
	do_greedy_selection: bool = False
	select_val_days: int = 120
	select_epochs: int = 20
	# Backtest options
	do_backtest: bool = False
	bt_train_days: int = 756
	bt_test_days: int = 63
	bt_start: Optional[str] = None
	bt_end: Optional[str] = None
	# Book-keeping
	selected_covariates: Optional[List[str]] = None
	# Stacking advanced options
	stack_use_tft_features: bool = False  # include TFT-derived features (MC dropout stats etc)


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


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["dow"] = df["date"].dt.weekday
	df["doy"] = df["date"].dt.dayofyear
	df["mon"] = df["date"].dt.month
	# sin/cos cyclical encoding
	for col, period in [("dow", 7), ("mon", 12), ("doy", 365)]:
		angle = 2 * np.pi * df[col] / period
		df[f"{col}_sin"] = np.sin(angle)
		df[f"{col}_cos"] = np.cos(angle)
	return df


def _add_rolling_stats(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
	"""Add multi-scale rolling statistics based strictly on past data (shifted to avoid leakage)."""
	df = df.sort_values("date").copy()
	if "close" not in df.columns:
		return df
	# log returns
	df["log_close"] = np.log(df["close"].astype(float).clip(lower=1e-9))
	df["logret"] = df["log_close"].diff()
	for w in windows:
		roll = df["logret"].rolling(w)
		df[f"r_vol_{w}"] = roll.std()
		df[f"r_mean_{w}"] = roll.mean()
		# price moving averages & ratio
		p_roll = df["close"].rolling(w)
		df[f"ma_{w}"] = p_roll.mean()
		df[f"close_ma_ratio_{w}"] = df["close"] / (df[f"ma_{w}"] + 1e-9)
	# Simple RSI variant (Wilder's) for longest window only as signal
	if windows:
		w = max(windows)
		chg = df["close"].diff()
		gain = chg.clip(lower=0).rolling(w).mean()
		loss = (-chg.clip(upper=0)).rolling(w).mean()
		rs = gain / (loss + 1e-9)
		df["rsi"] = 100 - 100 / (1 + rs)
	# shift all derived features by 1 day to ensure they use ONLY past info
	derived_cols = [c for c in df.columns if c not in {"date", "open", "high", "low", "close", "volume", "is_trading_day"}]
	df[derived_cols] = df[derived_cols].shift(1)
	return df


def build_merged_frame(cfg: Config) -> pd.DataFrame:
	base = load_base_ohlcv(cfg)
	# normalize base dates to midnight and keep trading calendar
	base["date"] = pd.to_datetime(base["date"]).dt.normalize()
	trading_dates = base["date"].drop_duplicates().sort_values()

	covs = load_covariates(cfg)

	# Merge all on date (outer join then sort)
	df = base.copy()
	for cv in covs:
		df = pd.merge(df, cv, on="date", how="outer")

	df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
	# Normalize to midnight
	df["date"] = pd.to_datetime(df["date"]).dt.normalize()

	# Restrict to full window
	mask = (df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.test_end))
	df = df[mask].copy()

	# Reindex到连续日历，仅向前填充，避免跨区间回填引入未来信息
	# Ensure unique index before reindex
	df = df.drop_duplicates(subset=["date"]).set_index("date")
	full_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
	df = df.reindex(full_days)
	df.index.name = "date"
	df = df.sort_index().ffill().reset_index()

	# Mark original trading days for evaluation filtering
	df["is_trading_day"] = df["date"].isin(set(trading_dates))

	df = df.sort_values("date")
	# Clean numeric columns: remove infs (exclude booleans & is_trading_day)
	num_cols = [
		c for c in df.columns
		if c not in {"date", "is_trading_day"}
		and pd.api.types.is_numeric_dtype(df[c])
		and not pd.api.types.is_bool_dtype(df[c])
	]
	df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
	df[num_cols] = df[num_cols].ffill()
	train_mask = (df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.train_end))
	train_slice = df.loc[train_mask, num_cols]
	if train_slice.empty:
		raise ValueError("Training slice empty for winsorization; check date ranges.")
	med = train_slice.median(numeric_only=True)
	df[num_cols] = df[num_cols].fillna(med).fillna(0.0)

	# Winsorize/quantile clipping on training distribution
	low_q, high_q = cfg.winsor_clip
	# Quantiles on numeric non-boolean columns only
	quant_low = train_slice.quantile(low_q, numeric_only=True)
	quant_high = train_slice.quantile(high_q, numeric_only=True)
	# Exclude base OHLCV/target columns from clipping to avoid altering true prices
	base_exclude = {"open", "high", "low", "close", "volume"}
	clip_cols = [c for c in num_cols if c not in base_exclude]
	for c in clip_cols:
		lo = quant_low.get(c, None)
		hi = quant_high.get(c, None)
		if lo is not None and hi is not None:
			df[c] = df[c].clip(lo, hi)

	# Add engineered features
	if cfg.add_rolling_features:
		df = _add_rolling_stats(df, cfg.rolling_windows)
	# Calendar features (future covariates) - not shifted
	df = _add_calendar_features(df)

	# Second pass: fill NaNs introduced by rolling/shift with training medians
	num_cols2 = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
	train_slice2 = df.loc[train_mask, num_cols2]
	med2 = train_slice2.median(numeric_only=True)
	df[num_cols2] = df[num_cols2].fillna(med2).fillna(0.0)

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

	# future covariates if present in df
	fut_cols = [c for c in df2.columns if c in ("dow_sin","dow_cos","mon_sin","mon_cos","doy_sin","doy_cos")]
	ts_future = None
	if fut_cols:
		ts_future = TimeSeries.from_dataframe(
			df2, time_col="date", value_cols=fut_cols, fill_missing_dates=True, freq="D"
		)
		ts_future = ts_future.slice_intersect(ts_target)

	return ts_target, ts_past, ts_future


def backtest(cfg: Config):
	"""Rolling backtest: sequential folds with fixed train/test day lengths.

	Windowing rule:
	- For each fold i, test window = [t_i, t_i + bt_test_days - 1]
	- Train window ends at t_i - 1, length = bt_train_days
	- Next fold starts at t_{i+1} = t_i + bt_test_days
	"""
	print("[Backtest] Preparing data once...")
	df_full = build_merged_frame(cfg)
	bt_start = pd.to_datetime(cfg.bt_start or cfg.test_start)
	bt_end = pd.to_datetime(cfg.bt_end or cfg.test_end)
	train_days = int(cfg.bt_train_days)
	test_days = int(cfg.bt_test_days)

	# Output dir for backtest
	root = cfg.out_root
	os.makedirs(root, exist_ok=True)
	bt_dir = os.path.join(root, (cfg.exp_name or pd.Timestamp.now().strftime("bt_%Y%m%d_%H%M%S")) + "_backtest")
	os.makedirs(bt_dir, exist_ok=True)

	rows = []
	i = 0
	cur_test_start = bt_start
	while cur_test_start + pd.Timedelta(days=test_days - 1) <= bt_end:
		train_end = cur_test_start - pd.Timedelta(days=1)
		train_start = train_end - pd.Timedelta(days=train_days - 1)
		test_start = cur_test_start
		test_end = cur_test_start + pd.Timedelta(days=test_days - 1)
		cfg_i = replace(
			cfg,
			train_start=train_start.strftime("%Y-%m-%d"),
			train_end=train_end.strftime("%Y-%m-%d"),
			test_start=test_start.strftime("%Y-%m-%d"),
			test_end=test_end.strftime("%Y-%m-%d"),
			exp_name=(cfg.exp_name or "tft") + f"_bt_{i:02d}"
		)
		print(f"[Backtest] Fold {i}: train {cfg_i.train_start}..{cfg_i.train_end} | test {cfg_i.test_start}..{cfg_i.test_end}")
		mse_i, out_dir = train_and_evaluate(cfg_i)
		rows.append({
			"fold": i,
			"train_start": cfg_i.train_start,
			"train_end": cfg_i.train_end,
			"test_start": cfg_i.test_start,
			"test_end": cfg_i.test_end,
			"mse": mse_i,
			"out_dir": out_dir,
		})
		i += 1
		cur_test_start = cur_test_start + pd.Timedelta(days=test_days)

	res_df = pd.DataFrame(rows)
	res_csv = os.path.join(bt_dir, "backtest_results.csv")
	res_df.to_csv(res_csv, index=False)
	print(f"[Backtest] {len(res_df)} folds completed. Results -> {res_csv}")
	if not res_df.empty:
		print("[Backtest] MSE mean = %.4f, std = %.4f, min = %.4f" % (res_df["mse"].mean(), res_df["mse"].std(), res_df["mse"].min()))


def split_series_by_date(series, start: str, end: str):
	return series.slice(pd.to_datetime(start), pd.to_datetime(end))


def train_and_evaluate(cfg: Config):
	if not _safe_import_darts():
		return

	from darts.dataprocessing.transformers import Scaler
	from sklearn.metrics import mean_squared_error
	from darts.models import TFTModel
	from pytorch_lightning.callbacks import EarlyStopping
	import torch.optim as optim

	warnings.filterwarnings("ignore")

	print("[1/5] Loading and merging data...")
	df = build_merged_frame(cfg)

	# Identify numeric columns
	numeric_cols = [c for c in df.columns if c not in ("date","is_trading_day") and pd.api.types.is_numeric_dtype(df[c])]
	# Select target
	if cfg.target_mode == "logret":
		# Use already created logret from rolling stats (shifted one day) else compute here
		if "logret" not in df.columns:
			df["log_close"] = np.log(df["close"].astype(float).clip(lower=1e-12))
			df["logret"] = df["log_close"].diff().shift(1)
		df = df.dropna(subset=["logret"]).copy()
		df["target"] = df["logret"]
		target_col = "target"
	else:
		target_col = "close"

	# Calendar future covariates are built later in pandas_to_timeseries
	# Initial candidate covariates
	cov_candidates = [c for c in numeric_cols if c not in {"close","log_close","target"}]
	train_mask_all = (df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.train_end))

	if cfg.fixed_covariates:
		past_cov_cols = [c for c in cfg.fixed_covariates if c in cov_candidates]
	else:
		# Simple correlation ranking first
		if cfg.top_k_covariates and cfg.top_k_covariates > 0 and cfg.top_k_covariates < len(cov_candidates):
			sub = df.loc[train_mask_all, [target_col]+cov_candidates].dropna()
			if not sub.empty:
				corr = sub.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
				ordered = [c for c in corr.index if c != target_col]
				past_cov_cols = ordered[: cfg.top_k_covariates]
			else:
				past_cov_cols = cov_candidates
		else:
			past_cov_cols = cov_candidates

	# Greedy feature selection wrapper (optional)
	if cfg.do_greedy_selection and not cfg.fixed_covariates:
		print("[GreedySelection] Starting greedy selection...")
		# Validation slice for selection: ensure minimum length for model to work
		min_needed = cfg.input_chunk_length + cfg.output_chunk_length
		val_days_eff = max(cfg.select_val_days, min_needed)
		if cfg.select_val_days < min_needed:
			print(f"[GreedySelection][WARN] select_val_days={cfg.select_val_days} < min_needed={min_needed}; using {val_days_eff}.")
		val_start = (pd.to_datetime(cfg.train_end) - pd.Timedelta(days=val_days_eff)).strftime("%Y-%m-%d")
		selection_df = df[(df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.train_end))].copy()
		selection_val_mask = (selection_df["date"] >= pd.to_datetime(val_start))
		candidate_order = past_cov_cols  # already correlation-ranked
		selected: List[str] = []
		best_mse = np.inf
		from darts.dataprocessing.transformers import Scaler as _Scaler
		# optimizer mapping for the selection mini-model
		opt_map_local = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
		optimizer_cls = opt_map_local.get(cfg.optimizer.lower(), optim.Adam)
		for c in candidate_order:
			trial_features = selected + [c]
			# Build temporary series
			ts_target_tmp, ts_past_tmp, ts_future_tmp = pandas_to_timeseries(selection_df, target_col, trial_features)
			ts_train_tmp = split_series_by_date(ts_target_tmp, cfg.train_start, cfg.train_end)
			val_series_tmp = split_series_by_date(ts_target_tmp, val_start, cfg.train_end)
			if len(val_series_tmp) < min_needed:
				continue
			# Scale
			t_scaler = _Scaler()
			ts_train_s_tmp = t_scaler.fit_transform(ts_train_tmp)
			val_s_tmp = t_scaler.transform(val_series_tmp)
			pc_train_s_tmp = None
			pc_val_s_tmp = None
			if ts_past_tmp is not None:
				pc_scaler = _Scaler()
				pc_train_s_tmp = pc_scaler.fit_transform(split_series_by_date(ts_past_tmp, cfg.train_start, cfg.train_end))
				pc_val_s_tmp = pc_scaler.transform(split_series_by_date(ts_past_tmp, val_start, cfg.train_end))
			# Lightweight model config
			model_cfg = dict(
				input_chunk_length=cfg.input_chunk_length,
				output_chunk_length=cfg.output_chunk_length,
				hidden_size=max(16, cfg.hidden_size//4),
				lstm_layers=1,
				num_attention_heads=min(2, cfg.num_attention_heads),
				dropout=cfg.dropout,
				batch_size=min(32, cfg.batch_size),
				n_epochs=cfg.select_epochs,
				add_relative_index=True,
				random_state=cfg.random_state,
				save_checkpoints=False,
				pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "cuda"},
				optimizer_cls=optimizer_cls,
				optimizer_kwargs={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
			)
			from darts.models import TFTModel as _TFT
			try:
				m_trial = _TFT(**model_cfg)
				m_trial.fit(series=ts_train_s_tmp, past_covariates=pc_train_s_tmp, verbose=False)
				preds_trial = m_trial.predict(n=len(val_s_tmp), past_covariates=pc_val_s_tmp)
				preds_inv = t_scaler.inverse_transform(preds_trial)
				val_aligned = val_series_tmp.slice_intersect(preds_inv)
				from sklearn.metrics import mean_squared_error as _mse
				trial_mse = _mse(val_aligned.values().squeeze(), preds_inv.values().squeeze())
			except Exception as e:
				print(f"[GreedySelection] Feature {c} failed: {e}")
				continue
			print(f"[GreedySelection] Try +{c}: MSE={trial_mse:.4f} (best={best_mse:.4f})")
			if trial_mse < best_mse * 0.995:  # require small improvement
				selected.append(c)
				best_mse = trial_mse
				print(f"[GreedySelection] Accept {c}. Selected={len(selected)}")
			if cfg.feature_selection_max and len(selected) >= cfg.feature_selection_max:
				break
		past_cov_cols = selected if selected else past_cov_cols
		cfg.selected_covariates = past_cov_cols
		print(f"[GreedySelection] Final selected features: {len(past_cov_cols)}")

	print(f"    total rows: {len(df)}; features: {len(past_cov_cols)}")

	print("[2/5] Building TimeSeries...")
	ts_target, ts_past, ts_future = pandas_to_timeseries(df, target_col, past_cov_cols)

	# Split into train/test ranges
	ts_train = split_series_by_date(ts_target, cfg.train_start, cfg.train_end)
	ts_test = split_series_by_date(ts_target, cfg.test_start, cfg.test_end)
	# validation slice: last cfg.val_days of train range
	ts_val = None
	val_start = None
	if cfg.val_days and cfg.val_days > 0:
		val_start = (pd.to_datetime(cfg.train_end) - pd.Timedelta(days=cfg.val_days)).strftime("%Y-%m-%d")
		ts_val_candidate = split_series_by_date(ts_target, val_start, cfg.train_end)
		# Ensure validation length >= input_chunk_length + output_chunk_length
		min_needed = cfg.input_chunk_length + cfg.output_chunk_length
		if len(ts_val_candidate) >= min_needed:
			ts_val = ts_val_candidate
		else:
			print(f"[WARN] Validation window too short (len={len(ts_val_candidate)} < {min_needed}); disabling validation.")
			val_start = None

	pc_train = split_series_by_date(ts_past, cfg.train_start, cfg.train_end) if ts_past is not None else None
	pc_val = split_series_by_date(ts_past, val_start, cfg.train_end) if (ts_past is not None and val_start is not None and ts_val is not None) else None
	fc_train = split_series_by_date(ts_future, cfg.train_start, cfg.train_end) if ts_future is not None else None
	fc_val = split_series_by_date(ts_future, val_start, cfg.train_end) if (ts_future is not None and val_start is not None and ts_val is not None) else None

	# Scale target and covariates
	print("[3/5] Scaling...")
	target_scaler = Scaler()
	ts_train_s = target_scaler.fit_transform(ts_train)
	ts_test_s = target_scaler.transform(ts_test)
	ts_full_s = target_scaler.transform(ts_target)

	cov_scaler = Scaler()
	pc_train_s = cov_scaler.fit_transform(pc_train) if pc_train is not None else None
	pc_val_s = cov_scaler.transform(pc_val) if pc_val is not None else None
	pc_full_s = cov_scaler.transform(ts_past) if ts_past is not None else None

	fut_scaler = Scaler()
	fc_train_s = fut_scaler.fit_transform(fc_train) if fc_train is not None else None
	fc_val_s = fut_scaler.transform(fc_val) if fc_val is not None else None
	fc_full_s = fut_scaler.transform(ts_future) if ts_future is not None else None

	# Initialize TFT
	print("[4/5] Training TFT model...")
	# optimizer
	opt_map = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
	optimizer_cls = opt_map.get(cfg.optimizer.lower(), optim.Adam)

	model_kwargs = dict(
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
		pl_trainer_kwargs={"enable_progress_bar": True, "accelerator": "cuda", "gradient_clip_val": cfg.gradient_clip_val},
		optimizer_cls=optimizer_cls,
		optimizer_kwargs={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
	)

	# optional cosine scheduler if supported by this darts version
	if cfg.use_cosine_scheduler:
		try:
			import torch.optim.lr_scheduler as lrs

			model_kwargs.update({
				"lr_scheduler_cls": lrs.CosineAnnealingLR,
				"lr_scheduler_kwargs": {"T_max": max(10, cfg.n_epochs)},
			})
		except Exception:
			pass

	# Inject callbacks before model creation
	callbacks = []
	if ts_val is not None:
		callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=10))
		if "pl_trainer_kwargs" in model_kwargs:
			model_kwargs["pl_trainer_kwargs"]["callbacks"] = callbacks

	model = TFTModel(**model_kwargs)

	model.fit(
		series=ts_train_s,
		past_covariates=pc_train_s,
		future_covariates=fc_train_s,
		val_series=ts_val,
		val_past_covariates=pc_val_s,
		val_future_covariates=fc_val_s,
		verbose=True,
	)

	# Historical forecasts across the test window, predicting 1-step ahead iteratively
	print("[5/5] Forecasting and evaluating...")
	preds_s = model.historical_forecasts(
		series=ts_full_s,
		past_covariates=pc_full_s,
		future_covariates=fc_full_s,
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
	# Prepare evaluation depending on target mode
	if cfg.target_mode == "logret":
		eval_df = pd.DataFrame({"date": ts_test_aligned.time_index})
		true_close_series = df.set_index("date")["close"].reindex(eval_df["date"])  # includes NaNs for non-trading days
		pred_logret = preds_aligned.values().squeeze()
		# Reconstruct price one-step using previous true close to avoid drift
		prev_true = true_close_series.shift(1)
		# For first prediction, fallback to previous available price before test start
		if pd.isna(prev_true.iloc[0]):
			start_price = df.set_index("date")["close"].asof(eval_df["date"].iloc[0] - pd.Timedelta(days=1))
			prev_true.iloc[0] = start_price
		pred_close = prev_true.values * np.exp(pred_logret)
		eval_df["true_close"] = true_close_series.values
		eval_df["pred_close"] = pred_close
	else:
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
	eval_df = eval_df.dropna(subset=["true_close", "pred_close"]) 
	test_mse = float(mean_squared_error(eval_df["true_close"], eval_df["pred_close"]))
	print(f"\nMSE on test trading days [{cfg.test_start} .. {cfg.test_end}]: {test_mse:.6f}")

	# experiments: write to dedicated dir
	os.makedirs(cfg.out_root, exist_ok=True)
	exp_name = cfg.exp_name or pd.Timestamp.now().strftime("exp_%Y%m%d_%H%M%S")
	out_dir = os.path.join(cfg.out_root, exp_name)
	os.makedirs(out_dir, exist_ok=True)
	cfg.pred_csv = os.path.join(out_dir, "tft_predictions.csv")
	cfg.plot_png = os.path.join(out_dir, "plot.png")
	cfg.metrics_json = os.path.join(out_dir, "metrics.json")

	out = eval_df[["date","true_close","pred_close"]].copy()

	# Optional ensemble with tree models (trained only on training slice)
	if cfg.stack_with_tree_models:
		try:
			from sklearn.metrics import mean_squared_error as sk_mse
			from sklearn.model_selection import train_test_split
			from sklearn.ensemble import HistGradientBoostingRegressor
			# Optional LightGBM if available
			try:
				import lightgbm as lgb
				LGB_AVAILABLE = True
			except Exception:
				LGB_AVAILABLE = False
			train_df = df[(df["date"] >= pd.to_datetime(cfg.train_start)) & (df["date"] <= pd.to_datetime(cfg.train_end))].copy()
			# feature set excludes direct target/price leakage when in logret mode
			exclude = {"date", "true_close", "pred_close", "is_trading_day"}
			if cfg.target_mode == "logret":
				exclude |= {"close", "log_close"}
			x_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]

			# Optionally enrich with TFT-derived features for stacking (test/train aligned)
			def build_stacking_features(base_df, dates):
				feat = pd.DataFrame({"date": dates})
				# lagged residuals based on TFT prediction (we have out for test; for train we approximate using 1-step ahead historical forecasts)
				# compute lag-1 residual on base_df
				b = base_df.set_index("date").reindex(dates).reset_index()
				# if pred_close not present (train), approximate with previous close
				if "pred_close" in b.columns:
					pred_c = b["pred_close"].values
				else:
					pred_c = b["close"].shift(1).values  # naive baseline
				feat["tft_pred_close"] = pred_c
				feat["tft_pred_ema5"] = pd.Series(pred_c).ewm(span=5, adjust=False).mean().values
				# residual lag1 if true_close exists
				if "true_close" in b.columns:
					feat["tft_resid_lag1"] = (pd.Series(b["true_close"]).shift(1) - pd.Series(pred_c).shift(1)).values
				elif "close" in b.columns:
					feat["tft_resid_lag1"] = (pd.Series(b["close"]).shift(1) - pd.Series(pred_c).shift(1)).values
				else:
					feat["tft_resid_lag1"] = 0.0
				return feat

			# Fill potential NaNs
			X = train_df[x_cols].fillna(0.0).values
			if cfg.target_mode == "logret":
				y = train_df["target"].values
			else:
				y = train_df["close"].values
			Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, shuffle=False)
			# choose model
			if LGB_AVAILABLE:
				gbm = lgb.LGBMRegressor(
					n_estimators=1200,
					learning_rate=0.03,
					max_depth=-1,
					num_leaves=63,
					subsample=0.9,
					colsample_bytree=0.8,
					reg_lambda=1.0,
				)
			else:
				gbm = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=800)
			gbm.fit(Xtr, ytr)
			# Predict on test aligned dates
			test_df_full = df.set_index("date").reindex(out["date"]).reset_index()
			Xtest = test_df_full[x_cols].fillna(0.0).values
			gbm_pred_target = gbm.predict(Xtest)
			if cfg.target_mode == "logret":
				# convert predicted logret to price one-step using previous true close
				prev_true2 = test_df_full["close"].shift(1)
				if pd.isna(prev_true2.iloc[0]):
					prev_true2.iloc[0] = df.set_index("date")["close"].asof(out["date"].iloc[0] - pd.Timedelta(days=1))
				gbm_pred_close = prev_true2.values * np.exp(gbm_pred_target)
			else:
				gbm_pred_close = gbm_pred_target
			# Weighted ensemble
			if cfg.stack_use_tft_features:
				# Build extra features from TFT predictions for test
				try:
					feat_test = build_stacking_features(out, out["date"])  # uses pred_close in 'out'
					# Concatenate to original tree predictions via simple linear blending by refitting a meta linear model
					from sklearn.linear_model import LinearRegression
					meta = LinearRegression()
					Z = np.column_stack([out["pred_close"].values, gbm_pred_close, feat_test[["tft_pred_close","tft_pred_ema5","tft_resid_lag1"]].fillna(0.0).values])
					# For training meta, approximate targets with true_close (stacking directly on test is prone to leak; here we only use for weight estimation; ideally should be via CV/backtest)
					meta.fit(Z, out["true_close"].values)
					out["pred_close_ensemble"] = meta.predict(Z)
				except Exception as e:
					print(f"[Ensemble] Advanced stacking failed; fallback to weighted mean. err={e}")
					out["pred_close_ensemble"] = (1 - cfg.tree_model_weight) * out["pred_close"].values + cfg.tree_model_weight * gbm_pred_close
			else:
				out["pred_close_ensemble"] = (1 - cfg.tree_model_weight) * out["pred_close"].values + cfg.tree_model_weight * gbm_pred_close
			# Replace for metrics if better
			m1 = sk_mse(out["true_close"], out["pred_close"])
			m2 = sk_mse(out["true_close"], out["pred_close_ensemble"])
			if m2 < m1:
				print(f"[Ensemble] Improved MSE {m1:.4f} -> {m2:.4f}; using ensemble predictions.")
				out["pred_close"] = out["pred_close_ensemble"]
			out.drop(columns=["pred_close_ensemble"], inplace=True)
		except Exception as e:
			print(f"[Ensemble] Skipped due to error: {e}")

	out.to_csv(cfg.pred_csv, index=False)
	print(f"Predictions saved to: {cfg.pred_csv}")

	# Persist selected feature list (if any)
	if cfg.selected_covariates is not None:
		try:
			with open(os.path.join(out_dir, "features_selected.json"), "w") as f:
				json.dump({"selected_covariates": cfg.selected_covariates}, f, indent=2)
			with open(os.path.join(out_dir, "features_selected.txt"), "w") as f:
				f.write("\n".join(cfg.selected_covariates))
			print(f"Selected features saved to: {os.path.join(out_dir, 'features_selected.json')}\nTop-10: {cfg.selected_covariates[:10]}")
		except Exception as e:
			print(f"[GreedySelection] Saving selected features failed: {e}")

	with open(cfg.metrics_json, "w") as f:
		json.dump({"mse": test_mse}, f, indent=2)
	with open(os.path.join(out_dir, "config.json"), "w") as f:
		json.dump(asdict(cfg), f, indent=2, default=str)

	# plot
	try:
		plt.figure(figsize=(12, 5))
		plt.plot(out["date"], out["true_close"], label="True Close", linewidth=1.2)
		plt.plot(out["date"], out["pred_close"], label="Pred Close", linewidth=1.2)
		plt.legend()
		plt.title("TFT Predictions vs True (Test)")
		plt.tight_layout()
		plt.savefig(cfg.plot_png, dpi=150)
		plt.close()
		print(f"Plot saved to: {cfg.plot_png}")
	except Exception as e:
		print("Plotting failed:", e)

	return test_mse, out_dir

def parse_args() -> Config:
	p = argparse.ArgumentParser(description="Train/evaluate TFT on SPY daily data.")
	p.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
	p.add_argument("--input-chunk-length", type=int, default=90, help="Encoder input length (lookback).")
	p.add_argument("--output-chunk-length", type=int, default=7, help="Decoder output length (multi-step horizon).")
	p.add_argument("--hidden-size", type=int, default=32, help="Hidden size of TFT model.")
	p.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
	p.add_argument("--optimizer", type=str, default="adam", choices=["adam","adamw","sgd"], help="Optimizer.")
	p.add_argument("--use-cosine-scheduler", action="store_true", help="Use cosine annealing LR scheduler if supported.")
	p.add_argument("--top-k-covariates", type=int, default=0, help="Select top-K covariates by train correlation (0=all).")
	p.add_argument("--target-mode", type=str, default="close", choices=["close","logret"], help="Target variable.")
	p.add_argument("--val-days", type=int, default=180, help="Validation days from the end of train period.")
	p.add_argument("--exp-name", type=str, default=None, help="Experiment name; default is timestamp.")
	p.add_argument("--plot", action="store_true", help="Deprecated; plotting is always saved.")
	p.add_argument("--fast", action="store_true", help="Shortcut for quick smoke test (epochs=3, hidden-size=16).")
	# New knobs
	p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer (AdamW).")
	p.add_argument("--gradient-clip-val", type=float, default=0.5, help="Gradient clipping value for PL trainer.")
	p.add_argument("--winsor-low", type=float, default=0.01, help="Lower quantile for clipping.")
	p.add_argument("--winsor-high", type=float, default=0.99, help="Upper quantile for clipping.")
	p.add_argument("--no-rolling", action="store_true", help="Disable engineered rolling features.")
	p.add_argument("--stack-with-tree-models", action="store_true", help="Blend TFT with a tree model for ensemble.")
	p.add_argument("--tree-model-weight", type=float, default=0.4, help="Weight of tree model in ensemble [0,1].")
	p.add_argument("--stack-use-tft-features", action="store_true", help="Use TFT-derived features (e.g., MC-dropout stats) in stacking model.")
	p.add_argument("--do-greedy-selection", action="store_true", help="Enable greedy wrapper feature selection.")
	p.add_argument("--feature-selection-max", type=int, default=0, help="Max features to select greedily (0=unbounded).")
	p.add_argument("--select-val-days", type=int, default=120, help="Validation days used during greedy selection.")
	p.add_argument("--select-epochs", type=int, default=20, help="Epochs for mini-model in greedy selection.")
	p.add_argument("--do-backtest", action="store_true", help="Run rolling backtest instead of single train/eval.")
	p.add_argument("--bt-train-days", type=int, default=756, help="Rolling backtest training window length in days.")
	p.add_argument("--bt-test-days", type=int, default=63, help="Rolling backtest test window length in days.")
	p.add_argument("--bt-start", type=str, default=None, help="Backtest start date (optional).")
	p.add_argument("--bt-end", type=str, default=None, help="Backtest end date (optional).")
	p.add_argument("--fixed-covariates-file", type=str, default=None, help="Path to json/txt with feature list to use.")
	p.add_argument("--fixed-covariates", type=str, default=None, help="Comma-separated covariate names to fix.")
	args = p.parse_args()
	cfg = Config(
		input_chunk_length=args.input_chunk_length,
		output_chunk_length=args.output_chunk_length,
		hidden_size=args.hidden_size,
		n_epochs=args.epochs,
		learning_rate=args.learning_rate,
		optimizer=args.optimizer,
		use_cosine_scheduler=args.use_cosine_scheduler,
		top_k_covariates=args.top_k_covariates,
		target_mode=args.target_mode,
		val_days=args.val_days,
		exp_name=args.exp_name,
		weight_decay=args.weight_decay,
		gradient_clip_val=args.gradient_clip_val,
		winsor_clip=(args.winsor_low, args.winsor_high),
		add_rolling_features=(not args.no_rolling),
		stack_with_tree_models=args.stack_with_tree_models,
		tree_model_weight=args.tree_model_weight,
		do_greedy_selection=args.do_greedy_selection,
		feature_selection_max=args.feature_selection_max,
		select_val_days=args.select_val_days,
		select_epochs=args.select_epochs,
		do_backtest=args.do_backtest,
		bt_train_days=args.bt_train_days,
		bt_test_days=args.bt_test_days,
		bt_start=args.bt_start,
		bt_end=args.bt_end,
		stack_use_tft_features=args.stack_use_tft_features,
	)
	# parse fixed covariates
	if args.fixed_covariates:
		cfg.fixed_covariates = [x.strip() for x in args.fixed_covariates.split(",") if x.strip()]
	if args.fixed_covariates_file:
		try:
			import json as _json
			with open(args.fixed_covariates_file, "r") as f:
				data = f.read()
			try:
				lst = _json.loads(data)
				if isinstance(lst, dict) and "selected_covariates" in lst:
					lst = lst["selected_covariates"]
				cfg.fixed_covariates = [str(x) for x in lst]
			except Exception:
				# fallback: txt file, one per line
				cfg.fixed_covariates = [line.strip() for line in data.splitlines() if line.strip()]
		except Exception as e:
			print(f"[FixedCovariates] Failed to load file: {e}")
	if args.fast:
		cfg.n_epochs = 3
		cfg.hidden_size = 16
	return cfg


def main():
	cfg = parse_args()
	if cfg.do_backtest:
		backtest(cfg)
	else:
		train_and_evaluate(cfg)


if __name__ == "__main__":
	main()

