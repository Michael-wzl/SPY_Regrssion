import os
import warnings
import argparse
import json
from datetime import datetime
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.optim as optim
from darts import TimeSeries, concatenate
from darts.models import TFTModel, LightGBMModel, XGBModel
from pytorch_lightning.callbacks import EarlyStopping
import yaml

from preprocess import Preprocessor, to_logret, from_logret
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def create_timeseries_with_int_index(df_features: pd.DataFrame, df_dates: pd.DataFrame) -> TimeSeries:
        """
        Create TimeSeries using integer index to avoid frequency inference issues.

        Args:
            df_features: DataFrame of features.
            df_dates: DataFrame with a single column of dates corresponding to df_features.

        Returns:
            Darts TimeSeries object with integer index.
        """
        df = df_features.reset_index(drop=True)
        # Check for NaNs
        if df.isnull().any().any():
            print(f"Warning: Found {df.isnull().sum().sum()} NaN values in features.")
            # Use ffill first (safe for time series)
            df = df.ffill()
            # For remaining NaNs (at the beginning), fill with 0 (assuming z-score scaled data where mean=0)
            # bfill() would leak future data if used on test set
            if df.isnull().any().any():
                raise ValueError("NaN values remain in features after forward fill. Please check the data.")
        return TimeSeries.from_dataframe(df)

def plot_predictions(
    train_dates: pd.Series,
    test_dates: pd.Series,
    train_preds: np.ndarray,
    test_preds: np.ndarray,
    train_gt: np.ndarray,
    test_gt: np.ndarray,
    save_path: str,
    title: str = "Model Predictions vs Ground Truth"
) -> None:
    """
    Plot model predictions and ground truth for the entire dataset.
    
    Args:
        train_dates: Date series for training data
        test_dates: Date series for test data
        train_preds: Model predictions on training data (in price space)
        test_preds: Model predictions on test data (in price space)
        train_gt: Ground truth for training data (in price space)
        test_gt: Ground truth for test data (in price space)
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(16, 8))
    
    # Combine dates and values for plotting
    all_dates = pd.concat([train_dates, test_dates]).reset_index(drop=True)
    all_gt = np.concatenate([train_gt, test_gt])
    all_preds = np.concatenate([train_preds, test_preds])
    
    # Plot ground truth
    plt.plot(all_dates, all_gt, label='Ground Truth', color='blue', alpha=0.7, linewidth=1.5)
    
    # Plot predictions
    plt.plot(all_dates, all_preds, label='Predictions', color='red', alpha=0.7, linewidth=1.5)
    
    # Add vertical line for train/test split
    split_date = test_dates.iloc[0]
    plt.axvline(x=split_date, color='green', linestyle='--', linewidth=2, label='Train/Test Split')
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def run(cfgs: OmegaConf) -> None:
    """
    Run the TFT time series forecasting experiment.

    Args:
        cfgs: Configuration dictionary.
    """
    # Setup paths
    src_data = f"{WORKING_DIR}/data/{cfgs.src_data}"
    res_save_path = f"{WORKING_DIR}/results/{cfgs.exp_name}"
    if os.path.exists(res_save_path):
        new_exp_name = cfgs.exp_name + datetime.now().strftime("_%Y%m%d_%H%M%S")
        res_save_path = f"{WORKING_DIR}/results/{new_exp_name}"
    os.makedirs(res_save_path, exist_ok=False)
    print(f"Results will be saved to {res_save_path}")
    # Save cfgs
    with open(f"{res_save_path}/cfgs.yaml", 'w') as f:
        yaml.dump(OmegaConf.to_container(cfgs, resolve=True), f)

    date_col = cfgs.date_col if 'date_col' in cfgs else 'date'
    # Ensure target column is lowercased to match raw_data columns
    if 'target_col' in cfgs:
        cfgs.target_col = cfgs.target_col.lower()
    
    # Load data
    raw_data = pd.read_csv(src_data)
    raw_data.columns = [col.lower() for col in raw_data.columns]
    if date_col not in raw_data.columns:
        raise ValueError(f"Input CSV must contain a '{date_col}' column (case-insensitive).")
    raw_data[date_col] = pd.to_datetime(raw_data[date_col])
    print(f"Loaded raw data from {src_data} with {len(raw_data)} rows.")

    # Convert target to logret BEFORE splitting (to avoid losing first test value)
    if cfgs.model_cfgs.output == 'logret':
        raw_data[cfgs.target_col] = to_logret(raw_data[cfgs.target_col])
        print("Converted target column to log returns before splitting.")

    # Split data
    print(f"Training data from {cfgs.train_start} to {cfgs.train_end}")
    print(f"Testing data from {cfgs.test_start} to {cfgs.test_end}")
    train_mask = (raw_data[date_col] >= cfgs.train_start) & (raw_data[date_col] <= cfgs.train_end)
    test_mask = (raw_data[date_col] >= cfgs.test_start) & (raw_data[date_col] <= cfgs.test_end)
    train_data = raw_data.loc[train_mask].copy()
    test_data = raw_data.loc[test_mask].copy()

    # Preprocessing
    preprocessor = Preprocessor(
        cfgs=cfgs.preprocess_cfgs, 
        steps=cfgs.pp_steps, 
        fit_dfs=train_data.drop(columns=[date_col, cfgs.target_col]).copy(),
        target=train_data[cfgs.target_col].copy())
    X_train = preprocessor.transform(train_data.drop(columns=[date_col, cfgs.target_col]).copy())
    y_train = train_data[cfgs.target_col].copy()
    X_test = preprocessor.transform(test_data.drop(columns=[date_col, cfgs.target_col]).copy())
    y_test = test_data[cfgs.target_col].copy()
    print(f"Preprocessing completed. Training features shape: {X_train.shape}, Testing features shape: {X_test.shape}")
    print(f"Training target sample after preprocessing:\n{y_train.head()}")

    # Prepare Darts TimeSeries
    
    ts_X_train = create_timeseries_with_int_index(X_train, train_data[[date_col]])
    ts_y_train = create_timeseries_with_int_index(y_train.to_frame(), train_data[[date_col]])
    ts_X_test = create_timeseries_with_int_index(X_test, test_data[[date_col]])
    ts_y_test = create_timeseries_with_int_index(y_test.to_frame(), test_data[[date_col]])
    
    # Save date indices for future reference
    train_dates = train_data[date_col].reset_index(drop=True)
    test_dates = test_data[date_col].reset_index(drop=True)
    
    print("Converted data to Darts TimeSeries format (using integer index for trading days).")

    # Load Training Configurations
    opt_map = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
    optimizer = opt_map[cfgs.model_cfgs.optimizer.lower()]
    model_cfgs = dict(
        input_chunk_length=cfgs.model_cfgs.input_chunk,
        output_chunk_length=cfgs.model_cfgs.output_chunk,
        hidden_size=cfgs.model_cfgs.hidden_size,
        lstm_layers=cfgs.model_cfgs.lstm_layers,
        num_attention_heads=cfgs.model_cfgs.num_attention_heads,
        dropout=cfgs.model_cfgs.dropout,
        batch_size=cfgs.model_cfgs.batch_size,
        n_epochs=cfgs.model_cfgs.n_epochs,
        add_relative_index=True,  # Required for TFT to generate future covariates
        random_state=cfgs.model_cfgs.random_state,
        save_checkpoints=False,
        pl_trainer_kwargs={
            "enable_progress_bar": True,
            "accelerator": ('cuda' if ('cuda' in str(cfgs.device)) else 'cpu'),
            "devices": 1,  # Use single GPU to avoid distributed issues
            "gradient_clip_val": (cfgs.model_cfgs.gradient_clip_val)
        },
        optimizer_cls=optimizer,
        optimizer_kwargs={"lr": cfgs.model_cfgs.learning_rate, "weight_decay": cfgs.model_cfgs.weight_decay},
    )
    print(f"Model configurations: {model_cfgs}")

    # Train or Load TFT Model
    if cfgs.use_ckpt:
        # Load model from checkpoint
        ckpt_path = f"{WORKING_DIR}/results/{cfgs.ckpt_name}/tft_model.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        print(f"Loading model from checkpoint: {ckpt_path}")
        # Note: For PyTorch 2.6+, Darts load doesn't support weights_only parameter
        # We temporarily patch torch.load to use weights_only=False
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})
        # Use 'cuda' instead of 'cuda:N' because CUDA_VISIBLE_DEVICES remaps device indices
        map_loc = 'cuda' if 'cuda' in cfgs.device else 'cpu'
        try:
            model = TFTModel.load(ckpt_path, map_location=map_loc)
        finally:
            torch.load = _original_torch_load
        print("Model loaded successfully. Skipping training.")
    else:
        # Train new model
        '''
        early_stopper = EarlyStopping(
            monitor="train_loss",
            patience=10,
            min_delta=1e-5,
            mode="min",
        )
        # Update pl_trainer_kwargs with callbacks
        model_cfgs["pl_trainer_kwargs"]["callbacks"] = [early_stopper]
        '''
        
        model = TFTModel(
            **model_cfgs,
            model_name=cfgs.exp_name,
            work_dir=res_save_path,
            force_reset=True,
        )
        
        print("Starting model training...")
        model.fit(
            series=ts_y_train,
            past_covariates=ts_X_train,
            verbose=True,
        )
        print("Model training completed.")

        # Save model weights
        model_save_path = f"{res_save_path}/tft_model.pt"
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    # ==================== Evaluation ====================
    print("\n" + "="*50)
    print("Starting model evaluation...")
    print("="*50)
    
    # Use the trained model directly for inference (avoiding reload issues with PyTorch 2.6+)
    loaded_model = model
    print(f"Using trained model for evaluation")
    
    # Get input_chunk_length for historical predictions
    input_chunk = cfgs.model_cfgs.input_chunk
    output_chunk = cfgs.model_cfgs.output_chunk
    
    # Combine train and test for continuous prediction using concatenate
    ts_y_full = concatenate([ts_y_train, ts_y_test], ignore_time_axis=True)
    ts_X_full = concatenate([ts_X_train, ts_X_test], ignore_time_axis=True)
    full_dates = pd.concat([train_dates, test_dates]).reset_index(drop=True)
    train_len = len(ts_y_train)
    
    # --- Use historical_forecasts for efficient batch prediction ---
    print("Generating historical forecasts")
    
    historical_preds = loaded_model.historical_forecasts(
        series=ts_y_full,
        past_covariates=ts_X_full,
        start=input_chunk,  # Start predicting from index input_chunk
        forecast_horizon=1,  # Predict 1 step ahead
        stride=1,  # Move 1 step each time
        retrain=False,  # Don't retrain the model
        verbose=True,
        show_warnings=False,
    )
    
    # Extract predictions as numpy array
    all_preds_logret = historical_preds.values().flatten()
    
    # Get corresponding ground truth and dates
    # historical_forecasts starts predicting at index `start`, so GT is from start to end
    all_gt_logret = ts_y_full.values()[input_chunk:, 0]
    all_pred_dates = full_dates.iloc[input_chunk:].reset_index(drop=True)
    
    # Split into train and test based on original train_len
    # Training predictions: from input_chunk to train_len
    # Test predictions: from train_len to end
    train_pred_count = train_len - input_chunk
    
    train_preds_logret = all_preds_logret[:train_pred_count]
    train_gt_logret = all_gt_logret[:train_pred_count]
    train_pred_dates = pd.Series(all_pred_dates.iloc[:train_pred_count].values)
    
    test_preds_logret = all_preds_logret[train_pred_count:]
    test_gt_logret = all_gt_logret[train_pred_count:]
    test_pred_dates = pd.Series(all_pred_dates.iloc[train_pred_count:].values)
    
    print(f"Training predictions generated: {len(train_preds_logret)} samples")
    print(f"Test predictions generated: {len(test_preds_logret)} samples")
    
    # --- Convert log returns back to prices ---
    # Load original price data for conversion
    raw_data_orig = pd.read_csv(src_data)
    raw_data_orig.columns = [col.lower() for col in raw_data_orig.columns]
    raw_data_orig[date_col] = pd.to_datetime(raw_data_orig[date_col])
    
    # Get original prices aligned with prediction dates
    # For training predictions
    train_mask_orig = raw_data_orig[date_col].isin(train_pred_dates)
    train_prices_orig = raw_data_orig.loc[train_mask_orig, cfgs.target_col].values
    
    # For test predictions
    test_mask_orig = raw_data_orig[date_col].isin(test_pred_dates)
    test_prices_orig = raw_data_orig.loc[test_mask_orig, cfgs.target_col].values
    
    # Ground truth prices are the original prices
    train_gt_prices = train_prices_orig.copy()
    test_gt_prices = test_prices_orig.copy()
    
    # Get initial prices for conversion (prices at t-1 for each prediction)
    # We need prices one step before each prediction date
    train_date_list = train_pred_dates.tolist()
    test_date_list = test_pred_dates.tolist()
    
    # Build a date-to-position mapping for efficient lookup
    raw_data_orig = raw_data_orig.reset_index(drop=True)  # Ensure continuous integer index
    date_to_pos = {d: i for i, d in enumerate(raw_data_orig[date_col])}
    
    train_init_prices = []
    for date in train_date_list:
        pos = date_to_pos.get(date)
        if pos is None:
            raise ValueError(f"Date {date} not found in original data")
        if pos > 0:
            train_init_prices.append(raw_data_orig.iloc[pos - 1][cfgs.target_col])
        else:
            # First data point: cannot compute log return properly, use same price
            # This should rarely happen as input_chunk skips initial points
            train_init_prices.append(raw_data_orig.iloc[pos][cfgs.target_col])
    train_init_prices = np.array(train_init_prices)
    
    test_init_prices = []
    for date in test_date_list:
        pos = date_to_pos.get(date)
        if pos is None:
            raise ValueError(f"Date {date} not found in original data")
        if pos > 0:
            test_init_prices.append(raw_data_orig.iloc[pos - 1][cfgs.target_col])
        else:
            test_init_prices.append(raw_data_orig.iloc[pos][cfgs.target_col])
    test_init_prices = np.array(test_init_prices)
    
    # Convert predicted log returns to prices using from_logret (non-cumulative mode)
    train_preds_prices = from_logret(train_preds_logret, train_init_prices, cumulative=False)
    test_preds_prices = from_logret(test_preds_logret, test_init_prices, cumulative=False)
    
    # --- Calculate MSE in price space ---
    train_mse = mean_squared_error(train_gt_prices, train_preds_prices)
    test_mse = mean_squared_error(test_gt_prices, test_preds_prices)
    
    print(f"\n{'='*50}")
    print("Evaluation Results (in Close Price space):")
    print(f"{'='*50}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training RMSE: {np.sqrt(train_mse):.6f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.6f}")
    
    # Save metrics
    metrics = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_rmse": float(np.sqrt(train_mse)),
        "test_rmse": float(np.sqrt(test_mse)),
        "train_samples": len(train_preds_prices),
        "test_samples": len(test_preds_prices),
    }
    with open(f"{res_save_path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {res_save_path}/metrics.json")
    
    # Save predictions to CSV
    train_results = pd.DataFrame({
        'date': train_pred_dates,
        'ground_truth': train_gt_prices,
        'prediction': train_preds_prices,
        'split': 'train'
    })
    test_results = pd.DataFrame({
        'date': test_pred_dates,
        'ground_truth': test_gt_prices,
        'prediction': test_preds_prices,
        'split': 'test'
    })
    all_results = pd.concat([train_results, test_results], ignore_index=True)
    all_results.to_csv(f"{res_save_path}/predictions.csv", index=False)
    print(f"Predictions saved to {res_save_path}/predictions.csv")
    
    # --- Plot predictions ---
    plot_predictions(
        train_dates=train_pred_dates,
        test_dates=test_pred_dates,
        train_preds=train_preds_prices,
        test_preds=test_preds_prices,
        train_gt=train_gt_prices,
        test_gt=test_gt_prices,
        save_path=f"{res_save_path}/predictions_plot.png",
        title=f"TFT Model Predictions vs Ground Truth - {cfgs.exp_name}"
    )
    
    print(f"\n{'='*50}")
    print("Experiment completed successfully!")
    print(f"{'='*50}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="TFT Time Series Forecasting")
    args.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cpu', 'cuda:0')")
    args.add_argument("--exp_name", type=str, default="default", help="Experiment name for saving outputs")
    args = args.parse_args()
    cfgs = {
        # Running env
        'device': 'cuda:0', 
        'exp_name': 'default', 
        'global_random_state': 42,
        'use_ckpt': True, # If True, will skip training and load model from checkpoint
        'ckpt_name': 'baseline', # The experiment name to load checkpoint from. 
                                  # The checkpoint itself is uniformly named 'tft_model.pt' and 'tft_model.pt.ckpt'
                                  # The folder should contain the weights and model configs.
        # Data  
        'src_data': 'final_dataset.csv', 
        'train_start': '2015-01-01',
        'train_end': '2021-12-31',
        'test_start': '2022-01-01',
        'test_end': '2025-05-30',
        'date_col': 'date',
        'target_col': 'spy_ohlcv_1drth_close', 
        # Preprocessing
        'preprocess_cfgs': {
            'cov_select': {
                'topk': 150, 
                'method': 'spearman',
            }, 
            'zscore': {},
            'winsor': {
                'lower_q': 0.01,
                'upper_q': 0.99
            },
            'pca': {
                'n_components': 100, 
                'rand_state': 42, # Will be overridden by global_random_state
            },
        },
        'pp_steps': [('cov_select', 'o'), ('zscore', 'o'), ('winsor', 'o'), ('pca', 'o')],
        # Training
        'model_cfgs': {
            # TFT Model parameters
            'output': 'logret',
            'input_chunk': 30, 
            'output_chunk': 1, 
            'hidden_size': 64, 
            'lstm_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.1,
            # TFT Training configs
            'learning_rate': 1e-3,
            'batch_size': 64,
            'n_epochs': 50,
            'random_state': 42,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'eta_min': 1e-6,
            'gradient_clip_val': 0.6,
        }
    }
    cfgs = OmegaConf.create(cfgs)
    # Always override from CLI
    cfgs.device = args.device
    cfgs.exp_name = args.exp_name
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.device.split(':')[-1] if 'cuda' in cfgs.device else ''
    torch.manual_seed(cfgs.global_random_state)
    torch.cuda.manual_seed_all(cfgs.global_random_state)
    np.random.seed(cfgs.global_random_state)
    random.seed(cfgs.global_random_state)
    cfgs.preprocess_cfgs.pca.rand_state = cfgs.global_random_state
    
    # Run training
    run(cfgs)