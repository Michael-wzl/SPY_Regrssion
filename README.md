# SPY Next-Trading-Day Regression with Temporal Fusion Transformer (TFT)

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/Michael-wzl/SPY_Regrssion
   ```

2. Install required packages:

   ```bash
   conda create -n spy_tft python=3.12
   conda activate spy_tft
   pip install -r requirements.txt
   ```

3. Run the prediction experiment:

   ```bash
   python tft.py --exp_name test_baseline --device cuda:0
   ```

## How to Train Your Own Model

1. Run the training script. Remember to set [use_ckpt](tft.py#L423) to False.

   ```bash
   python tft.py --exp_name your_experiment_name --device cuda:0
   ```

2. Check results in the `results/your_experiment_name/` directory.

3. Reuse the trained model for inference or further analysis as needed by setting [use_ckpt](tft.py#L423) to True and specifying the checkpoint path in [ckpt_name](tft.py#L424).

## Pipeline Overview

### 1. Data Loading & Target Transformation

- Load raw dataset from `data/final_dataset.csv`
- If configured to predict log returns (`output: 'logret'`), convert target column (e.g., `spy_ohlcv_1drth_close`) to log returns **before** splitting to avoid losing the first test value
- Split data into training and testing sets based on date ranges

### 2. Feature Preprocessing (fitted on training data only)

The preprocessing pipeline is modular and configurable via `pp_steps`. Each step can either **overwrite** (`'o'`) or **append** (`'a'`) features:

| Step | Class | Description |
|------|-------|-------------|
| `cov_select` | `CovSelector` | Select top-k features based on Spearman/Pearson correlation with target |
| `zscore` | `ZScoreScaler` | Z-Score normalization (mean=0, std=1) |
| `winsor` | `Winsorizer` | Clip outliers to specified quantiles (default: 1st and 99th percentile) |
| `pca` | `PCACompressor` | Reduce dimensionality via PCA |

**Default pipeline:** `cov_select → zscore → winsor → pca`

**Key design for avoiding data leakage:**

- All preprocessing steps are **fitted only on training data**
- The same fitted transformers are applied to test data without refitting
- Each step fits on the data **before** any transformation from the current step

### 3. TimeSeries Conversion

- Convert preprocessed features and targets to Darts `TimeSeries` objects
- Use integer index (instead of datetime frequency) to handle trading days with irregular gaps

### 4. Model Training

- Train a **Temporal Fusion Transformer (TFT)** model from the Darts library
- Key hyperparameters:
  - `input_chunk_length`: Number of historical time steps as input (default: 30)
  - `output_chunk_length`: Forecast horizon (default: 1)
  - `hidden_size`, `lstm_layers`, `num_attention_heads`: Model architecture
  - Optimizer: AdamW with gradient clipping
- Model weights are saved to `results/<exp_name>/tft_model.pt`

### 5. Evaluation

- Use `historical_forecasts` to generate rolling 1-step-ahead predictions on both train and test sets
- Convert predicted log returns back to price space using `from_logret()`
- Calculate MSE/RMSE in price space
- Save results:
  - `metrics.json`: Train/Test MSE and RMSE
  - `predictions.csv`: Date, ground truth, prediction, and split label
  - `predictions_plot.png`: Visualization of predictions vs ground truth

### 6. Avoiding Future Data Leakage

- **Preprocessing**: All scalers/selectors are fitted only on training data
- **Target transformation**: Log returns are computed on the full dataset before splitting, ensuring proper continuity
- **Evaluation**: `historical_forecasts` with `retrain=False` ensures no future information is used during prediction
- **NaN handling**: Only forward-fill (`ffill`) is used; backward-fill is avoided to prevent leaking future data
