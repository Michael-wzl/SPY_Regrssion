# SPY Regression

## Objective

This repo tries to use TFT (Temporal Fusion Transformer) for time series forecasting of daily SPY (S&P 500 ETF) close prices.
We use 2015-01-01 to 2021-12-31 as training data, and 2022-01-01 to 2025-06-01 as testing data. MSE is used as the benchmark metric.

## How to run

1. Install dependencies (recommend a clean virtualenv or conda env):

```bash
pip install -r requirements.txt
```

2. Run the TFT training/evaluation script:

```bash
python3 tft.py
```

This will:

- Load `data/ohlcv/spy_ohlcv_1drth_20141231_20250602.csv` as the base OHLCV.
- Join daily covariates from `data/volatility_and_options`, `data/macro_and_growth`, `data/sentiment_and_news`, `data/sector_and_cross_assets`, and `data/commodities_and_fx` by date.
- Train on 2015-01-01 .. 2021-12-31; evaluate on 2022-01-01 .. 2025-06-01 using MSE.
- Save predictions to `outputs/tft_predictions.csv` with columns: `date, true_close, pred_close`.

Notes:

- The script uses Darts' `TFTModel` with modest defaults (hidden_size=32, n_epochs=30) to keep runtime reasonable on CPU. Increase `n_epochs` if you want better accuracy.
- If you don't have GPU, PyTorch will run on CPU automatically.
