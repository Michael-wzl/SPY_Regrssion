# SPY Next-Trading-Day Regression with Temporal Fusion Transformer (TFT)

## How to Use

1. Install required packages:

   ```bash
   conda create -n spy_tft python=3.12
   conda activate spy_tft
   pip install -r requirements.txt
   ```

2. Run the training script: (You may set your own config file in the code)

   ```bash
   python tft.py --exp_name your_experiment_name --device cuda:0
   ```

3. Check results in the `results/your_experiment_name/` directory.

4. Reuse the trained model for inference or further analysis as needed by setting [use_ckpt](tft.py#L423) to True and specifying the checkpoint path in [ckpt_name](tft.py#L424).
