# WRAT: Wavelet Multi-Resolution Transformer

## Overview
This repository contains the PyTorch implementation of the **Wavelet Multi-Resolution Transformer (WRAT)**. 
WRAT is designed for efficient and accurate time-series forecasting, utilizing a novel frequency-sparse attention mechanism computed over wavelet coefficients.

## Architecture
The project has been modularized for research and production scale:
- `models/wrat.py`: Defines the learnable discrete wavelet transform (`LearnableDWT1D`) and the `WRATBlock`.
- `models/vanilla.py`: Defines the `VanillaTransformerBaseline` for benchmarking alongside.
- `data/dataset.py`: Handles downloading and loading the ETTm1 time-series dataset.
- `utils/losses.py`: Custom `WaveletTransformerLoss` imposing orthogonality and sparsity constraints.
- `utils/metrics.py`: Regression evaluation methods (MAE, MSE, RMSE, R², etc).
- `main.py`: Entrypoint combining the modules to download data, train, benchmark against the baseline, and plot results.

## Requirements
```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## Running the Benchmark Pipeline
You can launch the training and evaluation dynamically using `run_pipeline.py`, which offers a real-time terminal progress bar to let you know exactly where the script is at during execution. It wraps `main.py` while forwarding all arguments.

```bash
python run_pipeline.py --epochs 30 --batch_size 64 --lr 1e-3
```

**Common flags:**
- `--epochs`: Number of epochs to train.
- `--batch_size`: Defaults to 64.
- `--seq_len`: Length of the input series window (defaults to 128).
- `--d_model`: Embedding dimension (defaults to 16).
- `--num_heads`: Number of attention heads.

## Output
It will download the dataset if it is not present in the current working directory.
After training, it will evaluate on the test split and print out a comparative benchmark table. Finally, it generates and saves `wrat_benchmark_results.png` which contains visually informative diagnostic plots (loss curves, sparsity tracking, qualitative forecast plots, and bar charts computing performance metrics).


Auther's commands

 C:\Users\Asus\radioconda\python.exe main.py --epochs 30 --batch_size 64 --seq_len 128 --lr 0.001

 