#  Time Series Forecasting Model

## Overview
This project contains time series forecasting implementations including WRAT (Wavelet Multiresolution Transformer) and PWSA architectures.

## Important Notice

**Final RAT Architecture ID Location:**
The final RAT architecture implementation uses the ID located in:
```
/Rat/PWSA/x_publish.py
```

Please use this file for any further modifications to the RAT architecture. This contains the complete implementation of the Wavelet Multiresolution Transformer (WMRT) framework with multivariate time-series forecasting capabilities.

## Project Structure
- `Rat/` - Main RAT implementation directory
  - `PWSA/` - PWSA variant implementations
    - `x_publish.py` - **Final RAT architecture implementation**
  - `ETTm1.csv` - Dataset file
  - `wrat.py` - WRAT benchmark implementation
- `WRAT/` - Additional WRAT variants

## Key Features
- Multivariate time-series forecasting
- Channel Independence approach
- Full Ablation Suite
- Automated test-set visualizations
- Same data splits (60/20/20) and horizons [96, 192, 336, 720]

## Usage
```bash
python Rat/PWSA/x_publish.py
```

Ensure `ETTm1.csv` is available in the `Rat/` directory before running.
