# PatchWRAT — Time Series Forecasting Project

> Wavelet-augmented Patch Transformer for multivariate time-series forecasting.
> Developed as part of CS728, IIT Bombay, 2025.

The **final, clean implementation** lives in [`PatchWRAT_final/`](./PatchWRAT_final/).
See [`PatchWRAT_final/README.md`](./PatchWRAT_final/README.md) for the full architecture documentation, ablation details, and benchmark results.

---

## Quick Start

```bash
cd PatchWRAT_final

# ETTm1 (standard benchmark)
python train.py --dataset ett --data_path ../Rat/ETTm1.csv --horizons 96 192 336 720

# MPI Weather
python train.py --dataset weather --data_path ../Rat/mpi_roof_2017b/mpi_roof_2017b.csv
```

---

## Repository Layout

```
TTS/
├── PatchWRAT_final/        ← ✅ Final clean implementation (start here)
│   ├── model.py            — Architecture: RevIN, PatchEmbed, DWT, WRATBlock, PatchedWSA
│   ├── utils.py            — Datasets (ETT/Weather), metrics, early stopping, plots
│   ├── train.py            — CLI training script with full ablation suite
│   └── README.md           — Detailed architecture & benchmark documentation
│
├── Rat/                    ← Experiment workspace
│   ├── wrat.py             — WRAT ablation benchmark (ETTh1, 4 variants)
│   ├── modified_arch/      — PatchWRAT development iterations
│   │   ├── p_wsa.py        — ETTm1 version
│   │   └── p_wsa_weather.py — MPI Weather version
│   └── PWSA/               — Earlier P-WSA variants (reference only)
│
└── WRAT/                   ← Original WRAT baseline
    ├── models/wrat.py      — Original wavelet transformer (no patching)
    └── utils/              — Shared metrics and loss utilities
```

---

## Key Results (ETTm1, normalised MSE)

| Horizon | WRAT (original) | P-WSA v8 | **PatchWRAT (univariate)** |
|---------|----------------|----------|----------------------|
| H=96  | — | 0.3485 | **0.0476** |
| H=192 | — | 0.4072 | **0.0714** |
| H=336 | — | 0.4335 | **0.1023** |
| H=720 | — | 0.4984 | **0.1284** |
