"""
PatchWRAT — Patch-based Wavelet Routing Attention Transformer
=============================================================
Public API
----------
from PatchWRAT_final import PatchedWSA, DualHeadPWSA_Loss
from PatchWRAT_final import ETTDataset, WeatherDataset, evaluate
"""

from model import (
    RevIN,
    PatchEmbedding,
    LearnableDWT1D,
    FrequencySparseAttention,
    WRATBlock,
    LearnableTauWRATBlock,
    PatchedWSA,
    DualHeadPWSA_Loss,
)

from utils import (
    ETTDataset,
    WeatherDataset,
    EarlyStopping,
    evaluate,
    count_parameters,
    plot_learning_curves,
    plot_final_bar_charts,
    plot_learned_filters,
    plot_predictions,
)

__all__ = [
    "RevIN", "PatchEmbedding", "LearnableDWT1D",
    "FrequencySparseAttention", "WRATBlock", "LearnableTauWRATBlock",
    "PatchedWSA", "DualHeadPWSA_Loss",
    "ETTDataset", "WeatherDataset",
    "EarlyStopping", "evaluate", "count_parameters",
    "plot_learning_curves", "plot_final_bar_charts",
    "plot_learned_filters", "plot_predictions",
]
