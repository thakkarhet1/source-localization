"""
config.py — Central configuration for EEG Motor Imagery Parallel CNN-RNN.

Optimised for Apple M2 Pro (Metal GPU) — uses MPS when available, falls back to CPU.
All hyperparameters and path settings live here so every other module
imports from a single source of truth.
"""

import os
import torch

# ── Device ────────────────────────────────────────────────────────────────────
# Priority: MPS (Apple Silicon) → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[config] Using device: {DEVICE}")

# ── Data paths (absolute) ──────────────────────────────────────────────────────
PARALLEL_DATA_DIR = "/Users/hetthakkar/Desktop/R/Github/source-localization/npz_4class_parallel_W10"
OUTPUT_DIR        = "/Users/hetthakkar/Desktop/R/Github/source-localization/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared hyperparameters ─────────────────────────────────────────────────────
WINDOW       = 10
N_CLASSES    = 4       # 4 motor-imagery classes (set 5 if rest class kept)
EPOCHS       = 10
SEED         = 42
TRAIN_RATIO  = 0.75
NUM_SUBJECTS = 10      # Limit to first N subjects (0 or None for all 108)

# ── M2 Pro-optimised batch size and LR ────────────────────────────────────────
# M2 Pro has 16 GB unified memory; batch 128 is a good starting point.
BATCH   = 128
LR      = 1e-4
DROPOUT = 0.5

# ── DataLoader settings ────────────────────────────────────────────────────────
# num_workers > 0 can deadlock on macOS with MPS/OpenMP → 0 is safe.
# pin_memory is only beneficial for CUDA; MPS/CPU use False.
NUM_WORKERS     = 0
PREFETCH_FACTOR = None   # Must be None when num_workers == 0
PIN_MEMORY      = DEVICE.type == "cuda"

# ── Parallel model config ──────────────────────────────────────────────────────
# GRU replaces LSTM (see Gradcam EEG decoding paper).
PARALLEL_CFG = dict(
    conv_channels=32,
    cnn_fc=256,
    n_electrodes=64,
    rnn_fc_in=256,
    gru_hidden=16,
    gru_layers=2,
    rnn_fc_out=256,
    fusion="concat",   # 'concat' | 'add' | 'concat_fc' | 'concat_conv1d'
)

# ── Sweep ──────────────────────────────────────────────────────────────────────
SWEEP_EPOCHS = 10     # Keep short on M2 Pro; each epoch is slower than a T4 GPU
