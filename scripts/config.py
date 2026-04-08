"""
config.py — Central configuration for EEG Motor Imagery Parallel CNN-GRU.

Optimised for Apple M2 Pro (Metal GPU).
"""

import os
import torch

# ── Device ────────────────────────────────────────────────────────────────────
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
N_CLASSES    = 4
EPOCHS       = 100
SEED         = 42
TRAIN_RATIO  = 0.8     # (Of the training subjects pool)
NUM_SUBJECTS = 12      # 10 subjects pool + 2 subjects blind test

# ── M2 Pro-optimised batch size and LR ────────────────────────────────────────
BATCH   = 32
LR      = 1e-4
DROPOUT = 0.5    

# ── DataLoader settings ────────────────────────────────────────────────────────
NUM_WORKERS     = 0
PREFETCH_FACTOR = None
PIN_MEMORY      = DEVICE.type == "cuda"

# ── Parallel model config ──────────────────────────────────────────────────────
PARALLEL_CFG = dict(
    conv_channels=32,
    cnn_fc=256,
    n_electrodes=64,
    rnn_fc_in=256,
    gru_hidden=16,
    gru_layers=2,
    rnn_fc_out=256,
    fusion="add",
)

# ── Checkpointing & Training ───────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 10
SWEEP_EPOCHS        = 10
