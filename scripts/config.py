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
EPOCHS       = 50
SEED         = 42
TRAIN_RATIO  = 0.8     # (Of the training subjects pool)
NUM_SUBJECTS = 12      # 10 subjects pool + 2 subjects blind test

# ── M2 Pro-optimised batch size and LR ────────────────────────────────────────
BATCH   = 32
LR      = 1e-4
DROPOUT = 0.5

# ── Gradient Accumulation ─────────────────────────────────────────────────────
# Effective batch size = BATCH * GRAD_ACCUM_STEPS (32 * 4 = 128).
# The optimizer only steps every N micro-batches, simulating a larger batch
# without the memory cost. Set to 1 to disable (standard per-batch updates).
GRAD_ACCUM_STEPS = 4

# ── LR Scheduler (CosineAnnealingLR) ──────────────────────────────────────────
# T_max  : half-cycle length in epochs (set to EPOCHS for a single full cosine sweep)
# ETA_MIN: minimum LR at the bottom of the cosine curve
LR_T_MAX  = EPOCHS       # decay all the way to ETA_MIN over training
LR_ETA_MIN = 1e-6

# ── Data Augmentation ─────────────────────────────────────────────────────────
# Gaussian noise added *only* to training samples to regularise the model.
# Set to 0.0 to disable.
NOISE_STD = 0.05

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
