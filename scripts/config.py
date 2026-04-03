"""
config.py — Central configuration for EEG Motor Imagery Parallel CNN-RNN.

Optimised for Apple M2 (8 GB RAM) — uses MPS when available, falls back to CPU.
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

# ── Data paths ─────────────────────────────────────────────────────────────────
_REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARALLEL_DATA_DIR = os.path.join(_REPO_ROOT, "npz_4class_parallel_W10")
OUTPUT_DIR        = os.path.join(_REPO_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared hyperparameters ─────────────────────────────────────────────────────
WINDOW       = 10
N_CLASSES    = 4       # 4 motor-imagery classes (set 5 if rest class kept)
EPOCHS       = 50
SEED         = 42
TRAIN_RATIO  = 0.75
NUM_SUBJECTS = 20      # Limit to first N subjects (0 or None for all 108)

# ── M2-optimised batch size and LR ─────────────────────────────────────────────
# M2 unified memory: keep batches small enough to leave ~4 GB for the OS.
# 128 is a good starting point; raise to 256 if memory permits.
BATCH   = 32
LR      = 1e-4
DROPOUT = 0.5

# ── DataLoader settings ────────────────────────────────────────────────────────
# pin_memory is only useful for CUDA; MPS/CPU use False.
# num_workers > 0 can cause issues on macOS with some PyTorch builds → 0 is safe.
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY  = DEVICE.type == "cuda"

# ── Parallel model config ──────────────────────────────────────────────────────
# Reduced from paper defaults to stay within 8 GB unified memory.
PARALLEL_CFG = dict(
    conv_channels=32,
    cnn_fc=256,
    n_electrodes=64,
    rnn_fc_in=256,
    lstm_hidden=16,
    lstm_layers=2,
    rnn_fc_out=256,
    fusion="concat",   # 'concat' | 'add' | 'concat_fc' | 'concat_conv1d'
)

# ── Sweep ──────────────────────────────────────────────────────────────────────
SWEEP_EPOCHS = 10     # Keep short on M2; each epoch is slower than a T4 GPU
