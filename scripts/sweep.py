"""
sweep.py — Short hyperparameter sweep comparing fusion strategies.

Sweeps four fusion methods for the Parallel CNN-GRU:
    'concat', 'concat_fc', 'add', 'concat_conv1d'

Each run is independent; device memory is freed between runs to avoid
accumulation on M2 Pro unified memory.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.amp import GradScaler

import config
from models import build_model
from trainer import train_epoch, evaluate


# ── Sweep configurations ───────────────────────────────────────────────────────
# 'add' requires cnn_fc == rnn_fc_out; override both to 256 for M2 RAM budget.

SWEEP_CFGS = [
    {"name": "concat",          "fusion": "concat"},
    {"name": "concat_fc",       "fusion": "concat_fc"},
    {"name": "concat_conv1d",   "fusion": "concat_conv1d"},
    {"name": "add (256)",       "fusion": "add", "cnn_fc": 256, "rnn_fc_in": 256, "rnn_fc_out": 256},
]


# ─────────────────────────────────────────────────────────────────────────────
def _run_single(
    cfg_overrides: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    sweep_epochs: int,
    lr: float,
) -> List[float]:
    """Train one sweep configuration and return per-epoch test accuracies."""
    merged_cfg = {**config.PARALLEL_CFG, **cfg_overrides}
    model = build_model(
        cfg=merged_cfg,
        window=config.WINDOW,
        n_classes=config.N_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # GradScaler is only active for CUDA; disabled (no-op) on MPS/CPU
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    te_accs: List[float] = []

    for _ in range(1, sweep_epochs + 1):
        train_epoch(model, train_loader, criterion, optimiser, scaler, device)
        _, te_acc, _, _ = evaluate(model, test_loader, criterion, device)
        te_accs.append(te_acc)

    del model, optimiser
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return te_accs


# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
    sweep_epochs: int = config.SWEEP_EPOCHS,
    lr: float = config.LR,
    output_dir: str = config.OUTPUT_DIR,
) -> None:
    """Run all fusion sweep configurations and plot results.

    Args:
        train_loader : training DataLoader
        test_loader  : evaluation DataLoader
        device       : torch.device
        sweep_epochs : epochs per configuration
        lr           : learning rate
        output_dir   : directory to save the plot
    """
    results = []
    for raw_cfg in SWEEP_CFGS:
        name = raw_cfg["name"]
        overrides = {k: v for k, v in raw_cfg.items() if k != "name"}
        print(f"\n── Sweep: {name}")
        te_accs = _run_single(overrides, train_loader, test_loader, device, sweep_epochs, lr)
        best = max(te_accs)
        print(f"   Best acc = {best:.4f}")
        results.append({"name": name, "best_acc": best, "history": te_accs})

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        ax.plot(
            range(1, sweep_epochs + 1),
            [a * 100 for a in r["history"]],
            lw=2,
            label=f"{r['name']}  (best {r['best_acc'] * 100:.2f}%)",
        )
    ax.set(xlabel="Epoch", ylabel="Test Acc (%)", title="Parallel CNN-GRU — Fusion Sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parallel_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[sweep] Plot saved → {out_path}")
    plt.close(fig)

    print("\nRanking:")
    for r in sorted(results, key=lambda x: x["best_acc"], reverse=True):
        print(f"  {r['name']:25s}  {r['best_acc']:.4f}")
